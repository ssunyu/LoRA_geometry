from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from model import DenseAdapter, LoRAConfig, LoRALinear, make_teacher_world, smoke_check, train_regressor
from ops import (
    freedom_report,
    make_low_rank_update,
    numerical_rank,
    rank_energy,
    rank_for_energy,
    singular_values,
    subspace_alignment,
)
from utils import CLINICAL_METRICS, METRICS, VIS, compact_int, device, ensure_dirs, load_json, save_json, section, set_seed, shape_line


VIT_HF_NAME = "google/vit-base-patch16-224"
NUM_LABELS = 2
LORA_STATE_PATH = CLINICAL_METRICS / "lora_state.pt"
LORA_NAME_PATTERN = re.compile(
    r"(?P<layer>.+?)\.(?P<which>lora_A|lora_B)\.(?P<adapter>[^.]+)\.weight$"
)


@dataclass
class DataConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    train_subset: int | None = 1000
    test_subset: int | None = 400
    seed: int = 0


@dataclass
class TuneConfig:
    strategy: Literal["linear_probe", "lora"]
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_modules: tuple[str, ...] = ("query", "value")
    epochs: int = 3
    lr: float = 3e-4
    seed: int = 0


class PneumoniaTensorDataset(Dataset):
    def __init__(self, base_dataset, processor) -> None:
        self.base = base_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self.base[idx]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        if isinstance(label, np.ndarray):
            label = int(label.squeeze())
        else:
            label = int(label)
        return pixel_values, label


def print_protocol() -> None:
    section("ADAPTATION MANIFOLD ANALYSIS PROTOCOL")
    print("Task Shift     : Domain adaptation shifts pretrained behaviors on new tasks")
    print("Unit of Change : Delta W (The weight displacement tensor carrying task adaptation)")
    print("Hypothesis     : Weight adaptation happens on a low-rank task subspace")
    print("Constraint     : rank(Delta W) <= r (Restricting update degrees of freedom)")
    print("Observable     : Singular Value Spectrum Collapse (Energy concentration in a few directions)")
    print("Validation     : Recovers subspace in Toy world; profiles spectrum decay in Medical ViT")


def print_tensor_flow(lora: LoRALinear, batch_size: int) -> None:
    report = lora.freedom()
    section("tensor flow through the connection space")
    print(shape_line("x", (batch_size, lora.in_dim)))
    print(shape_line("A", lora.A.shape))
    print(shape_line("z_task = x A^T", (batch_size, lora.cfg.rank)))
    print(shape_line("B", lora.B.shape))
    print(shape_line("update = z B^T", (batch_size, lora.out_dim)))
    print(shape_line("Delta W = B A", lora.delta_weight.shape))
    print()
    print(f"Unconstrained space (Dense)  : {compact_int(report['dense_space_dim'])} parameters")
    print(f"LoRA active parameter count  : {compact_int(report['lora_raw_params'])} parameters")


def draw_toy(
    true_s: np.ndarray,
    dense_s: np.ndarray,
    lora_s: np.ndarray,
    dense_history: list[float],
    lora_history: list[float],
    report: dict[str, int],
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    ax = axes[0]
    ax.semilogy(true_s, "o-", label="teacher Delta W", color="black")
    ax.semilogy(dense_s, "o-", label="dense learned Delta W", color="#1f77b4")
    ax.semilogy(lora_s, "o-", label="LoRA learned Delta W", color="#d62728")
    ax.set_title("What directions did the task use?")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value")
    ax.grid(alpha=0.28, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    x_axis = np.arange(len(dense_history)) * 20
    ax.plot(x_axis, dense_history, label="dense Delta W", color="#1f77b4")
    ax.plot(x_axis, lora_history, label="x -> A -> z -> B", color="#d62728")
    ax.set_yscale("log")
    ax.set_title("Can the constrained path learn?")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=8)

    ax = axes[2]
    labels = ["dense space", "rank-r manifold", "A/B params"]
    values = [
        report["dense_space_dim"],
        report["rank_manifold_dim"],
        report["lora_raw_params"],
    ]
    ax.bar(labels, values, color=["#1f77b4", "#d62728", "#ff9896"])
    ax.set_yscale("log")
    ax.set_title("Freedom budget of Delta W")
    ax.set_ylabel("degrees / coordinates")
    ax.grid(axis="y", alpha=0.28, which="both")

    fig.suptitle("Toy causality: known low-rank cause, recovered through LoRA", fontweight="bold")
    fig.tight_layout()
    out = VIS / "toy_lora_recovery.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def run_toy(seed: int = 0) -> dict[str, float]:
    section("toy world: known cause")
    smoke_check()
    x, y, base, delta_true = make_teacher_world(seed=seed)

    dense = DenseAdapter(base)
    lora = LoRALinear(base, LoRAConfig(rank=4, alpha=4.0))
    print_tensor_flow(lora, batch_size=x.shape[0])

    dense_history = train_regressor(dense, x, y)
    lora_history = train_regressor(lora, x, y)

    dense_delta = dense.delta_weight.detach()
    lora_delta = lora.delta_weight.detach()
    report = lora.freedom()

    metrics = {
        "dense_final_mse": dense_history[-1],
        "lora_final_mse": lora_history[-1],
        "dense_trainable_parameters": sum(p.numel() for p in dense.parameters()),
        "lora_trainable_parameters": lora.trainable_parameter_count(),
        "dense_rank4_energy": rank_energy(dense_delta, 4),
        "lora_rank4_energy": rank_energy(lora_delta, 4),
        "dense_alignment_to_teacher": subspace_alignment(delta_true, dense_delta, 4),
        "lora_alignment_to_teacher": subspace_alignment(delta_true, lora_delta, 4),
        "teacher_numerical_rank": numerical_rank(delta_true),
        "lora_numerical_rank": numerical_rank(lora_delta),
        **{f"freedom_{k}": v for k, v in report.items()},
    }

    out = draw_toy(
        singular_values(delta_true),
        singular_values(dense_delta),
        singular_values(lora_delta),
        dense_history,
        lora_history,
        report,
    )
    metrics_path = METRICS / "toy_lora_metrics.json"
    save_json(metrics_path, metrics)
    print(f"saved: {out}")
    print(f"saved: {metrics_path}")
    print(json.dumps(metrics, indent=2))
    return metrics


def take_subset(dataset: Dataset, n: int | None, seed: int) -> Dataset:
    if n is None or n >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices: Iterable[int] = rng.choice(len(dataset), size=n, replace=False).tolist()
    return Subset(dataset, sorted(int(i) for i in indices))


def make_loaders(cfg: DataConfig, processor) -> tuple[DataLoader, DataLoader]:
    from medmnist import PneumoniaMNIST

    train_raw = PneumoniaMNIST(split="train", size=cfg.image_size, download=True)
    test_raw = PneumoniaMNIST(split="test", size=cfg.image_size, download=True)
    train_raw = take_subset(train_raw, cfg.train_subset, cfg.seed)
    test_raw = take_subset(test_raw, cfg.test_subset, cfg.seed)
    pin_memory = torch.cuda.is_available()

    return (
        DataLoader(
            PneumoniaTensorDataset(train_raw, processor),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            PneumoniaTensorDataset(test_raw, processor),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        ),
    )


def build_clinical_model(cfg: TuneConfig) -> nn.Module:
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        VIT_HF_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.classifier.parameters():
        p.requires_grad_(True)

    if cfg.strategy == "linear_probe":
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT failed to import. In Colab, run `pip uninstall -y torchao`, "
            "restart the session, then reinstall requirements.txt."
        ) from exc

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_target_modules),
        lora_dropout=0.0,
        bias="none",
        modules_to_save=["classifier"],
    )
    return get_peft_model(model, lora_cfg)


def trainable_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model: nn.Module, loader: DataLoader, optim, dev: torch.device) -> float:
    model.train()
    total = 0.0
    n = 0
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(dev)
        labels = labels.to(dev)
        out = model(pixel_values=pixel_values, labels=labels)
        loss = out.loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += float(loss.item()) * labels.size(0)
        n += labels.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, dev: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(dev)
        labels = labels.to(dev)
        out = model(pixel_values=pixel_values, labels=labels)
        pred = out.logits.argmax(dim=-1)
        loss_sum += float(out.loss.item()) * labels.size(0)
        correct += int((pred == labels).sum().item())
        total += int(labels.size(0))
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1)}


def run_strategy(strategy: Literal["linear_probe", "lora"], data_cfg: DataConfig, tune_cfg: TuneConfig) -> dict:
    if tune_cfg.strategy != strategy:
        tune_cfg = TuneConfig(strategy=strategy, **{
            k: v for k, v in asdict(tune_cfg).items() if k != "strategy"
        })

    set_seed(tune_cfg.seed)
    dev = device()

    from transformers import ViTImageProcessor

    processor = ViTImageProcessor.from_pretrained(VIT_HF_NAME)
    train_loader, test_loader = make_loaders(data_cfg, processor)
    model = build_clinical_model(tune_cfg).to(dev)
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=tune_cfg.lr)

    history = []
    for epoch in range(tune_cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optim, dev)
        eval_metrics = evaluate(model, test_loader, dev)
        row = {"epoch": epoch + 1, "train_loss": train_loss, **eval_metrics}
        history.append(row)
        print(
            f"  [{strategy} epoch {epoch + 1}/{tune_cfg.epochs}] "
            f"train_loss={train_loss:.4f} test_acc={eval_metrics['accuracy']:.3f}"
        )

    result = {
        "strategy": strategy,
        "trainable_parameters": trainable_count(model),
        "history": history,
        "final": history[-1],
    }

    if strategy == "lora":
        lora_state = {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad and "lora_" in name
        }
        torch.save(lora_state, LORA_STATE_PATH)
        result["lora_state_path"] = str(LORA_STATE_PATH)

    save_json(CLINICAL_METRICS / f"{strategy}_metrics.json", result)
    return result


def group_lora_pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for name, tensor in state.items():
        match = LORA_NAME_PATTERN.search(name)
        if match is None:
            continue
        key = match.group("layer")
        which = "A" if match.group("which") == "lora_A" else "B"
        pairs[key][which] = tensor
    return {k: v for k, v in pairs.items() if "A" in v and "B" in v}


def analyze_lora_state(state_path: Path, alpha: float, rank: int) -> dict:
    state = torch.load(state_path, map_location="cpu")
    pairs = group_lora_pairs(state)
    if not pairs:
        raise RuntimeError(f"no lora_A/lora_B pairs found in {state_path}")

    layers = []
    scale = alpha / rank
    for key, ab in pairs.items():
        delta = scale * (ab["B"].float() @ ab["A"].float())
        s = singular_values(delta)
        layers.append({
            "layer": key,
            "trained_rank": int(rank),
            "delta_shape": list(delta.shape),
            "singular_values": s.tolist(),
            "rank_for_80pct": rank_for_energy(s, 0.80),
            "rank_for_90pct": rank_for_energy(s, 0.90),
            "rank_for_95pct": rank_for_energy(s, 0.95),
        })

    summary = {
        "n_layers": len(layers),
        "alpha": alpha,
        "rank": rank,
        "median_rank_for_90pct": int(np.median([x["rank_for_90pct"] for x in layers])),
        "min_rank_for_90pct": int(np.min([x["rank_for_90pct"] for x in layers])),
        "max_rank_for_90pct": int(np.max([x["rank_for_90pct"] for x in layers])),
    }
    out = {"summary": summary, "layers": layers}
    save_json(CLINICAL_METRICS / "delta_w_analysis.json", out)
    return out


def draw_clinical_figure() -> Path:
    delta_analysis = load_json(CLINICAL_METRICS / "delta_w_analysis.json")
    lp_metrics = load_json(CLINICAL_METRICS / "linear_probe_metrics.json")
    lora_metrics = load_json(CLINICAL_METRICS / "lora_metrics.json")

    layers = delta_analysis["layers"]
    rank = delta_analysis["summary"]["rank"]
    median_r90 = delta_analysis["summary"]["median_rank_for_90pct"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    all_norm = []
    for layer in layers:
        s = np.array(layer["singular_values"])
        s_norm = s / s.max() if s.max() > 0 else s
        all_norm.append(s_norm)
        ax.plot(np.arange(1, len(s_norm) + 1), s_norm, color="#1f77b4", alpha=0.16, lw=1.0)

    median_curve = np.median(np.stack(all_norm), axis=0)
    ax.plot(
        np.arange(1, len(median_curve) + 1),
        median_curve,
        color="#d62728",
        lw=2.2,
        marker="o",
        ms=4,
        label="median across layers",
    )
    ax.axvline(
        median_r90,
        color="#2ca02c",
        ls="--",
        alpha=0.75,
        label=f"median 90% energy at r={median_r90}",
    )
    ax.set_yscale("log")
    ax.set_xlabel("singular value index of Delta W")
    ax.set_ylabel("sigma / sigma_max")
    ax.set_title(f"Clinical LoRA update spectrum, trained r={rank}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper right")

    ax = axes[1]
    strategies = ["linear probe", "LoRA"]
    accs = [lp_metrics["final"]["accuracy"], lora_metrics["final"]["accuracy"]]
    trainable = [lp_metrics["trainable_parameters"], lora_metrics["trainable_parameters"]]
    bars = ax.bar(strategies, accs, color=["#888888", "#d62728"], width=0.55)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("test accuracy")
    ax.set_title("Same frozen ViT, two adaptation strategies")
    ax.grid(axis="y", alpha=0.3)

    for bar, acc, tn in zip(bars, accs, trainable):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"acc {acc:.3f}\n{tn / 1e6:.2f}M params",
            ha="center",
            va="bottom",
            fontsize=9.5,
            family="monospace",
        )

    fig.suptitle("Real trace: the clinical Delta W collapses into a few directions", fontweight="bold")
    fig.tight_layout()
    out = VIS / "clinical_lora_bridge.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def print_clinical_freedom(rank: int) -> None:
    section("clinical connection space: ViT q/v projection")
    report = freedom_report(out_dim=768, in_dim=768, rank=rank)
    print("one dense q or v Delta W:", compact_int(report.dense_space_dim), "degrees")
    print("one LoRA q or v path    :", compact_int(report.lora_raw_params), "A/B coordinates")
    print("rank-r manifold dim     :", compact_int(report.rank_manifold_dim))
    print("observable              : spectrum collapse after training")


def run_real() -> None:
    data_cfg = DataConfig()
    tune_lp = TuneConfig(strategy="linear_probe", epochs=3)
    tune_lora = TuneConfig(strategy="lora", lora_rank=8, lora_alpha=16, epochs=3)
    print_clinical_freedom(tune_lora.lora_rank)

    section("real clinical task: unknown cause, observable trace")
    print("[stage 1/4] linear probe")
    run_strategy("linear_probe", data_cfg, tune_lp)
    print("\n[stage 2/4] LoRA fine-tuning")
    run_strategy("lora", data_cfg, tune_lora)
    print("\n[stage 3/4] Delta W spectrum analysis")
    analysis = analyze_lora_state(LORA_STATE_PATH, alpha=tune_lora.lora_alpha, rank=tune_lora.lora_rank)
    print(json.dumps(analysis["summary"], indent=2))
    print("\n[stage 4/4] figure")
    out = draw_clinical_figure()
    print(f"saved: {out}")


def run_clinical_figure_only() -> None:
    print_clinical_freedom(rank=8)
    analysis = analyze_lora_state(LORA_STATE_PATH, alpha=16, rank=8)
    out = draw_clinical_figure()
    print(json.dumps(analysis["summary"], indent=2))
    print(f"saved: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA parameter efficiency analysis in medical ViTs")
    parser.add_argument(
        "--stage",
        choices=["toy", "real", "clinical-figure", "all"],
        default="toy",
        help="real trains ViT and is intended for Colab/GPU; clinical-figure reuses saved metrics.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(args.seed)
    print_protocol()

    if args.stage in {"toy", "all"}:
        run_toy(seed=args.seed)
    if args.stage in {"real", "all"}:
        run_real()
    if args.stage == "clinical-figure":
        run_clinical_figure_only()


if __name__ == "__main__":
    main()
