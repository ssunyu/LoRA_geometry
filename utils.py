from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
VIS = ROOT / "visualization"
METRICS = ROOT / "metrics"
CLINICAL_METRICS = METRICS / "clinical_bridge"


def ensure_dirs() -> None:
    VIS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    CLINICAL_METRICS.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def shape_line(name: str, shape: tuple[int, ...] | torch.Size) -> str:
    dims = " x ".join(str(int(x)) for x in tuple(shape))
    return f"{name:<18} {dims}"


def compact_int(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)
