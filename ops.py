from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class FreedomReport:
    out_dim: int
    in_dim: int
    rank: int
    dense_space_dim: int
    lora_raw_params: int
    rank_manifold_dim: int
    gauge_redundancy: int

    def as_dict(self) -> dict[str, int]:
        return {
            "out_dim": self.out_dim,
            "in_dim": self.in_dim,
            "rank": self.rank,
            "dense_space_dim": self.dense_space_dim,
            "lora_raw_params": self.lora_raw_params,
            "rank_manifold_dim": self.rank_manifold_dim,
            "gauge_redundancy": self.gauge_redundancy,
        }


def freedom_report(out_dim: int, in_dim: int, rank: int) -> FreedomReport:
    """
    Count the spaces before choosing a model.

    Unit of change:
        Delta W in R^(out_dim x in_dim)

    Full fine-tuning:
        Delta W can move anywhere in that dense space.

    LoRA:
        Delta W must be generated as B @ A, so it lies on the rank-r manifold.
        A/B have r^2 redundant coordinates because B A = (B Q)(Q^-1 A).
    """
    dense = out_dim * in_dim
    raw = rank * in_dim + out_dim * rank
    gauge = rank * rank
    manifold = rank * (out_dim + in_dim - rank)
    return FreedomReport(
        out_dim=out_dim,
        in_dim=in_dim,
        rank=rank,
        dense_space_dim=dense,
        lora_raw_params=raw,
        rank_manifold_dim=manifold,
        gauge_redundancy=gauge,
    )


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        raise ValueError("cannot normalize a near-zero vector")
    return v / norm


def make_low_rank_update(out_dim: int, in_dim: int, rank: int, seed: int) -> np.ndarray:
    """
    Build Delta W from hidden causes:

        Delta W = sum_k strength_k * output_direction_k input_direction_k^T

    This is the top-down causal assumption. SVD is the bottom-up readout.
    """
    rng = np.random.default_rng(seed)
    delta = np.zeros((out_dim, in_dim))
    strengths = np.geomspace(4.0, 0.7, num=rank)

    for strength in strengths:
        u = normalize(rng.normal(size=out_dim))
        v = normalize(rng.normal(size=in_dim))
        delta += strength * np.outer(u, v)

    return delta


def singular_values(delta: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(delta, np.ndarray):
        return np.linalg.svd(delta, compute_uv=False)
    return torch.linalg.svdvals(delta.detach().float()).cpu().numpy()


def numerical_rank(delta: Tensor | np.ndarray, tol: float = 1e-4) -> int:
    return int(np.sum(singular_values(delta) > tol))


def rank_energy(delta: Tensor | np.ndarray, rank: int) -> float:
    s = singular_values(delta)
    energy = s ** 2
    return float(energy[:rank].sum() / (energy.sum() + 1e-12))


def rank_for_energy(s: np.ndarray, threshold: float) -> int:
    energy = s ** 2
    curve = np.cumsum(energy) / (energy.sum() + 1e-12)
    return min(int(np.searchsorted(curve, threshold)) + 1, len(s))


def subspace_alignment(delta_a: Tensor, delta_b: Tensor, rank: int) -> float:
    ua, _, _ = torch.linalg.svd(delta_a.detach().float(), full_matrices=False)
    ub, _, _ = torch.linalg.svd(delta_b.detach().float(), full_matrices=False)
    overlap = ua[:, :rank].T @ ub[:, :rank]
    return float((torch.linalg.norm(overlap) ** 2 / rank).item())
