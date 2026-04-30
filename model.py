from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ops import freedom_report


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 4.0
    init_scale: float = 0.01


def make_frozen_linear(in_dim: int, out_dim: int, seed: int = 0) -> nn.Linear:
    """
    W0 is the frozen operator.

    The chosen unit of adaptation is not the whole model and not accuracy. It is
    Delta W: the task-induced displacement of this operator.
    """
    generator = torch.Generator().manual_seed(seed)
    layer = nn.Linear(in_dim, out_dim)
    bound = math.sqrt(6.0 / (in_dim + out_dim))

    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        layer.bias.zero_()

    for param in layer.parameters():
        param.requires_grad_(False)
    return layer


class LoRALinear(nn.Module):
    """
    Minimal LoRA as a tensor flow through a constrained connection space.

    Dense update:
        x -> Delta W -> output displacement

    LoRA update:
        x -> A -> z_task -> B -> output displacement

    Shapes:
        x:       (batch, in)
        A:       (rank, in)
        z_task:  (batch, rank)
        B:       (out, rank)
        Delta W: (out, in)

    The bottleneck z_task is the modeling assumption. It says the task only
    needs r coordinates of force before being written back into model space.
    """

    def __init__(self, base: nn.Linear, cfg: LoRAConfig) -> None:
        super().__init__()
        if cfg.rank <= 0:
            raise ValueError("rank must be positive")

        self.cfg = cfg
        self.scale = cfg.alpha / cfg.rank
        self.out_dim, self.in_dim = base.weight.shape

        self.register_buffer("W0", base.weight.detach().clone())
        self.register_buffer("b0", base.bias.detach().clone())

        # A/B parameterize a rank-r manifold inside the full Delta W space.
        self.A = nn.Parameter(torch.empty(cfg.rank, self.in_dim))
        self.B = nn.Parameter(torch.zeros(self.out_dim, cfg.rank))
        nn.init.normal_(self.A, mean=0.0, std=cfg.init_scale)

    @property
    def delta_weight(self) -> Tensor:
        return self.scale * (self.B @ self.A)

    @property
    def merged_weight(self) -> Tensor:
        return self.W0 + self.delta_weight

    def lora_path(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z_task = F.linear(x, self.A)
        update = self.scale * F.linear(z_task, self.B)
        return z_task, update

    def forward(self, x: Tensor) -> Tensor:
        _, update = self.lora_path(x)
        frozen = F.linear(x, self.W0, self.b0)
        return frozen + update

    def trainable_parameter_count(self) -> int:
        return self.A.numel() + self.B.numel()

    def freedom(self) -> dict[str, int]:
        return freedom_report(self.out_dim, self.in_dim, self.cfg.rank).as_dict()

    @torch.no_grad()
    def copy_merged_into(self, target: nn.Linear) -> nn.Linear:
        target.weight.copy_(self.merged_weight)
        target.bias.copy_(self.b0)
        return target


class DenseAdapter(nn.Module):
    """
    Dense adaptation is the unconstrained comparison.

    It learns Delta W directly in R^(out x in). That gives maximum freedom, but
    it hides whether the task actually needed all those directions.
    """

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.register_buffer("W0", base.weight.detach().clone())
        self.register_buffer("b0", base.bias.detach().clone())
        self.delta = nn.Parameter(torch.zeros_like(base.weight))

    @property
    def delta_weight(self) -> Tensor:
        return self.delta

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.W0 + self.delta, self.b0)


def make_teacher_world(
    n: int = 384,
    in_dim: int = 32,
    out_dim: int = 24,
    true_rank: int = 4,
    seed: int = 0,
) -> tuple[Tensor, Tensor, nn.Linear, Tensor]:
    """
    Toy world with known causality.

    Assumed cause:
        the new task changes W0 through a low-rank Delta W_true.

    Logical test:
        if LoRA is the right constraint, it should recover behavior and
        subspace without being allowed to use the full dense space.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=generator)
    base = make_frozen_linear(in_dim, out_dim, seed=seed)

    A_true = torch.randn(true_rank, in_dim, generator=generator) / np.sqrt(in_dim)
    B_true = torch.randn(out_dim, true_rank, generator=generator) / np.sqrt(true_rank)
    delta_true = B_true @ A_true

    with torch.no_grad():
        y = F.linear(x, base.weight + delta_true, base.bias)
        y = y + 0.01 * torch.randn(y.shape, generator=generator)

    return x, y, base, delta_true


def train_regressor(model: nn.Module, x: Tensor, y: Tensor, steps: int = 300, lr: float = 3e-2) -> list[float]:
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    for step in range(steps):
        loss = F.mse_loss(model(x), y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 20 == 0 or step == steps - 1:
            history.append(float(loss.item()))

    return history


def smoke_check() -> None:
    base = make_frozen_linear(8, 6, seed=1)
    lora = LoRALinear(base, LoRAConfig(rank=2, alpha=2.0))
    x = torch.randn(4, 8)
    loss = lora(x).square().mean()
    loss.backward()

    assert lora.W0.requires_grad is False
    assert lora.A.grad is not None
    assert lora.B.grad is not None
    assert torch.linalg.matrix_rank(lora.delta_weight.detach()).item() <= 2

    z_task, update = lora.lora_path(x)
    assert z_task.shape == (4, 2)
    assert update.shape == (4, 6)

    merged = nn.Linear(8, 6)
    lora.copy_merged_into(merged)
    assert torch.allclose(lora(x), merged(x), atol=1e-6)
