"""Shared metrics for the recommender experiments."""

from __future__ import annotations

import torch


def gini(x: torch.Tensor) -> float:
    xs, _ = x.reshape(-1).clamp_min(0.0).sort()
    n = xs.numel()
    idx = torch.arange(1, n + 1, dtype=xs.dtype)
    return float((2.0 * (idx * xs).sum()) / (n * xs.sum()) - (n + 1) / n)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1) - a.mean()
    b = b.reshape(-1) - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))
