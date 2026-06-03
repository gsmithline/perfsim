"""Distributional collapse metrics for the opinion-dynamics PP loop.

Pure functions over a 1-D tensor of values in [lo, hi] (opinions or model
predictions). They quantify the model-collapse signatures we care about:

  variance / std       -- spread
  entropy              -- diversity of the binned distribution (nats)
  eff_support          -- exp(entropy): effective number of distinct levels
  occupied_frac        -- fraction of bins with any mass (raw support)
  low_prob_mass        -- mass on rare bins (the long tail that disappears)
  mode_mass            -- max bin fraction (high-prob events exaggerated)
  gini                 -- concentration of the histogram
  jaccard_support      -- occupied-bin overlap between two rounds

Cross-round drift / error amplification is computed in the runner loop from
the per-round mean (bias vs the innate truth) since it needs history.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _hist(v: Tensor, bins: int, lo: float, hi: float) -> Tensor:
    """Normalized histogram (probabilities per bin), shape (bins,)."""
    h = torch.histc(v.float().clamp(lo, hi), bins=bins, min=lo, max=hi)
    s = h.sum()
    return h / s if s > 0 else h


def entropy(v: Tensor, *, bins: int = 50, lo: float = 0.0, hi: float = 1.0) -> float:
    """Shannon entropy (nats) of the binned distribution. 0 = collapsed."""
    p = _hist(v, bins, lo, hi)
    nz = p[p > 0]
    return float(-(nz * nz.log()).sum())


def eff_support(v: Tensor, *, bins: int = 50, lo: float = 0.0, hi: float = 1.0) -> float:
    """exp(entropy): effective number of distinct occupied levels."""
    return float(torch.tensor(entropy(v, bins=bins, lo=lo, hi=hi)).exp())


def occupied_frac(v: Tensor, *, bins: int = 50, lo: float = 0.0, hi: float = 1.0) -> float:
    """Fraction of bins with any mass (raw support of the distribution)."""
    p = _hist(v, bins, lo, hi)
    return float((p > 0).float().mean())


def low_prob_mass(v: Tensor, *, bins: int = 50, lo: float = 0.0, hi: float = 1.0,
                  rare_q: float = 0.05) -> float:
    """Mass sitting on rare bins -- the long tail of low-probability events.

    A bin is rare if its probability is below `rare_q` of the modal bin's
    probability. Returns the total probability on rare-but-occupied bins.
    Shrinks toward 0 as the tail disappears.
    """
    p = _hist(v, bins, lo, hi)
    if p.max() == 0:
        return 0.0
    cutoff = rare_q * p.max()
    rare = p[(p > 0) & (p < cutoff)]
    return float(rare.sum())


def mode_mass(v: Tensor, *, bins: int = 50, lo: float = 0.0, hi: float = 1.0) -> float:
    """Fraction of mass in the single densest bin (high-prob exaggeration)."""
    return float(_hist(v, bins, lo, hi).max())


def gini(v: Tensor, *, bins: int = 50, lo: float = 0.0, hi: float = 1.0) -> float:
    """Gini concentration of the histogram counts. 0 = uniform, ->1 = a spike."""
    p = _hist(v, bins, lo, hi).sort().values
    n = p.numel()
    if n == 0 or p.sum() == 0:
        return 0.0
    idx = torch.arange(1, n + 1, dtype=p.dtype)
    return float((2.0 * (idx * p).sum()) / (n * p.sum()) - (n + 1.0) / n)


def occupied_bins(v: Tensor, *, bins: int = 50, lo: float = 0.0, hi: float = 1.0) -> set[int]:
    p = _hist(v, bins, lo, hi)
    return {int(i) for i in torch.nonzero(p > 0).flatten().tolist()}


def jaccard_support(a: Tensor, b: Tensor, *, bins: int = 50, lo: float = 0.0,
                    hi: float = 1.0) -> float:
    """Jaccard overlap of the occupied-bin sets of two distributions."""
    sa = occupied_bins(a, bins=bins, lo=lo, hi=hi)
    sb = occupied_bins(b, bins=bins, lo=lo, hi=hi)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def summary(v: Tensor, *, bins: int = 50, lo: float = 0.0, hi: float = 1.0) -> dict:
    """All single-distribution metrics in one dict."""
    v = v.float()
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "var": float(v.var()),
        "min": float(v.min()),
        "max": float(v.max()),
        "entropy": entropy(v, bins=bins, lo=lo, hi=hi),
        "eff_support": eff_support(v, bins=bins, lo=lo, hi=hi),
        "occupied_frac": occupied_frac(v, bins=bins, lo=lo, hi=hi),
        "low_prob_mass": low_prob_mass(v, bins=bins, lo=lo, hi=hi),
        "mode_mass": mode_mass(v, bins=bins, lo=lo, hi=hi),
        "gini": gini(v, bins=bins, lo=lo, hi=hi),
    }
