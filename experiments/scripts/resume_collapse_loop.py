"""Closed-loop AI-mediated model collapse on real resumes (perfsim).

The premise: a single LLM rewrite is mild (see embed_real_rewrites.py), but
recursion is the amplifier. This runs the SELF-CONSUMING loop: each round the
population of resumes is rewritten by a mediator calibrated to REAL GPT-4o
rewrites, the rewritten resumes become next round's population, and the screener
retrains on them. We watch whether the mild per-pass shift compounds into
diversity collapse and loss of label recoverability over rounds.

Mediator: LinearSurrogateMediator fit (ridge) from real (original, rewrite)
embedding pairs, so the per-round shift magnitude is real, not invented.
Compare regimes: 'replace' (pure self-consuming loop) vs 'clean_anchor' (mix a
fraction of real pre-AI resumes back in each round -- the textbook defense).

Prereq: run experiments/scripts/embed_and_cache.py once (writes embeddings.npz).
Run:    python experiments/scripts/resume_collapse_loop.py
Tweak the CONFIG block below.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perfsim.core.predictor import Predictor
from perfsim.environments.mediation import LinearSurrogateMediator, MediationWorld
from perfsim.learners import ERMLearner
from perfsim.losses import BCELoss
from perfsim.models import LogisticModel
from perfsim.scenarios.ai_mediated import model_auc, recoverability, run_mediated, style_variance

# ---------------- CONFIG (tweak me) ----------------
CACHE = Path.home() / ".cache/perfsim/datasets/two_tickets"
EMB = CACHE / "embeddings.npz"
OUT = Path("runs/two_tickets_analysis")
PCA_DIM = 64          # work in PCA space (speed + a square surrogate map)
RIDGE_ALPHA = 0.01    # small = faithful to the real rewrite (large over-contracts)
N_ROUNDS = 25
ANCHOR_ALPHA = 0.3    # fraction of real pre-AI data mixed in for clean_anchor
LABEL = "experience"  # "experience" (headroom) or "occupation" (trivially separable)
REGIMES = ("replace", "clean_anchor")
# ---------------------------------------------------


def _var(z):  # total variance summed over dims (same notion as style_variance)
    return float(np.var(z, axis=0).sum())


def load():
    d = np.load(EMB)
    orig, rewrite = d["orig"], d["rewrite"]
    pca = PCA(n_components=PCA_DIM, random_state=0).fit(orig)
    zo, zr = pca.transform(orig), pca.transform(rewrite)
    # surrogate: rewrite ~= orig @ W.T + b  (the per-round shift, fit from real pairs)
    n = len(zo); ntr = int(0.8 * n)
    reg = Ridge(alpha=RIDGE_ALPHA).fit(zo[:ntr], zr[:ntr])
    r2 = reg.score(zo[ntr:], zr[ntr:])  # how well a linear map reproduces the rewrite
    W = torch.tensor(reg.coef_, dtype=torch.float32)
    b = torch.tensor(reg.intercept_, dtype=torch.float32)
    sv = np.linalg.svd(reg.coef_, compute_uv=False)
    # CALIBRATION: one-pass diversity ratio, surrogate vs the REAL rewrite
    surr_once = zo @ reg.coef_.T + reg.intercept_
    cal = {
        "r2": float(r2),
        "real_div_ratio": _var(zr) / _var(zo),          # real GPT-4o rewrite, one pass
        "surrogate_div_ratio": _var(surr_once) / _var(zo),  # our map, one pass
    }
    exp = d["experience"]
    y_occ = torch.tensor(d["occupation"], dtype=torch.float32).unsqueeze(-1)
    y_exp = torch.tensor((exp > np.nanmedian(exp)).astype(np.float32)).unsqueeze(-1)
    x = torch.tensor(zo, dtype=torch.float32)
    return x, {"occupation": y_occ, "experience": y_exp}, (W, b), sv, cal


def fresh_predictor() -> Predictor:
    model = LogisticModel(in_features=PCA_DIM)
    learner = ERMLearner(model, BCELoss(), max_iter=300)
    return Predictor(model=model, loss=BCELoss(), learner=learner)


def main() -> None:
    if not EMB.exists():
        raise SystemExit(f"missing {EMB}; run embed_and_cache.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    x, labels, (W, b), sv, cal = load()
    y = labels[LABEL]
    print(f"resumes={x.shape[0]} pca_dim={PCA_DIM}  label={LABEL} balance={float(y.mean()):.2f}")
    print(f"surrogate map singular values: max={sv[0]:.3f} mean={sv.mean():.3f} min={sv[-1]:.3f}")
    print(f"CALIBRATION: surrogate R2 on held-out rewrites={cal['r2']:.3f}  | "
          f"one-pass diversity ratio  real={cal['real_div_ratio']:.3f}  "
          f"surrogate={cal['surrogate_div_ratio']:.3f}  (these should match)")
    for name, yy in labels.items():
        print(f"  baseline recoverability ({name}): {recoverability({'x': x, 'y': yy}):.3f}")
    print()

    results = {}
    for regime in REGIMES:
        world = MediationWorld(
            x, y, mediator=LinearSurrogateMediator(W, b), self_consuming=True,
        )
        pred = fresh_predictor()
        div, rec, raw = [], [], []
        run_mediated(
            world, pred, n_rounds=N_ROUNDS, regime=regime, seed=0, alpha=ANCHOR_ALPHA,
            probes={
                "rec": recoverability,
                "div": lambda d: style_variance(d, world.style_mask),
            },
            on_round=lambda t, r: (div.append(r["div"]), rec.append(r["rec"]),
                                   raw.append(model_auc(pred.model, world.raw_data))),
        )
        results[regime] = {"diversity": div, "recoverability": rec, "raw_auc": raw}
        print(f"{regime:14s} diversity {div[0]:.2f}->{div[-1]:.2f}   "
              f"recoverability {rec[0]:.3f}->{rec[-1]:.3f}   raw_auc {raw[0]:.3f}->{raw[-1]:.3f}")

    (OUT / "collapse_loop_trajectory.json").write_text(json.dumps(
        {"config": {"pca_dim": PCA_DIM, "ridge": RIDGE_ALPHA, "n_rounds": N_ROUNDS,
                    "label": LABEL, "anchor_alpha": ANCHOR_ALPHA},
         "singular_values": sv.tolist(), "results": results}, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for regime in REGIMES:
        r = results[regime]
        ax[0].plot(r["diversity"], marker="o", ms=3, label=regime)
        ax[1].plot(r["recoverability"], marker="o", ms=3, label=regime)
        ax[2].plot(r["raw_auc"], marker="o", ms=3, label=regime)
    ax[0].set_title("population diversity (style variance)")
    ax[1].set_title(f"recoverability of {LABEL}")
    ax[2].set_title("screener AUC on real (pre-AI) data")
    for a in ax:
        a.set_xlabel("round"); a.legend(); a.grid(alpha=0.3)
    fig.suptitle("Self-consuming AI-mediation loop on real resumes "
                 "(surrogate calibrated to GPT-4o rewrites)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "collapse_loop.png", dpi=140)
    print(f"\nsaved {OUT/'collapse_loop.png'} and trajectory json")


if __name__ == "__main__":
    main()
