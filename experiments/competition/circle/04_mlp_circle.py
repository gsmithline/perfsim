"""Block 4: learning platforms. K small MLPs compete for a circular-FJ population.

Each platform is an MLP (features -> opinion on the circle), fine-tuned each
round on the customers it won, with an output anchor toward its own pretrained
base net. Base nets are pretrained on corpora whose opinions are biased toward
centers spread delta apart: delta is prior separation. Conditions: shared (one
base for all), pair (two share, one apart), distinct (all spread delta).

Population: real Pokec graph, circular FJ (02's step), hard platform choice.

Run: python experiments/competition/circle/04_mlp_circle.py
"""

import copy
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "experiments/fj")
from _pokec import load_pokec  # noqa: E402

_spec = importlib.util.spec_from_file_location("b2", "experiments/competition/circle/02_phase_diagram.py")
b2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b2)

OUT = Path("experiments/competition/circle/figs")
TWO_PI = 2 * math.pi

TAU = 0.2
MALL = 0.45
PEER_SHARE = 0.5
ROUNDS = 50
K = 3
SIGMA_F = 0.15
ANCHOR_W = 0.3
TRAIN_STEPS = 15
SEEDS = [0, 1]
C0 = 0.5


def vonmises(center, kappa, n, rng):
    return (rng.vonmises(TWO_PI * center, kappa, n) / TWO_PI) % 1.0


def make_innate(kind, n, rng):
    if kind == "uniform":
        return rng.random(n)
    if kind == "bimodal":
        camps = np.where(rng.random(n) < 0.5, 0.0, 0.5)
        return vonmises(camps, 8.0, n, rng)
    if kind == "bimodal64":
        camps = np.where(rng.random(n) < 0.6, 0.0, 0.5)
        return vonmises(camps, 8.0, n, rng)
    if kind == "concentrated":
        return vonmises(0.25, 8.0, n, rng)
    raise ValueError(kind)


def make_features(angles, rng, sigma_f=SIGMA_F):
    f = angles + rng.normal(0, sigma_f, angles.shape)
    noise = rng.normal(0, 1.0, (len(angles), 2))
    return np.stack([np.cos(TWO_PI * f), np.sin(TWO_PI * f), noise[:, 0], noise[:, 1]], 1)


def mlp():
    return nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2))


def pred_vec(net, feats):
    return F.normalize(net(feats), dim=-1)


def circ_loss(v_pred, v_target):
    return (1.0 - (v_pred * v_target).sum(-1)).mean()


def pretrain_base(center, rng, seed, n_corpus=4000, steps=400):
    torch.manual_seed(seed)
    y = vonmises(center, 2.0, n_corpus, rng)
    feats = torch.tensor(make_features(y, rng), dtype=torch.float32)
    target = b2.to_vec(torch.tensor(y, dtype=torch.float32))
    net = mlp()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = circ_loss(pred_vec(net, feats), target)
        loss.backward()
        opt.step()
    return net


def centers_for(cond, delta):
    if cond == "shared":
        return [C0] * K
    if cond == "pair":
        return [C0, C0, (C0 + delta) % 1.0]
    if cond == "distinct":
        return [(C0 + (i - 1) * delta) % 1.0 for i in range(K)]
    raise ValueError(cond)


def run_cell(cond, delta, innate_kind, seed, sigma_f=SIGMA_F, centers=None, prox_w=None):
    rng = np.random.default_rng(seed)
    pk = load_pokec()
    w_graph = pk["W"]
    n = w_graph.shape[0]
    innate = torch.tensor(make_innate(innate_kind, n, rng), dtype=torch.float32)
    feats = torch.tensor(make_features(innate.numpy(), rng, sigma_f), dtype=torch.float32)

    if centers is None:
        centers = centers_for(cond, delta)
    bases = [pretrain_base(c, rng, seed * 10 + i) for i, c in enumerate(centers)]
    platforms = [copy.deepcopy(b) for b in bases]
    base_preds = [pred_vec(b, feats).detach() for b in bases]
    return _loop(platforms, base_preds, feats, innate.clone(), innate, w_graph, seed,
                 prox_w=prox_w)


def run_cell_merged(centers, seed, sigma_f=SIGMA_F, innate_kind="uniform"):
    """Merged start: population concentrated at C0, all platforms from one C0 net.

    Anchor losses still pull each platform toward its own base, so this asks
    whether segmentation re-forms out of a collapsed state.
    """
    rng = np.random.default_rng(seed)
    pk = load_pokec()
    w_graph = pk["W"]
    n = w_graph.shape[0]
    innate = torch.tensor(make_innate(innate_kind, n, rng), dtype=torch.float32)
    feats = torch.tensor(make_features(innate.numpy(), rng, sigma_f), dtype=torch.float32)
    bases = [pretrain_base(c, rng, seed * 10 + i) for i, c in enumerate(centers)]
    merged = pretrain_base(C0, rng, seed * 10 + 7)
    platforms = [copy.deepcopy(merged) for _ in range(K)]
    base_preds = [pred_vec(b, feats).detach() for b in bases]
    x0 = torch.tensor(vonmises(C0, 8.0, n, rng), dtype=torch.float32)
    return _loop(platforms, base_preds, feats, x0, innate, w_graph, seed)


def _loop(platforms, base_preds, feats, x, innate, w_graph, seed, prox_w=None):
    """prox_w: per-platform cost for chasing -- extra loss toward the platform's
    own previous-round predictions (inertia). None = free movement."""
    n = len(innate)
    opts = [torch.optim.Adam(p.parameters(), lr=3e-3) for p in platforms]
    gen = torch.Generator().manual_seed(seed)
    prev_preds = [pred_vec(p, feats).detach() for p in platforms]
    traj = []
    for t in range(ROUNDS):
        with torch.no_grad():
            pv = torch.stack([pred_vec(p, feats) for p in platforms])     # (K, N, 2)
            pa = b2.to_angle(pv)                                          # (K, N)
        d = circle_dist(pa, x.unsqueeze(0))
        assign = torch.multinomial(F.softmax(-d.t() / TAU, dim=1), 1, generator=gen).squeeze(1)

        for k in range(K):
            idx = (assign == k).nonzero(as_tuple=True)[0]
            if len(idx) < 2:
                continue
            tv = b2.to_vec(x[idx])
            for _ in range(TRAIN_STEPS):
                opts[k].zero_grad()
                out = pred_vec(platforms[k], feats[idx])
                loss = circ_loss(out, tv) + ANCHOR_W * circ_loss(out, base_preds[k][idx])
                if prox_w is not None and prox_w[k] > 0:
                    loss = loss + prox_w[k] * circ_loss(out, prev_preds[k][idx])
                loss.backward()
                opts[k].step()

        with torch.no_grad():
            pv = torch.stack([pred_vec(p, feats) for p in platforms])
            pa = b2.to_angle(pv)
        prev_preds = [pv[k].detach() for k in range(K)]
        served = pa[assign, torch.arange(n)]
        x = fj_step_personalized(x, innate, w_graph, served)

        plat_means = b2.to_angle(pv.mean(1))
        pdist = [float(circle_dist(plat_means[a], plat_means[b]))
                 for a in range(K) for b in range(a + 1, K)]
        traj.append({
            "pdist": pdist,
            "plat_means": [float(a) for a in plat_means],
            "round": t,
            "pred_div": float(torch.stack([circle_dist(pa[a], pa[b]).mean()
                                           for a in range(K) for b in range(a + 1, K)]).mean()),
            "gap": b2.plat_min_gap(plat_means) * K,
            "pop_var": b2.circ_var(x),
            "disp": float(circle_dist(circ_mean(x), circ_mean(innate))),
            "shares": [(assign == k).float().mean().item() for k in range(K)],
        })
    return traj


def circle_dist(a, b):
    d = (a - b).abs() % 1.0
    return torch.minimum(d, 1.0 - d)


def circ_mean(x):
    v = b2.to_vec(x).mean(0)
    return torch.atan2(v[1], v[0]) / TWO_PI % 1.0


def fj_step_personalized(x, innate, w_graph, served, inner=3):
    w_innate = 1.0 - MALL
    w_peer = MALL * PEER_SHARE
    w_plat = MALL * (1.0 - PEER_SHARE)
    z0 = b2.to_vec(innate)
    zp = b2.to_vec(served)
    for _ in range(inner):
        target = w_innate * z0 + w_plat * zp + w_peer * (w_graph @ b2.to_vec(x))
        x = b2.to_angle(target)
    return x


CELLS = (
    [("uniform", "shared", 0.0)]
    + [("uniform", "pair", 0.10)]
    + [("uniform", "distinct", d) for d in (0.03, 0.07, 0.15, 1 / 3)]
    + [("bimodal", "shared", 0.0), ("bimodal", "distinct", 1 / 3)]
    + [("concentrated", "shared", 0.0), ("concentrated", "distinct", 1 / 3)]
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    print(f"K={K} MLP platforms, Pokec graph, tau={TAU}, mall={MALL}, anchor_w={ANCHOR_W}")
    print(f"{'innate':>12} {'cond':>9} {'delta':>6} | {'div_T':>6} {'gap_T':>6} "
          f"{'var0':>6} {'var_T':>6} {'disp_T':>6}")
    for innate_kind, cond, delta in CELLS:
        rows = [run_cell(cond, delta, innate_kind, s) for s in SEEDS]
        key = f"{innate_kind}|{cond}|{delta:.3f}"
        res[key] = rows
        last = lambda f: np.mean([np.mean([r[f] for r in t[-5:]]) for t in rows])
        first = lambda f: np.mean([t[0][f] for t in rows])
        print(f"{innate_kind:>12} {cond:>9} {delta:>6.3f} | {last('pred_div'):>6.3f} "
              f"{last('gap'):>6.3f} {first('pop_var'):>6.3f} {last('pop_var'):>6.3f} "
              f"{last('disp'):>6.3f}", flush=True)
    json.dump(res, open(OUT / "mlp_circle.json", "w"))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    deltas = [0.03, 0.07, 0.15, 1 / 3]
    for f, i, ylab in [("pred_div", 0, "final platform divergence"),
                       ("pop_var", 1, "final population circ-var")]:
        mu = [np.mean([np.mean([r[f] for r in t[-5:]]) for t in res[f"uniform|distinct|{d:.3f}"]])
              for d in deltas]
        sh = np.mean([np.mean([r[f] for r in t[-5:]]) for t in res["uniform|shared|0.000"]])
        ax[i].plot(deltas, mu, "o-", label="distinct")
        ax[i].axhline(sh, ls="--", c="gray", label="shared (delta=0)")
        ax[i].set(xlabel="prior separation delta", ylabel=ylab)
        ax[i].legend(frameon=False)
    fig.savefig(OUT / "mlp_circle_delta.png", dpi=130)
    print(f"saved {OUT / 'mlp_circle_delta.png'}")


if __name__ == "__main__":
    main()
