"""Block 6: speed-capped platforms chasing a drifting population.

Two point platforms. The population's innate opinions rotate at speed u per
round (opinion drift). Each platform steps toward its own customers' circular
mean, capped at v_k per round (retraining-cadence budget). Population does the
circular FJ step toward drifted innate + served platform + peers.

Outcomes per (u, v_slow): both track / slow platform locked out / secession
(slow platform tears off and keeps a splinter crowd).

Run: python experiments/competition/06_speed_drift.py
"""

import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_spec = importlib.util.spec_from_file_location("b2", "experiments/competition/circle/02_phase_diagram.py")
b2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b2)

OUT = Path("experiments/competition/circle/figs")
TWO_PI = 2 * math.pi

N = 2000
TAU = 0.2
MALL = 0.45
PEER_SHARE = 0.3
ROUNDS = 200
V_FAST = 0.05
KAPPA = 4.0


def circle_dist(a, b):
    d = (a - b).abs() % 1.0
    return torch.minimum(d, 1.0 - d)


def circ_mean(x):
    v = b2.to_vec(x).mean(0)
    return float(torch.atan2(v[1], v[0]) / TWO_PI % 1.0)


def step_toward(p, target, cap):
    d = (target - p + 0.5) % 1.0 - 0.5
    return (p + float(np.clip(d, -cap, cap))) % 1.0


def splinter_mass(x, pos_slow):
    return float((circle_dist(x, torch.tensor(pos_slow)) < 0.1).float().mean())


def run(u, v_slow, seed=0):
    g = np.random.default_rng(seed)
    torch.manual_seed(seed)
    innate0 = torch.tensor(g.vonmises(TWO_PI * 0.5, KAPPA, N) / TWO_PI % 1.0,
                           dtype=torch.float32)
    w_graph = b2.random_graph(N, 10, seed)
    x = innate0.clone()
    pos = [0.5, 0.45]  # fast, slow; both start near the camp
    caps = [V_FAST, v_slow]
    gen = torch.Generator().manual_seed(seed)
    hist = []
    for t in range(ROUNDS):
        innate_t = (innate0 + u * t) % 1.0
        pt = torch.tensor(pos)
        d = circle_dist(pt[:, None], x[None, :])
        assign = torch.multinomial(F.softmax(-d.t() / TAU, dim=1), 1, generator=gen).squeeze(1)
        for k in range(2):
            mine = x[assign == k]
            if len(mine) > 5:
                pos[k] = step_toward(pos[k], circ_mean(mine), caps[k])
        served = pt[assign]
        z0 = b2.to_vec(innate_t)
        zp = b2.to_vec(served)
        target = (1 - MALL) * z0 + MALL * (1 - PEER_SHARE) * zp \
            + MALL * PEER_SHARE * (w_graph @ b2.to_vec(x))
        x = b2.to_angle(target)
        hist.append({
            "share_slow": float((assign == 1).float().mean()),
            "lag_slow": float(circle_dist(torch.tensor(pos[1]),
                                          torch.tensor(circ_mean(innate_t)))),
            "pos": list(pos),
            "splinter": splinter_mass(x, pos[1]),
            "pop_var": b2.circ_var(x),
        })
    return hist


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    print(f"drift u x slow speed (fast={V_FAST}); camp kappa={KAPPA}, {ROUNDS} rounds")
    print(f"{'u':>6} {'v_slow':>7} | {'share_slow':>10} {'lag_slow':>8} {'splinter':>8}")
    for u in (0.005, 0.01, 0.02):
        for v_slow in (0.05, 0.02, 0.01, 0.005, 0.002):
            h = run(u, v_slow)
            last = h[-20:]
            row = {k: float(np.mean([r[k] for r in last])) for k in
                   ("share_slow", "lag_slow", "splinter")}
            res[f"{u}|{v_slow}"] = h
            print(f"{u:>6.3f} {v_slow:>7.3f} | {row['share_slow']:>10.2f} "
                  f"{row['lag_slow']:>8.3f} {row['splinter']:>8.2f}", flush=True)
    json.dump({k: v[-1] for k, v in res.items()}, open(OUT / "speed_drift.json", "w"))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for u, c in [(0.005, "#1f77b4"), (0.01, "#2ca02c"), (0.02, "#d62728")]:
        for v_slow, ls in [(0.02, "-"), (0.005, "--")]:
            h = res[f"{u}|{v_slow}"]
            ax[0].plot([r["share_slow"] for r in h], c=c, ls=ls, lw=1.3,
                       label=f"u={u} v_slow={v_slow}")
            ax[1].plot([r["lag_slow"] for r in h], c=c, ls=ls, lw=1.3)
            ax[2].plot([r["splinter"] for r in h], c=c, ls=ls, lw=1.3)
    ax[0].set(xlabel="round", ylabel="slow platform share", title="share")
    ax[1].set(xlabel="round", ylabel="slow dist to camp", title="lag")
    ax[2].set(xlabel="round", ylabel="mass within 0.1 of slow", title="splinter crowd")
    ax[0].legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "speed_drift.png", dpi=130)
    print(f"saved {OUT / 'speed_drift.png'}")


if __name__ == "__main__":
    main()
