"""Block 8c: hysteresis test for the HK x anchored-MLP loop.

Ramp protocol: one continual run (MLP weights persist) with epsilon ramped
1.0 -> 0.1 over 120 rounds then back up over 120. HK alone is supposed to be
memoryless in epsilon except for frozen clusters; the platform's trained
weights are extra state that could open (or close) a hysteresis loop.
Fine scan: fixed-epsilon runs around the 2-cluster -> consensus transition
to see whether platform weight w shifts the critical epsilon.

Run: python experiments/competition/08c_hysteresis.py
"""

import importlib.util
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "experiments/competition")
_spec = importlib.util.spec_from_file_location(
    "b8", "experiments/competition/08_ndlib_mlp_loop.py")
b8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b8)

N = b8.N
RAMP_STEPS = 120
RAMP_W = [0.0, 0.3]
RAMP_SEEDS = [0, 1]

SCAN_EPS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
SCAN_W = [0.0, 0.3, 0.5]
SCAN_SEEDS = [0, 1, 2]
SCAN_ROUNDS = 60

_base_cache = {}


def get_nets(rng, seed):
    # pretrains are deterministic per seed (rng reaches them in the same
    # state every run), so cache the two state dicts
    if seed not in _base_cache:
        _base_cache[seed] = (b8.pretrain_base(rng, seed).state_dict(),
                             b8.pretrain_base(rng, seed).state_dict())
    base, net = b8.mlp(), b8.mlp()
    base.load_state_dict(_base_cache[seed][0])
    net.load_state_dict(_base_cache[seed][1])
    return base, net


def run_loop(eps_schedule, w, seed):
    pop = b8.build_pop(eps_schedule[0], seed)
    rng = np.random.default_rng(seed)
    x0 = np.array([pop.status[i] for i in range(N)])
    feats = torch.tensor(b8.make_features(x0, rng), dtype=torch.float32)
    base, net = get_nets(rng, seed)
    base_pred = base(feats).squeeze(1).detach()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    traj = []
    for eps in eps_schedule:
        pop.params["model"]["epsilon"] = float(eps)
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        target = torch.tensor(x, dtype=torch.float32)
        for _ in range(b8.TRAIN_STEPS):
            opt.zero_grad()
            pred = net(feats).squeeze(1)
            loss = ((pred - target) ** 2).mean() \
                + b8.ANCHOR_W * ((pred - base_pred) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(feats).squeeze(1).numpy()
        gate = np.abs(m - x) < eps
        if w > 0:
            x = np.where(gate, (1 - w) * x + w * m, x)
            for i in range(N):
                pop.status[i] = float(x[i])
        traj.append({
            "epsilon": float(eps),
            "pop_mean": float(x.mean()), "pop_std": float(x.std()),
            "clusters": b8.n_clusters(x), "contact": float(gate.mean()),
            "pred_std": float(m.std()), "loss": float(loss.item()),
        })
    return traj


def main():
    eps_down = np.linspace(1.0, 0.1, RAMP_STEPS)
    eps_up = np.linspace(0.1, 1.0, RAMP_STEPS)
    # hot start: down then up (the spec); cold start: population freezes into
    # clusters at eps=0.1 first, then we ask whether they re-merge on the way up
    schedules = {"hot": np.concatenate([eps_down, eps_up]),
                 "cold": np.concatenate([eps_up, eps_down])}

    ramp = {}
    gaps = {}
    for name, schedule in schedules.items():
        print(f"ramp [{name}]: eps {schedule[0]:.1f} -> {schedule[RAMP_STEPS - 1]:.1f} "
              f"-> {schedule[-1]:.1f}, {2 * RAMP_STEPS} rounds, w {RAMP_W}, "
              f"seeds {RAMP_SEEDS}")
        print(f"{'w':>5} {'seed':>4} | {'gap pop_std':>11} {'gap contact':>11} "
              f"{'gap pred_std':>12} | {'end std':>7} {'end clu':>7}")
        for w in RAMP_W:
            for s in RAMP_SEEDS:
                t = run_loop(schedule, w, s)
                ramp[f"{name}|{w}|{s}"] = t
                leg1, leg2 = t[:RAMP_STEPS], t[RAMP_STEPS:][::-1]  # same eps order
                g = {f: float(np.mean([abs(a[f] - b[f]) for a, b in zip(leg1, leg2)]))
                     for f in ["pop_std", "contact", "pred_std"]}
                gaps[f"{name}|{w}|{s}"] = g
                print(f"{w:>5.2f} {s:>4d} | {g['pop_std']:>11.4f} "
                      f"{g['contact']:>11.4f} {g['pred_std']:>12.4f} | "
                      f"{t[-1]['pop_std']:>7.3f} {t[-1]['clusters']:>7d}", flush=True)

    scan = {}
    print(f"\nfine scan: fixed eps, {SCAN_ROUNDS} rounds, seeds {SCAN_SEEDS}")
    print(f"{'eps':>5} | " + "  ".join(
        f"w={w}: clu std    " for w in SCAN_W))
    for eps in SCAN_EPS:
        row = []
        for w in SCAN_W:
            for s in SCAN_SEEDS:
                t = run_loop([eps] * SCAN_ROUNDS, w, s)
                scan[f"{eps}|{w}|{s}"] = {
                    "clusters": t[-1]["clusters"],
                    "pop_std": float(np.mean([r["pop_std"] for r in t[-5:]])),
                    "contact": float(np.mean([r["contact"] for r in t[-5:]])),
                }
            clu = np.mean([scan[f"{eps}|{w}|{s}"]["clusters"] for s in SCAN_SEEDS])
            std = np.mean([scan[f"{eps}|{w}|{s}"]["pop_std"] for s in SCAN_SEEDS])
            row.append(f"w={w}: {clu:.1f} {std:.3f}")
        print(f"{eps:>5.2f} | " + "  ".join(f"{r:<18}" for r in row), flush=True)

    crit = {}
    for w in SCAN_W:
        cons = [eps for eps in SCAN_EPS if all(
            scan[f"{eps}|{w}|{s}"]["clusters"] <= 1 for s in SCAN_SEEDS)]
        crit[str(w)] = cons[0] if cons else None
        print(f"critical eps (all seeds 1 cluster) w={w}: {crit[str(w)]}")

    json.dump({"ramp": ramp, "gaps": gaps, "scan": scan, "critical_eps": crit},
              open("experiments/competition/figs/hk_hysteresis.json", "w"))

    fig, ax = plt.subplots(4, 4, figsize=(17, 13), constrained_layout=True)
    rows = [(name, w) for name in schedules for w in RAMP_W]
    for r, (name, w) in enumerate(rows):
        for c, f in enumerate(["pop_std", "contact", "pred_std"]):
            for s in RAMP_SEEDS:
                t = ramp[f"{name}|{w}|{s}"]
                e = [p["epsilon"] for p in t]
                v = [p[f] for p in t]
                ax[r, c].plot(e[:RAMP_STEPS], v[:RAMP_STEPS], c="#1f77b4",
                              alpha=0.8, label="leg 1" if s == 0 else None)
                ax[r, c].plot(e[RAMP_STEPS:], v[RAMP_STEPS:], c="#d62728",
                              ls="--", alpha=0.8, label="leg 2" if s == 0 else None)
            ax[r, c].set(xlabel="epsilon", ylabel=f,
                         title=f"{name} start, w={w}")
            ax[r, c].legend(frameon=False, fontsize=8)
    colors = {0.0: "#7f7f7f", 0.3: "#d62728", 0.5: "#9467bd"}
    for r, (f, ylab) in enumerate([("clusters", "final clusters"),
                                   ("pop_std", "final pop_std")]):
        for w in SCAN_W:
            v = [np.mean([scan[f"{eps}|{w}|{s}"][f] for s in SCAN_SEEDS])
                 for eps in SCAN_EPS]
            ax[r, 3].plot(SCAN_EPS, v, "o-", c=colors[w], label=f"w={w}")
        ax[r, 3].set(xlabel="epsilon", ylabel=ylab, title="fixed-eps scan")
        ax[r, 3].legend(frameon=False, fontsize=8)
    ax[2, 3].axis("off")
    ax[3, 3].axis("off")
    fig.suptitle("HK x anchored MLP: epsilon ramp hysteresis and consensus threshold")
    fig.savefig("experiments/competition/figs/hk_hysteresis.png", dpi=130)
    print("saved experiments/competition/figs/hk_hysteresis.png")


if __name__ == "__main__":
    main()
