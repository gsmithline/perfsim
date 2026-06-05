"""Dynamic Hotelling on the real Pokec graph: lloyd vs strategic platforms,
drift and peers on/off.
Run: python experiments/fj/run_hotelling.py
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.environments.dynamics.fj import FJWorld, normalize_adjacency

OUT = Path("runs/fj_beach")
POKEC = Path("examples/pokec")
SEEDS = (0, 1, 2)
ROUNDS = 60
K_INNER = 10
TEMP = 0.05
LR_STRATEGIC = 0.1


class FixedPredictions(Model):
    def __init__(self, values: Tensor) -> None:
        super().__init__()
        self.values = values

    def forward(self, x: Tensor) -> Tensor:
        return self.values


def load_pokec() -> dict:
    with open(POKEC / "lcc_profiles_relation_to_smoking.pk", "rb") as fh:
        df = pickle.load(fh)
    with open(POKEC / "lcc_graph_relation_to_smoking.pk", "rb") as fh:
        graph = pickle.load(fh)
    pp = POKEC / "parametric_params"
    with open(pp / "y_label2163.pk", "rb") as fh:
        y_lab = pickle.load(fh)
    with open(pp / "y_unlabel_label2163.pk", "rb") as fh:
        y_unlab = pickle.load(fh)
    with open(pp / "hetero_peer_sus2163.pkl", "rb") as fh:
        peer_sus = pickle.load(fh)
    with open(pp / "hetero_platform_sus2163.pkl", "rb") as fh:
        platform_sus = pickle.load(fh)
    innate = torch.tensor(
        np.asarray(list(y_lab) + list(y_unlab), dtype=np.float64), dtype=torch.float32
    )
    adj = nx.to_numpy_array(graph, nodelist=df["user_id"].tolist())
    return {
        "innate": innate,
        "W": normalize_adjacency(torch.tensor(adj, dtype=torch.float32)),
        "peer_sus": torch.tensor(np.asarray(peer_sus), dtype=torch.float32),
        "platform_sus": torch.tensor(np.asarray(platform_sus), dtype=torch.float32),
    }


def patronage(b: Tensor, x: Tensor) -> Tensor:
    """Softmax-on-proximity choice probabilities, (N, P)."""
    return torch.softmax(-(b.unsqueeze(0) - x.unsqueeze(1)).abs() / TEMP, dim=1)


def run_market(setup: dict, *, update: str, drift: bool, peers: bool, seed: int,
               beach_bins: int = 0, b0: tuple[float, float] | None = None) -> dict:
    innate = setup["innate"]
    n = innate.shape[0]
    peer_sus = setup["peer_sus"] if peers else torch.ones(n)
    plat_sus = setup["platform_sus"] if drift else 0.0
    world = FJWorld(innate, setup["W"], peer_sus, platform_sus=plat_sus)
    world.reset(seed=seed)
    gen = torch.Generator()
    gen.manual_seed(seed + 1000)
    q = torch.quantile(innate, torch.tensor([0.25, 0.75]))
    b = q.clone() if b0 is None else torch.tensor(b0)
    x = innate.clone()
    out = {"gap": [], "std": [], "b": []}
    if beach_bins:
        out["beach"] = []
    for _ in range(ROUNDS):
        probs = patronage(b, x)
        assign = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        data = world.run(FixedPredictions(b[assign]), n_steps=K_INNER)
        x = data["y"].reshape(-1)
        if update == "lloyd":
            b_next = b.clone()
            for p in range(2):
                mask = assign == p
                if int(mask.sum()) > 0:
                    b_next[p] = x[mask].mean()
            b = b_next
        else:
            # each platform climbs its own expected share (gradient play)
            bg = b.clone().requires_grad_(True)
            s = patronage(bg, x)
            g = torch.stack([
                torch.autograd.grad(s[:, p].mean(), bg, retain_graph=(p == 0))[0][p]
                for p in range(2)
            ])
            b = (b + LR_STRATEGIC * g).detach()
        out["gap"].append(float((b[0] - b[1]).abs()))
        out["std"].append(float(x.std()))
        out["b"].append([float(v) for v in b])
        if beach_bins:
            out["beach"].append(torch.histc(x, bins=beach_bins, min=0.0, max=1.0))
    out["x_final"] = x
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup = load_pokec()
    innate = setup["innate"]
    print(f"pokec: N={innate.shape[0]}  innate mean={float(innate.mean()):.3f} "
          f"std={float(innate.std()):.3f}  range=({float(innate.min()):.2f}, "
          f"{float(innate.max()):.2f})")

    cells = [
        ("strategic", False, True), ("strategic", True, True),
        ("strategic", True, False), ("lloyd", False, True), ("lloyd", True, True),
    ]
    runs = {}
    for update, drift, peers in cells:
        key = f"{update} drift={'on' if drift else 'off'} peers={'on' if peers else 'off'}"
        runs[key] = [run_market(setup, update=update, drift=drift, peers=peers, seed=s)
                     for s in SEEDS]
        gap = sum(r["gap"][-1] for r in runs[key]) / len(SEEDS)
        std = sum(r["std"][-1] for r in runs[key]) / len(SEEDS)
        print(f"{key:>34s} | final gap {gap:.3f}  beach std {std:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = plt.cm.tab10.colors
    for i, (key, rs) in enumerate(runs.items()):
        curves = torch.tensor([r["gap"] for r in rs])
        axes[0].plot(curves.mean(dim=0), color=colors[i], label=key)
        curves = torch.tensor([r["std"] for r in rs])
        axes[1].plot(curves.mean(dim=0), color=colors[i], label=key)
    axes[0].set_title("|b1 - b2|: differentiation")
    axes[0].set_xlabel("round")
    axes[0].legend(fontsize=7)
    axes[1].set_title("beach std")
    axes[1].set_xlabel("round")
    fig.suptitle("Dynamic Hotelling on Pokec: minimum differentiation vs drift",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "hotelling_pokec.png", dpi=140)
    print(f"[fj] figure -> {OUT / 'hotelling_pokec.png'}  (seeds {SEEDS})")


if __name__ == "__main__":
    main()
