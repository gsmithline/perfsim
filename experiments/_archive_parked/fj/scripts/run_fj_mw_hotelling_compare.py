"""Same model (FJ + MW agents + Hotelling), two graphs: real Pokec vs a
synthetic 2-block graph with weak cross-community coupling. Susceptibilities
matched (0.89) so ONLY topology differs. Tests whether the modality ->
merge/segment transition that Pokec washes out reappears on a graph that does
not self-collapse.
Run: python experiments/fj/run_fj_mw_hotelling_compare.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from perfsim.environments.dynamics.fj import normalize_adjacency
from _pokec import load_pokec
from run_fj_mw_hotelling import SEEDS, SEPS, SEG, fiedler_split, run_market

P_IN = 0.01
P_OUT = 0.0005
SUS = 0.89


def synth_setup(n, seed):
    gen = torch.Generator(); gen.manual_seed(seed)
    community = (torch.arange(n) >= n // 2).long()
    same = community.unsqueeze(0) == community.unsqueeze(1)
    p = torch.where(same, P_IN, P_OUT)
    adj = (torch.rand(n, n, generator=gen) < p).float()
    adj = torch.triu(adj, diagonal=1); adj = adj + adj.t()
    setup = {"adj": adj, "W": normalize_adjacency(adj),
             "peer_sus": torch.full((n,), SUS), "platform_sus": torch.full((n,), SUS)}
    return setup, community


def sweep(setup, community, label):
    print(f"--- {label} ---")
    print(f"{'mode_sep':>9s} | {'mean_div':>9s} {'pop_std':>8s}  per-seed div")
    xs, dv, st = [], [], []
    for sep in SEPS:
        runs = [run_market(setup, community, sep, s) for s in SEEDS]
        msep = sum(r["mode_sep"] for r in runs) / len(runs)
        dvs = [r["div"] for r in runs]
        pstd = sum(r["pop_std"] for r in runs) / len(runs)
        xs.append(msep); dv.append(sum(dvs) / len(dvs)); st.append(pstd)
        print(f"{msep:>9.3f} | {sum(dvs)/len(dvs):>9.3f} {pstd:>8.3f}  {[round(d,2) for d in dvs]}")
    return xs, dv, st


def main() -> None:
    out = Path("runs/fj_mw_hotelling"); out.mkdir(parents=True, exist_ok=True)
    pk = load_pokec()
    pk_comm = fiedler_split(pk["adj"])
    n = pk_comm.shape[0]
    sy, sy_comm = synth_setup(n, 0)
    print(f"N={n}  pokec(dense connected) vs synthetic(p_in={P_IN}, p_out={P_OUT}), sus={SUS}\n")

    xp, dp, sp = sweep(pk, pk_comm, "POKEC")
    xs, ds, ss = sweep(sy, sy_comm, "SYNTHETIC block graph")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax[0].plot(xp, dp, "o-", color="#d8633b", label="Pokec (dense)")
    ax[0].plot(xs, ds, "o-", color="#3b6fd8", label="synthetic (weak cross)")
    ax[0].plot(xp, xp, ls=":", color="gray", lw=1, label="platforms at modes")
    ax[0].set_xlabel("injected mode separation"); ax[0].set_ylabel("final platform div |b_p - b_q|")
    ax[0].set_title("same model, two graphs: does modality drive segment?")
    ax[0].legend(fontsize=8)
    ax[1].plot(xp, sp, "o-", color="#d8633b", label="Pokec (dense)")
    ax[1].plot(xs, ss, "o-", color="#3b6fd8", label="synthetic (weak cross)")
    ax[1].set_xlabel("injected mode separation"); ax[1].set_ylabel("final population std")
    ax[1].set_title("population spread: Pokec self-collapses")
    ax[1].legend(fontsize=8)
    for a in ax:
        for spn in ("top", "right"):
            a.spines[spn].set_visible(False)
    fig.suptitle("FJ + MW + Hotelling: topology decides whether modality survives",
                 fontweight="bold")
    fig.savefig(out / "graph_compare.png", dpi=150)
    print(f"\n[compare] figure -> {out / 'graph_compare.png'}")


if __name__ == "__main__":
    main()
