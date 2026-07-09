"""Block 27: endogenous population baseline (no AI, no gate feedback).
Deffuant alone on the real ML-Action graph across eps and gamma. This is the
observer floor for the two-object story: any AI-driven diversity loss must be
read AGAINST this curve, not against the initial std. Shows that the trap cell
eps=0.40 self-collapses to dr~0.22 with no model at all, while eps~0.10 leaves
the population alive (dr~0.71) and susceptible.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/27_pop_eps_baseline.py
"""
import importlib.util, os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("rm", "experiments/real_mlp.py")
rm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rm)

ROUNDS, SEEDS = 30, [0, 1, 2]
EPS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
CURVE_EPS = [0.05, 0.10, 0.20, 0.40]       # subset drawn as trajectories
GAMMAS = [0.0, 1.0]

ds = rm.ml_dataset("Action")
G, x0 = ds["G"], np.asarray(ds["x0"], float)
N = len(x0); init = x0.std()


def endog(eps, gamma):
    """mean std-trajectory over seeds (length ROUNDS)."""
    curves = []
    for s in SEEDS:
        pop = rm.build_pop(G, eps, gamma, x0, s)
        stds = []
        for _ in range(ROUNDS):
            pop.iteration(node_status=False)
            x = np.array([pop.status[i] for i in range(N)])
            stds.append(x.std())
        curves.append(stds)
    return np.mean(curves, 0)


data = {g: {e: endog(e, g) for e in EPS} for g in GAMMAS}

print(f"ML-Action  N={N}  innate std={init:.3f}   (eps/std: 0.10={0.10/init:.1f}  0.40={0.40/init:.1f})")
for g in GAMMAS:
    print(f"\ngamma={g}")
    for e in EPS:
        print(f"  eps={e:.2f}  final dr={data[g][e][-1]/init:.3f}")

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)

# Panel A: endogenous div_ratio over rounds, gamma=0, sequential ramp by eps
ramp = plt.cm.GnBu(np.linspace(0.4, 0.95, len(CURVE_EPS)))
for e, c in zip(CURVE_EPS, ramp):
    y = data[0.0][e] / init
    axA.plot(np.arange(1, ROUNDS + 1), y, color=c, lw=2.4, label=f"{e:.2f}")
axA.set(xlabel="round", ylabel="diversity kept  (std / initial std)", ylim=(0, 1.02))
axA.legend(title="$\\epsilon$ (social)", frameon=False, fontsize=9, title_fontsize=10)
axA.set_title("Endogenous collapse (no AI): wider $\\epsilon$ = faster consensus", fontsize=12)

# Panel B: final div_ratio vs eps, gamma 0 vs 1
for g, mk, ls, col in [(0.0, "o", "-", "#2f6f9f"), (1.0, "s", "--", "#e08214")]:
    ys = [data[g][e][-1] / init for e in EPS]
    axB.plot(EPS, ys, ls, marker=mk, color=col, lw=2.2, ms=7, label=f"$\\gamma$={g:.0f}")
axB.axvspan(0.08, 0.13, color="#cfe6cf", alpha=0.6, zorder=0)
axB.text(0.105, 0.05, "alive +\nsusceptible", ha="center", va="bottom", fontsize=8, color="#3a6b3a")
for e, lab in [(0.10, "clean cell\n$\\epsilon$=0.10"), (0.40, "trap cell\n$\\epsilon$=0.40")]:
    axB.axvline(e, color="#999", lw=1, ls=":")
    axB.text(e, 1.0, lab, ha="center", va="top", fontsize=8, color="#555")
axB.set(xlabel="$\\epsilon$ (social confidence radius)", ylabel="final diversity kept", ylim=(0, 1.05))
axB.legend(frameon=False, fontsize=11, loc="center right")
axB.set_title("Homophily ($\\gamma$) rescues diversity at every $\\epsilon$", fontsize=12)

fig.suptitle("ML-Action population baseline with no AI: the trap cell self-collapses; "
             "$\\epsilon\\approx$0.10 keeps the population alive", fontsize=13)
out = "experiments/competition/figs/pop_eps_baseline.png"
fig.savefig(out, dpi=140); plt.close(fig)
print(f"\nsaved {out}")
