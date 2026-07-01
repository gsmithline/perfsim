"""Complete-picture phase plane: four equilibria across synthetic and real data,
for OLS, MLP-mean, and the MLP sampler, across all gammas and training regimes.

Two panels (synthetic | real). Each point is one (dataset, gamma, regime) cell.
Color = model class. Reads the combined MLP-study JSON (synthetic + real).
"""
import json, os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = "experiments/toy/figs"
d = json.load(open("experiments/toy/results/real_mlp_results.json"))
runs = d["runs"]
SYNTH = {"Synth-strong", "Synth-weak"}
col = {"ols": "#888888", "mlp_mean": "#1f77b4", "mlp_sample": "#d62728"}
lab = {"ols": "OLS (linear mean)", "mlp_mean": "MLP (nonlinear mean)", "mlp_sample": "MLP sampler P(y|x)"}

panels = {"Synthetic": {k: ([], []) for k in col}, "Real (MovieLens, Yelp)": {k: ([], []) for k in col}}
for key, v in runs.items():
    name, g, kind, regime = key.split("|")
    if not np.isfinite(v["vr"]):
        continue
    p = "Synthetic" if name in SYNTH else "Real (MovieLens, Yelp)"
    panels[p][kind][0].append(v["dr"]); panels[p][kind][1].append(v["vr"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), constrained_layout=True)
for ax, pname in zip(axes, ["Synthetic", "Real (MovieLens, Yelp)"]):
    ax.axhline(1.0, ls="--", c="black", lw=1)
    ax.axvline(0.5, ls=":", c="gray", lw=1)
    n = 0
    for kind in ["ols", "mlp_mean", "mlp_sample"]:
        dr, vr = panels[pname][kind]
        n += len(dr)
        ax.scatter(dr, np.clip(vr, 0, 3.2), c=col[kind], s=34, alpha=0.65, edgecolors="none", label=lab[kind])
    for x, y, t in [(0.03, 0.08, "1 both\ncollapsed"), (0.72, 0.08, "3 pop diverse,\nplatform below"),
                    (0.72, 1.10, "2 both\ndiverse"), (0.03, 2.7, "4 pop collapsed,\nplatform wide")]:
        ax.text(x, y, t, fontsize=9, color="#333", va="bottom", fontweight="bold")
    ax.set(xlabel="population spread (dr)", ylabel="platform vs population (vr)",
           xlim=(-0.02, 1.05), ylim=(0, 3.2), title=f"{pname}  ({n} cells)")
    ax.legend(frameon=False, fontsize=9, loc="upper center")
fig.suptitle("Four equilibria across data (synthetic / real), model class (OLS / MLP / sampler), "
             "and training dynamics (gamma x regime)", fontsize=12)
fig.savefig(f"{FIGS}/four_equilibria_complete.png", dpi=130)
print(f"saved {FIGS}/four_equilibria_complete.png")
