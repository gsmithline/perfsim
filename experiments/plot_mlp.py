"""Two MLP figures from real_mlp_results.json:
 (A) model class barely matters: population spread under MLP vs under OLS.
 (B) four corners: dr vs vr, colored by predictor. OLS and MLP span all four;
     the sampler is trapped in the top half (platform always wide).
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

cells = {}
for k, v in runs.items():
    name, g, kind, regime = k.split("|")
    cells.setdefault((name, g, regime), {})[kind] = (v["dr"], v["vr"])

ols_dr, mm_dr, ms_dr = [], [], []
pts = {"ols": ([], []), "mlp_mean": ([], []), "mlp_sample": ([], [])}
for kd in cells.values():
    if all(x in kd for x in ["ols", "mlp_mean", "mlp_sample"]):
        ols_dr.append(kd["ols"][0]); mm_dr.append(kd["mlp_mean"][0]); ms_dr.append(kd["mlp_sample"][0])
    for kind, (dr, vr) in kd.items():
        pts[kind][0].append(dr); pts[kind][1].append(vr)

col = {"ols": "#888888", "mlp_mean": "#1f77b4", "mlp_sample": "#d62728"}
lab = {"ols": "OLS (linear mean)", "mlp_mean": "MLP (nonlinear mean)", "mlp_sample": "MLP sampler P(y|x)"}

fig, ax = plt.subplots(1, 2, figsize=(13.5, 6), constrained_layout=True)

# (A) MLP dr vs OLS dr
ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="same as OLS")
ax[0].scatter(ols_dr, mm_dr, c=col["mlp_mean"], s=36, alpha=0.75, label="MLP-mean")
ax[0].scatter(ols_dr, ms_dr, c=col["mlp_sample"], s=36, alpha=0.75, label="MLP sampler")
ax[0].set(xlabel="population spread (dr) under OLS", ylabel="population spread (dr) under the MLP",
          title="(A) Model class barely changes the population", xlim=(0, 1.05), ylim=(0, 1.05))
ax[0].legend(frameon=False, fontsize=9, loc="upper left")
ax[0].annotate("on the line = same collapse as OLS\nbelow = sampler collapses more",
               (0.5, 0.06), fontsize=8, color="#555")

# (B) four corners: dr vs vr
ax[1].axhline(1.0, ls="--", c="black", lw=1)
ax[1].axvline(0.5, ls=":", c="gray", lw=1)
for kind in ["ols", "mlp_mean", "mlp_sample"]:
    ax[1].scatter(pts[kind][0], pts[kind][1], c=col[kind], s=36, alpha=0.7, label=lab[kind])
ax[1].set(xlabel="population spread (dr)", ylabel="platform vs population (vr)",
          title="(B) Four corners: the sampler is trapped in the top half",
          xlim=(0, 1.05), ylim=(0, 3.2))
for (x, y, t) in [(0.04, 0.10, "1 both\ncollapsed"), (0.78, 0.18, "3 platform\nbelow pop"),
                  (0.78, 1.12, "2 both\ndiverse"), (0.04, 2.7, "4 platform\nwide, pop dead")]:
    ax[1].text(x, y, t, fontsize=8, color="#333", va="bottom")
ax[1].legend(frameon=False, fontsize=9, loc="upper center")
ax[1].annotate("sampler never goes below vr=1\n(its noise floor forbids a narrow platform)",
               (0.30, 0.10), fontsize=8, color="#a00")

fig.suptitle("MLP study on real data (OLS vs nonlinear MLP vs sampler), 3 datasets x 3 gammas x 3 regimes")
fig.savefig(f"{FIGS}/mlp_modelclass_corners.png", dpi=130)
print(f"saved {FIGS}/mlp_modelclass_corners.png  ({len(ols_dr)} matched triples, {sum(len(v[0]) for v in pts.values())} points)")
