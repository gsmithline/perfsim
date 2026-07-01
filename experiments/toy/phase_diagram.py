"""Phase diagram for the toy OLS-in-AB loop.

Left panel = OUTCOME PLANE: every run is a point at (op_std, pred_std). The
diagonal pred_std=op_std is the variance cap (vr=1). The four equilibria are
four regions: near the origin both collapse; up the diagonal both are wide with
the model under the population; above the diagonal the model is wider than the
(collapsed) population; bottom-right the population is wide but the model is
collapsed. Color = regime, marker = config.

Right panel = PARAMETER PLANE: eps x gamma for the replace regime, strong
feature, complete graph, cell color = op_std (where the population collapses).
"""
import json, os
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "experiments/toy"
CONFIGS = {"strong_complete": "o", "weak_complete": "s", "strong_sparse": "^"}
RCOL = {"replace": "#d62728", "accumulate": "#1f77b4", "pristine": "#2ca02c"}


def load(tag):
    return json.load(open(f"{HERE}/results/toy_{tag}.json"))


def fin(runs, k, key):
    v = [r[key] for r in runs[k][-5:] if not (isinstance(r[key], float) and np.isnan(r[key]))]
    return sum(v) / len(v) if v else float("nan")


fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)

# ---------- left: outcome plane ----------
for tag, mk in CONFIGS.items():
    d = load(tag); runs = d["runs"]; m = d["meta"]
    for regime in ["replace", "accumulate", "pristine"]:
        for eps in m["eps_grid"]:
            for g in m["gamma_grid"]:
                k = f"{regime}_e{eps}_g{g}"
                ax[0].scatter(fin(runs, k, "op_std"), fin(runs, k, "pred_std"),
                              c=RCOL[regime], marker=mk, s=34, alpha=0.55,
                              edgecolors="none")
hi = 0.29
ax[0].plot([0, hi], [0, hi], "k--", lw=1.2)
ax[0].text(0.20, 0.225, "variance cap\npred_std = op_std", fontsize=8, rotation=34)
ax[0].axvline(0.10, color="gray", ls=":", lw=1)
ax[0].set(xlim=(-0.01, hi), ylim=(-0.01, hi),
          xlabel="op_std  (population spread)  ->  wide",
          ylabel="pred_std  (model output spread)  ->  wide",
          title="OUTCOME PLANE  (each point = one run)")
ann = [(0.012, 0.010, "1  both\ncollapsed"),
       (0.225, 0.20, "2  both wide\n(model <= pop)"),
       (0.045, 0.13, "3  pop collapsed,\nmodel WIDE  (vr>1)"),
       (0.20, 0.045, "4  pop wide,\nmodel collapsed")]
for x, y, t in ann:
    ax[0].annotate(t, (x, y), fontsize=9, fontweight="bold",
                   bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
# legends
h_reg = [plt.Line2D([], [], marker="o", ls="", c=c, label=r) for r, c in RCOL.items()]
h_cfg = [plt.Line2D([], [], marker=mk, ls="", c="0.3", label=t) for t, mk in CONFIGS.items()]
leg1 = ax[0].legend(handles=h_reg, title="regime", loc="upper left", frameon=False, fontsize=8)
ax[0].add_artist(leg1)
ax[0].legend(handles=h_cfg, title="config", loc="lower right", frameon=False, fontsize=8)

# ---------- right: parameter plane (strong/complete, replace) ----------
d = load("strong_complete"); runs = d["runs"]; m = d["meta"]
EPS, GAM = m["eps_grid"], m["gamma_grid"]
grid = np.array([[fin(runs, f"replace_e{e}_g{g}", "op_std") for e in EPS] for g in GAM])
im = ax[1].imshow(grid, origin="lower", aspect="auto", cmap="viridis")
ax[1].set(xticks=range(len(EPS)), xticklabels=EPS, yticks=range(len(GAM)), yticklabels=GAM,
          xlabel="eps  (population-spread knob)", ylabel="gamma  (homophily)",
          title="PARAMETER PLANE  op_std  (replace, strong, complete)")
for gi in range(len(GAM)):
    for ei in range(len(EPS)):
        ax[1].text(ei, gi, f"{grid[gi, ei]:.2f}", ha="center", va="center",
                   color="white", fontsize=9)
fig.colorbar(im, ax=ax[1], shrink=0.85, label="op_std")

fig.suptitle("Toy OLS-in-AB loop: phase diagram", fontsize=13)
out = f"{HERE}/figs/toy_phase_diagram.png"
fig.savefig(out, dpi=130)
print("saved", out)
