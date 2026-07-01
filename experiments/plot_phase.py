"""Phase plane of the four equilibria across parameterizations (OLS).

Each point is one parameterization (gamma x regime x eps_AI x dataset), placed at
(dr = population spread, vr = platform vs population). The four corners are the
four joint states. Pools the synthetic OLS sweep and the three real datasets.
"""
import json, os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = "experiments/toy/figs"


def fin(t, k):
    v = [r[k] for r in t[-5:] if not (isinstance(r[k], float) and np.isnan(r[k]))]
    return sum(v) / len(v) if v else float("nan")


pts = {}  # source -> (dr list, vr list)

# synthetic OLS sweep (toy_ols_eps_ai.json): keys "{regime}|{gamma}|{eps_ai}"
s = json.load(open("experiments/toy/results/toy_ols_eps_ai.json"))
init = s["meta"]["init_std"]
dr_s, vr_s = [], []
for key, seedtrajs in s["runs"].items():
    regime, gamma, ea = key.split("|")
    if float(ea) == 0.0:
        continue
    dr = np.mean([fin(t, "op_std") for t in seedtrajs]) / init
    vr = np.mean([fin(t, "vr") for t in seedtrajs])
    if np.isfinite(vr):
        dr_s.append(dr); vr_s.append(vr)
pts["synthetic OLS"] = (dr_s, vr_s)

# real OLS sweep (real_eps_ai.json): keys "{name}|{gamma}|{regime}|{eps_ai}" -> {dr,vr,gate}
r = json.load(open("experiments/toy/results/real_eps_ai.json"))
dr_r, vr_r = [], []
for key, v in r.items():
    name, gamma, regime, ea = key.split("|")
    if float(ea) == 0.0:
        continue
    if np.isfinite(v["vr"]):
        dr_r.append(v["dr"]); vr_r.append(v["vr"])
pts["real OLS (MovieLens, Yelp)"] = (dr_r, vr_r)

col = {"synthetic OLS": "#1f77b4", "real OLS (MovieLens, Yelp)": "#ff7f0e"}
fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
ax.axhline(1.0, ls="--", c="black", lw=1)
ax.axvline(0.5, ls=":", c="gray", lw=1)
for src, (dr, vr) in pts.items():
    ax.scatter(dr, np.clip(vr, 0, 3.2), c=col[src], s=34, alpha=0.65, edgecolors="none", label=src)

for x, y, t in [(0.03, 0.10, "1  both collapsed"),
                (0.70, 0.10, "3  population diverse,\n     platform below"),
                (0.70, 1.10, "2  both diverse\n     (platform tracks)"),
                (0.03, 2.6, "4  population collapsed,\n     platform stays wide")]:
    ax.text(x, y, t, fontsize=10, color="#222", va="bottom", fontweight="bold")

ax.set(xlabel="population spread retained  (dr = op_std / init)",
       ylabel="platform vs population  (vr = pred_std / op_std)",
       xlim=(-0.02, 1.05), ylim=(0, 3.2),
       title="Four equilibria across parameterizations (OLS)\neach point = one gamma x regime x eps_AI x dataset")
ax.legend(frameon=False, fontsize=10, loc="upper right")
fig.savefig(f"{FIGS}/four_equilibria_phase_ols.png", dpi=130)
print(f"saved {FIGS}/four_equilibria_phase_ols.png  ({len(dr_s)} synthetic + {len(dr_r)} real points)")
