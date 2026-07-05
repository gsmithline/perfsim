"""Does adding the platform ever WIDEN the population (AI as a diversity source)?

For every cell we have the no-platform baseline (eps_AI=0) and the with-platform
result. Plot dr_no (x) vs dr_with (y); points above the diagonal mean the platform
made the population MORE diverse than it would be alone. Colored by training regime.
Pools synthetic OLS (toy_ols_eps_ai) and real OLS (real_eps_ai), all gammas/gates.
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


reg_col = {"replace": "#d62728", "accumulate": "#1f77b4", "pristine": "#2ca02c"}
pts = {r: ([], [], []) for r in reg_col}   # regime -> (dr_no, dr_with, is_synth)

# synthetic OLS: keys "{regime}|{gamma}|{eps}"
s = json.load(open("experiments/toy/results/toy_ols_eps_ai.json"))
init = s["meta"]["init_std"]
base = {}
for key, trs in s["runs"].items():
    regime, gamma, ea = key.split("|")
    if float(ea) == 0.0:
        base[("syn", gamma)] = np.mean([fin(t, "op_std") for t in trs]) / init
for key, trs in s["runs"].items():
    regime, gamma, ea = key.split("|")
    if float(ea) == 0.0:
        continue
    dr = np.mean([fin(t, "op_std") for t in trs]) / init
    pts[regime][0].append(base[("syn", gamma)]); pts[regime][1].append(dr); pts[regime][2].append(1)

# real OLS: keys "{name}|{gamma}|{regime}|{eps}" -> {dr,vr,gate}
r = json.load(open("experiments/toy/results/real_eps_ai.json"))
rbase = {}
for key, v in r.items():
    name, gamma, regime, ea = key.split("|")
    if float(ea) == 0.0:
        rbase[(name, gamma)] = v["dr"]
for key, v in r.items():
    name, gamma, regime, ea = key.split("|")
    if float(ea) == 0.0:
        continue
    pts[regime][0].append(rbase[(name, gamma)]); pts[regime][1].append(v["dr"]); pts[regime][2].append(0)

fig, ax = plt.subplots(figsize=(8.5, 8), constrained_layout=True)
ax.plot([0, 1.1], [0, 1.1], "k--", lw=1, label="no change (with = without)")
counts = {}
for regime, (xn, yw, syn) in pts.items():
    xn, yw, syn = np.array(xn), np.array(yw), np.array(syn)
    n_up = int(np.sum(yw > xn + 0.02))
    counts[regime] = (n_up, len(xn))
    m_syn = syn == 1
    ax.scatter(xn[m_syn], yw[m_syn], c=reg_col[regime], marker="o", s=34, alpha=0.7,
               edgecolors="none", label=f"{regime} (synthetic)")
    ax.scatter(xn[~m_syn], yw[~m_syn], c=reg_col[regime], marker="^", s=40, alpha=0.7,
               edgecolors="k", linewidths=0.3, label=f"{regime} (real)")
ax.fill_between([0, 1.1], [0, 1.1], 1.15, color="#2ca02c", alpha=0.06)
ax.text(0.05, 1.0, "ABOVE diagonal:\nplatform WIDENS the population\n(AI as a diversity source)",
        fontsize=10, color="#1a6b1a", va="top", fontweight="bold")
ax.text(0.55, 0.10, "BELOW diagonal:\nplatform narrows the population",
        fontsize=10, color="#7a1a1a", va="bottom")
ax.set(xlabel="population spread WITHOUT platform (dr_no)",
       ylabel="population spread WITH platform (dr_with)",
       xlim=(0, 1.1), ylim=(0, 1.15),
       title="Does the platform widen or narrow the population? (OLS, all regimes/gammas/gates)")
ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)
fig.savefig(f"{FIGS}/diversity_source.png", dpi=130)
print(f"saved {FIGS}/diversity_source.png")
print("cells where platform WIDENS the population (dr_with > dr_no + 0.02), by regime:")
for regime, (nup, tot) in counts.items():
    print(f"  {regime:10s}: {nup}/{tot}")
