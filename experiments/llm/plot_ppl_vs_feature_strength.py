"""Log perplexity vs feature strength. x = realized probe R2 (weak -> strong),
y = final median log10 ppl, one line per beta. Continual (main line) | fresh.
Shows: at beta=0 ppl rises as features weaken (rot rate); the anchor (beta>=0.5)
flattens it, overriding feature strength -- except the fresh instabilities."""
import os, numpy as np, pandas as pd, torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"; OUT = "experiments/llm/figs/qwen"
KNOBS = ["p100", "p065", "p040", "p033", "p015", "p005", "nat"]
SHUFP = {"p100": 1.0, "p065": 0.65, "p040": 0.40, "p033": 0.33, "p015": 0.15, "p005": 0.05, "nat": 0.0}
BETAS = [("b0", "0", "#d1495b"), ("b0p5", "0.5", "#edae49"), ("b1", "1", "#2f6f9f")]
LEG = {("e040_a040", "b0"): "mla2dv2_e040_a040_b0_s0",
       ("e040_a040", "b0p5"): "mla2bv2_e040_a040_b0p5_s0",
       ("e040_a040", "b1"): "mla2bv2_e040_a040_b1_s0"}

def ex(t): return t and os.path.exists(f"{RUNS}/{t}/trajectory.pt")
def cont_tag(k, b):
    if k == "nat":
        for t in (f"mlat_e040_a040_rep_{b}_s0", LEG.get(("e040_a040", b), "")):
            if ex(t): return t
        return None
    t = f"mlatF_{k}_e040_a040_rep_{b}_s0"; return t if ex(t) else None
def fresh_tag(k, b):
    t = f"mlatF_{k}_e040_a040_frep_{b}_s0"; return t if ex(t) else None
def load(tag): return torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
def finlp(d):
    lp = np.log10(np.clip(np.asarray(d["ppl_raw"], np.float32), 1.0, None)); return float(np.median(lp, axis=1)[-1])
def r2(d):
    prof = pd.DataFrame(d["profiles"]); y = np.asarray(d["innate"], float); cols = []
    for c in prof.columns:
        v = prof[c]
        cols.append(v.values.astype(float)[:, None] if pd.api.types.is_numeric_dtype(v)
                    else pd.get_dummies(v).values.astype(float))
    X = np.hstack(cols); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); yh = np.zeros_like(y)
    for f in range(5):
        te = idx[f::5]; tr = np.setdiff1d(idx, te)
        w = np.linalg.solve(X[tr].T @ X[tr] + np.eye(X.shape[1]), X[tr].T @ (y[tr] - y[tr].mean()))
        yh[te] = y[tr].mean() + X[te] @ w
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())

r2cache = {}
def series(tf, bc):
    xs, ys = [], []
    for k in KNOBS:
        t = tf(k, bc)
        if not t: continue
        d = load(t); rr = r2cache.setdefault(k, r2(d))
        xs.append(rr); ys.append(finlp(d))
    o = np.argsort(xs)
    return np.array(xs)[o], np.array(ys)[o]

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True, constrained_layout=True)
for ax, (wname, tf) in zip(axes, (("continual", cont_tag), ("fresh", fresh_tag))):
    for bc, blab, col in BETAS:
        xs, ys = series(tf, bc)
        ax.plot(xs, ys, "-o", color=col, lw=2.2, ms=7, label=f"$\\beta$={blab}")
    ax.axhline(np.log10(1000), color="#999", lw=1, ls="--")
    ax.text(0.98, np.log10(1000) + 0.06, "ppl 1000", color="#777", fontsize=8,
            ha="right", transform=ax.get_yaxis_transform())
    ax.set_title(wname, fontsize=13)
    ax.set_xlabel("feature strength  (probe $R^2$:  shuffled $\\to$ natural)", fontsize=12)
    ax.invert_xaxis()  # weak features on the left visually? keep strong=right
    ax.invert_xaxis()  # (no-op guard) -- strong R2 stays on the right
axes[0].set_ylabel("final median $\\log_{10}$ ppl", fontsize=12)
axes[0].legend(frameon=False, fontsize=12, loc="upper right")
fig.suptitle("Log perplexity vs feature strength: weaker features rot faster at $\\beta$=0; "
             "the anchor ($\\beta\\geq$0.5) flattens the curve", fontsize=13)
fig.savefig(f"{OUT}/ppl_vs_feature_strength.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/ppl_vs_feature_strength.png")

# text table: log ppl by shuffle p
print(f"\n{'knob':5}{'shufP':>7}{'R2':>7} | " + " ".join(f"{'c-'+b:>7}{'f-'+b:>7}" for b, _, _ in BETAS))
for k in KNOBS:
    row = f"{k:5}{SHUFP[k]:>7.2f}"
    r2v = None
    cells = ""
    for bc, _, _ in BETAS:
        ct, ft = cont_tag(k, bc), fresh_tag(k, bc)
        cv = finlp(load(ct)) if ct else float("nan")
        fv = finlp(load(ft)) if ft else float("nan")
        if r2v is None and ct: r2v = r2cache.get(k)
        cells += f"{cv:>7.2f}{fv:>7.2f}"
    print(f"{row}{(r2v if r2v is not None else float('nan')):>7.2f} | {cells}")
