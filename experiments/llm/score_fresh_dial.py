"""Score fresh (frep) vs continual (rep) feature dial against the registered
prediction. Mirrors the loader + realized_r2 from plot_knob_ppl_lines.py."""
import os, numpy as np, pandas as pd, torch
RUNS = "runs/pokec_gated_lm"
KNOBS = ["p100", "p065", "p040", "p033", "p015", "p005", "nat"]
BETAS = [("b0", "0.0"), ("b0p5", "0.5"), ("b1", "1.0")]
LEG = {("e040_a040", "b0"): "mla2dv2_e040_a040_b0_s0",
       ("e040_a040", "b0p5"): "mla2bv2_e040_a040_b0p5_s0",
       ("e040_a040", "b1"): "mla2bv2_e040_a040_b1_s0"}

def ex(t): return t and os.path.exists(f"{RUNS}/{t}/trajectory.pt")

def cont_tag(k, b):
    if k == "nat":
        for t in (f"mlat_e040_a040_rep_{b}_s0", LEG.get(("e040_a040", b), "")):
            if ex(t): return t
        return None
    t = f"mlatF_{k}_e040_a040_rep_{b}_s0"
    return t if ex(t) else None

def fresh_tag(k, b):
    t = f"mlatF_{k}_e040_a040_frep_{b}_s0"
    return t if ex(t) else None

def load(tag):
    return torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)

def medlp(d):  # median log10 ppl per round
    lp = np.log10(np.clip(np.asarray(d["ppl_raw"], np.float32), 1.0, None))
    return np.median(lp, axis=1)

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

def stats(d):
    op = np.asarray(d["op_raw"], float); inn = np.asarray(d["innate"], float)
    lp = medlp(d)
    return dict(r2=r2(d), lp1=float(lp[0]), lpN=float(lp[-1]), climb=float(lp[-1]-lp[0]),
                pplN=float(10**lp[-1]), dr=float(op[-1].std()/(inn.std()+1e-12)),
                bias=float(op[-1].mean()-inn.mean()))

for bc, blab in BETAS:
    print(f"\n===== beta={blab} =====")
    print(f"{'knob':6} {'R2':>5} | {'cont ppl1->N':>14} {'clmb':>6} {'dr':>5} {'bias':>6} "
          f"|| {'fresh ppl1->N':>14} {'clmb':>6} {'dr':>5} {'bias':>6}")
    for k in KNOBS:
        ct, ft = cont_tag(k, bc), fresh_tag(k, bc)
        c = stats(load(ct)) if ct else None
        f = stats(load(ft)) if ft else None
        rr = (c or f or {}).get("r2", float("nan"))
        def fmt(s):
            if s is None: return f"{'--':>14} {'--':>6} {'--':>5} {'--':>6}"
            return (f"{10**s['lp1']:6.1f}->{s['pplN']:6.1f} {s['climb']:6.2f} "
                    f"{s['dr']:5.2f} {s['bias']:6.3f}")
        print(f"{k:6} {rr:5.2f} | {fmt(c)} || {fmt(f)}")
