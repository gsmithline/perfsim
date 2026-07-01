"""Controlled toy: OLS in the bounded-confidence (AB) performative loop, on a
synthetic distribution where the feature genuinely predicts the label.

Goal: show the loop reaches four qualitatively different equilibria, set by two
knobs we already isolated:
  eps   -> POPULATION variance. high eps = global consensus (collapse),
           low eps = confidence-gated fragmentation (spread).
  gamma -> how feature-SORTED the band is at fixed population variance.
           homophily (gamma>0) sorts boundary agents into feature-aligned camps,
           raising Cov(z,x); heterophily (gamma<0) scrambles it, lowering Cov.

Population: N synthetic agents on a complete graph. feature z ~ U(0,1);
initial opinion x0 = clip(0.5 + a*(z-0.5) + noise), so z predicts the label with
tunable strength. Dynamics = NDlib AlgorithmicBiasModel. Predictor = OLS of the
current opinion (the label / supervision) on the standardized feature z; the
data regime {replace, accumulate, pristine} sets which (z, opinion) pairs are
pooled for the fit. Feedback respects bounded confidence: agent i blends toward
m_i only if |m_i - x_i| < eps, weight W.

With z standardized and a single feature, pred_std = |slope| = |Cov(z_std, x)|,
and the variance cap pred_std <= op_std is exactly vr = pred_std/op_std <= 1.

  corner 1  collapse / collapse   low op_std, low pred_std   (high eps)
  corner 2  spread / spread       high op_std, high vr       (low eps, homophily)
  corner 3  pop fixed, model up   op_std flat, vr up         (fixed eps, gamma up)
  corner 4  pop fixed, model down op_std flat, vr down        (fixed eps, gamma down)
"""
import json, os, random
import numpy as np
import networkx as nx
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "experiments/toy"
OUT = f"{HERE}/results"
FIGS = f"{HERE}/figs"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

SMOKE = os.environ.get("SMOKE", "0") == "1"
FEATURE = os.environ.get("FEATURE", "strong")   # strong (R2~0.72) | weak (R2~0.1)
GRAPH = os.environ.get("GRAPH", "complete")     # complete (mean-field) | sparse (degree ~6)
TAG = os.environ.get("TAG", f"{FEATURE}_{GRAPH}")
N, W, SEED = 300, 0.3, 0
ROUNDS = 50
A_SLOPE, NOISE_SD = (0.8, 0.15) if FEATURE == "strong" else (0.25, 0.25)
EPS_GRID = [0.08, 0.15, 0.25, 0.40, 0.60]
GAMMA_GRID = [-1.5, 0.0, 1.5, 3.0]
REGIMES = ["replace", "accumulate", "pristine"]
if SMOKE:
    EPS_GRID, GAMMA_GRID, ROUNDS = [0.15, 0.60], [0.0, 1.5], 20

rng = np.random.default_rng(SEED)
z = rng.uniform(0.0, 1.0, N)
z_std = (z - z.mean()) / (z.std() + 1e-9)
x0 = np.clip(0.5 + A_SLOPE * (z - 0.5) + rng.normal(0.0, NOISE_SD, N), 0.0, 1.0)
if GRAPH == "sparse":
    G = nx.connected_watts_strogatz_graph(N, 6, 0.1, tries=200, seed=SEED)
else:
    G = nx.complete_graph(N)
MEAN_DEG = 2 * G.number_of_edges() / N
F = np.column_stack([np.ones(N), z_std])

_w0 = np.linalg.lstsq(F, x0, rcond=None)[0]
_fit0 = F @ _w0
INIT_R2 = float(1.0 - ((x0 - _fit0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum())
INIT_STD = float(x0.std())


def build_pop(gamma, eps, seed):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G)
    c = mc.Configuration()
    c.add_model_parameter("epsilon", eps)
    c.add_model_parameter("gamma", gamma)
    m.set_initial_status(c)
    for i in range(N):
        m.status[i] = float(x0[i])
    m.sts = x0.copy()          # complete-graph cache used for biased partner pick
    m.iteration(False)         # consume the iteration-0 no-op
    return m


def ols(Xtr, ytr, Xpr):
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return np.clip(Xpr @ w, 0.0, 1.0), w


def run(regime, gamma, eps, use_platform=True):
    pop = build_pop(gamma, eps, SEED)
    x = x0.copy()
    keep = np.random.default_rng(SEED).permutation(N)[: N // 2]
    F0keep = F[keep]
    HX, Hy = [], []
    traj = []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        slope = pstd = pmean = float("nan")
        if use_platform:
            if regime == "replace":
                Xtr, ytr = F, x
            elif regime == "accumulate":
                HX.append(F); Hy.append(x.copy())
                Xtr, ytr = np.vstack(HX), np.concatenate(Hy)
            else:
                Xtr = np.vstack([F, F0keep]); ytr = np.concatenate([x, x0[keep]])
            m, w = ols(Xtr, ytr, F)
            slope = float(w[1])
            g = np.abs(m - x) < eps
            x = np.where(g, (1 - W) * x + W * m, x)
            for i in range(N):
                pop.status[i] = float(x[i])
            pop.sts = x.copy()
            pstd, pmean = float(np.std(m)), float(np.mean(m))
        ostd = float(x.std())
        traj.append(dict(round=t, op_std=ostd, op_mean=float(x.mean()),
                         pred_std=pstd, pred_mean=pmean, slope=slope,
                         vr=(pstd / ostd) if (use_platform and ostd > 1e-9) else float("nan")))
    return traj


def fin(traj, k):
    v = [r[k] for r in traj[-5:] if not (isinstance(r[k], float) and np.isnan(r[k]))]
    return sum(v) / len(v) if v else float("nan")


def main():
    runs = {}
    for eps in EPS_GRID:
        runs[f"noplatform_e{eps}_g0.0"] = run("replace", 0.0, eps, use_platform=False)
        runs[f"noplatform_e{eps}_g{GAMMA_GRID[-1]}"] = run("replace", GAMMA_GRID[-1], eps, use_platform=False)
        for gamma in GAMMA_GRID:
            for regime in REGIMES:
                runs[f"{regime}_e{eps}_g{gamma}"] = run(regime, gamma, eps)

    meta = dict(N=N, W=W, rounds=ROUNDS, eps_grid=EPS_GRID, gamma_grid=GAMMA_GRID,
                a_slope=A_SLOPE, noise_sd=NOISE_SD, init_op_std=INIT_STD, init_R2=INIT_R2,
                feature=FEATURE, graph=GRAPH, mean_deg=MEAN_DEG, tag=TAG)
    json.dump({"meta": meta, "runs": runs}, open(f"{OUT}/toy_{TAG}.json", "w"))

    print(f"TOY [{TAG}]  N={N}  graph={GRAPH} (mean_deg={MEAN_DEG:.1f})  W={W}  rounds={ROUNDS}")
    print(f"feature={FEATURE}: z ~ U(0,1);  x0 = clip(0.5 + {A_SLOPE}*(z-0.5) + N(0,{NOISE_SD}))")
    print(f"init op_std = {INIT_STD:.3f}   init feature R2 = {INIT_R2:.3f}")
    print(f"columns:  op_std = population std   pred_std = model output std")
    print(f"          vr = pred_std/op_std (variance cap: vr<=1)   slope = OLS coef on z_std")
    print(f"          dr = op_std/init_op_std (population divergence ratio; <1 = collapsed)")

    # ---- Table 1: phase map (replace regime) ----
    print("\n================ TABLE 1: PHASE MAP (regime = replace) ================")
    print("rows = eps (population-variance knob), cols = gamma (feature-sorting knob)")
    head = "  eps  | nplt_dr | " + " | ".join(f"   gamma={g:>4}    " for g in GAMMA_GRID)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for eps in EPS_GRID:
        npl = fin(runs[f"noplatform_e{eps}_g0.0"], "op_std") / INIT_STD
        cells = []
        for g in GAMMA_GRID:
            k = f"replace_e{eps}_g{g}"
            cells.append(f"os{fin(k_traj(runs,k),'op_std'):.3f} vr{fin(k_traj(runs,k),'vr'):.2f}")
        print(f"  {eps:>4.2f} |  {npl:>5.2f}  | " + " | ".join(f"{c:>15s}" for c in cells))

    # ---- Table 2: eps x regime at gamma=0 (the model-population coupling knob) ----
    print("\n=========== TABLE 2: eps x REGIME at gamma=0 (textbook Deffuant) ===========")
    print("eps moves the POPULATION; the regime moves whether the MODEL stays tied to it.")
    print("replace fits the current pop (vr<=1). pristine/accumulate anchor to other data,")
    print("so the model can stay wide while the pop collapses (vr can exceed 1).")
    print(f"  {'eps':>5} {'regime':10s} | {'op_std':>7s} {'dr':>5s} {'pred_std':>8s} {'vr':>5s} {'slope':>7s}")
    for eps in EPS_GRID:
        for regime in REGIMES:
            t = runs[f"{regime}_e{eps}_g0.0"]
            print(f"  {eps:>5} {regime:10s} | {fin(t,'op_std'):7.3f} {fin(t,'op_std')/INIT_STD:5.2f} "
                  f"{fin(t,'pred_std'):8.3f} {fin(t,'vr'):5.2f} {fin(t,'slope'):7.3f}")

    # ---- Corner summary ----
    print("\n================ FOUR CORNERS (gamma=0; eps x regime) ================")
    eps_lo, eps_hi = EPS_GRID[0], EPS_GRID[-1]
    corners = [
        ("1 both collapsed", eps_hi, "replace", runs[f"replace_e{eps_hi}_g0.0"],
         "high eps collapses the pop; replace fits the collapsed pop -> model collapses too."),
        ("2 both wide (model<=pop)", eps_lo, "replace", runs[f"replace_e{eps_lo}_g0.0"],
         "low eps keeps the pop spread+feature-sorted; model fits it, vr<=1."),
        ("3 pop fixed(collapsed), model UP", eps_hi, "pristine", runs[f"pristine_e{eps_hi}_g0.0"],
         "pristine anchors the slope to the original data; model stays wide though the pop collapsed (vr>1)."),
        ("4 pop fixed(collapsed), model DOWN", eps_hi, "replace", runs[f"replace_e{eps_hi}_g0.0"],
         "same collapsed pop as corner 3, but replace lets the model collapse with it. 3 vs 4 = pristine vs replace."),
    ]
    for name, eps, regime, t, desc in corners:
        print(f"  corner {name:36s} eps={eps:<5} {regime:10s} | op_std {fin(t,'op_std'):.3f} "
              f"(dr {fin(t,'op_std')/INIT_STD:.2f})  pred_std {fin(t,'pred_std'):.3f}  vr {fin(t,'vr'):.2f}")
        print(f"  {'':45s} {desc}")
    print("\n  NOTE: corner 4 (pop WIDE + model collapsed) is NOT reached here: a wide pop in")
    print("  this strong-feature toy (R2=0.72) is always feature-sorted, so a model fit on it")
    print("  is never collapsed. Reaching it needs a weak/non-predictive feature (low Cov).")
    print("  NOTE: gamma COUPLES pop+model on this complete graph (both rise with gamma); the")
    print("  gamma-decoupling seen on the sparse Yelp graph does not reproduce mean-field.")

    make_fig(runs)
    print(f"\nsaved {OUT}/toy_{TAG}.json  and  {FIGS}/toy_{TAG}.png")


def k_traj(runs, k):
    return runs[k]


def make_fig(runs):
    colors = {"replace": "#d62728", "accumulate": "#1f77b4", "pristine": "#2ca02c"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    npl = [fin(runs[f"noplatform_e{eps}_g0.0"], "op_std") for eps in EPS_GRID]
    for metric, a, title in [("op_std", ax[0], "population std vs eps"),
                             ("pred_std", ax[1], "model output std vs eps"),
                             ("vr", ax[2], "variance ratio pred_std/op_std vs eps")]:
        for regime in REGIMES:
            y = [fin(runs[f"{regime}_e{eps}_g0.0"], metric) for eps in EPS_GRID]
            a.plot(EPS_GRID, y, marker="o", c=colors[regime], label=regime)
        a.set(xlabel="eps  (population-variance knob)", title=title)
    ax[0].plot(EPS_GRID, npl, ls="--", c="gray", marker="s", ms=4, label="no platform")
    ax[0].axhline(INIT_STD, ls=":", c="black", lw=1)
    ax[0].set_ylabel("std")
    ax[2].axhline(1.0, ls="--", c="black", lw=1)
    ax[2].annotate("variance cap vr=1", (EPS_GRID[0], 1.0), fontsize=8, va="bottom")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Toy OLS in the AB loop [{TAG}], gamma=0.  graph={GRAPH}, "
                 f"init op_std={INIT_STD:.2f}, R2={INIT_R2:.2f}")
    fig.savefig(f"{FIGS}/toy_{TAG}.png", dpi=130)


if __name__ == "__main__":
    main()
