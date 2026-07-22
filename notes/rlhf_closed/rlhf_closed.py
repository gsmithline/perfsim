#!/usr/bin/env python3
"""MINIMAL local closed-loop RLHF (DPO) test.

Question: does CLOSED-loop RLHF reinforce the proxy's INITIAL opinion bias, because
the proxy deploys into -> changes -> the very population that later grades its
preferences?  The closed - open contrast isolates the model-mediated feedback
channel (identical DPO, identical dynamics; only the source of preference labels
differs).

Surrogate (laptop, NO LLM). Reuses the real innate opinions + graph
(fv2.ml_action_setup) and the SAME gate/blend + calibrated peer dynamics as the
LM pipeline / rlvr_local.py.

--- policy (faithful LLM analog, no Gaussian) ----------------------------------
The proxy is a categorical distribution over a discretized opinion grid
y_k in {0, 1/(K-1), ..., 1}: p_theta(k | phi) = softmax(f_theta(phi)/T)_k, a
SHARED net f_theta over the per-node feature phi (= innate).  This mirrors an LLM
emitting a distribution over numeric tokens: sampling two candidates is multinomial,
and DPO log-probs are EXACT (log p_theta(k)).  No Gaussian policy, no injected bias.
  point prediction (what gets deployed):  yhat_i = sum_k p_theta(k|phi_i) * y_k
  initial bias:  pi_0 is FIT to the real frozen-Qwen prior -> it inherits the LLM's
                 own prediction bias (mean shift + regression-to-mean compression).
                 Measured + reported at round 0; NOT injected.

--- one round (learning arm) ---------------------------------------------------
  prefs from pop x_{t-1}:  for each agent i, sample kA,kB ~ pi_{t-1}(.|phi_i);
    P(A>B) = sigmoid( tau * [ U(yA;x) - U(yB;x) ] ),  U(y;x) = -|y - x|;
    sample winner ~ Bernoulli.   (tau -> inf  == deterministic closer-wins.)
  DPO update pi_{t-1} -> pi_t on those pairs; reference = MOVING anchor (prev
    round's checkpoint) by default (ANCHOR=fixed ablates to pi_0).
  deploy pi_t's point predictions -> gated blend -> peer step -> x_t.

--- three arms (differ ONLY in the preference-label source) --------------------
  frozen : no DPO; pi_0 deployed every round (pure deployment drift, static policy).
  open   : labels judged by the no-AI counterfactual twin  x_cf_{t-1}.
  closed : labels judged by the arm's OWN model-influenced pop x_own_{t-1}.
All three deploy their current policy into their OWN population under identical
dynamics.  One synchronized no-AI twin (same dynamics, no deployment) is the shared
W1 reference and the open arm's judge.

Config via env: EPS_SOCIAL EPS_AI W KAPPA H SEEDS K TAU DECODE_T DPO_BETA DPO_INNER
                LR ANCHOR(moving|fixed) CELLS(realistic,permissive) TAG.
Run from perfsim repo root.
"""
import importlib.util, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import wasserstein_distance
os.environ.setdefault("MPLCONFIGDIR",
    "/private/tmp/claude-501/-Users-gabesmithline-Desktop-ellis-work-nosync/4b0348b4-6440-472d-ac66-9ef073946cd4/scratchpad/.mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

torch.set_num_threads(6)
REPO = "/Users/gabesmithline/Desktop/ellis_work.nosync/perfsim"
FV2  = f"{REPO}/experiments/MMHD_restructured_project/scripts/build_feature_v2_data.py"
RUNS = f"{REPO}/runs/pokec_gated_lm"
OUT  = os.path.dirname(os.path.abspath(__file__))

def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
fv2 = _load("fv2", FV2)

def _f(k, d): return float(os.environ.get(k, d))
H         = int(_f("H", 30))
SEEDS     = int(_f("SEEDS", 6))
K         = int(_f("K", 41))            # opinion grid resolution (0.025)
SIGMA0    = _f("SIGMA0", 0.06)          # pi_0 predictive spread (LLM decode uncertainty)
TAU       = _f("TAU", 12.0)             # BT judge sharpness (log its value; ->inf = closer-wins)
DECODE_T  = _f("DECODE_T", 1.0)         # policy decode temperature
DPO_BETA  = _f("DPO_BETA", 0.10)
DPO_INNER = int(_f("DPO_INNER", 12))
LR        = _f("LR", 3e-3)
ANCHOR    = os.environ.get("ANCHOR", "moving")   # moving | fixed
TAG       = os.environ.get("TAG", "rlhf_closed")
RATE, TAU_P, TAU_G = 0.20, 0.02, 0.02   # peer step + soft gate (calibrated to real ab_sweep)

CELLS = {
    "realistic":  dict(eps_social=0.10, eps_ai=0.40, W=0.30, kappa=0.25),
    "permissive": dict(eps_social=0.10, eps_ai=2.00, W=0.50, kappa=0.00),
}
WANT = os.environ.get("CELLS", "realistic,permissive").split(",")

YK = torch.linspace(0, 1, K)           # opinion grid values

# --------------------------------------------------------------------------- policy
class CatPolicy(nn.Module):
    """Shared net phi([N,in_dim]) -> K logits.  p = softmax(logits/T)."""
    def __init__(s, in_dim=2, k=K, h=32):
        super().__init__()
        s.in_dim = in_dim
        s.n = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                            nn.Linear(h, k))
    def logits(s, phi): return s.n(phi)                         # phi [N,in_dim] -> [N,K]
    def logp(s, phi):   return F.log_softmax(s.logits(phi) / DECODE_T, dim=-1)
    def probs(s, phi):  return torch.softmax(s.logits(phi) / DECODE_T, dim=-1)
    def point(s, phi):                                          # E[y|phi]  (deployed pred)
        return (s.probs(phi) * YK[None, :]).sum(-1)

def qwen_prior():
    d = torch.load(f"{RUNS}/frz_qwen_e040_s0/trajectory.pt", map_location="cpu",
                   weights_only=False)
    return np.clip(np.asarray(d["pred_raw"], float)[0], 0, 1)

def fit_pi0(phi, prior, steps=1500, seed=0):
    """pi_0 fit to the real Qwen prior with a SOFT target (Gaussian bump of width SIGMA0
    over the grid, centered on each node's qwen value) -> pi_0 REPRODUCES the LLM's mean
    prediction (bias inherited) AND carries realistic predictive spread, so sampled
    candidate pairs are distinct and DPO gets a real preference signal."""
    torch.manual_seed(seed); net = CatPolicy(in_dim=phi.shape[1])
    yk = YK.numpy()[None, :]; q = prior[:, None]
    lt = -((yk - q) ** 2) / (2 * SIGMA0 ** 2)
    tgt = np.exp(lt - lt.max(1, keepdims=True)); tgt = tgt / tgt.sum(1, keepdims=True)
    tgt = torch.tensor(tgt, dtype=torch.float32)               # [N,K] soft target
    opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        loss = -(tgt * F.log_softmax(net.logits(phi), dim=-1)).sum(1).mean()
        loss.backward(); opt.step()
    net.eval(); return net

# --------------------------------------------------------------------------- dynamics
def peer_step(z, adjb, P):
    d = (z[:, None] - z[None, :]).abs()
    w = adjb * torch.sigmoid((P["eps_social"] - d) / TAU_P)
    nbar = (w * z[None, :]).sum(1) / w.sum(1).clamp_min(1e-6)
    return z + RATE * (nbar - z)

def deploy(x, yhat, innate, P, w_deploy):
    b = P["kappa"] * innate + (1.0 - P["kappa"]) * x
    gate = torch.sigmoid((P["eps_ai"] - (yhat - x).abs()) / TAU_G)
    return (1.0 - w_deploy * gate) * b + (w_deploy * gate) * yhat

def twin_traj(innate, adjb, P):
    x = innate.clone(); traj = [x.clone()]
    for _ in range(H):
        x = peer_step(deploy(x, x, innate, P, 0.0), adjb, P); traj.append(x.clone())
    return traj                                                # len H+1, traj[0]=innate

# --------------------------------------------------------------------------- prefs + DPO
def U(y, x): return -(y - x).abs()

def sample_prefs(net, phi, xjudge, gen):
    """For each node sample kA,kB ~ pi; BT winner under U(.;xjudge). Returns (kw,kl)."""
    with torch.no_grad():
        p = net.probs(phi)                                     # [N,K]
        kA = torch.multinomial(p, 1, generator=gen).squeeze(1)
        kB = torch.multinomial(p, 1, generator=gen).squeeze(1)
        yA, yB = YK[kA], YK[kB]
        pA = torch.sigmoid(TAU * (U(yA, xjudge) - U(yB, xjudge)))
        Awin = (torch.rand(len(phi), generator=gen) < pA)
        kw = torch.where(Awin, kA, kB); kl = torch.where(Awin, kB, kA)
    return kw, kl

def dpo_update(net, ref, phi, kw, kl, opt):
    idx = torch.arange(len(phi))
    for _ in range(DPO_INNER):
        opt.zero_grad()
        lp = net.logp(phi);
        with torch.no_grad(): lpr = ref.logp(phi)
        dw = lp[idx, kw] - lpr[idx, kw]
        dl = lp[idx, kl] - lpr[idx, kl]
        loss = -F.logsigmoid(DPO_BETA * (dw - dl)).mean()
        loss.backward(); opt.step()
    return float(loss.detach())

def snapshot(net):
    r = CatPolicy(in_dim=net.in_dim)
    r.load_state_dict({k: v.clone() for k, v in net.state_dict().items()})
    r.eval()
    return r

# --------------------------------------------------------------------------- one arm run
def run_arm(arm, phi, innate, adjb, P, twin, pi0_state, ref0_preds, held, seed):
    """arm in {frozen, open, closed}. Returns per-round metric lists (len H+1)."""
    net = CatPolicy(in_dim=phi.shape[1]); net.load_state_dict(pi0_state); net.eval()
    opt = torch.optim.Adam(net.parameters(), LR)
    ref_fixed = snapshot(net)                                  # pi_0 anchor (for fixed)
    gen = torch.Generator().manual_seed(1000 * seed + {"frozen":1,"open":2,"closed":3}[arm])

    pred0 = net.point(phi).detach()
    x = innate.clone()
    M = dict(pop_mean=[], pop_std=[], w1_twin=[], pred_w1_pi0=[], win_U0=[], win_Ut=[],
             flip=[], preds=[])

    def record(t, xcur, netcur):
        pr = netcur.point(phi).detach()
        M["pop_mean"].append(float(xcur.mean())); M["pop_std"].append(float(xcur.std()))
        M["w1_twin"].append(float(wasserstein_distance(xcur.numpy(), twin[t].numpy())))
        M["pred_w1_pi0"].append(float(wasserstein_distance(pr.numpy(), pred0.numpy())))
        # win rate vs fixed pi_0 reference candidates, judged under U_0 (innate) & U_t (xcur)
        with torch.no_grad():
            p = netcur.probs(phi)
            cand = YK[torch.multinomial(p, ref0_preds.shape[1], replacement=True, generator=gen)]
            refc = ref0_preds                                  # [N,R] pi_0 candidate values
            def winrate(x):                                    # tie-aware: gt + 0.5*tie
                du, dr = U(cand, x), U(refc, x)
                return float(((du > dr).float() + 0.5 * (du == dr).float()).mean())
            M["win_U0"].append(winrate(innate[:, None]))
            M["win_Ut"].append(winrate(xcur[:, None]))
        # held-out pref pairs: winner under U(.;xcur) vs round-0 winner (under innate)
        c1, c2, w0 = held
        wt = (U(c1, xcur) > U(c2, xcur))
        M["flip"].append(float((wt != w0).float().mean()))
        M["preds"].append(pr.numpy().astype(np.float32))

    record(0, x, net)                                          # t=0: pop=innate, pi_0
    for t in range(1, H + 1):
        if arm != "frozen":
            xjudge = x if arm == "closed" else twin[t - 1]
            kw, kl = sample_prefs(net, phi, xjudge, gen)
            ref = ref_fixed if ANCHOR == "fixed" else snapshot(net)   # moving = prev checkpoint
            dpo_update(net, ref, phi, kw, kl, opt); net.eval()
        yhat = net.point(phi).detach()
        x = peer_step(deploy(x, yhat, innate, P, P["W"]), adjb, P)
        record(t, x, net)
    M["preds"] = np.stack(M["preds"])                          # [H+1, N]
    return M

# --------------------------------------------------------------------------- main
def run_cell(name, P):
    innate, adj = fv2.ml_action_setup()
    adjb = (adj > 0).float()
    prior = qwen_prior()
    # feature = [innate, qwen_prior]: carries the LLM's own prediction signal so pi_0
    # REPRODUCES the real per-node Qwen predictions (bias inherited, not a fit artifact).
    phi = torch.stack([innate, torch.tensor(prior, dtype=torch.float32)], dim=1)
    net0 = fit_pi0(phi, prior)
    pi0_state = {k: v.clone() for k, v in net0.state_dict().items()}
    pred0 = net0.point(phi).detach()

    corr_fit = float(np.corrcoef(pred0.numpy(), prior)[0, 1])
    print(f"[pi_0 fit] corr(pi0_point, qwen_prior)={corr_fit:.3f}  "
          f"corr(qwen,innate)={float(np.corrcoef(prior, innate.numpy())[0,1]):.3f}", flush=True)
    # --- initial (inherited, NOT injected) bias report ---
    b_mean = float(pred0.mean() - innate.mean())
    b_std_ratio = float(pred0.std() / innate.std())
    b_w1 = float(wasserstein_distance(pred0.numpy(), innate.numpy()))
    print(f"\n===== CELL {name}  {P} =====", flush=True)
    print(f"[pi_0 inherited bias]  mean(pred)-mean(innate)={b_mean:+.4f}   "
          f"std_ratio(pred/innate)={b_std_ratio:.3f} (<1 = compression)   "
          f"W1(pred0,innate)={b_w1:.4f}", flush=True)
    print(f"[config] H={H} seeds={SEEDS} K={K} tau={TAU} decodeT={DECODE_T} "
          f"beta={DPO_BETA} inner={DPO_INNER} lr={LR} anchor={ANCHOR}", flush=True)

    twin = twin_traj(innate, adjb, {**P, "W": P["W"]})

    # fixed pi_0 reference candidate set (R per node) + held-out pref pairs (round 0)
    g0 = torch.Generator().manual_seed(777)
    p0 = net0.probs(phi)
    ref0_preds = YK[torch.multinomial(p0, 4, replacement=True, generator=g0)]   # [N,4]
    c1 = YK[torch.multinomial(p0, 1, generator=g0).squeeze(1)]
    c2 = YK[torch.multinomial(p0, 1, generator=g0).squeeze(1)]
    held = (c1, c2, (U(c1, innate) > U(c2, innate)))

    Pw = {**P, "W": P["W"]}
    arms = ["frozen", "open", "closed"]
    res = {a: [] for a in arms}
    all_preds = {}
    for seed in range(SEEDS):
        for a in arms:
            M = run_arm(a, phi, innate, adjb, Pw, twin, pi0_state, ref0_preds, held, seed)
            all_preds[f"{a}_s{seed}"] = M.pop("preds")
            res[a].append(M)
        print(f"  seed {seed}: "
              + " | ".join(f"{a} W1={res[a][seed]['w1_twin'][-1]:.4f} "
                           f"std={res[a][seed]['pop_std'][-1]:.4f} "
                           f"winUt={res[a][seed]['win_Ut'][-1]:.3f}" for a in arms), flush=True)

    twin_stats = dict(mean=[float(t.mean()) for t in twin], std=[float(t.std()) for t in twin])
    meta = dict(cell=name, params=P, H=H, seeds=SEEDS, K=K, tau=TAU, decode_T=DECODE_T,
                sigma0=SIGMA0, dpo_beta=DPO_BETA, dpo_inner=DPO_INNER, lr=LR, anchor=ANCHOR,
                innate_mean=float(innate.mean()), innate_std=float(innate.std()),
                pi0_bias_mean=b_mean, pi0_std_ratio=b_std_ratio, pi0_w1_innate=b_w1)
    json.dump(dict(meta=meta, results=res, twin=twin_stats),
              open(f"{OUT}/{TAG}_{name}.json", "w"))
    np.savez_compressed(f"{OUT}/{TAG}_{name}_preds.npz", **all_preds)
    return name, meta, res, twin_stats

def summarize(name, meta, res, twin):
    arms = ["frozen", "open", "closed"]
    def m(a, k): return np.array([res[a][s][k] for s in range(meta["seeds"])])  # [S,H+1]
    print(f"\n---- CELL {name} FINAL (round {H}) per-seed [mean over seeds] ----")
    print(f"{'metric':>16} |    frozen    |     open     |    closed")
    for k, lab in [("pop_std","pop std"),("w1_twin","W1 vs twin"),
                   ("pred_w1_pi0","predW1 vs pi0"),("win_U0","winrate U0"),
                   ("win_Ut","winrate Ut"),("flip","pref-flip frac")]:
        cells = []
        for a in arms:
            v = m(a, k)[:, -1]
            cells.append(f"{v.mean():.4f} ({v.min():.3f}-{v.max():.3f})")
        print(f"{lab:>16} | " + " | ".join(cells))
    # closed - open contrast (the model-mediated channel), per seed
    cs, op = m("closed","w1_twin")[:, -1], m("open","w1_twin")[:, -1]
    print(f"  closed-open  W1(vs twin) per seed: "
          + ", ".join(f"{c-o:+.4f}" for c, o in zip(cs, op)))
    cs, op = m("closed","pred_w1_pi0")[:, -1], m("open","pred_w1_pi0")[:, -1]
    print(f"  closed-open  predW1(vs pi0) per seed: "
          + ", ".join(f"{c-o:+.4f}" for c, o in zip(cs, op)))

def figure(name, meta, res, twin):
    arms = [("frozen","0.5"),("open","tab:blue"),("closed","tab:red")]
    panels = [("pop_std","opinion spread std(x_t)"),("w1_twin","W1 from no-AI twin"),
              ("pred_w1_pi0","W1(pred_t, pred_0)  [pred space]"),
              ("win_U0","win vs pi_0 ref  |  under U_0 (innate)"),
              ("win_Ut","win vs pi_0 ref  |  under U_t (current pop)"),
              ("flip","held-out pref-flip fraction")]
    R = range(H + 1)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5)); ax = ax.ravel()
    for j, (key, title) in enumerate(panels):
        for a, c in arms:
            for s in range(meta["seeds"]):
                ax[j].plot(R, res[a][s][key], "-", color=c, lw=0.8, alpha=0.35)
            mean = np.mean([res[a][s][key] for s in range(meta["seeds"])], 0)
            ax[j].plot(R, mean, "-", color=c, lw=2.4, label=a if j == 0 else None)
        if key == "pop_std":
            ax[j].plot(R, twin["std"], "--", color="k", lw=1.3, label="no-AI twin")
        ax[j].set_xlabel("round"); ax[j].set_title(title, fontsize=10)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"[exploratory] closed-loop RLHF (DPO) surrogate — cell={name}  "
                 f"eps_soc={meta['params']['eps_social']} eps_AI={meta['params']['eps_ai']} "
                 f"W={meta['params']['W']} kappa={meta['params']['kappa']}  "
                 f"| pi0 bias mean{meta['pi0_bias_mean']:+.3f} stdratio{meta['pi0_std_ratio']:.2f} "
                 f"| tau={meta['tau']} anchor={meta['anchor']} seeds={meta['seeds']}",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(f"{OUT}/{TAG}_{name}_fig.png", dpi=120)
    print(f"wrote {OUT}/{TAG}_{name}.json / _preds.npz / _fig.png")

if __name__ == "__main__":
    for nm in WANT:
        name, meta, res, twin = run_cell(nm, CELLS[nm])
        summarize(name, meta, res, twin)
        figure(name, meta, res, twin)
