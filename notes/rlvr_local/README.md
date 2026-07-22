# Local RLVR surrogate — can a predictor reward-hack by shaping the population? (2026-07-21)

**Question.** A model predicts each node's current pre-platform opinion; reward =
negative prediction error; the prediction is **deployed** and shapes next round's
opinion through the existing platform + peer dynamics. Optimizing **cumulative**
reward, does the model raise long-run reward by making the population *easier to
predict* — and does it distort the population to do so?

**Headline (honest, and it INVERTS the stated hypothesis).**
In this surrogate the destructive "reward-hack" (collapse the population to make it
predictable) is a **MYOPIC** pathology, **not** a long-horizon one. Where the AI
lever is strong, the **long-horizon planner Pareto-dominates the myopic optimizer:
higher reward AND less population distortion.** Where the lever is weak (the
realistic cluster operating point), horizon barely matters and distortion is small.
The stated pattern — *long-horizon* gets more reward *while distorting more* — is
**not** observed at either operating point tested.

This is a surrogate result and it contradicts the prior; it should be treated as a
hypothesis to verify on the real LLM, not as settled.

---

## Setup (all reused from the real code)
- Population/graph: `fv2.ml_action_setup()` → innate opinions (n=723, mean 0.630,
  std 0.137) + the real graph (mean degree 15). Same object the LM pipeline uses.
- Gate/blend algebra: identical to the pipeline
  `b=κx*+(1−κ)x ; z=where(|ŷ−x|<ε_AI,(1−W)b+Wŷ,b)` (soft gate, temp 0.02).
- Peer step: a **differentiable** soft bounded-confidence Deffuant, **calibrated to
  the real `ab_sweep`** dispersion trajectory at ε_social=0.10 (rmse 0.014,
  rate=0.20, τ=0.02). Differentiability is what lets long-horizon credit be computed
  by backprop-through-time; the real `ab_sweep` (pair sampling + hard threshold) is
  not differentiable, so the local test uses this faithful mirror.

## The crux: FEATURE_NOISE
The policy predicts from a **fixed** feature `φ`. If `φ==innate`, the easiest-AND-
least-distorted population is `x=innate` itself, so a reward-maximizer just *holds*
the population at innate — no hack is even possible. That is unfaithful to an LLM,
whose profile does **not** determine the opinion. So we degrade the feature:
`φ = innate + FEATURE_NOISE·η`. Now the natural (no-AI) population is only partly
predictable and distortion becomes a real lever. `FEATURE_NOISE=0` recovers the
trivial no-hack case; `>0` is the faithful regime.

## Policy (the anchor surrogate)
`ŷ = clip(g_θ(φ) + O_SCALE·o)`, `g_θ`=small MLP init from the **real Qwen-7B prior**
(`runs/pokec_gated_lm/frz_qwen_e040_s0`), `o`=frozen residual (dropped, `O_SCALE=0`,
so the policy *can* collapse the population onto a smooth function of φ).

## Conditions (differ by exactly one flag)
1. **myopic** — detach state each round → gradient never credits shaping (γ=0).
2. **long_horizon** — full BPTT through the dynamics → credits shaping (γ=1).
3. **long_nodeploy** — long-horizon objective but deploy weight `W=0` → no lever to
   shape; == the synchronized no-AI population. The clean control.
Plus **myopic_online** — a faithful greedy control that mirrors per-round LLM SFT:
persistent weights, each round fit `g_θ(φ)→x_t`, deploy, advance. (Confirms the
myopic collapse is not a stationary-optimization artifact.)

## Metrics logged per round
prediction reward `r_t=−mean((ŷ−x)²)`; opinion spread `std(x)`; distance from innate
`mean|x−x*|`; **Wasserstein `W1(x_t, no-AI twin_t)`** (synchronized counterfactual).

---

## Results

### A. Permissive lever — ε_AI=2.0 (open gate), W=0.5, κ=0, feat_noise=0.12, H=30
(`rlvr_hackon.json`, `rlvr_online.json`; figure `rlvr_traj_fig.png`)

| condition | cum_reward ↑ | final spread | \|x−innate\| | **W1 vs no-AI ↓** |
|---|---|---|---|---|
| **long_horizon** (plans) | **−0.016** | 0.067 | 0.075 | **0.044** |
| myopic (stationary) | −0.019 | 0.021 | 0.092 | 0.065 |
| myopic (online, greedy) | −0.064 | **0.007** | **0.143** | **0.122** |
| long_nodeploy (control) | −0.192 | 0.098 | 0.059 | 0.001 |

→ Long-horizon: **highest reward, LEAST distortion.** Myopic (esp. online) collapses
the population to a near-point (spread 0.007) far from innate — **most** distortion,
**worse** reward. No-deploy: zero distortion, worst reward (can't shape at all).

### B. Realistic lever (the cluster cell) — ε_AI=0.4, W=0.3, κ=0.25, feat_noise=0.12
(`rlvr_paperpt.json`)

| condition | cum_reward | final spread | \|x−innate\| | W1 vs no-AI |
|---|---|---|---|---|
| myopic | −0.049 | 0.085 | 0.048 | 0.030 |
| long_horizon | −0.049 | 0.082 | 0.049 | 0.033 |
| myopic (online) | −0.113 | 0.082 | 0.049 | 0.032 |
| long_nodeploy | −0.206 | 0.124 | 0.019 | 0.000 |

→ Weak lever (tight gate + innate anchor): myopic ≈ long-horizon on everything; all
distortion is small. The script prints "SUPPORTED: True" here only on **negligible**
margins (W1 0.033 vs 0.030, reward −0.0490 vs −0.0495) — within optimization noise,
not a real effect.

### C. FEATURE_NOISE sweep at the permissive lever (`rlvr_sweep.json`, `rlvr_sweep_fig.png`)

| feat_noise | long R / W1 | myopic_online R / W1 | nodeploy R |
|---|---|---|---|
| 0.00 | −0.005 / 0.024 | −0.060 / 0.108 | −0.078 |
| 0.08 | −0.012 / 0.028 | −0.065 / 0.130 | −0.143 |
| 0.16 | −0.020 / 0.055 | −0.067 / 0.143 | −0.232 |
| 0.24 | −0.022 / 0.067 | −0.068 / 0.145 | −0.286 |

→ Robust across the lever: **long-horizon always gets more reward and less distortion
than myopic**; the distortion gap widens as the feature gets less informative.

---

## Interpretation
- **Reward-hacking-via-collapse is real but MYOPIC.** Greedy per-round accuracy
  maximization, in a loop where predictions are deployed, spirals into a
  performative collapse (everyone dragged together → trivially predictable). This is
  most severe with a strong lever (open gate, high W, weak anchor).
- **Long-horizon planning is gentler,** not worse. Crediting the downstream effect of
  its predictions, it settles the population onto a low-error manifold near innate
  *without* over-collapsing — higher reward, less distortion.
- **The hack needs an uninformative feature** (`FEATURE_NOISE>0`). If the model's
  features determine the opinion, the reward-optimal move is non-distortive
  (hold at truth). LLMs predict opinions from profiles that *don't* determine them,
  so the faithful regime is `>0`.
- **At the realistic operating point** (tight gate + anchor) the whole effect is muted.

## Caveats
- Differentiable **surrogate** dynamics (calibrated mirror of `ab_sweep`), not the
  real sampler; single seed; small MLP policy; long-horizon is analytic policy
  gradient (BPTT), not model-free RL. The real LLM loop cannot be BPTT'd — the
  cluster stage must use model-free RLVR (GRPO). **This surrogate contradicts the
  prior and must be checked on the real model.**

## Exact commands (run from repo root; `MPLCONFIGDIR` set to a writable dir)
```
# realistic operating point (cluster cell)
EPS_AI=0.40 W=0.30 KAPPA=0.25 FEATURE_NOISE=0.12 O_SCALE=0 STEPS=400 H=30 TAG=rlvr_paperpt python3 rlvr_local.py
# permissive operating point (strong lever)
EPS_AI=2.0  W=0.50 KAPPA=0.00 FEATURE_NOISE=0.12 O_SCALE=0 STEPS=400 H=30 TAG=rlvr_hackon  python3 rlvr_local.py
# faithful online-myopic control (same env vars as the point you want)
EPS_AI=2.0  W=0.50 KAPPA=0.00 FEATURE_NOISE=0.12 O_SCALE=0 H=30 INNER=8 python3 rlvr_online.py
# feature-noise sweep (permissive lever; fixes its own env)
python3 rlvr_sweep.py
# consolidated 4-arm trajectory figure (reads rlvr_hackon.json + rlvr_online.json)
python3 plot_rlvr_traj.py
```

## Files
- `rlvr_local.py` — 3-condition BPTT surrogate (myopic/long/long-nodeploy).
- `rlvr_online.py` — faithful online-myopic control.
- `rlvr_sweep.py` — FEATURE_NOISE lever sweep.
- `plot_rlvr_traj.py` — consolidated trajectory figure.
- `rlvr_traj_fig.png` — **headline** 4-arm trajectories (permissive lever).
- `rlvr_sweep_fig.png` — reward + distortion vs feature noise.
- `rlvr_paperpt_fig.png` / `rlvr_hackon_fig.png` — 3-arm figs at each operating point.
- `*.json` — full per-round logs + metadata for every run.

## Cluster stage (NOT run — needs decisions + submit permission)
The LM pipeline already logs the reward and all distortion metrics: `acc_served`
(reward = accuracy on the deployed population), `w1_cf` (Wasserstein vs the no-AI
twin), `op_std` (spread), `op_bias` (distance from innate), and maintains the
matched no-AI counterfactual `ab_x_cf`. The **existing closed-loop SFT** is already a
myopic accuracy-maximizer in the performative loop, so the *myopic* and *no-influence*
arms are cheap/mostly built. A true **long-horizon** arm (credit through a 30-round
7B loop) is the expensive/hard piece and has no cheap faithful form. See the message
thread for the proposed cheap design and the open decisions.
