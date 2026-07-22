# Closed-loop RLHF (DPO): does it reinforce the proxy's initial bias? (2026-07-22)

**Question.** A proxy predicts each agent's opinion and its prediction is **deployed**
into the population (same gate/blend + peer dynamics as the LM pipeline). It is then
updated by **DPO** on stochastic pairwise preferences. Does **closed-loop** RLHF —
where the preference labels come from the model's *own, deployment-shifted*
population — **reinforce the proxy's initial opinion bias**, relative to an
**open-loop** arm that gets labels from a synchronized **no-AI twin**?

**Headline (honest).** **Yes — but only when the deployment lever is strong.**
With everything else identical, judging preferences on the model's own shifted
population keeps both the population mean and the policy ~0.10 nearer the initial
bias than the open arm, on **every seed**. At the realistic operating point (tight
gate + innate anchor) the effect vanishes: closed ≈ open ≈ *corrects* the bias.
This is a **surrogate** result (DPO, not full online RLHF; no LLM) and should be
verified on the real model.

---

## Design (minimal; reuses the real dynamics)
- Population/graph: `fv2.ml_action_setup()` (n=723, innate mean 0.630, std 0.137).
- Deploy + peer step: identical gate/blend algebra and calibrated soft
  bounded-confidence peer step as `rlvr_local.py` / the LM pipeline.
- **Policy = categorical over a discretized opinion grid** (K=41): `p_theta(k|phi)=
  softmax(f_theta(phi))`, the faithful analog of an LLM emitting a distribution over
  numeric tokens. Sampling two candidates is `multinomial`; **DPO log-probs are
  exact**. No Gaussian policy.
- **Feature** `phi=[innate, qwen_prior]` (2-D). `pi_0` is fit (soft target, spread
  sigma0=0.06) to the **real frozen-Qwen prior** (`frz_qwen_e040_s0`), so it
  **reproduces the LLM's own per-node predictions** (`corr(pi0,qwen)=1.00`).
- **Inherited bias, NOT injected.** The real frozen-Qwen prior is a *poor* predictor
  here: `corr(qwen, innate)=0.04`, mean **-0.20** below innate, only ~5 distinct
  values. That mean shift is the bias the closed loop can reinforce.
- **Preferences:** for each agent sample kA,kB ~ pi; `P(A>B)=sigmoid(tau*[U(yA;x)-
  U(yB;x)])`, `U(y;x)=-|y-x|`, tau=12 (log it; tau->inf = deterministic closer-wins).
- **DPO:** moving anchor (prev round's checkpoint) by default; `ANCHOR=fixed` ablates
  to a pi_0 anchor. beta=0.1, 12 inner steps/round, lr=3e-3.

## Three arms (differ ONLY in the preference-label source)
- **frozen** — no DPO; pi_0 deployed every round (pure deployment drift baseline).
- **open** — labels judged by the no-AI twin `x_cf_{t-1}`.
- **closed** — labels judged by the arm's OWN model-influenced pop `x_own_{t-1}`.
All three deploy their current policy into their own population under identical
dynamics. One synchronized no-AI twin (no deployment) is the shared W1 reference and
the open arm's judge.

## Metrics (per round, per seed): population mean & std; W1 vs twin; policy drift
`W1(pred_t,pred_0)`; win rate of pi_t vs a fixed pi_0 reference set under U_0 (innate)
and U_t (current pop), tie-aware; held-out pref-flip fraction; full pred matrices
saved to `*_preds.npz`.

---

## Results (6 seeds, 30 rounds, H=30)

### A. Permissive cell (strong lever) — eps_AI=2.0, W=0.5, kappa=0  — EFFECT PRESENT
`rlhf_closed_permissive_fig.png`

| final (round 30) | frozen | open | closed |
|---|---|---|---|
| population **mean** | 0.429 | 0.660 (+0.03 vs twin) | **0.556 (-0.08 vs twin)** |
| policy drift W1(pred,pi0) | 0.000 | 0.242 | **0.140** |
| W1 vs twin | 0.205 | 0.075 | 0.102 |
| opinion std | 0.200 | 0.009 | 0.072 |
| win rate U_t | 0.51 | 0.97 | 0.96 |

- **closed - open population mean, per seed:** -0.028, -0.127, -0.162, -0.088,
  -0.045, -0.172  (**all 6 negative**, mean **-0.104**). Closed drags the pop toward
  the bias; open corrects past the twin.
- **closed - open policy drift, per seed:** all 6 negative (mean -0.102). Closed
  keeps its policy near its *initial biased* predictions; open moves away.
- Closed locks in by round ~5 (pop mean 0.630 -> 0.556, flat); open drifts up.

### B. Realistic cell (weak lever) — eps_AI=0.4, W=0.3, kappa=0.25 — NO EFFECT
`rlhf_closed_realistic_fig.png`

| final | frozen | open | closed |
|---|---|---|---|
| population mean | 0.541 | 0.636 | 0.637 |
| policy drift | 0.000 | 0.220 | 0.222 |

- **closed - open pop mean per seed:** -0.031, +0.027, -0.002, +0.004, +0.007,
  -0.001 (mean **+0.001**). Mixed signs -> null. Both learning arms correct the bias.
- Tight gate + innate anchor => closed can't reshape its own feedback pop enough to
  differ from the twin.

### C. Fixed-anchor control (permissive) — isolates the feedback channel
`rlhf_fixedanchor_permissive_fig.png`

- With a **fixed pi_0 anchor** (which brakes policy drift), the closed-open gap gets
  **cleaner**, not smaller: pop-mean gap -0.132,-0.139,-0.137,-0.140,-0.151,-0.116
  (all 6, mean **-0.136**); W1-vs-twin positive on **all 6 seeds**; policy-drift gap
  all negative. => the reinforcement is the **model-mediated population-feedback
  channel**, not a moving-anchor drift artifact.

---

## Interpretation
Closed-loop RLHF entrenches the initial bias because the model **deploys -> reshapes
the very population that later grades its preferences**, manufacturing feedback that
validates its prior. Open-loop, graded by the un-influenced twin, instead *corrects*
the bias. The fingerprint shows up in **both** population space (mean pulled toward
the bias) and policy space (predictions stay near pi_0), unanimously across seeds —
and only when the deployment lever is strong enough to move the feedback population.

## Caveats
Surrogate: DPO (not full online RLHF); categorical policy over a 2-D `[innate,qwen]`
feature standing in for the LLM's profile; tau=12 (finite); this frozen-Qwen prior is
a poor predictor (⊥ innate, -0.20 shift), so the inherited bias is strong; one
operating cell per regime. **Verify on the real model.**

## Exact commands (run from perfsim repo root)
```
# main run: both cells, 6 seeds, moving anchor
python3 notes/rlhf_closed/rlhf_closed.py
# fixed-anchor control (permissive)
ANCHOR=fixed CELLS=permissive SEEDS=6 TAG=rlhf_fixedanchor python3 notes/rlhf_closed/rlhf_closed.py
# (re)generate labeled figures from saved JSON (no re-sim)
python3 notes/rlhf_closed/plot_rlhf.py
# env knobs: EPS_SOCIAL EPS_AI W KAPPA H SEEDS K TAU DECODE_T SIGMA0 DPO_BETA DPO_INNER LR ANCHOR CELLS TAG
```

## Files
- `rlhf_closed.py` — the 3-arm closed-loop DPO surrogate (both cells).
- `plot_rlhf.py` — re-plot labeled figures from JSON.
- `rlhf_closed_permissive_fig.png` — **headline** (effect present).
- `rlhf_closed_realistic_fig.png` — realistic cell (null).
- `rlhf_fixedanchor_permissive_fig.png` — fixed-anchor control.
- `*.json` — per-round scalar metrics + metadata; `*_preds.npz` — full pred matrices.

## Not yet run (next steps)
- tau->inf closer-wins sanity check; lever sweep between the two cells to map where
  the effect turns on; cluster LoRA RLHF on Qwen-7B with the same 3 arms.
