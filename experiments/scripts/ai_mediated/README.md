# ai_mediated experiments

AI-mediated human data / market-mediated collapse. Pairs with `perfsim/scenarios/ai_mediated`.
Run all from the repo root (`python experiments/scripts/ai_mediated/<file>.py`).

## Real-LLM performative loop (cluster)
- `run_resume_llm_loop.py` — the loop: θ scores resumes, score conditions a Qwen rewrite, θ retrains on the mediated data, rewrites carry forward (self-consuming). Conditions phenomenon/identification/generic; regimes replace/accumulate/clean_anchor/mediated_anchor. Submitted via `experiments/condor/at_resume_llm_{smoke,loop}.sub`.
- `embed_and_cache.py` — one-time: embed originals + GPT-4o rewrites (bge-base-en) to `~/.cache/perfsim/datasets/two_tickets/embeddings.npz`.

## One-shot analysis of the two-tickets data (no loop)
- `embed_real_rewrites.py` — embed real LLM rewrites, measure recoverability / diversity / idempotence / cross-model.
- `plot_real_rewrites.py` — figure for the above.

## Quasi loops (no LLM, surrogate mediator)
- `resume_mediation_loop.py` — TF-IDF features, quasi mediator, retrain regimes.
- `resume_collapse_loop.py` — self-consuming loop with a surrogate mediator fit from real rewrite pairs.

## Bandit-market / replicator (no LLM)
- `bandit_market_sweep.py` — Δ(p): fitness gap of using the assistant vs adoption share.
- `bandit_kappa_sweep.py` — adoption-vs-cost phase diagram (imports the sweep).
- `bandit_competitive.py` — top-k zero-sum reward; deadweight-loss / red-queen check.
- `bandit_regimes.py` — coupled (adoption, homogenization) system; the three regimes + (μ, ρ) map.

Figures land in `runs/two_tickets_analysis/`.
