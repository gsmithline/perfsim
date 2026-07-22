# Open-loop material — set aside (2026-07-21)

Parked here because the OPEN loop may not be part of the story (undecided).
Nothing is lost: the `.json` files hold **both** loops' full results, so the
open-loop arm can be restored into any figure at any time.

## What "open" vs "closed" means
- **closed** (solid, the main story): the LLM proxy **retrains each round** on the
  population's realized opinions — the performative loop.
- **open** (dashed, parked): same anchored serve function, but the model is
  **frozen** (reads current opinion x_t, never retrains). Isolates the loop.

## Files
- `eps_ai_sweep.py` / `eps_ai_sweep.json` — ε_AI single-knob, W=0.30, κ=0.25,
  3 regimes (ε_social 0.10/0.30/0.60), OPEN + CLOSED. JSON keys: `{regime}|{loop}|{eps_ai}`.
- `w_sweep.py` / `w_sweep.json` — W single-knob, ε_AI=0.40, κ=0.25, same 3 regimes,
  OPEN + CLOSED. JSON keys: `{regime}|{loop}|{W}`.
- `eps_ai_sweep_OPEN_vs_CLOSED_fig.png`, `w_sweep_OPEN_vs_CLOSED_fig.png` — the
  combined figures (dashed=open, solid=closed).

## Headline the open arm carried (if you want it back)
Closing the loop **launders** the AI's fingerprint: vs the frozen model, the
closed loop shows ~30–40% LOWER model separation and LOWER population distortion,
and slightly DEEPER dispersion collapse. i.e. the performative loop dissolves part
of the model's identity into the population it shaped rather than amplifying it.

## To restore into the story
Re-run `eps_ai_sweep.py` / `w_sweep.py` (regenerates both arms), or re-plot from
the JSON including the `|open|` keys. The main story currently uses closed-only
figures at `notes/eps_ai_sweep_closed_fig.png` and `notes/w_sweep_closed_fig.png`.
