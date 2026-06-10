# Competition experiments

Platforms compete for users by where they position themselves in an opinion
space; users pick platforms by proximity; the population may move toward the
platforms it uses. The question: when does competition keep opinions diverse,
and when does the feedback loop collapse them?

Run scripts from the repo root, e.g. `python experiments/competition/01_baseline.py`.

| # | File | Where | What it does |
|---|------|-------|--------------|
| 0 | `metrics.py` | local | (todo) shared readouts: platform spread, population diversity, audience overlap, share entropy, time-to-collapse, cycling |
| 1 | `01_baseline.py` | local | static benchmark: best-response landscape + basins. Done. |
| 2 | `02_phase_diagram.py` | local | (todo) moving population vs competition; malleability x competition phase diagram, with the no-feedback control |
| 3 | `03_anchoring.py` | local | (todo) users pulled back to baseline identity; does an anchor stop collapse |
| 4 | `04_adaptation.py` | local | (todo) platform update rule: none / move-to-average / best-response |
| 7 | `07_crowd_shapes.py` | local | (todo) uniform / balanced / lopsided / rare-minority populations |
| 8 | `08_geometry.py` | local | line vs circle, MLP vs scalar, gradient vs best-response. Done. |

Cluster (real LLMs) lives in `experiments/scripts/cluster_pipelines/`:
- adaptation with LLMs (in-context vs fine-tuning)
- label-structure harm (does collapse hurt prediction)

## What we know so far

- Static, fixed population: under softmax choice the best response is to spread
  (peak at the equidistant spot); under hard nearest choice the payoff is flat.
- The circle has multiple equilibria (segmented and paired). Basins are
  parity-dependent: K=3 always segments, K=4 mostly pairs, K=5 mostly segments.
- Use **K=3, tau=0.2** as the baseline where segmentation is the single attractor,
  so any collapse under feedback is attributable to the feedback.
- Geometry alone does not rescue diversity: the line collapses to its center,
  the circle clumps into pairs. Spreading needs smooth (softmax) choice.

`_common.py` holds the shared primitives (distances, the gradient competition
loop, the best-response landscape helpers).
