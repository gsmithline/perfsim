# perdomo_loan

Faithful replication of the strategic loan example from:

> Perdomo, Zrnic, Mendler-Dünner, Hardt. **Performative Prediction.** ICML 2020.
> [proceedings.mlr.press/v119/perdomo20a](https://proceedings.mlr.press/v119/perdomo20a/perdomo20a.pdf)

## Setup

- **Dataset:** GiveMeSomeCredit Kaggle competition, loaded via `KaggleDataset("GiveMeSomeCredit")` (requires Kaggle CLI credentials in `~/.kaggle/kaggle.json`).
- **Predictor:** logistic regression (`models/linear.py`).
- **World:** strategic best-response with quadratic cost on feature shifts (Perdomo et al. Section 5). Implemented in `world.py`, extending `worlds/strategic_linear.py`.
- **Learners:** ERM (RRM) and one-step gradient (RGD k=1).

## Reproduction

```bash
python -m perfsim.scenarios.perdomo_loan --mu 100 --n-rounds 50 --learner erm
```

Produces the headline convergence figure from the paper. Visual tolerance gate; v1 literature-replication target (DESIGN.md Section 15 test 9).

## Synthetic fallback

Users without Kaggle credentials can run a synthetic-data config; it is not the replication claim, only a smoke test:

```bash
python -m perfsim.scenarios.perdomo_loan --synthetic
```

## Status

v0 skeleton stub. No implementation yet; see top-level DESIGN.md.
