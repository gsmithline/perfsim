# perdomo_loan

Faithful replication of the strategic loan example from:

> Perdomo, Zrnic, Mendler-Dünner, Hardt. **Performative Prediction.** ICML 2020.
> [proceedings.mlr.press/v119/perdomo20a](https://proceedings.mlr.press/v119/perdomo20a/perdomo20a.pdf)

## Setup

- **Dataset:** GiveMeSomeCredit Kaggle competition, loaded via `KaggleDataset("GiveMeSomeCredit")` (requires Kaggle CLI credentials in `~/.kaggle/kaggle.json`).
- **Predictor:** logistic regression (`models/linear.py`) wrapped in a `Predictor` facade (`core/predictor.py`) over (Model, Loss, Learner).
- **Environment:** strategic best-response with quadratic cost on feature shifts (Perdomo et al. Section 5). Implemented in `world.py`, extending `environments/dynamics/strategic_linear.py`.
- **Learners:** ERM (RRM) and one-step gradient (RGD k=1).
- **Epoch size:** 1 (strategic best-response is one-shot; see `max_meaningful_epoch_size` on `StrategicLinearWorld`).

## Reproduction

```bash
python -m perfsim.scenarios.perdomo_loan --mu 100 --n-rounds 50 --learner erm
```

Produces the headline convergence figure from the paper. Visual tolerance gate; literature-replication target.

## Synthetic fallback

Users without Kaggle credentials can run a synthetic-data config; it is not the replication claim, only a smoke test:

```bash
python -m perfsim.scenarios.perdomo_loan --synthetic
```

## Status

Implemented and tested (see `tests/test_perdomo_scenario.py`).
