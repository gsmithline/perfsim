"""End-to-end demo: Perdomo strategic-loan scenario, single mu, in-process.

Drives the full Perdomo loan scenario through `Simulator.run` without any
agent-shell / Executor indirection. Prints per-round PR and stability gap.

Defaults to GiveMeSomeCredit via Kaggle (cached locally). Set
`USE_SYNTHETIC = True` in `config` cell to use the in-process synthetic
fallback (no Kaggle creds required; not the replication claim).
"""

import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def imports():
    import torch

    from perfsim.scenarios.perdomo_loan.config import PerdomoLoanConfig
    from perfsim.scenarios.perdomo_loan.reproduction import run
    return PerdomoLoanConfig, run, torch


@app.cell
def _intro():
    import marimo as mo
    mo.md(
        """
        # Perdomo strategic loan (in-process)

        Replicates Perdomo et al. 2020 strategic-classification setup against
        GiveMeSomeCredit data. Each round, the lender deploys a logistic
        classifier; loan applicants shift their features along the gradient
        of their predicted default probability; the lender retrains; repeat.
        """
    )
    return (mo,)


@app.cell
def config(PerdomoLoanConfig):
    cfg = PerdomoLoanConfig(
        mu=1.0,
        n_rounds=15,
        learner="erm",
        learner_lr=0.01,
        learner_steps=1,
        weight_decay=5e-5,
        seed=0,
        use_synthetic_fallback=False,
    )
    print(
        f"# mu={cfg.mu} learner={cfg.learner} rounds={cfg.n_rounds} "
        f"weight_decay={cfg.weight_decay} synthetic={cfg.use_synthetic_fallback}"
    )
    print(f"# config hash: {cfg.content_hash()}")
    print(f"# strat_features: {list(cfg.strat_features)}")
    return (cfg,)


@app.cell
def execute(cfg, run):
    history = run(cfg)
    return (history,)


@app.cell
def per_round_table(history, torch):
    print(f"# {'round':>5}  {'PR':>12}  {'||delta theta||':>16}")
    for _r in history.records:
        _pr = _r.get("PR")
        _gap = _r.get("stability_gap")
        _pr_s = f"{_pr.item():12.6f}" if isinstance(_pr, torch.Tensor) else f"{'n/a':>12}"
        _gap_s = (
            f"{_gap.item():16.3e}"
            if isinstance(_gap, torch.Tensor)
            else f"{'(init)':>16}"
        )
        print(f"  {_r['round']:>5d}  {_pr_s}  {_gap_s}")
    return


@app.cell
def final_theta(history):
    final = history.records[-1]["theta"]
    print(f"# final ||theta||_2 = {final.norm().item():.6f}")
    print(f"# final theta first 4 entries: {final.flatten()[:4].tolist()}")
    return (final,)


if __name__ == "__main__":
    app.run()
