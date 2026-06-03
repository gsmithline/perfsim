"""Local MLP beta sweep on macro or covid ABM with structured targets.

Uses multi-feature input + per-bucket target so the MLP has to learn many
distinct per-demographic outputs. L2 anchor to init weights then has
meaningful work to do at high beta.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from perfsim.core.loss import Loss
from perfsim.core.model import Model
from perfsim.core.types import Data
from perfsim.learners.gradient import GradientLearner
from perfsim.losses import MSELoss
from perfsim.models.mlp import MLPModel
from perfsim.simulator import Simulator


class ClampedMLP(MLPModel):
    def forward(self, x):
        return torch.sigmoid(super().forward(x))


class L2AnchoredLoss(Loss):
    def __init__(self, beta, anchor_params):
        self.beta = beta
        self.anchor = {n: p.detach().clone() for n, p in anchor_params}
        self._mse = MSELoss()

    def __call__(self, model: Model, data: Data, reduction: str = "mean") -> torch.Tensor:
        base = self._mse(model, data, reduction=reduction)
        if self.beta == 0:
            return base
        reg = sum(((p - self.anchor[n]) ** 2).sum() for n, p in model.named_parameters())
        return base + self.beta * reg


def macro_setup(n_agents=100, calibrated_uac_path=None):
    from perfsim.scenarios._deprecated.at_macro import make_macro_env

    env = make_macro_env(
        init_seed=0,
        yaml_name="config_100_agents.yaml",
        n_agents=n_agents,
        keep_trajectory=True,
        strict_signal=False,
    )

    if calibrated_uac_path is not None:
        uac_data = torch.load(calibrated_uac_path, weights_only=False)
        for tf_key in env.runner.initializer.transition_function:
            tf = env.runner.initializer.transition_function[tf_key]
            for _, module in tf.named_modules():
                if hasattr(module, "external_UAC"):
                    n_steps_local = module.external_UAC.shape[0]
                    with torch.no_grad():
                        module.external_UAC.copy_(uac_data[:n_steps_local])
                    print(f"  [macro] loaded calibrated UAC from {calibrated_uac_path}", flush=True)
                    break
    _cit = env.runner.state["agents"]["consumers"]
    _age = _cit["age"].squeeze().long()
    _gen = _cit["gender"].squeeze().long()
    _eth = _cit["ethnicity"].squeeze().long()
    n = _age.shape[0]

    _key = _age * 100 + _gen * 10 + _eth
    _unique_keys, _bucket = torch.unique(_key, return_inverse=True)
    n_bk = int(_unique_keys.shape[0])
    torch.manual_seed(42)
    _bucket_targets = 0.15 + 0.7 * torch.rand(n_bk)

    feats = torch.stack([_age.float() / 5.0, _gen.float(), _eth.float() / 5.0], dim=-1)

    def state_extractor(runner):
        env_state = runner.state["environment"]
        inflation = 0.0
        pi = env_state.get("P_i")
        if pi is not None:
            try:
                val = float(pi[-1][-1].item())
                if abs(val) < 1.0:
                    inflation = val
            except Exception:
                pass
        target = (_bucket_targets[_bucket] - 0.5 * inflation).clamp(0.05, 0.95)
        return {
            "x": feats.detach(),
            "y": target.detach().reshape(-1, 1),
            "agent_idx": torch.arange(n),
        }

    env._state_extractor = state_extractor
    env._feature_provider = lambda runner: feats.detach()

    def metrics(sim):
        c = sim.env.runner.state["agents"]["consumers"]
        e = sim.env.runner.state["environment"]
        assets = float(c["assets"].float().mean().item())
        cons = c.get("consumption_propensity")
        cons_mean = float(cons.float().mean().item()) if cons is not None else 0.0
        pi = e.get("P_i")
        inflation = float(pi[-1][-1].item()) if pi is not None else 0.0
        u = e.get("U")
        unemp = 0.0
        if u is not None:
            row = u[-1]
            nz = row.nonzero(as_tuple=True)[0]
            unemp = float(row[nz[-1]].item()) if len(nz) else 0.0
        with torch.no_grad():
            preds = sim.predictor.model(feats).squeeze()
        train_loss = float(((preds - _bucket_targets[_bucket].clamp(0.05, 0.95)) ** 2).mean().item())
        return {
            "assets": assets,
            "consumption": cons_mean,
            "inflation": inflation,
            "unemployment": unemp,
            "pred_mean": float(preds.mean().item()),
            "pred_std": float(preds.std().item()),
            "train_loss_vs_target": train_loss,
            "n_buckets": n_bk,
        }

    return env, metrics, feats.shape[-1]


def covid_setup(calibrated_r2_path=None):
    from perfsim.scenarios.at_covid import make_covid_env

    env = make_covid_env(
        init_seed=0,
        initial_infections_fraction=0.05,
        keep_trajectory=True,
        strict_signal=False,
    )

    if calibrated_r2_path is not None:
        r2_data = torch.load(calibrated_r2_path, weights_only=False)
        for tf_key in env.runner.initializer.transition_function:
            tf = env.runner.initializer.transition_function[tf_key]
            for _, module in tf.named_modules():
                if hasattr(module, "calibrate_R2"):
                    with torch.no_grad():
                        module.calibrate_R2.copy_(r2_data.expand_as(module.calibrate_R2))
                    print(f"  [covid] loaded calibrated R2 from {calibrated_r2_path}", flush=True)
                    break
    _cit = env.runner.state["agents"]["citizens"]
    _envs = env.runner.state["environment"]
    _age = _cit["age"].squeeze().long()
    _mi = _envs["mean_interactions"].squeeze().float()
    _mi_bucket = _mi.round().long()
    n = _age.shape[0]

    _key = _age * 100 + _mi_bucket
    _unique_keys, _bucket = torch.unique(_key, return_inverse=True)
    n_bk = int(_unique_keys.shape[0])
    torch.manual_seed(42)
    _bucket_targets = 0.1 + 0.8 * torch.rand(n_bk)

    feats = torch.stack([_age.float() / 5.0, _mi / _mi.max().clamp(min=1.0)], dim=-1)

    def state_extractor(runner):
        c = runner.state["agents"]["citizens"]
        ds = c["disease_stage"].squeeze()
        infected = ((ds == 1) | (ds == 2)).float()
        bucket_inf = torch.zeros(n_bk).scatter_add_(0, _bucket, infected)
        bucket_n = torch.zeros(n_bk).scatter_add_(0, _bucket, torch.ones(n))
        local_rate = (bucket_inf / bucket_n.clamp(min=1.0))[_bucket]
        target = (_bucket_targets[_bucket] + 0.3 * local_rate).clamp(0.05, 0.95)
        return {
            "x": feats.detach(),
            "y": target.detach().reshape(-1, 1),
            "agent_idx": torch.arange(n),
        }

    env._state_extractor = state_extractor
    env._feature_provider = lambda runner: feats.detach()

    def metrics(sim):
        c = sim.env.runner.state["agents"]["citizens"]
        e = sim.env.runner.state["environment"]
        ds = c["disease_stage"].squeeze()
        n_total = int(ds.shape[0])
        n_susceptible = int((ds == 0).sum().item())
        di = e.get("daily_infected")
        di_sum = float(di.sum().item()) if di is not None else 0.0
        with torch.no_grad():
            preds = sim.predictor.model(feats).squeeze()
        train_loss = float(((preds - _bucket_targets[_bucket].clamp(0.05, 0.95)) ** 2).mean().item())
        return {
            "daily_infected_sum": di_sum,
            "fraction_non_S": (n_total - n_susceptible) / n_total,
            "pred_mean": float(preds.mean().item()),
            "pred_std": float(preds.std().item()),
            "train_loss_vs_target": train_loss,
            "n_buckets": n_bk,
        }

    return env, metrics, feats.shape[-1]


def run_one(simulator: str, beta: float, n_rounds=15, k_steps=3, n_agents=100, calibrated_uac_path=None, calibrated_r2_path=None):
    torch.manual_seed(0)
    if simulator == "macro":
        env, metrics_fn, in_features = macro_setup(n_agents=n_agents, calibrated_uac_path=calibrated_uac_path)
    elif simulator == "covid":
        env, metrics_fn, in_features = covid_setup(calibrated_r2_path=calibrated_r2_path)
    else:
        raise ValueError(simulator)

    model = ClampedMLP(in_features=in_features, out_features=1, hidden_dims=[64, 64, 32])
    loss = L2AnchoredLoss(beta=beta, anchor_params=list(model.named_parameters()))
    learner = GradientLearner(model, loss, lr=0.02, steps_per_round=100)

    sim = Simulator(env=env, learner=learner, loss=loss, metrics={"m": metrics_fn})
    sim.run(n_rounds=n_rounds, epoch_size=k_steps, seed=0)
    return sim.history[-1]["m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulator", choices=["macro", "covid", "both"], default="both")
    ap.add_argument("--betas", nargs="*", type=float, default=[0.0, 0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--n-agents", type=int, default=100)
    ap.add_argument("--n-rounds", type=int, default=15)
    ap.add_argument("--calibrated-uac", type=str, default=None)
    ap.add_argument("--calibrated-r2", type=str, default=None)
    args = ap.parse_args()

    sims = ["macro", "covid"] if args.simulator == "both" else [args.simulator]

    for sim_name in sims:
        print(f"\n=== {sim_name.upper()} beta sweep ===")
        results = []
        for beta in args.betas:
            t0 = time.time()
            final = run_one(
                sim_name, beta, n_rounds=args.n_rounds, n_agents=args.n_agents,
                calibrated_uac_path=args.calibrated_uac if sim_name == "macro" else None,
                calibrated_r2_path=args.calibrated_r2 if sim_name == "covid" else None,
            )
            elapsed = time.time() - t0
            print(f"  beta={beta:7.2f}  pred_mean={final['pred_mean']:.4f}  pred_std={final['pred_std']:.4f}  "
                  f"train_loss={final['train_loss_vs_target']:.4f}  ({elapsed:.1f}s)")
            for k, v in final.items():
                if k not in ("pred_mean", "pred_std", "train_loss_vs_target", "n_buckets"):
                    print(f"           {k}={v:.4f}")
            results.append({"beta": beta, **final})
        out_path = Path(f"experiments/runs/mlp_beta_sweep_{sim_name}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
