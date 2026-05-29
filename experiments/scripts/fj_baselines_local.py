"""Non-LLM baselines for the Fig 2a opinion-dynamics panel (local, no GPU).

Runs the same FJ performative loop (deploy every round, replace, settle to
equilibrium, 30 rounds) on the real Pokec graph with four non-LLM predictors,
the dashed reference lines of the figure:

  perfect -- predicts the true innate opinion (no error)
  mean    -- constant predictor (fits the training mean each round)
  ridge   -- L2-regularized LinearModel on the features
  mlp     -- small MLPModel on the features

Logs population AND prediction mean/variance per round, writes trajectories to
JSON, and renders the two-panel (mean, variance vs round) figure to PNG.
"""

import importlib.util
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from perfsim.core.model import Model
from perfsim.environments.dynamics import FJWorld, normalize_adjacency
from perfsim.learners import ERMLearner, GradientLearner
from perfsim.losses import L2RegularizedLoss, MSELoss
from perfsim.models import LinearModel, MLPModel

POKEC_DIR = Path("examples/pokec")
_CM = Path(__file__).resolve().parent / "_collapse_metrics.py"
_spec = importlib.util.spec_from_file_location("_collapse_metrics", _CM)
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)


def load_pokec():
    def L(f):
        return pickle.load(open(POKEC_DIR / f, "rb"))
    graph = L("lcc_graph_relation_to_smoking.pk")
    feats = np.concatenate(
        [L("labeled_feature_matrix_relation_to_smoking_False.pk"),
         L("unlabeled_feature_matrix_relation_to_smoking_False.pk")], axis=0)
    innate = np.array(list(L("parametric_params/y_label2163.pk"))
                      + list(L("parametric_params/y_unlabel_label2163.pk")), dtype=np.float64)
    peer = L("parametric_params/hetero_peer_sus2163.pkl")
    plat = L("parametric_params/hetero_platform_sus2163.pkl")
    W = normalize_adjacency(torch.tensor(nx.to_numpy_array(graph, nodelist=list(graph.nodes)),
                                         dtype=torch.float32))
    feats = torch.tensor(feats, dtype=torch.float32)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)  # standardize
    return {
        "graph": W, "features": feats,
        "innate": torch.tensor(innate, dtype=torch.float32),
        "peer_sus": torch.tensor(peer, dtype=torch.float32),
        "platform_sus": torch.tensor(plat, dtype=torch.float32),
    }


class PerfectModel(Model):
    """Predicts the true innate opinion (the no-error oracle)."""
    def __init__(self, innate):
        super().__init__()
        self.register_buffer("innate", innate.clone())
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.innate[: x.shape[0]].unsqueeze(-1)


class ConstantModel(Model):
    """Single bias; ERM on MSE drives it to the training mean."""
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.b.expand(x.shape[0], 1)


def build(kind, d, innate):
    loss = MSELoss()
    if kind == "perfect":
        return PerfectModel(innate), None
    if kind == "mean":
        m = ConstantModel()
        return m, ERMLearner(m, loss, max_iter=50)
    if kind == "ridge":
        m = LinearModel(in_features=d, out_features=1, bias=True)
        return m, ERMLearner(m, L2RegularizedLoss(loss, weight_decay=1e-2), max_iter=100)
    if kind == "mlp":
        m = MLPModel(in_features=d, hidden_dims=[16, 16], out_features=1,
                     activation="relu", final_activation="sigmoid", init_seed=0)
        return m, GradientLearner(m, loss, lr=0.05, steps_per_round=300, optimizer="adam")
    raise ValueError(kind)


def select_train_data(buffer, regime, cur_dep):
    """Same deployment-schedule aggregation as the cluster runner."""
    if not buffer:
        return None
    if regime == "replace":
        chosen = [buffer[-1]]
    elif regime == "accumulate":
        chosen = buffer
    elif regime == "deployed_into":
        chosen = [b for b in buffer if b["dep"] == cur_dep] or [buffer[-1]]
    elif regime == "not_deployed_into":
        chosen = [b for b in buffer if b["dep"] < cur_dep] or buffer
    else:
        raise ValueError(f"unknown regime {regime!r}")
    return {"x": torch.cat([b["x"] for b in chosen], 0),
            "y": torch.cat([b["y"] for b in chosen], 0)}


def run_baseline(kind, data, n_rounds=30, epoch_size=20, n_labeled=1730, seed=0,
                 deploy_every=1, data_regime="replace"):
    torch.manual_seed(seed)
    n, d = data["features"].shape
    feats, innate = data["features"], data["innate"]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:n_labeled] = True
    world = FJWorld(innate=innate, graph=data["graph"], peer_sus=data["peer_sus"],
                    platform_sus=data["platform_sus"], features=feats)
    world.reset(seed=seed)
    model, learner = build(kind, d, innate)

    initial = {"x": feats[mask], "y": innate[mask].unsqueeze(-1)}
    buffer = []
    cur_dep = -1
    rows = []
    op0 = None
    prev_op = None
    for t in range(n_rounds):
        if t % deploy_every == 0:
            train_data = initial if t == 0 else select_train_data(buffer, data_regime, cur_dep)
            if learner is not None and train_data is not None:
                learner.train(train_data)
            cur_dep += 1
        with torch.no_grad():
            preds = model(feats).squeeze(-1).float()
        world.run(model, n_steps=epoch_size)
        op = world.state["opinion"].float()
        if op0 is None:
            op0 = op.clone()
        # full collapse suite for BOTH distributions (perplexity is LLM-only)
        row = {"round": t, "deployment": cur_dep}
        row.update({f"op_{k}": v for k, v in cm.summary(op).items()})
        row.update({f"pred_{k}": v for k, v in cm.summary(preds).items()})
        row["jaccard_init"] = cm.jaccard_support(op, op0)
        if prev_op is not None:
            row["jaccard_prev"] = cm.jaccard_support(op, prev_op)
        rows.append(row)
        prev_op = op.clone()
        buffer.append({"t": t, "dep": cur_dep,
                       "x": feats[mask], "y": op[mask].detach().unsqueeze(-1)})
    return rows


def main():
    data = load_pokec()
    innate = data["innate"]
    out = Path("runs/fj_baselines_local")
    out.mkdir(parents=True, exist_ok=True)

    traj = {}
    print(f"innate: mean={innate.mean():.4f} var={innate.var():.4f}")
    for kind in ("perfect", "mean", "ridge", "mlp"):
        traj[kind] = run_baseline(kind, data)
        r = traj[kind][-1]
        print(f"{kind:8} final  op_mean={r['op_mean']:.4f} op_var={r['op_var']:.5f}  "
              f"pred_mean={r['pred_mean']:.4f} pred_var={r['pred_var']:.5f}")
    (out / "baselines.json").write_text(json.dumps(traj, indent=2))

    # population-side panels for the full suite; innate reference where defined
    panels = [
        ("op_mean", "mean opinion", float(innate.mean())),
        ("op_var", "variance", float(innate.var())),
        ("op_entropy", "entropy", cm.entropy(innate)),
        ("op_eff_support", "eff support", cm.eff_support(innate)),
        ("op_mode_mass", "mode mass", cm.mode_mass(innate)),
        ("jaccard_init", "Jaccard vs round 0", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (key, title, ref) in zip(axes.ravel(), panels):
        for kind, rows in traj.items():
            ts = [r["round"] for r in rows if key in r]
            ys = [r[key] for r in rows if key in r]
            ax.plot(ts, ys, "--", label=kind)
        if ref is not None:
            ax.axhline(ref, ls=":", c="gray", label="innate")
        ax.set(title=f"population {title} across rounds", xlabel="retraining step t",
               ylabel=title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fj_baselines.png", dpi=110)
    print(f"wrote {out}/baselines.json and {out}/fj_baselines.png")


if __name__ == "__main__":
    main()
