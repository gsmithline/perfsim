"""β prior-anchor sweep on the FJ performative loop.

Local analog of KL-to-pretrained: a penalty of strength β pulls the model's
predictions toward the original diverse innate signal, so the model resists
collapsing to the mean. Crossed with replace vs accumulate training regimes.

Tracks, per round:
  pred_std  -- spread of the model's predictions (model-side collapse diagnostic)
  op_std    -- spread of the population opinions (the thing that matters)
  tail_frac -- fraction of agents with |opinion - 0.5| > 0.15 (rare/extreme survival)

The dissociation claim: a β range where pred_std looks healthy while op_std /
tail_frac are already collapsed (model dashboard green, population tail gone).
"""

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from perfsim.core.loss import Loss
from perfsim.environments.dynamics import FJWorld, normalize_adjacency
from perfsim.losses import MSELoss
from perfsim.models import MLPModel

POKEC_DIR = Path("examples/pokec")


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
    return {
        "graph": W,
        "features": torch.tensor(feats, dtype=torch.float32),
        "innate": torch.tensor(innate, dtype=torch.float32),
        "peer_sus": torch.tensor(peer, dtype=torch.float32),
        "platform_sus": torch.tensor(plat, dtype=torch.float32),
    }


class AnchoredMSELoss(Loss):
    """MSE on data + β * MSE(prediction, original innate anchor).

    anchor is the (n_labeled,) original innate signal; for accumulated data
    (rows = k copies of the labeled block) it is tiled to match.
    """

    def __init__(self, beta: float, anchor: torch.Tensor) -> None:
        self.beta = beta
        self.anchor = anchor

    def __call__(self, model, data, *, reduction="mean"):
        y_hat = model(data["x"])
        y = data["y"].to(y_hat.dtype).view(y_hat.shape)
        fit = (y_hat - y).pow(2).mean()
        if self.beta == 0.0:
            return fit
        m = y_hat.shape[0]
        reps = m // self.anchor.shape[0]
        anc = self.anchor.repeat(reps).view(y_hat.shape).to(y_hat.dtype)
        return fit + self.beta * (y_hat - anc).pow(2).mean()


def run_loop(data, beta, regime, n_rounds=15, epoch_size=30, n_labeled=1730, seed=0):
    torch.manual_seed(seed)
    n, d = data["features"].shape
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:n_labeled] = True
    feats = data["features"]
    innate = data["innate"]
    anchor = innate[mask].clone()

    world = FJWorld(innate=innate, graph=data["graph"], peer_sus=data["peer_sus"],
                    platform_sus=data["platform_sus"], features=feats)
    world.reset(seed=seed)
    model = MLPModel(in_features=d, hidden_dims=[16, 16], out_features=1,
                     activation="relu", final_activation="sigmoid", init_seed=seed)
    loss = AnchoredMSELoss(beta, anchor)

    def train(train_x, train_y):
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        dd = {"x": train_x, "y": train_y}
        for _ in range(300):
            opt.zero_grad()
            loss(model, dd).backward()
            opt.step()

    xs = [feats[mask]]
    ys = [innate[mask].unsqueeze(-1)]
    rows = []
    for t in range(n_rounds):
        if regime == "replace":
            train(xs[-1], ys[-1])
        else:  # accumulate
            train(torch.cat(xs, 0), torch.cat(ys, 0))
        world.run(model, n_steps=epoch_size)
        op = world.state["opinion"]
        with torch.no_grad():
            pred = model(feats).squeeze(-1)
        rows.append({
            "op_std": float(op.std()),
            "op_range": (float(op.min()), float(op.max())),
            "tail": float((op - 0.5).abs().gt(0.15).float().mean()),
            "pred_std": float(pred.std()),
        })
        xs.append(feats[mask])
        ys.append(op[mask].detach().unsqueeze(-1))
    return rows


def main():
    data = load_pokec()
    innate = data["innate"]
    base_tail = float((innate - 0.5).abs().gt(0.15).float().mean())
    print(f"innate: std={innate.std():.4f} tail_frac={base_tail:.3f} "
          f"range=[{innate.min():.3f}, {innate.max():.3f}]\n")

    betas = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
    for regime in ("replace", "accumulate"):
        print(f"=== regime: {regime} ===")
        print("  beta   pred_std  op_std  tail_frac  op_range  (final round)")
        for b in betas:
            rows = run_loop(data, b, regime)
            r = rows[-1]
            print(f"  {b:>5}   {r['pred_std']:.4f}   {r['op_std']:.4f}  "
                  f"{r['tail']:.3f}     [{r['op_range'][0]:.3f}, {r['op_range'][1]:.3f}]")
        print()


if __name__ == "__main__":
    main()
