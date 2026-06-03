"""Does an MLP-driven FJ loop homogenize the population?

FJWorld on the real Pokec graph, a small MLP retrained each round on the
settled opinions. Track the population opinion spread per round. If the
performative loop homogenizes, std shrinks and the range collapses.
"""

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from perfsim.environments.dynamics import FJWorld, normalize_adjacency
from perfsim.learners import GradientLearner
from perfsim.losses import MSELoss
from perfsim.models import MLPModel
from perfsim.simulator import Simulator

POKEC_DIR = Path("examples/pokec")


def load_pokec():
    with open(POKEC_DIR / "lcc_graph_relation_to_smoking.pk", "rb") as fh:
        graph = pickle.load(fh)
    with open(POKEC_DIR / "labeled_feature_matrix_relation_to_smoking_False.pk", "rb") as fh:
        X_lab = pickle.load(fh)
    with open(POKEC_DIR / "unlabeled_feature_matrix_relation_to_smoking_False.pk", "rb") as fh:
        X_unlab = pickle.load(fh)
    features = np.concatenate([X_lab, X_unlab], axis=0)
    with open(POKEC_DIR / "parametric_params/y_label2163.pk", "rb") as fh:
        y_lab = pickle.load(fh)
    with open(POKEC_DIR / "parametric_params/y_unlabel_label2163.pk", "rb") as fh:
        y_unlab = pickle.load(fh)
    innate = np.array(list(y_lab) + list(y_unlab), dtype=np.float64)
    with open(POKEC_DIR / "parametric_params/hetero_peer_sus2163.pkl", "rb") as fh:
        peer_sus = pickle.load(fh)
    with open(POKEC_DIR / "parametric_params/hetero_platform_sus2163.pkl", "rb") as fh:
        platform_sus = pickle.load(fh)
    adj = nx.to_numpy_array(graph, nodelist=list(graph.nodes))
    W = normalize_adjacency(torch.tensor(adj, dtype=torch.float32))
    return {
        "graph": W,
        "features": torch.tensor(features, dtype=torch.float32),
        "innate": torch.tensor(innate, dtype=torch.float32),
        "peer_sus": torch.tensor(peer_sus, dtype=torch.float32),
        "platform_sus": torch.tensor(platform_sus, dtype=torch.float32),
    }


def main():
    torch.manual_seed(0)
    data = load_pokec()
    n, d = data["features"].shape
    world = FJWorld(
        innate=data["innate"],
        graph=data["graph"],
        peer_sus=data["peer_sus"],
        platform_sus=data["platform_sus"],
        features=data["features"],
    )
    model = MLPModel(
        in_features=d,
        hidden_dims=[16, 16],
        out_features=1,
        activation="relu",
        final_activation="sigmoid",
        init_seed=0,
    )
    loss = MSELoss()
    learner = GradientLearner(model, loss, lr=0.05, steps_per_round=300, optimizer="adam")

    n_labeled = 1730
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:n_labeled] = True

    metrics = {
        "op_std": lambda s: float(s.env.state["opinion"].std()),
        "op_mean": lambda s: float(s.env.state["opinion"].mean()),
        "op_min": lambda s: float(s.env.state["opinion"].min()),
        "op_max": lambda s: float(s.env.state["opinion"].max()),
        "pred_std": lambda s: float(s.predictor.model(data["features"]).std()),
    }
    sim = Simulator(world=world, learner=learner, loss=loss, metrics=metrics)

    innate = data["innate"]
    print(f"innate:   mean={innate.mean():.4f}  std={innate.std():.4f}  "
          f"range=[{innate.min():.3f}, {innate.max():.3f}]")

    initial_data = {"x": data["features"], "y": innate.unsqueeze(-1)}
    history = sim.run(
        n_rounds=15,
        epoch_size=30,
        seed=0,
        initial_data=initial_data,
        train_mask=train_mask,
    )

    print("\nround  op_std  op_mean  op_range            pred_std")
    for t, rec in enumerate(history.records):
        print(f"{t:>4}  {rec['op_std']:.4f}  {rec['op_mean']:.4f}  "
              f"[{rec['op_min']:.3f}, {rec['op_max']:.3f}]   {rec['pred_std']:.4f}")


if __name__ == "__main__":
    main()
