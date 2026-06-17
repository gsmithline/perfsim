"""Pokec loaders and the fixed-prediction model, shared by the fj experiments."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.environments.dynamics.fj import normalize_adjacency

POKEC = Path("examples/pokec")


class FixedPredictions(Model):
    def __init__(self, values: Tensor) -> None:
        super().__init__()
        self.values = values

    def forward(self, x: Tensor) -> Tensor:
        return self.values


def load_pokec() -> dict:
    with open(POKEC / "lcc_profiles_relation_to_smoking.pk", "rb") as fh:
        df = pickle.load(fh)
    with open(POKEC / "lcc_graph_relation_to_smoking.pk", "rb") as fh:
        graph = pickle.load(fh)
    pp = POKEC / "parametric_params"
    with open(pp / "y_label2163.pk", "rb") as fh:
        y_lab = pickle.load(fh)
    with open(pp / "y_unlabel_label2163.pk", "rb") as fh:
        y_unlab = pickle.load(fh)
    with open(pp / "hetero_peer_sus2163.pkl", "rb") as fh:
        peer_sus = pickle.load(fh)
    with open(pp / "hetero_platform_sus2163.pkl", "rb") as fh:
        platform_sus = pickle.load(fh)
    innate = torch.tensor(
        np.asarray(list(y_lab) + list(y_unlab), dtype=np.float64), dtype=torch.float32
    )
    adj = nx.to_numpy_array(graph, nodelist=df["user_id"].tolist())
    adj = torch.tensor(adj, dtype=torch.float32)
    return {
        "innate": innate,
        "adj": adj,
        "W": normalize_adjacency(adj),
        "peer_sus": torch.tensor(np.asarray(peer_sus), dtype=torch.float32),
        "platform_sus": torch.tensor(np.asarray(platform_sus), dtype=torch.float32),
    }
