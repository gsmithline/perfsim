import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def _():
    """Pokec FJ smoke test: drive perfsim's Simulator with the real Pokec graph
    (LCC restricted to the relation-to-smoking labeled subgraph; 2163 nodes,
    2346 edges) and the heterogeneous peer / platform susceptibility vectors
    shipped with the `examples/pokec/` snapshot.

    Plumbing test, not a numerical replication of
    `Opinion-dynamics-post-training/pokec_simulations.py`. One simplification
    remaining vs that script:

    - No standardization or feature engineering. The 4-column feature matrix
      is fed straight to a LinearModel.

    The labeled/unlabeled split IS exercised here via `Simulator.run(...,
    train_mask=...)`: the first 1730 agents (labeled) contribute opinions as
    training targets each round; the remaining 433 agents (unlabeled)
    participate in FJ peer dynamics but their opinions are never used as
    training labels. Matches pokec's `x_labeled_prior` convention.

    """
    return


@app.cell
def _():
    import pickle
    from pathlib import Path

    import argparse
    import networkx as nx
    import numpy as np
    import torch

    from perfsim.environments.dynamics import FJWorld, normalize_adjacency
    from perfsim.learners import ERMLearner, GradientLearner
    from perfsim.losses import MSELoss
    from perfsim.models import LinearModel, LogisticModel, MLPModel
    from perfsim.simulator import Simulator

    return (
        ERMLearner,
        FJWorld,
        GradientLearner,
        LinearModel,
        LogisticModel,
        MLPModel,
        MSELoss,
        Path,
        Simulator,
        normalize_adjacency,
        np,
        nx,
        pickle,
        torch,
    )


@app.cell
def _(Path, np, nx, pickle, torch):
    POKEC_DIR = Path("examples/pokec")


    def load_pokec() -> dict[str, torch.Tensor | nx.Graph]:
        """Load the pieces we need for an FJ smoke run."""
        with open(POKEC_DIR / "lcc_graph_relation_to_smoking.pk", "rb") as fh:
            graph: nx.Graph = pickle.load(fh)

        with open(POKEC_DIR / "labeled_feature_matrix_relation_to_smoking_False.pk", "rb") as fh:
            X_lab = pickle.load(fh)
        with open(POKEC_DIR / "unlabeled_feature_matrix_relation_to_smoking_False.pk", "rb") as fh:
            X_unlab = pickle.load(fh)
        features = np.concatenate([X_lab, X_unlab], axis=0)  # (2163, 4)

        with open(POKEC_DIR / "parametric_params/y_label2163.pk", "rb") as fh:
            y_lab = pickle.load(fh)
        with open(POKEC_DIR / "parametric_params/y_unlabel_label2163.pk", "rb") as fh:
            y_unlab = pickle.load(fh)
        innate = np.array(list(y_lab) + list(y_unlab), dtype=np.float64)  # (2163,)

        with open(POKEC_DIR / "parametric_params/hetero_peer_sus2163.pkl", "rb") as fh:
            peer_sus = pickle.load(fh)
        with open(POKEC_DIR / "parametric_params/hetero_platform_sus2163.pkl", "rb") as fh:
            platform_sus = pickle.load(fh)

        return {
            "graph": graph,
            "features": torch.tensor(features, dtype=torch.float32),
            "innate": torch.tensor(innate, dtype=torch.float32),
            "peer_sus": torch.tensor(peer_sus, dtype=torch.float32),
            "platform_sus": torch.tensor(platform_sus, dtype=torch.float32),
        }

    return (load_pokec,)


@app.cell
def _(FJWorld, normalize_adjacency, nx, torch):
    def build_world(data: dict) -> FJWorld:
        """Build FJWorld from pokec data.

        Caveat: graph node order vs feature/opinion order alignment requires
        the `lcc_profiles` DataFrame's user_id column, which fails to unpickle
        under our pandas version. We fall back to `list(graph.nodes)` as the
        nodelist; this misaligns agents with features in the topological sense,
        but the dynamics still run and FJ still converges (just to a graph
        permutation of the canonical equilibrium). Sufficient for a plumbing
        smoke test.
        """
        graph: nx.Graph = data["graph"]
        nodelist = list(graph.nodes)
        adj = nx.to_numpy_array(graph, nodelist=nodelist)
        W = normalize_adjacency(torch.tensor(adj, dtype=torch.float32))
        return FJWorld(
            innate=data["innate"],
            graph=W,
            peer_sus=data["peer_sus"],
            platform_sus=data["platform_sus"],
            features=data["features"],
        )

    return (build_world,)


@app.cell
def _(
    ERMLearner,
    GradientLearner,
    LinearModel,
    LogisticModel,
    MLPModel,
    MSELoss,
):
    def build_predictor(kind: str, d: int) -> tuple:
        """Return (model, loss, learner) for the requested predictor kind."""
        loss = MSELoss()
        if kind == "linear":
            model = LinearModel(in_features=d, out_features=1, bias=True)
            learner = ERMLearner(model, loss, max_iter=50)
        elif kind == "logistic":
            model = LogisticModel(in_features=d, out_features=1, bias=True)
            learner = ERMLearner(model, loss, max_iter=100) 
        elif kind == "mlp":
            model = MLPModel(
                in_features=d,
                hidden_dims=[16, 16],
                out_features=1,
                activation="relu",
                final_activation="sigmoid",
                init_seed=0,
            )
            learner = GradientLearner(model, loss, lr=0.05, steps_per_round=300)
        else:
            raise ValueError(f"unknown model kind {kind!r}")
        return model, loss, learner

    return (build_predictor,)


@app.cell
def _(load_pokec):
    data = load_pokec()
    model = "linear" #can be any of 3 base models 
    print(f"Loading pokec data (predictor={model})...")
    data = load_pokec()
    print( f" graph: {data['graph'].number_of_nodes()} nodes, "
        f"{data['graph'].number_of_edges()} edges"
    )
    return data, model


@app.cell
def _(data):
    print(f"features: {tuple(data['features'].shape)}")
    print(f" innate: range [{data['innate'].min():.3f}, {data['innate'].max():.3f}]")
    print(f" peer_sus: range [{data['peer_sus'].min():.3f}, "
        f"{data['peer_sus'].max():.3f}], mean={data['peer_sus'].mean():.3f}"
    )
    print(f" platform_sus: range [{data['platform_sus'].min():.3f}, "
        f"{data['platform_sus'].max():.3f}], mean={data['platform_sus'].mean():.3f}"
    )
    return


@app.cell
def _(Simulator, build_predictor, build_world, data, model, torch):
    world = build_world(data)
    n, d = data['features'].shape
    n_labeled = 1730
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:n_labeled] = True  #last 433 unlabeled following oriignal implementation
    model_1, loss, learner = build_predictor(model, d=d)
    sim = Simulator(world=world, learner=learner, loss=loss)
    return n, sim, train_mask, world


@app.cell
def _(data, sim, train_mask):
    n_rounds = 5
    epoch_size = 30

    print( f"\nRunning: n_rounds={n_rounds}, epoch_size={epoch_size} "
        f"(so {epoch_size} FJ inner ticks per outer round, one K-agent "
        f"query per round)."
    )
    print(
        f"train_mask: {int(train_mask.sum())} labeled / "
        f"{int((~train_mask).sum())} unlabeled (predictor trains on "
        f"labeled only)."
    )

    initial_data = {
        "x": data["features"],
        "y": data["innate"].unsqueeze(-1),
    }

    print("\nPer-round opinion summary (after train + deploy + FJ rollout):")
    print(
        f" init: opinion mean={data['innate'].mean():.4f}  "
        f"std={data['innate'].std():.4f}"
    )

    history = sim.run(
        n_rounds=n_rounds,
        epoch_size=epoch_size,
        seed=0,
        initial_data=initial_data,
        train_mask=train_mask,
    )
    return (history,)


@app.cell
def _(data, history, n, train_mask, world):
    final_opinions = world.state["opinion"]
    for t, rec in enumerate(history.records):
        print(f"round {t}: |theta|={rec['theta'].norm():.4f}" #deployed predictor for this round
            f"stability_gap={rec.get('stability_gap', float('nan')):.6f}"
        )
        print( f"final: opinion mean={final_opinions.mean():.4f}  "
            f"std={final_opinions.std():.4f}  "
            f"range=[{final_opinions.min():.3f}, {final_opinions.max():.3f}]"
        )

        lab_idx = train_mask
        unl_idx = ~train_mask
        print(
            f"  labeled (N={int(lab_idx.sum())}):   "
            f"mean={final_opinions[lab_idx].mean():.4f}  "
            f"std={final_opinions[lab_idx].std():.4f}"
        )
        print(
            f"  unlabeled (N={int(unl_idx.sum())}): "
            f"mean={final_opinions[unl_idx].mean():.4f}  "
            f"std={final_opinions[unl_idx].std():.4f}"
        )

        drift = (final_opinions - data["innate"]).norm() / (n ** 0.5)
        print(f" RMS drift from innate: {drift:.4f}")
    return


if __name__ == "__main__":
    app.run()
