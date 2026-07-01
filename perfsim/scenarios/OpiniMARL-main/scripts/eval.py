import pickle as pkl
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from opinimarl.eval import make_evaluate, make_evaluate_nash_test


@hydra.main(version_base=None, config_path="../config/", config_name="config")
def main(cfg: DictConfig) -> None:

    alphas = jnp.array(cfg.env.ALPHAS)

    # If list of seeds is specified this is taken prioritised
    if cfg.train.seed.SEEDS:
        seeds = jnp.array(cfg.train.seed.SEEDS, dtype=jnp.int32)
    else:
        seeds = jnp.arange(
            cfg.train.seed.STARTING_SEED,
            cfg.train.seed.STARTING_SEED + cfg.train.seed.NUM_SEEDS,
            dtype=jnp.int32
        )


    for seed in seeds:
        print(f"Starting run for seed {seed}")

        rng = jax.random.PRNGKey(cfg.eval.seed.EVAL_SEED)

        cfg.NN_KWARGS.use_centralised_logic = False

        base_dir = Path(__file__).resolve().parent.parent if cfg.path.BASE_DIR == "" else Path(cfg.path.BASE_DIR)
        algorithm = f"{cfg.train.ALGORITHM}_other_play" if cfg.train.OTHER_PLAY else f"{cfg.train.ALGORITHM}"
        eval_op = "eval_op_true" if cfg.eval.OTHER_PLAY else "eval_op_false"
        nash_test = f"_nash_test_{'_'.join(str(x) for x in cfg.train.local_nash.SELECTED_AGENTS)}" if cfg.train.local_nash.NASH_TEST else ""
        graph_kwargs_path = Path(*[f"{k}_{v:.4g}" for k, v in cfg.env.GRAPH_KWARGS.items()])


        if cfg.train.local_nash.NASH_TEST:
            frozen_params_list = []
            params_list = []
            for alpha in alphas:
                params_path = base_dir / "output_data" / "train" / algorithm / f"graph_{cfg.env.GRAPH}" / graph_kwargs_path / f"num_agents_{cfg.env.NUM_AGENTS}" / f"alpha_{alpha:.2f}" / f"seed_{seed}" 
                
                with (params_path / f"params.pkl").open("rb") as f:
                    frozen_params = pkl.load(f)
                with (params_path / f"params{nash_test}.pkl").open("rb") as f:
                    params = pkl.load(f)
                
                frozen_params_list.append(frozen_params)
                params_list.append(params)

            frozen_params_batched = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *frozen_params_list)
            params_batched = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *params_list)

            params_tuple = (frozen_params_batched, params_batched)

            evaluate_jit = jax.jit(
                make_evaluate_nash_test(cfg),
            )

        else:
            params_list = []
            for alpha in alphas:
                params_path = base_dir / "output_data" / "train" / algorithm / f"graph_{cfg.env.GRAPH}" / graph_kwargs_path / f"num_agents_{cfg.env.NUM_AGENTS}" / f"alpha_{alpha:.2f}" / f"seed_{seed}" 
                with (params_path / "params.pkl").open("rb") as f:
                    params = pkl.load(f)
                
                params_list.append(params)

            params_batched = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *params_list)
            params_tuple = (params_batched, )

            evaluate_jit = jax.jit(
                make_evaluate(cfg),
            )

        @jax.jit
        def _scan_alpha(carry, x):
            rng = carry
            out = evaluate_jit(rng, *x)

            return rng, out

        _, out = jax.lax.scan(
            _scan_alpha,
            rng,
            (*params_tuple, alphas),
        )
        _, metrics_tuple = out

        metrics = {
            "rewards": metrics_tuple.rewards,
            # "aux_rewards": metrics_tuple.aux_rewards,
            # "accuracy": metrics_tuple.accuracy,
            # "entropy": metrics_tuple.entropy,
            # "beliefs": metrics_tuple.beliefs,
            "lying": metrics_tuple.lying,
            "outputs": metrics_tuple.actions,
            "private_signals": metrics_tuple.private_signals,
            "ground_truth": metrics_tuple.ground_truth,
            "interaction_graph": metrics_tuple.interaction_graph,
            "graph": metrics_tuple.graph,
            "attn_weights": metrics_tuple.attn_weights,
            # "timestep": metrics_tuple.timestep,
        }

        for i, alpha in enumerate(alphas):
            out_path = base_dir / "output_data" / "eval" / algorithm / eval_op / f"graph_{cfg.env.GRAPH}" / graph_kwargs_path / f"num_agents_{cfg.env.NUM_AGENTS}" / f"alpha_{alpha:.2f}" / f"seed_{seed}"
            out_path.mkdir(parents=True, exist_ok=True)

            alpha_metrics = jax.tree.map(lambda x: x[i], metrics)

            with (out_path / f"metrics{nash_test}.pkl").open("wb") as f:
                pkl.dump(alpha_metrics, f)

    return None


if __name__ == "__main__":
    main()
