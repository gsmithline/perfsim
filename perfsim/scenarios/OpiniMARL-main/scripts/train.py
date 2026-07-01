import pickle as pkl
import time
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import wandb
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf import DictConfig, OmegaConf

from opinimarl.train import make_train_ppo, make_train_ppo_nash_test


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:

    alphas = jnp.array(cfg.env.ALPHAS)

    # Pad alphas so shape fits with number of gpus for sharding
    padding = (-alphas.shape[0]) % cfg.resources.NUM_GPUS
    padded_alphas = jnp.pad(alphas, (0, padding))

    # Shard alphas
    if cfg.resources.NUM_GPUS > 1:
        devices = mesh_utils.create_device_mesh((jax.device_count(),))
        mesh = Mesh(devices, axis_names=("x"))
        sharding = NamedSharding(mesh, P("x"))
        sharded_alphas = jax.device_put(padded_alphas, sharding)
        print("Sharding alphas across the following GPUS:")
        jax.debug.visualize_array_sharding(sharded_alphas)
        print(f"padded_alphas: {alphas}")
    else:
        sharded_alphas = alphas
        jax.debug.visualize_array_sharding(sharded_alphas)
        print(f"padded_alphas: {alphas}")

    # Initialise timing variables
    start = 0
    end = 0


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

        rng = jax.random.PRNGKey(seed)
    
        base_dir = Path(__file__).resolve().parent.parent if cfg.path.BASE_DIR == "" else Path(cfg.path.BASE_DIR)
        algorithm = f"{cfg.train.ALGORITHM}_other_play" if cfg.train.OTHER_PLAY else f"{cfg.train.ALGORITHM}"
        nash_test = f"_nash_test_{'_'.join(str(x) for x in cfg.train.local_nash.SELECTED_AGENTS)}" if cfg.train.local_nash.NASH_TEST else ""
        graph_kwargs_path = Path(*[f"{k}_{v:.4g}" for k, v in cfg.env.GRAPH_KWARGS.items()])

        if cfg.train.TIME_IT:
            cfg.wandb.WANDB_MODE == "disabled"

        if cfg.wandb.WANDB_MODE == "online":
            wandb.init(
                entity=cfg.wandb.ENTITY,
                project=cfg.wandb.PROJECT,
                tags=[cfg.env.NAME.upper(), f"jax_{jax.__version__}"],
                name=f'{cfg.env.NAME}_{cfg.env.GRAPH}_n{cfg.env.NUM_AGENTS}_{algorithm}{nash_test}_seed{seed}',
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=cfg.wandb.WANDB_MODE,
            )

        if cfg.train.local_nash.NASH_TEST:
            cfg.NN_KWARGS.use_centralised_logic = False
            frozen_params_list = []
            for alpha in alphas:
                params_path = base_dir / "output_data" / "train" / algorithm / f"graph_{cfg.env.GRAPH}" / graph_kwargs_path / f"num_agents_{cfg.env.NUM_AGENTS}" / f"alpha_{alpha:.2f}" / f"seed_{seed}"
                with (params_path / "params.pkl").open("rb") as f:
                    params = pkl.load(f)
                frozen_params_list.append(params)
            frozen_params_batched = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *frozen_params_list)

            params_tuple = (frozen_params_batched, )

            if cfg.train.TIME_IT:
               start = time.time() 
            train_jit = jax.jit(
                jax.vmap(
                    make_train_ppo_nash_test(cfg),
                    in_axes=(None, 0, 0),
                ),
            )
        else:
            params_tuple = ()

            if cfg.train.TIME_IT:
               start = time.time() 
            train_jit = jax.jit(
                jax.vmap(
                    make_train_ppo(cfg),
                    in_axes=(None, 0),
                ),
            )

        out = train_jit(rng, *params_tuple, sharded_alphas)

        if cfg.train.TIME_IT:
            jax.block_until_ready(out)
            end = time.time()

        params, metrics = out


        for i, alpha in enumerate(alphas):
            out_path = base_dir / "output_data" / "train" / algorithm / f"graph_{cfg.env.GRAPH}" / graph_kwargs_path / f"num_agents_{cfg.env.NUM_AGENTS}" / f"alpha_{alpha:.2f}" / f"seed_{seed}"
            out_path.mkdir(parents=True, exist_ok=True)

            alpha_params = jax.tree.map(lambda x: x[i], params)
            alpha_metrics = jax.tree.map(lambda x: x[i], metrics)

            with (out_path / f"params{nash_test}.pkl").open("wb") as f:
                pkl.dump(alpha_params, f)
            with (out_path / f"metrics{nash_test}.pkl").open("wb") as f:
                pkl.dump(alpha_metrics, f)

            if cfg.train.TIME_IT:
                num_alphas = alphas.size
                with (out_path / f"training_time_alphas_{num_alphas}.txt").open("w") as f:
                    f.write(f" {end - start}\n")

        if cfg.wandb.WANDB_MODE == "online":
            wandb.finish()
        
    return None
    
    
if __name__ == "__main__":
    main()
