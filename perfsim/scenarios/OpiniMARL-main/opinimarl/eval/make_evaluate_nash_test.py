from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from opinimarl.environments import NetworkBinary
from opinimarl.graphs import make_graph
from opinimarl.models import ActorCriticRNNAttention, ScannedRNN
from opinimarl.utils import (actions_batched_sym_flip, batchify_arr,
                             batchify_dict, obs_batched_sym_flip,
                             reset_symmetry_flip, unbatchify_arr,
                             unbatchify_dict)
from opinimarl.wrappers import NetworkBinaryLogWrapper


# Class to store metrics
class Metrics(NamedTuple):
    rewards_0: jnp.array
    rewards: jnp.array

# Make evaluate function
def make_evaluate_nash_test(cfg):
    NUM_ACTORS = cfg.env.NUM_AGENTS * cfg.eval.NUM_ENVS
    NUM_LOOPS = cfg.eval.NUM_EPISODES // cfg.eval.NUM_ENVS
    
    selected_agents = jnp.array(cfg.train.local_nash.SELECTED_AGENTS)
    rest_agents = jnp.setdiff1d(jnp.arange(cfg.env.NUM_AGENTS), selected_agents)

    graph_adj, graph_aux = make_graph(cfg.env.GRAPH, cfg.env.NUM_AGENTS, **cfg.env.GRAPH_KWARGS)

    def evaluate(rng, frozen_params, params, alpha):
        rng, _rng = jax.random.split(rng)

        # INIT ENV
        env = NetworkBinary(
            graph_adj=graph_adj,
            alpha=alpha,
            q=graph_aux.get("q"),
            r=cfg.env.GRAPH_KWARGS.get("r"),
            **cfg.env.ENV_KWARGS,
        )
        env = NetworkBinaryLogWrapper(env)

        # INIT NETWORK
        network = ActorCriticRNNAttention(
            action_dim=env.action_space(env.agents[0]).n,
            num_agents=env.num_agents,
            agent_id_num_bits=cfg.env.AGENT_ID_NUM_BITS,
            **cfg.NN_KWARGS,
        )
        init_hstate = ScannedRNN.initialize_carry(cfg.eval.NUM_ENVS, cfg.nn.GRU_HIDDEN_SIZE)
        init_frozen_hstate = ScannedRNN.initialize_carry(NUM_ACTORS - cfg.eval.NUM_ENVS, cfg.nn.GRU_HIDDEN_SIZE)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, cfg.eval.NUM_ENVS)
        prob_of_signal = jax.lax.select(
            jnp.isclose(alpha, 0.0),
            0.0,
            cfg.env.ENV_KWARGS.p_signal,
        )
        reset_params = (prob_of_signal, 1.0)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, reset_params)
        dones = {a: jnp.zeros(cfg.eval.NUM_ENVS, dtype=bool) for a in env.agents + ["__all__"]}

        rng, _rng = jax.random.split(rng)
        symmetry_rng = jax.random.split(_rng, cfg.eval.NUM_ENVS)
        symmetry_flips = reset_symmetry_flip(symmetry_rng, env.agents, cfg.eval.OTHER_PLAY)

        def _env_step(runner_state, unused):
            env_state, last_obs, last_dones, symmetry_flips, frozen_hstate, hstate, timestep, rng = runner_state

            interaction_graph = env_state.env_state.interaction_network

            # SPECIFY SYMMETRY FLIPPING FOR "OTHER PLAY"
            rng, _rng = jax.random.split(rng)
            rng_sym = jax.random.split(_rng, cfg.eval.NUM_ENVS)
            symmetry_flips = jax.tree.map(
                lambda x, y: jnp.where(last_dones["__all__"], x, y),
                reset_symmetry_flip(rng_sym, env.agents, cfg.eval.OTHER_PLAY),
                symmetry_flips,
            )
            symmetry_flips_batch = batchify_arr(symmetry_flips, env.agents)

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            last_obs_batch = batchify_dict(last_obs, env.agents)
            last_obs_batch_sym_flip = obs_batched_sym_flip(last_obs_batch, symmetry_flips_batch)
            last_dones_batch = batchify_arr(last_dones, env.agents)
            
            # SEPARATE OUT SINGLE AGENT
            last_obs_batch_sym_flip_reshaped = jax.tree.map(
                lambda leaf: leaf.reshape(env.num_agents, cfg.train.NUM_ENVS, -1).squeeze(),
                last_obs_batch_sym_flip,
            )
            last_obs_batch_sym_flip_0 = jax.tree.map(
                lambda leaf: leaf[selected_agents].reshape(selected_agents.size * cfg.train.NUM_ENVS, -1).squeeze(),
                last_obs_batch_sym_flip_reshaped,
            )
            last_obs_batch_sym_flip_rest = jax.tree.map(
                lambda leaf: leaf[rest_agents].reshape(rest_agents.size * cfg.train.NUM_ENVS, -1).squeeze(),
                last_obs_batch_sym_flip_reshaped
            )

            last_dones_batch_reshaped = last_dones_batch.reshape(env.num_agents, cfg.eval.NUM_ENVS)
            last_dones_batch_0 = last_dones_batch_reshaped[selected_agents].reshape(selected_agents.size * cfg.eval.NUM_ENVS)
            last_dones_batch_rest = last_dones_batch_reshaped[rest_agents].reshape(rest_agents.size * cfg.eval.NUM_ENVS)

            ac_in_0 = (
                jax.tree.map(lambda x: x[np.newaxis, :], last_obs_batch_sym_flip_0),
                last_dones_batch_0[np.newaxis, :],
            )
            ac_in_rest = (
                jax.tree.map(lambda x: x[np.newaxis, :], last_obs_batch_sym_flip_rest),
                last_dones_batch_rest[np.newaxis, :],
            )

            hstate, pi, values, beliefs_0 = network.apply(params, hstate, ac_in_0)
            actions_0 = pi.sample(seed=_rng)
            log_probs = pi.log_prob(actions_0)
            values, actions_0, beliefs_0, log_probs = (
                values.squeeze(0),
                actions_0.squeeze(0),
                beliefs_0.squeeze(0),
                log_probs.squeeze(0),
            )
            frozen_hstate, frozen_pi, _, beliefs_rest = network.apply(frozen_params, frozen_hstate, ac_in_rest)
            actions_rest = frozen_pi.sample(seed=_rng)
            actions_rest = actions_rest.squeeze(0)
            beliefs_rest = beliefs_rest.squeeze(0)

            actions_0 = actions_0.reshape(selected_agents.size, cfg.eval.NUM_ENVS)
            actions_rest = actions_rest.reshape(rest_agents.size, cfg.eval.NUM_ENVS)

            actions = jnp.zeros((env.num_agents, cfg.eval.NUM_ENVS), dtype=actions_0.dtype)
            actions = actions.at[selected_agents].set(actions_0).at[rest_agents].set(actions_rest)
            actions = actions.reshape(env.num_agents * cfg.eval.NUM_ENVS)

            actions_sym_flip = actions_batched_sym_flip(actions, symmetry_flips_batch)
            env_act = unbatchify_arr(actions_sym_flip, env.agents)
            env_act = {k: v.squeeze() for k, v in env_act.items()}

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, cfg.eval.NUM_ENVS)
            obs, env_state, rewards, dones, infos = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, env_act, reset_params)
            
            # rewards_batch = batchify(rewards, env.agents, NUM_ACTORS).squeeze()
            # rewards_batch_reshape = rewards_batch.reshape((env.num_agents, -1))
            # accuracy = rewards_batch_reshape.mean(axis=0)

            accuracy = infos["accuracy"].mean()
            entropy = infos["entropy"].mean()

            beliefs = jnp.concatenate([beliefs_0, beliefs_rest])
            beliefs_sym_flip = jnp.where(
                symmetry_flips_batch,
                1 - beliefs,
                beliefs,
            ).reshape(cfg.eval.NUM_ENVS, env.num_agents, order="F")

            lying = jnp.where(
                ((beliefs - 0.5) * (actions - 0.5) < -1e-6) & (actions < 2.0 - 1e-6),
                jnp.ones(beliefs.shape),
                jnp.zeros(beliefs.shape),
            ).reshape(cfg.eval.NUM_ENVS, env.num_agents, order="F")

            outputs = actions_sym_flip.reshape(cfg.eval.NUM_ENVS, env.num_agents, order="F")

            private_signals = last_obs_batch["private_signal"].reshape(cfg.eval.NUM_ENVS, env.num_agents, order="F")

            ground_truth = env_state.env_state.truth_value

            rewards_batch = batchify_arr(rewards, env.agents).reshape(cfg.eval.NUM_ENVS, env.num_agents, order="F")
            rewards_batch_0 = rewards_batch[:, selected_agents]

            metrics = Metrics(
                rewards_batch_0,
                rewards_batch,
            )
            timestep = jax.lax.select(
                timestep + 1 ==cfg.env.MAX_STEPS,
                0,
                timestep + 1,
            )

            runner_state = (env_state, obs, dones, symmetry_flips, frozen_hstate, hstate, timestep, rng)
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (env_state, obs, dones, symmetry_flips, init_frozen_hstate, init_hstate, 0, _rng)

        runner_state, metrics = jax.lax.scan(
            _env_step, runner_state, None, cfg.env.MAX_STEPS * NUM_LOOPS
        )
        metrics = Metrics(
            metrics.rewards_0.reshape([cfg.env.MAX_STEPS, NUM_LOOPS * cfg.eval.NUM_ENVS] + list(metrics.rewards_0.shape[2:]), order="F"),
            metrics.rewards.reshape([cfg.env.MAX_STEPS, NUM_LOOPS * cfg.eval.NUM_ENVS] + list(metrics.rewards.shape[2:]), order="F"),
        )

        return runner_state, metrics
    
    return evaluate
