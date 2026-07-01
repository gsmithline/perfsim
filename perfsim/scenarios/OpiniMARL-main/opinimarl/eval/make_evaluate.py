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
    rewards: jnp.array
    # aux_rewards: jnp.array
    # accuracy: jnp.array
    # entropy: jnp.array
    # beliefs: jnp.array
    lying: jnp.array
    actions: jnp.array
    private_signals: jnp.array
    ground_truth: jnp.array
    interaction_graph: jnp.array
    graph: jnp.array
    attn_weights: jnp.array
    # timestep: int

# Make evaluate function
def make_evaluate(cfg):
    NUM_ACTORS = cfg.env.NUM_AGENTS * cfg.eval.NUM_ENVS
    NUM_LOOPS = cfg.eval.NUM_EPISODES // cfg.eval.NUM_ENVS
    TOTAL_STEPS = cfg.env.MAX_STEPS * NUM_LOOPS

    graph_adj, graph_aux = make_graph(cfg.env.GRAPH, cfg.env.NUM_AGENTS, **cfg.env.GRAPH_KWARGS)
    graph_with_self_loops = graph_adj + jnp.eye(cfg.env.NUM_AGENTS, dtype=jnp.int32)
    receivers, senders = jnp.nonzero(graph_with_self_loops)
    receivers = receivers.astype(jnp.int32)
    senders = senders.astype(jnp.int32)

    def evaluate(rng, params, alpha):
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
            senders=senders,
            receivers=receivers,
        )
        init_hstate = ScannedRNN.initialize_carry(NUM_ACTORS, cfg.NN_KWARGS.gru_hidden_size)

        # INIT ENV
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

        avg_attn_weights = jnp.zeros_like(graph_adj, dtype=jnp.float32)

        def _env_step(runner_state, unused):
            env_state, last_obs, last_dones, symmetry_flips, hstate, episode_timestep, global_timestep, avg_attn_weights, rng = runner_state

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
            
            ac_in = (
                jax.tree.map(lambda x: x[np.newaxis, :], last_obs_batch_sym_flip),
                last_dones_batch[np.newaxis, :],
            )
            symmetry_flips_in = symmetry_flips_batch[np.newaxis, :]

            (hstate, pi, values, beliefs), attn_weights_batch = network.apply(params, hstate, ac_in, symmetry_flips_in, sow_attention=True)
            actions = pi.sample(seed=_rng)
            log_probs = pi.log_prob(actions)
            values, actions, beliefs, log_probs = (
                values.squeeze(0),
                actions.squeeze(0),
                beliefs.squeeze(0),
                log_probs.squeeze(0),
            )
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

            accuracies = infos["accuracy"].mean()
            entropy = infos["entropy"].mean()

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
            aux_rewards_batch = batchify_arr(infos["aux_rewards"], env.agents).reshape(cfg.eval.NUM_ENVS, env.num_agents, order="F")

            metrics = Metrics(
                rewards_batch,
                # aux_rewards_batch,
                # accuracy,
                # entropy,
                # beliefs_sym_flip,
                lying,
                outputs,
                private_signals,
                ground_truth,
                interaction_graph, 
                None,
                None,
                # episode_timestep,
            )
            episode_timestep = jax.lax.select(
                episode_timestep + 1 ==cfg.env.MAX_STEPS,
                0,
                episode_timestep + 1,
            )

            attn_weights = attn_weights_batch.squeeze().reshape(env.num_agents, cfg.eval.NUM_ENVS, -1).mean(axis=1)
            avg_attn_weights = (global_timestep * avg_attn_weights + attn_weights) / (global_timestep + 1)

            global_timestep += 1

            runner_state = (env_state, obs, dones, symmetry_flips, hstate, episode_timestep, global_timestep, avg_attn_weights, rng)
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (env_state, obs, dones, symmetry_flips, init_hstate, 0, 0, avg_attn_weights, _rng)

        runner_state, metrics = jax.lax.scan(
            _env_step, runner_state, None, TOTAL_STEPS
        )
        # metrics = jax.tree.map(lambda x: x.reshape([cfg.env.MAX_STEPS, NUM_LOOPS * cfg.eval.NUM_ENVS] + list(x.shape[2:]), order="F"), metrics)
        metrics_interaction_graph = metrics.interaction_graph.reshape([cfg.env.MAX_STEPS, NUM_LOOPS * cfg.eval.NUM_ENVS] + list(metrics.interaction_graph.shape[2:]), order="F") if env.q != None else None
        metrics = Metrics(
            metrics.rewards.reshape([cfg.env.MAX_STEPS, NUM_LOOPS * cfg.eval.NUM_ENVS] + list(metrics.rewards.shape[2:]), order="F"),
            # metrics.aux_rewards.reshape([cfg.env.MAX_STEPS, NUM_LOOPS * cfg.eval.NUM_ENVS] + list(metrics.aux_rewards.shape[2:]), order="F"),
            # metrics.accuracy.reshape(cfg.env.MAX_STEPS, -1, order="F").mean(axis=-1),
            # metrics.entropy.reshape(cfg.env.MAX_STEPS, -1, order="F").mean(axis=-1),
            # metrics.belief_accuracy.reshape(cfg.env.MAX_STEPS, -1, order="F").mean(axis=-1),
            # metrics.beliefs.reshape([cfg.env.MAX_STEPS, NUM_LOOPS * cfg.eval.NUM_ENVS] + list(metrics.beliefs.shape[2:]), order="F"),
            metrics.lying.reshape([cfg.env.MAX_STEPS, NUM_LOOPS * cfg.eval.NUM_ENVS] + list(metrics.lying.shape[2:]), order="F"),
            metrics.actions.reshape([cfg.env.MAX_STEPS, NUM_LOOPS * cfg.eval.NUM_ENVS] + list(metrics.actions.shape[2:]), order="F"),
            metrics.private_signals.reshape([cfg.env.MAX_STEPS, NUM_LOOPS * cfg.eval.NUM_ENVS] + list(metrics.private_signals.shape[2:]), order="F"),
            metrics.ground_truth.reshape([cfg.env.MAX_STEPS, NUM_LOOPS * cfg.eval.NUM_ENVS] + list(metrics.ground_truth.shape[2:]), order="F"),
            metrics_interaction_graph,
            runner_state[0].env_state.network[0],
            runner_state[7]
            # metrics.timestep[:20],
        )

        return runner_state, metrics


    
    return evaluate
