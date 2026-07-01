from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict

import opinimarl.utils.schedulars as sch
from opinimarl.environments import NetworkBinary
from opinimarl.graphs import make_graph
from opinimarl.models import ActorCriticRNNAttention, ScannedRNN
from opinimarl.utils import (actions_batched_sym_flip, batchify_arr,
                                batchify_dict, obs_batched_sym_flip,
                                reset_symmetry_flip, unbatchify_arr,
                                unbatchify_dict)
from opinimarl.wrappers import NetworkBinaryLogWrapper


# Class for storing transition information
class Transition(NamedTuple):
    global_dones: jnp.ndarray
    dones: jnp.ndarray
    actions: jnp.ndarray
    values: jnp.ndarray
    rewards: jnp.ndarray
    log_probs: jnp.ndarray
    obs: jnp.ndarray
    truth_values: jnp.ndarray
    infos: jnp.ndarray

# TrainState class
class CustomTrainState(TrainState):
    update_step: int

# Make train function
def make_train_ppo_nash_test(cfg):
    NUM_ACTORS = cfg.env.NUM_AGENTS * cfg.train.NUM_ENVS
    NUM_UPDATES = cfg.train.TOTAL_TIMESTEPS // cfg.train.TRAINING_INTERVAL // cfg.train.NUM_ENVS
    CLIP_EPS = (
        cfg.train.epsiilon.CLIP / cfg.env.NUM_AGENTS
        if cfg.train.epsilon.SCALE_CLIP
        else cfg.train.epsilon.CLIP
    )

    selected_agents = jnp.array(cfg.train.local_nash.SELECTED_AGENTS)
    rest_agents = jnp.setdiff1d(jnp.arange(cfg.env.NUM_AGENTS), selected_agents)

    lr_schedule = partial(
        sch.linear_schedule,
        total_counts=cfg.train.param_update.NUM_MINIBATCHES * cfg.train.param_update.UPDATE_EPOCHS * NUM_UPDATES,
        init_val=cfg.train.LR,
        final_val=0.0,
    )
    graph_adj, graph_aux = make_graph(cfg.env.GRAPH, cfg.env.NUM_AGENTS, **cfg.env.GRAPH_KWARGS)

    def train(rng, frozen_params, alpha):
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
        rng, _rng = jax.random.split(rng)
        network_params = frozen_params
        init_hstate = ScannedRNN.initialize_carry(cfg.train.NUM_ENVS, cfg.NN_KWARGS.gru_hidden_size)
        init_frozen_hstate = ScannedRNN.initialize_carry(NUM_ACTORS - cfg.train.NUM_ENVS, cfg.NN_KWARGS.gru_hidden_size)

        # INIT OPTIMIZER
        lr = lr_schedule if cfg.train.ANNEAL_LR else cfg.train.LR
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.train.MAX_GRAD_NORM),
            optax.adam(learning_rate=lr, eps=1e-5),
        )

        # INIT TRAIN_STATE
        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            update_step=0,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, cfg.train.NUM_ENVS)
        prob_of_signal = jax.lax.select(
            jnp.isclose(alpha, 0.0),
            0.0,
            cfg.env.ENV_KWARGS.p_signal,
        )
        reset_params = (prob_of_signal, 1.0)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, reset_params)
        dones = {a: jnp.zeros(cfg.train.NUM_ENVS, dtype=bool) for a in env.agents + ["__all__"]}

        rng, _rng = jax.random.split(rng)
        symmetry_rng = jax.random.split(_rng, cfg.train.NUM_ENVS)
        symmetry_flips = reset_symmetry_flip(symmetry_rng, env.agents, cfg.train.OTHER_PLAY)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_dones, symmetry_flips, frozen_hstate, hstate, rng = runner_state

                # SPECIFY SYMMETRY FLIPPING FOR "OTHER PLAY"
                rng, _rng = jax.random.split(rng)
                rng_sym = jax.random.split(_rng, cfg.train.NUM_ENVS)
                symmetry_flips = jax.tree.map(
                    lambda x, y: jnp.where(last_dones["__all__"], x, y),
                    reset_symmetry_flip(rng_sym, env.agents, cfg.train.OTHER_PLAY),
                    symmetry_flips,
                )
                symmetry_flips_batch = batchify_arr(symmetry_flips, env.agents)

                # GET CURRENT TRUTH VALUE
                last_truth_values = env_state.env_state.truth_value # Need to double this since we have wrapper on environment
                last_truth_values_expanded = jnp.tile(last_truth_values, env.num_agents)
                last_truth_values_sym_flip = actions_batched_sym_flip(last_truth_values_expanded, symmetry_flips_batch)

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                last_obs_batch = batchify_dict(last_obs, env.agents)
                last_obs_batch_sym_flip = obs_batched_sym_flip(last_obs_batch, symmetry_flips_batch)
                last_dones_batch = batchify_arr(last_dones, env.agents)

                # SEPARATE OUT SELECTED AGENTS
                last_truth_values_sym_flip_reshaped = last_truth_values_sym_flip.reshape(env.num_agents, cfg.train.NUM_ENVS)
                last_truth_values_sym_flip_0 = last_truth_values_sym_flip_reshaped[selected_agents].reshape(selected_agents.size * cfg.train.NUM_ENVS)

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

                last_dones_batch_reshaped = last_dones_batch.reshape(env.num_agents, cfg.train.NUM_ENVS)
                last_dones_batch_0 = last_dones_batch_reshaped[selected_agents].reshape(selected_agents.size * cfg.train.NUM_ENVS)
                last_dones_batch_rest = last_dones_batch_reshaped[rest_agents].reshape(rest_agents.size * cfg.train.NUM_ENVS)

                ac_in_0 = (
                    jax.tree.map(lambda x: x[np.newaxis, :], last_obs_batch_sym_flip_0),
                    last_dones_batch_0[np.newaxis, :],
                )
                ac_in_rest = (
                    jax.tree.map(lambda x: x[np.newaxis, :], last_obs_batch_sym_flip_rest),
                    last_dones_batch_rest[np.newaxis, :],
                )

                hstate, pi, values, _ = network.apply(train_state.params, hstate, ac_in_0)
                actions_0 = pi.sample(seed=_rng)
                log_probs = pi.log_prob(actions_0)
                values, actions_0, log_probs = (
                    values.squeeze(0),
                    actions_0.squeeze(0),
                    log_probs.squeeze(0),
                )

                frozen_hstate, frozen_pi, _, _ = network.apply(frozen_params, frozen_hstate, ac_in_rest)
                actions_rest = frozen_pi.sample(seed=_rng)
                actions_rest = actions_rest.squeeze(0)

                actions_batch_0 = actions_0
                actions_0 = actions_0.reshape(selected_agents.size, cfg.train.NUM_ENVS)
                actions_rest = actions_rest.reshape(rest_agents.size, cfg.train.NUM_ENVS)

                actions = jnp.zeros((env.num_agents, cfg.train.NUM_ENVS), dtype=actions_0.dtype)
                actions = actions.at[selected_agents].set(actions_0).at[rest_agents].set(actions_rest)
                actions = actions.reshape(env.num_agents * cfg.train.NUM_ENVS)

                actions_sym_flip = actions_batched_sym_flip(actions, symmetry_flips_batch)
                env_act = unbatchify_arr(actions_sym_flip, env.agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, cfg.train.NUM_ENVS)
                # Don't symmetry flip obs in next line since we do this for last_obs which will be this obs in the next iteration.
                obs, env_state, rewards, dones, infos = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, env_act, reset_params)
                
                rewards_batch = batchify_arr(rewards, env.agents)
                rewards_batch_reshaped = rewards_batch.reshape(env.num_agents, cfg.train.NUM_ENVS)
                rewards_batch_0 = rewards_batch_reshaped[selected_agents].reshape(selected_agents.size * cfg.train.NUM_ENVS)

                # We don't take the symemtry flipped versions of the actions here since the neural network does not technically output the symmetry flips.
                # Instead we flip the actions when acting on the environment.
                # We must remember to flip the observations here though since these are actually passed into the network.
                transition = Transition(
                    jnp.tile(last_dones["__all__"], selected_agents.size),
                    last_dones_batch_0,
                    actions_batch_0, 
                    values,
                    rewards_batch_0,
                    log_probs,
                    last_obs_batch_sym_flip_0,
                    last_truth_values_sym_flip_0,
                    infos,
                )
                runner_state = (train_state, env_state, obs, dones, symmetry_flips, frozen_hstate, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, cfg.train.TRAINING_INTERVAL
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_dones, symmetry_flips, frozen_hstate, hstate, rng = runner_state
            last_obs_batch = batchify_dict(last_obs, env.agents)
            symmetry_flips_batch = batchify_arr(symmetry_flips, env.agents)
            last_obs_batch_sym_flip_0 = jax.tree.map(
                lambda leaf: leaf.reshape(env.num_agents, cfg.train.NUM_ENVS, -1)[selected_agents].reshape(selected_agents.size * cfg.train.NUM_ENVS, -1).squeeze(),
                obs_batched_sym_flip(last_obs_batch, symmetry_flips_batch),
            )
            last_dones_batch_0 = batchify_arr(last_dones, env.agents).reshape(env.num_agents, cfg.train.NUM_ENVS)[selected_agents].reshape(selected_agents.size * cfg.train.NUM_ENVS)
            ac_in = (
                jax.tree.map(lambda x: x[np.newaxis, :], last_obs_batch_sym_flip_0),
                last_dones_batch_0[np.newaxis, :],
            )
            _, _, last_vals, _ = network.apply(train_state.params, hstate, ac_in)
            last_vals = last_vals.squeeze()

            def _calculate_gae(traj_batch, last_vals):
                def _get_advantages(gae_and_next_values, transition):
                    gae, next_values = gae_and_next_values
                    dones, values, rewards = (
                        transition.global_dones,
                        transition.values,
                        transition.rewards,
                    )
                    delta = rewards + cfg.train.GAMMA * next_values * (1 - dones) - values
                    gae = (
                        delta
                        + cfg.train.GAMMA * cfg.train.GAE_LAMBDA * (1 - dones) * gae
                    )
                    return (gae, values), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_vals), last_vals),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.values

            advantages, targets = _calculate_gae(traj_batch, last_vals)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, values, _ = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.dones),
                        )
                        log_probs = pi.log_prob(traj_batch.actions)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.values + (
                            values - traj_batch.values
                        ).clip(-CLIP_EPS, CLIP_EPS)
                        value_losses = jnp.square(values - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS
                        logratios = log_probs - traj_batch.log_probs
                        ratios = jnp.exp(logratios)
                        ratio_0 = ratios.mean()  # Average ratios
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        actor_losses1 = ratios * gae
                        actor_losses2 = jnp.clip(ratios, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * gae
                        actor_losses = -jnp.minimum(actor_losses1, actor_losses2)
                        actor_loss = actor_losses.mean()

                        # CALCULATE ENTROPY
                        entropy = pi.entropy().mean()

                        # KL DIVERGENCE
                        approx_kl = ((ratios - 1) - logratios).mean()

                        # FRACTION OF CLIPPED RATIOS
                        clip_frac = jnp.where(jnp.abs(ratios - 1) > CLIP_EPS, 1, 0).mean()

                        total_loss = (
                            actor_loss
                            + cfg.train.coef.VF_COEF * value_loss
                            - cfg.train.coef.ENT_COEF * entropy
                        )
                        return total_loss, (value_loss, actor_loss, entropy, ratio_0, approx_kl, clip_frac)

                    def _loss_fn_belief(params, init_hstate, traj_batch): 
                        # FREEZE BACKBONE
                        def _freeze_non_belief(params):
                            flat_params = flatten_dict(params)
                            frozen_params = {
                                key: jax.lax.select(
                                    jnp.asarray(key[1].startswith("BeliefHead"), dtype=bool),
                                    val,
                                    jax.lax.stop_gradient(val),
                                ) for key, val in flat_params.items()
                            }
                            frozen_params = unflatten_dict(frozen_params)
                            return frozen_params

                        frozen_params = _freeze_non_belief(params)

                        # RERUN NETWORK
                        _, _, _, beliefs = network.apply(
                            frozen_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.dones),
                        )
                        belief_loss = jnp.square(beliefs - traj_batch.truth_values).mean()
                        return belief_loss

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    loss_info, grads_ac = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    grad_fn_belief = jax.value_and_grad(_loss_fn_belief, has_aux=False)
                    loss_belief, grads_belief = grad_fn_belief(
                        train_state.params, init_hstate, traj_batch
                    )
                    grads = jax.tree.map(
                        lambda x, y: x + y,
                        grads_ac,
                        grads_belief,
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    loss_info = (*loss_info, loss_belief)

                    return train_state, loss_info

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jnp.reshape(init_hstate, (1, cfg.train.NUM_ENVS, -1))
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, cfg.train.NUM_ENVS)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], cfg.train.param_update.NUM_MINIBATCHES, -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, loss_info = jax.lax.scan(
                    _update_minibatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, cfg.train.param_update.UPDATE_EPOCHS
            )
            train_state = update_state[0]
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            metrics = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio_0": loss_info[1][3],
                "approx_KL": loss_info[1][4],
                "clip_frac": loss_info[1][5],
                "belief_loss": loss_info[2],
                "returns": traj_batch.infos["returned_episode_returns"].mean(),
                "normalised_return_agent_0": traj_batch.rewards.mean(),
                "normalised_returns": traj_batch.infos["returned_episode_returns"].mean() / (selected_agents.size * cfg.env.MAX_STEPS),
                "final_accuracy": traj_batch.infos["accuracy"][-1].mean(),
                "total_step": train_state.update_step * cfg.train.NUM_ENVS * cfg.train.TRAINING_INTERVAL,
            }
            if cfg.wandb.get("WANDB_MODE", "disabled") == "online":
                def callback(metrics, alpha):
                    metrics_alpha = {k + f"/{alpha:.2f}": v for k, v in metrics.items()}
                    wandb.log(metrics_alpha)
                jax.experimental.io_callback(callback, None, metrics, alpha)

            train_state = train_state.replace(update_step=train_state.update_step + 1)
            rng = update_state[-1]
            runner_state = (train_state, env_state, last_obs, last_dones, symmetry_flips, frozen_hstate, hstate, rng)
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obs,
            dones,
            symmetry_flips,
            init_frozen_hstate,
            init_hstate,
            _rng,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, NUM_UPDATES
        )
        train_state, _, _, _, _, _, _, _ = runner_state
        params = train_state.params
        return params, metrics

    return train


