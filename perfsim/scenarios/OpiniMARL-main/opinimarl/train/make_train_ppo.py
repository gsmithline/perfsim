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
    aux_rewards: jnp.ndarray
    log_probs: jnp.ndarray
    obs: jnp.ndarray
    truth_values: jnp.ndarray
    symmetry_flips: jnp.ndarray
    infos: jnp.ndarray

# TrainState class
class CustomTrainState(TrainState):
    update_step: int

# Make train function
def make_train_ppo(cfg):
    NUM_ACTORS = cfg.env.NUM_AGENTS * cfg.train.NUM_ENVS
    NUM_UPDATES = cfg.train.TOTAL_TIMESTEPS // cfg.train.TRAINING_INTERVAL // cfg.train.NUM_ENVS
    CLIP_EPS = (
        cfg.train.epsiilon.CLIP / cfg.env.NUM_AGENTS
        if cfg.train.epsilon.SCALE_CLIP
        else cfg.train.epsilon.CLIP
    )

    lr_schedule = partial(
        sch.linear_schedule,
        total_counts=cfg.train.param_update.NUM_MINIBATCHES * cfg.train.param_update.UPDATE_EPOCHS * NUM_UPDATES,
        init_val=cfg.train.LR,
        final_val=0.0,
    )

    # Reward shaping for alpha = 0
    reward_shaping_schedule = partial(
        sch.linear_schedule,
        total_counts=NUM_UPDATES / 2,
        init_val=0.05,
        final_val=0.0,
    )

    prob_of_signal_schedule = partial(
        sch.exponential_schedule,
        total_counts=NUM_UPDATES,
        init_val=1.0,
        final_val=cfg.env.ENV_KWARGS.p_signal,
    )
    
    prob_of_signal_schedule_zero = partial(
        sch.exponential_linear_schedule,
        total_counts=NUM_UPDATES,
        init_val=1.0,
        final_val=0.0,
        epsilon=1e-2,
        switch_frac=0.95,
    )

    percolation_prob_schedule = partial(
        sch.sigmoid_schedule,
        total_counts=NUM_UPDATES,
        init_val=cfg.train.PERCOLATION_PROB_INIT,
        final_val=1.0,
        a=1.05,
        b=24,
        c=7/30,
    )


    graph_adj, graph_aux = make_graph(cfg.env.GRAPH, cfg.env.NUM_AGENTS, **cfg.env.GRAPH_KWARGS)
    graph_with_self_loops = graph_adj + jnp.eye(cfg.env.NUM_AGENTS, dtype=jnp.int32)
    receivers, senders = jnp.nonzero(graph_with_self_loops)
    receivers = receivers.astype(jnp.int32)
    senders = senders.astype(jnp.int32)

    def train(rng, alpha):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
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
        rng, _rng = jax.random.split(rng)
        init_x = (
            {"agent_id": jnp.zeros((1, NUM_ACTORS)),
             "private_signal": jnp.zeros((1, NUM_ACTORS)),
             "neighbourhood_outputs": jnp.zeros((1, NUM_ACTORS, env.num_agents))},
            jnp.zeros((1, NUM_ACTORS)),
        )
        init_symmetry_flips = jnp.zeros((1, NUM_ACTORS))
        init_hstate = ScannedRNN.initialize_carry(NUM_ACTORS, cfg.NN_KWARGS.gru_hidden_size)
        network_params = network.init(_rng, init_hstate, init_x, init_symmetry_flips)

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
        reset_params = (1.0, cfg.train.PERCOLATION_PROB_INIT)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, reset_params)
        dones = {a: jnp.zeros(cfg.train.NUM_ENVS, dtype=bool) for a in env.agents + ["__all__"]}

        rng, _rng = jax.random.split(rng)
        symmetry_rng = jax.random.split(_rng, cfg.train.NUM_ENVS)
        symmetry_flips = reset_symmetry_flip(symmetry_rng, env.agents, cfg.train.OTHER_PLAY)
        rng, _rng = jax.random.split(rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_dones, symmetry_flips, hstate, rng = runner_state

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

                ac_in = (
                    jax.tree.map(lambda x: x[np.newaxis, :], last_obs_batch_sym_flip),
                    last_dones_batch[np.newaxis, :],
                )
                symmetry_flips_in = symmetry_flips_batch[np.newaxis, :]

                hstate, pi, values, _ = network.apply(
                    train_state.params,
                    hstate,
                    ac_in,
                    symmetry_flips_in,
                )
                actions = pi.sample(seed=_rng)
                log_probs = pi.log_prob(actions)
                values, actions, log_probs = (
                    values.squeeze(0),
                    actions.squeeze(0),
                    log_probs.squeeze(0),
                )
                actions_sym_flip = actions_batched_sym_flip(actions, symmetry_flips_batch)
                env_act = unbatchify_arr(actions_sym_flip, env.agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                prob_of_signal = jax.lax.cond(
                    jnp.isclose(alpha, 0.0),
                    prob_of_signal_schedule_zero,
                    prob_of_signal_schedule,
                    train_state.update_step,
                )
                percolation_prob = percolation_prob_schedule(train_state.update_step)
                reset_params = (prob_of_signal, percolation_prob)
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, cfg.train.NUM_ENVS)
                # Don't symmetry flip obs in next line since we do this for last_obs which will be this obs in the next iteration.
                obs, env_state, rewards, dones, infos = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, env_act, reset_params)
                
                rewards_batch = batchify_arr(rewards, env.agents)

                aux_rewards_batch = batchify_arr(infos["aux_rewards"], env.agents)

                # Reward shape for alpha = 0
                rewards_batch = jax.lax.select(
                    jnp.isclose(alpha, 0.0),
                    (1 - reward_shaping_schedule(train_state.update_step)) * rewards_batch + reward_shaping_schedule(train_state.update_step) * aux_rewards_batch,
                    rewards_batch,
                )

                # We don't take the symemtry flipped versions of the actions here since the neural network does not technically output the symmetry flips.
                # Instead we flip the actions when acting on the environment.
                # We must remember to flip the observations here though since these are actually passed into the network.
                transition = Transition(
                    jnp.tile(last_dones["__all__"], env.num_agents),
                    last_dones_batch,
                    actions, 
                    values,
                    rewards_batch,
                    aux_rewards_batch,
                    log_probs,
                    last_obs_batch_sym_flip,
                    last_truth_values_sym_flip,
                    symmetry_flips_batch,
                    infos,
                )
                runner_state = (train_state, env_state, obs, dones, symmetry_flips, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, cfg.train.TRAINING_INTERVAL
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_dones, symmetry_flips, hstate, rng = runner_state
            last_obs_batch = batchify_dict(last_obs, env.agents)
            symmetry_flips_batch = batchify_arr(symmetry_flips, env.agents)
            last_obs_batch_sym_flip = obs_batched_sym_flip(last_obs_batch, symmetry_flips_batch)
            last_dones_batch = batchify_arr(last_dones, env.agents)
            ac_in = (
                jax.tree.map(lambda x: x[np.newaxis, :], last_obs_batch_sym_flip),
                last_dones_batch[np.newaxis, :],
            )
            symmetry_flips_in = symmetry_flips_batch[np.newaxis, :]
            _, _, last_vals, _ = network.apply(train_state.params, hstate, ac_in, symmetry_flips_in)
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
                            traj_batch.symmetry_flips,
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
                            traj_batch.symmetry_flips,
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

                init_hstate = jnp.reshape(init_hstate, (1, NUM_ACTORS, -1))
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, cfg.train.NUM_ENVS)

                # Note that the following method for batching is unique to the centralised logic of the neural network.
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(
                        x.reshape([x.shape[0], cfg.train.NUM_ENVS, -1] + list(x.shape[2:]), order="F"),
                        permutation,
                        axis=1).reshape([x.shape[0], -1] + list(x.shape[2:])), # No order F here since we want to order my env not agent
                    batch,
                )

                assert cfg.train.NUM_ENVS % cfg.train.param_update.NUM_MINIBATCHES == 0, "NUM_MINIBATCHES must divide NUM_ENVS perfectly."
                BATCH_NUM_ENVS = cfg.train.NUM_ENVS // cfg.train.param_update.NUM_MINIBATCHES
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], cfg.train.param_update.NUM_MINIBATCHES, BATCH_NUM_ENVS, -1]
                            + list(x.shape[2:]),
                        ).reshape([x.shape[0], cfg.train.param_update.NUM_MINIBATCHES, -1] + list(x.shape[2:]), order="F"),
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

            aux_rewards_dict = {f"aux_rewards/step_{i}": traj_batch.aux_rewards[i].mean() for i in range(cfg.train.TRAINING_INTERVAL)}

            metrics = {
                **aux_rewards_dict,
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio_0": loss_info[1][3],
                "approx_KL": loss_info[1][4],
                "clip_frac": loss_info[1][5],
                "belief_loss": loss_info[2],
                "rewards": traj_batch.rewards.mean(),
                "returns": traj_batch.infos["returned_episode_returns"].sum(),
                "normalised_returns": traj_batch.infos["returned_episode_returns"].mean() / cfg.env.MAX_STEPS,
                "mean_episode_accuracy": traj_batch.infos["accuracy"].mean(),
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
            runner_state = (train_state, env_state, last_obs, last_dones, symmetry_flips, hstate, rng)
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obs,
            dones,
            symmetry_flips,
            init_hstate,
            _rng,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, NUM_UPDATES
        )
        train_state, _, _, _, _, _, _ = runner_state
        params = train_state.params
        return params, metrics

    return train


