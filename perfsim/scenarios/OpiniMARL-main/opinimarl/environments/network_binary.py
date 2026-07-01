"""
This environment does not change the personal signal with each timestep and only at resets.
"""
from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from distrax import Categorical
from flax import struct
from jaxmarl.environments import spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv


@struct.dataclass
class State:
    truth_value: int # 1 (True) or 0 (False)
    agent_outputs: jnp.ndarray # num_agents x truth_dim
    step: int
    network: jnp.ndarray
    interaction_network: jnp.ndarray
    instant_interaction_network: jnp.ndarray
    agents_with_signal: jnp.ndarray
    noise_key: chex.PRNGKey = jax.random.key(0)

class NetworkBinary(MultiAgentEnv):
    def __init__(
        self,
        graph_adj,
        alpha=1.0,
        max_steps=20,
        null_action=True,
        sigma=1,
        p_signal=1.0,
        expert_agents=[],
        extra_info=False,
        q=None, # Probability of interaction matrix
        r=None, # Interaction_switch_rate
    ):
        self.num_agents = graph_adj.shape[0]
        assert self.num_agents >= 2, "The minimum number of agents for this environment is 2"
        super().__init__(self.num_agents)
        self.graph_adj = graph_adj
        self.agent_range = jnp.arange(self.num_agents)
        self.alpha = alpha
        self.max_steps = max_steps
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.a_to_i = {a: i for i, a in enumerate(self.agents)}
        self.null_action = null_action
        self.sigma = sigma
        self.p_signal = p_signal if p_signal != None else 1.0
        self.expert_agents = jnp.array(expert_agents)
        self.extra_info = extra_info

        # Dynamic graph quantities
        self.q = q
        self.r = r

        # Action space
        if self.null_action:
            action_space = spaces.Discrete(num_categories=3, dtype=jnp.int32)
        else:
            action_space = spaces.Discrete(num_categories=2, dtype=jnp.int32)
        self.action_spaces = {a: action_space for a in self.agents}

        # Observation space
        observation_space = spaces.Dict({
            "agent_id": spaces.Discrete(self.num_agents + 1),
            "private_signal": spaces.Box(
                low=-jnp.inf, 
                high=jnp.inf, 
                shape=(), 
                dtype=jnp.float32
            ),
            # 0 - guess 0, 1 - guess 1, 2 - wait, 3, no edge connection
            "neighbourhood_outputs": spaces.Discrete(4) 
        })
        self.observation_spaces = {a: observation_space for a in self.agents}


    # Added an extra argument "params" compared to the base class
    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey, params) -> Tuple[Dict[str, chex.Array], State]:
        """Resets the environment"""

        p_signal, percolation_prob = params

        rngs = jax.random.split(key, 7)

        new_truth = jax.random.randint(rngs[0], shape=(), minval=0, maxval=2) # Select either 0 or 1

        new_agent_outputs = 2*jnp.ones(shape=(self.num_agents), dtype=jnp.int32)

        if p_signal == None:
            new_agents_with_signal = jax.random.bernoulli(rngs[1], p=self.p_signal, shape=(self.num_agents))
        else:
            new_agents_with_signal = jax.random.bernoulli(rngs[1], p=p_signal, shape=(self.num_agents))

        new_network = self.graph_adj

        if percolation_prob != None:
            mask = jax.random.bernoulli(key=rngs[3], p=percolation_prob, shape=(self.num_agents, self.num_agents))
            new_network = mask & self.graph_adj

        if (self.q  != None) & (self.r != None):
            mask = jax.random.bernoulli(key=rngs[4], p=self.q)
            mask = jnp.triu(mask, k=1) + jnp.triu(mask, k=1).T
            instant_interaction_network = mask & new_network
            instant_interaction_network, interaction_network = self.update_interaction_network(rngs[5], instant_interaction_network, new_network)
        else:
            instant_interaction_network = None
            interaction_network = None

        new_state = State(
            truth_value=new_truth,
            agent_outputs=new_agent_outputs,
            step=0,
            network=new_network,
            interaction_network=interaction_network,
            instant_interaction_network=instant_interaction_network,
            agents_with_signal=new_agents_with_signal,
            noise_key=rngs[6],
        )

        obs = self.get_obs(new_state)

        return obs, new_state


    # Overwrite step function to allow for extra input parameters to reset function
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        params=(None, None),
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment. Resets the environment if done.
        To control the reset state, pass `reset_state`. Otherwise, the environment will reset using `self.reset`.

        Args:
            key (chex.PRNGKey): random key
            state (State): environment state
            actions (Dict[str, chex.Array]): agent actions, keyed by agent name
            reset_state (Optional[State], optional): Optional environment state to reset to on episode completion. Defaults to None.

        Returns:
            Observations (Dict[str, chex.Array]): next observations
            State (State): next environment state
            Rewards (Dict[str, float]): rewards, keyed by agent name
            Dones (Dict[str, bool]): dones, keyed by agent name:
            Info (Dict): info dictionary
        """

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset, params)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)

        # Auto-reset environment based on termination
        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos


    @partial(jax.jit, static_argnums=[0])
    def update_interaction_network(self, key, instant_interaction_network, full_network):
        no_instant_interaction_network = full_network - instant_interaction_network
        a = self.q * self.r

        # Update instantaneous interaction network
        mask_switch_interact = jax.random.bernoulli(
            jax.random.fold_in(key, 0),
            p = self.q * (1 - jnp.exp(-self.r)),
        )
        mask_stay_interact = jax.random.bernoulli(
            jax.random.fold_in(key, 1),
            p = self.q + (1 - self.q) * jnp.exp(-self.r),
        )

        new_instant_interaction_network = ((no_instant_interaction_network & mask_switch_interact) | (instant_interaction_network & mask_stay_interact))
        upper = jnp.triu(new_instant_interaction_network, k=1) # Interactions are symmetric and we sample once for each edge.
        new_instant_interaction_network = upper | upper.T
        new_no_instant_interacton_network = full_network - new_instant_interaction_network

        # Update accumulated interaction_network
        mask_interact_from_no_to_no = jax.random.bernoulli(
            jax.random.fold_in(key, 2),
            p = 1 - jnp.exp(-a)/(1 - self.q * (1 - jnp.exp(-self.r))),
        )
        interact_from_no_to_no = no_instant_interaction_network & new_no_instant_interacton_network &  mask_interact_from_no_to_no
        new_interaction_network = interact_from_no_to_no | instant_interaction_network | new_instant_interaction_network
        upper = jnp.triu(new_interaction_network, k=1) # Interactions are symmetric and we sample once for each edge.
        new_interaction_network = upper | upper.T

        return new_instant_interaction_network.astype(jnp.int32), new_interaction_network.astype(jnp.int32)

    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]: # Technically we output just the Dict and not Tuple for the observations.
        """Performs step in environment"""
        # New state
        new_agent_outputs = jnp.array([actions[a] for a in self.agents])

        new_step = state.step + 1

        key, _key = jax.random.split(key)
        if (self.q != None) & (self.r != None):
            instant_interaction_network, interaction_network = self.update_interaction_network(_key, state.instant_interaction_network, state.network)
        else:
            instant_interaction_network = None
            interaction_network = None

        new_state = State(
            truth_value=state.truth_value,
            agent_outputs=new_agent_outputs,
            step=new_step,
            network=state.network,
            interaction_network=interaction_network,
            instant_interaction_network=instant_interaction_network,
            agents_with_signal=state.agents_with_signal,
            noise_key=state.noise_key,
        )

        # New observations
        obs = self.get_obs(new_state)

        # Rewards
        @partial(jax.vmap, in_axes=[0, None, None])
        def _reward_and_accuracy(agent_id: int, new_state: State, old_state: State):
            self_output_new = new_state.agent_outputs[agent_id]
            # self_output_old = old_state.agent_outputs[agent_id]

            # Truth reward
            truth_reward = jax.lax.select(
                self_output_new == old_state.truth_value, # Must take old here in case of truth value flip
                0.1,
                jax.lax.select(
                    self_output_new == 2,
                    0.0,
                    -0.2,
                ),
            )

            # Conformity reward
            if (self.q != None) & (self.r != None):
                edges = old_state.interaction_network[agent_id]
            else:
                edges = old_state.network[agent_id]
            num_neighbours = edges.sum()
            conform_reward = jnp.where(
                (edges == 1) & (new_state.agent_outputs == self_output_new) & (new_state.agent_outputs != 2),
                0.1,
                0.0,
            )
            conform_penalty = jnp.where(
                (edges == 1) & (new_state.agent_outputs != self_output_new) & (new_state.agent_outputs != 2) & (self_output_new != 2),
                -0.2,
                0.0,
            )
            conform_reward_total = (conform_reward + conform_penalty).sum() / jnp.maximum(num_neighbours, 1)

            intrinsic_reward = self.alpha * truth_reward + (1 - self.alpha) * conform_reward_total
            accuracy = self_output_new == old_state.truth_value # Must take old here in case of truth value flip

            aux_reward = truth_reward

            return intrinsic_reward, aux_reward, accuracy
        
        rewards, aux_rewards, accuracies = _reward_and_accuracy(self.agent_range, new_state, state)
        rewards = {a: rewards[i] for i, a in enumerate(self.agents)}
        aux_rewards = {a: aux_rewards[i] for i, a in enumerate(self.agents)}
        global_accuracy = accuracies.mean()

        # Compute entropy of belief distribution
        p1 = jnp.mean(new_state.agent_outputs) # Probabliity of believing 1
        p0 = 1 - p1 # Probability of believing 0
        dist = Categorical(probs=jnp.array([p0, p1]))
        entropy = dist.entropy()
        
        # Dones
        done = jnp.equal(new_state.step, self.max_steps)
        dones = {a: done for a in self.agents + ["__all__"]}

        # Info
        info = {"accuracy": global_accuracy, "entropy": entropy, "aux_rewards": aux_rewards} 

        return obs, new_state, rewards, dones, info
    

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict:
        """Returns the observations of the agents"""
        @partial(jax.vmap, in_axes=[0, None])
        def _observation(agent_id: int, state: State) -> jnp.ndarray:
            """Returns observation for agent i"""
            noise_key = jax.random.split(state.noise_key, self.num_agents)[agent_id] # Constant for episode

            # Agent ID
            agent_id_arr = jnp.array([agent_id])
            agent_id_one_hot = jax.nn.one_hot(agent_id, self.num_agents)

            obs_truth = jax.lax.select(
                jnp.any(self.expert_agents == agent_id),
                jnp.float32(state.truth_value),
                jnp.float32(state.truth_value) + self.sigma * jax.random.normal(noise_key, dtype=jnp.float32),
            )

            obs_truth = jax.lax.select(
                state.agents_with_signal[agent_id] == 1,
                obs_truth,
                0.5,
            )

            if (self.q != None) & (self.r != None):
                mask_neighbours = state.interaction_network[agent_id]
            else:
                mask_neighbours = state.network[agent_id]
            mask_neighbours_plus_self = mask_neighbours + agent_id_one_hot

            obs_outputs = jnp.where(
                mask_neighbours_plus_self,
                state.agent_outputs,
                3, # Indicates no edge
            )

            obs_dict = {
                "agent_id": agent_id,
                "private_signal": obs_truth,
                "neighbourhood_outputs": obs_outputs,
            }
            return obs_dict
        
        obs = _observation(self.agent_range, state)
        return {a: jax.tree.map(lambda x: x[i], obs) for i, a in enumerate(self.agents)}

    
    def render(self, state: State):
        print(f"\nCurrent step: {state.step}")
        print(f"Current truth: {state.truth_value}")
        return None


    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__



