from typing import Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from opinimarl.utils import segment_softmax

from .scanned_rnn import ScannedRNN


class UnifiedAttentionHead(nn.Module):
    d: int
    num_agents: int
    use_edge_list: bool
    use_agent_ids: bool = True

    def setup(self):
        # Embedding layers
        self.action_embed = nn.Embed(num_embeddings=4, features=self.d)
        self.id_embed = nn.Embed(num_embeddings=self.num_agents, features=self.d)

        # QKV projections
        self.q_proj = nn.Dense(self.d)
        self.k_proj = nn.Dense(self.d)
        self.v_proj = nn.Dense(self.d)

    def __call__(self, obs, senders=None, receivers=None, symmetry_flips=None):
        """
        senders: (L,) int32 – indices of neighbour nodes (who sends the message)
        receivers: (L,) int32 – indices of nodes receiving the message
        obs: (t, num_actors, -1) float32 - observations
        symmetry_flips: (t, num_actors) int32
        """
        if self.use_edge_list:
            assert symmetry_flips != None, "symmetry_flips must not be None when use_edge_list is True."
            assert senders != None, "senders must not be None when use_edge_list is True."
            assert receivers != None, "receivers must not be None when use_edge_list is True."

            num_envs = obs["agent_id"].shape[1] // self.num_agents
            symmetry_flips_reshaped = symmetry_flips.reshape(-1, num_envs, self.num_agents, order="F") # (t, num_envs, num_agents)
            sender_symmetry_flips  = symmetry_flips_reshaped[:, :, senders] # (t, num_envs, L)
            receiver_symmetry_flips  = symmetry_flips_reshaped[:, :, receivers] # (t, num_envs, L)
            symmetry_mismatch = sender_symmetry_flips != receiver_symmetry_flips # (t, num_envs, L)

            own_ids = obs["agent_id"].astype(jnp.int32) # (t, num_actors)
            own_ids_reshaped = own_ids.reshape(-1, num_envs, self.num_agents, order="F") # (t, num_envs, num_agents)

            neighbour_actions = obs["neighbourhood_outputs"].astype(jnp.int32) # (t, num_actors, num_agents)
            own_actions = jax.vmap(
                lambda a, oid: a[jnp.arange(a.shape[0]), oid].astype(jnp.int32), # (num_actors,)
                in_axes=(0, 0),
            )(neighbour_actions, own_ids).astype(jnp.int32) # (t, num_actors)
            own_actions_reshaped = own_actions.reshape(-1, num_envs, self.num_agents, order="F") # (t, num_envs, num_agents)

            neighbour_actions_reshaped = neighbour_actions.reshape(-1, num_envs, self.num_agents, self.num_agents, order="F") # (t, num_envs, num_agents, num_agents)
            interaction_mask = neighbour_actions_reshaped[:, :, receivers, senders] != 3 # (t, num_envs, L)

            sender_ids_reshaped = own_ids_reshaped[:, :, senders] # (t, num_envs, L)
            sender_actions_reshaped = own_actions_reshaped[:, :, senders] # (t, num_envs, L)
            sender_actions_symmetry_flips_reshaped = jnp.where(
                symmetry_mismatch & (sender_actions_reshaped != 2),
                1 - sender_actions_reshaped,
                sender_actions_reshaped,
            ) # (t, num_envs, L)

            # Embed agents
            # own_emb = self.action_embed(receiver_actions_reshaped) + self.id_embed(receiver_ids_reshaped)   # (t, num_envs, L, d)
            if self.use_agent_ids:
                own_emb = self.action_embed(own_actions_reshaped) + self.id_embed(own_ids_reshaped) # (t, num_envs, num_agents, d)
                send_emb = self.action_embed(sender_actions_symmetry_flips_reshaped) + self.id_embed(sender_ids_reshaped)  # (t, num_envs, L, d)
            else:
                own_emb = self.action_embed(own_actions_reshaped) # (t, num_envs, num_agents, d)
                send_emb = self.action_embed(sender_actions_symmetry_flips_reshaped)  # (t, num_envs, L, d)

            # Projections
            q = self.q_proj(own_emb)           # (t, num_envs, num_agents, d)
            k = self.k_proj(send_emb)          # (t, num_envs, L, d)
            v = self.v_proj(send_emb)          # (t, num_envs, L, d)

            # Compute attention logits for each edge
            logits = jnp.einsum("teld,teld->tel", q[:, :, receivers, :], k) / jnp.sqrt(self.d)  # (t, num_envs, L)
            logits = jnp.where(interaction_mask, logits, -jnp.inf)

            # Normalize per receiver (softmax over incoming edges)
            attn = jax.vmap(
                jax.vmap(
                    lambda x: segment_softmax(x, receivers, num_segments=self.num_agents), # (L)
                    in_axes=0,
                ),
                in_axes=0,
            )(logits) # shape (t, num_envs, L)

            # Weight values by attention
            msgs = attn[:, :, :, None] * v  # (t, num_envs, L, d)

            # Aggregate messages into nodes
            out_reshaped = jax.vmap(
                jax.vmap(
                    lambda x: jax.ops.segment_sum(x, receivers, num_segments=self.num_agents),  # (num_agents, d)
                    in_axes=0,
                ),
                in_axes=0,
            )(msgs) # shape (t, num_envs, num_agents, d)
            out_1 = out_reshaped.reshape(-1, num_envs*self.num_agents, self.d, order="F") # (t, num_actors, d)

            return out_1, attn

        else:
            own_id = obs["agent_id"].astype(jnp.int32) # (t, num_actors)
            neighbour_actions = obs["neighbourhood_outputs"].astype(jnp.int32) # (t, num_actors, num_agents)
            print(f"shape own_id: {own_id.shape}")
            print(f"shape neighbour_acitons: {neighbour_actions.shape}")
            own_action = jax.vmap(
                lambda a, oid: a[jnp.arange(a.shape[0]), oid].astype(jnp.int32), # (num_actors,)
                in_axes=(0, 0),
            )(neighbour_actions, own_id).astype(jnp.int32) # (t, num_actors)
            neighbour_ids = jnp.broadcast_to(jnp.arange(self.num_agents), neighbour_actions.shape).astype(jnp.int32)  # (t, num_actors, num_agents)
            neighbour_mask = neighbour_actions != 3

            # Embed self and neighbours
            if self.use_agent_ids:
                own_emb = self.action_embed(own_action) + self.id_embed(own_id) # (t, num_actors, d)
                neighbour_emb = self.action_embed(neighbour_actions) + self.id_embed(neighbour_ids) # (t, num_actors, num_agents, d)
            else:
                own_emb = self.action_embed(own_action)# (t, num_actors, d)
                neighbour_emb = self.action_embed(neighbour_actions)# (t, num_actors, num_agents, d)


            q = self.q_proj(own_emb)         # (t, num_actors, d)
            k = self.k_proj(neighbour_emb)   # (t, num_actors, num_agents, d)
            v = self.v_proj(neighbour_emb)   # (t, num_actors, num_agents, d)

            logits = jnp.einsum("tnd,tnmd->tnm", q, k) / jnp.sqrt(self.d) # (t, num_actors, num_agents)

            mask = jnp.where(neighbour_mask, 0.0, -1e18) # (t, num_actors, num_agents)
            logits = logits + mask

            attn = nn.softmax(logits, axis=-1) # (t, num_actors, num_agents)

            out_2 = jnp.einsum("tnm,tnmd->tnd", attn, v) # (t, num_actors, d)

            return out_2, attn



class ActorCriticRNNAttention(nn.Module):
    action_dim: Sequence[int]
    num_agents: int
    agent_id_num_bits: int
    fc_dim_size: int = 50
    gru_hidden_size: int = 50
    use_layer_norm: bool = True
    attention_d: int = 8
    use_centralised_logic: bool = True
    senders: jnp.array = None
    receivers: jnp.array = None
    use_rnn: bool = True
    use_agent_ids: bool = True

    @nn.compact
    def __call__(self, hidden, x, symmetry_flips=None, sow_attention=False):
        obs, dones = x

        attn_output, attn_weights = UnifiedAttentionHead(
            d=self.attention_d,
            num_agents=self.num_agents,
            use_edge_list=self.use_centralised_logic,
            use_agent_ids=self.use_agent_ids,
        )(obs, self.senders, self.receivers, symmetry_flips)

        private_signal = obs["private_signal"][:, :, None]

        # Create binary representation of own id
        own_id = obs["agent_id"].astype(jnp.int32) # (t, num_actors)
        own_id_binary_mask = 2 ** jnp.arange(self.agent_id_num_bits - 1, -1, -1)
        own_id_binary = ((own_id[..., None] & own_id_binary_mask) > 0).astype(jnp.int32)


        if self.use_agent_ids:
            embedding = jnp.concatenate([private_signal, own_id_binary, attn_output], axis=-1) # (batch_size, 1+ceil(log2(num_agents))+d)
        else:
            embedding = jnp.concatenate([private_signal, attn_output], axis=-1) # (batch_size, 1+d)

        embedding = nn.Dense(
            self.fc_dim_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        if self.use_layer_norm:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        embedding = nn.relu(embedding)
        
        embedding = nn.Dense(
            self.fc_dim_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        if self.use_layer_norm:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        embedding = nn.relu(embedding)

        if self.use_rnn:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        actor_mean = nn.Dense(self.gru_hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.use_layer_norm:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        actor_mean = nn.relu(actor_mean)
        
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(self.fc_dim_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.use_layer_norm:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        critic = nn.relu(critic)
        
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        critic = jnp.squeeze(critic, axis=-1)
        
        belief = nn.Dense(self.fc_dim_size, kernel_init=orthogonal(2), bias_init=constant(0.0), name="BeliefHead_0")(
            embedding
        )
        if self.use_layer_norm:
            embedding = nn.LayerNorm(use_scale=False, name="BeliefHead_LayerNorm")(embedding)
        belief = nn.relu(belief)
        
        belief = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="BeliefHead_1")(
            belief
        )
        belief = nn.sigmoid(belief)
        belief = jnp.squeeze(belief, axis=-1)

        if sow_attention:
            return (hidden, pi, critic, belief), attn_weights
        else:
            return hidden, pi, critic, belief
