from functools import partial

import jax
import jax.numpy as jnp


def batchify_arr(x: dict, agent_list):
    y = jnp.stack([x[a] for a in agent_list])
    y = y.reshape(-1, *y.shape[2:])
    return y


def unbatchify_arr(x: jnp.ndarray, agent_list):
    num_agents = len(agent_list)
    y = x.reshape(num_agents, -1, *x.shape[1:])
    y = {a: y[i] for i, a in enumerate(agent_list)}
    return y


def batchify_dict(x: dict, agent_list: list):
    """
    Transposes a list of agent dictionaries into a single dictionary 
    where each value is batched.
    """
    list_of_dicts = [x[a] for a in agent_list]
    y = jax.tree.map(lambda *xs: jnp.stack(xs), *list_of_dicts)
    y = jax.tree.map(lambda leaf: leaf.reshape(-1, *leaf.shape[2:]), y)
    return y

def unbatchify_dict(x: dict, agent_list: list):
    """
    Reverses the batchify process for a dictionary of batched arrays.
    Works for any array dimension (scalars, vectors, tensors).
    """
    num_agents = len(agent_list)
    y = jax.tree.map(lambda leaf: leaf.reshape((num_agents, -1, *leaf.shape[1:])), x)
    y = {a: jax.tree.map(lambda leaf: leaf[i], y) for i, a in enumerate(agent_list)}
    return y


@partial(jax.vmap, in_axes=(0, None, None)) 
def reset_symmetry_flip(rng, agent_list, other_play=False):
    if other_play:
        symmetry_flips = jax.random.randint(rng, shape=(len(agent_list)), minval=0, maxval=2)
    else:
        symmetry_flips = jnp.zeros(shape=(len(agent_list)), dtype=jnp.int32)
    return {a: symmetry_flips[i] for i, a in enumerate(agent_list)}


def obs_batched_sym_flip(obs, symmetry_flips):
    """
    Returns symmetry flipped versions of batched dictionary observations given by batched array symmetry_flips for "other play"
    obs: batched dictionary
    symmetry_flips: batched array
    """
    # Flip private signals
    private_signals_flipped = jnp.where(symmetry_flips, 1 - obs["private_signal"], obs["private_signal"])

    # Flip neighbourhood outputs
    symmetry_flips_expanded = jnp.expand_dims(symmetry_flips, -1)
    symmetry_flips_tiled = jnp.tile(symmetry_flips_expanded, obs["neighbourhood_outputs"].shape[-1])

    outputs_flipped = jnp.where(
        symmetry_flips_tiled & (obs["neighbourhood_outputs"] < 2),
        1 - obs["neighbourhood_outputs"],
        obs["neighbourhood_outputs"],
    )

    obs_flipped = {
        "agent_id": obs["agent_id"],
        "private_signal": private_signals_flipped,
        "neighbourhood_outputs": outputs_flipped,
    }

    return obs_flipped


def actions_batched_sym_flip(actions, symmetry_flips):
    actions_flipped = jnp.where(
        symmetry_flips & (actions != 2),
        1 - actions,
        actions
    )
    return actions_flipped

def segment_softmax(logits, segment_ids, num_segments):
    """
    logits: (L,)
    segment_ids: (L,)
    num_segments: int = num_agents
    """

    # 1. Subtract segment-wise max for numerical stability
    maxes = jax.ops.segment_max(logits, segment_ids, num_segments) # (num_agents,)
    safe_maxes = jnp.where(jnp.isneginf(maxes), 0.0, maxes) # (num_agents,)
    logits_minus_max = logits - safe_maxes[segment_ids] # (L,)
    
    # 2. Exponentiate
    exp_logits = jnp.exp(logits_minus_max) # (L,)
    
    # 3. Sum per segment
    sums = jax.ops.segment_sum(exp_logits, segment_ids, num_segments) # (num_agents,)
    
    # 4. Normalise
    normalised = jnp.where(
        sums[segment_ids] > 0.0,
        exp_logits / sums[segment_ids],
        0.0,
    ) # (L,)

    return normalised



