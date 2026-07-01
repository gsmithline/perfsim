from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.random as jr


def generate(graph, num_nodes, **kwargs):
    if graph == "random":
        return random(num_nodes, **kwargs)
    elif graph == "ring":
        return ring(num_nodes, **kwargs)
    elif graph == "star":
        return star(num_nodes, **kwargs)
    elif graph == "barabasi_albert":
        return barabasi_albert(num_nodes, **kwargs)
    elif graph == "directed_barabasi_albert":
        return directed_barabasi_albert(num_nodes, **kwargs)
    elif graph == "holme_kim":
        return holme_kim(num_nodes, **kwargs)
    elif graph == "sbm":
        return sbm(num_nodes, **kwargs)
    elif graph == "watts_strogatz":
        return watts_strogatz(num_nodes, **kwargs)
    elif graph == "fixed_edge":
        return fixed_edge(num_nodes, **kwargs)
    elif graph == "custom":
        return custom(**kwargs)
    else:
        raise ValueError(f"Graph {graph} is not supported.")
    

@partial(jax.jit, static_argnums=[0])
def random(num_nodes, seed: int = 0, p: float = 0.5) -> chex.Array:
    """
    An undirected graph which sets the entries of the upper triange (excl. diagonal)
    randomly to either 0 or 1 according to p and then symmetrises.
    I.e. 50% chance of two nodes having an edge or not if p=0.5
    """
    if not isinstance(num_nodes, int):
        raise ValueError("Graph type random requires 'num_nodes' to be specified")
    key = jr.key(seed)
    shape = (num_nodes, num_nodes)
    random_matrix = jr.bernoulli(key, p, shape).astype(jnp.int32)
    adj_matrix = jnp.triu(random_matrix, k=1) + jnp.triu(random_matrix, k=1).T
    return adj_matrix

@partial(jax.jit, static_argnums=[0])
def ring(num_nodes) -> chex.Array:
    """"
    A ring graph
    """
    if not isinstance(num_nodes, int):
        raise ValueError("Graph type ring requires 'num_nodes' to be specified")
    shape = (num_nodes, num_nodes)
    irows, icols = jnp.indices(shape)
    triangle_matrix = jnp.where((icols - irows == 1) | (icols - irows == num_nodes -1), 1, 0).astype(jnp.int32)
    adj_matrix = triangle_matrix + triangle_matrix.T
    return adj_matrix

@partial(jax.jit, static_argnums=[0])
def star(num_nodes) -> chex.Array:
    """"
    A star graph
    """
    if not isinstance(num_nodes, int):
        raise ValueError("Graph type star requires 'num_nodes' to be specified")
    shape = (num_nodes, num_nodes)
    irows, icols = jnp.indices(shape)
    triangle_matrix = jnp.where((irows == 0) & (icols != 0), 1, 0).astype(jnp.int32)
    adj_matrix = triangle_matrix + triangle_matrix.T
    return adj_matrix

@partial(jax.jit, static_argnums=[0, 2])
def barabasi_albert(num_nodes, seed: int = 0, m: int = 1) -> chex.Array:
    """"
    Generates a Barabasi-Albert graph.
    m is the number of edges each new node attaches to existing nodes.
    """
    if not isinstance(num_nodes, int):
        raise ValueError("Graph type barabasi_albert requires 'num_nodes' to be specified")
    key = jr.key(seed)

    # Initialise adjacency matrix
    shape = (num_nodes, num_nodes)
    adj_matrix = jnp.zeros(shape)

    # Set initial m fully connected nodes
    adj_matrix = adj_matrix.at[:m, :m].set(1 - jnp.eye(m))

    def _add_node(carry, i):
        key, adj_matrix = carry
        key, _key = jr.split(key)

        degrees = jnp.sum(adj_matrix, axis=1)
        probs = degrees / degrees.sum()  # Probabilities of attaching to existing nodes
        selected_nodes = jr.choice(_key, jnp.arange(num_nodes), shape=(m,), p=probs, replace=False)

        # Update adjacency matrix and symmetrise
        adj_matrix = adj_matrix.at[i, selected_nodes].set(1)
        adj_matrix = adj_matrix.at[selected_nodes, i].set(1)

        return (key, adj_matrix), None
    
    (key, adj_matrix), _ = jax.lax.scan(_add_node, (key, adj_matrix), jnp.arange(m, num_nodes))
    adj_matrix = adj_matrix.astype(jnp.int32)

    return adj_matrix

@partial(jax.jit, static_argnums=[0, 2, 3])
def directed_barabasi_albert(num_nodes, seed: int = 0, m: int = 1, b: float = 0.2) -> chex.Array:
    """"
    Generates a Barabasi-Albert graph.
    m is the number of edges each new node attaches to existing nodes.
    """
    if not isinstance(num_nodes, int):
        raise ValueError("Graph type directed_barabasi_albert requires 'num_nodes' to be specified")
    key = jr.key(seed)

    # Initialise adjacency matrix
    shape = (num_nodes, num_nodes)
    adj_matrix = jnp.zeros(shape)

    # Set initial m fully connected nodes
    adj_matrix = adj_matrix.at[:m, :m].set(1 - jnp.eye(m))

    def _add_node(carry, i):
        key, adj_matrix = carry
        key, _key = jr.split(key)

        # Preferential attachment to in degrees
        in_degrees = jnp.sum(adj_matrix, axis=0)
        probs = in_degrees / in_degrees.sum()  # Probabilities of attaching to existing nodes
        selected_nodes = jr.choice(_key, jnp.arange(num_nodes), shape=(m,), p=probs, replace=False)

        # Reciprocation from attached nodes
        selected_nodes_in_degrees = in_degrees[selected_nodes]
        reciprocation_probs = 1 / (selected_nodes_in_degrees + 1)**b
        key, _key = jr.split(key)
        reciprocate = jr.uniform(_key, shape=(m,)) < reciprocation_probs

        # Update adjacency matrix
        adj_matrix = adj_matrix.at[i, selected_nodes].set(1)
        adj_matrix = adj_matrix.at[selected_nodes, i].set(reciprocate)

        return (key, adj_matrix), None
    
    (key, adj_matrix), _ = jax.lax.scan(_add_node, (key, adj_matrix), jnp.arange(m, num_nodes))
    adj_matrix = adj_matrix.astype(jnp.int32)

    return adj_matrix

@partial(jax.jit, static_argnums=[0, 2, 3])
def holme_kim(num_nodes, seed: int = 0, m: int = 1, p_t: float = 0.5) -> chex.Array:
    """
    Generates a Holme-Kim graph.
    m is the number of edges each new node attaches to existing nodes.
    p_t is the probability of performing a triad fromation step instead of a preferential attachment step.
    """
    if not isinstance(num_nodes, int):
        raise ValueError("Graph type holme_kim requires 'num_nodes' to be specified")
    key = jr.key(seed)

    # Initialise adjacency matrix
    shape = (num_nodes, num_nodes)
    adj_matrix = jnp.zeros(shape)

    # Set initial m fully connected nodes
    adj_matrix = adj_matrix.at[:m, :m].set(1 - jnp.eye(m))

    def _add_node(carry, i):
        key, adj_matrix = carry
        selected_node = -1 # -1 Represents no initial selected node

        def _single_attachment(attach_carry, _):
            key, adj_matrix, i, selected_node = attach_carry
            key, _key = jr.split(key)
            
            def _pref_attach(state):
                key, adj_matrix, i, selected_node = state
                key, _key = jr.split(key)
                degrees = jnp.where(
                    # Ensure new connection is to a node that i is not connected to already. (And avoid self-loops)
                    (jnp.sum(adj_matrix, axis=1) != 0) & (adj_matrix[i] == 0) & (jnp.arange(num_nodes) != i),
                    jnp.sum(adj_matrix, axis=1),
                    0,
                )
                probs = degrees / degrees.sum()  # Probabilities of attaching to exisitng nodes
                selected_node = jr.choice(_key, jnp.arange(num_nodes), p=probs)

                # Update adjacency matrix and symmetrise
                adj_matrix = adj_matrix.at[i, selected_node].set(1)
                adj_matrix = adj_matrix.at[selected_node, i].set(1)

                state = (key, adj_matrix, i, selected_node)
                return state

            def _triad_attach(state):
                key, adj_matrix, i, selected_node = state
                key, _key = jr.split(key)
                available_nodes = jnp.where(
                    # Ensure new connection is to neighbouring nodes of PA step that are not already connected. (And avoid self-loops)
                    (adj_matrix[selected_node] == 1) & (adj_matrix[i] == 0) & (jnp.arange(num_nodes) != i),
                    1,
                    0,
                )
                probs = available_nodes / available_nodes.sum()
                attach_node = jr.choice(_key, jnp.arange(num_nodes), p=probs)

                # Update adjacency matrix and symmetrise
                adj_matrix = adj_matrix.at[i, attach_node].set(1)
                adj_matrix = adj_matrix.at[attach_node, i].set(1)

                state = (key, adj_matrix, i, selected_node)
                return state
            
            attach_carry = jax.lax.cond(
                selected_node == -1,
                _pref_attach,
                lambda operand: jax.lax.cond(
                    jnp.any((adj_matrix[selected_node] == 1) & (adj_matrix[i] == 0) & (jnp.arange(num_nodes) != i)) & (jr.uniform(_key) < p_t),
                    _triad_attach,
                    _pref_attach,
                    operand=operand
                ),
                operand=(key, adj_matrix, i, selected_node),
            )

            return attach_carry, None

        attach_carry = (key, adj_matrix, i, selected_node)
        (key, adj_matrix, i, selected_node), _ = jax.lax.scan(_single_attachment, attach_carry, jnp.arange(m))
        
        return (key, adj_matrix), None
    
    (key, adj_matrix), _ = jax.lax.scan(_add_node, (key, adj_matrix), jnp.arange(m, num_nodes))
    adj_matrix = adj_matrix.astype(jnp.int32)

    return adj_matrix

@partial(jax.jit, static_argnums=[0, 2, 3, 4])
def sbm(num_nodes, seed: int = 0, num_communities: int = 5, p_in: float = 0.5, p_ext: float = 0.1) -> chex.Array:
    """
    An undirected stochastic block model graph.
    p_in is the probability of connecting within a community
    p_ext is the probability of connecting to an external community
    num_com is the number of communities
    """
    if not isinstance(num_nodes, int):
        raise ValueError("Graph type sbm requires 'num_nodes' to be specified")
    key = jr.key(seed)
    shape = (num_nodes, num_nodes)
    irows, icols = jnp.indices(shape)

    community_rows = irows // (num_nodes / num_communities)
    community_cols = icols // (num_nodes / num_communities)
    # community_rows = irows % num_communities
    # community_cols = icols % num_communities
    
    random_matrix = jnp.where(
        community_rows == community_cols,
        jr.bernoulli(key, p_in, shape).astype(jnp.int32),
        jr.bernoulli(key, p_ext, shape).astype(jnp.int32),
    )
    adj_matrix = jnp.triu(random_matrix, k=1) + jnp.triu(random_matrix, k=1).T
    return adj_matrix

@partial(jax.jit, static_argnums=[0, 2, 3])
def watts_strogatz(num_nodes, seed: int = 0, k: int = 4, p: float = 0.1) -> chex.Array:
    """"
    A Watts Strogatz small world graph
    """
    if not isinstance(num_nodes, int):
        raise ValueError("Graph type watts_strogatz requires 'num_nodes' to be specified")
    key = jr.key(seed)

    assert k % 2 == 0, "k must be even"

    shape = (num_nodes, num_nodes)
    irows, icols = jnp.indices(shape)
    triangle_matrix = jnp.where(
        ((icols - irows >= 1) & (icols - irows <= k/2)) | ((icols - irows <= num_nodes -1) & (icols - irows >= num_nodes - k/2)),
        1,
        0,
    ).astype(jnp.int32)
    adj_matrix = triangle_matrix + triangle_matrix.T

    def _rewire(carry, edge_id, num_nodes, p):
        key, adj_matrix = carry

        i = edge_id // num_nodes # Quotient gives row
        j = edge_id % num_nodes # Remainder gives column

        selected_row = adj_matrix[i]

        key, _key = jr.split(key)
        move_edge = jr.uniform(_key) < p

        available_rewirings = jnp.where(
            (selected_row == 0) & (jnp.arange(num_nodes) != i),
            1,
            0,
        )

        key, _key = jr.split(key)
        rewire_id = jax.lax.select(
            move_edge,
            jr.choice(_key, jnp.arange(num_nodes), shape=(), p=available_rewirings),
            j,
        )

        adj_matrix = adj_matrix.at[i, j].set(0)
        adj_matrix = adj_matrix.at[j, i].set(0)
        adj_matrix = adj_matrix.at[i, rewire_id].set(1)
        adj_matrix = adj_matrix.at[rewire_id, i].set(1)

        carry = (key, adj_matrix)

        return carry, None

    carry = (key, adj_matrix)

    triangle_matrix_flat = triangle_matrix.flatten()
    num_edge_ids = num_nodes * (k // 2)
    edge_ids = jnp.where(triangle_matrix_flat, size=num_edge_ids, fill_value=-1)[0]

    (_, adj_matrix), _ = jax.lax.scan(
        partial(_rewire, num_nodes=num_nodes, p=p),
        carry,
        edge_ids
    )

    return adj_matrix


@partial(jax.jit, static_argnums=[0, 2])
def fixed_edge(num_nodes, seed: int = 0, num_edges: int = 1) -> chex.Array:
    """
    An undirected graph with a fixed length of edges
    """
    if not isinstance(num_nodes, int):
        raise ValueError("Graph type fixed_edge requires 'num_nodes' to be specified")
    key = jr.key(seed)
    shape = (num_nodes, num_nodes)
    irows, icols = jnp.indices(shape)

    p = jnp.where((irows < icols), 1, 0).astype(jnp.int32).flatten()
    indices = jax.random.choice(key, jnp.arange(p.shape[0]), shape=(num_edges,), p=p, replace=False)

    flattened_matrix = jnp.zeros(p.shape, dtype=jnp.int32)
    flattened_matrix = flattened_matrix.at[indices].set(1)
    triangle_matrix = flattened_matrix.reshape(shape)
    adj_matrix = triangle_matrix + triangle_matrix.T

    return adj_matrix

def custom(adj_matrix: chex.Array) -> chex.Array:
    chex.assert_rank(adj_matrix, 2)
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {adj_matrix.shape}")
    if not jnp.all(jnp.diag(adj_matrix) == 0):
        raise ValueError("Adjacency matrix must have a zero diagonal.")
        
    return adj_matrix   
