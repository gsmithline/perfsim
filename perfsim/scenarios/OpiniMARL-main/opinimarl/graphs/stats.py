import jax.numpy as jnp

def local_normalised_in_degree(adj):
    """
    Calculates the In-Degree, normalised by the 2-hop neighbourhood.
    
    Args:
        adj: Directed adjacency matrix (n, n). 
           adj[i, j] = 1 means Agent i pays attention to Agent j.
           
    Returns:
        norm_in_degree: Vector (n,) of scale-invariant structural features.
    """
    n = adj.shape[0]

    d_in = jnp.sum(adj, axis=0)
    adj_sq = adj @ adj
    reachability = adj + adj_sq # Limit to 2-hop neighbourhood

    local_mask = (reachability + jnp.eye(n)) > 0
    n_local = jnp.sum(local_mask, axis=0)

    norm_in_degree = d_in / jnp.where(n_local != 0, n_local, 1.0)

    return norm_in_degree

def attentional_reach(adj, k=4):
    """
    Calculates the 'Attentional Reach' (Structural Prestige) of each node.
    
    This identifies 'Information Sources' or 'Authorities' by measuring how much 
    global attention flows into a node over k steps. It captures nodes that are 
    destinations for information chains, even if they have low out-degree.
    
    The result is scale-invariant: a value of 1.0 represents the 'average' reach, 
    while values > 1.0 indicate nodes that attract more attention than the mean.

    Args:
        adj: Directed adjacency matrix (n, n). 
             adj[i, j] = 1 means Agent i pays attention to Agent j.
        k: The number of attention steps (default 4). 
           Captures the local-to-intermediate 'social horizon'.

    Returns:
        scale_invariant_reach: A (n,) JAX array where each entry is the scale-invariant reach score.
    """
    n = adj.shape[0]

    row_sums = adj.sum(axis=1)
    normalised_adj = adj / jnp.where(row_sums != 0, row_sums, 1.0)

    reach = jnp.ones(n) / n

    for _ in range(k):
        reach = reach @ normalised_adj

    scale_invariant_reach = reach * n

    return scale_invariant_reach

