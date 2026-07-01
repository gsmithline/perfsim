import json
import pickle as pkl
from pathlib import Path

import numpy as np
import jax.numpy as jnp


def load_dataset(dataset: str, **kwargs):
    base_dir = Path(__file__).resolve().parent.parent.parent
    if dataset == "congress_twitter":
        return load_congress_twitter(base_dir, **kwargs)
    elif dataset == "bluesky":
        return load_bluesky(base_dir)
    elif dataset == "bluesky_small":
        return load_bluesky_small(base_dir)
    elif dataset == "hadza_out_camp":
        return load_hadza_out_camp(base_dir)
    else:
        raise ValueError(f"The dataset {dataset} does not exist.")

def load_congress_twitter(base_dir):
    path = base_dir / "datasets/congress/congress_network/adj_matrix.csv"
    adj_matrix = np.loadtxt(path, delimiter=",", dtype=np.int8)
    adj_matrix = jnp.array(adj_matrix)
    return adj_matrix

def load_bluesky(base_dir):
    path = base_dir / "datasets/bluesky/ml/adj_matrix.csv"
    adj_matrix = np.loadtxt(path, delimiter=",", dtype=np.int8)
    adj_matrix = jnp.array(adj_matrix)
    return adj_matrix

def load_bluesky_small(base_dir):
    path = base_dir / "datasets/bluesky/ml_small/adj_matrix.csv"
    adj_matrix = np.loadtxt(path, delimiter=",", dtype=np.int8)
    adj_matrix = jnp.array(adj_matrix)
    return adj_matrix

def load_hadza_out_camp(base_dir):
    path = base_dir / f"datasets/hadza/camp2/out_camp/adj_matrix.csv"
    path_prob_of_interaction = base_dir / f"datasets/hadza/camp2/out_camp/prob_of_interaction.csv"
    adj_matrix = np.loadtxt(path, delimiter=",", dtype=np.int8)
    adj_matrix = jnp.array(adj_matrix)
    prob_of_interaction = np.loadtxt(path_prob_of_interaction, delimiter=",", dtype=np.float32)
    prob_of_interaction = jnp.array(prob_of_interaction)
    aux = {"q": prob_of_interaction}
    return adj_matrix, aux
