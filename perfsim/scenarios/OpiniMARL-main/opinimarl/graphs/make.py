from .datasets import load_dataset
from .generate import generate


def make_graph(graph, num_nodes=None, **kwargs):
    if graph in ["bluesky", "bluesky_small", "congress_twitter", "hadza_out_camp"]:
        graph_out =  load_dataset(graph, **kwargs)
    else:
        graph_out = generate(graph, num_nodes, **kwargs)

    if isinstance(graph_out, tuple):
        if len(graph_out) != 2:
            raise ValueError(f"Expected a tuple of length 2, but got length {len(graph_out)}")
        return graph_out
    else:
        return (graph_out, {})




