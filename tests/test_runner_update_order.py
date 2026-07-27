"""Structural guard on run_pokec_gated_lm's AB round: mixture once, peers last.

The runner imports transformers, so it cannot be executed here (and must not
load a model on a laptop). These checks parse its source instead, which is
enough to catch the regression that matters: the platform/innate mixture
drifting back inside the peer-sweep loop, or peers being reordered ahead of it.
"""
import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, os.pardir, "experiments", "scripts",
                   "cluster_pipelines", "run_pokec_gated_lm.py")


def _tree():
    with open(SRC) as fh:
        return ast.parse(fh.read()), fh


def _calls(node, name):
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Attribute) and n.func.attr == name)
                 or (isinstance(n.func, ast.Name) and n.func.id == name))]


def _sweep_loops(tree):
    """for sw in range(ab_sweeps): ... loops."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.For) and isinstance(n.iter, ast.Call) \
                and isinstance(n.iter.func, ast.Name) and n.iter.func.id == "range" \
                and n.iter.args and isinstance(n.iter.args[0], ast.Name) \
                and n.iter.args[0].id == "ab_sweeps":
            out.append(n)
    return out


def test_mixture_called_once_and_outside_the_sweep_loop():
    tree, _ = _tree()
    calls = _calls(tree, "nested_presocial_update")
    assert len(calls) == 1, \
        f"expected exactly one nested_presocial_update call, found {len(calls)}"
    loops = _sweep_loops(tree)
    assert loops, "no `for sw in range(ab_sweeps)` loop found -- source moved?"
    inside = [c for lp in loops for c in _calls(lp, "nested_presocial_update")]
    assert not inside, \
        "the platform/innate mixture must happen ONCE per round, not per sweep"


def test_peer_sweeps_run_inside_the_sweep_loop():
    tree, _ = _tree()
    loops = _sweep_loops(tree)
    inside = [c for lp in loops for c in _calls(lp, "ab_sweep")]
    assert inside, "ab_sweep must run inside the per-sweep loop"


def test_no_legacy_reordering_or_post_platform_anchor():
    with open(SRC) as fh:
        src = fh.read()
    assert 'pop_order == "platform_first"' not in src, \
        "POP_ORDER must no longer reorder peers ahead of the mixture"
    # the legacy anchor was an unconditional re-anchor over everyone applied
    # after the blend; its exact form must be gone from the runner
    assert "(1.0 - innate_lambda) * ab_x + innate_lambda * ab_innate" not in src, \
        "legacy post-platform innate re-anchor still present"


def test_config_records_the_population_update_marker():
    with open(SRC) as fh:
        src = fh.read()
    assert '"population_update": "nested_ai_then_social_v1"' in src
