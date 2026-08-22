"""Wu-replication CONTEXT / ICL construction (2026-08-22).

Sibling of _gated_pop.py: deliberately transformers-free and importable
on a laptop, so every guarantee below is testable without loading a
single model weight.

THREE MECHANISMS, NEVER ONE WORD
--------------------------------
Wu's platform is a FROZEN predictor that is given context. There are
three different things people call "context" here and collapsing them
is the failure this module exists to prevent, because two of them are
faithful to Wu's observation model and one of them is not:

  observed_context     STRICT. K demonstrations drawn ONLY from the
                       OBSERVED set O, each rendered as (that agent's
                       profile -> that agent's CURRENT opinion x_j(t)).
                       history_source = "observed_peer".
                       The platform genuinely observes O, so showing
                       O's labels is inside Wu's information set.

  prediction_history   STRICT. The held-out agent's OWN past PLATFORM
                       PREDICTIONS -- what the platform previously
                       SERVED to it, m_i(t-1), m_i(t-2), ... depth D.
                       history_source = "platform_prediction".
                       This is the platform remembering what IT said.
                       It reveals nothing the platform did not already
                       know: it is a function of the platform's own
                       outputs.

  expressed_history    EXTENSION -- NOT Wu. The held-out agent's OWN
                       past POST-FJ OPINIONS x_i(t-1), x_i(t-2), ...
                       history_source = "post_fj_opinion".
                       Wu's platform NEVER observes a held-out agent's
                       opinion; that is the entire point of holding it
                       out. Feeding it back is a different observation
                       model, so every log line and config row this mode
                       produces carries wu_icl_extension = True.

The mode strings are separate, the history_source strings are separate,
and is_extension() is the single place that says which is which. A run
cannot accidentally get expressed_history by asking for "history".

NESTED SELECTION
----------------
select_observed_demos draws ONE permutation of the eligible pool per
(agent, round) from a dedicated stream whose seed does NOT depend on K,
then takes the first K. So the K=8 demonstrations are a PREFIX of the
K=32 demonstrations for the same agent and round: a K sweep varies the
DOSE of the same evidence, never the identity of the evidence. If the
seed depended on K, a K=8-vs-K=32 difference could be "different people"
rather than "more people", and the sweep would measure nothing.

D = 0 and K = 0 render the EMPTY string, which is the required frozen
no-memory baseline: build_prompt treats "" exactly like None, so the
prompt bytes are those of the plain zero-shot prompt.

AUDIT
-----
audit_entry() re-derives, from the artifact alone, that a strict entry
used only observed ids, never the target, and displayed exactly the
values those observed agents actually held -- and that the rendered TEXT
contains no number beyond the logged ones. That last check is what makes
a held-out label detectable: a 2-decimal opinion value collides with
somebody by chance, so scanning the text for "some held-out agent's
value" alone is vacuous. Pinning text numbers == logged values, logged
ids subset of O, and logged values == x_O[ids] is not.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import torch

# ------------------------------------------------------------------ modes

MODES = ("none", "observed_context", "prediction_history", "expressed_history")

#: modes whose information set is inside Wu's platform observation model
STRICT_MODES = ("none", "observed_context", "prediction_history")

#: modes that reveal something Wu's platform does not observe
EXTENSION_MODES = ("expressed_history",)

#: what the displayed numbers ARE, per mode. Never "history" or "context":
#: the whole point is that these three are different quantities.
HISTORY_SOURCE = {
    "none": None,
    "observed_context": "observed_peer",
    "prediction_history": "platform_prediction",
    "expressed_history": "post_fj_opinion",
}

HISTORY_SOURCES = ("platform_prediction", "post_fj_opinion", "observed_peer")

#: dedicated RNG stream id. Distinct from every other stream in the
#: runner (70000 legacy ICL, 424243 population, 777331 SFT subsample,
#: 202608 replay, 52000/52100 permute, 9700/31000 profile dials) so a
#: context draw can never shadow or be shadowed by a simulation draw.
WU_CTX_STREAM = 90210

#: displayed numeric format, shared by the renderers and the audit
VALUE_FMT = "{:.2f}"

_NUM_RE = re.compile(r"\d+\.\d\d")


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(
            f"WU_ICL_MODE must be one of {MODES}; got {mode!r}")
    return mode


def is_extension(mode: str) -> bool:
    """True iff `mode` shows the model something Wu's platform cannot see.

    Stated once, here, so no caller has to remember which of the three
    mechanisms is the extension.
    """
    return validate_mode(mode) in EXTENSION_MODES


def is_strict(mode: str) -> bool:
    return validate_mode(mode) in STRICT_MODES


def mode_needs_k(mode: str) -> bool:
    return mode == "observed_context"


def mode_needs_d(mode: str) -> bool:
    return mode in ("prediction_history", "expressed_history")


# -------------------------------------------------------------- selection

def select_observed_demos(agent, observed_ids, k, *, seed, round_t):
    """K demonstration ids for `agent`, drawn ONLY from `observed_ids`.

    NESTED IN K by construction: the permutation is seeded by
    (stream, seed, round, agent) -- NOT by k -- so the length-8 prefix of
    the length-32 draw is the length-8 draw. Returns an int64 ndarray.

    The target is removed from the pool before the draw, so it can never
    demonstrate itself. Under the passthrough design the targets are the
    HELD-OUT agents and are absent from `observed_ids` anyway; the guard
    is kept because this function is also usable when the target is
    observed, and a self-demonstration would leak the answer.
    """
    pool = np.asarray(observed_ids, dtype=np.int64)
    pool = pool[pool != int(agent)]
    k = int(k)
    if k <= 0 or pool.size == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(
        [WU_CTX_STREAM, int(seed), int(round_t), int(agent)])
    return rng.permutation(pool)[:min(k, pool.size)].astype(np.int64)


def assert_selection_safe(ids, observed_ids, agent):
    """Runtime guard: demonstrations are observed agents, never the target.

    Cheap (set arithmetic on <= K ids) so it runs every agent every
    round rather than being a thing the checker might get to later.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return
    obs = set(int(x) for x in np.asarray(observed_ids, dtype=np.int64))
    bad = [int(x) for x in ids if int(x) not in obs]
    if bad:
        raise ValueError(
            f"wu_context: agent {int(agent)} drew demonstration ids {bad} "
            f"that are NOT in the observed set -- a strict context would "
            f"be showing held-out information")
    if int(agent) in set(int(x) for x in ids):
        raise ValueError(
            f"wu_context: agent {int(agent)} appears in its own "
            f"demonstrations -- that is the label, not context")


# -------------------------------------------------------------- rendering

def _isnan(v) -> bool:
    try:
        return v != v
    except Exception:
        return False


def pokec_profile_bits(row, *, alcohol_translator=None) -> str:
    """Compact one-line Pokec profile for a demonstration.

    Same three columns the zero-shot prompt shows (age, gender,
    relation_to_alcohol) and the SAME alcohol translation -- injected
    rather than re-implemented, so a demonstration can never describe a
    user differently from the way the target's own prompt would.
    """
    bits = []
    age = row.get("age") if hasattr(row, "get") else None
    if age is not None and not _isnan(age):
        try:
            if float(age) > 0:
                bits.append(f"age {int(float(age))}")
        except (TypeError, ValueError):
            pass
    gender = row.get("gender") if hasattr(row, "get") else None
    if gender is not None and not _isnan(gender) and str(gender) != "":
        try:
            bits.append({0.0: "female", 1.0: "male"}.get(
                float(gender), "unknown gender"))
        except (TypeError, ValueError):
            bits.append(str(gender))
    alc = row.get("relation_to_alcohol") if hasattr(row, "get") else None
    if alc is not None and not _isnan(alc) and str(alc) not in ("", "nan"):
        bits.append(alcohol_translator(alc) if alcohol_translator
                    else str(alc))
    return ", ".join(bits) if bits else "(no profile info)"


def render_observed_block(ids, values, profile_fn) -> str:
    """The observed_context demonstration block.

    profile_fn(agent_id) -> the compact profile string for that agent.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return ""
    lines = [
        f"- {profile_fn(int(a))} -> attitude {VALUE_FMT.format(float(v))}"
        for a, v in zip(ids, values)
    ]
    return ("Here are the current attitudes toward smoking of some other "
            "users the platform observes:\n" + "\n".join(lines))


def render_history_block(values, *, source) -> str:
    """The personal-memory block for the two history mechanisms.

    The two sentences are DIFFERENT ON PURPOSE. "what the platform
    previously showed you" and "your own recorded attitude" are not the
    same claim, and a reader of a saved prompt must be able to tell which
    experiment produced it from the prompt text alone.
    """
    vals = [float(v) for v in values]
    if not vals:
        return ""
    shown = ", ".join(VALUE_FMT.format(v) for v in vals)
    if source == "platform_prediction":
        return ("Attitude scores the platform previously showed this user, "
                f"oldest to newest: {shown}.")
    if source == "post_fj_opinion":
        return ("This user's own recorded attitudes toward smoking over the "
                f"most recent rounds, oldest to newest: {shown}.")
    raise ValueError(f"unknown history source {source!r}")


# ------------------------------------------------------------ the builder

def _tail(history, agent, depth):
    """Last `depth` values of `agent`'s own series, oldest to newest."""
    depth = int(depth)
    if depth <= 0 or not history:
        return []
    return [float(h[int(agent)]) for h in history[-depth:]]


def build_context(mode, agent, *, observed_ids=(), opinion=None, k=0, d=0,
                  pred_history=None, expr_history=None, profile_fn=None,
                  seed=0, round_t=0):
    """Build one agent's context block and its audit log entry.

    Returns (text, entry). `text` is "" for mode "none" and for the
    K=0 / D=0 baselines -- build_prompt treats "" like None, so those
    runs carry the plain zero-shot prompt bytes.

    entry keys (the per-agent record written to wu_ctx_log.json.gz):
        agent, ids, values, text, history_source, mode, extension
    """
    validate_mode(mode)
    entry = {
        "agent": int(agent),
        "ids": [],
        "values": [],
        "text": "",
        "history_source": HISTORY_SOURCE[mode],
        "mode": mode,
        # carried on EVERY row, including the strict ones (as False), so
        # "flag absent" can never be read as "flag false"
        "extension": bool(is_extension(mode)),
    }
    if mode == "none":
        return "", entry

    if mode == "observed_context":
        if profile_fn is None:
            raise ValueError("observed_context needs profile_fn")
        if opinion is None:
            raise ValueError("observed_context needs the current opinion "
                             "vector (demonstrations show x_j(t))")
        ids = select_observed_demos(agent, observed_ids, k,
                                    seed=seed, round_t=round_t)
        assert_selection_safe(ids, observed_ids, agent)
        op = torch.as_tensor(opinion).detach().cpu().float()
        vals = [float(op[int(a)]) for a in ids]
        text = render_observed_block(ids, vals, profile_fn)
        entry["ids"] = [int(a) for a in ids]
        entry["values"] = vals
        entry["text"] = text
        return text, entry

    hist = pred_history if mode == "prediction_history" else expr_history
    vals = _tail(hist, agent, d)
    text = render_history_block(vals, source=HISTORY_SOURCE[mode])
    # the id "used" for personal memory is the agent itself, once per
    # displayed value -- so the log answers "whose numbers are these"
    # with the same shape for all three mechanisms
    entry["ids"] = [int(agent)] * len(vals)
    entry["values"] = vals
    entry["text"] = text
    return text, entry


def round_log_line(round_t, mode, entries, *, k=0, d=0):
    """One gzip-JSONL line for wu_ctx_log.json.gz."""
    return {
        "round": int(round_t),
        "mode": mode,
        "wu_icl_k": int(k),
        "wu_icl_d": int(d),
        "history_source": HISTORY_SOURCE[mode],
        # round-level copy of the per-agent flag: an analyst grepping the
        # log for extension runs must not have to open the agent list
        "wu_icl_extension": bool(is_extension(mode)),
        "agents": entries,
    }


# ----------------------------------------------------------------- audit

def text_values(text):
    """Every 2-decimal number the rendered block displays, in order."""
    return [float(m) for m in _NUM_RE.findall(text or "")]


def audit_entry(entry, *, observed_ids=(), opinion=None, tol=1e-6):
    """Re-derive a log entry's safety from the artifact. Returns a list of
    violation strings (empty == clean).

    For strict modes this is a PROOF, not a heuristic:
      1. every id is observed and is not the target        (no held-out id)
      2. displayed values equal x_O[ids] at that round     (no swapped label)
      3. the rendered TEXT displays exactly the logged values, in order
         (nothing extra smuggled into the prose)
    (1)+(2)+(3) together leave no room for a held-out label in the text.
    Scanning the text for "some held-out agent's value" instead would be
    vacuous: at 2 decimals almost every value collides with somebody.
    """
    bad = []
    mode = entry.get("mode")
    if mode not in MODES:
        return [f"unknown mode {mode!r}"]
    src = entry.get("history_source")
    if src != HISTORY_SOURCE[mode]:
        bad.append(f"history_source {src!r} does not match mode {mode!r}")
    if bool(entry.get("extension")) != is_extension(mode):
        bad.append(f"extension flag {entry.get('extension')!r} does not "
                   f"match mode {mode!r}")
    ids = [int(a) for a in entry.get("ids", [])]
    vals = [float(v) for v in entry.get("values", [])]
    if len(ids) != len(vals):
        bad.append(f"{len(ids)} ids vs {len(vals)} values")
    tvals = text_values(entry.get("text", ""))
    if len(tvals) != len(vals) or any(
            abs(a - b) > 5e-3 for a, b in zip(tvals, vals)):
        bad.append(f"rendered text shows {tvals} but the log records {vals}")
    if mode == "observed_context":
        obs = set(int(x) for x in np.asarray(observed_ids, dtype=np.int64))
        leaked = [a for a in ids if a not in obs]
        if leaked:
            bad.append(f"HELD-OUT ids in a strict context: {leaked}")
        if int(entry.get("agent", -1)) in set(ids):
            bad.append("target agent demonstrates itself")
        if len(set(ids)) != len(ids):
            bad.append("duplicate demonstration ids")
        if opinion is not None:
            op = torch.as_tensor(opinion).detach().cpu().float()
            for a, v in zip(ids, vals):
                if abs(float(op[a]) - v) > tol:
                    bad.append(
                        f"displayed value {v} for agent {a} is not that "
                        f"agent's opinion {float(op[a])}")
    elif mode in ("prediction_history", "expressed_history"):
        wrong = [a for a in ids if a != int(entry.get("agent", -1))]
        if wrong:
            bad.append(f"personal memory carrying OTHER agents' ids: {wrong}")
    return bad


def nested_prefix_ok(agent, observed_ids, k_small, k_big, *, seed, round_t):
    """True iff the K=k_small draw is a prefix of the K=k_big draw."""
    a = select_observed_demos(agent, observed_ids, k_small,
                              seed=seed, round_t=round_t)
    b = select_observed_demos(agent, observed_ids, k_big,
                              seed=seed, round_t=round_t)
    return len(a) <= len(b) and list(a) == list(b[:len(a)])


def idx_sha256(idx) -> str:
    """sha256 of a sorted int64 index set -- routing-treatment provenance."""
    a = np.sort(np.asarray(idx, dtype=np.int64))
    return hashlib.sha256(a.tobytes()).hexdigest()


# ------------------------------------------------- passthrough assembly

def served_vector(x_current, model_pred, observed_mask):
    """s(t) = x_O(t) on the observed set, the model's output on held-out.

    The one line the whole passthrough design reduces to, kept here so
    the runner, the tests and any checker share ONE definition. Returns a
    fresh float32 tensor; `model_pred` may be NaN on O (the model was
    never asked there) and those NaNs must NOT survive.
    """
    x = torch.as_tensor(x_current).detach().cpu().float().reshape(-1)
    m = torch.as_tensor(model_pred).detach().cpu().float().reshape(-1)
    obs = torch.as_tensor(observed_mask).detach().cpu().bool().reshape(-1)
    if not (x.shape == m.shape == obs.shape):
        raise ValueError(
            f"served_vector shape mismatch: x{tuple(x.shape)} "
            f"m{tuple(m.shape)} mask{tuple(obs.shape)}")
    s = torch.where(obs, x, m)
    if not torch.isfinite(s).all():
        n_bad = int((~torch.isfinite(s)).sum())
        raise ValueError(
            f"served_vector: {n_bad} non-finite entries -- the model was "
            f"not asked for every held-out agent")
    return s
