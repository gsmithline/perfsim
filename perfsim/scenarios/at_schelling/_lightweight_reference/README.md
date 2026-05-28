# `perfsim.scenarios.at_schelling`

Schelling-style segregation as a clean LM-driven performative-prediction
loop. The LM predicts per-agent happiness via binary token logits, the
prediction modulates each agent's Schelling move threshold, the runner
relocates unhappy agents, and the realized post-move happiness is the
SFT target for the next round.

## Mental model

```
round t:
  prompt_i  = build_prompt(profile_i, neighborhood_i, previous_state_i)
  p_i       = softmax( logit(HAPPY), logit(UNHAPPY) )[0]   # one LM fwd
  H_i       = clip(H_0 - lambda * (p_i - 0.5), 0, 1)       # modulate
  move_i    = 1[ same_frac_i < H_i ]                        # decide
  relocate movers to random empty cells
  s_i_new   = same_frac after moves
  y_i       = 1[ s_i_new >= H_0 ]                          # baseline
                                                            # (NOT H_i)
SFT round t+1 LM on (prompt_i, "HAPPY" or "UNHAPPY") by y_i.
```

H_0 is the baseline threshold (default 0.4). lambda is the LM
modulation strength (default 0.15). Realized happiness is judged at H_0
on purpose: the LM's job is to predict the BASELINE happiness label,
not its own modulated label.

## Quickstart

```python
from perfsim.scenarios.at_schelling import make_schelling_env, BinaryLMScorer
from perfsim.models.hf_causal_lm import HFCausalLMModel
from perfsim.simulator import Simulator
from perfsim.learners.lm.sft import SFTLearner
from perfsim.losses import MSELoss
import pandas as pd


def schelling_prompt(profile, tokenizer) -> str:
    return (
        f"Agent type: {profile['type_name']}.\n"
        f"Same-type neighbor fraction: {profile['same_frac']:.2f}.\n"
        f"Opposite-type neighbor fraction: {profile['opp_frac']:.2f}.\n"
        f"Empty-neighbor fraction: {profile['empty_frac']:.2f}.\n"
        f"Baseline happiness threshold: 0.40.\n"
        f"Previous state: {profile['previous_state']}.\n"
        f"After one Schelling update, will this agent be happy?\n"
        f"Answer with one token: HAPPY or UNHAPPY.\n"
        f"Answer:"
    )


def happy_or_unhappy(y: float) -> str:
    return "HAPPY" if float(y) >= 0.5 else "UNHAPPY"


env = make_schelling_env(
    num_agents=200,
    grid_height=20,
    grid_width=20,
    baseline_threshold=0.4,
    lambda_=0.15,
    num_steps_per_episode=5,
)

# Build profiles DataFrame from the runner's initial state.
state = env.runner.state["agents"]["residents"]
profiles = pd.DataFrame({
    "type_name": [
        ["White", "Black", "Hispanic", "Asian"][int(t)]
        for t in state["type"].tolist()
    ],
    "same_frac": state["same_frac"].tolist(),
    "opp_frac": state["opp_frac"].tolist(),
    "empty_frac": state["empty_frac"].tolist(),
    "previous_state": ["unknown"] * 200,
})

lm = HFCausalLMModel(
    base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    profiles=profiles,
    prompt_builder=schelling_prompt,
    use_lora=True,
)
scorer = BinaryLMScorer(lm, yes_token="HAPPY", no_token="UNHAPPY")

learner = SFTLearner(
    lm,                    # learner trains the underlying HFCausalLMModel
    MSELoss(),             # loss is ignored by SFTLearner (uses CE); plumbing
    target_formatter=happy_or_unhappy,
    max_steps=20,
    response_template="Answer:",
)

# The Simulator deploys `scorer` (the model the env queries) but trains
# `lm` (the wrapped HF model whose params SFT updates). Since `scorer` is
# a thin wrapper that always reads `lm.inner_model`, this works.
sim = Simulator(env=env, learner=learner, loss=MSELoss())
hist = sim.run(n_rounds=3, epoch_size=1)
```

The `Simulator` passes `learner.model` to the env as the deployed
handle. If you want the env to call `BinaryLMScorer.forward`, set
`learner.model = scorer` after construction, or write a small wrapper
predictor (see `perfsim/core/predictor.py`).

## Wiring profiles to the live state

The profiles DataFrame is stale after the first move: an agent's
neighborhood changes when it moves. For correct prompts on round t+1
you need to rebuild profiles from `env.runner.state` after each round.

Two patterns:

1. **Override the prompt builder** so it reads the live tensors:

```python
def schelling_prompt(agent_idx, tokenizer, runner=env.runner) -> str:
    r = runner.state["agents"]["residents"]
    i = int(agent_idx)
    same = float(r["same_frac"][i]); opp = float(r["opp_frac"][i])
    ...
```

   Then set `lm._profiles = list(range(N))` so `profile_at(i)` returns
   `i` and the prompt builder pulls from the runner.

2. **Run an on_round callback** that rebuilds the DataFrame from
   `env.runner.state` and assigns `lm._profiles = new_df` before round
   t+1's SFT call.

Pattern 1 is simpler; pattern 2 keeps the LM-side code unaware of the
runner.

## KL-anchored SFT

To match the user's PP setup, replace `SFTLearner` with
`KLSFTLearner(ref_model_name="Qwen/Qwen2.5-0.5B-Instruct", kl_beta=1.0)`.
The KL anchor is on the LM's full token distribution, computed against
a frozen copy of the pretrained reference. The Schelling target text
("HAPPY" / "UNHAPPY") is treated as the SFT label; KL acts on every
non-padded token in the rendered (prompt + label) sequence.

## What's hardcoded vs config-driven

Config-driven (`build_schelling_config` kwargs):
  - num_agents, grid_height, grid_width
  - n_types (number of demographic categories)
  - baseline_threshold (H_0), lambda_, neighborhood_radius
  - num_steps_per_episode, move_hardness, device, seed

Hardcoded for Stage 1:
  - 4-type ACS proportions in `data.NYC_ACS_PROPORTIONS_4TYPE`.
    Override by passing `proportions=` to `make_schelling_env`.
  - Empty-cell relocation strategy (random pick from the empty pool).
  - 8-neighbor Moore neighborhood (radius 1). Setting
    `neighborhood_radius=2` gives a 24-neighbor structure (5x5 minus
    self), but the substep formula handles any r >= 1.

## Stage 2 TODO

1. **Real NYC ACS data loading**: per-tract demographic counts from
   bundled `agent_torch.populations.NYC` pickles. Grid would become
   one cell per tract, not a flat 20x20 lattice.
2. **Group prompting**: bin agents by (type, same_frac_bin,
   opp_frac_bin, empty_frac_bin, previous_state) and share one LM
   forward per bin. The `BinaryLMScorer.score_binary` interface already
   accepts a list of prompts, so this is a small wrapper.
3. **Differentiable move execution**: replace the empty-cell sampling
   with a soft permutation over candidate cells weighted by
   `compare_soft(empty_frac, threshold)`. Needed for `grad_run` to
   produce non-zero gradient through the move step.
4. **KL-SFT integration test**: end-to-end run with KLSFTLearner +
   bifurcation sweep over beta in {0, 1, 3, 10}, parallel to the
   pokec / EPG-LLM sweeps.

## See also

- `perfsim/scenarios/at_covid/README.md` -- analogous wiring for covid.
- `perfsim/adapters/agenttorch.py` -- adapter contract.
- `STAGE_1_REPORT.md` (in this dir) -- honest verification log.
