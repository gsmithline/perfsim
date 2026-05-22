"""Single-config covid + LM + KL-SFT run for the condor sweep.

Reads run parameters from environment variables (set by `condor/run_one.sh`)
and executes one (training_style, kl_beta) configuration of the
AT covid sim driven by a HuggingFace causal LM predictor.

Outputs:
  $OUT_DIR/history.pt          torch-pickled list of per-round records
  $OUT_DIR/trajectory.json     per-round summary (daily_infected, theta_norm, ...)
  $OUT_DIR/config.json         resolved config for this run
  $OUT_DIR/stderr.log          stderr capture (also tee'd by condor)

Environment variables (with defaults):
  RUN_TAG              required. Identifies the run; used in OUT_DIR + wandb.
  KL_BETA              required. KL coefficient for KLSFTLearner (0.0 -> plain SFT).
  TRAINING_STYLE       "sft" or "sft_kl". Default "sft_kl".
  BASE_MODEL           HF model ID. Default "Qwen/Qwen2.5-0.5B-Instruct".
  N_ROUNDS             outer rounds. Default 5.
  K_STEPS              inner AT substeps per round. Default 3.
  SEED_FRAC            initial infected fraction. Default 0.05.
  SEED                 random seed. Default 0.
  DEVICE               "cuda" / "cpu". Default: auto.
  OUT_DIR              run output dir. Default runs/at_covid_lm/$RUN_TAG.
  WANDB_PROJECT        optional. If set, log to wandb.
  WANDB_KEY_FILE       optional path. run_one.sh exports WANDB_API_KEY from this.

Honest caveats:
  - perfsim's HFCausalLMModel + KLSFTLearner stack has NOT been smoke-tested
    end-to-end (macOS tokenizers hang on the development box). First cluster
    submit serves as the smoke test; expect to debug.
  - AT covid bundled population is fixed at 37,518 agents. No subsampling.
    Generating 37k LM completions per round is expensive (~5-10 min on A100
    with batched HF generate). Five rounds x five betas easily costs many
    hours of GPU time.
  - SFTTrainer with 37k examples is also non-trivial. max_steps is the
    primary cost knob; we default to 50.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def _env_or(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var {name!r} not set")
    return val


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def main() -> int:
    import torch
    import pandas as pd

    run_tag = _env_or("RUN_TAG")
    kl_beta = _env_float("KL_BETA", 0.0)
    training_style = _env_or("TRAINING_STYLE", "sft_kl")
    base_model = _env_or("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    n_rounds = _env_int("N_ROUNDS", 5)
    k_steps = _env_int("K_STEPS", 3)
    seed_frac = _env_float("SEED_FRAC", 0.05)
    seed = _env_int("SEED", 0)
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(os.environ.get("OUT_DIR", f"runs/at_covid_lm/{run_tag}"))
    wandb_project = os.environ.get("WANDB_PROJECT")
    max_steps = _env_int("SFT_MAX_STEPS", 50)
    gen_batch_size = _env_int("GEN_BATCH_SIZE", 64)
    max_new_tokens = _env_int("MAX_NEW_TOKENS", 8)
    sft_batch_size = _env_int("SFT_BATCH_SIZE", 32)
    sft_full_epoch = os.environ.get("SFT_FULL_EPOCH", "0").lower() in ("1", "true", "yes")
    target_kind = os.environ.get("TARGET_KIND", "exposed_binary")  # or "disease_stage"

    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_tag": run_tag,
        "kl_beta": kl_beta,
        "training_style": training_style,
        "base_model": base_model,
        "n_rounds": n_rounds,
        "k_steps": k_steps,
        "seed_frac": seed_frac,
        "seed": seed,
        "device": device,
        "max_steps": max_steps,
        "sft_batch_size": sft_batch_size,
        "sft_full_epoch": sft_full_epoch,
        "gen_batch_size": gen_batch_size,
        "max_new_tokens": max_new_tokens,
        "target_kind": target_kind,
        "host": os.uname().nodename,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[run] {json.dumps(config)}", flush=True)

    wandb = None
    if wandb_project:
        import wandb as _wandb

        wandb = _wandb
        wandb.init(project=wandb_project, name=run_tag, config=config)

    from perfsim.models.hf_causal_lm import HFCausalLMModel
    from perfsim.learners.lm.sft import SFTLearner
    from perfsim.learners.lm.kl_sft import KLSFTLearner
    from perfsim.losses import MSELoss
    from perfsim.scenarios.at_covid import make_covid_env
    from perfsim.simulator import Simulator

    torch.manual_seed(seed)
    print("[run] building covid env (~5s init for 37,518 agents)", flush=True)
    t0 = time.time()
    # Seed initial infections via the factory so Simulator.run's env.reset
    # does not wipe them between sim.run() calls. Without this, the
    # population stays effectively all-Susceptible and the LM's isolation
    # decisions have nothing to gate (daily_infected stays at the baseline
    # exposed count across all rounds and betas).
    env = make_covid_env(init_seed=seed, initial_infections_fraction=seed_frac)
    n_agents = env.runner.state["agents"]["citizens"]["age"].shape[0]
    citizens = env.runner.state["agents"]["citizens"]
    n_seeded = int((citizens["disease_stage"].squeeze() == 2.0).sum().item())
    print(f"[run] env ready: {n_agents} agents, {n_seeded} initially infected, "
          f"in {time.time() - t0:.1f}s", flush=True)

    # Build per-agent profile with the two features the bundled covid substep
    # actually uses for transmission: age bucket (SFSusceptibility multiplier)
    # and mean_interactions (per-agent contacts/day).
    ages = env.runner.state["agents"]["citizens"]["age"].squeeze().long().tolist()
    mean_int = env.runner.state["environment"]["mean_interactions"].squeeze().tolist()
    AGE_LABELS = ["under 18", "18-29", "30-44", "45-59", "60-74", "75+"]
    profiles = pd.DataFrame({
        "age_bucket": ages,
        "age_label": [AGE_LABELS[min(int(a), len(AGE_LABELS) - 1)] for a in ages],
        "mean_interactions": mean_int,
        "agent_id": list(range(n_agents)),
    })
    print(f"[run] profile features: age_buckets={sorted(set(ages))} "
          f"interactions={sorted(set(mean_int))}", flush=True)

    def prompt_builder(profile_row, tokenizer):  # noqa: ARG001
        return (
            f"You are an adult in the {profile_row['age_label']} age group "
            f"living in New York City. On a typical day you have close contact "
            f"with about {profile_row['mean_interactions']:.0f} other people. "
            f"An infectious respiratory disease is currently spreading. "
            f"Public health officials have recommended that residents isolate. "
            f"On a scale from 0 to 1, what is the probability you will choose "
            f"to isolate today? Reply with only a number between 0 and 1.\n"
            f"Answer: "
        )

    print(f"[run] loading LM: {base_model} on {device}", flush=True)
    t0 = time.time()
    model = HFCausalLMModel(
        base_model_name=base_model,
        profiles=profiles,
        prompt_builder=prompt_builder,
        use_lora=True,
        device=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        max_new_tokens=max_new_tokens,
        gen_batch_size=gen_batch_size,
        load_now=True,
    )
    print(f"[run] LM loaded in {time.time() - t0:.1f}s", flush=True)

    # Resolve max_steps. If SFT_FULL_EPOCH=1, override to one full epoch
    # over the 37k examples at the chosen batch size.
    if sft_full_epoch:
        effective_max_steps = -(-n_agents // sft_batch_size)  # ceil div
        print(
            f"[run] SFT_FULL_EPOCH=1: max_steps={effective_max_steps} "
            f"(= ceil({n_agents}/{sft_batch_size}))",
            flush=True,
        )
    else:
        effective_max_steps = max_steps

    # Override the default state_extractor if the user asked for the
    # binary-exposed target instead of disease_stage.
    if target_kind == "exposed_binary":
        def custom_state_extractor(runner):
            citizens = runner.state["agents"]["citizens"]
            age = citizens["age"].float().detach()
            exposed = (citizens["disease_stage"].squeeze() > 0).float().detach().reshape(-1, 1)
            return {
                "x": age,
                "y": exposed,
                "agent_idx": torch.arange(age.shape[0]),
            }
        # Hot-swap the extractor on the existing env.
        env._state_extractor = custom_state_extractor  # noqa: SLF001
        print("[run] target = (disease_stage > 0).float() (binary exposed)", flush=True)
    elif target_kind == "disease_stage":
        print("[run] target = disease_stage (float 0-4)", flush=True)
    else:
        raise ValueError(f"unknown TARGET_KIND: {target_kind!r}")

    loss = MSELoss()
    learner_kwargs = dict(
        model=model,
        loss=loss,
        max_steps=effective_max_steps,
        per_device_batch_size=sft_batch_size,
        output_dir=str(out_dir / "trl"),
    )
    if training_style == "sft":
        learner = SFTLearner(**learner_kwargs)
    elif training_style == "sft_kl":
        learner = KLSFTLearner(
            **learner_kwargs,
            ref_model_name=base_model,
            kl_beta=kl_beta,
        )
    else:
        raise ValueError(f"unknown TRAINING_STYLE: {training_style!r}")

    sim = Simulator(env=env, learner=learner, loss=loss)
    print(f"[run] starting outer loop: n_rounds={n_rounds} K={k_steps}", flush=True)
    t_loop = time.time()
    hist = sim.run(n_rounds=n_rounds, epoch_size=k_steps, seed=seed)
    print(f"[run] loop done in {time.time() - t_loop:.1f}s", flush=True)

    torch.save([dict(r) for r in hist.records], out_dir / "history.pt")

    trajectory = []
    for r in hist.records:
        theta = r.get("theta")
        di = env.runner.state["environment"]["daily_infected"]
        ds = env.runner.state["agents"]["citizens"]["disease_stage"]
        row = {
            "round": int(r["round"]),
            "theta_norm": float(theta.norm().item()) if hasattr(theta, "norm") else None,
            "daily_infected_sum": float(di.sum().item()),
            "fraction_non_S": float((ds > 0).float().mean().item()),
        }
        gap = r.get("stability_gap")
        if hasattr(gap, "item"):
            row["stability_gap"] = float(gap.item())
        trajectory.append(row)
        if wandb is not None:
            wandb.log(row)
        print(f"[round {row['round']}] {row}", flush=True)

    (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    print(f"[run] outputs in {out_dir}", flush=True)
    if wandb is not None:
        wandb.finish()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
