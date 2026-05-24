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
    # LoRA rank. perfsim default is r=8 (Hu et al. minimal); opinion-dyn
    # uses r=32. With r=8 + (q,v) on Qwen-0.5B we observed unconditional
    # token-bias collapse (LM emits "0" repeatedly regardless of position).
    # r=32 quadruples adapter capacity; matches the working opinion-dyn setup.
    lora_r = _env_int("LORA_R", 32)
    # Full fine-tuning vs LoRA. Qwen-0.5B fits trivially on a 24GB+ GPU
    # for full FT (~10GB for model + Adam state + activations). Set
    # USE_LORA=0 to test full FT, which gives direct gradient updates to
    # every parameter and bypasses any LoRA-capacity bottleneck.
    use_lora = _env_int("USE_LORA", 1) == 1
    # SFT learning rate. LoRA tolerates ~1e-5 to 5e-5; full FT typically
    # needs 10-100x smaller because LoRA's low-rank projection naturally
    # bounds per-parameter updates while full FT applies the LR to every
    # parameter directly. With LR=1e-5 we observed full FT collapse on
    # Qwen-0.5B after just 20 steps; 1e-6 is the gentler retry.
    sft_lr = _env_float("SFT_LR", 1e-5)

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
        # WANDB_RUN_SUFFIX lets parallel sweeps (e.g. full-FT vs LoRA) share
        # the same `configs_betas.txt` tags without colliding in wandb. The
        # condor `output=` paths separate the .out files; this separates
        # the wandb runs.
        wandb_suffix = os.environ.get("WANDB_RUN_SUFFIX", "")
        wandb_name = f"{run_tag}{wandb_suffix}" if wandb_suffix else run_tag
        wandb.init(project=wandb_project, name=wandb_name, config=config)

    from perfsim.models.hf_causal_lm import HFCausalLMModel
    from perfsim.learners.lm.sft import SFTLearner
    from perfsim.learners.lm.kl_sft import KLSFTLearner
    from perfsim.losses import MSELoss
    from perfsim.scenarios.at_covid import make_covid_env
    from perfsim.simulator import Simulator

    # Logit signal_writer: PerfsimIsolationDecision applies sigmoid() to
    # platform_signal. If we want `will_isolate` to equal the LM's emitted
    # value `p`, we must write logit(p) so sigmoid(logit(p)) = p. Without
    # this, an LM that emits {0.1, 0.9} only spans will_isolate=[0.525,
    # 0.711] after the substep's sigmoid; with it, will_isolate spans the
    # full [0.1, 0.9] and the env actually responds.
    def logit_signal_writer(runner, preds):
        if preds.ndim == 2 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        p = preds.clamp(min=0.01, max=0.99)
        logit_p = torch.log(p / (1.0 - p))
        runner.state["agents"]["citizens"]["platform_signal"] = logit_p.detach().clone()

    torch.manual_seed(seed)
    print("[run] building covid env (~5s init for 37,518 agents)", flush=True)
    t0 = time.time()
    # Seed initial infections via the factory so Simulator.run's env.reset
    # does not wipe them between sim.run() calls. Without this, the
    # population stays effectively all-Susceptible and the LM's isolation
    # decisions have nothing to gate (daily_infected stays at the baseline
    # exposed count across all rounds and betas).
    # keep_trajectory=True: AT runner state PERSISTS across env.run() calls.
    # Without this, env.run resets state to the initial seeded state every
    # round (only LM weights carry across rounds), so the "performative" loop
    # is degenerate: every round is an independent K-step rollout. With this
    # flag, the AT sim evolves continuously and round t starts from the end
    # state of round t-1, so the LM's earlier policy choices have cumulative
    # downstream effect (the actual performative-prediction semantics).
    env = make_covid_env(
        init_seed=seed,
        initial_infections_fraction=seed_frac,
        signal_writer=logit_signal_writer,
        keep_trajectory=True,
    )
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

    # Qwen2.5-Instruct was trained with chat templates. Running it on raw
    # completion text is off-distribution and triggers LoRA-SFT collapse.
    # We route ALL prompts through `tokenizer.apply_chat_template` so the
    # SFT-time prompt and the generation-time prompt are byte-identical
    # (avoids the train/gen format-mismatch trap we hit in opinion-dyn).
    SYSTEM_MSG = (
        "You are a public health official assigning personalized "
        "isolation recommendations during an active respiratory disease "
        "outbreak. Your goal is to minimize new transmissions."
    )

    def prompt_builder(profile_row, tokenizer):
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": (
                    f"Person: age group {profile_row['age_label']}, has about "
                    f"{profile_row['mean_interactions']:.0f} close contacts per "
                    f"day.\n\nOutput a single number between 0 and 1 (e.g. 0.50) "
                    f"where 0 means no isolation and 1 means full isolation."
                ),
            },
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    print(f"[run] loading LM: {base_model} on {device}", flush=True)
    t0 = time.time()
    model = HFCausalLMModel(
        base_model_name=base_model,
        profiles=profiles,
        prompt_builder=prompt_builder,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=2 * lora_r,
        device=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        max_new_tokens=max_new_tokens,
        gen_batch_size=gen_batch_size,
        load_now=True,
    )
    print(
        f"[run] LM training mode: {'LoRA r=' + str(lora_r) if use_lora else 'FULL fine-tuning'}",
        flush=True,
    )
    print(f"[run] LM loaded in {time.time() - t0:.1f}s", flush=True)

    # Diagnostic: dump (prompt, raw LM text, parsed value) for 20 random
    # agents BEFORE any SFT, so we see what the base model is actually
    # emitting and what the parser is doing with it. Crucial for debugging
    # the "LM signal collapses to 0 in env" issue.
    import random as _random
    _rng = _random.Random(seed)
    _sample_idx = _rng.sample(range(n_agents), min(20, n_agents))
    _sample_prompts = [model.build_prompt(model.profile_at(i)) for i in _sample_idx]
    print("[diag] sample LM outputs (pre-SFT):", flush=True)
    _sample_texts = model._generate(_sample_prompts)
    _sample_log = []
    for _idx, _prompt, _txt in zip(_sample_idx, _sample_prompts, _sample_texts):
        _parsed = model._parse(_txt)
        _sample_log.append({"agent_idx": int(_idx), "raw_text": _txt, "parsed": float(_parsed)})
        print(f"  agent {_idx}: text={_txt!r}  parsed={_parsed:.3f}", flush=True)
    (out_dir / "diagnostic_pre_sft.json").write_text(json.dumps(_sample_log, indent=2))

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
        env._state_extractor = custom_state_extractor  # noqa: SLF001
        print("[run] target = (disease_stage > 0).float() (binary exposed)", flush=True)
    elif target_kind == "risk_recommendation":
        # Recommend isolation level from per-agent risk features (age x
        # mean_interactions). Static across rounds; the SFT target is a real
        # policy the LM is being supervised on, not a prediction of outcome.
        # Three tiers: high (elderly OR high-contact) -> 0.9,
        # medium (middle-aged, not high-contact) -> 0.5,
        # low (young AND low-contact) -> 0.2.
        # KL beta now controls how strongly each fine-tune anchors back to
        # base Qwen vs commits to this policy.
        def custom_state_extractor(runner):
            citizens = runner.state["agents"]["citizens"]
            env_state = runner.state["environment"]
            age_long = citizens["age"].squeeze().long()
            mi = env_state["mean_interactions"].squeeze().float()
            high_risk = (age_long >= 4) | (mi >= 3.5)
            medium_risk = (age_long >= 2) & ~high_risk
            target = torch.where(
                high_risk,
                torch.tensor(0.9),
                torch.where(medium_risk, torch.tensor(0.5), torch.tensor(0.2)),
            ).float().reshape(-1, 1)
            return {
                "x": citizens["age"].float().detach(),
                "y": target.detach(),
                "agent_idx": torch.arange(target.shape[0]),
            }
        env._state_extractor = custom_state_extractor  # noqa: SLF001
        # Pre-compute label histogram so we know the class balance.
        _cit = env.runner.state["agents"]["citizens"]
        _envs = env.runner.state["environment"]
        _age = _cit["age"].squeeze().long()
        _mi = _envs["mean_interactions"].squeeze().float()
        _hr = ((_age >= 4) | (_mi >= 3.5)).sum().item()
        _mr = ((_age >= 2) & ~((_age >= 4) | (_mi >= 3.5))).sum().item()
        _lr = n_agents - _hr - _mr
        print(
            f"[run] target = risk_recommendation: high(0.9)={_hr} "
            f"medium(0.5)={_mr} low(0.2)={_lr}",
            flush=True,
        )
    elif target_kind == "exposure_aware":
        # Performative target: per-agent recommendation depends on the
        # CURRENT local exposure pressure in the agent's demographic slice.
        #
        # Bundled AT covid has no explicit contact graph; we proxy "neighbors
        # of i" by bucketing on (age_bucket, round(mean_interactions)) and
        # using the fraction of that bucket currently in E or I state.
        #
        # target_i = clamp( base_risk(age_i, mi_i) + local_rate_i, 0.05, 0.95 )
        #
        # base_risk gives a static prior (high-risk slice -> baseline higher
        # isolation even when no one is infected). local_rate lifts the
        # target as cases mount in the slice. True performative loop: LM
        # policy this round -> who gets exposed -> next round's local_rate.
        _cit_init = env.runner.state["agents"]["citizens"]
        _envs_init = env.runner.state["environment"]
        _age_long_init = _cit_init["age"].squeeze().long()
        _mi_init = _envs_init["mean_interactions"].squeeze().float()
        _keys = _age_long_init * 100 + _mi_init.round().long()
        _unique_keys, _inverse = torch.unique(_keys, return_inverse=True)
        _n_buckets = int(_unique_keys.shape[0])
        _ones = torch.ones(_age_long_init.shape[0])
        _bucket_n = torch.zeros(_n_buckets).scatter_add_(0, _inverse, _ones)
        _high = (_age_long_init >= 4) | (_mi_init >= 3.5)
        _med = (_age_long_init >= 2) & ~_high
        _base = torch.where(
            _high, torch.tensor(0.5),
            torch.where(_med, torch.tensor(0.3), torch.tensor(0.15)),
        ).float()

        def custom_state_extractor(runner):
            citizens = runner.state["agents"]["citizens"]
            ds = citizens["disease_stage"].squeeze()
            infected = ((ds == 1) | (ds == 2)).float()
            bucket_inf = torch.zeros(_n_buckets).scatter_add_(0, _inverse, infected)
            local_rate = (bucket_inf / _bucket_n.clamp(min=1.0))[_inverse]
            target = (_base + local_rate).clamp(0.05, 0.95)
            return {
                "x": citizens["age"].float().detach(),
                "y": target.detach().reshape(-1, 1),
                "agent_idx": torch.arange(target.shape[0]),
            }
        env._state_extractor = custom_state_extractor  # noqa: SLF001
        _y0 = custom_state_extractor(env.runner)["y"].squeeze()
        print(
            f"[run] target = exposure_aware: n_buckets={_n_buckets} "
            f"init y min={float(_y0.min()):.3f} max={float(_y0.max()):.3f} "
            f"mean={float(_y0.mean()):.3f} std={float(_y0.std()):.3f}",
            flush=True,
        )
    elif target_kind == "disease_stage":
        print("[run] target = disease_stage (float 0-4)", flush=True)
    else:
        raise ValueError(f"unknown TARGET_KIND: {target_kind!r}")

    loss = MSELoss()
    # Mirror opinion-dynamics-post-training/llm_predictor.py:sft_on_round,
    # which is the working setup for Qwen2.5-Instruct + LoRA + KL-SFT.
    # Plain "0.42" target (no EOS): TRL with `completion_only_loss=True`
    # handles EOS appending internally; manually appending double-stacks it.
    # Response_template matches Qwen2.5-Instruct chat template's assistant
    # opener (model-specific; change if switching model families).
    learner_kwargs = dict(
        model=model,
        loss=loss,
        max_steps=effective_max_steps,
        per_device_batch_size=sft_batch_size,
        output_dir=str(out_dir / "trl"),
        response_template="<|im_start|>assistant\n",
        learning_rate=sft_lr,
    )
    print(
        f"[run] SFT learning_rate={sft_lr} max_steps={effective_max_steps}",
        flush=True,
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

    # Register per-round metrics. The Simulator calls these at the END of
    # each round (after env.run + state extraction), so they capture the
    # actual per-round env state. Without this, the post-sim trajectory
    # dump reads `env.runner.state` long after the rollout finished and
    # returns the *final* state every time, masquerading as per-round data.
    def _di_metric(sim_obj) -> float:
        di = sim_obj.env.runner.state["environment"]["daily_infected"]
        return float(di.sum().item())

    def _fnS_metric(sim_obj) -> float:
        ds = sim_obj.env.runner.state["agents"]["citizens"]["disease_stage"]
        return float((ds > 0).float().mean().item())

    sim_metrics = {
        "daily_infected_sum": _di_metric,
        "fraction_non_S": _fnS_metric,
    }
    sim = Simulator(env=env, learner=learner, loss=loss, metrics=sim_metrics)
    print(f"[run] starting outer loop: n_rounds={n_rounds} K={k_steps}", flush=True)
    t_loop = time.time()
    hist = sim.run(n_rounds=n_rounds, epoch_size=k_steps, seed=seed)
    print(f"[run] loop done in {time.time() - t_loop:.1f}s", flush=True)

    torch.save([dict(r) for r in hist.records], out_dir / "history.pt")

    # ---- LM output diagnostic (POST-SFT) -----------------------------------
    # Force one more forward pass through the LM on the same population to
    # capture the final per-agent parsed recommendation. Group by profile_type
    # (age_bucket, mean_interactions) and report mean/stdev/min/max so we can
    # compare across betas and see if the LM's per-profile outputs actually
    # differ between sweeps.
    print("[run] dumping final per-profile LM recommendations...", flush=True)
    t0 = time.time()
    final_features = env.runner.state["agents"]["citizens"]["age"].float()
    with torch.no_grad():
        final_preds = model(final_features).squeeze().detach().cpu()
    print(f"[run] final-pass forward in {time.time() - t0:.1f}s", flush=True)

    # Stats on final_preds (what platform_signal would be set to).
    print(
        f"[diag] final preds stats: min={float(final_preds.min()):.4f} "
        f"max={float(final_preds.max()):.4f} mean={float(final_preds.mean()):.4f} "
        f"std={float(final_preds.std()):.4f}",
        flush=True,
    )
    # With logit_signal_writer the env path is: preds -> logit -> sigmoid
    # so will_isolate ~= preds (within the clamp(0.01, 0.99) range applied
    # by the writer). Reporting preds directly instead of double-sigmoiding.
    _p = final_preds.clamp(min=0.01, max=0.99)
    print(
        f"[diag] will_isolate (= preds clamped): min={float(_p.min()):.4f} "
        f"max={float(_p.max()):.4f} mean={float(_p.mean()):.4f}",
        flush=True,
    )

    # Also dump 20 random (prompt, raw text, parsed) samples POST-SFT to
    # compare against pre-SFT.
    _sample_texts_post = model._generate([model.build_prompt(model.profile_at(i)) for i in _sample_idx])
    _sample_log_post = []
    print("[diag] sample LM outputs (post-SFT):", flush=True)
    for _idx, _txt in zip(_sample_idx, _sample_texts_post):
        _parsed = model._parse(_txt)
        _sample_log_post.append({"agent_idx": int(_idx), "raw_text": _txt, "parsed": float(_parsed)})
        print(f"  agent {_idx}: text={_txt!r}  parsed={_parsed:.3f}", flush=True)
    (out_dir / "diagnostic_post_sft.json").write_text(json.dumps(_sample_log_post, indent=2))

    # Group by (age_bucket, mean_interactions). Use a Python dict so we can
    # save as JSON cleanly.
    grp = {}
    age_t = profiles["age_bucket"].tolist()
    mi_t = profiles["mean_interactions"].tolist()
    for i in range(n_agents):
        key = f"age={int(age_t[i])}_mi={float(mi_t[i]):.1f}"
        grp.setdefault(key, []).append(float(final_preds[i].item()))

    profile_summary = []
    for key in sorted(grp):
        vals = torch.tensor(grp[key])
        profile_summary.append({
            "profile_type": key,
            "n_agents": int(vals.shape[0]),
            "rec_mean": float(vals.mean()),
            "rec_std": float(vals.std()) if vals.shape[0] > 1 else 0.0,
            "rec_min": float(vals.min()),
            "rec_max": float(vals.max()),
        })
    (out_dir / "recommendations.json").write_text(json.dumps(profile_summary, indent=2))
    print(f"[run] {len(profile_summary)} profile types saved -> recommendations.json", flush=True)

    trajectory = []
    for r in hist.records:
        theta = r.get("theta")
        row = {
            "round": int(r["round"]),
            "theta_norm": float(theta.norm().item()) if hasattr(theta, "norm") else None,
            "daily_infected_sum": float(r["daily_infected_sum"]),
            "fraction_non_S": float(r["fraction_non_S"]),
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
