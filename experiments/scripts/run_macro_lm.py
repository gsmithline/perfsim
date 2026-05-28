"""Single-config macro_economics + LM + KL-SFT run for condor sweep."""

from __future__ import annotations

import json
import os
import random as _random
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

try:
    import wandb as _wandb
    _HAS_WANDB = True
except ImportError:
    _wandb = None
    _HAS_WANDB = False

from perfsim.core.learner import Learner
from perfsim.core.types import SUPERVISED_SCHEMA
from perfsim.learners.lm.kl_sft import KLSFTLearner
from perfsim.learners.lm.sft import SFTLearner
from perfsim.losses import MSELoss
from perfsim.models.hf_causal_lm import HFCausalLMModel
from perfsim.scenarios.at_macro import make_macro_env
from perfsim.simulator import Simulator


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
    run_tag = _env_or("RUN_TAG")
    kl_beta = _env_float("KL_BETA", 0.0)
    training_style = _env_or("TRAINING_STYLE", "sft_kl")
    base_model = _env_or("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    n_rounds = _env_int("N_ROUNDS", 5)
    k_steps = _env_int("K_STEPS", 3)
    seed = _env_int("SEED", 0)
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(os.environ.get("OUT_DIR", f"runs/at_macro_lm/{run_tag}"))
    wandb_project = os.environ.get("WANDB_PROJECT")
    max_steps = _env_int("SFT_MAX_STEPS", 20)
    gen_batch_size = _env_int("GEN_BATCH_SIZE", 32)
    max_new_tokens = _env_int("MAX_NEW_TOKENS", 16)
    sft_batch_size = _env_int("SFT_BATCH_SIZE", 16)
    sft_full_epoch = os.environ.get("SFT_FULL_EPOCH", "0").lower() in ("1", "true", "yes")
    yaml_name = os.environ.get("MACRO_YAML", "config_100_agents.yaml")
    n_agents_cfg = _env_int("N_AGENTS", 100)
    lora_r = _env_int("LORA_R", 32)
    use_lora = _env_int("USE_LORA", 1) == 1
    sft_lr = _env_float("SFT_LR", 1e-5)
    group_prompting = os.environ.get("GROUP_PROMPTING", "0").lower() in ("1", "true", "yes")
    consumption_noise = _env_float("CONSUMPTION_NOISE", 0.05)
    calibrated_uac_path = os.environ.get("CALIBRATED_UAC", "")

    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_tag": run_tag,
        "kl_beta": kl_beta,
        "training_style": training_style,
        "base_model": base_model,
        "n_rounds": n_rounds,
        "k_steps": k_steps,
        "seed": seed,
        "device": device,
        "max_steps": max_steps,
        "sft_batch_size": sft_batch_size,
        "sft_full_epoch": sft_full_epoch,
        "gen_batch_size": gen_batch_size,
        "max_new_tokens": max_new_tokens,
        "yaml_name": yaml_name,
        "lora_r": lora_r,
        "use_lora": use_lora,
        "sft_lr": sft_lr,
        "host": os.uname().nodename,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[run] {json.dumps(config)}", flush=True)

    wandb = None
    if wandb_project and _HAS_WANDB:
        wandb = _wandb
        wandb_suffix = os.environ.get("WANDB_RUN_SUFFIX", "")
        wandb_name = f"{run_tag}{wandb_suffix}" if wandb_suffix else run_tag
        wandb.init(project=wandb_project, name=wandb_name, config=config)

    # Logit signal_writer mirrors the covid setup: PerfsimEarningDecision
    # applies sigmoid() to platform_signal. To make consumption_propensity
    # equal the LM's emitted value `p`, we write logit(p) so
    # sigmoid(logit(p)) = p. Recovers full [0, 1] expressivity instead of
    # being squashed to [sigmoid(0), sigmoid(1)] = [0.5, 0.73].
    def logit_signal_writer(runner, preds):
        if preds.ndim == 2 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        if consumption_noise > 0:
            noise = torch.randn_like(preds) * consumption_noise
            preds = preds + noise
        p = preds.clamp(min=0.01, max=0.99)
        logit_p = torch.log(p / (1.0 - p))
        runner.state["agents"]["consumers"]["platform_signal"] = logit_p.detach().clone()

    torch.manual_seed(seed)
    print("[run] building macro env", flush=True)
    t0 = time.time()
    env = make_macro_env(
        init_seed=seed,
        yaml_name=yaml_name,
        n_agents=n_agents_cfg,
        signal_writer=logit_signal_writer,
        keep_trajectory=True,
    )
    n_agents = env.runner.state["agents"]["consumers"]["age"].shape[0]
    print(f"[run] env ready: {n_agents} agents in {time.time() - t0:.1f}s", flush=True)

    if calibrated_uac_path:
        uac_data = torch.load(calibrated_uac_path, weights_only=False)
        for tf_key in env.runner.initializer.transition_function:
            tf = env.runner.initializer.transition_function[tf_key]
            for _, module in tf.named_modules():
                if hasattr(module, "external_UAC"):
                    n_steps_local = module.external_UAC.shape[0]
                    uac_to_load = uac_data[:n_steps_local]
                    with torch.no_grad():
                        module.external_UAC.copy_(uac_to_load)
                    print(
                        f"[run] loaded calibrated UAC from {calibrated_uac_path} "
                        f"(sliced {tuple(uac_data.shape)} -> {tuple(uac_to_load.shape)})",
                        flush=True,
                    )
                    break

    # Per-agent profile DataFrame for prompt construction. Pulls features
    # from the AT runner state; the categorical labels come from
    # populations/NYC/mapping.json (loaded via the YAML's mapping_path).
    AGE_LABELS = ["20-29", "30-39", "40-49", "50-64", "65+", "Under 19"]
    GENDER_LABELS = ["female", "male"]
    ETHNICITY_LABELS = ["white", "other", "hispanic", "black", "asian", "native"]

    ages = env.runner.state["agents"]["consumers"]["age"].squeeze().long().tolist()
    genders = env.runner.state["agents"]["consumers"]["gender"].squeeze().long().tolist()
    ethnicities = env.runner.state["agents"]["consumers"]["ethnicity"].squeeze().long().tolist()

    profiles = pd.DataFrame({
        "age_bucket": ages,
        "age_label": [AGE_LABELS[min(int(a), len(AGE_LABELS) - 1)] for a in ages],
        "gender": [GENDER_LABELS[min(int(g), 1)] for g in genders],
        "ethnicity": [ETHNICITY_LABELS[min(int(e), len(ETHNICITY_LABELS) - 1)] for e in ethnicities],
        "agent_id": list(range(n_agents)),
    })
    print(
        f"[run] profile features: age_buckets={sorted(set(ages))} "
        f"genders={sorted(set(genders))} ethnicities={sorted(set(ethnicities))}",
        flush=True,
    )

    
    SYSTEM_MSG = (
        "You are a financial advisor providing personalized consumption "
        "recommendations during ongoing economic uncertainty. Your goal "
        "is to help each household balance current spending with future "
        "financial security."
    )

    def _get_econ_state():
        env_state = env.runner.state["environment"]
        inflation = 0.0
        pi = env_state.get("P_i")
        if pi is not None:
            try:
                val = float(pi[-1][-1].item())
                if abs(val) < 1.0:
                    inflation = val
            except Exception:
                pass
        unemployment = 0.0
        u = env_state.get("U")
        if u is not None:
            try:
                row = u[-1]
                nz = row.nonzero(as_tuple=True)[0]
                if len(nz):
                    unemployment = float(row[nz[-1]].item())
            except Exception:
                pass
        price = 0.0
        p = env_state.get("P")
        if p is not None:
            try:
                price = float(p[-1][-1].item())
            except Exception:
                pass
        return inflation, unemployment, price

    def prompt_builder(profile_row, tokenizer):
        inflation, unemployment, price = _get_econ_state()
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": (
                    f"Household profile: age group {profile_row['age_label']}, "
                    f"{profile_row['gender']}, {profile_row['ethnicity']}.\n\n"
                    f"Current economic conditions: "
                    f"inflation {inflation * 100:.1f}%, "
                    f"unemployment {unemployment:.1f}%, "
                    f"price index {price:.1f}.\n\n"
                    f"Respond with ONLY a single number between 0.00 and 1.00. "
                    f"0 means save everything, 1 means spend everything. "
                    f"No explanation."
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
        group_prompting=group_prompting,
        load_now=True,
    )
    print(
        f"[run] LM training mode: {'LoRA r=' + str(lora_r) if use_lora else 'FULL fine-tuning'}",
        flush=True,
    )
    print(f"[run] SFT learning_rate={sft_lr} max_steps={max_steps}", flush=True)
    print(f"[run] LM loaded in {time.time() - t0:.1f}s", flush=True)

    _rng = _random.Random(seed)
    _sample_idx = _rng.sample(range(n_agents), min(20, n_agents))
    _sample_prompts = [model.build_prompt(model.profile_at(i)) for i in _sample_idx]
    print("[diag] sample prompt (agent 0):", flush=True)
    print(_sample_prompts[0], flush=True)
    print("[diag] sample LM outputs (pre-SFT):", flush=True)
    _sample_texts = model._generate(_sample_prompts)
    _sample_log = []
    for _idx, _prompt, _txt in zip(_sample_idx, _sample_prompts, _sample_texts):
        _parsed = model._parse(_txt)
        _sample_log.append({"agent_idx": int(_idx), "raw_text": _txt, "parsed": float(_parsed)})
        print(f"  agent {_idx}: text={_txt!r}  parsed={_parsed:.3f}", flush=True)
    (out_dir / "diagnostic_pre_sft.json").write_text(json.dumps(_sample_log, indent=2))

    if sft_full_epoch:
        effective_max_steps = -(-n_agents // sft_batch_size)
        print(
            f"[run] SFT_FULL_EPOCH=1: max_steps={effective_max_steps} "
            f"(= ceil({n_agents}/{sft_batch_size}))",
            flush=True,
        )
    else:
        effective_max_steps = max_steps

    _cit_init = env.runner.state["agents"]["consumers"]
    _age_init = _cit_init["age"].squeeze().long()
    _gender_init = _cit_init["gender"].squeeze().long()
    _ethnicity_init = _cit_init["ethnicity"].squeeze().long()
    _keys_init = (_age_init * 100 + _gender_init * 10 + _ethnicity_init).long()
    _unique_keys, _bucket_idx = torch.unique(_keys_init, return_inverse=True)
    _n_buckets = int(_unique_keys.shape[0])

    _prev_assets = _cit_init["assets"].float().detach().mean(dim=1).clone()

    _fallback_target = torch.where(
        _age_init >= 4, torch.tensor(0.3),
        torch.where(_age_init >= 2, torch.tensor(0.5), torch.tensor(0.7)),
    ).float()

    def custom_state_extractor(runner):
        nonlocal _prev_assets
        consumers = runner.state["agents"]["consumers"]
        cur_assets = consumers["assets"].float().detach().mean(dim=1)
        asset_gain = cur_assets - _prev_assets
        cons_used = consumers["consumption_propensity"].float().detach().squeeze()

        target = _fallback_target.clone()
        for b in range(_n_buckets):
            mask = _bucket_idx == b
            if mask.sum() < 2:
                continue
            bucket_gains = asset_gain[mask]
            bucket_cons = cons_used[mask]
            best = bucket_gains.argmax()
            target[mask] = bucket_cons[best]

        target = target.clamp(0.05, 0.95)
        _prev_assets = cur_assets.clone()

        return {
            "x": consumers["age"].float().detach(),
            "y": target.detach().reshape(-1, 1),
            "agent_idx": torch.arange(target.shape[0]),
        }
    env._state_extractor = custom_state_extractor  # noqa: SLF001
    _y0 = custom_state_extractor(env.runner)["y"].squeeze()
    print(
        f"[run] target = best-outcome consumption per bucket (noise={consumption_noise}): "
        f"n_buckets={_n_buckets} "
        f"init y min={float(_y0.min()):.3f} max={float(_y0.max()):.3f} "
        f"mean={float(_y0.mean()):.3f} std={float(_y0.std()):.3f}",
        flush=True,
    )

    loss = MSELoss()
    learner_kwargs = dict(
        model=model,
        loss=loss,
        max_steps=effective_max_steps,
        per_device_batch_size=sft_batch_size,
        output_dir=str(out_dir / "trl"),
        response_template="<|im_start|>assistant\n",
        learning_rate=sft_lr,
    )
    if training_style == "sft":
        learner = SFTLearner(**learner_kwargs)
    elif training_style == "sft_kl":
        learner = KLSFTLearner(
            **learner_kwargs,
            ref_model_name=base_model,
            kl_beta=kl_beta,
        )
    elif training_style == "frozen":
        class _FrozenLearner(Learner):
            accepted_schemas = (SUPERVISED_SCHEMA,)

            def __init__(self, model, loss):
                super().__init__(model, loss)

            def train(self, data):
                print("[FrozenLearner] skipping SFT (frozen baseline)", flush=True)

            def reset(self):
                pass

        learner = _FrozenLearner(model=model, loss=loss)
    else:
        raise ValueError(f"unknown TRAINING_STYLE: {training_style!r}")

    def _mean_assets(sim_obj) -> float:
        a = sim_obj.env.runner.state["agents"]["consumers"]["assets"]
        return float(a.float().mean().item())

    def _mean_consumption_prop(sim_obj) -> float:
        c = sim_obj.env.runner.state["agents"]["consumers"].get("consumption_propensity")
        if c is None:
            return 0.0
        return float(c.float().mean().item())

    def _inflation(sim_obj) -> float:
        pi = sim_obj.env.runner.state["environment"].get("P_i")
        if pi is None:
            return 0.0
        try:
            return float(pi[-1][-1].item())
        except Exception:
            return 0.0

    def _unemployment(sim_obj) -> float:
        u = sim_obj.env.runner.state["environment"].get("U")
        if u is None:
            return 0.0
        try:
            row = u[-1]
            nonzero = row.nonzero(as_tuple=True)[0]
            if len(nonzero) == 0:
                return 0.0
            return float(row[nonzero[-1]].item())
        except Exception:
            return 0.0

    def _pred_stats(sim_obj) -> dict:
        features = sim_obj.env.runner.state["agents"]["consumers"]["age"].float()
        with torch.no_grad():
            preds = sim_obj.predictor.model(features).squeeze().detach().cpu()
        return {
            "pred_mean": float(preds.mean()),
            "pred_std": float(preds.std()),
            "pred_min": float(preds.min()),
            "pred_max": float(preds.max()),
        }

    def _subgroup_assets(sim_obj) -> dict:
        assets = sim_obj.env.runner.state["agents"]["consumers"]["assets"].float()
        ages = sim_obj.env.runner.state["agents"]["consumers"]["age"].squeeze().long()
        col_mean = assets.mean(dim=1)
        out = {}
        for bucket in range(6):
            mask = ages == bucket
            if mask.any():
                out[f"assets_age{bucket}"] = float(col_mean[mask].mean())
        return out

    def _subgroup_preds(sim_obj) -> dict:
        features = sim_obj.env.runner.state["agents"]["consumers"]["age"].float()
        ages = features.squeeze().long()
        with torch.no_grad():
            preds = sim_obj.predictor.model(features).squeeze().detach().cpu()
        out = {}
        for bucket in range(6):
            mask = ages == bucket
            if mask.any():
                out[f"pred_age{bucket}"] = float(preds[mask].mean())
        return out

    def _train_loss(sim_obj) -> float:
        data = sim_obj.env._state_extractor(sim_obj.env.runner)
        return float(sim_obj.predictor.loss(sim_obj.predictor.model, data).item())

    def _price_of_goods(sim_obj) -> float:
        p = sim_obj.env.runner.state["environment"].get("P")
        if p is None:
            return 0.0
        try:
            return float(p[-1][-1].item())
        except Exception:
            return 0.0

    sim_metrics = {
        "mean_assets": _mean_assets,
        "mean_consumption_prop": _mean_consumption_prop,
        "inflation": _inflation,
        "unemployment": _unemployment,
        "pred_stats": _pred_stats,
        "subgroup_assets": _subgroup_assets,
        "subgroup_preds": _subgroup_preds,
        "train_loss": _train_loss,
        "price_of_goods": _price_of_goods,
    }

    trajectory = []

    def _on_round(t, record):
        theta = record.get("theta")
        row = {
            "round": t,
            "theta_norm": float(theta.norm().item()) if hasattr(theta, "norm") else None,
            "mean_assets": float(record["mean_assets"]),
            "mean_consumption_prop": float(record["mean_consumption_prop"]),
            "inflation": float(record["inflation"]),
            "unemployment": float(record["unemployment"]),
            "train_loss": float(record["train_loss"]),
            "price_of_goods": float(record["price_of_goods"]),
        }
        gap = record.get("stability_gap")
        if hasattr(gap, "item"):
            row["stability_gap"] = float(gap.item())
        ps = record.get("pred_stats", {})
        row.update(ps)
        sa = record.get("subgroup_assets", {})
        row.update(sa)
        sp = record.get("subgroup_preds", {})
        row.update(sp)

        trajectory.append(row)
        if wandb is not None:
            wandb.log(row)
        print(f"[round {t}] {row}", flush=True)

    sim = Simulator(env=env, learner=learner, loss=loss, metrics=sim_metrics)
    print(f"[run] starting outer loop: n_rounds={n_rounds} K={k_steps}", flush=True)
    t_loop = time.time()
    hist = sim.run(n_rounds=n_rounds, epoch_size=k_steps, seed=seed, on_round=_on_round)
    print(f"[run] loop done in {time.time() - t_loop:.1f}s", flush=True)

    torch.save([dict(r) for r in hist.records], out_dir / "history.pt")

    _sample_texts_post = model._generate([model.build_prompt(model.profile_at(i)) for i in _sample_idx])
    _sample_log_post = []
    print("[diag] sample LM outputs (post-SFT):", flush=True)
    for _idx, _txt in zip(_sample_idx, _sample_texts_post):
        _parsed = model._parse(_txt)
        _sample_log_post.append({"agent_idx": int(_idx), "raw_text": _txt, "parsed": float(_parsed)})
        print(f"  agent {_idx}: text={_txt!r}  parsed={_parsed:.3f}", flush=True)
    (out_dir / "diagnostic_post_sft.json").write_text(json.dumps(_sample_log_post, indent=2))

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
