import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def _():
    """
    This notebook contains an example of how to wire a small LLM to the FJ dynamics environment
    Pokec FJ smoke test with an HF causal LM as the predictor.

    Runs FJ on a small synthetic graph with synthetic profiles, using
    Qwen2.5-0.5B-Instruct as the platform. Each outer round queries the LM
    once per agent to set FJ initial conditions, then runs the inner FJ
    dynamics under those fixed predictions for `epoch_size` ticks. SFT
    fine-tunes the LM on the labeled subset's evolved opinions at the start
    of the next round.

    End-to-end CPU run, small N and few rounds to keep it tractable.

    To swap in real Pokec data:

    1. In an older-pandas env (where lcc_profiles_relation_to_smoking.pk
    unpickles cleanly), save the relevant columns to a parquet file:

    import pandas as pd, pickle
    with open("lcc_profiles_relation_to_smoking.pk", "rb") as fh:
    df = pickle.load(fh)
    df[["user_id", "age", "gender", "relation_to_alcohol"]].to_parquet(
    "examples/pokec/profiles.parquet"
    )

    2. Replace `_make_synthetic_setup` below with a loader that reads
    `examples/pokec/profiles.parquet` (and the graph + parameter pickles
    from `examples/pokec/`).

    Run from the repo root: `python examples/pokec_fj_llm.py`.
    """
    return


@app.cell
def _():
    from __future__ import annotations

    import argparse
    import random

    import numpy as np
    import pandas as pd
    import torch

    from perfsim.environments.dynamics import FJWorld, normalize_adjacency
    from perfsim.learners.lm.sft import SFTLearner
    from perfsim.losses import MSELoss
    from perfsim.models.hf_causal_lm import HFCausalLMModel
    from perfsim.simulator import Simulator

    return (
        FJWorld,
        HFCausalLMModel,
        MSELoss,
        SFTLearner,
        Simulator,
        normalize_adjacency,
        np,
        pd,
        random,
        torch,
    )


@app.cell
def _():
    ALCOHOL_VALUES = [
        "I drink occasionally",
        "I drink regularly",
        "I don't drink",
        "I abstain from alcohol",
        "I no longer drink",
    ]
    return (ALCOHOL_VALUES,)


@app.cell
def _(ALCOHOL_VALUES, normalize_adjacency, np, pd, random, torch):
    def _make_synthetic_setup(n: int = 30, seed: int = 0):
        """
        test function to generate data
        """
        rng = random.Random(seed)
        nprng = np.random.default_rng(seed)

        profiles = pd.DataFrame(
            {
                "age": [rng.randint(17, 30) for _ in range(n)],
                "gender": [rng.randint(0, 1) for _ in range(n)],
                "relation_to_alcohol": [rng.choice(ALCOHOL_VALUES) for _ in range(n)],
            }
        )
        innate = torch.tensor(nprng.uniform(0.0, 1.0, size=n), dtype=torch.float32)

        # This makes the random sparse graph
        p_edge = 0.2
        adj = (torch.rand(n, n) < p_edge).float()
        adj = (adj + adj.T).clamp(0.0, 1.0)
        adj.fill_diagonal_(0.0)
        W = normalize_adjacency(adj)

        peer_sus = torch.tensor(nprng.uniform(0.5, 0.95, size=n), dtype=torch.float32)
        platform_sus = torch.tensor(nprng.uniform(0.5, 0.95, size=n), dtype=torch.float32)

        return {
            "profiles": profiles,
            "innate": innate,
            "W": W,
            "peer_sus": peer_sus,
            "platform_sus": platform_sus,
            "n": n,
        }

    return


@app.cell
def _(pd):
    def build_prompt(profile: pd.Series, tokenizer) -> str:
        """Profile row -> chat-templated prompt asking for an opinion in [0, 1]."""
        age = int(profile["age"])
        gender_str = "female" if int(profile["gender"]) == 0 else "male"
        alcohol = str(profile["relation_to_alcohol"])
        user_msg = (
            "Estimate this person's attitude toward smoking. "
            "0 means very negative (anti-smoking), 1 means very positive (pro-smoking).\n\n"
            "Person profile:\n"
            f"- Age: {age}\n"
            f"- Gender: {gender_str}\n"
            f"- Alcohol use: {alcohol}\n\n"
            "Output a single number in [0, 1], e.g. 0.42. Respond with only the number."
        )
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (build_prompt,)


@app.cell
def _():
    n=30
    n_labeled = int(n * 0.8)
    seed=42
    setup = _make_synthetic_setup(seed=seed)
    epoch_size = 100
    n_rounds = 10
    sft_max_steps = 1
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    print(
    f"Synthetic setup: N={n} agents, n_labeled={n_labeled}, "
    f"n_rounds={epoch_size}, epoch_size={100}, "
    f"sft_max_steps={sft_max_steps}"
    )
    return (
        base_model,
        epoch_size,
        n,
        n_labeled,
        n_rounds,
        seed,
        setup,
        sft_max_steps,
    )


@app.cell
def _(HFCausalLMModel, base_model, build_prompt, setup, torch):
    model = HFCausalLMModel(
        base_model_name=base_model,
        profiles=setup["profiles"],
        prompt_builder=build_prompt,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        device="cpu",
        dtype=torch.float32,
        max_new_tokens=6,
        gen_batch_size=4,
        load_now=True,
     )
    return (model,)


@app.cell
def _(FJWorld, MSELoss, SFTLearner, model, setup, sft_max_steps):
    loss = MSELoss()
    learner = SFTLearner(
        model,
        loss,
        max_steps=sft_max_steps,
        learning_rate=5e-5,
        per_device_batch_size=2,
        max_seq_length=256,
    )

    world = FJWorld(
        innate=setup["innate"],
        graph=setup["W"],
        peer_sus=setup["peer_sus"],
        platform_sus=setup["platform_sus"],
        features=setup["innate"],  
        profiles=setup["profiles"],
    )
    return learner, loss, world


@app.cell
def _(Simulator, learner, loss, world):
    sim = Simulator(world=world, learner=learner, loss=loss)
    return (sim,)


@app.cell
def _(n, n_labeled, setup, torch):
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:n_labeled] = True
    initial_data = {
        "x": setup["innate"].unsqueeze(-1),
        "y": setup["innate"].unsqueeze(-1),
        "agent_idx": torch.arange(n),
    }

    print(
        f"  init: opinion mean={setup['innate'].mean():.4f}  "
        f"std={setup['innate'].std():.4f}"
    )
    return initial_data, train_mask


@app.cell
def _(epoch_size, initial_data, n_rounds, seed, sim, train_mask):
    history = sim.run(
        n_rounds=n_rounds,
        epoch_size=epoch_size,
        seed=seed,
        initial_data=initial_data,
        train_mask=train_mask,
    )
    return (history,)


@app.cell
def _(history, train_mask, world):
    final = world.state["opinion"]
    for t, rec in enumerate(history.records):
        print(f"  round {t}: |theta|={float(rec['theta']):.4f}  "
            f"stability_gap={rec.get('stability_gap', float('nan')):.6f}"
        )
        print(f"  final opinion: mean={final.mean():.4f}  std={final.std():.4f}  "
            f"range=[{final.min():.3f}, {final.max():.3f}]"
        )
        print(f"    labeled: mean={final[train_mask].mean():.4f}  "
            f"unlabeled: mean={final[~train_mask].mean():.4f}"
        )
        print("\nDone. End-to-end LM-as-predictor FJ run completed.")
    return


if __name__ == "__main__":
    app.run()
