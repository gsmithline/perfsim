#!/usr/bin/env python3
"""FEATURE_NOISE sweep at a permissive lever (eps_AI=2.0, W=0.5, kappa=0).
FEATURE_NOISE = how UNpredictable the natural population is from the model's
feature (0 => feature is the truth; larger => the model must distort to predict).
For each level records cum_reward + distortion (final W1 vs no-AI, final spread)
for: myopic(online, greedy), long-horizon(plans), long-no-deploy(control).
Shows the reward/distortion ordering is robust, not a single-point fluke."""
import os
# fix the permissive operating point BEFORE importing rlvr_local (it reads env at import)
os.environ.update(EPS_AI="2.0", W="0.5", KAPPA="0.0", O_SCALE="0", H="30", STEPS="300")
import importlib.util, json
import numpy as np, torch
from scipy.stats import wasserstein_distance
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = os.path.dirname(os.path.abspath(__file__))
def L(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
R = L("rlvr", f"{SP}/rlvr_local.py")

NOISES = [0.0, 0.08, 0.16, 0.24]
INNER = 8

def myopic_online(feat, innate, o, adjb, ref):
    net, _ = R.make_proxy(feat, R.qwen_prior(), draw=0)
    opt = torch.optim.Adam(net.parameters(), R.LR)
    rew, w1, spr = [], [], []
    x = innate.clone()
    for t in range(R.H):
        net.eval()
        with torch.no_grad(): yhat = torch.clip(net(feat) + o, 0, 1)
        rew.append(float(-((yhat - x) ** 2).mean())); spr.append(float(x.std()))
        w1.append(float(wasserstein_distance(x.numpy(), ref[t].numpy())))
        with torch.no_grad():
            z, _ = R.deploy(x, yhat, innate, R.W); x = R.peer_step(z, adjb)
        net.train(); tgt = x.detach()
        for _ in range(INNER):
            opt.zero_grad()
            (((torch.clip(net(feat) + o, 0, 1) - tgt) ** 2).mean()).backward(); opt.step()
    return dict(cum_reward=float(np.sum(rew)), final_w1=w1[-1], final_spread=spr[-1])

def main():
    innate, adj = R.fv2.ml_action_setup(); adjb = (adj > 0).float()
    g = torch.Generator().manual_seed(12345)
    eta = torch.randn(innate.shape[0], generator=g)      # shared noise direction across levels
    ref = R.noai_reference(innate, adjb)
    rows = {}
    for fn in NOISES:
        feat = innate + fn * eta
        _, o = R.make_proxy(feat, R.qwen_prior(), draw=0); o = R.O_SCALE * o
        lh = R.eval_rollout(R.train(feat, innate, o, adjb, R.W, True, f"long fn={fn}"),
                            o, feat, innate, adjb, R.W, ref)
        nd = R.eval_rollout(R.train(feat, innate, o, adjb, 0.0, True, f"ndp fn={fn}"),
                            o, feat, innate, adjb, 0.0, ref)
        mo = myopic_online(feat, innate, o, adjb, ref)
        rows[fn] = dict(
            long=dict(cum_reward=lh["cum_reward"], final_w1=lh["final_w1"], final_spread=lh["final_spread"]),
            myopic_online=mo,
            nodeploy=dict(cum_reward=nd["cum_reward"], final_w1=nd["final_w1"], final_spread=nd["final_spread"]))
        print(f"\nfn={fn}: long R={rows[fn]['long']['cum_reward']:.3f} W1={rows[fn]['long']['final_w1']:.3f} | "
              f"myopic_online R={mo['cum_reward']:.3f} W1={mo['final_w1']:.3f} | "
              f"nodeploy R={rows[fn]['nodeploy']['cum_reward']:.3f}", flush=True)
    json.dump(dict(meta=dict(eps_AI=2.0, W=0.5, kappa=0.0, H=30, steps=300, noises=NOISES),
                   rows={str(k): v for k, v in rows.items()}),
              open(f"{SP}/rlvr_sweep.json", "w"))

    cols = {"long": "tab:red", "myopic_online": "tab:orange", "nodeploy": "tab:blue"}
    labs = {"long": "long-horizon", "myopic_online": "myopic (online)", "nodeploy": "no-deploy"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for cond in ("long", "myopic_online", "nodeploy"):
        ax[0].plot(NOISES, [rows[n][cond]["cum_reward"] for n in NOISES], "o-", color=cols[cond], label=labs[cond])
        ax[1].plot(NOISES, [rows[n][cond]["final_w1"] for n in NOISES], "o-", color=cols[cond], label=labs[cond])
    ax[0].set_title("cumulative reward (higher=better prediction)", fontsize=10)
    ax[1].set_title("population distortion  W1 from no-AI (higher=more distortion)", fontsize=10)
    for a in ax: a.set_xlabel("FEATURE_NOISE  (feature ⟂ opinion →)"); a.legend(fontsize=8)
    fig.suptitle("[exploratory] RLVR surrogate lever sweep — long-horizon: more reward, LESS distortion than myopic",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f"{SP}/rlvr_sweep_fig.png", dpi=130)
    print(f"\nwrote {SP}/rlvr_sweep.json and rlvr_sweep_fig.png")

if __name__ == "__main__":
    main()
