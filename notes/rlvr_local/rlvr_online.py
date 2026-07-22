#!/usr/bin/env python3
"""Faithful ONLINE myopic control (mirrors LLM per-round SFT/RLVR):
persistent weights theta, each round take a few SGD steps to fit g_theta(feat)
-> current opinion x_t (gamma=0, greedy), then DEPLOY and advance the real loop.
No planning / no through-dynamics credit. Question: does a faithful online-myopic
agent also over-collapse the population (as the stationary-detach myopic did), or
was that an artifact?

Reuses rlvr_local's calibrated dynamics + feature construction by importing it
(env vars set before import so the operating point matches)."""
import importlib.util, os, json
import numpy as np, torch, torch.nn as nn
from scipy.stats import wasserstein_distance

SP = os.path.dirname(os.path.abspath(__file__))
def L(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
R = L("rlvr", f"{SP}/rlvr_local.py")   # picks up EPS_AI/W/KAPPA/FEATURE_NOISE/... from env

INNER = int(os.environ.get("INNER", "8"))   # SGD steps/round for online myopic

def main():
    innate, adj = R.fv2.ml_action_setup(); adjb = (adj > 0).float()
    g = torch.Generator().manual_seed(12345)
    feat = innate + R.FEATURE_NOISE * torch.randn(innate.shape[0], generator=g)
    net, o = R.make_proxy(feat, R.qwen_prior(), draw=0); o = R.O_SCALE * o
    ref = R.noai_reference(innate, adjb)
    opt = torch.optim.Adam(net.parameters(), R.LR)

    rew, spread, dist, w1 = [], [], [], []
    x = innate.clone()
    for t in range(R.H):
        net.eval()
        with torch.no_grad():
            yhat = torch.clip(net(feat) + o, 0, 1)
        rew.append(float(-((yhat - x) ** 2).mean()))
        spread.append(float(x.std())); dist.append(float((x - innate).abs().mean()))
        w1.append(float(wasserstein_distance(x.numpy(), ref[t].numpy())))
        # deploy + advance the REAL loop
        with torch.no_grad():
            z, _ = R.deploy(x, yhat, innate, R.W)
            x = R.peer_step(z, adjb)
        # MYOPIC online update: fit g_theta(feat) -> current opinion x_t (detached)
        net.train(); tgt = x.detach()
        for _ in range(INNER):
            opt.zero_grad()
            loss = ((torch.clip(net(feat) + o, 0, 1) - tgt) ** 2).mean()
            loss.backward(); opt.step()

    out = dict(condition="myopic_online", cum_reward=float(np.sum(rew)),
               final_spread=spread[-1], final_dist=dist[-1], final_w1=w1[-1],
               reward=rew, spread=spread, dist_innate=dist, w1_noai=w1)
    json.dump(out, open(f"{SP}/rlvr_online.json", "w"))
    print(f"eps_AI={R.EPS_AI} W={R.W} kappa={R.KAPPA} feat_noise={R.FEATURE_NOISE} INNER={INNER}")
    print(f"myopic_online: cum_reward={out['cum_reward']:.3f}  final_spread={out['final_spread']:.4f}  "
          f"final|x-innate|={out['final_dist']:.4f}  final_W1={out['final_w1']:.4f}")
    print("(compare stationary myopic / long_horizon from rlvr_hackon.json)")

if __name__ == "__main__":
    main()
