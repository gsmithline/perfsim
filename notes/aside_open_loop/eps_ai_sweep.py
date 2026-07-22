#!/usr/bin/env python3
"""SINGLE-KNOB demo: sweep ONLY eps_AI (AI acceptance/gate) under fixed regimes,
OPEN loop vs CLOSED loop overlaid (the contrast = effect of the performative loop).

Held constant EXCEPT eps_AI:  W=0.30, kappa=0.25 (in (0,1)), anchored LLM proxies.
3 regimes differ ONLY in peer coupling eps_social {0.10 tight,0.30 medium,0.60 wide}.
Knob: eps_AI 0.05..2.0 (narrow gate -> open gate).

OPEN  loop: model FROZEN at its deployed-prior anchor; reads current opinion x_t
            each round but NEVER retrains (AI does not learn from what it shaped).
CLOSED loop: model retrains each round on the population's realized opinions
            (performative feedback) -- reuses closed_env_sweep.closed_run VERBATIM.
Run from perfsim repo root.
"""
import importlib.util, os, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

torch.set_num_threads(6)
HERE=os.path.dirname(os.path.abspath(__file__))
def _load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ce=_load("ce", f"{HERE}/closed_env_sweep.py")            # closed_run, make_proxy, ROUNDS
sw,gp,fv2,MODELS,ROUNDS=ce.sw,ce.gp,ce.fv2,ce.MODELS,ce.ROUNDS

REGIMES=[("tight",0.10),("medium",0.30),("wide",0.60)]
W=0.30; KAPPA=0.25; SEED=0
DRAWS=[0,1]
EPS_AI=[0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.60,1.00,2.00]   # the ONE knob

def open_run(innate,adj,net,o,g0i,eps,eps_ai,W,kappa,seed):
    """OPEN loop: same serve shape, reads current x, but net is FROZEN (no retrain)."""
    x=innate.clone(); x_star=innate; g=torch.Generator().manual_seed(seed)
    inn=innate.numpy(); inn_std=float(innate.std()); conts=[]
    for t in range(ROUNDS):
        with torch.no_grad(): yhat=torch.clip(net(x)+o,0,1)     # frozen model, reactive to x
        b=kappa*x_star+(1.0-kappa)*x
        gate=(yhat-x).abs()<eps_ai
        z=torch.where(gate,(1.0-W)*b+W*yhat,b); conts.append(float(gate.float().mean()))
        gp.ab_sweep(z,adj,eps,0.0,gen=g); x=z                   # NO retraining
    xn=x.numpy()
    return dict(mean=float(xn.mean()), dr=float(xn.std()/inn_std),
                corr=float(np.corrcoef(xn,inn)[0,1]) if xn.std()>1e-9 else 0.0,
                contact=float(np.mean(conts)))

def cell(prox, eps, eps_ai, runner):
    mm={};dr=[];cor=[];con=[]
    for m in MODELS:
        fm=[]
        for d in DRAWS:
            net,o,g0i=prox[(m,d)]
            r=runner(innate,adj,net,o,g0i,eps,eps_ai,W,KAPPA,SEED)
            fm.append(r["mean"]); dr.append(r["dr"]); cor.append(r["corr"]); con.append(r["contact"])
        mm[m]=float(np.mean(fm))
    sep=float(np.std([mm[m] for m in MODELS]))
    capture=float(np.mean([abs(mm[m]-INN) for m in MODELS]))
    return dict(reach=float(np.mean(con)), dr=float(np.mean(dr)), sep=sep,
                capture=capture, corr=float(np.mean(cor)))

def main():
    global innate,adj,INN
    innate,adj=fv2.ml_action_setup(); INN=float(innate.mean()); inn_std=float(innate.std())
    llm={m:sw.prior(m) for m in MODELS}
    prox={}
    for m in MODELS:
        for d in DRAWS: prox[(m,d)]=ce.make_proxy(innate,llm[m],d)
    print(f"innate mean/std={INN:.3f}/{inn_std:.4f}  W={W} kappa={KAPPA} (FIXED)  knob=eps_AI",flush=True)

    LOOPS={"open":open_run,"closed":ce.closed_run}
    res={}
    for lab,eps in REGIMES:
        for loop,runner in LOOPS.items():
            for eai in EPS_AI:
                res[f"{lab}|{loop}|{eai}"]=dict(regime=lab,loop=loop,eps=eps,eps_ai=eai,**cell(prox,eps,eai,runner))
        print(f"  done regime {lab} (eps_social={eps})",flush=True)
    json.dump(dict(meta=dict(W=W,kappa=KAPPA,innate_mean=round(INN,4),knob="eps_AI",
                             loops=["open(frozen)","closed(FT,beta=1.0)"]),results=res),
              open(f"{HERE}/eps_ai_sweep.json","w"))

    # ---- report: open vs closed side by side ----
    print(f"\n=== SINGLE-KNOB eps_AI: OPEN vs CLOSED (W={W}, kappa={KAPPA} FIXED) ===")
    for lab,eps in REGIMES:
        print(f"\n {lab} peers (eps_social={eps}):   [open | closed]")
        print(f"   {'epsAI':>6} |   reach     |    dr       |    sep       |  capture")
        for eai in EPS_AI:
            o=res[f"{lab}|open|{eai}"]; c=res[f"{lab}|closed|{eai}"]
            print(f"   {eai:>6} | {o['reach']:.2f} | {c['reach']:.2f} | {o['dr']:.2f} | {c['dr']:.2f} "
                  f"| {o['sep']:.3f} | {c['sep']:.3f} | {o['capture']:.3f} | {c['capture']:.3f}")

    # ---- FIGURE: open (dashed) vs closed (solid), color=regime ----
    cols={"tight":"tab:red","medium":"tab:orange","wide":"tab:blue"}
    fig,ax=plt.subplots(1,4,figsize=(20,5))
    panels=[("reach  (fraction clearing AI gate)","reach",None),
            ("dr  (dispersion: amplify>1 / collapse<1)","dr",1.0),
            ("model separation  (std of per-LLM means)","sep",None),
            ("population distortion  |mean−innate|","capture",None)]
    for k,(title,key,hline) in enumerate(panels):
        for lab,eps in REGIMES:
            yo=[res[f"{lab}|open|{eai}"][key] for eai in EPS_AI]
            yc=[res[f"{lab}|closed|{eai}"][key] for eai in EPS_AI]
            ax[k].plot(EPS_AI,yc,"o-",color=cols[lab],lw=2,label=f"{lab} closed")
            ax[k].plot(EPS_AI,yo,"o--",color=cols[lab],lw=1.4,alpha=0.7,label=f"{lab} open")
        if hline is not None: ax[k].axhline(hline,ls="--",c="0.4",lw=1)
        ax[k].set_xscale("log"); ax[k].set_xlabel("ε_AI  (AI gate; narrow → open)")
        ax[k].set_title(title,fontsize=10)
        if k==0: ax[k].legend(fontsize=7,ncol=2)
    fig.suptitle("[exploratory] ε_AI single-knob: OPEN (dashed, frozen model) vs CLOSED (solid, performative loop) "
                 f"— 3 regimes, W={W}, κ={KAPPA}",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{HERE}/eps_ai_sweep_fig.png",dpi=130)
    print(f"\nwrote {HERE}/eps_ai_sweep.json and eps_ai_sweep_fig.png")

if __name__=="__main__": main()
