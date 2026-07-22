#!/usr/bin/env python3
"""SINGLE-KNOB demo #2: sweep ONLY W (AI weight in the blend) under fixed regimes,
OPEN loop vs CLOSED loop overlaid.

W = how heavily the AI recommendation counts when accepted:
     z = where(gate, (1-W) b + W yhat, b).  W=0 -> no AI ; W large -> AI-dominated.
INTENSIVE margin of AI impact (how much), vs eps_AI = EXTENSIVE (how many).

Held constant EXCEPT W:  eps_AI=0.40 (gate wide, reach~1.0 so W is the sole lever),
                         kappa=0.25 (in (0,1)), same anchored LLM proxies.
3 regimes differ ONLY in peer coupling eps_social {0.10,0.30,0.60}. Knob: W 0.0..0.80.
OPEN = model frozen (no retrain, reads x_t) ; CLOSED = model retrains (performative).
Run from perfsim repo root.
"""
import importlib.util, os, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

torch.set_num_threads(6)
HERE=os.path.dirname(os.path.abspath(__file__))
def _load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ce=_load("ce", f"{HERE}/closed_env_sweep.py")
sw,gp,fv2,MODELS,ROUNDS=ce.sw,ce.gp,ce.fv2,ce.MODELS,ce.ROUNDS

REGIMES=[("tight",0.10),("medium",0.30),("wide",0.60)]
EPS_AI=0.40; KAPPA=0.25; SEED=0
DRAWS=[0,1]
WGRID=[0.0,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.80]   # the ONE knob

def open_run(innate,adj,net,o,g0i,eps,eps_ai,W,kappa,seed):
    x=innate.clone(); x_star=innate; g=torch.Generator().manual_seed(seed)
    inn=innate.numpy(); inn_std=float(innate.std()); conts=[]
    for t in range(ROUNDS):
        with torch.no_grad(): yhat=torch.clip(net(x)+o,0,1)
        b=kappa*x_star+(1.0-kappa)*x
        gate=(yhat-x).abs()<eps_ai
        z=torch.where(gate,(1.0-W)*b+W*yhat,b); conts.append(float(gate.float().mean()))
        gp.ab_sweep(z,adj,eps,0.0,gen=g); x=z
    xn=x.numpy()
    return dict(mean=float(xn.mean()), dr=float(xn.std()/inn_std),
                corr=float(np.corrcoef(xn,inn)[0,1]) if xn.std()>1e-9 else 0.0,
                contact=float(np.mean(conts)))

def cell(prox, eps, W, runner):
    mm={};dr=[];cor=[];con=[]
    for m in MODELS:
        fm=[]
        for d in DRAWS:
            net,o,g0i=prox[(m,d)]
            r=runner(innate,adj,net,o,g0i,eps,EPS_AI,W,KAPPA,SEED)
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
    print(f"innate mean/std={INN:.3f}/{inn_std:.4f}  eps_AI={EPS_AI} kappa={KAPPA} (FIXED)  knob=W",flush=True)

    LOOPS={"open":open_run,"closed":ce.closed_run}
    res={}
    for lab,eps in REGIMES:
        for loop,runner in LOOPS.items():
            for W in WGRID:
                res[f"{lab}|{loop}|{W}"]=dict(regime=lab,loop=loop,eps=eps,W=W,**cell(prox,eps,W,runner))
        print(f"  done regime {lab} (eps_social={eps})",flush=True)
    json.dump(dict(meta=dict(eps_AI=EPS_AI,kappa=KAPPA,innate_mean=round(INN,4),knob="W",
                             loops=["open(frozen)","closed(FT,beta=1.0)"]),results=res),
              open(f"{HERE}/w_sweep.json","w"))

    print(f"\n=== SINGLE-KNOB W: OPEN vs CLOSED (eps_AI={EPS_AI}, kappa={KAPPA} FIXED) ===")
    for lab,eps in REGIMES:
        print(f"\n {lab} peers (eps_social={eps}):   [open | closed]")
        print(f"   {'W':>5} |   dr        |    sep       |  capture      |  corr")
        for W in WGRID:
            o=res[f"{lab}|open|{W}"]; c=res[f"{lab}|closed|{W}"]
            print(f"   {W:>5} | {o['dr']:.2f} | {c['dr']:.2f} | {o['sep']:.3f} | {c['sep']:.3f} "
                  f"| {o['capture']:.3f} | {c['capture']:.3f} | {o['corr']:+.2f} | {c['corr']:+.2f}")

    cols={"tight":"tab:red","medium":"tab:orange","wide":"tab:blue"}
    fig,ax=plt.subplots(1,4,figsize=(20,5))
    panels=[("reach  (fraction clearing AI gate)","reach",None),
            ("dr  (dispersion: amplify>1 / collapse<1)","dr",1.0),
            ("model separation  (std of per-LLM means)","sep",None),
            ("population distortion  |mean−innate|","capture",None)]
    for k,(title,key,hline) in enumerate(panels):
        for lab,eps in REGIMES:
            yo=[res[f"{lab}|open|{W}"][key] for W in WGRID]
            yc=[res[f"{lab}|closed|{W}"][key] for W in WGRID]
            ax[k].plot(WGRID,yc,"o-",color=cols[lab],lw=2,label=f"{lab} closed")
            ax[k].plot(WGRID,yo,"o--",color=cols[lab],lw=1.4,alpha=0.7,label=f"{lab} open")
        if hline is not None: ax[k].axhline(hline,ls="--",c="0.4",lw=1)
        ax[k].set_xlabel("W  (AI weight in blend; none → AI-dominated)")
        ax[k].set_title(title,fontsize=10)
        if k==0: ax[k].legend(fontsize=7,ncol=2)
    fig.suptitle("[exploratory] W single-knob: OPEN (dashed, frozen model) vs CLOSED (solid, performative loop) "
                 f"— 3 regimes, ε_AI={EPS_AI}, κ={KAPPA}",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{HERE}/w_sweep_fig.png",dpi=130)
    print(f"\nwrote {HERE}/w_sweep.json and w_sweep_fig.png")

if __name__=="__main__": main()
