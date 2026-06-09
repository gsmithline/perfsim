# `perfsim.scenarios.recommender`

A performative recommender ecosystem. The deployed model **theta** is the
platform ranker: it scores items, exposure is allocated by softmax over scores,
a batched user population chooses items via a differentiable multinomial-logit
**biased by exposure**, and the logged engagement becomes theta's training data
for the next round. Meanwhile user interests **drift** toward whatever was
surfaced. Both the labels (exposure bias) and the population (taste drift) are
performative.

Design follows the **entity / behavior / story** decomposition of RecSim NG
(Mladenov et al. 2021, [arXiv:2103.08057](https://arxiv.org/abs/2103.08057)),
reimplemented in torch and wrapped to perfsim's `Environment` /
`FullyDifferentiable` contract. The user population is modeled RecSim NG-style:
one batched `(N, D)` interest tensor with heterogeneity drawn from a community
mixture, not one object per user.

> **Fidelity note.** This is a perfsim tool *inspired by* RecSim NG, not a port.
> In particular the producer dynamics use a **replicator** over producer
> availability share (perfsim's `ReplicatorWorld` math), whereas RecSim NG's own
> provider model accumulates unnormalized discounted engagement and scales item
> production by it. The qualitative phenomena match; the mechanism differs.

## Demand side — `RecommenderEcosystemWorld`

One epoch (Algorithm 1; theta queried once, frozen, population evolves `n_steps`):

1. scores  $\hat{y} = \theta(\text{features})$  — query theta once, frozen
2. expose  $u = \mathrm{softmax}(\beta\,\hat{y})$  — slate propensity, (K,)
3. choose  $c = \mathrm{softmax}((\text{interest}\cdot x + \alpha\log u)/\text{temp})$  — (N, K), MNL
4. label   $y = \mathrm{mean}_n\, c \cdot \text{appeal}$  — logged, exposure-biased engagement
5. drift   $\text{interest} \leftarrow \text{interest} + \eta\,(c\,@\,x - \text{interest})$
6. train   $\theta$ on $\{\text{features}, y\}$

`epoch_size > 1` evolves tastes for several ticks under the frozen ranking
before retraining; `epoch_size = 1` is the lockstep PP loop.

### Knobs

| Param | Meaning |
|-------|---------|
| `alpha` | exposure bias — how much being surfaced inflates engagement (label performativity) |
| `beta` | ranker sharpness — large beta concentrates exposure (winner-take-all) |
| `temp` | choice temperature |
| `eta` | taste-drift rate (population performativity) |
| `n_communities`, `community_spread` | user heterogeneity / community structure |

### Hidden ground truth

`interest0` (the initial population) is never trained on. `world.hidden_quality()`
returns the true item relevance it implies. Probes compare theta / engagement to
this truth.

### Probes (`probes.py`)

The headline performative signal is **`engagement_faithfulness`** with `alpha=0`
as the baseline: faithfulness is flat with no exposure bias and settles lower
once the feedback loop is on (empirically ~0.33 -> ~0.22 at `alpha>=4`, `eta=0`).

| Probe | Story |
|-------|-------|
| `engagement_faithfulness(world)` | corr(induced engagement, true relevance). **The performative data distortion.** Flat at `alpha=0`; settles lower for `alpha>0`. |
| `exposure_faithfulness(world)` | corr(exposure, true relevance). Does attention reach truly-relevant items? Degrades under bias. |
| `exposure_gini(world)` | feed concentration, 0 = uniform -> 1 = winner-take-all |
| `engagement_entropy(data)` | consumption diversity (filter-bubble collapse) |
| `interest_drift(world)` | mean movement of user interests (population channel; isolate with `alpha=0`) |
| `induced_calibration(model, data)` | MSE on logged engagement (platform's own accuracy) |
| `alignment_induced(model, data)` | corr(theta, engagement). **Diagnostic only** — ~regression fit quality, since theta is MSE-trained on this target. *Not* a performativity signal. |
| `alignment_truth(model, world)` | corr(theta, true relevance). Loose: `hidden_quality` is linear affinity while engagement is nonlinear, so absolute values aren't comparable to 1 — use for relative/over-time reads only. |

The two channels are separable: sweep `alpha` with `eta=0` to isolate the
exposure (label) channel; sweep `eta` with `alpha=0` for the taste-drift
(population) channel.

## Supply side — `ProducerEcosystemWorld`

Adds a content-producer population. Each of the K items is a producer (a topic)
carrying a **replicating availability mass** $a \in \Delta^K$. The loop extends
the demand-side world:

- expose  $u = \mathrm{softmax}(\beta\,\hat{y} - \text{boost}\cdot\log a)$  — optional ecosystem-aware boost
- choose  adds a supply term: $\dots + \gamma_{\text{supply}}\log a_k$
- per-producer engagement $y$ -> fitness $f_k = \text{rate}\,(y_k/\overline{y} - 1) - \kappa$
- replicator update $a \leftarrow a(1+f)/\langle a, 1+f\rangle$

So theta's exposure shapes which producers get engagement, which (via the
replicator) shapes which producers survive, which shapes next round's content
and data — a supply-side performative loop.

### Extra knobs

| Param | Meaning |
|-------|---------|
| `gamma_supply` | supply feedback (availability -> consumption). **Producer channel is only performative when > 0.** |
| `repl_rate` | replicator speed. Default `0.1` gives gradual, observable collapse; high values fixate almost immediately. |
| `kappa` | flat production cost |
| `boost` | ecosystem-aware exposure that amplifies small producers. `0` = myopic ranker. |

### Extra probes

| Probe | Story |
|-------|-------|
| `producer_diversity(world)` | effective # live producers = `exp(entropy(availability))`. Collapses under rich-get-richer. |
| `surviving_producers(world)` | count of producers above a survival threshold |
| `user_welfare(world)` | mean realized TRUE affinity of consumption; falls as preferred topics die |

### Observed dynamics

A myopic ranker (`boost=0`) drives **rich-get-richer**: producer diversity falls
from ~K to 1 (a single producer captures all attention). An ecosystem-aware
**boost** counteracts it — e.g. at `repl_rate=0.1`, K=30: `boost=0 -> diversity 1`,
`boost=1 -> ~22`, `boost=2 -> ~29`. Note pure replicator with relative fitness
fixates to a single vertex; a positive `kappa` or `boost` is what yields an
interior, diverse equilibrium.

## Competing platforms — `CompetingRecommendersWorld`

Several rankers over one shared, drifting population. Each user splits attention
across the platforms; each platform trains only on the engagement its own share
of attention produced. `eta_mob` moves attention between platforms over time by
multiplicative weights on satisfaction. Run it with `run_competition` (the
single-model `Simulator` loop fanned over platforms). Use it to ask whether
competing platforms differentiate or converge over a malleable population.

## Quickstart

```python
from perfsim.scenarios.recommender import build_recommender_env, build_producer_env
from perfsim.models import LinearModel
from perfsim.losses import MSELoss
from perfsim.learners import ERMLearner
from perfsim.simulator import Simulator

# demand-side exposure bias
env = build_recommender_env(n_items=30, n_users=200, dim=8, alpha=4.0, beta=4.0, eta=0.15)

# supply-side producer ecosystem (myopic ranker -> diversity collapse)
env = build_producer_env(n_items=30, n_users=200, dim=8, gamma_supply=1.0, boost=0.0)

model = LinearModel(8, 1)
loss = MSELoss()
sim = Simulator(env=env, learner=ERMLearner(model, loss), loss=loss)
hist = sim.run(n_rounds=40, epoch_size=1, seed=0)
```

## Differentiability / PerfGD

All behaviors are smooth, so both worlds satisfy `FullyDifferentiable`
(`grad_sample`, `grad_step`, `grad_run`). With `PerfGDLearner`, the autograd path
`theta -> exposure -> choice -> engagement` lets you optimize the performative
objective directly (e.g. via `env.grad_engagement`).

## Files

- `entities.py` — `Corpus`, `UserPopulation`, community-mixture samplers
- `choice.py` — differentiable MNL choice (with optional supply term)
- `behaviors.py` — `exposure`, `engagement`, `drift_interest`, `replicator_step`, `producer_fitness`, `exposure_with_boost`
- `env.py` — `RecommenderEcosystemWorld` (demand side)
- `env_producers.py` — `ProducerEcosystemWorld` (supply side)
- `probes.py` — ecosystem-health metrics
- `story.py` — `build_recommender_env`, `build_producer_env`
