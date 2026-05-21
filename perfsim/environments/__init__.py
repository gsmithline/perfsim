"""perfsim.environments: concrete Environment implementations.

Two sub-packages mirror the design's top-level siblings (DESIGN.md §4):

- `dynamics/`: closed-form / ODE-style tensor-state envs (FJ, replicator,
  strategic best-response, gaussian shift, accumulating shift,
  stateful population, ...).
- `agent_based/`: per-agent stateful object envs. v1 stub; first concrete
  implementation lands in v2.
"""
