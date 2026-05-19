# perfsim

General high-performance simulator for performative prediction (PP), with the Agent-to-Agent (A2A) protocol as the native communication substrate between simulated agents.

**Status:** pre-implementation skeleton. No implementations yet. See [DESIGN.md](DESIGN.md) for the architecture, decisions, validation strategy, and phasing.

## Layout

```
perfsim/         # the package; numerical core + A2A agent shell
examples/        # end-to-end runnable scenarios
tests/           # gating tests per DESIGN.md Section 15
DESIGN.md        # full design document (19 sections)
pyproject.toml   # build config and optional extras
```

## Install (when ready)

```bash
pip install -e .             # core only
pip install -e ".[tabular]"  # adds pandas + pyarrow for TabularDataset
pip install -e ".[kaggle]"   # adds kaggle CLI for KaggleDataset (Perdomo replication)
pip install -e ".[a2a]"      # adds a2a-sdk for A2AExecutor (v2)
pip install -e ".[dev]"      # pytest, ruff, mypy
```

Optional extras for v2 and later: `[hf]`, `[trl]`, `[vllm]`, `[agenttorch]`. See `pyproject.toml`.

## Phasing

- **v0**: numerical core; GaussianShiftWorld; ERM and gradient Learners; TensorDataset.
- **v1**: full architecture, supervised only; InProcessExecutor; faithful Perdomo replication via Kaggle GiveMeSomeCredit.
- **v2**: A2A wire (`A2AExecutor` over `a2a-sdk`); RL Learners; LM-backed agents (TRL + vLLM); multi-step Coordinator.
- **v3**: optional CUDA-fused Simulator; AgentTorch adapter; Mendler-Dunner RGD replication; performatively-optimal outer-RL stretch.

See DESIGN.md Section 17 for the full phasing table.
