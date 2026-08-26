"""node_pack.py: whole-node packing reproduces each cell's single-GPU launch."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NP = ROOT / "experiments" / "condor" / "node_pack.py"
MAN = ROOT / "experiments" / "condor" / "nodepack_fig4_anchor_tradeoff.json"


def _load():
    spec = importlib.util.spec_from_file_location("_node_pack", NP)
    m = importlib.util.module_from_spec(spec); sys.modules["_node_pack"] = m
    spec.loader.exec_module(m)
    return m


def test_manifest_covers_every_production_tag_once_and_the_smoke_group():
    man = json.loads(MAN.read_text())
    groups = man["groups"]
    prod = [e["tag"] for g, es in groups.items() if g != "smoke0" for e in es]
    rows = (ROOT / "experiments" / "condor" /
            "configs_pofd_fig4_anchor_tradeoff.txt").read_text().splitlines()
    assert prod == [r.split(",")[0].strip() for r in rows if r.strip()]
    assert len(prod) == 60 and len(set(prod)) == 60
    assert all(1 <= len(es) <= 8 for es in groups.values())
    smoke = [e["tag"] for e in groups["smoke0"]]
    assert len(smoke) == 4 and sum("pofdf4asmk_" in t for t in smoke) == 2 \
        and sum("pofdzsprior_" in t and "_a100_" in t for t in smoke) == 2
    assert man["hardware"] == "NVIDIA A100-SXM4-80GB"


def test_build_cell_reproduces_the_single_gpu_env_and_args():
    np_ = _load()
    man = json.loads(MAN.read_text())
    seen = set()
    for g, es in man["groups"].items():
        for e in es:
            exe, argv, env, row = np_.build_cell(e, ROOT)
            assert exe.endswith("run_one_pokec_gated_idempotent.sh")
            assert argv[0] == e["tag"] and e["tag"] not in seen
            seen.add(e["tag"])
            assert len(argv) == 14                      # the wrapper's positionals
            assert env["DATASET"] == "movielens" and env["ML_TARGET"] == "Action"
            assert "$(" not in " ".join(argv) and not any("$(" in v for v in env.values())
            if e["tag"].startswith("pofdzsprior_"):
                assert "TRAIN_WITNESS" not in env and env["SAVE_RAW_GEN"] == "1"
                assert argv[1] == "frozen"
            else:
                assert env["TRAIN_WITNESS"] == "1" and env["PARSE_MODE"] == "strict"
                assert env["PEER_GATE_MODE"] == "threshold" and env["AB_SWEEPS"] == "1"
                assert env["INNATE_LAMBDA"] == row["lam"] and argv[1] == "sft_kl"
    assert len(seen) == 64


def test_dry_run_of_every_group_launches_nothing_and_exits_zero():
    man = json.loads(MAN.read_text())
    for g in man["groups"]:
        p = subprocess.run([sys.executable, str(NP), str(MAN.relative_to(ROOT)), g,
                            "--dry-run", "--repo", str(ROOT)],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        assert "dry run: nothing launched" in p.stdout
        assert p.stdout.count("gpu=") == len(man["groups"][g])
    q = subprocess.run([sys.executable, str(NP), str(MAN.relative_to(ROOT)),
                        "nope", "--dry-run", "--repo", str(ROOT)],
                       capture_output=True, text=True)
    assert q.returncode == 2


def test_assigned_gpus_follow_condors_visible_devices(monkeypatch):
    np_ = _load()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaa,GPU-bbb,GPU-ccc")
    assert np_.assigned_gpus(2) == ["GPU-aaa", "GPU-bbb"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert np_.assigned_gpus(3) == ["0", "1", "2"]


def test_node_subs_request_exact_node_totals_and_pin_the_pair():
    for pair, mem, hosts in (("a", 2051936, ("g181", "g183")),
                             ("b", 2051934, ("g182", "g184"))):
        sub = (ROOT / "experiments" / "condor" /
               f"at_pofd_fig4_anchor_tradeoff_node_{pair}.sub").read_text()
        assert f"request_memory    = {mem}" in sub
        assert "request_cpus      = 128" in sub and "request_gpus      = 8" in sub
        assert "request_disk      = 22345412368" in sub
        assert f"(TARGET.TotalMemory == {mem})" in sub and "H100" not in sub
        for h in hosts:
            assert f'TARGET.Machine == "{h}.internal.cluster.is.localnet"' in sub
        assert "run_node_pack.sh" in sub and "queue group from" in sub
    smoke = (ROOT / "experiments" / "condor" /
             "at_pofd_fig4_anchor_tradeoff_node_smoke_a.sub").read_text()
    assert "request_memory    = 2051936" in smoke
