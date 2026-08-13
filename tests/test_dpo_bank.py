"""Unit tests for the matched-randomness DPO bank (2026-08-13).

Covers: write/read roundtrip, corruption detection (candidates, uniforms,
bank seed, round-0 snapshot digest), derived RNG stream disjointness and
determinism, and the reader-path guarantees -- preference rows are rebuilt
from fixed candidates + uniforms with NO global RNG draws, orientation
follows uniform < P(A|judge), and judge-dependent label disagreement is
counted correctly.
"""
import math

import pytest
import torch

from perfsim.learners.lm.dpo import DPOLearner
from perfsim.learners.lm.dpo_bank import (DpoBank, derive_seed, sha_tensor,
                                          state_digest)


def _mk_bank(tmp_path, seed=100):
    return DpoBank(str(tmp_path / "bank"), seed)


def _round_payload(n=8):
    g = torch.Generator().manual_seed(7)
    return dict(
        agent_ids=torch.arange(n),
        prompt_hash="ph",
        cand_a=[f"0.{10 + i}" for i in range(n)],
        cand_b=[f"0.{60 + i}" for i in range(n)],
        parsed_a=[0.10 + i / 100 for i in range(n)],
        parsed_b=[0.60 + i / 100 for i in range(n)],
        valid=[True] * n,
        tie=[False] * n,
        uniforms=torch.rand(n, generator=g, dtype=torch.float64),
        writer_orient=torch.tensor([i % 2 == 0 for i in range(n)]),
    )


def test_roundtrip(tmp_path):
    bank = _mk_bank(tmp_path)
    rec_w = bank.write_round(3, **_round_payload())
    rec_r = bank.read_round(3)
    assert rec_r["cand_hash"] == rec_w["cand_hash"]
    assert rec_r["cand_a"] == rec_w["cand_a"]
    assert torch.equal(rec_r["uniforms"], rec_w["uniforms"])
    assert torch.equal(rec_r["writer_orient"], rec_w["writer_orient"])


def test_corrupted_candidates_fail(tmp_path):
    bank = _mk_bank(tmp_path)
    bank.write_round(0, **_round_payload())
    rec = torch.load(bank._round_path(0), weights_only=False)
    rec["cand_a"][2] = "0.99"          # tamper without rehashing
    torch.save(rec, bank._round_path(0))
    with pytest.raises(ValueError, match="candidate hash"):
        bank.read_round(0)


def test_corrupted_uniforms_fail(tmp_path):
    bank = _mk_bank(tmp_path)
    bank.write_round(0, **_round_payload())
    rec = torch.load(bank._round_path(0), weights_only=False)
    rec["uniforms"][0] = 0.123456789
    torch.save(rec, bank._round_path(0))
    with pytest.raises(ValueError, match="uniform hash"):
        bank.read_round(0)


def test_wrong_bank_seed_fails(tmp_path):
    bank = _mk_bank(tmp_path, seed=100)
    bank.write_round(0, **_round_payload())
    with pytest.raises(ValueError, match="bank_seed"):
        DpoBank(str(tmp_path / "bank"), 101).read_round(0)


def test_round0_digest_corruption_fails(tmp_path):
    bank = _mk_bank(tmp_path)
    snap = {"w": torch.randn(4, 4)}
    bank.write_round0_state(snapshot=snap, preds=torch.rand(8),
                            x0=torch.rand(8), labels_hash="lh", stats={})
    st = torch.load(bank.round0_path(), weights_only=False)
    st["snapshot"]["w"][0, 0] += 1.0
    torch.save(st, bank.round0_path())
    with pytest.raises(ValueError, match="snapshot digest"):
        bank.read_round0_state()


def test_derived_streams_disjoint_and_deterministic():
    seeds = {derive_seed(100, t, lab)
             for t in range(30) for lab in ("cand", "bt", "train")}
    assert len(seeds) == 90                      # no collisions across streams
    assert derive_seed(100, 5, "bt") == derive_seed(100, 5, "bt")
    assert derive_seed(100, 5, "bt") != derive_seed(101, 5, "bt")
    assert derive_seed(100, 5, "bt") != derive_seed(100, 6, "bt")


class _FakeSelf:
    """Just enough of a DPOLearner for _rows_from_candidates."""
    _tau = 12.0


def _rows(xj, uniforms, n=8):
    pay = _round_payload(n)
    return DPOLearner._rows_from_candidates(
        _FakeSelf(), [f"p{i}" for i in range(n)], pay["cand_a"],
        pay["cand_b"], pay["parsed_a"], pay["parsed_b"], pay["valid"],
        pay["tie"], uniforms, xj, None, None)


def test_reader_path_draws_no_global_rng():
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    g = torch.Generator().manual_seed(7)
    _rows(torch.full((8,), 0.2), torch.rand(8, generator=g,
                                            dtype=torch.float64))
    assert torch.equal(before, torch.get_rng_state()), \
        "preference reconstruction consumed global RNG"


def test_orientation_follows_uniform_and_judge():
    n = 8
    g = torch.Generator().manual_seed(7)
    uniforms = torch.rand(n, generator=g, dtype=torch.float64)
    pay = _round_payload(n)
    # judge at 0.0: candidate A (~0.1x) is closer for every agent
    rows_a, orient_a, _ = _rows(torch.zeros(n), uniforms)
    for i in range(n):
        ya, yb = pay["parsed_a"][i], pay["parsed_b"][i]
        du = -abs(ya - 0.0) + abs(yb - 0.0)
        p_a = 1.0 / (1.0 + math.exp(-12.0 * du))
        assert bool(orient_a[i]) == (float(uniforms[i]) < p_a)
    # judge at 1.0: candidate B is closer -> orientation flips for most
    _, orient_b, _ = _rows(torch.ones(n), uniforms)
    dis = int((orient_a != orient_b).sum())
    assert dis > 0, "opposite judges produced identical labels"


def test_disagreement_counted_against_writer(tmp_path):
    n = 8
    g = torch.Generator().manual_seed(7)
    uniforms = torch.rand(n, generator=g, dtype=torch.float64)
    _, orient_writer, _ = _rows(torch.zeros(n), uniforms)
    _, orient_reader, _ = _rows(torch.ones(n), uniforms)
    dis = int((orient_reader != orient_writer).sum())
    assert 0 < dis <= n


def test_state_digest_orders_stably():
    a = {"x": torch.ones(3), "y": torch.zeros(2)}
    b = {"y": torch.zeros(2), "x": torch.ones(3)}
    assert state_digest(a) == state_digest(b)
    b["x"] = b["x"] + 1e-6
    assert state_digest(a) != state_digest(b)
