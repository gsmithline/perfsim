"""Shared candidate/uniform bank for MATCHED-RANDOMNESS paired DPO runs.

A pair = one WRITER arm (closed feedback) + one READER arm (open feedback)
run sequentially on the same GPU. The writer records, per round, everything
stochastic about preference construction:

  - ordered agent ids and a hash of the exact prompt strings,
  - raw candidate A/B strings, their parsed values, validity + tie masks,
  - ONE Bradley-Terry uniform per agent (drawn from a dedicated CPU
    generator, never the global stream),
  - the writer's own chosen/rejected orientation bits,
  - content hashes over all of the above.

The reader replays the identical candidates and uniforms and recomputes the
preference ORIENTATION with its own judge -- it never reuses the writer's
labels. The reader path draws NO random numbers (unit-tested via global RNG
state comparison), so from round 1 onward the only designed difference
between the arms is the judge:

  closed -> the deployment-shaped population;  open -> the no-platform twin.

RNG streams are derived, disjoint, and deterministic:
  candidate generation : derive_seed(bank_seed, round, "cand")  (global
                         torch/cuda seed set right before sampling)
  BT uniforms          : derive_seed(bank_seed, round, "bt")    (dedicated
                         CPU torch.Generator)
  DPO training         : derive_seed(train_seed, round, "train") -- the SAME
                         across matched arms and across bank seeds, so the
                         production wave varies preference sampling, not
                         optimizer randomness.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def derive_seed(base: int, round_idx: int, label: str) -> int:
    """Deterministic, stream-disjoint 63-bit seed from (base, round, label)."""
    h = hashlib.sha256(f"{int(base)}|{int(round_idx)}|{label}".encode())
    return int.from_bytes(h.digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def sha_strings(items) -> str:
    h = hashlib.sha256()
    for s in items:
        b = s.encode() if isinstance(s, str) else bytes(s)
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_digest(snapshot: dict) -> str:
    """Order-stable digest of a trainable-state snapshot {name: tensor}."""
    h = hashlib.sha256()
    for name in sorted(snapshot):
        h.update(name.encode())
        h.update(snapshot[name].detach().cpu().contiguous()
                 .to(torch.float32).numpy().tobytes())
    return h.hexdigest()


class DpoBank:
    """Directory-backed per-round bank. Writer and reader share the path."""

    def __init__(self, path: str, bank_seed: int):
        self.dir = Path(path)
        self.bank_seed = int(bank_seed)

    def _round_path(self, t: int) -> Path:
        return self.dir / f"round_{t:03d}.pt"

    def meta_path(self) -> Path:
        return self.dir / "bank_meta.json"

    def round0_path(self) -> Path:
        return self.dir / "round0_state.pt"

    # ------------------------------------------------------------- writer
    def write_meta(self, **kw) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        kw = dict(kw)
        kw["bank_seed"] = self.bank_seed
        self.meta_path().write_text(json.dumps(kw, indent=2, sort_keys=True))

    def write_round(self, t: int, *, agent_ids, prompt_hash, cand_a, cand_b,
                    parsed_a, parsed_b, valid, tie, uniforms,
                    writer_orient) -> dict:
        self.dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "round": int(t),
            "bank_seed": self.bank_seed,
            "agent_ids": torch.as_tensor(agent_ids, dtype=torch.long),
            "prompt_hash": prompt_hash,
            "cand_a": list(cand_a),
            "cand_b": list(cand_b),
            "parsed_a": torch.as_tensor(parsed_a, dtype=torch.float64),
            "parsed_b": torch.as_tensor(parsed_b, dtype=torch.float64),
            "valid": torch.as_tensor(valid, dtype=torch.bool),
            "tie": torch.as_tensor(tie, dtype=torch.bool),
            "uniforms": torch.as_tensor(uniforms, dtype=torch.float64),
            "writer_orient": torch.as_tensor(writer_orient, dtype=torch.bool),
        }
        rec["cand_hash"] = sha_strings(
            [str(int(i)) for i in rec["agent_ids"]] + rec["cand_a"]
            + rec["cand_b"])
        rec["unif_hash"] = sha_tensor(rec["uniforms"])
        rec["orient_hash"] = sha_tensor(rec["writer_orient"])
        torch.save(rec, self._round_path(t))
        return rec

    def write_round0_state(self, *, snapshot, preds, x0, labels_hash,
                           stats) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        torch.save({"snapshot": snapshot,
                    "preds": preds.detach().cpu(),
                    "x0": x0.detach().cpu(),
                    "labels_hash": labels_hash,
                    "stats": dict(stats),
                    "snapshot_digest": state_digest(snapshot)},
                   self.round0_path())

    # ------------------------------------------------------------- reader
    def read_meta(self) -> dict:
        return json.loads(self.meta_path().read_text())

    def read_round(self, t: int) -> dict:
        rec = torch.load(self._round_path(t), map_location="cpu",
                         weights_only=False)
        if int(rec["round"]) != int(t):
            raise ValueError(f"bank round file {t} carries round={rec['round']}")
        if int(rec["bank_seed"]) != self.bank_seed:
            raise ValueError(f"bank_seed mismatch: file {rec['bank_seed']} "
                             f"vs env {self.bank_seed}")
        got_c = sha_strings([str(int(i)) for i in rec["agent_ids"]]
                            + rec["cand_a"] + rec["cand_b"])
        if got_c != rec["cand_hash"]:
            raise ValueError(f"bank round {t}: candidate hash mismatch "
                             f"(corrupted candidates)")
        if sha_tensor(rec["uniforms"]) != rec["unif_hash"]:
            raise ValueError(f"bank round {t}: uniform hash mismatch "
                             f"(corrupted uniforms)")
        return rec

    def read_round0_state(self) -> dict:
        st = torch.load(self.round0_path(), map_location="cpu",
                        weights_only=False)
        if state_digest(st["snapshot"]) != st["snapshot_digest"]:
            raise ValueError("bank round-0 snapshot digest mismatch "
                             "(corrupted adapter state)")
        return st

    def n_rounds(self) -> int:
        return len(list(self.dir.glob("round_*.pt")))


def bank_from_env() -> "DpoBank | None":
    """DPO_BANK_MODE in {write, read} + DPO_BANK_DIR + DPO_BANK_SEED -> bank.
    Anything else -> None (the legacy path; behavior byte-identical)."""
    mode = os.environ.get("DPO_BANK_MODE", "")
    if mode not in ("write", "read"):
        return None
    return DpoBank(os.environ["DPO_BANK_DIR"],
                   int(os.environ["DPO_BANK_SEED"]))
