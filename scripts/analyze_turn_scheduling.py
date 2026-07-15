#!/usr/bin/env python3
"""Join opencode profile turns with dynamo worker SCHED_DELAY records.

Research question: do turns adjacent to particular tools carry a SMALL
LLM load (few output tokens) yet suffer LARGE scheduling delay under
high input QPS?  If so, those turns are candidates for CPU offloading
(the GPU->CPU->GPU turn transition pays a full scheduler queue wait for
a tiny decode).

Join strategy (auto-detected per run):
  exact     profile llm.end carries `request_id` (the dynamo Context
            UUID; recorded by the updated opencode-profile.patch from
            the response id `chatcmpl-<uuid>`) -> dict join.
  timestamp fallback for runs captured BEFORE the patch update: match
            profile `dynamo.request_received_unix_s` against prefill
            SCHED_DELAY `queued_ts` with a tolerance window, greedy
            1:1 assignment by |dt| (optionally corroborated by prompt
            -dump num_prompt_tokens vs profile input tokens).

Inputs:
  --profiles <dir>   opencode profile NDJSON dir (one <sessionID>.jsonl per session)
  --logs <path>      worker log file or dir (vllm-*.log) with SCHED_DELAY lines
  --prompts <dir>    optional prompt-dump dir (prompt-*.jsonl) for ISL corroboration
  --tolerance-s      timestamp-match window (default 0.25)
  --isl-band         max |engine_isl - profile_input| in tokens for a
                     timestamp match to be accepted when prompt-dump data
                     is available (default 512; engine ISL includes chat
                     template so it is normally LARGER than profile input)
  --small-tokens     "small turn" cutoff on output tokens (default 64)
  --out <dir>        output dir (default: <profiles>/turn_sched)
  --emit-session-map <csv>  also write request_id,session_id map (consumable
                     by analyze_request_wait.py --session-map)

Outputs (under --out):
  turn_sched.csv     one row per matched turn:
                     session_id,step,request_id,match,prev_tools,output_tokens,
                     input_tokens,llm_wall_s,elapsed_s,prefill_queue_ms,
                     decode_queue_ms,total_queue_ms,queue_share
  by_tool.csv        per preceding-tool aggregation (count, small-turn share,
                     p50/p90/p99 of output_tokens/llm_wall_s/queue metrics,
                     queue_share conditioned small vs large)
  stdout             match-quality report + by-tool table
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Same line format as analyze_request_wait.py, extended with the ts fields
# the dynamo-scheduling-log.patch emits.
_SCHED_RE = re.compile(
    r"SCHED_DELAY\s+request_id=(?P<rid>\S+)\s+role=(?P<role>\S+)\s+"
    r"queue_ms=(?P<queue_ms>[-0-9.eE+]+)"
    r"(?:\s+queued_ts=(?P<queued_ts>[-0-9.eE+]+))?"
    r"(?:\s+scheduled_ts=(?P<scheduled_ts>[-0-9.eE+]+))?"
)


# ---------- profile side ----------


@dataclass
class TurnRec:
    session_id: str
    step: int
    request_id: str | None = None
    recv_ts: float | None = None       # dynamo.request_received_unix_s
    elapsed_s: float | None = None     # dynamo.total_time_ms/1000
    output_tokens: int | None = None
    input_tokens: int | None = None
    cache_read: int = 0
    llm_wall_s: float | None = None    # llm.end duration_s (stream wall)
    tool_names: list[str] = field(default_factory=list)  # tools run IN this step
    prev_tools: tuple[str, ...] = ()   # tools of step-1 (adjacency key)

    @property
    def effective_input(self) -> int | None:
        if self.input_tokens is None:
            return None
        return self.input_tokens + self.cache_read

    @property
    def prev_key(self) -> str:
        return "+".join(sorted(set(self.prev_tools))) if self.prev_tools else "(none)"


def load_turns(profiles_dir: Path) -> list[TurnRec]:
    turns: dict[tuple[str, int], TurnRec] = {}
    for f in sorted(profiles_dir.glob("*.jsonl")):
        sid = f.stem
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev_type = ev.get("ev")
            step = ev.get("step")
            if step is None:
                continue
            key = (sid, int(step))
            if ev_type == "llm.end":
                t = turns.setdefault(key, TurnRec(session_id=sid, step=int(step)))
                rid = ev.get("request_id")
                if rid:
                    t.request_id = str(rid)
                dyn = ev.get("dynamo") or {}
                if isinstance(dyn, dict):
                    if dyn.get("request_received_unix_s") is not None:
                        t.recv_ts = float(dyn["request_received_unix_s"])
                    if dyn.get("elapsed_s") is not None:
                        t.elapsed_s = float(dyn["elapsed_s"])
                if ev.get("duration_s") is not None:
                    t.llm_wall_s = float(ev["duration_s"])
                tokens = ev.get("tokens") or {}
                inp = tokens.get("input")
                out = tokens.get("output")
                if inp is None:  # classic OpenAI usage shape
                    inp = tokens.get("prompt_tokens")
                    out = tokens.get("completion_tokens")
                cache = tokens.get("cache") or {}
                t.input_tokens = inp
                t.output_tokens = out
                t.cache_read = (cache.get("read") if isinstance(cache, dict) else 0) or 0
            elif ev_type == "tool.end":
                t = turns.setdefault(key, TurnRec(session_id=sid, step=int(step)))
                name = ev.get("name")
                if name:
                    t.tool_names.append(str(name))
    # adjacency: preceding tools of turn N = tools executed in step N-1
    out: list[TurnRec] = []
    for (sid, step), t in sorted(turns.items()):
        prev = turns.get((sid, step - 1))
        t.prev_tools = tuple(prev.tool_names) if prev else ()
        # only keep turns that actually had an LLM call
        if t.llm_wall_s is not None or t.output_tokens is not None or t.recv_ts is not None:
            out.append(t)
    return out


# ---------- worker side ----------


@dataclass
class SchedRec:
    prefill_queue_ms: float | None = None
    prefill_queued_ts: float | None = None
    decode_queue_ms: float | None = None
    decode_queued_ts: float | None = None

    @property
    def total_queue_ms(self) -> float:
        return (self.prefill_queue_ms or 0.0) + (self.decode_queue_ms or 0.0)

    @property
    def anchor_ts(self) -> float | None:
        """Timestamp used for fallback matching: prefill entry, else decode."""
        return self.prefill_queued_ts if self.prefill_queued_ts is not None else self.decode_queued_ts


def _iter_log_files(path: Path):
    if path.is_file():
        yield path
        return
    for f in sorted(path.glob("vllm-*.log")):
        yield f


def load_sched(path: Path) -> dict[str, SchedRec]:
    out: dict[str, SchedRec] = {}
    for fpath in _iter_log_files(path):
        with fpath.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _SCHED_RE.search(line)
                if not m:
                    continue
                rec = out.setdefault(m.group("rid"), SchedRec())
                q = float(m.group("queue_ms"))
                qts = float(m.group("queued_ts")) if m.group("queued_ts") else None
                if m.group("role") == "prefill":
                    rec.prefill_queue_ms = q
                    rec.prefill_queued_ts = qts
                else:
                    rec.decode_queue_ms = q
                    rec.decode_queued_ts = qts
    return out


def load_prompt_isl(prompts_dir: Path) -> dict[str, int]:
    """request_id -> num_prompt_tokens from a DYN_PROMPT_DUMP dir."""
    out: dict[str, int] = {}
    for f in sorted(prompts_dir.glob("prompt-*.jsonl")):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("request_id")
            n = rec.get("num_prompt_tokens")
            if rid and n is not None:
                out.setdefault(str(rid), int(n))
    return out


# ---------- matching ----------


@dataclass
class MatchResult:
    matched: list[tuple[TurnRec, str, SchedRec]]  # (turn, request_id, sched)
    mode: str                                     # "exact" | "timestamp"
    unmatched_turns: int = 0
    ambiguous: int = 0
    dt_abs: list[float] = field(default_factory=list)


def join_exact(turns: list[TurnRec], sched: dict[str, SchedRec]) -> MatchResult:
    res = MatchResult(matched=[], mode="exact")
    for t in turns:
        if t.request_id and t.request_id in sched:
            res.matched.append((t, t.request_id, sched[t.request_id]))
        else:
            res.unmatched_turns += 1
    return res


def join_timestamp(
    turns: list[TurnRec],
    sched: dict[str, SchedRec],
    tolerance_s: float,
    isl: dict[str, int] | None = None,
    isl_band: int = 512,
) -> MatchResult:
    res = MatchResult(matched=[], mode="timestamp")
    candidates: list[tuple[float, int, str]] = []  # (|dt|, turn_idx, rid)
    per_turn_cands: dict[int, int] = defaultdict(int)
    for i, t in enumerate(turns):
        if t.recv_ts is None:
            continue
        for rid, rec in sched.items():
            ats = rec.anchor_ts
            if ats is None:
                continue
            dt = abs(ats - t.recv_ts)
            if dt > tolerance_s:
                continue
            if isl and rid in isl and t.effective_input is not None:
                # engine ISL includes the chat template, so it should be
                # >= profile input; reject wildly-off pairings.
                if abs(isl[rid] - t.effective_input) > isl_band:
                    continue
            candidates.append((dt, i, rid))
            per_turn_cands[i] += 1
    res.ambiguous = sum(1 for n in per_turn_cands.values() if n > 1)
    used_turn: set[int] = set()
    used_rid: set[str] = set()
    for dt, i, rid in sorted(candidates):
        if i in used_turn or rid in used_rid:
            continue
        used_turn.add(i)
        used_rid.add(rid)
        res.matched.append((turns[i], rid, sched[rid]))
        res.dt_abs.append(dt)
    res.unmatched_turns = len(turns) - len(used_turn)
    return res


# ---------- aggregation ----------


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return math.nan
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _dist(vals: list[float]) -> dict[str, float]:
    return {
        "p50": _pct(vals, 0.50),
        "p90": _pct(vals, 0.90),
        "p99": _pct(vals, 0.99),
    }


def by_tool_rows(match: MatchResult, small_tokens: int) -> list[dict]:
    groups: dict[str, list[tuple[TurnRec, SchedRec]]] = defaultdict(list)
    for t, _rid, rec in match.matched:
        groups[t.prev_key].append((t, rec))
    rows = []
    for key, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        outs = [float(t.output_tokens) for t, _ in items if t.output_tokens is not None]
        walls = [t.llm_wall_s for t, _ in items if t.llm_wall_s is not None]
        pq = [r.prefill_queue_ms for _, r in items if r.prefill_queue_ms is not None]
        dq = [r.decode_queue_ms for _, r in items if r.decode_queue_ms is not None]
        tq = [r.total_queue_ms for _, r in items]
        shares = [
            r.total_queue_ms / (t.elapsed_s * 1000.0)
            for t, r in items
            if t.elapsed_s and t.elapsed_s > 0
        ]
        small = [(t, r) for t, r in items
                 if t.output_tokens is not None and t.output_tokens <= small_tokens]
        large = [(t, r) for t, r in items
                 if t.output_tokens is not None and t.output_tokens > small_tokens]

        def share_of(sub):
            vals = [r.total_queue_ms / (t.elapsed_s * 1000.0)
                    for t, r in sub if t.elapsed_s and t.elapsed_s > 0]
            return _pct(vals, 0.50) if vals else math.nan

        row = {
            "prev_tools": key,
            "count": len(items),
            "small_share": (len(small) / len(items)) if items else math.nan,
            "queue_share_small_p50": share_of(small),
            "queue_share_large_p50": share_of(large),
        }
        for name, vals in (
            ("output_tokens", outs),
            ("llm_wall_s", walls),
            ("prefill_queue_ms", pq),
            ("decode_queue_ms", dq),
            ("total_queue_ms", tq),
            ("queue_share", shares),
        ):
            d = _dist(vals)
            for k, v in d.items():
                row[f"{name}_{k}"] = v
        rows.append(row)
    return rows


# ---------- outputs ----------


def write_turn_csv(path: Path, match: MatchResult) -> None:
    cols = [
        "session_id", "step", "request_id", "match", "prev_tools",
        "output_tokens", "input_tokens", "llm_wall_s", "elapsed_s",
        "prefill_queue_ms", "decode_queue_ms", "total_queue_ms", "queue_share",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for t, rid, rec in match.matched:
            share = (
                rec.total_queue_ms / (t.elapsed_s * 1000.0)
                if t.elapsed_s and t.elapsed_s > 0 else ""
            )
            w.writerow({
                "session_id": t.session_id,
                "step": t.step,
                "request_id": rid,
                "match": match.mode,
                "prev_tools": t.prev_key,
                "output_tokens": t.output_tokens if t.output_tokens is not None else "",
                "input_tokens": t.effective_input if t.effective_input is not None else "",
                "llm_wall_s": t.llm_wall_s if t.llm_wall_s is not None else "",
                "elapsed_s": t.elapsed_s if t.elapsed_s is not None else "",
                "prefill_queue_ms": rec.prefill_queue_ms if rec.prefill_queue_ms is not None else "",
                "decode_queue_ms": rec.decode_queue_ms if rec.decode_queue_ms is not None else "",
                "total_queue_ms": rec.total_queue_ms,
                "queue_share": share,
            })


def write_by_tool_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_session_map(path: Path, match: MatchResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["request_id", "session_id"])
        for t, rid, _rec in match.matched:
            w.writerow([rid, t.session_id])


def print_report(match: MatchResult, total_turns: int, rows: list[dict],
                 small_tokens: int) -> None:
    n = len(match.matched)
    print(f"join mode: {match.mode}")
    print(f"turns: {total_turns}  matched: {n} "
          f"({100.0 * n / total_turns:.1f}%)" if total_turns else "turns: 0")
    print(f"unmatched turns: {match.unmatched_turns}")
    if match.mode == "timestamp":
        print(f"ambiguous (>=2 candidates in tolerance): {match.ambiguous}")
        if match.dt_abs:
            print(f"|dt| of accepted matches: median={_pct(match.dt_abs, 0.5)*1000:.1f}ms "
                  f"max={max(match.dt_abs)*1000:.1f}ms")
        if total_turns and match.ambiguous / max(1, total_turns) > 0.10:
            print("WARNING: >10% ambiguous matches — re-run with the request_id-"
                  "capturing opencode-profile.patch for an exact join.",
                  file=sys.stderr)
    if not rows:
        return
    print(f"\nPer preceding-tool (small turn = output_tokens <= {small_tokens}):")
    hdr = (f"{'prev_tools':30s} {'n':>5s} {'small%':>7s} "
           f"{'out_p50':>8s} {'llm_p50':>8s} {'pq_p50':>8s} {'dq_p50':>8s} "
           f"{'qshare_p50':>10s} {'qs_small':>9s} {'qs_large':>9s}")
    print(hdr)
    for r in rows:
        print(f"{r['prev_tools'][:30]:30s} {r['count']:5d} "
              f"{100.0 * r['small_share']:6.1f}% "
              f"{r['output_tokens_p50']:8.0f} {r['llm_wall_s_p50']:8.2f} "
              f"{r['prefill_queue_ms_p50']:8.1f} {r['decode_queue_ms_p50']:8.1f} "
              f"{r['queue_share_p50']:10.3f} "
              f"{r['queue_share_small_p50']:9.3f} {r['queue_share_large_p50']:9.3f}")


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", required=True, type=Path)
    ap.add_argument("--logs", required=True, type=Path)
    ap.add_argument("--prompts", type=Path, default=None)
    ap.add_argument("--tolerance-s", type=float, default=0.25)
    ap.add_argument("--isl-band", type=int, default=512)
    ap.add_argument("--small-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--emit-session-map", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.profiles.is_dir():
        print(f"error: profiles dir not found: {args.profiles}", file=sys.stderr)
        return 2
    turns = load_turns(args.profiles)
    sched = load_sched(args.logs)
    if not turns:
        print("error: no turns parsed from profiles", file=sys.stderr)
        return 2
    if not sched:
        print("error: no SCHED_DELAY records parsed from logs", file=sys.stderr)
        return 2

    have_ids = sum(1 for t in turns if t.request_id)
    if have_ids:
        match = join_exact(turns, sched)
    else:
        isl = load_prompt_isl(args.prompts) if args.prompts else None
        match = join_timestamp(turns, sched, args.tolerance_s, isl, args.isl_band)

    out_dir = args.out or (args.profiles / "turn_sched")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_turn_csv(out_dir / "turn_sched.csv", match)
    rows = by_tool_rows(match, args.small_tokens)
    write_by_tool_csv(out_dir / "by_tool.csv", rows)
    if args.emit_session_map:
        write_session_map(args.emit_session_map, match)
        print(f"wrote session map: {args.emit_session_map}")

    print_report(match, len(turns), rows, args.small_tokens)
    print(f"\nwrote {out_dir / 'turn_sched.csv'}")
    print(f"wrote {out_dir / 'by_tool.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
