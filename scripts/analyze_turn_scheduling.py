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
  away_cache.csv     away-time buckets -> prefix-cache hit ratio + re-prefilled
                     tokens (PROFILE-ONLY: works without worker logs). Measures
                     the KV-eviction cost of leaving the GPU between turns:
                     away_s = llm.start(N) - llm.end(N-1); cache_hit_ratio =
                     tokens.cache.read / (tokens.input + tokens.cache.read).
  stdout             match-quality report + by-tool table + away/cache table
                     with pearson r(away_s, cache_hit_ratio)
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
    llm_start_ts: float | None = None  # llm.start event ts (unix s)
    llm_end_ts: float | None = None    # llm.end event ts (unix s)
    tool_names: list[str] = field(default_factory=list)  # tools run IN this step
    prev_tools: tuple[str, ...] = ()   # tools of step-1 (adjacency key)
    away_s: float | None = None        # llm.start(N) - llm.end(N-1): time the
                                       # session was OFF the GPU (tool exec +
                                       # scaffold) before this turn re-entered
    # Proxy for eviction pressure DURING the away window: sum of
    # (input + output) tokens of OTHER sessions' turns whose llm.end fell
    # inside this turn's away window — i.e. how much new KV concurrent
    # traffic allocated while this session was off the GPU. vLLM's LRU
    # evicts by recency, not by session liveness, so this displaced
    # traffic is what pushes a still-active session's blocks out.
    away_displaced_tokens: int | None = None

    @property
    def effective_input(self) -> int | None:
        if self.input_tokens is None:
            return None
        return self.input_tokens + self.cache_read

    @property
    def cache_hit_ratio(self) -> float | None:
        """Fraction of this turn's prompt served from prefix cache.

        Profile `tokens.input` already has cache tokens subtracted
        (CLAUDE.md: ISL = input + cache.read), so `input` IS the
        re-prefilled token count and `cache.read` the reused count.
        """
        eff = self.effective_input
        if not eff:
            return None
        return self.cache_read / eff

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
            if ev_type == "llm.start":
                t = turns.setdefault(key, TurnRec(session_id=sid, step=int(step)))
                if ev.get("ts") is not None:
                    t.llm_start_ts = float(ev["ts"])
            elif ev_type == "llm.end":
                t = turns.setdefault(key, TurnRec(session_id=sid, step=int(step)))
                if ev.get("ts") is not None:
                    t.llm_end_ts = float(ev["ts"])
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
    # adjacency: preceding tools of turn N = tools executed in step N-1;
    # away_s = llm.start(N) - llm.end(N-1) = time the session spent OFF the
    # GPU (tool execution + scaffold) before re-entering.
    out: list[TurnRec] = []
    for (sid, step), t in sorted(turns.items()):
        prev = turns.get((sid, step - 1))
        t.prev_tools = tuple(prev.tool_names) if prev else ()
        if prev is not None and prev.llm_end_ts is not None and t.llm_start_ts is not None:
            gap = t.llm_start_ts - prev.llm_end_ts
            if gap >= 0:
                t.away_s = gap
        # only keep turns that actually had an LLM call
        if t.llm_wall_s is not None or t.output_tokens is not None or t.recv_ts is not None:
            out.append(t)
    _fill_displaced_tokens(out)
    return out


def _fill_displaced_tokens(turns: list[TurnRec]) -> None:
    """For each turn with an away window, sum the new-KV tokens
    (input + output) of OTHER sessions' turns whose llm.end fell inside
    [llm.end(N-1), llm.start(N)]. This approximates how much KV was
    allocated by concurrent traffic while the session was away — the
    displacement pressure that evicts its blocks under LRU."""
    import bisect
    events = sorted(
        (t.llm_end_ts, t.session_id,
         (t.input_tokens or 0) + (t.output_tokens or 0))
        for t in turns if t.llm_end_ts is not None
    )
    ts_list = [e[0] for e in events]
    for t in turns:
        if t.away_s is None or t.llm_start_ts is None:
            continue
        lo = t.llm_start_ts - t.away_s
        hi = t.llm_start_ts
        i = bisect.bisect_left(ts_list, lo)
        j = bisect.bisect_right(ts_list, hi)
        t.away_displaced_tokens = sum(
            tok for ts, sid, tok in events[i:j] if sid != t.session_id
        )


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


def _denom_ms(t: "TurnRec") -> float | None:
    """End-to-end denominator for queue_share, in ms.

    Prefer dynamo's server-side elapsed_s (nvext.timing); when that is
    absent (non-dynamo provider, or a run where the frontend didn't emit
    timing) fall back to the client-side LLM stream wall llm_wall_s so
    queue_share stays computable instead of collapsing to nan.
    """
    if t.elapsed_s and t.elapsed_s > 0:
        return t.elapsed_s * 1000.0
    if t.llm_wall_s and t.llm_wall_s > 0:
        return t.llm_wall_s * 1000.0
    return None


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
            r.total_queue_ms / d
            for t, r in items
            if (d := _denom_ms(t)) is not None
        ]
        small = [(t, r) for t, r in items
                 if t.output_tokens is not None and t.output_tokens <= small_tokens]
        large = [(t, r) for t, r in items
                 if t.output_tokens is not None and t.output_tokens > small_tokens]

        def share_of(sub):
            vals = [r.total_queue_ms / d
                    for t, r in sub if (d := _denom_ms(t)) is not None]
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


# ---------- away-time vs cache-hit (KV eviction cost of leaving the GPU) ----------

_AWAY_BUCKETS = [
    (0.0, 1.0, "<1s"),
    (1.0, 5.0, "1-5s"),
    (5.0, 15.0, "5-15s"),
    (15.0, 60.0, "15-60s"),
    (60.0, math.inf, ">60s"),
]


def away_cache_rows(turns: list[TurnRec]) -> list[dict]:
    """Bucket non-first turns by away_s -> cache-hit / re-prefill stats.

    The claim being tested: the longer a session is off the GPU (tool
    execution between turns), the more of its KV blocks get evicted by
    concurrent traffic -> lower prefix-cache hit ratio -> more tokens
    re-prefilled on re-entry. This is the direct measurement of the
    "host-GPU transition loss" beyond queue wait.
    """
    eligible = [t for t in turns
                if t.away_s is not None and t.cache_hit_ratio is not None]
    rows = []
    for lo, hi, label in _AWAY_BUCKETS:
        sub = [t for t in eligible if lo <= t.away_s < hi]
        hits = [t.cache_hit_ratio for t in sub]
        reprefill = [float(t.input_tokens) for t in sub if t.input_tokens is not None]
        rows.append({
            "away_bucket": label,
            "count": len(sub),
            "cache_hit_p50": _pct(hits, 0.50) if hits else math.nan,
            "cache_hit_p10": _pct(hits, 0.10) if hits else math.nan,
            "reprefill_tokens_p50": _pct(reprefill, 0.50) if reprefill else math.nan,
            "reprefill_tokens_p90": _pct(reprefill, 0.90) if reprefill else math.nan,
        })
    return rows


def _pearson(pairs: list[tuple[float, float]]) -> tuple[float, int]:
    n = len(pairs)
    if n < 2:
        return math.nan, n
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return math.nan, n
    return cov / math.sqrt(vx * vy), n


def away_cache_correlation(turns: list[TurnRec]) -> tuple[float, int]:
    """Pearson r between away_s and cache_hit_ratio over non-first turns."""
    return _pearson([(t.away_s, t.cache_hit_ratio) for t in turns
                     if t.away_s is not None and t.cache_hit_ratio is not None])


def displaced_cache_correlation(turns: list[TurnRec]) -> tuple[float, int]:
    """Pearson r between away_displaced_tokens and cache_hit_ratio.

    Stronger causal proxy than away_s alone: what evicts blocks is not
    time itself but the KV allocated by OTHER traffic during the away
    window (LRU displaces by allocation pressure)."""
    return _pearson([
        (float(t.away_displaced_tokens), t.cache_hit_ratio) for t in turns
        if t.away_displaced_tokens is not None and t.cache_hit_ratio is not None
    ])


def write_away_cache_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["away_bucket", "count"])
        w.writeheader()
        w.writerows(rows)


def print_away_cache(rows: list[dict], r: float, n: int) -> None:
    print("\nAway-time vs prefix-cache hit (non-first turns; away_s = "
          "llm.start(N) - llm.end(N-1)):")
    print(f"{'away':>8s} {'n':>6s} {'hit_p50':>8s} {'hit_p10':>8s} "
          f"{'reprefill_p50':>13s} {'reprefill_p90':>13s}")
    for row in rows:
        print(f"{row['away_bucket']:>8s} {row['count']:6d} "
              f"{row['cache_hit_p50']:8.3f} {row['cache_hit_p10']:8.3f} "
              f"{row['reprefill_tokens_p50']:13.0f} {row['reprefill_tokens_p90']:13.0f}")
    print(f"pearson r(away_s, cache_hit_ratio) = {r:.3f}  (n={n})")
    if not math.isnan(r) and r < -0.2:
        print("  -> negative correlation: longer off-GPU time costs prefix "
              "cache (KV evicted while away); re-entry pays extra prefill.")


def print_displaced(r: float, n: int) -> None:
    print(f"pearson r(away_displaced_tokens, cache_hit_ratio) = {r:.3f}  (n={n})")
    if not math.isnan(r) and r < -0.2:
        print("  -> displacement pressure (other sessions' KV allocation "
              "during the away window) drives the eviction, not elapsed "
              "time per se — LRU is traffic-driven.")


# ---------- outputs ----------


def write_turn_csv(path: Path, match: MatchResult) -> None:
    cols = [
        "session_id", "step", "request_id", "match", "prev_tools",
        "output_tokens", "input_tokens", "llm_wall_s", "elapsed_s",
        "prefill_queue_ms", "decode_queue_ms", "total_queue_ms",
        "queue_share", "queue_share_basis",
        "away_s", "away_displaced_tokens", "cache_read", "cache_hit_ratio",
        "cur_tools",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for t, rid, rec in match.matched:
            d = _denom_ms(t)
            share = rec.total_queue_ms / d if d is not None else ""
            basis = ("elapsed_s" if t.elapsed_s and t.elapsed_s > 0
                     else "llm_wall_s" if d is not None else "")
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
                "queue_share_basis": basis,
                "away_s": t.away_s if t.away_s is not None else "",
                "away_displaced_tokens": (t.away_displaced_tokens
                                          if t.away_displaced_tokens is not None else ""),
                "cache_read": t.cache_read,
                "cache_hit_ratio": (t.cache_hit_ratio
                                    if t.cache_hit_ratio is not None else ""),
                "cur_tools": "+".join(sorted(set(t.tool_names))),
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
    # Coverage of the per-turn signal so nan columns are self-explanatory:
    # prefill=0 usually means agg/colocation (only decode records) OR the
    # prefill worker log wasn't in --logs; elapsed=0 means the profile had
    # no dynamo.nvext.timing (queue_share then falls back to llm_wall_s).
    if n:
        has_pf = sum(1 for _t, _r, rec in match.matched if rec.prefill_queue_ms is not None)
        has_dc = sum(1 for _t, _r, rec in match.matched if rec.decode_queue_ms is not None)
        has_el = sum(1 for t, _r, _rec in match.matched if t.elapsed_s and t.elapsed_s > 0)
        has_wall = sum(1 for t, _r, _rec in match.matched if t.llm_wall_s and t.llm_wall_s > 0)
        print(f"coverage: prefill_queue {has_pf}/{n}  decode_queue {has_dc}/{n}  "
              f"elapsed_s {has_el}/{n}  llm_wall_s {has_wall}/{n}")
        if has_pf == 0:
            print("NOTE: no prefill queue records matched — agg/colocation "
                  "deployment (decode-only), or prefill worker log missing "
                  "from --logs. The interesting scheduler wait is usually on "
                  "prefill.", file=sys.stderr)
        if has_el == 0:
            print("NOTE: no dynamo elapsed_s in profiles — queue_share uses "
                  "llm_wall_s as the denominator (see queue_share_basis "
                  "column).", file=sys.stderr)
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
    out_dir = args.out or (args.profiles / "turn_sched")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Away-time vs cache-hit is PROFILE-ONLY (no worker logs needed): the
    # KV-eviction cost of leaving the GPU between turns.
    ac_rows = away_cache_rows(turns)
    r, n_ac = away_cache_correlation(turns)
    rd, n_d = displaced_cache_correlation(turns)
    write_away_cache_csv(out_dir / "away_cache.csv", ac_rows)

    if not sched:
        print("warning: no SCHED_DELAY records parsed from logs — skipping "
              "queue-wait join, reporting away/cache analysis only",
              file=sys.stderr)
        print_away_cache(ac_rows, r, n_ac)
        print_displaced(rd, n_d)
        print(f"\nwrote {out_dir / 'away_cache.csv'}")
        return 0

    have_ids = sum(1 for t in turns if t.request_id)
    if have_ids:
        match = join_exact(turns, sched)
    else:
        isl = load_prompt_isl(args.prompts) if args.prompts else None
        match = join_timestamp(turns, sched, args.tolerance_s, isl, args.isl_band)

    write_turn_csv(out_dir / "turn_sched.csv", match)
    rows = by_tool_rows(match, args.small_tokens)
    write_by_tool_csv(out_dir / "by_tool.csv", rows)
    if args.emit_session_map:
        write_session_map(args.emit_session_map, match)
        print(f"wrote session map: {args.emit_session_map}")

    print_report(match, len(turns), rows, args.small_tokens)
    print_away_cache(ac_rows, r, n_ac)
    print_displaced(rd, n_d)
    print(f"\nwrote {out_dir / 'turn_sched.csv'}")
    print(f"wrote {out_dir / 'by_tool.csv'}")
    print(f"wrote {out_dir / 'away_cache.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
