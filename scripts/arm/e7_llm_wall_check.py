#!/usr/bin/env python3
"""E7: measurement cross-check — profile LLM wall vs dynamo frontend.log.

Joins each profile `llm.end` event to its frontend.log `request completed`
line by request_id (the dynamo Context UUID the profile patch extracts
from the chunk id) and reports how far apart the two clocks are:

  client_wall_s   profile stream wall. basis "stream_end" = anchored on
                  the AI SDK finish event (either an explicit
                  stream_end_s field, or duration_s on a step whose
                  llm.stream-finish fired / post_stream_overhead_s is
                  set — the current patch's encoding). basis "duration"
                  = legacy first-tool/last-text approximation (known to
                  collapse on buffered turns).
  frontend_s      frontend elapsed_ms / 1000 (server-side wall from HTTP
                  receipt to last chunk)
  dynamo_s        profile llm.end.dynamo.elapsed_s (the SAME dynamo wall
                  riding in-band via nvext.timing) — sanity: must match
                  frontend_s almost exactly; a gap here means a broken
                  join, not a real timing difference

Expected physics: client_wall >= frontend (client adds request upload,
TTFB network, SSE consumption); the delta is the client/network overhead.
A NEGATIVE delta or a huge one usually means the legacy duration_s
under-measurement (buffered turns) — the `basis` column says which
source was used.

Output:
  <out>/wall_check.csv    per-turn join rows
  stdout                  coverage + delta stats (abs + relative),
                          split by basis, worst-10 offenders

Usage:
  scripts/arm/e7_llm_wall_check.py --profiles <dir> \
      --frontend logs/frontend.log [--out e7_wall]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

_ARM = Path(__file__).resolve().parent
_E4_PATH = _ARM / "e4_prefill_decode.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- profiles

def load_llm_ends(profiles: Path) -> list[dict]:
    """One record per llm.end event: {session, step, request_id?,
    stream_end_s?, duration_s?, dynamo_elapsed_s?}."""
    files = [profiles] if profiles.is_file() else \
        sorted(profiles.glob("*.jsonl"))
    out: list[dict] = []
    # (session, step) -> tool names called in that turn, call order
    tools: dict[tuple, list[str]] = {}
    # (session, step) that saw an llm.stream-finish event: on such steps
    # llm.end's duration_s is anchored on the AI SDK finish event (true
    # stream wall) rather than the first-tool/last-text approximation.
    stream_finish: set[tuple] = set()
    for f in files:
        with f.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                et = ev.get("ev")
                if et == "tool.end":
                    key = (ev.get("sessionID"), ev.get("step"))
                    name = ev.get("tool") or ev.get("name")
                    if name:
                        tools.setdefault(key, []).append(str(name))
                    continue
                if et == "llm.stream-finish":
                    stream_finish.add((ev.get("sessionID"), ev.get("step")))
                    continue
                if et != "llm.end":
                    continue
                dyn = ev.get("dynamo") or {}
                out.append({
                    "session": ev.get("sessionID"),
                    "step": ev.get("step"),
                    "request_id": ev.get("request_id"),
                    "stream_end_s": ev.get("stream_end_s"),
                    "duration_s": ev.get("duration_s"),
                    "post_stream_overhead_s": ev.get("post_stream_overhead_s"),
                    "dynamo_elapsed_s": (dyn.get("elapsed_s")
                                         if isinstance(dyn, dict) else None),
                })
    for e in out:
        key = (e["session"], e["step"])
        e["tools"] = "+".join(tools.get(key, []))
        e["stream_anchored"] = key in stream_finish or \
            e["post_stream_overhead_s"] is not None
    return out


# ---------------------------------------------------------------- stats

def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return math.nan
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    idx = q / 100.0 * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _stat_line(name: str, vals: list[float]) -> None:
    if not vals:
        print(f"  {name:<28s} (no samples)")
        return
    mean = sum(vals) / len(vals)
    print(f"  {name:<28s} n={len(vals):<5d} mean={mean:8.3f} "
          f"p50={_pct(vals, 50):8.3f} p90={_pct(vals, 90):8.3f} "
          f"max={max(vals):8.3f} min={min(vals):8.3f}")


CSV_COLS = ["session", "step", "request_id", "tools", "basis",
            "client_wall_s", "frontend_s", "dynamo_s",
            "delta_s", "delta_rel", "dynamo_vs_frontend_s"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", required=True, type=Path,
                    help="profile NDJSON dir (or a single .jsonl)")
    ap.add_argument("--frontend", required=True, type=Path,
                    help="dynamo frontend.log")
    ap.add_argument("--out", type=Path, default=Path("e7_wall"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    e4 = _load("_e7_e4", _E4_PATH)
    # e6-style frontend parse is not needed — e4's row already carries
    # request_id + elapsed_ms, which is all this check consumes.
    fe = {r["request_id"]: r for r in e4.parse_frontend(args.frontend)}
    ends = load_llm_ends(args.profiles)

    n_rid = sum(1 for e in ends if e["request_id"])
    rows: list[dict] = []
    unmatched = 0
    for e in ends:
        rid = e["request_id"]
        if not rid:
            continue
        fr = fe.get(rid)
        if fr is None:
            unmatched += 1
            continue
        if e["stream_end_s"] is not None:
            # older patch naming: explicit stream_end_s field
            basis, cw = "stream_end", float(e["stream_end_s"])
        elif e["duration_s"] is not None:
            # current patch: duration_s is anchored on the AI SDK finish
            # event when llm.stream-finish fired for this step (true
            # stream wall) — only otherwise is it the legacy
            # first-tool/last-text approximation
            basis = "stream_end" if e["stream_anchored"] else "duration"
            cw = float(e["duration_s"])
        else:
            continue
        f_s = fr["elapsed_ms"] / 1000.0
        d_s = e["dynamo_elapsed_s"]
        rows.append({
            "session": e["session"], "step": e["step"], "request_id": rid,
            "tools": e.get("tools", ""),
            "basis": basis,
            "client_wall_s": cw,
            "frontend_s": f_s,
            "dynamo_s": d_s,
            "delta_s": cw - f_s,
            "delta_rel": (cw - f_s) / f_s if f_s > 0 else math.nan,
            "dynamo_vs_frontend_s": (d_s - f_s) if d_s is not None else None,
        })

    print(f"profile llm.end events: {len(ends)} "
          f"(with request_id: {n_rid})")
    print(f"frontend requests: {len(fe)}")
    print(f"joined: {len(rows)}  unmatched-rid: {unmatched}  "
          f"no-rid: {len(ends) - n_rid}")
    if not rows:
        print("no joined rows — check that the profile patch is applied "
              "(request_id) and the frontend log matches this run",
              file=sys.stderr)
        return 1

    print("\nclient_wall - frontend elapsed (s):")
    for basis in ("stream_end", "duration"):
        sub = [r["delta_s"] for r in rows if r["basis"] == basis]
        _stat_line(f"delta_s [{basis}]", sub)
    _stat_line("delta_rel (all)",
               [r["delta_rel"] for r in rows
                if not math.isnan(r["delta_rel"])])
    neg = [r for r in rows if r["delta_s"] < 0]
    if neg:
        print(f"  NEGATIVE deltas: {len(neg)}/{len(rows)} "
              "(client wall < server wall — measurement artifact, "
              "typically legacy duration_s collapse)")

    dvf = [r["dynamo_vs_frontend_s"] for r in rows
           if r["dynamo_vs_frontend_s"] is not None]
    print("\nin-band dynamo elapsed vs frontend elapsed (same clock — "
          "sanity, should be ~0):")
    _stat_line("dynamo - frontend (s)", dvf)
    if not dvf:
        print("  (no in-band dynamo timing — nvext opt-in missing?)")

    worst = sorted(rows, key=lambda r: abs(r["delta_s"]), reverse=True)[:10]
    print("\nworst 10 |delta|:")
    for r in worst:
        tl = f" tools={r['tools']}" if r["tools"] else " tools=(none)"
        print(f"  {r['session']} step {r['step']} [{r['basis']}] "
              f"client {r['client_wall_s']:.3f}s vs frontend "
              f"{r['frontend_s']:.3f}s (delta {r['delta_s']:+.3f}s){tl}")

    path = args.out / "wall_check.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k])
                        for k in CSV_COLS})
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
