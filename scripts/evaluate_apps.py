#!/usr/bin/env python3
"""Post-hoc correctness evaluation for an APPS run (the APPS counterpart
of extract_predictions + evaluate_predictions for SWE-bench).

Reads a run directory produced by `python -m testbed run --workload apps`,
re-derives the exact sample set from config.json's (split, seed,
num_samples) via testbed.apps.load_samples (deterministic), locates each
task's `solution.py` inside its workspace, and executes it against the
problem's hidden `input_output` tests.

Judging modes (per problem, from input_output's fn_name):
  stdio       run `python solution.py`, pipe the test input to stdin,
              compare stdout to the expected output after per-line
              trailing-whitespace + trailing-newline normalization.
  call-based  import solution.py in a subprocess harness and call
              fn_name(*args) (LeetCode-style `class Solution` methods are
              resolved on an instance); compare the JSON-serialized return
              value with float tolerance.

This is a pragmatic approximation of the official APPS metric (the
official testing_util has extra leniencies -- e.g. multiple accepted
outputs); expect it to be slightly stricter, uniformly across routers, so
cross-run comparisons stay fair.

SECURITY: this EXECUTES model-generated code with the invoking user's
privileges. Run it inside a container/VM or at least a throwaway user on
the eval host. There is deliberately no network/file sandboxing here.

Verdicts (per instance):
  resolved      every executed test passed (and >= 1 test ran)
  unresolved    at least one test failed / timed out / crashed
  no_solution   workspace or solution.py missing (incl. HTTP-failed tasks)
  no_tests      problem has no input_output tests to run
Aggregates mirror analyze_eval_results.py: resolve_rate_all (over every
trace record) and resolve_rate_http_ok (over success=true records only).

Usage:
  scripts/evaluate_apps.py --run results/apps1 \
      [--workspace-root /tmp/testbed-workspaces] \
      [--max-tests 20] [--timeout-s 10] [--python /usr/bin/python3]

Outputs:
  <run>/apps_eval.json   per-instance verdicts + aggregates
  stdout                 pretty table
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from testbed import apps  # noqa: E402


# ---------- output comparison ----------


def _norm_stdio(text: str) -> str:
    """Per-line trailing-whitespace strip + trailing-blank-line strip.
    The dominant benign formatting differences in APPS outputs."""
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    return "\n".join(lines)


def _as_text(v: Any) -> str:
    """input_output entries are usually strings but occasionally lists of
    lines (the official harness joins them)."""
    if isinstance(v, (list, tuple)):
        return "\n".join(str(x) for x in v)
    return str(v)


def _json_close(a: Any, b: Any, *, tol: float = 1e-6) -> bool:
    """Structural equality with float tolerance for call-based returns."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_json_close(x, y, tol=tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_close(a[k], b[k], tol=tol) for k in a)
    return a == b


# ---------- per-mode execution ----------


# Runs inside a THROWAWAY subprocess per test: reads {"args": [...]} on
# stdin, imports the solution module, resolves fn_name (module-level or on
# a Solution() instance), prints json.dumps(result) as the LAST stdout
# line (solutions that print extra noise during import don't break the
# protocol because we only parse the final line).
_CALL_HARNESS = r"""
import importlib.util, json, sys
payload = json.load(sys.stdin)
spec = importlib.util.spec_from_file_location("solution", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn_name = sys.argv[2]
if hasattr(mod, fn_name):
    fn = getattr(mod, fn_name)
elif hasattr(mod, "Solution"):
    fn = getattr(mod.Solution(), fn_name)
else:
    raise AttributeError(f"solution.py defines neither {fn_name!r} nor Solution")
result = fn(*payload["args"])
print("\n__APPS_RESULT__" + json.dumps(result))
"""


def _run_stdio_test(python: str, solution: Path, inp: Any, expected: Any,
                    timeout_s: float) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [python, str(solution)],
            input=_as_text(inp).encode(),
            capture_output=True,
            timeout=timeout_s,
            cwd=str(solution.parent),
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode != 0:
        return False, f"exit={proc.returncode}"
    ok = _norm_stdio(proc.stdout.decode(errors="replace")) == _norm_stdio(_as_text(expected))
    return ok, "" if ok else "wrong_output"


def _run_call_test(python: str, solution: Path, fn_name: str, args: Any,
                   expected: Any, timeout_s: float) -> tuple[bool, str]:
    if not isinstance(args, (list, tuple)):
        args = [args]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_CALL_HARNESS)
        harness = f.name
    try:
        proc = subprocess.run(
            [python, harness, str(solution), fn_name],
            input=json.dumps({"args": list(args)}).encode(),
            capture_output=True,
            timeout=timeout_s,
            cwd=str(solution.parent),
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        Path(harness).unlink(missing_ok=True)
    if proc.returncode != 0:
        return False, f"exit={proc.returncode}"
    marker = "__APPS_RESULT__"
    out = proc.stdout.decode(errors="replace")
    idx = out.rfind(marker)
    if idx < 0:
        return False, "no_result_marker"
    try:
        got = json.loads(out[idx + len(marker):].strip())
    except json.JSONDecodeError:
        return False, "unserializable_result"
    # APPS expected outputs for call-based problems are usually a
    # one-element list wrapping the return value; accept either shape.
    if _json_close(got, expected):
        return True, ""
    if isinstance(expected, list) and len(expected) == 1 and _json_close(got, expected[0]):
        return True, ""
    return False, "wrong_output"


def evaluate_instance(sample: dict[str, Any], workspace: Path, *,
                      python: str, max_tests: int, timeout_s: float) -> dict[str, Any]:
    solution = workspace / apps.SOLUTION_FILE
    if not solution.is_file():
        return {"verdict": "no_solution", "n_tests": 0, "n_passed": 0, "fails": []}
    io = apps.parse_input_output(sample)
    inputs, outputs, fn_name = io["inputs"], io["outputs"], io["fn_name"]
    n = min(len(inputs), len(outputs))
    if n == 0:
        return {"verdict": "no_tests", "n_tests": 0, "n_passed": 0, "fails": []}
    n_run = min(n, max_tests) if max_tests > 0 else n
    n_passed = 0
    fails: list[dict[str, Any]] = []
    for i in range(n_run):
        if fn_name:
            ok, why = _run_call_test(python, solution, fn_name,
                                     inputs[i], outputs[i], timeout_s)
        else:
            ok, why = _run_stdio_test(python, solution,
                                      inputs[i], outputs[i], timeout_s)
        if ok:
            n_passed += 1
        elif len(fails) < 5:  # keep the report small
            fails.append({"test": i, "reason": why})
    verdict = "resolved" if n_passed == n_run else "unresolved"
    return {
        "verdict": verdict,
        "mode": "call" if fn_name else "stdio",
        "n_tests": n_run,
        "n_tests_total": n,
        "n_passed": n_passed,
        "fails": fails,
    }


# ---------- run-dir plumbing ----------


def load_trace(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path,
                    help="Run directory (config.json + trace.jsonl)")
    ap.add_argument("--workspace-root", default=None, type=Path,
                    help="Override the workspace root (default: the "
                         "resolved snapshot recorded in config.json)")
    ap.add_argument("--max-tests", default=20, type=int,
                    help="Cap on tests executed per problem (some APPS "
                         "problems carry hundreds); <=0 = all. Default 20.")
    ap.add_argument("--timeout-s", default=10.0, type=float,
                    help="Wall-clock cap per test execution. Default 10s.")
    ap.add_argument("--python", default=sys.executable,
                    help="Interpreter used to execute solutions. "
                         "Default: this script's interpreter.")
    args = ap.parse_args(argv)

    cfg = json.loads((args.run / "config.json").read_text())
    if cfg.get("workload", "swebench") != "apps":
        print(f"run {args.run} has workload={cfg.get('workload')!r}, not 'apps' "
              f"-- use the SWE-bench harness scripts for swebench runs",
              file=sys.stderr)
        return 2

    workspace_root = args.workspace_root
    if workspace_root is None:
        ws = (cfg.get("config") or {}).get("workspace_root")
        if not ws:
            print("workspace_root not found in config.json; pass --workspace-root",
                  file=sys.stderr)
            return 2
        workspace_root = Path(ws)
    workspace_root = workspace_root.expanduser().resolve()

    print("WARNING: executing model-generated code from the workspaces; "
          "run this inside a container/VM.", file=sys.stderr)

    samples = apps.load_samples(cfg["split"], cfg["seed"], cfg["num_samples"])
    by_id = {s["instance_id"]: s for s in samples}
    records = load_trace(args.run)

    per_instance: dict[str, dict[str, Any]] = {}
    for rec in records:
        iid = rec["instance_id"]
        sample = by_id.get(iid)
        if sample is None:
            per_instance[iid] = {"verdict": "not_in_sample_set",
                                 "n_tests": 0, "n_passed": 0, "fails": []}
            continue
        res = evaluate_instance(
            sample, workspace_root / rec["directory"],
            python=args.python, max_tests=args.max_tests, timeout_s=args.timeout_s,
        )
        res["http_success"] = bool(rec.get("success"))
        res["difficulty"] = sample.get("difficulty")
        per_instance[iid] = res
        print(f"  {iid:<14} {res['verdict']:<12} "
              f"{res['n_passed']}/{res['n_tests']} tests "
              f"[{res.get('mode', '-')}]", flush=True)

    n_all = len(per_instance)
    resolved = [i for i, r in per_instance.items() if r["verdict"] == "resolved"]
    http_ok = [i for i, r in per_instance.items() if r.get("http_success")]
    resolved_http_ok = [i for i in resolved if i in set(http_ok)]
    summary = {
        "count": n_all,
        "resolved": len(resolved),
        "resolve_rate_all": (len(resolved) / n_all) if n_all else None,
        "resolve_rate_http_ok": (len(resolved_http_ok) / len(http_ok)) if http_ok else None,
        "verdicts": {v: sum(1 for r in per_instance.values() if r["verdict"] == v)
                     for v in sorted({r["verdict"] for r in per_instance.values()})},
    }

    out_path = args.run / "apps_eval.json"
    out_path.write_text(json.dumps(
        {"summary": summary, "per_instance": per_instance}, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
