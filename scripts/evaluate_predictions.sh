#!/usr/bin/env bash
# Judge TRUE resolve/fail for a finished testbed run with the official
# SWE-bench evaluation harness (per-instance Docker images; runs each
# instance's FAIL_TO_PASS + PASS_TO_PASS tests against the model patch).
#
# Pipeline:
#   1. scripts/extract_predictions.py  -> <run>/predictions.jsonl   (skipped if it exists)
#   2. python -m swebench.harness.run_evaluation                    (Docker required)
#   3. report lands at <run>/<model_name>.<run_id>.json
#      then summarize with: scripts/analyze_eval_results.py --run <run>
#
# Prerequisites on the eval host:
#   pip install swebench       # official harness (pulls docker SDK)
#   docker daemon running, current user in the docker group
#   ~120GB free disk for SWE-bench Lite images at default cache level
#
# Usage:
#   scripts/evaluate_predictions.sh --run results/run1 [--max-workers 8] \
#       [--run-id myeval] [--model-name testbed] [--python /path/to/python]
set -Eeuo pipefail

RUN=""
MAX_WORKERS=4
RUN_ID=""
MODEL_NAME="testbed"
PYTHON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)         RUN="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --run-id)      RUN_ID="$2"; shift 2 ;;
    --model-name)  MODEL_NAME="$2"; shift 2 ;;
    --python)      PYTHON="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$RUN" ]] || { echo "usage: $0 --run results/<dir> [--max-workers N] [--run-id ID]" >&2; exit 2; }
[[ -f "$RUN/config.json" ]] || { echo "error: $RUN/config.json not found" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Prefer the project venv; the harness itself may live in a different env --
# override with --python if so.
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

# Map the run's split to the HF dataset id (same map as src/testbed/swebench.py).
SPLIT=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['split'])" "$RUN/config.json")
case "$SPLIT" in
  lite)     DATASET="princeton-nlp/SWE-bench_Lite" ;;
  verified) DATASET="princeton-nlp/SWE-bench_Verified" ;;
  full)     DATASET="princeton-nlp/SWE-bench" ;;
  *) echo "error: unknown split '$SPLIT' in $RUN/config.json" >&2; exit 1 ;;
esac

[[ -n "$RUN_ID" ]] || RUN_ID="$(basename "$RUN")"

PREDS="$RUN/predictions.jsonl"
if [[ ! -f "$PREDS" ]]; then
  echo "== extracting predictions -> $PREDS"
  "$PYTHON" "$SCRIPT_DIR/extract_predictions.py" --run "$RUN" --model-name "$MODEL_NAME"
else
  echo "== reusing existing $PREDS (delete it to re-extract)"
fi

# Run from inside the run dir: the harness writes its report
# (<model_name>.<run_id>.json) and logs/run_evaluation/ relative to CWD,
# keeping all eval artifacts next to trace.jsonl.
echo "== running official harness (dataset=$DATASET run_id=$RUN_ID workers=$MAX_WORKERS)"
ABS_PREDS="$(cd "$(dirname "$PREDS")" && pwd)/$(basename "$PREDS")"
cd "$RUN"
"$PYTHON" -m swebench.harness.run_evaluation \
  --dataset_name "$DATASET" \
  --predictions_path "$ABS_PREDS" \
  --max_workers "$MAX_WORKERS" \
  --run_id "$RUN_ID"

echo "== done. summarize with:"
echo "   scripts/analyze_eval_results.py --run $RUN"
