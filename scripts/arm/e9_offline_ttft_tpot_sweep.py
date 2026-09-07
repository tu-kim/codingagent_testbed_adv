#!/usr/bin/env python3
"""E9: offline vLLM sweep of TTFT / TPOT over (batch size x prompt length).

Runs the model through vLLM's OFFLINE `LLM` API -- no dynamo frontend, no
opencode, no HTTP -- so the numbers are the engine's own, with none of
the queueing a live server adds. Every cell of the
(batch size x prompt tokens) grid is measured twice:

  pass 1: max_tokens=1   -> wall == TTFT (prefill + first decode step)
  pass 2: max_tokens=G   -> TPOT = (wall - TTFT) / (G - 1)

The two-pass split is what vLLM's own benchmark_latency.py does; it needs
no per-request engine metrics, which have moved around across versions.
Because the whole batch is submitted at once and waited on, the reported
value is the BATCH's time, i.e. what the last request in it experienced.

Measurement hygiene (all of it matters, none of it is optional):
  * chunked prefill OFF (--enable-chunked-prefill flips it back on): with
    it on, a long prompt is split across scheduler steps and its "TTFT"
    stops being one prefill.
  * prefix caching OFF, and every request in every batch gets its own
    RANDOM token ids -- identical prompts would otherwise be served from
    cache and collapse TTFT to noise.
  * ignore_eos=True so exactly G tokens are generated, always.
  * a warmup cell runs first (CUDA graph capture / autotune / allocator
    warm-up would otherwise land entirely on the first measured cell).

Grid: prompt lengths default to powers of two from 1k up to the model's
max_model_len minus the generation budget -- deliberately coarse, since
each cell costs two full prefills. Cells that do not fit (KV OOM at large
batch x length) are recorded with an `error` and the sweep continues.

This script ONLY MEASURES. Tables, SLA limits and figures come from
`e9_plot_ttft_tpot_sla.py`, which reads the CSV written here -- so a
sweep is paid for once on the GPU host and re-analysed anywhere, with
any SLA, as often as you like. That script also merges several sweep
CSVs, which is how a coarse full-range run and a targeted long-context
tail run end up on one plot.

Inputs:
  --model <id|path>     HF id or local path (default qwen3-coder-30b-a3b)
  --batch-sizes 1,4,16  batch sizes to sweep
  --prompt-lens ...     prompt token counts (default: powers of two)
  --gen-tokens 128      generated tokens for the TPOT pass

Output:
  <out>/sweep.csv       one row per cell: batch, prompt_tokens, gen_tokens,
                        ttft_ms, tpot_ms, decode_ms, total_ms, error
                        (failed cells keep their error text and are kept)

Usage:
  scripts/arm/e9_offline_ttft_tpot_sweep.py --model qwen3-coder-30b-a3b \
      --tp 2 --max-model-len 65536 --batch-sizes 1,2,4,8 \
      --gen-tokens 128 --out results/e9-64k
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "qwen3-coder-30b-a3b"
DEFAULT_BATCHES = [1, 2, 4, 8, 16]
# Coarse on purpose: every cell is two full prefills, and the point of the
# sweep is the SHAPE of the curve, not a dense sampling of it.
POW2_LENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]


def build_grid(max_prompt_tokens: int) -> list[int]:
    """Powers of two from 1k up to what the context window leaves for the
    prompt. Anything above the cap would be rejected by the engine."""
    return [n for n in POW2_LENS if n <= max_prompt_tokens]


def parse_int_list(s: str) -> list[int]:
    out = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"not an integer: {part!r}") from None
    if not out:
        raise argparse.ArgumentTypeError("expected a comma-separated int list")
    return out


def random_prompts(batch: int, length: int, vocab_size: int,
                   rng: random.Random) -> list[list[int]]:
    """One DISTINCT random token sequence per request.

    Distinctness is load-bearing: with a shared prompt the engine would
    serve every request after the first from the prefix cache (even with
    caching off, identical sequences in one batch let the scheduler reuse
    blocks), and the measured TTFT would be a cache hit, not a prefill.
    Ids are drawn below vocab_size so detokenization never trips on a
    reserved/added-token id.
    """
    lo, hi = 10, max(11, vocab_size - 1)
    return [[rng.randint(lo, hi) for _ in range(length)] for _ in range(batch)]


# ---------- engine ----------


def _tokens_prompt(ids: list[int]):
    """vllm.inputs.TokensPrompt when available, else the dict form both
    old and new vLLM accept."""
    try:
        from vllm.inputs import TokensPrompt
        return TokensPrompt(prompt_token_ids=ids)
    except Exception:
        return {"prompt_token_ids": ids}


def chunked_prefill_enabled(llm) -> bool | None:
    """Read the ENGINE's effective setting back, rather than trusting the
    constructor kwarg: vLLM has moved chunked prefill between config
    objects and defaults across versions, and a silently-on chunked
    prefill invalidates every TTFT in the sweep. None = could not tell."""
    for path in (("vllm_config", "scheduler_config"),
                 ("llm_engine", "vllm_config", "scheduler_config"),
                 ("llm_engine", "scheduler_config")):
        obj = llm
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is None:
            continue
        for name in ("chunked_prefill_enabled", "enable_chunked_prefill"):
            v = getattr(obj, name, None)
            if isinstance(v, bool):
                return v
    return None


def build_engine(args):
    from vllm import LLM
    kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=args.enable_chunked_prefill,
        # A cache hit is not a prefill; leaving this on would measure the
        # wrong thing for every repeated cell.
        enable_prefix_caching=False,
        max_num_seqs=max(args.batch_sizes),
        trust_remote_code=True,
        seed=args.seed,
    )
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len
    if args.max_num_batched_tokens:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    elif not args.enable_chunked_prefill:
        # With chunked prefill off a prefill must fit in ONE scheduler
        # step, so the token budget has to cover the longest prompt.
        kwargs["max_num_batched_tokens"] = args.max_model_len or None
        if kwargs["max_num_batched_tokens"] is None:
            del kwargs["max_num_batched_tokens"]
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if args.extra_engine_kwargs:
        kwargs.update(json.loads(args.extra_engine_kwargs))
    return LLM(**kwargs)


def engine_limits(llm) -> tuple[int, int]:
    """(max_model_len, vocab_size) as the engine resolved them."""
    mc = None
    for path in (("llm_engine", "model_config"), ("vllm_config", "model_config"),
                 ("llm_engine", "vllm_config", "model_config")):
        obj = llm
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            mc = obj
            break
    max_len = getattr(mc, "max_model_len", None) if mc else None
    vocab = getattr(mc, "get_vocab_size", None)
    vocab_size = vocab() if callable(vocab) else getattr(mc, "vocab_size", None)
    return int(max_len or 32768), int(vocab_size or 32000)


def measure_cell(llm, batch: int, length: int, gen_tokens: int,
                 vocab_size: int, rng: random.Random) -> dict:
    """One (batch, length) cell: a max_tokens=1 pass for TTFT and a
    max_tokens=gen_tokens pass for TPOT, on FRESH random prompts each
    time so neither pass warms the other."""
    from vllm import SamplingParams
    row = {"batch": batch, "prompt_tokens": length, "gen_tokens": gen_tokens,
           "ttft_ms": "", "tpot_ms": "", "decode_ms": "", "total_ms": "",
           "error": ""}
    try:
        greedy = dict(temperature=0.0, ignore_eos=True)
        p1 = [_tokens_prompt(ids)
              for ids in random_prompts(batch, length, vocab_size, rng)]
        t0 = time.perf_counter()
        llm.generate(p1, SamplingParams(max_tokens=1, **greedy), use_tqdm=False)
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        p2 = [_tokens_prompt(ids)
              for ids in random_prompts(batch, length, vocab_size, rng)]
        t0 = time.perf_counter()
        llm.generate(p2, SamplingParams(max_tokens=gen_tokens, **greedy),
                     use_tqdm=False)
        total_ms = (time.perf_counter() - t0) * 1000.0

        decode_ms = total_ms - ttft_ms
        row["ttft_ms"] = round(ttft_ms, 3)
        row["total_ms"] = round(total_ms, 3)
        row["decode_ms"] = round(decode_ms, 3)
        row["tpot_ms"] = round(decode_ms / max(gen_tokens - 1, 1), 4)
    except Exception as exc:                      # KV OOM at the big cells
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return row


def run_sweep(llm, batches: list[int], lengths: list[int], gen_tokens: int,
              vocab_size: int, seed: int, warmup: bool = True) -> list[dict]:
    rng = random.Random(seed)
    if warmup:
        # Absorbs CUDA graph capture, kernel autotune and allocator growth,
        # which would otherwise all be charged to the first real cell.
        measure_cell(llm, 1, min(lengths), min(gen_tokens, 8), vocab_size, rng)
    rows = []
    for batch in batches:
        for length in lengths:
            row = measure_cell(llm, batch, length, gen_tokens, vocab_size, rng)
            rows.append(row)
            status = row["error"] or (f"ttft {row['ttft_ms']:.0f} ms  "
                                      f"tpot {row['tpot_ms']:.2f} ms")
            print(f"  batch={batch:<4} prompt={length:<7} {status}", flush=True)
    return rows


# ---------- SLA ----------


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch-sizes", type=parse_int_list,
                    default=DEFAULT_BATCHES)
    ap.add_argument("--prompt-lens", type=parse_int_list, default=None,
                    help="default: powers of two from 1k to the context limit")
    ap.add_argument("--gen-tokens", type=int, default=128,
                    help="tokens generated in the TPOT pass (default 128)")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None,
                    help="default with chunked prefill off: --max-model-len, "
                         "the minimum the engine accepts (a prefill must fit "
                         "in one scheduler step)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--enable-chunked-prefill", action="store_true",
                    help="re-enable chunked prefill (OFF by default: it "
                         "splits a long prefill across scheduler steps, so "
                         "the measured TTFT is no longer one prefill)")
    ap.add_argument("--extra-engine-kwargs", default=None,
                    help="JSON dict merged into the LLM(...) constructor")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("e9_offline_sweep"))
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    llm = build_engine(args)
    cp = chunked_prefill_enabled(llm)
    if cp and not args.enable_chunked_prefill:
        print("error: the engine reports chunked prefill ENABLED despite "
              "enable_chunked_prefill=False -- every TTFT in this sweep "
              "would be a partial prefill. Re-run with "
              "--enable-chunked-prefill if that is what you want.",
              file=sys.stderr)
        return 3
    if cp is None:
        print("warning: could not read the engine's chunked-prefill "
              "setting back; proceeding on the constructor kwarg alone",
              file=sys.stderr)
    max_len, vocab_size = engine_limits(llm)
    max_prompt = max_len - args.gen_tokens
    lengths = args.prompt_lens or build_grid(max_prompt)
    too_long = [n for n in lengths if n > max_prompt]
    if too_long:
        print(f"warning: dropping {too_long} -- above max_model_len "
              f"({max_len}) minus {args.gen_tokens} generated tokens",
              file=sys.stderr)
        lengths = [n for n in lengths if n <= max_prompt]
    if not lengths:
        print("error: no prompt length fits the context window",
              file=sys.stderr)
        return 2
    print(f"model={args.model} max_model_len={max_len} vocab={vocab_size}")
    print(f"batches={args.batch_sizes} lengths={lengths} "
          f"gen_tokens={args.gen_tokens} chunked_prefill={bool(cp)}")
    rows = run_sweep(llm, args.batch_sizes, lengths, args.gen_tokens,
                     vocab_size, args.seed, warmup=not args.no_warmup)
    out_csv = args.out / "sweep.csv"
    write_csv(out_csv, rows,
              ["batch", "prompt_tokens", "gen_tokens", "ttft_ms",
               "tpot_ms", "decode_ms", "total_ms", "error"])

    n_err = sum(1 for r in rows if r.get("error"))
    if n_err:
        print(f"\n{n_err}/{len(rows)} cells failed (typically KV OOM at "
              "large batch x length); they are kept in the CSV with their "
              "error text")
    print(f"\nwrote {out_csv}")
    print("plot it with: scripts/arm/e9_plot_ttft_tpot_sla.py "
          f"--sweep {out_csv} --ttft-sla-ms <ms> --tpot-sla-ms <ms>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
