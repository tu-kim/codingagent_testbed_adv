"""Tests for scripts/arm/e9_offline_ttft_tpot_sweep.py.

No network, no GPU: this script drives vLLM's OFFLINE `LLM` API, which
needs a real accelerator and is not installed on this dev machine. Loaded
via importlib (script, not a package module) -- see
tests/test_e8_ttft_tpot.py for the same pattern.

This script is now MEASUREMENT ONLY (the SLA/plot/--from-csv reporting
side moved to `e9_plot_ttft_tpot_sla.py` -- see tests/test_e9_plot_sla.py
for sla_limits / print_sweep / print_sla / plot_metric / merge_sweeps).
`main()` always builds an engine; there is no --from-csv path any more.
It is exercised here by monkeypatching the module's own
build_engine/chunked_prefill_enabled/engine_limits/run_sweep to fakes, so
no real vLLM install or GPU is touched even for the end-to-end main()
tests.

Everything below main() is still exercised against `_FakeLLM` (a
`generate()` that records every call and returns a dummy per-prompt list;
nested attribute stubs are hung off an instance per-test for the
config-reader helpers) plus a fake `vllm` module injected into
`sys.modules` via `monkeypatch.setitem` so `measure_cell`'s in-function
`from vllm import SamplingParams` resolves without a real install.
"""
import argparse
import csv
import importlib.util
import random
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "arm" / "e9_offline_ttft_tpot_sweep.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def e9():
    return _load_module("e9_offline_ttft_tpot_sweep", _SCRIPT_PATH)


# ---------------------------------------------------------------------------
# fake vLLM engine + module plumbing
# ---------------------------------------------------------------------------

class _FakeLLM:
    """Stand-in for vllm.LLM. `generate` records every call and returns a
    dummy per-prompt list (measure_cell never inspects the return value
    beyond nothing -- vLLM's own RequestOutput objects aren't needed).
    `chunked_prefill_enabled`/`engine_limits` walk arbitrary getattr chains
    with a None default, so nested attribute stubs (plain objects, e.g.
    SimpleNamespace, hung directly off an instance) are enough -- no
    spec'd mock needed."""

    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params, use_tqdm=None):
        self.calls.append({"prompts": prompts,
                           "sampling_params": sampling_params,
                           "use_tqdm": use_tqdm})
        return ["dummy"] * len(prompts)


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_fake_vllm(monkeypatch, with_tokens_prompt=False):
    """Inject a fake top-level `vllm` module exposing SamplingParams, via
    monkeypatch.setitem so it is cleanly reverted after the test -- matters
    on a GPU host where a REAL vllm is installed and must not be left
    clobbered for later tests in the same session.

    `measure_cell` does `from vllm import SamplingParams` on every call.
    Separately, `_tokens_prompt` (called while building each pass's prompt
    list) does `from vllm.inputs import TokensPrompt`. Verified empirically
    (see the pytest-mock-author agent's memory note on this script): Python
    resolves a dotted `from X.Y import Z` by checking sys.modules["X.Y"]
    directly -- it does NOT require the parent module to be a real package
    with a matching `.Y` attribute. So:

      * with_tokens_prompt=False (default): no "vllm.inputs" entry exists
        -> `import vllm.inputs` raises ModuleNotFoundError ("'vllm' is not
        a package"), caught by _tokens_prompt's broad `except Exception` ->
        exercises the dict FALLBACK path (`{"prompt_token_ids": ids}`).
      * with_tokens_prompt=True: "vllm.inputs" is registered with a
        TokensPrompt class -> exercises the non-fallback path.
    """
    fake_vllm = ModuleType("vllm")
    fake_vllm.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    if with_tokens_prompt:
        fake_inputs = ModuleType("vllm.inputs")

        class _FakeTokensPrompt:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_inputs.TokensPrompt = _FakeTokensPrompt
        monkeypatch.setitem(sys.modules, "vllm.inputs", fake_inputs)
    else:
        # Force the ModuleNotFoundError path even if a REAL vllm (with a
        # real .inputs submodule) already sits in sys.modules from an
        # earlier-imported test in this session.
        monkeypatch.delitem(sys.modules, "vllm.inputs", raising=False)


# ---------------------------------------------------------------------------
# build_grid
# ---------------------------------------------------------------------------

class TestBuildGrid:
    def test_filters_powers_of_two_to_the_cap(self, e9):
        assert e9.build_grid(2048) == [1024, 2048]

    def test_cap_below_1024_is_empty(self, e9):
        assert e9.build_grid(1023) == []
        assert e9.build_grid(0) == []

    def test_cap_above_top_of_table_returns_everything(self, e9):
        assert e9.build_grid(300_000) == e9.POW2_LENS


# ---------------------------------------------------------------------------
# parse_int_list
# ---------------------------------------------------------------------------

class TestParseIntList:
    def test_normal_comma_separated(self, e9):
        assert e9.parse_int_list("1,2,4,8") == [1, 2, 4, 8]

    def test_whitespace_around_values_is_stripped(self, e9):
        assert e9.parse_int_list(" 1, 2 , 4 ") == [1, 2, 4]

    def test_single_value(self, e9):
        assert e9.parse_int_list("5") == [5]

    def test_empty_string_raises_argument_type_error(self, e9):
        with pytest.raises(argparse.ArgumentTypeError):
            e9.parse_int_list("")

    def test_whitespace_only_raises_argument_type_error(self, e9):
        # collapses to "" after the leading .replace(" ", ""), same as empty
        with pytest.raises(argparse.ArgumentTypeError):
            e9.parse_int_list("   ")

    def test_bare_commas_raise_argument_type_error(self, e9):
        with pytest.raises(argparse.ArgumentTypeError):
            e9.parse_int_list(",,,")

    def test_non_numeric_garbage_raises_argument_type_error(self, e9):
        # Same exception type as the zero-parts cases above: a caller of the
        # bare function catches ONE type for any bad input. (ArgumentTypeError
        # is not a ValueError subclass, so this uniformity has to be built in
        # rather than inherited.)
        with pytest.raises(argparse.ArgumentTypeError):
            e9.parse_int_list("abc")
        with pytest.raises(argparse.ArgumentTypeError):
            e9.parse_int_list("1,abc,3")

    def test_non_numeric_value_still_fails_cleanly_through_real_argparse(
            self, e9, capsys):
        # End to end through argparse: a bad --batch-sizes value ends in a
        # clean usage error + SystemExit(2), never a traceback. This is what
        # actually protects `--batch-sizes`/`--prompt-lens` in main().
        ap = argparse.ArgumentParser()
        ap.add_argument("--batch-sizes", type=e9.parse_int_list)
        with pytest.raises(SystemExit) as excinfo:
            ap.parse_args(["--batch-sizes", "abc"])
        assert excinfo.value.code == 2
        capsys.readouterr()  # swallow the usage/error text argparse printed


# ---------------------------------------------------------------------------
# random_prompts
# ---------------------------------------------------------------------------

class TestRandomPrompts:
    def test_batch_count_and_exact_length(self, e9):
        out = e9.random_prompts(batch=5, length=20, vocab_size=32000,
                                rng=random.Random(42))
        assert len(out) == 5
        assert all(len(seq) == 20 for seq in out)

    def test_ids_within_bounds(self, e9):
        out = e9.random_prompts(batch=5, length=20, vocab_size=32000,
                                rng=random.Random(42))
        flat = [tok for seq in out for tok in seq]
        assert min(flat) >= 10
        assert max(flat) <= 32000 - 1

    def test_sequences_are_distinct(self, e9):
        # Load-bearing: identical prompts across a batch would let the
        # engine serve later requests from a shared-prefix/cache hit even
        # with prefix caching disabled, silently corrupting the TTFT
        # measurement into a cache-hit measurement.
        out = e9.random_prompts(batch=8, length=24, vocab_size=32000,
                                rng=random.Random(7))
        assert len({tuple(seq) for seq in out}) == len(out)


# ---------------------------------------------------------------------------
# chunked_prefill_enabled
# ---------------------------------------------------------------------------

class TestChunkedPrefillEnabled:
    def test_finds_via_vllm_config_scheduler_config(self, e9):
        llm = _FakeLLM()
        llm.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(chunked_prefill_enabled=True))
        assert e9.chunked_prefill_enabled(llm) is True

    def test_finds_via_llm_engine_vllm_config_scheduler_config(self, e9):
        llm = _FakeLLM()
        llm.llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    chunked_prefill_enabled=True)))
        assert e9.chunked_prefill_enabled(llm) is True

    def test_finds_via_llm_engine_scheduler_config(self, e9):
        # llm_engine present but with NO vllm_config attribute at all, so
        # path 2 (llm_engine.vllm_config.scheduler_config) must fall
        # through to path 3 (llm_engine.scheduler_config directly).
        llm = _FakeLLM()
        llm.llm_engine = SimpleNamespace(
            scheduler_config=SimpleNamespace(chunked_prefill_enabled=False))
        assert e9.chunked_prefill_enabled(llm) is False

    def test_reads_enable_chunked_prefill_name_too(self, e9):
        llm = _FakeLLM()
        llm.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True))
        assert e9.chunked_prefill_enabled(llm) is True

    def test_none_when_nothing_exposes_it(self, e9):
        assert e9.chunked_prefill_enabled(_FakeLLM()) is None

    def test_none_when_attribute_present_but_not_bool(self, e9):
        # A non-bool value must not be mistaken for a found flag --
        # isinstance(v, bool) gates every candidate.
        llm = _FakeLLM()
        llm.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(chunked_prefill_enabled=None))
        assert e9.chunked_prefill_enabled(llm) is None


# ---------------------------------------------------------------------------
# engine_limits
# ---------------------------------------------------------------------------

class TestEngineLimits:
    def test_llm_engine_model_config_with_plain_vocab_size_attribute(
            self, e9):
        llm = _FakeLLM()
        llm.llm_engine = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=4096, vocab_size=15000))
        assert e9.engine_limits(llm) == (4096, 15000)

    def test_vllm_config_model_config_with_callable_get_vocab_size(self, e9):
        llm = _FakeLLM()
        llm.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=8192,
                                         get_vocab_size=lambda: 20000))
        assert e9.engine_limits(llm) == (8192, 20000)

    def test_llm_engine_vllm_config_model_config_path(self, e9):
        # Neither of the first two paths is present directly on llm_engine;
        # only the nested llm_engine.vllm_config.model_config resolves.
        llm = _FakeLLM()
        llm.llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(max_model_len=2048,
                                             vocab_size=9000)))
        assert e9.engine_limits(llm) == (2048, 9000)

    def test_falls_back_when_config_unreachable(self, e9):
        assert e9.engine_limits(_FakeLLM()) == (32768, 32000)


# ---------------------------------------------------------------------------
# measure_cell
# ---------------------------------------------------------------------------

class TestMeasureCell:
    def test_deterministic_timings_and_tpot_formula(self, e9, monkeypatch):
        _install_fake_vllm(monkeypatch)
        llm = _FakeLLM()
        # Exactly 4 perf_counter() reads happen, in this order:
        #   t0(pass1)=0.0, end(pass1)=0.5   -> ttft_ms  = (0.5-0.0)*1000 = 500
        #   t0(pass2)=10.0, end(pass2)=12.0 -> total_ms = (12.0-10.0)*1000 = 2000
        ticks = iter([0.0, 0.5, 10.0, 12.0])
        monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

        row = e9.measure_cell(llm, batch=2, length=8, gen_tokens=4,
                              vocab_size=100, rng=random.Random(0))

        assert row["error"] == ""
        assert row["ttft_ms"] == 500.0
        assert row["total_ms"] == 2000.0
        assert row["decode_ms"] == 1500.0
        assert row["tpot_ms"] == row["decode_ms"] / (4 - 1)

    def test_two_generate_calls_with_max_tokens_one_then_gen_tokens(
            self, e9, monkeypatch):
        _install_fake_vllm(monkeypatch)
        llm = _FakeLLM()
        ticks = iter([0.0, 0.1, 1.0, 1.4])
        monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

        e9.measure_cell(llm, batch=3, length=16, gen_tokens=7,
                        vocab_size=200, rng=random.Random(1))

        assert len(llm.calls) == 2
        first, second = llm.calls
        assert first["sampling_params"].kwargs["max_tokens"] == 1
        assert second["sampling_params"].kwargs["max_tokens"] == 7
        assert first["use_tqdm"] is False
        assert second["use_tqdm"] is False
        assert len(first["prompts"]) == 3
        assert len(second["prompts"]) == 3

    def test_generate_exception_captured_other_fields_left_blank(
            self, e9, monkeypatch):
        _install_fake_vllm(monkeypatch)

        class _RaisingLLM:
            def generate(self, *a, **k):
                raise RuntimeError("boom-OOM")

        monkeypatch.setattr(time, "perf_counter", lambda: 0.0)

        row = e9.measure_cell(_RaisingLLM(), batch=1, length=4, gen_tokens=2,
                              vocab_size=50, rng=random.Random(2))

        assert row["error"] == "RuntimeError: boom-OOM"
        assert row["ttft_ms"] == ""
        assert row["tpot_ms"] == ""
        assert row["decode_ms"] == ""
        assert row["total_ms"] == ""
        assert row["batch"] == 1
        assert row["prompt_tokens"] == 4
        assert row["gen_tokens"] == 2

    def test_fallback_prompt_shape_when_tokens_prompt_unavailable(
            self, e9, monkeypatch):
        _install_fake_vllm(monkeypatch, with_tokens_prompt=False)
        llm = _FakeLLM()
        ticks = iter([0.0, 0.1, 1.0, 1.1])
        monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

        e9.measure_cell(llm, batch=1, length=4, gen_tokens=2,
                        vocab_size=50, rng=random.Random(3))

        prompt = llm.calls[0]["prompts"][0]
        assert isinstance(prompt, dict)
        assert list(prompt.keys()) == ["prompt_token_ids"]
        assert len(prompt["prompt_token_ids"]) == 4

    def test_tokens_prompt_used_when_vllm_inputs_available(
            self, e9, monkeypatch):
        _install_fake_vllm(monkeypatch, with_tokens_prompt=True)
        llm = _FakeLLM()
        ticks = iter([0.0, 0.1, 1.0, 1.1])
        monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

        e9.measure_cell(llm, batch=1, length=4, gen_tokens=2,
                        vocab_size=50, rng=random.Random(4))

        prompt = llm.calls[0]["prompts"][0]
        assert not isinstance(prompt, dict)
        assert len(prompt.kwargs["prompt_token_ids"]) == 4


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------

class TestRunSweep:
    def test_warmup_adds_exactly_one_extra_measured_cell(
            self, e9, monkeypatch):
        _install_fake_vllm(monkeypatch)
        llm_with = _FakeLLM()
        e9.run_sweep(llm_with, batches=[1, 4], lengths=[100, 200],
                    gen_tokens=8, vocab_size=1000, seed=42, warmup=True)
        llm_without = _FakeLLM()
        e9.run_sweep(llm_without, batches=[1, 4], lengths=[100, 200],
                    gen_tokens=8, vocab_size=1000, seed=42, warmup=False)
        # Each cell is 2 generate() calls (pass1 + pass2); warmup is ONE
        # extra cell measured before the grid, so it must add exactly 2
        # generate() calls without changing the returned row count.
        assert len(llm_with.calls) - len(llm_without.calls) == 2

    def test_warmup_cell_not_included_in_returned_rows(self, e9, monkeypatch):
        _install_fake_vllm(monkeypatch)
        llm = _FakeLLM()
        rows = e9.run_sweep(llm, batches=[1, 4], lengths=[100, 200],
                            gen_tokens=8, vocab_size=1000, seed=42,
                            warmup=True)
        assert len(rows) == 4  # len(batches) * len(lengths), NOT +1

    def test_rows_are_batch_major_order(self, e9, monkeypatch):
        _install_fake_vllm(monkeypatch)
        llm = _FakeLLM()
        rows = e9.run_sweep(llm, batches=[1, 4], lengths=[100, 200],
                            gen_tokens=8, vocab_size=1000, seed=42,
                            warmup=False)
        assert [(r["batch"], r["prompt_tokens"]) for r in rows] == [
            (1, 100), (1, 200), (4, 100), (4, 200),
        ]

    def test_warmup_false_skips_the_warmup_cell(self, e9, monkeypatch):
        _install_fake_vllm(monkeypatch)
        llm = _FakeLLM()
        e9.run_sweep(llm, batches=[1], lengths=[100], gen_tokens=8,
                    vocab_size=1000, seed=42, warmup=False)
        assert len(llm.calls) == 2  # exactly one cell, no warmup cell


# ---------------------------------------------------------------------------
# main() -- always builds an engine now (no --from-csv path); everything
# that would touch a real vLLM engine is monkeypatched to a fake at the
# module level.
# ---------------------------------------------------------------------------

_SWEEP_COLS = ["batch", "prompt_tokens", "gen_tokens", "ttft_ms", "tpot_ms",
              "decode_ms", "total_ms", "error"]


def _fake_row(batch=1, prompt_tokens=1000, gen_tokens=8, ttft_ms=10.0,
             tpot_ms=1.0):
    decode_ms = tpot_ms * (gen_tokens - 1)
    return {"batch": batch, "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens, "ttft_ms": ttft_ms, "tpot_ms": tpot_ms,
            "decode_ms": decode_ms, "total_ms": ttft_ms + decode_ms,
            "error": ""}


def _patch_engine(monkeypatch, e9, *, chunked_prefill=False,
                  max_model_len=65536, vocab_size=32000, rows=None,
                  record_run_sweep_args=None):
    """Stub out everything main() uses to talk to a real engine, leaving
    the grid-building / CSV-writing / exit-code logic in main() itself
    under test."""
    monkeypatch.setattr(e9, "build_engine", lambda args: "engine-sentinel")
    monkeypatch.setattr(e9, "chunked_prefill_enabled",
                        lambda llm: chunked_prefill)
    monkeypatch.setattr(e9, "engine_limits",
                        lambda llm: (max_model_len, vocab_size))

    def fake_run_sweep(llm, batches, lengths, gen_tokens, vocab_size, seed,
                       warmup=True):
        if record_run_sweep_args is not None:
            record_run_sweep_args.append({
                "llm": llm, "batches": batches, "lengths": lengths,
                "gen_tokens": gen_tokens, "vocab_size": vocab_size,
                "seed": seed, "warmup": warmup,
            })
        return rows if rows is not None else [_fake_row()]

    monkeypatch.setattr(e9, "run_sweep", fake_run_sweep)


class TestMain:
    def test_writes_sweep_csv_with_expected_header_and_prints_pointer(
            self, e9, monkeypatch, tmp_path, capsys):
        _patch_engine(monkeypatch, e9, rows=[_fake_row(1, 1000, 8, 10.0, 1.0)])
        out_dir = tmp_path / "out"

        rc = e9.main(["--batch-sizes", "1", "--gen-tokens", "8",
                     "--out", str(out_dir)])

        assert rc == 0
        out_csv = out_dir / "sweep.csv"
        assert out_csv.exists()
        with out_csv.open() as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == _SWEEP_COLS
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["batch"] == "1"
        out = capsys.readouterr().out
        assert f"wrote {out_csv}" in out
        assert "e9_plot_ttft_tpot_sla.py" in out
        assert f"--sweep {out_csv}" in out

    def test_chunked_prefill_enabled_without_flag_returns_3(
            self, e9, monkeypatch, tmp_path, capsys):
        calls = []
        _patch_engine(monkeypatch, e9, chunked_prefill=True,
                      record_run_sweep_args=calls)
        out_dir = tmp_path / "out"

        rc = e9.main(["--out", str(out_dir)])

        assert rc == 3
        assert calls == []  # run_sweep must never be reached
        assert not (out_dir / "sweep.csv").exists()
        err = capsys.readouterr().err
        assert "chunked prefill" in err.lower()

    def test_chunked_prefill_enabled_with_flag_proceeds(
            self, e9, monkeypatch, tmp_path):
        _patch_engine(monkeypatch, e9, chunked_prefill=True)
        out_dir = tmp_path / "out"

        rc = e9.main(["--enable-chunked-prefill", "--out", str(out_dir)])

        assert rc == 0
        assert (out_dir / "sweep.csv").exists()

    def test_no_prompt_length_fits_returns_2(
            self, e9, monkeypatch, tmp_path, capsys):
        calls = []
        # max_model_len - gen_tokens (default 128) is negative, so every
        # power-of-two default length is dropped and the grid is empty.
        _patch_engine(monkeypatch, e9, max_model_len=100,
                      record_run_sweep_args=calls)
        out_dir = tmp_path / "out"

        rc = e9.main(["--out", str(out_dir)])

        assert rc == 2
        assert calls == []
        assert not (out_dir / "sweep.csv").exists()
        err = capsys.readouterr().err
        assert "no prompt length fits" in err.lower()

    def test_over_long_prompt_len_dropped_with_warning(
            self, e9, monkeypatch, tmp_path, capsys):
        calls = []
        # max_model_len=2000, gen_tokens=100 -> max_prompt=1900. 1000 fits,
        # 999999 does not.
        _patch_engine(monkeypatch, e9, max_model_len=2000,
                      rows=[_fake_row(1, 1000, 100, 10.0, 1.0)],
                      record_run_sweep_args=calls)
        out_dir = tmp_path / "out"

        rc = e9.main(["--prompt-lens", "1000,999999", "--gen-tokens", "100",
                     "--out", str(out_dir)])

        assert rc == 0
        assert len(calls) == 1
        assert calls[0]["lengths"] == [1000]
        err = capsys.readouterr().err
        assert "999999" in err
        assert "dropping" in err.lower()
        assert (out_dir / "sweep.csv").exists()
