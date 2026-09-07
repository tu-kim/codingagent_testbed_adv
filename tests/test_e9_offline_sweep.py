"""Tests for scripts/arm/e9_offline_ttft_tpot_sweep.py.

No network, no GPU: this script drives vLLM's OFFLINE `LLM` API, which
needs a real accelerator and is not installed on this dev machine. Loaded
via importlib (script, not a package module) -- see
tests/test_e8_ttft_tpot.py for the same pattern.

Everything that would otherwise touch a real engine is exercised against
`_FakeLLM` (a `generate()` that records every call and returns a dummy
per-prompt list; nested attribute stubs are hung off an instance per-test
for the config-reader helpers) plus a fake `vllm` module injected into
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
# sla_limits
# ---------------------------------------------------------------------------

def _rw(batch, prompt_tokens, ttft_ms="", error=""):
    return {"batch": batch, "prompt_tokens": prompt_tokens,
            "ttft_ms": ttft_ms, "error": error}


class TestSlaLimits:
    def test_interpolated_crossing(self, e9):
        rows = [_rw(1, 1000, 50.0), _rw(1, 2000, 150.0)]
        out = e9.sla_limits(rows, "ttft_ms", 100.0)
        # y_ok=50 at x=1000, y_bad=150 at x=2000; frac=(100-50)/(150-50)=0.5
        # -> interp = 1000 + 0.5*(2000-1000) = 1500
        assert out == [{"metric": "ttft_ms", "sla": 100.0, "batch": 1,
                        "max_ok_measured": 1000, "max_ok_interp": 1500,
                        "crossed": True}]

    def test_never_crosses_reports_top_of_grid(self, e9):
        rows = [_rw(1, 1000, 10.0), _rw(1, 2000, 20.0), _rw(1, 4000, 30.0)]
        out = e9.sla_limits(rows, "ttft_ms", 100.0)
        assert out == [{"metric": "ttft_ms", "sla": 100.0, "batch": 1,
                        "max_ok_measured": 4000, "max_ok_interp": 4000,
                        "crossed": False}]

    def test_all_fail_reports_zero_and_crossed(self, e9):
        rows = [_rw(1, 1000, 500.0), _rw(1, 2000, 600.0)]
        out = e9.sla_limits(rows, "ttft_ms", 100.0)
        assert out == [{"metric": "ttft_ms", "sla": 100.0, "batch": 1,
                        "max_ok_measured": 0, "max_ok_interp": 0,
                        "crossed": True}]

    def test_error_and_blank_rows_excluded(self, e9):
        # An error row (tiny value, which would otherwise become the new
        # max_ok) and a blank-metric row both sit strictly between two real
        # points. Correct exclusion leaves the single real point as a
        # "never crosses" result rather than corrupting the curve with
        # either excluded row.
        rows = [
            _rw(1, 1000, 50.0),
            _rw(1, 1500, ""),
            _rw(1, 2000, 5.0, error="OOM: boom"),
        ]
        out = e9.sla_limits(rows, "ttft_ms", 100.0)
        assert out == [{"metric": "ttft_ms", "sla": 100.0, "batch": 1,
                        "max_ok_measured": 1000, "max_ok_interp": 1000,
                        "crossed": False}]

    def test_multiple_batches_sorted_ascending(self, e9):
        rows = [_rw(4, 1000, 10.0), _rw(1, 1000, 10.0)]
        out = e9.sla_limits(rows, "ttft_ms", 100.0)
        assert [r["batch"] for r in out] == [1, 4]


# ---------------------------------------------------------------------------
# print_sweep / print_sla
# ---------------------------------------------------------------------------

class TestPrintSweep:
    def test_error_cell_renders_as_x_missing_cell_as_dash(self, e9, capsys):
        rows = [
            _rw(1, 1000, 10.0),
            _rw(1, 2000, 20.0),
            _rw(4, 1000, "", error="OOM: boom"),
            # (4, 2000) is simply absent from `rows` -> a missing cell.
        ]
        e9.print_sweep(rows, "ttft_ms", "TTFT (ms)")
        out = capsys.readouterr().out
        assert "TTFT (ms) (rows = batch size, cols = prompt tokens):" in out
        lines = [l for l in out.splitlines() if l.strip()]
        row1 = next(l for l in lines if l.lstrip().startswith("1"))
        row4 = next(l for l in lines if l.lstrip().startswith("4"))
        assert row1.split()[1:] == ["10.0", "20.0"]
        assert row4.split()[1:] == ["x", "-"]


class TestPrintSla:
    def test_crossed_and_never_crossed_notes(self, e9, capsys):
        limits = [
            {"metric": "ttft_ms", "sla": 100.0, "batch": 1,
             "max_ok_measured": 1000, "max_ok_interp": 1500, "crossed": True},
            {"metric": "ttft_ms", "sla": 100.0, "batch": 2,
             "max_ok_measured": 4000, "max_ok_interp": 4000,
             "crossed": False},
        ]
        e9.print_sla(limits, "ms")
        out = capsys.readouterr().out
        assert "Longest prompt meeting ttft_ms <= 100 ms:" in out
        lines = [l for l in out.splitlines() if l.strip()]
        row1 = next(l for l in lines if l.split()[:1] == ["1"])
        row2 = next(l for l in lines if l.split()[:1] == ["2"])
        assert row1.split()[:3] == ["1", "1000", "1500"]
        assert "(never crossed within the grid)" not in row1
        assert "(never crossed within the grid)" in row2

    def test_empty_limits_prints_nothing(self, e9, capsys):
        e9.print_sla([], "ms")
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# main() -- --from-csv end to end
# ---------------------------------------------------------------------------

_SWEEP_COLS = ["batch", "prompt_tokens", "gen_tokens", "ttft_ms", "tpot_ms",
              "decode_ms", "total_ms", "error"]


def _write_sweep_csv(path: Path, rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_SWEEP_COLS)
        w.writeheader()
        w.writerows(rows)


def _sweep_row(batch, prompt_tokens, ttft_ms, tpot_ms, gen_tokens=8):
    decode_ms = tpot_ms * (gen_tokens - 1)
    return {"batch": batch, "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens, "ttft_ms": ttft_ms, "tpot_ms": tpot_ms,
            "decode_ms": decode_ms, "total_ms": ttft_ms + decode_ms,
            "error": ""}


class TestMainFromCsv:
    def test_end_to_end_returns_zero_and_prints_both_tables(
            self, e9, tmp_path, capsys):
        csv_path = tmp_path / "sweep.csv"
        _write_sweep_csv(csv_path, [
            _sweep_row(1, 1000, 50.0, 5.0),
            _sweep_row(1, 2000, 150.0, 6.0),
        ])
        out_dir = tmp_path / "out"
        rc = e9.main(["--from-csv", str(csv_path), "--out", str(out_dir),
                     "--no-figures"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "TTFT (ms)" in out
        assert "TPOT (ms/token)" in out
        # --from-csv skips measurement entirely -- it must not re-derive or
        # overwrite a sweep.csv of its own in --out.
        assert not (out_dir / "sweep.csv").exists()

    def test_sla_limits_csv_written_with_header_when_sla_flag_given(
            self, e9, tmp_path):
        csv_path = tmp_path / "sweep.csv"
        _write_sweep_csv(csv_path, [
            _sweep_row(1, 1000, 50.0, 5.0),
            _sweep_row(1, 2000, 150.0, 6.0),
        ])
        out_dir = tmp_path / "out"
        rc = e9.main(["--from-csv", str(csv_path), "--out", str(out_dir),
                     "--no-figures", "--ttft-sla-ms", "100"])
        assert rc == 0
        sla_path = out_dir / "sla_limits.csv"
        assert sla_path.exists()
        with sla_path.open() as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == [
                "metric", "sla", "batch", "max_ok_measured",
                "max_ok_interp", "crossed",
            ]
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["metric"] == "ttft_ms"

    def test_sla_limits_csv_not_written_without_any_sla_flag(
            self, e9, tmp_path):
        csv_path = tmp_path / "sweep.csv"
        _write_sweep_csv(csv_path, [_sweep_row(1, 1000, 50.0, 5.0)])
        out_dir = tmp_path / "out"
        rc = e9.main(["--from-csv", str(csv_path), "--out", str(out_dir),
                     "--no-figures"])
        assert rc == 0
        assert not (out_dir / "sla_limits.csv").exists()

    def test_empty_csv_returns_two(self, e9, tmp_path):
        csv_path = tmp_path / "empty.csv"
        _write_sweep_csv(csv_path, [])  # header only, zero data rows
        out_dir = tmp_path / "out"
        rc = e9.main(["--from-csv", str(csv_path), "--out", str(out_dir),
                     "--no-figures"])
        assert rc == 2


# ---------------------------------------------------------------------------
# plot_metric (matplotlib-dependent)
# ---------------------------------------------------------------------------

class TestPlotMetric:
    def test_writes_a_file_with_data(self, e9, tmp_path):
        pytest.importorskip("matplotlib")
        rows = [_rw(1, 1000, 50.0), _rw(1, 2000, 150.0),
                _rw(4, 1000, 80.0), _rw(4, 2000, 220.0)]
        limits = e9.sla_limits(rows, "ttft_ms", 100.0)
        out = tmp_path / "fig.pdf"
        e9.plot_metric(rows, "ttft_ms", "TTFT (ms)",
                       "TTFT vs prompt tokens", 100.0, limits, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_all_error_rows_render_no_data_without_raising(self, e9, tmp_path):
        pytest.importorskip("matplotlib")
        rows = [_rw(1, 1000, "", error="boom"),
                _rw(1, 2000, "", error="boom2")]
        out = tmp_path / "fig_empty.pdf"
        e9.plot_metric(rows, "ttft_ms", "TTFT (ms)", "title", None, [], out)
        assert out.exists()
        assert out.stat().st_size > 0
