"""Tests for scripts/monitor_resources.py.

DCGM and psutil are mocked -- no real GPU / system access. We only
exercise the pure-Python logic: PID discovery from .pid files, blank-
value filtering, and the main loop's row shape.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "monitor_resources.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("monitor_resources", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["monitor_resources"] = module
    spec.loader.exec_module(module)
    return module


# ---------- DCGM blank-value filtering ----------


def test_is_dcgm_blank_recognises_sentinel_ints(mod):
    # Includes both int64 max (0x7FFFFFFFFFFFFFFF) and DCGM's "blank
    # float" bit pattern (9223372036854775792). Both are in the
    # _DCGM_BLANK_VALUES set and must be filtered.
    for v in (-1, -2, -3, -4, -5, -6, 0x7FFFFFFFFFFFFFFF, 9223372036854775792):
        assert mod._is_dcgm_blank(v), f"{v} should be detected as blank"


def test_is_dcgm_blank_recognises_nan(mod):
    assert mod._is_dcgm_blank(float("nan"))


def test_is_dcgm_blank_accepts_normal_values(mod):
    for v in (0.0, 0, 1, 42.5, 100):
        assert not mod._is_dcgm_blank(v)


def test_extract_dcgm_value_unwraps_time_series(mod):
    """`samples.GetLatest()` returns a DcgmFieldValueTimeSeries wrapper
    (despite the singular name). Without unwrapping, json.dumps blows
    up at runtime with "Object of type DcgmFieldValueTimeSeries is
    not JSON serializable" -- pin the unwrap path."""

    class FakeFieldValue:
        def __init__(self, value):
            self.value = value

    class FakeTimeSeries:
        def __init__(self, values):
            self.values = values  # list of FakeFieldValue

    # Shape 1: TimeSeries with samples → last .value wins
    ts = FakeTimeSeries([FakeFieldValue(0.85), FakeFieldValue(0.92)])
    assert mod._extract_dcgm_value(ts) == 0.92

    # Shape 2: singleton FieldValue with .value attribute
    fv = FakeFieldValue(42)
    assert mod._extract_dcgm_value(fv) == 42

    # Shape 3: already-scalar
    assert mod._extract_dcgm_value(3.14) == 3.14
    assert mod._extract_dcgm_value(7) == 7

    # Empty TimeSeries falls through gracefully (no sample yet)
    empty = FakeTimeSeries([])
    assert mod._extract_dcgm_value(empty) is None

    # None propagates as None (caller should skip the field)
    assert mod._extract_dcgm_value(None) is None


def test_prof_sampled_fields_constant_matches_required_set(mod):
    """The PROF_SAMPLED_FIELDS constant is the contract that drives the
    profiling.WatchFields split. Pin its exact members so a typo or
    accidental rename surfaces here, not as silent SM_ACTIVE=0 readings
    on the GPU host."""
    assert mod.PROF_SAMPLED_FIELDS == frozenset({
        "DCGM_FI_PROF_SM_ACTIVE",
        "DCGM_FI_PROF_SM_OCCUPANCY",
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
        "DCGM_FI_PROF_PIPE_FP32_ACTIVE",
        "DCGM_FI_PROF_PIPE_FP16_ACTIVE",
        "DCGM_FI_PROF_DRAM_ACTIVE",
    })
    # Things that MUST NOT be moved into the profiling group: PCIE/
    # NVLINK byte counters (hardware counters, work without perfworks)
    # and all DCGM_FI_DEV_* (NVML-backed).
    must_stay_regular = {
        "DCGM_FI_PROF_PCIE_RX_BYTES", "DCGM_FI_PROF_PCIE_TX_BYTES",
        "DCGM_FI_PROF_NVLINK_RX_BYTES", "DCGM_FI_PROF_NVLINK_TX_BYTES",
        "DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_FB_USED",
        "DCGM_FI_DEV_POWER_USAGE", "DCGM_FI_DEV_SM_CLOCK",
    }
    assert not (must_stay_regular & mod.PROF_SAMPLED_FIELDS), (
        "PROF byte counters and DEV fields must stay on samples.WatchFields"
    )


# ---------- DcgmSampler watch routing ----------


def _build_fake_dcgm(gpu_ids=(0, 1)):
    """Construct fake (ds, df, pd) triple emulating pydcgm enough for
    DcgmSampler.__init__ + sample() to run. Returns (triple, recorder)
    where:
      recorder.calls          [(api, fg_name, update_freq_us, fids), ...]
      recorder.drain_responses queue of dicts that the next
                              GetAllSinceLastCall returns. Each entry is
                              {gpu_id: {fid: [FakeFieldValueRecord, ...]}}.
                              Pop one per sample() call. When empty,
                              GetAllSinceLastCall returns an empty
                              collection (no buffered samples).
      recorder.drain_calls    number of GetAllSinceLastCall invocations
      recorder.last_dfvc_args [(dfvc_arg, fieldGroup_name), ...] -- pinned
                              so we can assert the SAME dfvc instance is
                              reused across calls (cursor preservation).
    """
    recorder = type("R", (), {})()
    recorder.calls = []
    recorder.drain_responses = []
    recorder.drain_calls = 0
    recorder.last_dfvc_args = []
    # FakeFieldValueCollection instances we've returned, in order. Tests
    # cross-check against `last_dfvc_args` to pin "the sampler reused the
    # collection we handed back" semantics (cursor preservation).
    recorder.returned_collections = []

    fake_df = MagicMock()
    # Auto-resolve any DCGM_FI_* name to a stable fake int id.
    fake_df._ids = {}
    def _attr(name):
        if not name.startswith("DCGM_FI_"):
            raise AttributeError(name)
        if name not in fake_df._ids:
            fake_df._ids[name] = len(fake_df._ids) + 100
        return fake_df._ids[name]
    fake_df.__getattr__ = lambda self, name: _attr(name)
    fake_df.configure_mock(**{
        n: _attr(n) for n in (
            "DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_FB_USED",
            "DCGM_FI_PROF_SM_ACTIVE", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
            "DCGM_FI_PROF_PCIE_RX_BYTES",
        )
    })

    fake_ds = MagicMock()
    fake_ds.DCGM_OPERATION_MODE_AUTO = 0
    fake_ds.DCGM_GROUP_DEFAULT = 0

    class FakeFieldGroup:
        def __init__(self, handle, name, fieldIds):
            self.name = name
            self.fieldIds = list(fieldIds)

    class FakeFieldValueCollection:
        def __init__(self, payload):
            # payload: {gpu_id: {fid: FakeTimeSeries}}
            self.values = payload

    class FakeWatcher:
        def __init__(self, api_label):
            self.api_label = api_label
        def WatchFields(self, fg, updateFreq, maxKeepAge, maxKeepSamples):
            recorder.calls.append((self.api_label, fg.name, updateFreq, tuple(fg.fieldIds)))
        def GetAllSinceLastCall(self, dfvc, fg):
            recorder.drain_calls += 1
            recorder.last_dfvc_args.append((dfvc, fg.name))
            payload_raw = (
                recorder.drain_responses.pop(0)
                if recorder.drain_responses else {}
            )
            payload = {
                gpu: {fid: FakeTimeSeries(records) for fid, records in by_field.items()}
                for gpu, by_field in payload_raw.items()
            }
            collection = FakeFieldValueCollection(payload)
            recorder.returned_collections.append(collection)
            return collection

    class FakeGroup:
        def __init__(self, *a, **kw):
            self.samples = FakeWatcher("samples")
            self.profiling = FakeWatcher("profiling")

    class FakeHandle:
        def __init__(self, *a, **kw):
            pass
        def GetSystem(self):
            sys_obj = MagicMock()
            sys_obj.discovery.GetAllSupportedGpuIds.return_value = list(gpu_ids)
            return sys_obj
        def Shutdown(self):
            pass

    fake_pd = MagicMock()
    fake_pd.DcgmHandle.side_effect = FakeHandle
    fake_pd.DcgmGroup.side_effect = FakeGroup
    fake_pd.DcgmFieldGroup.side_effect = FakeFieldGroup
    return (fake_ds, fake_df, fake_pd), recorder


class FakeFieldValueRecord:
    """Mirrors DcgmFieldValue_v1: exposes `.value`, `.ts`, `.isBlank`."""
    def __init__(self, value, ts=0, is_blank=False):
        self.value = value
        self.ts = ts
        self.isBlank = is_blank


class FakeTimeSeries:
    """Mirrors DcgmFieldValueTimeSeries -- `.values` list of records."""
    def __init__(self, records):
        self.values = list(records)


def test_dcgm_sampler_splits_prof_sampled_into_dedicated_field_group(mod, monkeypatch):
    """Real bug regression-guard: PROF_SM_ACTIVE / PIPE_* / DRAM_ACTIVE
    MUST live in a SEPARATE field group from non-profiling fields.
    Mixing them into the regular group suppresses perfworks auto-
    activation in DCGM 3.x and they silently return 0 (observed live
    with vLLM serving for 30+ minutes).

    Both groups still go through samples.WatchFields — DCGM 3.x's
    auto-detection looks at field group membership, not which API
    method was called."""
    triple, recorder = _build_fake_dcgm()
    monkeypatch.setattr(mod, "_import_dcgm", lambda: triple)

    mod.DcgmSampler([
        "DCGM_FI_DEV_GPU_UTIL",
        "DCGM_FI_DEV_FB_USED",
        "DCGM_FI_PROF_PCIE_RX_BYTES",
        "DCGM_FI_PROF_SM_ACTIVE",
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    ], update_freq_us=5_000_000)

    samples_calls = [c for c in recorder.calls if c[0] == "samples"]
    assert len(samples_calls) == 2, (
        f"expected TWO samples.WatchFields calls (regular fg + PROF "
        f"sampled fg, separated for perfworks activation); got "
        f"{samples_calls}"
    )

    # Locate which call is which by field-group name.
    by_name = {fg_name: (freq, fids) for _, fg_name, freq, fids in samples_calls}
    assert "testbed_monitor_regular_fg" in by_name
    assert "testbed_monitor_prof_fg" in by_name

    # Regular group: DEV_* + PROF byte counters (which don't need perfworks)
    _, regular_fids = by_name["testbed_monitor_regular_fg"]
    regular_names = {
        n for n in ("DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_FB_USED",
                    "DCGM_FI_PROF_PCIE_RX_BYTES")
        if getattr(triple[1], n) in regular_fids
    }
    assert regular_names == {
        "DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_FB_USED",
        "DCGM_FI_PROF_PCIE_RX_BYTES",
    }
    # PROF sampled must NOT leak into the regular group (the whole bug)
    assert getattr(triple[1], "DCGM_FI_PROF_SM_ACTIVE") not in regular_fids

    # PROF group: only PROF sampled fields
    prof_freq, prof_fids = by_name["testbed_monitor_prof_fg"]
    prof_names = {
        n for n in ("DCGM_FI_PROF_SM_ACTIVE", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE")
        if getattr(triple[1], n) in prof_fids
    }
    assert prof_names == {"DCGM_FI_PROF_SM_ACTIVE", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"}

    # PROF update freq capped at 1s even when caller asks for 5s.
    assert prof_freq == 1_000_000, (
        f"prof updateFreq must be capped at 1s (1_000_000us), got {prof_freq}"
    )


def test_dcgm_sampler_skips_prof_group_when_no_prof_sampled_requested(mod, monkeypatch):
    """Caller asked only for regular fields → no PROF group created,
    only one samples.WatchFields call."""
    triple, recorder = _build_fake_dcgm()
    monkeypatch.setattr(mod, "_import_dcgm", lambda: triple)

    mod.DcgmSampler([
        "DCGM_FI_DEV_GPU_UTIL",
        "DCGM_FI_PROF_PCIE_RX_BYTES",  # counter, not perfworks-sampled
    ])

    samples_calls = [c for c in recorder.calls if c[0] == "samples"]
    assert len(samples_calls) == 1, (
        f"expected exactly one samples.WatchFields call when no PROF "
        f"sampled fields requested; got {samples_calls}"
    )
    fg_name = samples_calls[0][1]
    assert fg_name == "testbed_monitor_regular_fg"


def test_dcgm_sampler_default_update_freq_is_100ms(mod, monkeypatch):
    """Default DCGM internal sampling period is 100ms (10Hz). The
    output drain cadence (--interval) is decoupled and defaults to 1s
    in the CLI -- so each row aggregates ~10 samples per field."""
    triple, recorder = _build_fake_dcgm()
    monkeypatch.setattr(mod, "_import_dcgm", lambda: triple)
    mod.DcgmSampler(["DCGM_FI_DEV_GPU_UTIL"])
    samples_calls = [c for c in recorder.calls if c[0] == "samples"]
    assert len(samples_calls) == 1
    _, _, freq, _ = samples_calls[0]
    assert freq == 100_000, f"default update freq should be 100_000us, got {freq}"


def test_counter_fields_constant_matches_required_set(mod):
    """COUNTER_FIELDS is the contract that decides which fields keep
    LAST-value semantics (vs gauge → mean/min/max aggregation).
    Cumulative byte counters MUST stay last-value or downstream
    (last_curr - last_prev) / interval bandwidth math goes negative
    when window-averaging."""
    assert mod.COUNTER_FIELDS == frozenset({
        "DCGM_FI_PROF_PCIE_RX_BYTES",
        "DCGM_FI_PROF_PCIE_TX_BYTES",
        "DCGM_FI_PROF_NVLINK_RX_BYTES",
        "DCGM_FI_PROF_NVLINK_TX_BYTES",
    })
    # Gauges (point-in-time) must NOT be in COUNTER_FIELDS or we'd
    # lose the variance information the user explicitly asked for.
    gauges = {
        "DCGM_FI_PROF_SM_ACTIVE", "DCGM_FI_DEV_FB_USED",
        "DCGM_FI_DEV_POWER_USAGE", "DCGM_FI_DEV_GPU_TEMP",
        "DCGM_FI_DEV_SM_CLOCK", "DCGM_FI_DEV_GPU_UTIL",
    }
    assert not (gauges & mod.COUNTER_FIELDS)


def test_iter_buffered_values_skips_blanks_and_handles_shapes(mod):
    """`_iter_buffered_values` drives the gauge aggregation; pin the
    shapes it accepts so a DCGM-binding upgrade can't silently make
    the entire window aggregate empty."""
    # Records with .isBlank flag → skipped.
    ts = FakeTimeSeries([
        FakeFieldValueRecord(0.10),
        FakeFieldValueRecord(0.20, is_blank=True),  # skip
        FakeFieldValueRecord(0.30),
    ])
    assert list(mod._iter_buffered_values(ts)) == [0.10, 0.30]

    # Records with sentinel int values (older bindings without isBlank).
    ts = FakeTimeSeries([
        FakeFieldValueRecord(42),
        FakeFieldValueRecord(-1),                    # sentinel → skip
        FakeFieldValueRecord(0x7FFFFFFFFFFFFFFF),    # sentinel → skip
        FakeFieldValueRecord(99),
    ])
    assert list(mod._iter_buffered_values(ts)) == [42, 99]

    # None series → empty.
    assert list(mod._iter_buffered_values(None)) == []

    # Empty TimeSeries.
    assert list(mod._iter_buffered_values(FakeTimeSeries([]))) == []

    # Bare iterable of records (fallback for bindings that don't wrap).
    bare = [FakeFieldValueRecord(1.0), FakeFieldValueRecord(2.0)]
    assert list(mod._iter_buffered_values(bare)) == [1.0, 2.0]


def test_dcgm_sampler_aggregates_gauges_to_mean_min_max_n(mod, monkeypatch):
    """The headline change: gauge fields drain ALL buffered samples per
    window and emit {mean, min, max, n} -- not just GetLatest's
    point-in-time scalar. Without this you can't tell whether SM was
    flat at 0.4 or oscillating between 0.1 and 0.8 inside the second."""
    triple, recorder = _build_fake_dcgm(gpu_ids=(0,))
    monkeypatch.setattr(mod, "_import_dcgm", lambda: triple)
    sampler = mod.DcgmSampler(["DCGM_FI_PROF_SM_ACTIVE"])

    fid = triple[1].DCGM_FI_PROF_SM_ACTIVE
    recorder.drain_responses.append({
        0: {fid: [
            FakeFieldValueRecord(0.10), FakeFieldValueRecord(0.30),
            FakeFieldValueRecord(0.50), FakeFieldValueRecord(0.70),
            FakeFieldValueRecord(0.90),
        ]},
    })
    rows = sampler.sample()
    gpu0 = next(r for r in rows if r["index"] == 0)
    sm = gpu0["DCGM_FI_PROF_SM_ACTIVE"]
    assert isinstance(sm, dict), f"gauge field must aggregate to a dict, got {type(sm)}"
    assert sm["n"] == 5
    assert sm["mean"] == pytest.approx(0.50)
    assert sm["min"] == pytest.approx(0.10)
    assert sm["max"] == pytest.approx(0.90)


def test_dcgm_sampler_keeps_counter_fields_as_last_value(mod, monkeypatch):
    """Cumulative counters must NOT aggregate -- downstream computes
    bandwidth as (last_curr - last_prev) / interval, which only works
    if every sample preserves the latest cumulative reading."""
    triple, recorder = _build_fake_dcgm(gpu_ids=(0,))
    monkeypatch.setattr(mod, "_import_dcgm", lambda: triple)
    sampler = mod.DcgmSampler(["DCGM_FI_PROF_PCIE_RX_BYTES"])

    fid = triple[1].DCGM_FI_PROF_PCIE_RX_BYTES
    recorder.drain_responses.append({
        0: {fid: [
            FakeFieldValueRecord(1_000_000),
            FakeFieldValueRecord(2_500_000),
            FakeFieldValueRecord(4_750_000),   # last wins
        ]},
    })
    rows = sampler.sample()
    gpu0 = next(r for r in rows if r["index"] == 0)
    assert gpu0["DCGM_FI_PROF_PCIE_RX_BYTES"] == 4_750_000, (
        "counter must keep the last buffered value, not aggregate"
    )


def test_dcgm_sampler_skips_field_with_empty_buffer(mod, monkeypatch):
    """If a field has nothing buffered (perfworks slow to come up,
    field unsupported on this GPU, etc), the field is OMITTED from
    the entry rather than emitting `null` or an empty dict."""
    triple, recorder = _build_fake_dcgm(gpu_ids=(0,))
    monkeypatch.setattr(mod, "_import_dcgm", lambda: triple)
    sampler = mod.DcgmSampler([
        "DCGM_FI_PROF_SM_ACTIVE",
        "DCGM_FI_DEV_GPU_UTIL",
    ])
    fid_util = triple[1].DCGM_FI_DEV_GPU_UTIL
    recorder.drain_responses.append({
        0: {fid_util: [FakeFieldValueRecord(45.0)]},
        # SM_ACTIVE intentionally absent from this drain
    })
    rows = sampler.sample()
    gpu0 = next(r for r in rows if r["index"] == 0)
    assert "DCGM_FI_DEV_GPU_UTIL" in gpu0
    assert "DCGM_FI_PROF_SM_ACTIVE" not in gpu0, (
        "absent field must not appear in the entry"
    )


def test_dcgm_sampler_reuses_dfvc_across_drains(mod, monkeypatch):
    """`GetAllSinceLastCall` carries a since-timestamp cursor inside
    the DcgmFieldValueCollection it returns. Constructing a fresh
    collection on every drain would replay the entire ring buffer.
    Pin that the sampler passes its prior dfvc back in on call 2+."""
    triple, recorder = _build_fake_dcgm(gpu_ids=(0,))
    monkeypatch.setattr(mod, "_import_dcgm", lambda: triple)
    sampler = mod.DcgmSampler(["DCGM_FI_DEV_GPU_UTIL"])

    recorder.drain_responses.extend([{}, {}])
    sampler.sample()
    sampler.sample()

    assert recorder.drain_calls == 2
    first_dfvc_arg = recorder.last_dfvc_args[0][0]
    second_dfvc_arg = recorder.last_dfvc_args[1][0]
    assert first_dfvc_arg is None, "first drain should pass None (no cursor yet)"
    # IDENTITY check (not `is not None`): the sampler must hand back the
    # exact collection returned by drain #1. Any code that resets the
    # cursor (e.g. `self._dfvc = None` at the top of sample(), or
    # constructing a fresh collection) replays DCGM's entire ring buffer
    # each tick -- a silent correctness bug that "is not None" wouldn't
    # catch because a fresh collection is also not None.
    assert second_dfvc_arg is recorder.returned_collections[0], (
        "second drain MUST pass the SAME collection object returned by "
        "drain #1 (carries _nextSinceTimestamp cursor). Producing a "
        "fresh collection replays the entire DCGM ring buffer."
    )


def test_dcgm_sampler_emits_nothing_for_gpu_with_no_data(mod, monkeypatch):
    """Multi-GPU host where one GPU has no samples this drain (e.g.
    just started). Entry still appears with only the index field."""
    triple, recorder = _build_fake_dcgm(gpu_ids=(0, 1))
    monkeypatch.setattr(mod, "_import_dcgm", lambda: triple)
    sampler = mod.DcgmSampler(["DCGM_FI_DEV_GPU_UTIL"])

    fid = triple[1].DCGM_FI_DEV_GPU_UTIL
    recorder.drain_responses.append({
        0: {fid: [FakeFieldValueRecord(30.0), FakeFieldValueRecord(50.0)]},
        # gpu 1 absent
    })
    rows = sampler.sample()
    by_idx = {r["index"]: r for r in rows}
    assert by_idx[0]["DCGM_FI_DEV_GPU_UTIL"]["mean"] == pytest.approx(40.0)
    assert by_idx[1] == {"index": 1}


def test_to_json_value_coerces_bytes_and_passes_scalars(mod):
    """Driver-version / model-name fields come back as bytes; need a
    safe coerce so the whole NDJSON line doesn't crash on str()."""
    assert mod._to_json_value(None) is None
    assert mod._to_json_value(42) == 42
    assert mod._to_json_value(3.14) == 3.14
    assert mod._to_json_value("ok") == "ok"
    assert mod._to_json_value(True) is True
    assert mod._to_json_value(b"driver-535.86") == "driver-535.86"

    class Weird: pass
    # Unknown types stringify rather than crash json.dumps later
    out = mod._to_json_value(Weird())
    assert isinstance(out, str)


# ---------- CpuSampler PID discovery ----------


def test_cpu_sampler_discovers_pids_from_directory(mod, tmp_path, monkeypatch):
    # Build a fake psutil module that records which PIDs are constructed.
    fake_psutil = MagicMock()
    fake_psutil.cpu_percent.return_value = 0.0
    fake_psutil.cpu_count.return_value = 8

    constructed: list[int] = []

    class FakeProcess:
        def __init__(self, pid: int):
            constructed.append(pid)
            self.pid = pid
            self._alive = True

        def is_running(self) -> bool:
            return self._alive

        def cpu_percent(self, interval=None):
            return 42.0

        def oneshot(self):
            class _Ctx:
                def __enter__(self_): return None
                def __exit__(self_, *a): return False
            return _Ctx()

        def memory_info(self):
            m = MagicMock()
            m.rss = 1024 * 1024
            return m

        def num_threads(self):
            return 4

        def children(self, recursive=False):
            return []

    fake_psutil.Process.side_effect = FakeProcess
    fake_psutil.NoSuchProcess = Exception
    fake_psutil.AccessDenied = Exception

    mem = MagicMock()
    mem.total = 8 * (1 << 30)
    mem.used = 2 * (1 << 30)
    mem.available = 6 * (1 << 30)
    fake_psutil.virtual_memory.return_value = mem

    monkeypatch.setattr(mod, "_import_psutil", lambda: fake_psutil)

    # Lay down two .pid files that point at this process (alive PIDs).
    (tmp_path / "vllm-d0.pid").write_text(f"{12345}\n")
    (tmp_path / "opencode.pid").write_text(f"{23456}\n")

    cpu = mod.CpuSampler(tmp_path)
    sample = cpu.sample()

    assert sorted(p["name"] for p in sample["processes"]) == ["opencode", "vllm-d0"]
    for p in sample["processes"]:
        assert p["cpu_util_pct"] == 42.0
        assert p["rss_bytes"] == 1024 * 1024
        assert p["n_threads"] == 4
        assert p["n_procs"] == 1   # no children → tree size 1
    assert sample["host"]["n_cores"] == 8
    assert sample["host"]["mem_total_bytes"] == 8 * (1 << 30)


def test_cpu_sampler_sums_process_tree(mod, tmp_path, monkeypatch):
    """The headline fix: cpu/rss/threads are SUMMED over the watched PID
    plus its recursive children. `bun run dev` (the recorded PID) is an
    idle launcher; the real opencode server runs in a child that setsid's
    out -- sampling only the parent reads ~0%. Also pins child warm-up
    semantics: a child's cpu_percent(interval=None) returns 0.0 on its
    FIRST sighting (no prior reading to delta against) and the real value
    only on the NEXT sample, so the cached child object must be reused
    across samples rather than reconstructed."""
    fake_psutil = MagicMock()
    fake_psutil.cpu_percent.return_value = 7.0
    fake_psutil.cpu_count.return_value = 16
    fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    fake_psutil.AccessDenied = type("AccessDenied", (Exception,), {})

    # cpu_percent warm-up emulation: first call per Process returns 0.0,
    # subsequent calls return `steady`.
    class FakeProc:
        def __init__(self, pid, steady, threads, rss, kids=()):
            self.pid = pid
            self._steady = steady
            self._threads = threads
            self._rss = rss
            self._kids = list(kids)
            self._warmed = False

        def is_running(self): return True

        def cpu_percent(self, interval=None):
            if not self._warmed:
                self._warmed = True
                return 0.0
            return self._steady

        def oneshot(self):
            class _Ctx:
                def __enter__(s): return None
                def __exit__(s, *a): return False
            return _Ctx()

        def memory_info(self):
            m = MagicMock(); m.rss = self._rss; return m

        def num_threads(self): return self._threads

        def children(self, recursive=False):
            return list(self._kids)

    # child_spec stores the child's parameters so we can manufacture fresh
    # objects on each parent.children() call.  Real psutil.children() returns
    # newly-constructed Process objects from the live process table; always
    # returning the same object would mask a caching-removal regression: the
    # pre-warmed _warmed flag on the shared object would persist across
    # samples and the sample-2 assertion (cpu_util_pct==83.0) would pass
    # even if _tree() dropped its cache entirely.
    child_spec = dict(pid=1002, steady=80.0, threads=10, rss=200 << 20)
    parent = FakeProc(1001, steady=3.0, threads=2, rss=50 << 20)
    # Override children() on the parent instance to return a FRESH FakeProc
    # on every call.  Without the cache in _tree(), sample 2 would receive a
    # brand-new _warmed=False object whose cpu_percent() returns 0.0, making
    # the 83.0 assertion fail and catching the regression.
    parent.children = lambda recursive=False: [FakeProc(**child_spec)]

    # Parent is warmed at _refresh_pids() (so it returns 3.0 from sample 1);
    # the child is only discovered inside sample() and warms there.
    fake_psutil.Process.side_effect = lambda pid: parent
    mem = MagicMock(); mem.total = 0; mem.used = 0; mem.available = 0
    fake_psutil.virtual_memory.return_value = mem
    monkeypatch.setattr(mod, "_import_psutil", lambda: fake_psutil)

    (tmp_path / "opencode.pid").write_text("1001\n")
    cpu = mod.CpuSampler(tmp_path)

    # Sample 1: child seen for the first time → contributes 0.0 (warm-up),
    # but rss/threads/n_procs are summed immediately.
    s1 = next(p for p in cpu.sample()["processes"] if p["name"] == "opencode")
    assert s1["n_procs"] == 2, "parent + 1 child must both be counted"
    assert s1["cpu_util_pct"] == pytest.approx(3.0), "child still warming → 0 this sample"
    assert s1["rss_bytes"] == (50 << 20) + (200 << 20)
    assert s1["n_threads"] == 12

    # Sample 2: cached child now warmed → its real 80% is summed in.
    s2 = next(p for p in cpu.sample()["processes"] if p["name"] == "opencode")
    assert s2["cpu_util_pct"] == pytest.approx(83.0), (
        "child must be reused across samples so cpu_percent deltas accrue; "
        "reconstructing it each tick would peg it at 0.0 forever"
    )
    assert s2["n_procs"] == 2


def test_cpu_sampler_evicts_dead_children_from_cache(mod, tmp_path, monkeypatch):
    """Per-request child processes churn. A child present in one sample
    but gone the next must be dropped from the tree (and its cache entry),
    not keep contributing a stale reading."""
    fake_psutil = MagicMock()
    fake_psutil.cpu_percent.return_value = 0.0
    fake_psutil.cpu_count.return_value = 8
    fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    fake_psutil.AccessDenied = type("AccessDenied", (Exception,), {})

    class FakeProc:
        def __init__(self, pid, threads=1, rss=0):
            self.pid = pid
            self._threads = threads
            self._rss = rss
            self.kids = []
        def is_running(self): return True
        def cpu_percent(self, interval=None): return 1.0
        def oneshot(self):
            class _Ctx:
                def __enter__(s): return None
                def __exit__(s, *a): return False
            return _Ctx()
        def memory_info(self):
            m = MagicMock(); m.rss = self._rss; return m
        def num_threads(self): return self._threads
        def children(self, recursive=False): return list(self.kids)

    parent = FakeProc(2001)
    transient = FakeProc(2002)
    parent.kids = [transient]

    fake_psutil.Process.side_effect = lambda pid: parent
    mem = MagicMock(); mem.total = 0; mem.used = 0; mem.available = 0
    fake_psutil.virtual_memory.return_value = mem
    monkeypatch.setattr(mod, "_import_psutil", lambda: fake_psutil)

    (tmp_path / "opencode.pid").write_text("2001\n")
    cpu = mod.CpuSampler(tmp_path)

    s1 = next(p for p in cpu.sample()["processes"] if p["name"] == "opencode")
    assert s1["n_procs"] == 2
    assert transient.pid in cpu._children[parent.pid], "child should be cached"

    parent.kids = []  # transient child exited
    s2 = next(p for p in cpu.sample()["processes"] if p["name"] == "opencode")
    assert s2["n_procs"] == 1, "dead child must drop out of the tree"
    assert transient.pid not in cpu._children[parent.pid], (
        "dead child must be evicted from the cache, not linger"
    )


def test_cpu_sampler_drops_pids_when_file_disappears(mod, tmp_path, monkeypatch):
    """testbed.sh removes a component's .pid file when down'd. The
    sampler should forget that process at the next refresh."""
    fake_psutil = MagicMock()
    fake_psutil.cpu_percent.return_value = 0.0
    fake_psutil.cpu_count.return_value = 4
    fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    fake_psutil.AccessDenied = type("AccessDenied", (Exception,), {})

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def is_running(self): return True
        def cpu_percent(self, interval=None): return 1.0

        def oneshot(self):
            class _Ctx:
                def __enter__(self_): return None
                def __exit__(self_, *a): return False
            return _Ctx()

        def memory_info(self):
            m = MagicMock(); m.rss = 0; return m

        def num_threads(self): return 1

        def children(self, recursive=False): return []

    fake_psutil.Process.side_effect = FakeProcess
    mem = MagicMock(); mem.total = 0; mem.used = 0; mem.available = 0
    fake_psutil.virtual_memory.return_value = mem

    monkeypatch.setattr(mod, "_import_psutil", lambda: fake_psutil)

    pidfile = tmp_path / "frontend.pid"
    pidfile.write_text("999\n")
    cpu = mod.CpuSampler(tmp_path)
    assert any(p["name"] == "frontend" for p in cpu.sample()["processes"])

    pidfile.unlink()  # testbed.sh down frontend would do this
    sample = cpu.sample()
    assert not any(p["name"] == "frontend" for p in sample["processes"])


def test_cpu_sampler_replaces_dead_process_when_pidfile_unchanged(mod, tmp_path, monkeypatch):
    """When a worker crashes but its .pid file still points at the same
    PID, the sampler's refresh path must detect `is_running()==False`
    on the existing Process reference and reconstruct a fresh
    Process(pid) at the next tick -- pin that uncovered branch."""
    fake_psutil = MagicMock()
    fake_psutil.cpu_percent.return_value = 0.0
    fake_psutil.cpu_count.return_value = 4
    fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    fake_psutil.AccessDenied = type("AccessDenied", (Exception,), {})

    constructed_pids: list[int] = []

    class FakeProcess:
        def __init__(self, pid):
            constructed_pids.append(pid)
            self.pid = pid

        # Every existing-instance check returns False -> the refresh
        # path must always replace the stored Process(123).
        def is_running(self): return False
        def cpu_percent(self, interval=None): return 0.0

        def oneshot(self):
            class _Ctx:
                def __enter__(self_): return None
                def __exit__(self_, *a): return False
            return _Ctx()

        def memory_info(self):
            m = MagicMock(); m.rss = 0; return m

        def num_threads(self): return 1

        def children(self, recursive=False): return []

    fake_psutil.Process.side_effect = FakeProcess
    mem = MagicMock(); mem.total = 0; mem.used = 0; mem.available = 0
    fake_psutil.virtual_memory.return_value = mem
    monkeypatch.setattr(mod, "_import_psutil", lambda: fake_psutil)

    (tmp_path / "worker.pid").write_text("123\n")
    cpu = mod.CpuSampler(tmp_path)   # init's _refresh_pids constructs Process(123) #1
    cpu.sample()                      # sample()'s _refresh_pids: is_running()==False → Process(123) #2
    cpu.sample()                      # again → Process(123) #3
    assert constructed_pids.count(123) >= 2, (
        "stale dead Process should be replaced when .pid file's PID is "
        f"unchanged but the process is dead; constructed={constructed_pids}"
    )


def test_cpu_sampler_handles_empty_pids_dir(mod, tmp_path, monkeypatch):
    fake_psutil = MagicMock()
    fake_psutil.cpu_percent.return_value = 0.0
    fake_psutil.cpu_count.return_value = 1
    mem = MagicMock(); mem.total = 0; mem.used = 0; mem.available = 0
    fake_psutil.virtual_memory.return_value = mem
    monkeypatch.setattr(mod, "_import_psutil", lambda: fake_psutil)
    cpu = mod.CpuSampler(tmp_path)  # empty directory
    sample = cpu.sample()
    assert sample["processes"] == []
    assert sample["host"]["n_cores"] == 1


def test_cpu_sampler_handles_no_pids_dir(mod, monkeypatch):
    """--pids-from omitted → host metrics only, no per-process."""
    fake_psutil = MagicMock()
    fake_psutil.cpu_percent.return_value = 0.0
    fake_psutil.cpu_count.return_value = 1
    mem = MagicMock(); mem.total = 0; mem.used = 0; mem.available = 0
    fake_psutil.virtual_memory.return_value = mem
    monkeypatch.setattr(mod, "_import_psutil", lambda: fake_psutil)
    cpu = mod.CpuSampler(None)
    sample = cpu.sample()
    assert sample["processes"] == []


# ---------- run_sampler main loop ----------


def test_run_sampler_writes_one_line_per_sample(mod, tmp_path):
    out_path = tmp_path / "resource.ndjson"
    samples = [
        {"gpus": [{"index": 0, "DCGM_FI_PROF_SM_ACTIVE": 0.85}]},
        {"gpus": [{"index": 0, "DCGM_FI_PROF_SM_ACTIVE": 0.92}]},
    ]
    fake_gpu = MagicMock()
    fake_gpu.sample.side_effect = [s["gpus"] for s in samples]

    fake_cpu = MagicMock()
    fake_cpu.sample.return_value = {"host": {"cpu_util_pct": 5.0}, "processes": []}

    state = {"count": 0}

    def stop():
        state["count"] += 1
        return state["count"] > 2   # stop after 2 samples

    n = mod.run_sampler(out_path, 0.0, fake_gpu, fake_cpu, stop)
    assert n == 2
    lines = [json.loads(line) for line in out_path.read_text().splitlines() if line]
    assert len(lines) == 2
    for line in lines:
        assert "ts" in line and "interval_s" in line
        assert "gpus" in line and line["gpus"][0]["index"] == 0
        assert line["host"]["cpu_util_pct"] == 5.0


def test_run_sampler_catches_gpu_exception_and_records_error(mod, tmp_path):
    out_path = tmp_path / "resource.ndjson"

    fake_gpu = MagicMock()
    fake_gpu.sample.side_effect = RuntimeError("DCGM died")
    fake_cpu = MagicMock()
    fake_cpu.sample.return_value = {"host": {"cpu_util_pct": 1.0}, "processes": []}

    state = {"count": 0}
    def stop():
        state["count"] += 1
        return state["count"] > 1
    n = mod.run_sampler(out_path, 0.0, fake_gpu, fake_cpu, stop)
    assert n == 1
    sample = json.loads(out_path.read_text().strip())
    assert sample["gpu_error"] == "DCGM died"
    # CPU still recorded -- one source failing must not block the other
    assert sample["host"]["cpu_util_pct"] == 1.0


def test_run_sampler_gpu_only(mod, tmp_path):
    out_path = tmp_path / "resource.ndjson"
    fake_gpu = MagicMock()
    fake_gpu.sample.return_value = [{"index": 0}]
    state = {"count": 0}
    def stop():
        state["count"] += 1
        return state["count"] > 1
    mod.run_sampler(out_path, 0.0, fake_gpu, None, stop)
    sample = json.loads(out_path.read_text().strip())
    assert sample["gpus"] == [{"index": 0}]
    assert "host" not in sample
