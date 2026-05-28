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


def _build_fake_dcgm():
    """Construct fake (ds, df, pd) triple emulating pydcgm enough for
    DcgmSampler.__init__ to wire up watches. Returns (triple, recorder)
    where recorder.calls tracks every WatchFields invocation as
    (api, field_group_name, update_freq_us)."""
    recorder = type("R", (), {"calls": []})()

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

    class FakeWatcher:
        def __init__(self, api_label):
            self.api_label = api_label
        def WatchFields(self, fg, updateFreq, maxKeepAge, maxKeepSamples):
            recorder.calls.append((self.api_label, fg.name, updateFreq, tuple(fg.fieldIds)))

    class FakeGroup:
        def __init__(self, *a, **kw):
            self.samples = FakeWatcher("samples")
            self.profiling = FakeWatcher("profiling")

    class FakeHandle:
        def __init__(self, *a, **kw):
            pass
        def GetSystem(self):
            sys_obj = MagicMock()
            sys_obj.discovery.GetAllSupportedGpuIds.return_value = [0, 1]
            return sys_obj
        def Shutdown(self):
            pass

    fake_pd = MagicMock()
    fake_pd.DcgmHandle.side_effect = FakeHandle
    fake_pd.DcgmGroup.side_effect = FakeGroup
    fake_pd.DcgmFieldGroup.side_effect = FakeFieldGroup
    return (fake_ds, fake_df, fake_pd), recorder


def test_dcgm_sampler_routes_prof_sampled_to_profiling_watch(mod, monkeypatch):
    """Real bug regression-guard: PROF_SM_ACTIVE / PIPE_* / DRAM_ACTIVE
    MUST be watched via DcgmGroup.profiling.WatchFields (perfworks
    activation), not samples.WatchFields — otherwise they silently
    return 0 for the full run. Observed live with vLLM serving for
    30+ minutes and SM_ACTIVE pinned at 0 until this split landed."""
    triple, recorder = _build_fake_dcgm()
    monkeypatch.setattr(mod, "_import_dcgm", lambda: triple)

    mod.DcgmSampler([
        "DCGM_FI_DEV_GPU_UTIL",
        "DCGM_FI_DEV_FB_USED",
        "DCGM_FI_PROF_PCIE_RX_BYTES",
        "DCGM_FI_PROF_SM_ACTIVE",
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    ], update_freq_us=5_000_000)

    by_api = {api: (fg_name, freq, fids) for api, fg_name, freq, fids in recorder.calls}
    assert set(by_api) == {"samples", "profiling"}, (
        f"both APIs must be invoked when both classes of fields are "
        f"requested; got {set(by_api)}"
    )

    # samples.WatchFields gets DEV_* + PROF byte counters
    _, _, regular_fids = by_api["samples"]
    regular_names = {
        n for n in ("DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_FB_USED",
                    "DCGM_FI_PROF_PCIE_RX_BYTES")
        if getattr(triple[1], n) in regular_fids
    }
    assert regular_names == {
        "DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_FB_USED",
        "DCGM_FI_PROF_PCIE_RX_BYTES",
    }

    # profiling.WatchFields gets PROF sampled
    _, prof_freq, prof_fids = by_api["profiling"]
    prof_names = {
        n for n in ("DCGM_FI_PROF_SM_ACTIVE", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE")
        if getattr(triple[1], n) in prof_fids
    }
    assert prof_names == {"DCGM_FI_PROF_SM_ACTIVE", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"}

    # Profiling update freq is capped at 1s even when caller asks for 5s
    # — perfworks rounds long windows toward 0 on short bursts.
    assert prof_freq == 1_000_000, (
        f"prof updateFreq must be capped at 1s (1_000_000us), got {prof_freq}"
    )


def test_dcgm_sampler_skips_profiling_when_no_prof_sampled_requested(mod, monkeypatch):
    """Caller asked only for regular fields → don't touch profiling
    subsystem at all (leaves the perfworks lock free for dcgm-exporter
    or another monitoring process)."""
    triple, recorder = _build_fake_dcgm()
    monkeypatch.setattr(mod, "_import_dcgm", lambda: triple)

    mod.DcgmSampler([
        "DCGM_FI_DEV_GPU_UTIL",
        "DCGM_FI_PROF_PCIE_RX_BYTES",  # counter, not sampled
    ])

    apis = {api for api, *_ in recorder.calls}
    assert apis == {"samples"}, (
        f"profiling.WatchFields must not be called when no PROF sampled "
        f"fields requested; got {apis}"
    )


def test_dcgm_sampler_falls_back_to_samples_on_profiling_failure(mod, monkeypatch, capsys):
    """DCGM 3.0+ deprecated dcgmProfWatchFields. When the profiling API
    raises, fall back to samples.WatchFields for the PROF fields rather
    than crashing — modern DCGM's samples.WatchFields auto-activates
    profiling internally on 3.0+."""
    triple, recorder = _build_fake_dcgm()
    fake_ds, fake_df, fake_pd = triple

    # Replace DcgmGroup.profiling.WatchFields with a raising one.
    class RaisingFakeWatcher:
        def __init__(self, api_label):
            self.api_label = api_label
        def WatchFields(self, fg, updateFreq, maxKeepAge, maxKeepSamples):
            if self.api_label == "profiling":
                raise RuntimeError("dcgmProfWatchFields removed in DCGM 3.0+")
            recorder.calls.append((self.api_label, fg.name, updateFreq, tuple(fg.fieldIds)))

    class RaisingFakeGroup:
        def __init__(self, *a, **kw):
            self.samples = RaisingFakeWatcher("samples")
            self.profiling = RaisingFakeWatcher("profiling")

    fake_pd.DcgmGroup.side_effect = RaisingFakeGroup
    monkeypatch.setattr(mod, "_import_dcgm", lambda: triple)

    mod.DcgmSampler([
        "DCGM_FI_DEV_GPU_UTIL",
        "DCGM_FI_PROF_SM_ACTIVE",
    ])

    # PROF fields should now appear in a SECOND samples.WatchFields call
    samples_calls = [c for c in recorder.calls if c[0] == "samples"]
    assert len(samples_calls) == 2, (
        f"expected one regular + one fallback samples.WatchFields call; "
        f"got {samples_calls}"
    )
    fallback_fids = samples_calls[1][3]
    assert getattr(fake_df, "DCGM_FI_PROF_SM_ACTIVE") in fallback_fids

    # User must see the warning so they can debug if the fallback also yields 0
    err = capsys.readouterr().err
    assert "profiling.WatchFields failed" in err
    assert "RuntimeError" in err


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
    assert sample["host"]["n_cores"] == 8
    assert sample["host"]["mem_total_bytes"] == 8 * (1 << 30)


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
