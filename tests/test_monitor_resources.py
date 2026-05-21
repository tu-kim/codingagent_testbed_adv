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
    for v in (-1, -2, -3, -4, -5, -6, 0x7FFFFFFFFFFFFFFF):
        assert mod._is_dcgm_blank(v), f"{v} should be detected as blank"


def test_is_dcgm_blank_recognises_nan(mod):
    assert mod._is_dcgm_blank(float("nan"))


def test_is_dcgm_blank_accepts_normal_values(mod):
    for v in (0.0, 0, 1, 42.5, 100):
        assert not mod._is_dcgm_blank(v)


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
