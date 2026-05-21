"""Tests for scripts/analyze_session_resources.py.

Pure file-based parsing + window filtering + stats. No external
dependencies beyond numpy / json / csv. Optional testbed.yaml is
written under tmp_path; PyYAML is required only when that arg is
used (gracefully degrades otherwise).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_session_resources.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_session_resources", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_session_resources"] = module
    spec.loader.exec_module(module)
    return module


def _write_profile(path: Path, *, session_id: str = "ses_001",
                    start: float = 100.0, end: float = 200.0) -> None:
    lines = [
        {"ev": "query.start", "ts": start, "sessionID": session_id, "directory": "/x"},
        {"ev": "turn.start",  "ts": start + 1, "sessionID": session_id, "step": 1},
        {"ev": "turn.end",    "ts": end - 1,   "sessionID": session_id, "step": 1,
         "duration_s": end - start - 2, "llm_wall_s": 1.0, "tool_wall_s": 0.5,
         "post_overhead_s": 0.5},
        {"ev": "query.end",   "ts": end,   "sessionID": session_id,
         "duration_s": end - start, "steps": 1, "aborted": False},
    ]
    path.write_text("\n".join(json.dumps(l) for l in lines) + "\n")


def _write_resources(path: Path, samples: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(s) for s in samples) + "\n")


# ---------- window detection ----------


def test_load_session_windows_picks_query_start_end(mod, tmp_path):
    p = tmp_path / "ses.jsonl"
    _write_profile(p, session_id="ses_abc", start=10.0, end=50.0)
    w = mod.load_session_windows(p)
    assert w == {"ses_abc": (10.0, 50.0)}


def test_load_session_windows_falls_back_to_last_ts_when_query_end_missing(mod, tmp_path):
    """A crashed / mid-flight run won't have query.end. The window
    should fall back to the latest ts seen for that session so the
    analyzer still has something usable."""
    p = tmp_path / "ses.jsonl"
    lines = [
        {"ev": "query.start", "ts": 10.0, "sessionID": "ses_x"},
        {"ev": "turn.start",  "ts": 12.0, "sessionID": "ses_x", "step": 1},
        {"ev": "llm.start",   "ts": 13.0, "sessionID": "ses_x", "step": 1},
        # no query.end
    ]
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    w = mod.load_session_windows(p)
    assert w == {"ses_x": (10.0, 13.0)}


# ---------- window filtering ----------


def test_load_samples_skips_broken_json_lines(mod, tmp_path):
    """monitor_resources writes via appendFileSync which CAN produce
    a partial last line if the process is SIGKILL'd. The analyzer
    must skip malformed lines rather than crash."""
    p = tmp_path / "res.ndjson"
    p.write_text(
        json.dumps({"ts": 110.0, "host": {"cpu_util_pct": 25}}) + "\n"
        + "this is not json\n"                                                 # malformed
        + '{"ts": 120.0, "host": {"cpu_util_pct": 35}\n'                       # missing closing brace
        + json.dumps({"ts": 130.0, "host": {"cpu_util_pct": 45}}) + "\n"
    )
    samples = mod.load_samples_in_window(p, 100.0, 200.0)
    cpus = [s["host"]["cpu_util_pct"] for s in samples]
    assert cpus == [25, 45]   # two malformed lines skipped silently


def test_load_samples_filters_to_window(mod, tmp_path):
    p = tmp_path / "res.ndjson"
    _write_resources(p, [
        {"ts":  50.0, "host": {"cpu_util_pct": 5}},
        {"ts": 105.0, "host": {"cpu_util_pct": 30}},
        {"ts": 150.0, "host": {"cpu_util_pct": 45}},
        {"ts": 195.0, "host": {"cpu_util_pct": 60}},
        {"ts": 250.0, "host": {"cpu_util_pct": 10}},   # outside window
    ])
    samples = mod.load_samples_in_window(p, 100.0, 200.0)
    cpus = [s["host"]["cpu_util_pct"] for s in samples]
    assert cpus == [30, 45, 60]   # 50 and 250 outside [100,200]


# ---------- testbed.yaml role map ----------


def test_parse_gpu_role_map_from_yaml(mod, tmp_path):
    yaml_path = tmp_path / "testbed.yaml"
    yaml_path.write_text(
        "vllm:\n"
        "  prefill_workers:\n"
        '    - { name: p0, host: 127.0.0.1, gpus: "0,1", tp: 2, pp: 1 }\n'
        "  decode_workers:\n"
        '    - { name: d0, host: 127.0.0.1, gpus: "2,3", tp: 2, pp: 1 }\n'
    )
    pytest.importorskip("yaml")   # PyYAML required for this test
    mp = mod.parse_gpu_role_map(yaml_path)
    assert mp == {
        0: ("p0", "prefill"),
        1: ("p0", "prefill"),
        2: ("d0", "decode"),
        3: ("d0", "decode"),
    }


def test_parse_gpu_role_map_returns_empty_when_missing(mod, tmp_path):
    assert mod.parse_gpu_role_map(None) == {}
    assert mod.parse_gpu_role_map(tmp_path / "does-not-exist.yaml") == {}


def test_parse_gpu_role_map_handles_empty_vllm_section(mod, tmp_path):
    """testbed.yaml may exist but have no vllm.{prefill,decode}_workers
    keys (e.g. a partial config or a model-only yaml). Must return {}
    without KeyError."""
    pytest.importorskip("yaml")
    cases = [
        "vllm: {}\n",                       # vllm exists but empty
        "model: { name: x, served_name: y }\n",   # no vllm key at all
        "vllm:\n  prefill_workers: []\n",   # empty list explicitly
        "vllm:\n  decode_workers:\n",       # null value
    ]
    for i, content in enumerate(cases):
        p = tmp_path / f"empty_{i}.yaml"
        p.write_text(content)
        assert mod.parse_gpu_role_map(p) == {}, f"case {i}: {content!r}"


# ---------- metric extraction ----------


def test_extract_metrics_collects_host_gpus_processes(mod):
    samples = [
        {
            "ts": 1.0,
            "host": {"cpu_util_pct": 30.0, "mem_used_bytes": 16 * (1 << 30),
                     "mem_available_bytes": 8 * (1 << 30)},
            "gpus": [
                {"index": 0, "DCGM_FI_PROF_SM_ACTIVE": 0.5,
                 "DCGM_FI_DEV_FB_USED": 40000},
                {"index": 2, "DCGM_FI_PROF_SM_ACTIVE": 0.8,
                 "DCGM_FI_DEV_FB_USED": 42000},
            ],
            "processes": [
                {"name": "vllm-d0", "pid": 1, "cpu_util_pct": 220.0,
                 "rss_bytes": 18 * (1 << 30), "n_threads": 8},
            ],
        },
        {
            "ts": 2.0,
            "host": {"cpu_util_pct": 40.0, "mem_used_bytes": 17 * (1 << 30),
                     "mem_available_bytes": 7 * (1 << 30)},
            "gpus": [
                {"index": 0, "DCGM_FI_PROF_SM_ACTIVE": 0.6,
                 "DCGM_FI_DEV_FB_USED": 41000},
                {"index": 2, "DCGM_FI_PROF_SM_ACTIVE": 0.9,
                 "DCGM_FI_DEV_FB_USED": 43000},
            ],
            "processes": [
                {"name": "vllm-d0", "pid": 1, "cpu_util_pct": 250.0,
                 "rss_bytes": 19 * (1 << 30), "n_threads": 8},
            ],
        },
    ]
    metrics = mod.extract_metrics(samples)
    assert metrics["host.cpu_util_pct"] == [30.0, 40.0]
    assert metrics["host.mem_used_gib"] == [16.0, 17.0]
    assert metrics["gpu0.DCGM_FI_PROF_SM_ACTIVE"] == [0.5, 0.6]
    assert metrics["gpu2.DCGM_FI_PROF_SM_ACTIVE"] == [0.8, 0.9]
    assert metrics["gpu0.DCGM_FI_DEV_FB_USED"] == [40000, 41000]
    assert metrics["process.vllm-d0.cpu_util_pct"] == [220.0, 250.0]
    assert metrics["process.vllm-d0.rss_gib"] == [18.0, 19.0]


def test_extract_metrics_role_aggregate_when_role_map_present(mod):
    """When testbed.yaml maps GPUs to roles, emit per-role aggregate
    rows that average across GPUs sharing a role (per-sample)."""
    samples = [
        {
            "ts": 1.0,
            "gpus": [
                {"index": 0, "DCGM_FI_PROF_SM_ACTIVE": 0.4},   # prefill
                {"index": 1, "DCGM_FI_PROF_SM_ACTIVE": 0.6},   # prefill
                {"index": 2, "DCGM_FI_PROF_SM_ACTIVE": 0.8},   # decode
                {"index": 3, "DCGM_FI_PROF_SM_ACTIVE": 1.0},   # decode
            ],
        },
    ]
    gpu_role = {
        0: ("p0", "prefill"), 1: ("p0", "prefill"),
        2: ("d0", "decode"),  3: ("d0", "decode"),
    }
    metrics = mod.extract_metrics(samples, gpu_role=gpu_role)
    # per-GPU rows include the role suffix
    assert "gpu0[p0/prefill].DCGM_FI_PROF_SM_ACTIVE" in metrics
    assert "gpu2[d0/decode].DCGM_FI_PROF_SM_ACTIVE" in metrics
    # role aggregate = mean across GPUs in that role
    assert metrics["prefill.DCGM_FI_PROF_SM_ACTIVE"] == [0.5]   # mean(0.4, 0.6)
    assert metrics["decode.DCGM_FI_PROF_SM_ACTIVE"] == [0.9]    # mean(0.8, 1.0)


def test_extract_metrics_role_aggregate_across_multiple_samples(mod):
    """Per-sample mean → one entry per sample in the role aggregate
    list. p90/p99 at the role level should reflect the SAMPLE-level
    distribution, not a flattened pool of (n_gpus × n_samples)."""
    samples = [
        {"ts": 1.0, "gpus": [
            {"index": 0, "DCGM_FI_PROF_SM_ACTIVE": 0.4},
            {"index": 1, "DCGM_FI_PROF_SM_ACTIVE": 0.6},   # prefill mean = 0.5
        ]},
        {"ts": 2.0, "gpus": [
            {"index": 0, "DCGM_FI_PROF_SM_ACTIVE": 0.6},
            {"index": 1, "DCGM_FI_PROF_SM_ACTIVE": 0.8},   # prefill mean = 0.7
        ]},
        {"ts": 3.0, "gpus": [
            {"index": 0, "DCGM_FI_PROF_SM_ACTIVE": 0.8},
            {"index": 1, "DCGM_FI_PROF_SM_ACTIVE": 1.0},   # prefill mean = 0.9
        ]},
    ]
    gpu_role = {0: ("p0", "prefill"), 1: ("p0", "prefill")}
    metrics = mod.extract_metrics(samples, gpu_role=gpu_role)
    assert metrics["prefill.DCGM_FI_PROF_SM_ACTIVE"] == [0.5, 0.7, 0.9]
    # per-GPU lists still have one value per sample
    assert metrics["gpu0[p0/prefill].DCGM_FI_PROF_SM_ACTIVE"] == [0.4, 0.6, 0.8]
    assert metrics["gpu1[p0/prefill].DCGM_FI_PROF_SM_ACTIVE"] == [0.6, 0.8, 1.0]


def test_extract_metrics_handles_empty_processes_list(mod):
    """Sample without `processes` (e.g. CPU sampler off) must not emit
    spurious `process.*` keys."""
    samples = [{"ts": 1.0, "host": {"cpu_util_pct": 20.0}, "gpus": [], "processes": []}]
    metrics = mod.extract_metrics(samples)
    assert metrics["host.cpu_util_pct"] == [20.0]
    assert not any(k.startswith("process.") for k in metrics)
    assert not any(k.startswith("gpu") for k in metrics)


def test_extract_metrics_skips_non_numeric_fields(mod):
    """DCGM bytes/string fields shouldn't enter the numeric stats pool."""
    samples = [{
        "ts": 1.0,
        "gpus": [{"index": 0,
                  "DCGM_FI_DEV_DRIVER_VERSION": "535.86",
                  "DCGM_FI_PROF_SM_ACTIVE": 0.5}],
    }]
    metrics = mod.extract_metrics(samples)
    assert metrics["gpu0.DCGM_FI_PROF_SM_ACTIVE"] == [0.5]
    assert "gpu0.DCGM_FI_DEV_DRIVER_VERSION" not in metrics


# ---------- stats ----------


def test_stats_empty_returns_none_n_zero(mod):
    s = mod.stats([])
    assert s == {"n": 0, "mean": None, "median": None,
                  "p90": None, "p99": None, "max": None}


def test_stats_basic_correctness(mod):
    s = mod.stats([10.0, 20.0, 30.0, 40.0, 50.0])
    assert s["n"] == 5
    assert s["mean"] == pytest.approx(30.0)
    assert s["median"] == pytest.approx(30.0)
    assert s["max"] == pytest.approx(50.0)


# ---------- CSV ----------


def test_write_csv_columns_and_rows(mod, tmp_path):
    stats_per_metric = {
        "host.cpu_util_pct": {"n": 3, "mean": 35.0, "median": 33.0,
                               "p90": 50.0, "p99": 58.0, "max": 60.0},
    }
    csv_path = tmp_path / "out.csv"
    mod.write_csv(stats_per_metric, "ses_x", (100.0, 200.0), csv_path)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["session_id"] == "ses_x"
    assert row["window_duration_s"] == "100.000"
    assert row["metric"] == "host.cpu_util_pct"
    assert row["mean"] == "35.0000"
    assert row["p99"] == "58.0000"


# ---------- main end-to-end ----------


def test_main_end_to_end_writes_csv_and_prints_table(mod, tmp_path, capsys):
    profile = tmp_path / "ses.jsonl"
    resource = tmp_path / "res.ndjson"
    out = tmp_path / "figs"
    _write_profile(profile, session_id="ses_x", start=10.0, end=20.0)
    _write_resources(resource, [
        {"ts":  5.0, "host": {"cpu_util_pct": 5}},   # outside
        {"ts": 12.0, "host": {"cpu_util_pct": 30}},
        {"ts": 16.0, "host": {"cpu_util_pct": 45}},
        {"ts": 25.0, "host": {"cpu_util_pct": 5}},   # outside
    ])
    rc = mod.main(["--profile", str(profile),
                    "--resource", str(resource),
                    "--output", str(out)])
    assert rc == 0
    csv_path = out / "session_resources_stats.csv"
    assert csv_path.exists()
    text = csv_path.read_text()
    assert "host.cpu_util_pct" in text
    # mean of (30, 45) = 37.5
    captured = capsys.readouterr().out
    assert "Session: ses_x" in captured
    assert "host.cpu_util_pct" in captured


def test_main_returns_nonzero_when_no_samples_in_window(mod, tmp_path, capsys):
    profile = tmp_path / "ses.jsonl"
    resource = tmp_path / "res.ndjson"
    _write_profile(profile, session_id="ses_x", start=10.0, end=20.0)
    _write_resources(resource, [
        {"ts":  5.0, "host": {"cpu_util_pct": 5}},   # all outside the window
        {"ts": 25.0, "host": {"cpu_util_pct": 6}},
    ])
    rc = mod.main(["--profile", str(profile),
                    "--resource", str(resource),
                    "--output", str(tmp_path / "out")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no resource samples inside the window" in err


def test_main_with_testbed_yaml_emits_role_labels(mod, tmp_path, capsys):
    """End-to-end wire-up of --testbed-yaml: per-GPU rows get the
    [<worker>/<role>] suffix and role-aggregate rows (prefill.*,
    decode.*) appear in the CSV + stdout table."""
    pytest.importorskip("yaml")
    profile = tmp_path / "ses.jsonl"
    _write_profile(profile, session_id="ses_x", start=10.0, end=30.0)
    resource = tmp_path / "res.ndjson"
    _write_resources(resource, [
        {"ts": 15.0, "gpus": [
            {"index": 0, "DCGM_FI_PROF_SM_ACTIVE": 0.4},
            {"index": 1, "DCGM_FI_PROF_SM_ACTIVE": 0.6},
            {"index": 2, "DCGM_FI_PROF_SM_ACTIVE": 0.8},
            {"index": 3, "DCGM_FI_PROF_SM_ACTIVE": 1.0},
        ]},
        {"ts": 25.0, "gpus": [
            {"index": 0, "DCGM_FI_PROF_SM_ACTIVE": 0.5},
            {"index": 1, "DCGM_FI_PROF_SM_ACTIVE": 0.7},
            {"index": 2, "DCGM_FI_PROF_SM_ACTIVE": 0.9},
            {"index": 3, "DCGM_FI_PROF_SM_ACTIVE": 0.9},
        ]},
    ])
    yaml_path = tmp_path / "testbed.yaml"
    yaml_path.write_text(
        "vllm:\n"
        "  prefill_workers:\n"
        '    - { name: p0, host: 127.0.0.1, gpus: "0,1", tp: 2, pp: 1 }\n'
        "  decode_workers:\n"
        '    - { name: d0, host: 127.0.0.1, gpus: "2,3", tp: 2, pp: 1 }\n'
    )
    out = tmp_path / "out"
    rc = mod.main(["--profile", str(profile),
                    "--resource", str(resource),
                    "--output", str(out),
                    "--testbed-yaml", str(yaml_path)])
    assert rc == 0
    csv_path = out / "session_resources_stats.csv"
    text = csv_path.read_text()
    # role-labeled per-GPU rows
    assert "gpu0[p0/prefill].DCGM_FI_PROF_SM_ACTIVE" in text
    assert "gpu2[d0/decode].DCGM_FI_PROF_SM_ACTIVE" in text
    # role-aggregate rows
    assert "prefill.DCGM_FI_PROF_SM_ACTIVE" in text
    assert "decode.DCGM_FI_PROF_SM_ACTIVE" in text
    # stdout also surfaces them
    out_text = capsys.readouterr().out
    assert "prefill.DCGM_FI_PROF_SM_ACTIVE" in out_text


def test_main_with_multiple_sessions_picks_first_or_explicit(mod, tmp_path, capsys):
    profile = tmp_path / "ses.jsonl"
    # Two sessions: ses_a first, then ses_b
    lines = []
    for sid, start, end in [("ses_a", 10.0, 20.0), ("ses_b", 30.0, 40.0)]:
        lines += [
            {"ev": "query.start", "ts": start, "sessionID": sid},
            {"ev": "query.end",   "ts": end,   "sessionID": sid,
             "duration_s": end - start, "steps": 1, "aborted": False},
        ]
    profile.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    resource = tmp_path / "res.ndjson"
    _write_resources(resource, [
        {"ts": 15.0, "host": {"cpu_util_pct": 11}},
        {"ts": 35.0, "host": {"cpu_util_pct": 22}},
    ])

    # default: picks first
    rc = mod.main(["--profile", str(profile),
                    "--resource", str(resource),
                    "--output", str(tmp_path / "out1")])
    assert rc == 0
    out1 = (tmp_path / "out1" / "session_resources_stats.csv").read_text()
    assert "ses_a" in out1

    # explicit: picks ses_b
    rc = mod.main(["--profile", str(profile),
                    "--resource", str(resource),
                    "--output", str(tmp_path / "out2"),
                    "--session-id", "ses_b"])
    assert rc == 0
    out2 = (tmp_path / "out2" / "session_resources_stats.csv").read_text()
    assert "ses_b" in out2
