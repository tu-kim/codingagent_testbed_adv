"""Tests for scripts/filter_hanging_tools.py.

Flags server-type / long-running / backgrounded bash tool calls in an
opencode run's trace.jsonl and emits an exclusion callID list. Key risks:
the command heuristics (background `&` vs `&&`/`2>&1`, small vs big
`sleep`, foreground vs `-d` docker), the ms->s duration conversion, the
is_hang classification, and the main() output/exclusion-set contract.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "filter_hanging_tools.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("filter_hanging_tools", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["filter_hanging_tools"] = module
    spec.loader.exec_module(module)
    return module


def _tool_part(call_id: str, command, status: str,
               start: float | None, end: float | None,
               *, exit_code: int = 0) -> dict:
    time_obj: dict = {}
    if start is not None:
        time_obj["start"] = start
    if end is not None:
        time_obj["end"] = end
    inp: dict = {"description": "x"}
    if command is not None:
        inp["command"] = command
    return {
        "type": "tool",
        "tool": "bash",
        "callID": call_id,
        "state": {
            "status": status,
            "input": inp,
            "metadata": {"exit": exit_code},
            "time": time_obj,
        },
    }


def _record(instance_id: str, parts: list[dict], session_id: str = "ses1") -> dict:
    return {
        "instance_id": instance_id,
        "session_id": session_id,
        "messages": [{"info": {"id": "m1"}, "parts": parts}],
    }


def _write_trace(run: Path, records: list[dict]) -> None:
    (run / "trace.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


# ---------- classify_command: server ----------


@pytest.mark.parametrize("cmd", [
    "python3 -m http.server 8080",
    "python -m http.server",
    "python2 -m SimpleHTTPServer 8000",
    "uvicorn app:main --port 9000",
    "flask run --host 0.0.0.0",
    "gunicorn wsgi:app",
    "php -S localhost:8000",
    "nc -l 4444",
    "ncat -lk 5555",
    "npm run dev",
    "yarn start",
    "pnpm serve",
    "python manage.py runserver",
    "serve ./dist",
    "http-server -p 8080",
])
def test_classify_server(mod, cmd):
    assert "server" in mod.classify_command(cmd)


def test_classify_non_server_not_flagged_as_server(mod):
    assert "server" not in mod.classify_command("python3 script.py")
    assert "server" not in mod.classify_command("grep -rn foo .")


# ---------- classify_command: background ----------


@pytest.mark.parametrize("cmd", [
    "./daemon &",
    "python3 -m http.server 8080 > /dev/null 2>&1 &",
    "nohup ./x & echo done",
    "cmd_a & cmd_b",
])
def test_classify_background_true(mod, cmd):
    assert "background" in mod.classify_command(cmd)


@pytest.mark.parametrize("cmd", [
    "make && ./run",
    "echo hi 2>&1",
    "cat x &> out.log",
    "cmd >& out.log",
    "a && b && c",
])
def test_classify_background_false(mod, cmd):
    assert "background" not in mod.classify_command(cmd)


# ---------- classify_command: longrun ----------


@pytest.mark.parametrize("cmd", [
    "tail -f app.log",
    "tail -F app.log",
    "sleep 60",
    "sleep 300",
    "while true; do echo x; done",
    "docker run ubuntu bash",
    "watch ls",
    "journalctl -f",
])
def test_classify_longrun_true(mod, cmd):
    assert "longrun" in mod.classify_command(cmd)


@pytest.mark.parametrize("cmd", [
    "sleep 5",
    "sleep 2",
    "docker run -d nginx",
    "docker run --detach nginx",
    "tail -n 20 app.log",
])
def test_classify_longrun_false(mod, cmd):
    assert "longrun" not in mod.classify_command(cmd)


def test_classify_none(mod):
    assert mod.classify_command("ls -la") == []


# ---------- _duration_s ----------


def test_duration_ms_to_s(mod):
    assert mod._duration_s({"start": 1000, "end": 61000}) == pytest.approx(60.0)


def test_duration_missing_end_is_none(mod):
    assert mod._duration_s({"start": 1000}) is None


def test_duration_non_numeric_is_none(mod):
    assert mod._duration_s({"start": "a", "end": 5}) is None
    assert mod._duration_s(None) is None


def test_duration_negative_is_none(mod):
    assert mod._duration_s({"start": 5000, "end": 1000}) is None


# ---------- flag_calls ----------


def test_flag_calls_only_bash_with_string_command(mod):
    recs = [_record("i1", [
        _tool_part("c_nocmd", None, "completed", 0, 1000),
        {"type": "text", "text": "hi"},
    ])]
    assert mod.flag_calls(recs, 30.0) == []


def test_flag_calls_unflagged_omitted(mod):
    recs = [_record("i1", [_tool_part("c1", "grep -rn foo .", "completed", 0, 500)])]
    assert mod.flag_calls(recs, 30.0) == []


def test_flag_calls_server_hung(mod):
    recs = [_record("tb", [_tool_part(
        "c1", "cd ./webroot && python3 -m http.server 8080 > /dev/null 2>&1 &",
        "error", 1000, 301000)])]
    out = mod.flag_calls(recs, 30.0)
    assert len(out) == 1
    row = out[0]
    assert set(row["reasons"]) == {"server", "background", "duration"}
    assert row["is_hang"] is True
    assert row["duration_s"] == pytest.approx(300.0)


def test_flag_calls_running_no_end_is_hang(mod):
    recs = [_record("tb", [_tool_part("c3", "uvicorn app:main", "running", 5000, None)])]
    out = mod.flag_calls(recs, 30.0)
    assert len(out) == 1
    assert out[0]["reasons"] == ["server"]
    assert out[0]["is_hang"] is True
    assert out[0]["duration_s"] is None


def test_flag_calls_short_background_completed_not_hang(mod):
    recs = [_record("tb", [_tool_part("c4", "nohup ./daemon &", "completed", 6000, 6300)])]
    out = mod.flag_calls(recs, 30.0)
    assert len(out) == 1
    assert out[0]["reasons"] == ["background"]
    assert out[0]["is_hang"] is False


def test_flag_calls_duration_reason_added(mod):
    recs = [_record("tb", [_tool_part("c5", "make -j8", "completed", 7000, 52000)])]
    out = mod.flag_calls(recs, 30.0)
    assert out[0]["reasons"] == ["duration"]
    assert out[0]["is_hang"] is True


def test_flag_calls_min_duration_boundary(mod):
    # exactly at threshold -> duration reason + hang
    recs = [_record("tb", [_tool_part("c6", "make", "completed", 0, 30000)])]
    out = mod.flag_calls(recs, 30.0)
    assert out and out[0]["reasons"] == ["duration"]
    # just under -> not flagged (no other reason)
    recs2 = [_record("tb", [_tool_part("c7", "make", "completed", 0, 29000)])]
    assert mod.flag_calls(recs2, 30.0) == []


# ---------- total_bash_wall_s ----------


def test_total_bash_wall(mod):
    recs = [
        _record("i1", [_tool_part("c1", "make", "completed", 0, 10000)]),      # 10s
        _record("i1", [_tool_part("c2", "grep x", "completed", 0, 2000)]),     # 2s
        _record("i1", [_tool_part("c3", "run", "running", 0, None)]),          # no dur
    ]
    assert mod.total_bash_wall_s(recs) == pytest.approx(12.0)


# ---------- main() ----------


def _run_records():
    return [
        _record("tb-web", [_tool_part(
            "c1", "cd ./webroot && python3 -m http.server 8080 > /dev/null 2>&1 &",
            "error", 1000, 301000)]),
        _record("tb-web", [_tool_part("c2", "grep -rn foo .", "completed", 302000, 302500)]),
        _record("tb-api", [_tool_part("c3", "uvicorn app:main", "running", 5000, None)]),
        _record("tb-api", [_tool_part("c4", "nohup ./daemon &", "completed", 6000, 6300)]),
        _record("tb-build", [_tool_part("c5", "make -j8", "completed", 7000, 52000)]),
    ]


def test_main_writes_outputs_hangs_only(mod, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_trace(run, _run_records())
    rc = mod.main(["--run", str(run)])
    assert rc == 0
    out = run / "hanging_tools"
    flagged = [json.loads(l) for l in (out / "flagged_tools.jsonl").read_text().splitlines()]
    # c1, c3, c4, c5 flagged; c2 (grep) not
    assert {f["call_id"] for f in flagged} == {"c1", "c3", "c4", "c5"}
    excl = (out / "exclude_calls.txt").read_text().split()
    # default = hangs only: c4 (short background, completed) excluded from set
    assert excl == ["c1", "c3", "c5"]


def test_main_all_flagged(mod, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_trace(run, _run_records())
    rc = mod.main(["--run", str(run), "--all-flagged"])
    assert rc == 0
    excl = (run / "hanging_tools" / "exclude_calls.txt").read_text().split()
    assert excl == ["c1", "c3", "c4", "c5"]   # c4 now included


def test_main_dedups_call_ids_preserving_order(mod, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    # same callID hang appears twice (list_messages can repeat a part)
    rec = _record("tb", [
        _tool_part("c_dup", "python3 -m http.server 8080 &", "error", 0, 300000),
        _tool_part("c_dup", "python3 -m http.server 8080 &", "error", 0, 300000),
    ])
    _write_trace(run, [rec])
    assert mod.main(["--run", str(run)]) == 0
    excl = (run / "hanging_tools" / "exclude_calls.txt").read_text().split()
    assert excl == ["c_dup"]


def test_main_custom_out_dir(mod, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_trace(run, _run_records())
    dest = tmp_path / "elsewhere"
    assert mod.main(["--run", str(run), "--out", str(dest)]) == 0
    assert (dest / "flagged_tools.jsonl").is_file()
    assert (dest / "exclude_calls.txt").is_file()


def test_main_missing_trace_returns_2(mod, tmp_path, capsys):
    rc = mod.main(["--run", str(tmp_path / "nope")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


def test_main_min_duration_flag(mod, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    # a 20s make: flagged only when threshold <= 20
    _write_trace(run, [_record("i", [_tool_part("c1", "make", "completed", 0, 20000)])])
    assert mod.main(["--run", str(run), "--min-duration-s", "10"]) == 0
    excl = (run / "hanging_tools" / "exclude_calls.txt").read_text().split()
    assert excl == ["c1"]
    assert mod.main(["--run", str(run), "--min-duration-s", "30"]) == 0
    excl2 = (run / "hanging_tools" / "exclude_calls.txt").read_text().split()
    assert excl2 == []
