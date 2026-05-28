"""Tests for scripts/jsonl_to_json.py.

Stream conversion (NDJSON -> JSON array). No external deps; covers
file/stdin input, file/stdout output, --pretty indenting, malformed-
line handling (default skip, --strict raise), empty input.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "jsonl_to_json.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("jsonl_to_json", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["jsonl_to_json"] = module
    spec.loader.exec_module(module)
    return module


def _write_ndjson(path: Path, objects: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(o) for o in objects) + "\n")


# ---------- iter_objects ----------


def test_iter_objects_yields_one_per_line(mod):
    buf = io.StringIO('{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    out = list(mod.iter_objects(buf))
    assert out == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_iter_objects_skips_blank_lines(mod):
    buf = io.StringIO('{"a": 1}\n\n\n{"a": 2}\n')
    out = list(mod.iter_objects(buf))
    assert out == [{"a": 1}, {"a": 2}]


def test_iter_objects_default_skips_malformed_with_warning(mod, capsys):
    buf = io.StringIO('{"a": 1}\nnot json\n{"a": 2}\n')
    out = list(mod.iter_objects(buf, strict=False))
    assert out == [{"a": 1}, {"a": 2}]
    err = capsys.readouterr().err
    assert "malformed line 2" in err


def test_iter_objects_strict_raises(mod):
    buf = io.StringIO('{"a": 1}\nnot json\n')
    with pytest.raises(json.JSONDecodeError):
        list(mod.iter_objects(buf, strict=True))


# ---------- convert ----------


def test_convert_compact_is_valid_json(mod):
    in_buf = io.StringIO('{"a": 1}\n{"b": 2}\n')
    out_buf = io.StringIO()
    n = mod.convert(in_buf, out_buf, pretty=False, strict=False)
    assert n == 2
    parsed = json.loads(out_buf.getvalue())
    assert parsed == [{"a": 1}, {"b": 2}]


def test_convert_pretty_is_valid_and_indented(mod):
    in_buf = io.StringIO('{"a": 1}\n{"b": 2}\n')
    out_buf = io.StringIO()
    n = mod.convert(in_buf, out_buf, pretty=True, strict=False)
    assert n == 2
    text = out_buf.getvalue()
    # round-trips
    parsed = json.loads(text)
    assert parsed == [{"a": 1}, {"b": 2}]
    # has indentation (at least one 2-space indent inside)
    assert "\n  " in text


def test_convert_empty_input_emits_empty_array(mod):
    out_buf = io.StringIO()
    n = mod.convert(io.StringIO(""), out_buf, pretty=False, strict=False)
    assert n == 0
    assert json.loads(out_buf.getvalue()) == []


def test_convert_preserves_non_ascii(mod):
    """ensure_ascii=False so Korean/etc. characters round-trip without
    \\uXXXX escaping (compactness + readability)."""
    in_buf = io.StringIO('{"msg": "안녕"}\n')
    out_buf = io.StringIO()
    mod.convert(in_buf, out_buf, pretty=False, strict=False)
    text = out_buf.getvalue()
    assert "안녕" in text
    assert json.loads(text) == [{"msg": "안녕"}]


# ---------- main: file -> file ----------


def test_main_file_input_to_file_output(mod, tmp_path):
    src = tmp_path / "in.ndjson"
    dst = tmp_path / "out.json"
    _write_ndjson(src, [{"i": 1}, {"i": 2}, {"i": 3}])
    rc = mod.main([str(src), "--output", str(dst)])
    assert rc == 0
    assert json.loads(dst.read_text()) == [{"i": 1}, {"i": 2}, {"i": 3}]


def test_main_pretty_flag(mod, tmp_path):
    src = tmp_path / "in.ndjson"
    dst = tmp_path / "out.json"
    _write_ndjson(src, [{"i": 1}, {"i": 2}])
    rc = mod.main([str(src), "--output", str(dst), "--pretty"])
    assert rc == 0
    text = dst.read_text()
    assert "\n  " in text   # indented
    assert json.loads(text) == [{"i": 1}, {"i": 2}]


def test_main_creates_output_parent_directory(mod, tmp_path):
    src = tmp_path / "in.ndjson"
    dst = tmp_path / "nested" / "deeper" / "out.json"
    _write_ndjson(src, [{"x": 1}])
    rc = mod.main([str(src), "--output", str(dst)])
    assert rc == 0
    assert dst.exists()


def test_main_missing_input_returns_2(mod, tmp_path, capsys):
    rc = mod.main([str(tmp_path / "does-not-exist.ndjson")])
    assert rc == 2
    assert "input not found" in capsys.readouterr().err


# ---------- main: stdin -> stdout ----------


def test_main_stdin_to_stdout(mod, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"a": 1}\n{"b": 2}\n'))
    rc = mod.main(["-"])
    assert rc == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == [{"a": 1}, {"b": 2}]


def test_main_default_argv_reads_stdin(mod, monkeypatch, capsys):
    """Omitting the positional arg should also read stdin (input defaults
    to '-')."""
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"k": "v"}\n'))
    rc = mod.main([])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == [{"k": "v"}]


# ---------- main: strict flag ----------


def test_main_strict_raises_on_malformed(mod, tmp_path):
    src = tmp_path / "bad.ndjson"
    src.write_text('{"ok": 1}\nnot json\n')
    with pytest.raises(json.JSONDecodeError):
        mod.main([str(src), "--strict"])


def test_main_default_skips_malformed_and_succeeds(mod, tmp_path, capsys):
    src = tmp_path / "bad.ndjson"
    dst = tmp_path / "out.json"
    src.write_text('{"ok": 1}\nnot json\n{"ok": 2}\n')
    rc = mod.main([str(src), "--output", str(dst)])
    assert rc == 0
    assert json.loads(dst.read_text()) == [{"ok": 1}, {"ok": 2}]
    # warning surfaced on stderr
    assert "malformed line" in capsys.readouterr().err
