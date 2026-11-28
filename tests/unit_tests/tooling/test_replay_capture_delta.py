# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`scripts/replay_capture_delta.py` converts a capture to the delta format, verifies it, and times merges."""

import base64
import importlib.util
from pathlib import Path

import orjson
import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "replay_capture_delta.py"
_spec = importlib.util.spec_from_file_location("replay_capture_delta", SCRIPT)
replay_capture_delta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay_capture_delta)


def _full_capture(path: Path, calls: int) -> list[bytes]:
    image = "data:image/png;base64," + base64.b64encode(b"\x89PNG" * 2000).decode()
    history: list = [{"role": "user", "content": "task"}]
    lines = []
    for index in range(calls):
        history = history + [
            {
                "type": "function_call_output",
                "call_id": f"t{index}",
                "output": [{"type": "input_image", "image_url": image}],
            }
        ]
        exchange = {
            "model_call_id": f"c{index}",
            "dialect": "responses",
            "request": {"model": "m", "instructions": "x" * 500, "input": history},
            "response": {"id": f"r{index}", "output": []},
        }
        lines.append(orjson.dumps(exchange))
    path.write_bytes(b"\n".join(lines[:3]) + b"\n{damaged\n" + b"\n".join(lines[3:]) + b"\n")
    return lines


def test_convert_verifies_every_exchange_and_leaves_the_source_alone(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source" / "4-0.capture.jsonl"
    source.parent.mkdir()
    _full_capture(source, 12)
    before = source.read_bytes()

    assert replay_capture_delta.main(["convert", str(source), "--out-dir", str(tmp_path / "delta")]) == 0

    report = orjson.loads(capsys.readouterr().out)
    assert source.read_bytes() == before
    assert report["lines"] == 13 and report["unreadable_source_lines"] == 1
    assert report["verified_equal"] == report["verified_byte_identical"] == 12
    assert report["mismatch_count"] == 0
    assert report["delta_blob_files"] == 1
    assert report["delta_total_bytes"] * 5 < report["full_bytes"]


def test_convert_refuses_a_non_capture_file_and_a_used_output(tmp_path: Path) -> None:
    source = tmp_path / "source" / "4-0.capture.jsonl"
    source.parent.mkdir()
    _full_capture(source, 2)
    other = tmp_path / "source" / "notes.jsonl"
    other.write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="is not a"):
        replay_capture_delta.convert(other, tmp_path / "delta")
    replay_capture_delta.convert(source, tmp_path / "delta")
    with pytest.raises(FileExistsError, match="empty directory"):
        replay_capture_delta.convert(source, tmp_path / "delta")


def test_convert_refuses_a_capture_already_in_the_delta_format(tmp_path: Path) -> None:
    source = tmp_path / "source" / "4-0.capture.jsonl"
    source.parent.mkdir()
    _full_capture(source, 3)
    replay_capture_delta.convert(source, tmp_path / "delta")

    with pytest.raises(ValueError, match="already in the delta format"):
        replay_capture_delta.convert(tmp_path / "delta" / "4-0.capture.jsonl", tmp_path / "again")


def test_merge_reports_the_merged_calls(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source" / "4-0.capture.jsonl"
    source.parent.mkdir()
    _full_capture(source, 5)

    assert replay_capture_delta.main(["merge", str(source.parent), "4-0", "--payloads", "reference"]) == 0

    report = orjson.loads(capsys.readouterr().out)
    assert report["calls"] == 5 and report["gaps"] == [
        {"code": "model_call_capture_records_unreadable", "detail": "count=1"}
    ]
