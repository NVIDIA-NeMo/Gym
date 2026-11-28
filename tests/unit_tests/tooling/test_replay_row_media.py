# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`scripts/replay_row_media.py` references one row's images, restores the row and verifies it byte for byte."""

import base64
import importlib.util
from pathlib import Path

import orjson

from nemo_gym.models.model_call_capture import CaptureStore
from tests.unit_tests.rollouts.harness_capabilities.synthetic import evidence_record


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "replay_row_media.py"
_spec = importlib.util.spec_from_file_location("replay_row_media", SCRIPT)
replay_row_media = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay_row_media)

IMAGE = "data:image/png;base64," + base64.b64encode(b"\x89PNG" * 2000).decode()


def _rows(path: Path) -> list[bytes]:
    record = evidence_record()
    record["_ng_rollout_id"] = "0-0"
    image_output = [{"type": "input_image", "image_url": IMAGE, "detail": "auto"}]
    record["ng_agent_observations"]["records"][0]["conversation"][2]["output"] = image_output
    record["ng_trajectory"]["invocations"][0]["conversation"][2]["output"] = image_output
    record["ng_trajectory"]["tool_calls"][0]["output"] = image_output
    lines = [orjson.dumps({"_ng_rollout_id": "9-0", "reward": 0.0}), orjson.dumps(record)]
    path.write_bytes(b"".join(line + b"\n" for line in lines))
    return lines


def test_convert_verifies_a_row_and_reports_what_it_stored(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    lines = _rows(source)
    capture_dir = tmp_path / "model-calls"
    request = {"model": "m", "input": [{"type": "input_image", "image_url": IMAGE}]}
    CaptureStore(capture_dir, capture_format="delta").record(
        "0-0", {"model_call_id": "c1", "dialect": "responses", "request": request, "response": {}}
    )

    report = replay_row_media.convert(source, tmp_path / "out", line=1, capture_dir=capture_dir)

    assert report["restored_identical"] and report["orjson_round_trip_identical"]
    assert report["original_row_bytes"] == len(lines[1])
    assert (report["image_references"], report["unique_images"], report["blobs_written"]) == (3, 1, 1)
    assert report["referenced_row_bytes"] < report["original_row_bytes"] - 2 * len(IMAGE)
    assert report["blob_dir_bytes"] == len(IMAGE)
    assert report["capture_blobs_already_held"]["held_by_capture"] == 1
    assert report["capture_blobs_already_held"]["bytes_a_capture_root_row_would_add"] == 0
    assert set(report["phases"]) == {"read_and_parse", "reference", "serialize_referenced", "restore", "verify"}
    [written] = (tmp_path / "out" / "rollouts.jsonl").read_bytes().splitlines()
    assert len(written) == report["referenced_row_bytes"]


def test_convert_reads_a_row_by_offset(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    lines = _rows(source)

    report = replay_row_media.convert(source, tmp_path / "out", offset=len(lines[0]) + 1, length=len(lines[1]) + 1)

    assert report["restored_identical"] and report["image_references"] == 3


def test_main_exits_zero_on_a_verified_row(tmp_path: Path, capsys) -> None:
    source = tmp_path / "rollouts.jsonl"
    _rows(source)

    assert replay_row_media.main(["convert", str(source), "--line", "0", "--out-dir", str(tmp_path / "out")]) == 0
    assert orjson.loads(capsys.readouterr().out)["image_references"] == 0
