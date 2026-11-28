# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rollout rows that store each large data URL once (``row_media: reference``), and the readers that need them."""

import base64
import copy
import json
from pathlib import Path
from typing import Any

import orjson
import pytest

from nemo_gym.artifacts.rollout_rows import _rollout_for_export
from nemo_gym.models.model_call_capture import CaptureStore
from nemo_gym.rollouts.harness_capabilities import cli as conformance_cli
from nemo_gym.rollouts.harness_capabilities.checker import inspect_record
from nemo_gym.trajectory.atif_export import AtifExportError, ExportAtifConfig, export_rollouts_to_atif
from nemo_gym.trajectory.row_media import (
    BLOB_MIN_CHARS,
    NG_ROW_MEDIA_KEY,
    RowMediaUnavailable,
    capture_media_location,
    reference_row_media,
    restore_row_media,
    rollouts_media_location,
    row_media_dirs,
    row_media_from_config,
)
from tests.unit_tests.rollouts.harness_capabilities.synthetic import evidence_record


def _image(seed: bytes, size: int = 3000) -> str:
    return "data:image/png;base64," + base64.b64encode(seed * size).decode()


SCREENSHOT = _image(b"\x89PNG")
OTHER = _image(b"\x89PNX")


def _image_output(url: str) -> list[dict[str, Any]]:
    return [{"type": "input_image", "image_url": url, "detail": "auto"}]


def _row(url: str = SCREENSHOT) -> dict[str, Any]:
    """A finished row holding one screenshot where a Codex row holds it: observations, conversation, tool output."""
    tool_result = {"type": "function_call_output", "call_id": "view-1", "output": _image_output(url)}
    return {
        "responses_create_params": {"input": [{"role": "user", "content": _image_output(OTHER)}]},
        "response": {"output": [tool_result]},
        "reward": 1.0,
        "_ng_task_index": 3,
        "_ng_rollout_index": 0,
        "_ng_rollout_id": "3-0",
        "ng_agent_observations": {"source": "codex", "records": [{"conversation": [tool_result]}]},
        "ng_trajectory": {
            "invocations": [{"conversation": [tool_result, {"role": "user", "content": _image_output(OTHER)}]}],
            "tool_calls": [{"output": _image_output(url)}, {"output": "short data:image/png;base64,AAAA"}],
            "model_calls": [{"request": {"input": [tool_result]}, "response": {"output": []}}],
        },
        "ng_perf": {"turns": 1},
    }


def _blobs(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if not path.name.startswith(".")) if directory.is_dir() else []


def test_a_referenced_row_restores_to_the_exact_bytes_it_was_written_as(tmp_path: Path) -> None:
    row = _row()
    original = orjson.dumps(row)
    location = rollouts_media_location(tmp_path / "rollouts.jsonl", "3-0")

    stats = reference_row_media(row, location)

    written = orjson.dumps(row)
    evidence = orjson.dumps({key: row[key] for key in ("ng_agent_observations", "ng_trajectory")})
    assert SCREENSHOT.encode() not in evidence and OTHER.encode() not in evidence
    # Seven copies of two images were written; the two the harness output and task input hold remain.
    assert len(written) < len(original) * 2 // 7 + 4096
    # The listing follows the payload, as the last key of the row.
    assert list(row)[-1] == NG_ROW_MEDIA_KEY
    assert row[NG_ROW_MEDIA_KEY]["root"] == "rollouts_dir"
    assert row[NG_ROW_MEDIA_KEY]["dir"] == "rollouts_blobs/3-0"
    assert stats.references == len(row[NG_ROW_MEDIA_KEY]["blobs"]) == 5
    restored = restore_row_media(orjson.loads(written), rollouts_dir=tmp_path)
    assert orjson.dumps(restored) == original


def test_each_image_is_stored_once_however_many_places_hold_it(tmp_path: Path) -> None:
    row = _row()
    location = rollouts_media_location(tmp_path / "rollouts.jsonl", "3-0")

    stats = reference_row_media(row, location)

    # The screenshot sits in observations, the invocation conversation, a tool call and a request; the user image in
    # the conversation. Two files, each holding its URL's text under its SHA-256.
    assert (stats.references, stats.unique, stats.written_blobs) == (5, 2, 2)
    files = _blobs(location.directory)
    assert sorted(path.read_text() for path in files) == sorted([SCREENSHOT, OTHER])
    places = [path for path, _ in row[NG_ROW_MEDIA_KEY]["blobs"]]
    assert ["ng_agent_observations", "records", 0, "conversation", 0, "output", 0, "image_url"] in places
    assert ["ng_trajectory", "tool_calls", 0, "output", 0, "image_url"] in places
    assert row["ng_trajectory"]["tool_calls"][0]["output"][0]["image_url"] is None


def test_the_harness_output_and_task_input_keep_their_images(tmp_path: Path) -> None:
    row = _row()
    before = copy.deepcopy({key: row[key] for key in ("responses_create_params", "response", "reward")})

    reference_row_media(row, rollouts_media_location(tmp_path / "rollouts.jsonl", "3-0"))

    assert {key: row[key] for key in before} == before
    assert all(path[0] not in before for path, _ in row[NG_ROW_MEDIA_KEY]["blobs"])


@pytest.mark.parametrize(
    ("value", "referenced"),
    [
        pytest.param(_image(b"x", 10), False, id="a data URL below the threshold stays inline"),
        pytest.param("data:image/png;base64," + "A" * (BLOB_MIN_CHARS - 22), True, id="one at the threshold goes"),
        pytest.param("see data:image/png;base64," + "A" * 4000, False, id="a string that only contains one stays"),
        pytest.param("A" * 5000, False, id="long text stays"),
        pytest.param("data:text/plain," + "A" * 4000, True, id="any media type goes"),
    ],
)
def test_only_data_urls_at_or_over_the_threshold_are_referenced(tmp_path: Path, value: str, referenced: bool) -> None:
    row = {"_ng_rollout_id": "1-0", "ng_trajectory": {"tool_calls": [{"output": value}]}}

    stats = reference_row_media(row, rollouts_media_location(tmp_path / "r.jsonl", "1-0"))

    assert (stats.references == 1) is referenced
    assert (row["ng_trajectory"]["tool_calls"][0]["output"] is None) is referenced
    assert (NG_ROW_MEDIA_KEY in row) is referenced


def test_the_threshold_can_be_raised(tmp_path: Path) -> None:
    row = {"ng_trajectory": {"tool_calls": [{"output": SCREENSHOT}]}}

    stats = reference_row_media(
        row, rollouts_media_location(tmp_path / "r.jsonl", "1-0"), min_chars=len(SCREENSHOT) + 1
    )

    assert stats.references == 0 and row["ng_trajectory"]["tool_calls"][0]["output"] == SCREENSHOT


def test_a_row_without_large_data_urls_is_left_byte_identical(tmp_path: Path) -> None:
    row = {"reward": 0.5, "ng_trajectory": {"tool_calls": [{"output": "plain text"}]}}
    original = orjson.dumps(row)
    location = rollouts_media_location(tmp_path / "r.jsonl", "1-0")

    stats = reference_row_media(row, location)

    assert stats.references == 0 and orjson.dumps(row) == original
    assert not location.directory.exists()
    assert restore_row_media(row) is row


def test_restoring_a_copy_leaves_the_referenced_row_unchanged(tmp_path: Path) -> None:
    row = _row()
    reference_row_media(row, rollouts_media_location(tmp_path / "rollouts.jsonl", "3-0"))
    referenced = copy.deepcopy(row)

    restored = restore_row_media(row, rollouts_dir=tmp_path)

    assert row == referenced
    assert restored["ng_trajectory"]["tool_calls"][0]["output"][0]["image_url"] == SCREENSHOT
    # One blob read serves every place that names it.
    assert (
        restored["ng_trajectory"]["tool_calls"][0]["output"][0]["image_url"]
        is restored["ng_agent_observations"]["records"][0]["conversation"][0]["output"][0]["image_url"]
    )


def test_restoring_in_place_reuses_the_row(tmp_path: Path) -> None:
    row = _row()
    reference_row_media(row, rollouts_media_location(tmp_path / "rollouts.jsonl", "3-0"))

    restored = restore_row_media(row, rollouts_dir=tmp_path, in_place=True)

    assert restored is row and NG_ROW_MEDIA_KEY not in row
    assert orjson.dumps(row) == orjson.dumps(_row())


def test_a_missing_blob_names_itself_and_where_it_was_looked_for(tmp_path: Path) -> None:
    row = _row()
    location = rollouts_media_location(tmp_path / "rollouts.jsonl", "3-0")
    reference_row_media(row, location)
    _blobs(location.directory)[0].unlink()

    with pytest.raises(RowMediaUnavailable, match=r"rollouts_blobs/3-0.*row_media: reference"):
        restore_row_media(row, rollouts_dir=tmp_path)
    with pytest.raises(RowMediaUnavailable, match="no directory"):
        restore_row_media(row)


@pytest.mark.parametrize(
    "listing",
    [
        pytest.param({"version": 1, "root": "rollouts_dir", "dir": "../elsewhere", "blobs": []}, id="escaping dir"),
        pytest.param({"version": 1, "root": "rollouts_dir", "dir": "/abs", "blobs": []}, id="absolute dir"),
        pytest.param({"version": 2, "root": "rollouts_dir", "dir": "d", "blobs": []}, id="unknown version"),
        pytest.param({"version": 1, "root": "rollouts_dir", "dir": "d", "blobs": [[[], "a" * 64]]}, id="empty path"),
        pytest.param({"version": 1, "root": "rollouts_dir", "dir": "d", "blobs": [[["x"], "nothex"]]}, id="digest"),
        pytest.param({"version": 1, "root": "rollouts_dir", "dir": "d", "blobs": [], "extra": 1}, id="unknown field"),
    ],
)
def test_a_malformed_listing_is_refused(tmp_path: Path, listing: dict[str, Any]) -> None:
    with pytest.raises(RowMediaUnavailable, match="malformed"):
        restore_row_media({NG_ROW_MEDIA_KEY: listing}, rollouts_dir=tmp_path)


def test_a_listed_place_that_holds_a_value_is_refused(tmp_path: Path) -> None:
    row = _row()
    reference_row_media(row, rollouts_media_location(tmp_path / "rollouts.jsonl", "3-0"))
    row["ng_trajectory"]["tool_calls"][0]["output"][0]["image_url"] = "something else"

    with pytest.raises(RowMediaUnavailable, match="placeholder"):
        restore_row_media(row, rollouts_dir=tmp_path)


def test_images_the_delta_capture_already_holds_are_not_written_again(tmp_path: Path) -> None:
    capture_dir = tmp_path / "model-calls"
    store = CaptureStore(capture_dir, capture_format="delta")
    request = {
        "model": "m",
        "input": [{"type": "function_call_output", "call_id": "v", "output": _image_output(SCREENSHOT)}],
    }
    store.record("3-0", {"model_call_id": "c1", "dialect": "responses", "request": request, "response": {}})
    [capture_blob] = _blobs(store.blob_dir_for("3-0"))
    row = _row()
    row["ng_trajectory"]["invocations"][0]["conversation"].pop()  # the capture never saw the user's image

    stats = reference_row_media(row, capture_media_location(capture_dir, "3-0"))

    assert (stats.unique, stats.written_blobs, stats.written_bytes) == (1, 0, 0)
    assert _blobs(store.blob_dir_for("3-0")) == [capture_blob]
    assert row[NG_ROW_MEDIA_KEY]["root"] == "model_call_capture_dir"
    assert row[NG_ROW_MEDIA_KEY]["dir"] == "blobs/3-0"
    assert (
        restore_row_media(row, capture_dirs=[tmp_path / "other", capture_dir])["ng_trajectory"]["tool_calls"][0][
            "output"
        ][0]["image_url"]
        == SCREENSHOT
    )
    # The capture's own reader still rebuilds its line from the shared blob.
    assert store.read("3-0")[0]["request"] == request


def test_a_second_pass_adds_new_references_to_the_listing(tmp_path: Path) -> None:
    row = _row()
    location = rollouts_media_location(tmp_path / "rollouts.jsonl", "3-0")
    reference_row_media(row, location)
    row["ng_trajectory"]["tool_calls"].append({"output": _image(b"late")})

    stats = reference_row_media(row, location)

    assert stats.references == 1 and len(row[NG_ROW_MEDIA_KEY]["blobs"]) == 6
    restored = restore_row_media(row, rollouts_dir=tmp_path)
    assert restored["ng_trajectory"]["tool_calls"][-1]["output"] == _image(b"late")
    with pytest.raises(ValueError, match="already references"):
        reference_row_media(row, capture_media_location(tmp_path, "3-0"))


def test_a_failed_blob_write_leaves_the_row_inline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_gym.trajectory.row_media as row_media

    def failing(*args: Any) -> bool:
        raise OSError("disk full")

    monkeypatch.setattr(row_media, "write_blob", failing)
    row = _row()

    with pytest.raises(OSError):
        reference_row_media(row, rollouts_media_location(tmp_path / "rollouts.jsonl", "3-0"))

    assert row == _row()


def test_the_setting_is_read_from_the_global_config() -> None:
    assert row_media_from_config({}) == "inline"
    assert row_media_from_config({"row_media": "reference", "other": 1}) == "reference"
    with pytest.raises(ValueError):
        row_media_from_config({"row_media": "sometimes"})


def test_the_blob_dirs_of_a_rollouts_file_come_from_its_saved_config(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    assert row_media_dirs(rollouts) == ([], tmp_path)
    (tmp_path / "rollouts_resolved_config.yaml").write_text("model_call_capture_dir: /runs/x/model-calls\n")

    assert row_media_dirs(rollouts) == ([Path("/runs/x/model-calls")], tmp_path)


# --- Readers ---


def test_the_exporter_view_lists_only_the_references_it_keeps(tmp_path: Path) -> None:
    record = evidence_record()
    record["ng_agent_observations"]["records"][0]["conversation"][2]["output"] = _image_output(SCREENSHOT)
    record["ng_trajectory"]["invocations"][0]["conversation"][2]["output"] = _image_output(SCREENSHOT)
    inline_view = _rollout_for_export(copy.deepcopy(record))

    reference_row_media(record, rollouts_media_location(tmp_path / "rollouts.jsonl", "0-0"))
    view = _rollout_for_export(record)

    # The view drops conversations, so only the observation record's reference is left, and it still resolves.
    assert [path for path, _ in view[NG_ROW_MEDIA_KEY]["blobs"]] == [
        ["ng_agent_observations", "records", 0, "conversation", 2, "output", 0, "image_url"]
    ]
    assert len(record[NG_ROW_MEDIA_KEY]["blobs"]) == 2
    assert restore_row_media(view, rollouts_dir=tmp_path) == inline_view


def test_the_exporter_view_of_an_inline_row_is_unchanged() -> None:
    record = evidence_record()
    assert NG_ROW_MEDIA_KEY not in _rollout_for_export(record)


@pytest.mark.parametrize("make_row", [_row, evidence_record], ids=["codex-shaped", "conformance-evidence"])
def test_the_health_digest_of_a_referenced_row_equals_the_original(tmp_path: Path, make_row) -> None:
    """The rollout health checks read tool output text, never image data, so referencing a row's images changes
    none of its findings."""
    from nemo_gym.rollouts.rollout_health import evaluate_record

    row = make_row()
    referenced = copy.deepcopy(row)
    reference_row_media(referenced, rollouts_media_location(tmp_path / "rollouts.jsonl", "3-0"), min_chars=64)

    if make_row is _row:
        assert referenced[NG_ROW_MEDIA_KEY]["blobs"]
    digest = evaluate_record(referenced, frozenset())
    assert digest.model_dump(mode="json") == evaluate_record(row, frozenset()).model_dump(mode="json")


def test_conformance_refuses_a_referenced_record_and_checks_the_restored_one(tmp_path: Path) -> None:
    inline = evidence_record()
    inline["ng_trajectory"]["model_calls"][1]["request"]["input"][2]["output"] = _image_output(SCREENSHOT)
    expected = inspect_record(copy.deepcopy(inline))
    referenced = copy.deepcopy(inline)
    reference_row_media(referenced, rollouts_media_location(tmp_path / "rollouts.jsonl", "0-0"))

    with pytest.raises(RowMediaUnavailable, match=r"content.media.*restore_row_media"):
        inspect_record(referenced)
    assert inspect_record(restore_row_media(referenced, rollouts_dir=tmp_path)) == expected


def test_a_conformance_bundle_restores_its_rows(tmp_path: Path) -> None:
    record = evidence_record()
    record["ng_trajectory"]["model_calls"][1]["request"]["input"][2]["output"] = _image_output(SCREENSHOT)
    inline_dir, referenced_dir = tmp_path / "inline", tmp_path / "referenced"
    inline_dir.mkdir()
    referenced_dir.mkdir()
    (inline_dir / "rollouts.jsonl").write_bytes(orjson.dumps(record) + b"\n")
    reference_row_media(record, rollouts_media_location(referenced_dir / "rollouts.jsonl", "0-0"))
    (referenced_dir / "rollouts.jsonl").write_bytes(orjson.dumps(record) + b"\n")

    _, inline_summary = conformance_cli.inspect_bundle(inline_dir / "rollouts.jsonl", output=tmp_path / "a")
    _, summary = conformance_cli.inspect_bundle(referenced_dir / "rollouts.jsonl", output=tmp_path / "b")

    assert summary["verdict"] == inline_summary["verdict"] == "fulfilled"


def _atif_config(source: Path, output: Path) -> ExportAtifConfig:
    return ExportAtifConfig(
        rollouts_jsonl_fpath=source, output_dirpath=output, session_id="evaluation-42", agent_version="2.3.1"
    )


def _atif_rollout() -> dict[str, Any]:
    from tests.unit_tests.trajectory.test_atif_export import _canonical_rollout

    rollout = _canonical_rollout()
    rollout["_ng_rollout_id"] = "3-7"
    rollout["ng_agent_observations"] = {"source": "test", "records": [{"conversation": _image_output(SCREENSHOT)}]}
    return rollout


def test_atif_export_restores_referenced_rows_from_the_run(tmp_path: Path) -> None:
    inline = tmp_path / "inline" / "rollouts.jsonl"
    referenced = tmp_path / "referenced" / "rollouts.jsonl"
    inline.parent.mkdir()
    referenced.parent.mkdir()
    inline.write_bytes(orjson.dumps(_atif_rollout()) + b"\n")
    row = _atif_rollout()
    reference_row_media(row, rollouts_media_location(referenced, "3-7"))
    assert row[NG_ROW_MEDIA_KEY]["blobs"]
    referenced.write_bytes(orjson.dumps(row) + b"\n")

    export_rollouts_to_atif(_atif_config(inline, tmp_path / "inline-atif"))
    export_rollouts_to_atif(_atif_config(referenced, tmp_path / "referenced-atif"))

    assert (tmp_path / "referenced-atif" / "3-7.json").read_bytes() == (
        tmp_path / "inline-atif" / "3-7.json"
    ).read_bytes()

    for blob in _blobs(rollouts_media_location(referenced, "3-7").directory):
        blob.unlink()
    with pytest.raises(AtifExportError, match=r"line 1\.ng_row_media: row media blob"):
        export_rollouts_to_atif(_atif_config(referenced, tmp_path / "missing-atif"))


def test_atif_conversion_asks_for_referenced_trajectory_images_to_be_restored(tmp_path: Path) -> None:
    from nemo_gym.trajectory.atif_export import gym_rollout_to_atif

    rollout = _atif_rollout()
    rollout["ng_trajectory"]["tool_calls"][0]["output"] = SCREENSHOT
    reference_row_media(rollout, rollouts_media_location(tmp_path / "rollouts.jsonl", "3-7"))

    with pytest.raises(AtifExportError, match=r"ng_trajectory\.tool_calls\.0\.output: .*restore_row_media"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_the_listing_is_plain_json(tmp_path: Path) -> None:
    row = _row()
    reference_row_media(row, rollouts_media_location(tmp_path / "rollouts.jsonl", "3-0"))

    assert json.loads(json.dumps(row[NG_ROW_MEDIA_KEY])) == row[NG_ROW_MEDIA_KEY]


# --- Collection ---


def test_a_record_s_media_goes_to_the_capture_dir_else_beside_the_rollouts(tmp_path: Path) -> None:
    from nemo_gym.rollouts.model_calls import reference_record_media

    with_capture, beside_rows, nowhere, anonymous = _row(), _row(), _row(), _row()
    anonymous.pop("_ng_rollout_id")
    anonymous.pop("_ng_task_index")

    reference_record_media(with_capture, [tmp_path / "model-calls"], tmp_path / "rollouts.jsonl")
    reference_record_media(beside_rows, [], tmp_path / "rollouts.jsonl")

    assert with_capture[NG_ROW_MEDIA_KEY]["root"] == "model_call_capture_dir"
    assert (tmp_path / "model-calls" / "blobs" / "3-0").is_dir()
    assert beside_rows[NG_ROW_MEDIA_KEY]["dir"] == "rollouts_blobs/3-0"
    assert reference_record_media(nowhere, [], None) is None and nowhere == _row()
    assert reference_record_media(anonymous, [], tmp_path / "rollouts.jsonl") is None


async def test_collection_keeps_a_row_inline_when_its_media_cannot_be_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import nemo_gym.trajectory.row_media as row_media
    from nemo_gym.rollouts.rollout_collection import _reference_record_media

    def failing(*args: Any) -> bool:
        raise OSError("disk full")

    monkeypatch.setattr(row_media, "write_blob", failing)
    row = _row()

    await _reference_record_media(row, [], tmp_path / "rollouts.jsonl")

    assert row == _row()
    assert "keeps them inline" in caplog.text


def test_behavior_inspection_refuses_a_referenced_record_and_matches_the_restored_one(tmp_path: Path) -> None:
    from nemo_gym.rollouts.harness_capabilities.behavior import inspect_behavior
    from tests.unit_tests.rollouts.harness_capabilities.synthetic import evidence_and_witness
    from tests.unit_tests.rollouts.harness_capabilities.test_behavior import fingerprint

    record, witness = evidence_and_witness()
    record["ng_trajectory"]["model_calls"][1]["request"]["input"][2]["output"] = _image_output(SCREENSHOT)
    witness["attempts"][1]["request"] = record["ng_trajectory"]["model_calls"][1]["request"]
    options = dict(http_errors=(), terminal_error=False, tool_steps=2, expected_reward=0.0, fingerprint=fingerprint)
    expected = inspect_behavior(witness, copy.deepcopy(record), **options)
    reference_row_media(record, rollouts_media_location(tmp_path / "rollouts.jsonl", "0-0"))

    with pytest.raises(RowMediaUnavailable, match="Harness behavior inspection"):
        inspect_behavior(witness, record, **options)
    assert inspect_behavior(witness, restore_row_media(record, rollouts_dir=tmp_path), **options) == expected
