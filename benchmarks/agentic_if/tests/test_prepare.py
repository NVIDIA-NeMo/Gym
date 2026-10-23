# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agentic_if import prepare
from resources_servers.agentic_if.task_variants.builder import digest
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec, seal_runtime_row


EXAMPLES = Path(prepare.__file__).parents[2] / "resources_servers/agentic_if/data/example.jsonl"
FOLDER = f"{prepare.TASKS_PREFIX}/tasks/demo/task-a"


def _terminal_row(folder: str) -> dict:
    """A public example terminal row, resealed to name a private-split task folder."""
    example = next(row for row in map(json.loads, EXAMPLES.read_text().splitlines()) if "task_folder" in row)
    spec = copy.deepcopy(example["agentic_if"])
    spec["receipt"] = None
    row = {key: value for key, value in example.items() if key not in ("task_id", "agentic_if")}
    row["task_folder"] = folder
    sealed = seal_runtime_row(row, RuntimeSpec.model_validate(spec))
    return {"task_id": digest(sealed), **sealed}


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Point preparation at temporary paths and serve the two artifacts from memory."""
    artifacts = {
        prepare.ROWS_ARTIFACT: _jsonl([_terminal_row(FOLDER)]),
        prepare.TESTS_ARTIFACT: _jsonl(
            [
                {
                    "path": f"{FOLDER}/tests/test.sh",
                    "mode": 0o755,
                    "content_b64": base64.b64encode(b"exit 0\n").decode(),
                },
                {"path": f"{FOLDER}/tests/data.txt", "mode": 0o644, "content_b64": base64.b64encode(b"x").decode()},
            ]
        ),
    }

    def download(config):
        Path(config.output_fpath).write_bytes(artifacts[config.artifact_fpath])

    monkeypatch.setattr(prepare, "download_jsonl_dataset", download)
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "benchmark/data/instruction-following.jsonl")
    monkeypatch.setattr(prepare, "TASKS_DIR", tmp_path / "gym" / prepare.TASKS_PREFIX)
    monkeypatch.setattr(prepare, "ROWS_SHA256", hashlib.sha256(artifacts[prepare.ROWS_ARTIFACT]).hexdigest())
    monkeypatch.setattr(prepare, "TESTS_SHA256", hashlib.sha256(artifacts[prepare.TESTS_ARTIFACT]).hexdigest())
    return artifacts


def test_prepare_writes_rows_and_task_tests(registry):
    output = prepare.prepare()

    assert output.read_bytes() == registry[prepare.ROWS_ARTIFACT]
    task = prepare.TASKS_DIR / "tasks/demo/task-a/tests"
    assert (task / "test.sh").read_bytes() == b"exit 0\n"
    assert (task / "test.sh").stat().st_mode & 0o777 == 0o755
    assert (task / "data.txt").stat().st_mode & 0o777 == 0o644
    assert prepare.is_prepared_data_current(output)


def test_prepare_replaces_an_earlier_preparation(registry):
    prepare.prepare()
    stale = prepare.TASKS_DIR / "tasks/stale/task-z/tests/test.sh"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale")

    prepare.prepare()

    assert not stale.exists()


def test_a_wrong_digest_fails_before_anything_is_written(registry, monkeypatch):
    monkeypatch.setattr(prepare, "ROWS_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="mlflow_tracking_token"):
        prepare.prepare()

    assert not prepare.OUTPUT_FPATH.exists()
    assert not prepare.TASKS_DIR.exists()


def test_a_task_folder_without_tests_fails(registry, monkeypatch):
    rows = _jsonl([_terminal_row(f"{prepare.TASKS_PREFIX}/tasks/demo/task-b")])
    registry[prepare.ROWS_ARTIFACT] = rows
    monkeypatch.setattr(prepare, "ROWS_SHA256", hashlib.sha256(rows).hexdigest())

    with pytest.raises(RuntimeError, match="no tests/test.sh"):
        prepare.prepare()

    assert not prepare.OUTPUT_FPATH.exists()


@pytest.mark.parametrize("folder", ["benchmarks/other/tasks/x", f"{prepare.TASKS_PREFIX}/../escape"])
def test_rows_must_name_folders_under_the_prefix(registry, monkeypatch, folder):
    rows = _jsonl([_terminal_row(folder)])
    registry[prepare.ROWS_ARTIFACT] = rows
    monkeypatch.setattr(prepare, "ROWS_SHA256", hashlib.sha256(rows).hexdigest())

    with pytest.raises(ValueError, match="is not a folder of"):
        prepare.prepare()


def test_a_tampered_row_fails_its_identity_check(registry, monkeypatch):
    row = _terminal_row(FOLDER)
    row["task_name"] = "changed after sealing"
    rows = _jsonl([row])
    registry[prepare.ROWS_ARTIFACT] = rows
    monkeypatch.setattr(prepare, "ROWS_SHA256", hashlib.sha256(rows).hexdigest())

    with pytest.raises(ValueError, match="content digest"):
        prepare.prepare()


def test_is_prepared_data_current_needs_both_parts(registry, tmp_path):
    output = prepare.prepare()
    (prepare.TASKS_DIR / prepare.TESTS_MARKER).write_text("0" * 64)

    assert not prepare.is_prepared_data_current(output)
    assert not prepare.is_prepared_data_current(tmp_path / "missing.jsonl")
