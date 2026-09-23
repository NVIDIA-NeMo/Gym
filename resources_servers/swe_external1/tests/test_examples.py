# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public example assets and empty-log grading checks, not live sandbox rollouts."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.task_data import load_task_data_schema, validate_jsonl_rows
from resources_servers.swe_external1.task_data import TaskRow


SERVER = Path(__file__).resolve().parents[1]
ROWS = [json.loads(line) for line in (SERVER / "data/example.jsonl").read_text().splitlines()]
PUBLIC_TASK_IDS = {
    "intel__rohd-458",
    "syuilo__aiscript-257",
    "taiki-e__cargo-hack-70",
    "tox-dev__pipdeptree-279",
    "cta-observatory__ctapipe-2397",
}


def files(task, name):
    return {asset.path: asset.decoded() for asset in getattr(task, name)}


def test_five_public_examples_have_complete_assets():
    assert len(ROWS) == 5
    assert {row["verifier_metadata"]["task_id"] for row in ROWS} == PUBLIC_TASK_IDS
    for row in ROWS:
        task = TaskRow.model_validate(row).verifier_metadata
        tests, solution = files(task, "test_files"), files(task, "solution_files")
        assert row["public_source"]["instance_id"] == task.task_id
        assert row["responses_create_params"]["input"]
        assert task.image_ref.startswith("docker.io/swerebenchv2/")
        assert task.setup_script.startswith("git checkout --detach ")
        assert {"test.sh", "test.patch", "grade.py", "expected.json", "lib/agent/log_parsers.py", "LICENSE"} <= set(tests)
        assert set(solution) == {"solve.sh", "solution.patch"}
        assert tests["test.patch"].strip() and solution["solution.patch"].strip()
        expected = json.loads(tests["expected.json"])
        assert expected["parser"] and expected["FAIL_TO_PASS"]
        assert isinstance(expected["PASS_TO_PASS"], list)
        test_script = tests["test.sh"].decode()
        assert "git reset" not in test_script and "git checkout" not in test_script
        assert "git apply --check /tests/test.patch" in test_script
        assert "git apply --check /solution/solution.patch" in solution["solve.sh"].decode()
        assert b"MIT License" in tests["LICENSE"]


def run_packaged_grader(row, log, directory):
    task = TaskRow.model_validate(row).verifier_metadata
    for name, data in files(task, "test_files").items():
        target = directory / "tests" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    output = directory / "logs/verifier"
    output.mkdir(parents=True)
    (output / "test-output.txt").write_text(log)
    # Relocate the packaged script's two absolute roots for host-side parser tests.
    # All grading logic and parser bytes are unchanged; no task commands run here.
    grade = directory / "tests/grade.py"
    source = grade.read_text().replace('Path("/tests/', f'Path("{directory}/tests/')
    source = source.replace('Path("/logs/', f'Path("{directory}/logs/')
    grade.write_text(source)
    result = subprocess.run([sys.executable, str(grade)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return float((output / "reward.txt").read_text()), json.loads((output / "report.json").read_text())


def test_gym_flattened_schema_and_legacy_locations():
    adapter = load_task_data_schema(SERVER)
    report = validate_jsonl_rows("swe_external1", adapter, "examples", map(json.dumps, ROWS))
    assert report.rows == 5 and report.clean, report.summary()
    flat = {**ROWS[0]["verifier_metadata"], "responses_create_params": ROWS[0]["responses_create_params"]}
    report = validate_jsonl_rows("swe_external1", adapter, "wrong-location", [json.dumps(flat)])
    assert not report.clean and "task_id" in report.misplaced_keys
    migrated = {"task_data": ROWS[0]["verifier_metadata"]}
    report = validate_jsonl_rows("swe_external1", adapter, "migrated", [json.dumps(migrated)])
    assert report.clean, report.summary()


@pytest.mark.parametrize("row", ROWS, ids=[row["verifier_metadata"]["task_id"] for row in ROWS])
def test_empty_logs_never_pass(row, tmp_path):
    reward, report = run_packaged_grader(row, "", tmp_path)
    assert reward == 0
    assert report["passed"] == []
    assert len(report["missing_or_failed"]) == report["required"]
