# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public fixture integrity and offline parser tests, not live sandbox rollouts."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.task_data import load_task_data_schema, validate_jsonl_rows
from resources_servers.swe_external1.task_data import TaskRow


SERVER = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
ROWS = [json.loads(line) for line in (SERVER / "data/example.jsonl").read_text().splitlines()]
SOURCE_ROWS = {row["instance_id"]: row for row in json.loads((FIXTURES / "source-examples.json").read_text())}
RECORDED_TEST_OUTPUTS = json.loads((FIXTURES / "recorded-test-outputs.json").read_text())


def files(task, name):
    return {asset.path: asset.decoded() for asset in getattr(task, name)}


def test_exactly_the_same_five_public_examples():
    assert len(ROWS) == 5
    assert {row["verifier_metadata"]["task_id"] for row in ROWS} == set(SOURCE_ROWS)
    for row in ROWS:
        task = TaskRow.model_validate(row).verifier_metadata
        source = SOURCE_ROWS[task.task_id]
        tests, solution = files(task, "test_files"), files(task, "solution_files")
        assert row["responses_create_params"] == source["responses_create_params"]
        assert task.image_ref == source["image_name"]
        assert task.workdir == "/" + source["repo"].split("/", 1)[1]
        assert task.setup_script == "git checkout --detach " + source["base_commit"]
        assert solution["solution.patch"] == source["patch"].encode()
        assert tests["test.patch"] == source["test_patch"].encode()
        assert json.loads(tests["expected.json"]) == {
            "parser": source["install_config"]["log_parser"],
            "FAIL_TO_PASS": source["FAIL_TO_PASS"],
            "PASS_TO_PASS": source["PASS_TO_PASS"],
        }
        test_script = tests["test.sh"].decode()
        assert "git reset" not in test_script and "git checkout" not in test_script
        assert "git apply --check /tests/test.patch" in test_script
        for field in ("install", "test_cmd"):
            commands = source["install_config"].get(field) or []
            for command in [commands] if isinstance(commands, str) else commands:
                assert command in test_script
        assert b"MIT License" in tests["LICENSE"]


def run_packaged_grader(row, log, directory, missing_required=False):
    task = TaskRow.model_validate(row).verifier_metadata
    for name, data in files(task, "test_files").items():
        target = directory / "tests" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    output = directory / "logs/verifier"
    output.mkdir(parents=True)
    (output / "test-output.txt").write_text(log)
    if missing_required:
        expected_path = directory / "tests/expected.json"
        expected = json.loads(expected_path.read_text())
        expected["FAIL_TO_PASS"].append("missing-test-never-in-the-log")
        expected_path.write_text(json.dumps(expected))
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


@pytest.mark.parametrize("row", ROWS, ids=[row["verifier_metadata"]["task_id"] for row in ROWS])
def test_recorded_source_logs_and_missing_required_tests(row, tmp_path):
    recorded = RECORDED_TEST_OUTPUTS[row["verifier_metadata"]["task_id"]]
    reward, report = run_packaged_grader(row, recorded, tmp_path / "recorded")
    # These are existing source logs, not evidence that this adapter ran the tasks.
    # A truncated source log may lack required tests; it must not receive a free pass.
    if len(recorded) < 100_000:
        assert reward == 1, report
    else:
        assert reward == (not report["missing_or_failed"])
    assert report["passed"]
    reward, report = run_packaged_grader(row, recorded, tmp_path / "missing", missing_required=True)
    assert reward == 0
    assert "missing-test-never-in-the-log" in report["missing_or_failed"]
