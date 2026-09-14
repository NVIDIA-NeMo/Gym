# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.gdpval.validate_gdpval_batch import (
    _validate_private_file,
    _write_jsonl_atomic,
    apply_reference_overrides,
    validate_input_rows,
    validate_rollouts,
)


def _input_row(**overrides):
    row = {
        "responses_create_params": {"input": []},
        "task_id": "GDP-00001",
        "sector": "Technology",
        "occupation": "Software Engineer",
        "prompt": "Use https://example.test/paper.pdf as the source.",
        "reference_files": [],
        "reference_file_urls": [],
        "rubric_json": [{"criterion": "Create the result", "score": 1}],
        "rubric_pretty": "[+1] Create the result",
    }
    row.update(overrides)
    return row


def test_reference_overrides_are_prompt_backed_and_do_not_mutate_source(tmp_path):
    source_row = _input_row()
    override_path = tmp_path / "overrides.json"
    override_path.write_text(
        json.dumps(
            {
                "GDP-00001": {
                    "reference_files": ["paper.pdf"],
                    "reference_file_urls": ["https://example.test/paper.pdf"],
                }
            }
        ),
        encoding="utf-8",
    )

    launch_rows, applied = apply_reference_overrides([source_row], override_path)

    assert applied == ["GDP-00001"]
    assert source_row["reference_files"] == []
    assert launch_rows[0]["reference_files"] == ["paper.pdf"]
    assert validate_input_rows(launch_rows, expected_count=1) == ([], [])


def test_overrides_for_tasks_outside_this_input_are_ignored(tmp_path):
    """One overrides file serves a whole dataset; an input is routinely a slice of it.

    A re-collection of the tasks a repeat missed carries the dataset-wide overrides file,
    so most of its keys name tasks that are legitimately absent from this input. Rejecting
    those blocked a 29-task recovery run outright.
    """
    override_path = tmp_path / "overrides.json"
    override_path.write_text(
        json.dumps(
            {
                "GDP-00001": {
                    "reference_files": ["paper.pdf"],
                    "reference_file_urls": ["https://example.test/paper.pdf"],
                },
                "GDP-09999": {
                    "reference_files": ["absent.pdf"],
                    "reference_file_urls": ["https://example.test/absent.pdf"],
                },
            }
        ),
        encoding="utf-8",
    )

    launch_rows, applied = apply_reference_overrides([_input_row()], override_path)

    assert applied == ["GDP-00001"]
    assert launch_rows[0]["reference_files"] == ["paper.pdf"]


def test_overrides_matching_nothing_in_the_input_are_not_an_error(tmp_path):
    """Matching nothing is normal, because these files are sparse.

    The AfterQuery overrides file repairs 2 tasks out of 1013, so a 29-task recovery slice
    sharing none of them is the expected case rather than a mispassed file. Treating it as
    an error is what blocked that recovery run.
    """
    override_path = tmp_path / "overrides.json"
    override_path.write_text(
        json.dumps(
            {
                "GDP-09999": {
                    "reference_files": ["absent.pdf"],
                    "reference_file_urls": ["https://example.test/absent.pdf"],
                }
            }
        ),
        encoding="utf-8",
    )

    launch_rows, applied = apply_reference_overrides([_input_row()], override_path)

    assert applied == []
    assert launch_rows[0]["reference_files"] == []


def test_reference_override_rejects_url_absent_from_prompt(tmp_path):
    override_path = tmp_path / "overrides.json"
    override_path.write_text(
        json.dumps(
            {
                "GDP-00001": {
                    "reference_files": ["other.pdf"],
                    "reference_file_urls": ["https://example.test/other.pdf"],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="added URL is absent"):
        apply_reference_overrides([_input_row()], override_path)


def test_input_validation_rejects_nested_gold_output_keys():
    row = _input_row(responses_create_params={"input": [], "metadata": {"deliverable_files": ["gold.pdf"]}})

    errors, _ = validate_input_rows([row], expected_count=1)

    assert any("gold-output keys" in error for error in errors)
    assert any("responses_create_params must be exactly" in error for error in errors)


def test_private_file_validation_rejects_group_or_world_permissions(tmp_path):
    private_path = tmp_path / "private.jsonl"
    private_path.write_text("{}\n", encoding="utf-8")
    private_path.chmod(0o600)
    assert _validate_private_file(private_path) is None

    private_path.chmod(0o640)
    assert "group/world permissions 0640" in str(_validate_private_file(private_path))


def test_execute_only_rollout_validation_checks_cache_and_materialized_input(tmp_path):
    rollouts_path = tmp_path / "rollouts.jsonl"
    deliverables_dir = tmp_path / "deliverables"
    task_dir = deliverables_dir / "task_GDP-00001" / "repeat_0"
    task_dir.mkdir(parents=True)
    (task_dir / "answer.pdf").write_bytes(b"%PDF-test")
    (task_dir / "finish_params.json").write_text(
        json.dumps({"paths": ["/root/answer.pdf"], "reason": "done"}), encoding="utf-8"
    )
    rollout = {
        "task_id": "GDP-00001",
        "execute_only": True,
        "response": {"model": "glm-5.2-bf16", "error": None},
        "agent_ref": {"type": "responses_api_agents", "name": "gdpval_stirrup_agent"},
        "deliverables_dir": str(task_dir),
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
    }
    _write_jsonl_atomic(rollouts_path, [rollout])
    _write_jsonl_atomic(
        tmp_path / "rollouts_materialized_inputs.jsonl",
        [_input_row(agent_ref={"type": "responses_api_agents", "name": "gdpval_stirrup_agent"})],
    )

    errors, warnings = validate_rollouts(
        rollouts_path,
        expected_task_ids={"GDP-00001"},
        deliverables_dir=deliverables_dir,
        require_deliverable=True,
        expected_response_model="glm-5.2-bf16",
    )

    assert errors == []
    assert warnings == []


def test_execute_only_rollout_validation_rejects_judge_fields(tmp_path):
    rollouts_path = tmp_path / "rollouts.jsonl"
    rollout = {
        "task_id": "GDP-00001",
        "execute_only": True,
        "reward": 1.0,
        "response": {"model": "glm-5.2-bf16", "error": None},
        "agent_ref": {"type": "responses_api_agents", "name": "gdpval_stirrup_agent"},
    }
    _write_jsonl_atomic(rollouts_path, [rollout])
    _write_jsonl_atomic(
        tmp_path / "rollouts_materialized_inputs.jsonl",
        [_input_row(agent_ref={"type": "responses_api_agents", "name": "gdpval_stirrup_agent"})],
    )

    errors, _ = validate_rollouts(
        rollouts_path,
        expected_task_ids={"GDP-00001"},
        deliverables_dir=None,
        require_deliverable=False,
        expected_response_model="glm-5.2-bf16",
    )

    assert any("judge/reward fields" in error for error in errors)


def test_rollout_validation_requires_expected_model_on_every_row(tmp_path):
    rollouts_path = tmp_path / "rollouts.jsonl"
    rollout = {
        "task_id": "GDP-00001",
        "execute_only": True,
        "response": {"error": None},
        "agent_ref": {"type": "responses_api_agents", "name": "gdpval_stirrup_agent"},
    }
    _write_jsonl_atomic(rollouts_path, [rollout])
    _write_jsonl_atomic(
        tmp_path / "rollouts_materialized_inputs.jsonl",
        [_input_row(agent_ref={"type": "responses_api_agents", "name": "gdpval_stirrup_agent"})],
    )

    errors, _ = validate_rollouts(
        rollouts_path,
        expected_task_ids={"GDP-00001"},
        deliverables_dir=None,
        require_deliverable=False,
        expected_response_model="glm-5.2-bf16",
    )

    assert any("GDP-00001: expected response model" in error for error in errors)


def test_full_rollout_allows_completed_model_outcome_without_deliverable(tmp_path):
    rollouts_path = tmp_path / "rollouts.jsonl"
    deliverables_dir = tmp_path / "deliverables"
    task_dir = deliverables_dir / "task_GDP-00001" / "repeat_0"
    task_dir.mkdir(parents=True)
    (task_dir / "finish_params.json").write_text(json.dumps({"paths": [], "reason": "unable"}), encoding="utf-8")
    rollout = {
        "task_id": "GDP-00001",
        "execute_only": True,
        "response": {"model": "glm-5.2-bf16", "error": None},
        "agent_ref": {"type": "responses_api_agents", "name": "gdpval_stirrup_agent"},
        "deliverables_dir": str(task_dir),
    }
    _write_jsonl_atomic(rollouts_path, [rollout])
    _write_jsonl_atomic(
        tmp_path / "rollouts_materialized_inputs.jsonl",
        [_input_row(agent_ref={"type": "responses_api_agents", "name": "gdpval_stirrup_agent"})],
    )

    errors, warnings = validate_rollouts(
        rollouts_path,
        expected_task_ids={"GDP-00001"},
        deliverables_dir=deliverables_dir,
        require_deliverable=False,
        expected_response_model="glm-5.2-bf16",
    )

    assert errors == []
    assert warnings == []


def _row(task_id: str, urls: list[str], files: list[str]) -> dict:
    return {
        "responses_create_params": {"input": []},
        "task_id": task_id,
        "sector": "",
        "occupation": "Accountants and Auditors",
        "prompt": "do the thing",
        "reference_files": files,
        "reference_file_urls": urls,
        "rubric_json": [{"criterion": "c"}],
        "rubric_pretty": "c",
    }


def test_default_profile_accepts_any_id_but_still_requires_https():
    """The default id pattern is permissive so batches from any GDPVal source
    validate out of the box; the HTTPS default is a separate, stricter check."""
    rows = [_row("task_" + "a" * 32, ["/shared/x.xlsx"], ["x.xlsx"])]
    errors, _ = validate_input_rows(rows, expected_count=1)

    assert not any("invalid task_id" in e for e in errors)
    assert any("must use HTTPS" in e for e in errors)


def test_dataset_profile_can_pin_the_id_format():
    """A batch that needs a stricter contract sets it in its dataset profile."""
    rows = [_row("NOT-THE-FORMAT", ["https://example.test/x.xlsx"], ["x.xlsx"])]
    errors, _ = validate_input_rows(rows, expected_count=1, task_id_pattern=r"task_[0-9a-f]{32}$")

    assert any("invalid task_id" in e for e in errors)


def test_local_profile_accepts_hex_ids_and_absolute_local_paths():
    """Some batches ship `task_<32hex>` ids and absolute staged paths, and an
    empty `sector` — all valid under a profile that declares them."""
    rows = [_row("task_" + "b" * 32, ["/shared/data/x.xlsx"], ["x.xlsx"])]
    errors, _ = validate_input_rows(
        rows,
        expected_count=1,
        task_id_pattern=r"task_[0-9a-f]{32}$",
        reference_mode="local",
    )

    assert errors == []


def test_local_mode_rejects_relative_paths_and_urls():
    """A relative path would resolve against an arbitrary CWD; an https URL in
    local mode means the profile is wrong."""
    pattern = r"task_[0-9a-f]{32}$"
    relative, _ = validate_input_rows(
        [_row("task_" + "c" * 32, ["data/x.xlsx"], ["x.xlsx"])],
        expected_count=1,
        task_id_pattern=pattern,
        reference_mode="local",
    )
    remote, _ = validate_input_rows(
        [_row("task_" + "d" * 32, ["https://example.com/x.xlsx"], ["x.xlsx"])],
        expected_count=1,
        task_id_pattern=pattern,
        reference_mode="local",
    )

    assert any("must be absolute" in e for e in relative)
    assert any("must be a local path" in e for e in remote)
