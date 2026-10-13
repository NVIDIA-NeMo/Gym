# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from nemo_gym.task_materialization import materialize_task
from resources_servers.nemo_user_sim.app import UserSimResourcesServerConfig, _row_digest, _validate_resolved_row
from resources_servers.nemo_user_sim.episode_contracts import UserSimEpisodeRequest


DATA_DIR = Path(__file__).parents[1] / "data"


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_examples_are_safe_task_rows_that_materialize_as_episode_requests() -> None:
    rows = _jsonl(DATA_DIR / "example.jsonl")

    assert len(rows) == 5
    for index, row in enumerate(rows):
        task_id = row["task_id"]
        resolved_row = row["resolved_row"]
        persona = resolved_row["persona"]
        assert isinstance(task_id, str)
        assert task_id == resolved_row["trajectory_id"]
        assert persona["first_name"] == f"Example{index + 1}"
        assert persona["last_name"] == "User"
        assert persona["email_address"] == f"example{index + 1}@example.invalid"
        assert persona["national_id"] == "000-00-0000"
        assert "assets_dir" not in resolved_row["usersim_config"]

        task = materialize_task(row, taskset="nemo_user_sim:example", task_index=index)
        request = UserSimEpisodeRequest.model_validate(
            {"episode_id": {"rollout_id": f"{index}-0", "attempt": 0}, "task": task}
        )
        assert request.task.task_id.task_id == task_id
        expected_revision = UserSimResourcesServerConfig.model_fields["usersim_revision"].default
        expected_personas_version = UserSimResourcesServerConfig.model_fields["nemotron_personas_version"].default
        assert isinstance(expected_revision, str)
        assert isinstance(expected_personas_version, str)
        _validate_resolved_row(
            request.task.task_input.resolved_row,
            expected_revision=expected_revision,
            expected_personas_version=expected_personas_version,
        )
        provenance = json.loads(resolved_row["usersim_provenance"])
        assert provenance["nemotron_personas_version"] == "synthetic"


def test_rollouts_match_examples_and_do_not_duplicate_episode_evidence() -> None:
    examples = _jsonl(DATA_DIR / "example.jsonl")
    rollouts_path = DATA_DIR / "example_rollouts.jsonl"
    rollouts = _jsonl(rollouts_path)

    assert len(rollouts) == len(examples) == 5
    assert rollouts_path.stat().st_size < 1_000_000
    for index, (example, rollout) in enumerate(zip(examples, rollouts, strict=True)):
        assert rollout["_ng_task_index"] == index
        assert rollout["_ng_task_id"] == {
            "taskset": "nemo_user_sim:example",
            "task_id": example["task_id"],
        }
        assert rollout["mask_sample"] is False
        assert isinstance(rollout["reward"], float)

        verification = rollout["verification"]
        assert verification["usersim_result"] is None
        assert not {"invocations", "resolved_row", "usersim_result"} & verification["verifier_data"].keys()
        assert verification["verifier_data"]["resolved_row_sha256"] == _row_digest(example["resolved_row"])
        provenance = json.loads(rollout["usersim_result"]["usersim_provenance"])
        assert provenance["nemotron_personas_version"] == "synthetic"
