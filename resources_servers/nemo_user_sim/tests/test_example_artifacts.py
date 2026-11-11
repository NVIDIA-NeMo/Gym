# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path

from environments.nemo_user_sim.prepare import DEFAULT_LOCALES
from nemo_gym.data import materialize_task
from resources_servers.nemo_user_sim.app import (
    UserSimResourcesServerConfig,
    _row_digest,
    _validate_resolved_row,
    installed_usersim_revision,
)
from resources_servers.nemo_user_sim.task_data import TaskData


DATA_DIR = Path(__file__).parents[1] / "data"


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_examples_are_safe_task_rows_that_materialize_as_task_data() -> None:
    rows = _jsonl(DATA_DIR / "example.jsonl")

    assert len(rows) == 5
    # Each example covers a different probe, and together they cover every default preparation locale.
    assert len({row["resolved_row"]["probe_type"] for row in rows}) == 5
    assert {row["resolved_row"]["locale"] for row in rows} == set(DEFAULT_LOCALES)
    for index, row in enumerate(rows):
        task_id = row["task_id"]
        resolved_row = row["resolved_row"]
        persona = resolved_row["persona"]
        assert isinstance(task_id, str)
        assert task_id == resolved_row["trajectory_id"]
        assert re.fullmatch(r"Example\d+", persona["first_name"])
        assert persona["last_name"] == "User"
        assert persona["email_address"] == f"{persona['first_name'].lower()}@example.invalid"
        assert persona["locale"] == resolved_row["locale"]
        assert persona["national_id"] == "000-00-0000"
        assert "assets_dir" not in resolved_row["usersim_config"]

        task = materialize_task(row, taskset="nemo_user_sim:example", task_index=index)
        assert task["task_id"] == {"taskset": "nemo_user_sim:example", "task_id": task_id}
        task_input = TaskData.model_validate(task["task_input"])
        expected_revision = installed_usersim_revision()
        expected_personas_versions = UserSimResourcesServerConfig.model_fields[
            "nemotron_personas_versions"
        ].default_factory()
        assert isinstance(expected_revision, str)
        _validate_resolved_row(
            task_input.resolved_row,
            expected_revision=expected_revision,
            expected_personas_versions=expected_personas_versions,
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
