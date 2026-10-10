# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from nemo_gym.episode_types import MaterializedTask
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput


ROOT = Path(__file__).resolve().parents[3]


def test_native_counter_tasks_have_explicit_identity_and_verifier_state() -> None:
    path = ROOT / "resources_servers/example_session_state_mgmt/data/example_nooa_native.jsonl"
    for line in path.read_text().splitlines():
        row = MaterializedTask[SingleAgentTurnTaskInput].model_validate_json(line)
        assert row.task_id.taskset == "nooa-counter"
        assert row.task_input.task_data["expected_count"] > row.task_input.task_data["initial_count"]
        assert len(row.task_input.responses_create_params.tools) == 2
