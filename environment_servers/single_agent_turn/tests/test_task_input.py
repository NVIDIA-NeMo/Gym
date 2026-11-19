# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The episode request validates the TaskInput that data tooling loads."""

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServer
from environment_servers.single_agent_turn.task_input import TaskInput
from nemo_gym.core import MaterializedTask


def test_the_episode_request_takes_the_task_input_modules_task_input() -> None:
    assert (
        SingleAgentTurnEnvironmentServer.request_model.model_fields["task"].annotation is MaterializedTask[TaskInput]
    )
