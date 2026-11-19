# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The episode request validates the TaskInput that collation and ``gym env validate`` load."""

from environment_servers.swe_together.app import SWETEnvironmentServer
from environment_servers.swe_together.task_input import TaskInput
from nemo_gym.core.episode_types import MaterializedTask


def test_the_episode_request_takes_the_task_input_modules_task_input() -> None:
    assert SWETEnvironmentServer.request_model.model_fields["task"].annotation is MaterializedTask[TaskInput]
