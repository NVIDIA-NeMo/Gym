# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from responses_api_agents.kernel_gym.app import KernelGymInstanceConfig


def test_instance_uses_instance_id_as_task_name() -> None:
    config = KernelGymInstanceConfig.model_construct(problem_info={"instance_id": "kernelbench::level1::1"})

    assert config.task_name == "kernelbench::level1::1"
