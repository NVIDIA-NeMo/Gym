# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from nemo_gym.dataset_metrics import load_dataset_metrics_hook
from resources_servers.nemo_user_sim.episode_contracts import UserSimTaskInput


def test_compute_task_metrics_reports_usersim_input_contract() -> None:
    hook = load_dataset_metrics_hook(Path(__file__).parents[1])
    assert hook is not None

    task_input = UserSimTaskInput.model_validate(
        {
            "resolved_row": {
                "locale": "en_US",
                "persona": {"first_name": "Morgan"},
                "probe_type": "general_open_ended",
                "theme": {"type": "local food", "description": "Find dinner."},
                "goal": "Find dinner.",
                "usersim_config": {"random_seed": 1042},
            },
            "role_request_params": {
                "assistant": {"input": []},
                "judge": {"input": []},
                "tool_simulation": {"input": []},
            },
        }
    ).model_dump(mode="json")
    metrics = hook(task_input)

    assert metrics == {
        "UserSim locales": "en_US",
        "UserSim probe types": "general_open_ended",
        "UserSim sampling seeds": "1042",
        "UserSim user Responses override coverage": False,
        "UserSim assistant Responses override coverage": True,
        "UserSim judge Responses override coverage": True,
        "UserSim summary Responses override coverage": False,
        "UserSim tool_simulation Responses override coverage": True,
    }
