# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from nemo_gym.dataset_metrics import load_dataset_metrics_hook


def test_compute_task_metrics_reports_usersim_input_contract() -> None:
    hook = load_dataset_metrics_hook(Path(__file__).parents[1])
    assert hook is not None

    metrics = hook(
        {
            "scenario": {
                "locale": "en_US",
                "persona": {"first_name": "Morgan"},
                "probe_type": "general_open_ended",
                "theme": {"type": "local food", "description": "Find dinner."},
                "goal": "Find dinner.",
            },
            "usersim_context": {
                "locale": "en_US",
                "seed": 1042,
                "personas_dataset_version": "0.0.2",
                "personas_panel_sha256": "a" * 64,
                "usersim_revision": "b" * 40,
            },
            "responses_create_params": {
                "assistant": {"input": []},
                "judge": {"input": []},
            },
        }
    )

    assert metrics == {
        "UserSim locales": "en_US",
        "UserSim probe types": "general_open_ended",
        "UserSim sampling seeds": "1042",
        "UserSim user Responses override coverage": False,
        "UserSim assistant Responses override coverage": True,
        "UserSim judge Responses override coverage": True,
        "UserSim summary Responses override coverage": False,
    }
