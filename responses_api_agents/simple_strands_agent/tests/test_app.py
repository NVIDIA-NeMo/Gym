# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseFunctionToolCallForTraining,
)
from responses_api_agents.simple_strands_agent.app import (
    _extract_instruction,
    trajectory_to_output_items,
)
from responses_api_agents.simple_strands_agent.setup_ssa import _package_dir


def test_extract_instruction() -> None:
    instruction, system = _extract_instruction(
        [
            NeMoGymEasyInputMessage(role="system", content="Be precise"),
            NeMoGymEasyInputMessage(role="developer", content="Use tools"),
            NeMoGymEasyInputMessage(role="user", content="Old task"),
            NeMoGymEasyInputMessage(role="user", content="Solve this"),
        ]
    )
    assert instruction == "Solve this"
    assert system == "Be precise\n\nUse tools"


def test_trajectory_preserves_reasoning_tools_and_training() -> None:
    messages = [
        {"role": "user", "content": [{"text": "Solve"}]},
        {
            "role": "assistant",
            "content": [
                {"reasoningContent": {"reasoningText": {"text": "I should calculate."}}},
                {
                    "toolUse": {
                        "toolUseId": "call-1",
                        "name": "bash",
                        "input": {"command": "printf 42"},
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "call-1",
                        "status": "success",
                        "content": [{"text": "42"}],
                    }
                }
            ],
        },
        {"role": "assistant", "content": [{"text": "\\boxed{42}"}]},
    ]
    training_turns = [
        {
            "prompt_token_ids": [1],
            "generation_token_ids": [2],
            "generation_log_probs": [-0.1],
        },
        {},
    ]

    output = trajectory_to_output_items(messages, training_turns)

    assert [item.type for item in output] == [
        "reasoning",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert isinstance(output[1], NeMoGymResponseFunctionToolCallForTraining)
    assert output[1].prompt_token_ids == [1]
    assert output[2].output == "42"
    assert output[3].content[0].text == "\\boxed{42}"


def test_package_dir_accepts_monorepo_or_package(tmp_path: Path) -> None:
    package = tmp_path / "simple-strands-agent"
    (package / "src" / "ssa").mkdir(parents=True)
    (package / "pyproject.toml").touch()
    assert _package_dir(tmp_path) == package.resolve()
    assert _package_dir(package) == package.resolve()


def test_package_dir_rejects_missing_package(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="package not found"):
        _package_dir(tmp_path)
