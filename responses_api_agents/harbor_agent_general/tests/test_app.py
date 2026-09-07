# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from harbor.models.trajectories import Trajectory

from responses_api_agents.harbor_agent_general.app import HarborAgent, HarborVerifyResponse


def _trajectory(**step_updates) -> Trajectory:
    step = {
        "step_id": 1,
        "source": "agent",
        "message": "answer",
    }
    step.update(step_updates)
    return Trajectory.model_validate(
        {
            "agent": {"name": "test-agent", "version": "1"},
            "steps": [step],
        }
    )


def test_prompt_token_ids_do_not_emit_training_metadata() -> None:
    warnings = []
    output = HarborAgent.convert_atif_to_gym_responses(_trajectory(metrics={"prompt_token_ids": [1, 2]}), warnings)

    assert "prompt_token_ids" not in output[0]
    assert "generation_token_ids" not in output[0]
    assert "generation_log_probs" not in output[0]
    assert "completion_token_ids are missing or empty" in warnings[0]


def test_mismatched_completion_metadata_is_omitted_with_warning() -> None:
    warnings = []
    trajectory = _trajectory(metrics={"prompt_token_ids": [0], "completion_token_ids": [1, 2], "logprobs": [-0.1]})

    output = HarborAgent.convert_atif_to_gym_responses(trajectory, warnings)

    assert "generation_token_ids" not in output[0]
    assert "completion_token_ids and logprobs have different lengths" in warnings[0]


def test_completion_token_ids_emit_aligned_training_metadata() -> None:
    output = HarborAgent.convert_atif_to_gym_responses(
        _trajectory(
            metrics={
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3, 4],
                "logprobs": [-0.1, -0.2],
            }
        )
    )

    assert output[0]["prompt_token_ids"] == [1, 2]
    assert output[0]["generation_token_ids"] == [3, 4]
    assert output[0]["generation_log_probs"] == [-0.1, -0.2]


def test_multimodal_message_is_serialized_with_warning() -> None:
    warnings = []
    output = HarborAgent.convert_atif_to_gym_responses(
        _trajectory(
            message=[
                {"type": "text", "text": "look"},
                {"type": "image", "source": {"media_type": "image/png", "path": "/tmp/image.png"}},
            ]
        ),
        warnings,
    )

    assert json.loads(output[0]["content"][0]["text"]) == [
        {"type": "text", "text": "look"},
        {"type": "image", "source": {"media_type": "image/png", "path": "/tmp/image.png"}},
    ]
    assert "multimodal message serialized as JSON" in warnings[0]


def test_multimodal_tool_output_uses_gym_content_parts() -> None:
    output = HarborAgent.convert_atif_to_gym_responses(
        _trajectory(
            tool_calls=[{"tool_call_id": "call-1", "function_name": "inspect", "arguments": {}}],
            observation={
                "results": [
                    {
                        "source_call_id": "call-1",
                        "content": [
                            {"type": "text", "text": "result"},
                            {
                                "type": "image",
                                "source": {"media_type": "image/png", "path": "https://example.com/image.png"},
                            },
                        ],
                    }
                ]
            },
        )
    )

    assert output[2]["output"] == [
        {"text": "result", "type": "input_text"},
        {"detail": "auto", "image_url": "https://example.com/image.png", "type": "input_image"},
    ]


def test_local_tool_image_and_missing_call_id_are_documented() -> None:
    warnings = []
    output = HarborAgent.convert_atif_to_gym_responses(
        _trajectory(
            observation={
                "results": [
                    {
                        "content": [
                            {
                                "type": "image",
                                "source": {"media_type": "image/png", "path": "/tmp/image.png"},
                            }
                        ]
                    }
                ]
            }
        ),
        warnings,
    )

    assert json.loads(output[1]["output"])[0]["source"]["path"] == "/tmp/image.png"
    assert output[1]["call_id"] == "atif-step-1-observation-0"
    assert any("local image paths" in warning for warning in warnings)
    assert any("synthetic call_id" in warning for warning in warnings)


def test_copied_context_training_metadata_is_omitted_with_warning() -> None:
    warnings = []
    output = HarborAgent.convert_atif_to_gym_responses(
        _trajectory(
            is_copied_context=True,
            metrics={
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3, 4],
                "logprobs": [-0.1, -0.2],
            },
        ),
        warnings,
    )

    assert "generation_token_ids" not in output[0]
    assert "step is copied context" in warnings[0]


def test_user_step_is_preserved_without_warning() -> None:
    warnings = []

    output = HarborAgent.convert_atif_to_gym_responses(_trajectory(source="user"), warnings)

    assert output == [{"content": "answer", "role": "user", "type": "message", "phase": None}]
    assert warnings == []


def test_system_multimodal_step_uses_gym_content_parts() -> None:
    warnings = []
    output = HarborAgent.convert_atif_to_gym_responses(
        _trajectory(
            source="system",
            message=[
                {"type": "text", "text": "inspect"},
                {
                    "type": "image",
                    "source": {"media_type": "image/png", "path": "data:image/png;base64,YQ=="},
                },
            ],
        ),
        warnings,
    )

    assert output[0]["role"] == "system"
    assert output[0]["content"] == [
        {"text": "inspect", "type": "input_text"},
        {"detail": "auto", "image_url": "data:image/png;base64,YQ==", "type": "input_image"},
    ]
    assert warnings == []


def test_verify_response_preserves_conversion_diagnostics() -> None:
    response = HarborVerifyResponse.model_validate(
        {
            "responses_create_params": {"input": []},
            "response": {
                "id": "response-id",
                "created_at": 0,
                "model": "test-model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "status": "completed",
            },
            "reward": 1.0,
            "atif_conversion": {"lossless": False, "warnings": ["documented limitation"]},
        }
    )

    assert response.model_dump()["atif_conversion"] == {
        "lossless": False,
        "warnings": ["documented limitation"],
    }
