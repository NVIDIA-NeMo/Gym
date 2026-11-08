# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ceiling extensions preserve saved actor requests and allow answer completion."""

import json
from uuid import uuid4

import pytest

from benchmarks.partner_if.tests.fixture_data import SOURCE
from responses_api_agents.ngc_recorded_context.app import resume_settings_changes
from responses_api_agents.ngc_recorded_context.replay import ReplayConfig, execute


def config(tmp_path):
    return ReplayConfig(
        package_dir=SOURCE / "assets",
        artifacts_dir=tmp_path,
        coverage_profile="expanded-v4",
        max_new_turns=8,
        max_policy_calls=16,
        max_delegate_turns=12,
    )


def test_explicit_extension_records_both_ceilings_without_changing_actor_settings(tmp_path):
    old = config(tmp_path)
    extended = old.model_copy(update={"max_new_turns": 24, "max_policy_calls": 40})
    assert resume_settings_changes(old.model_dump(), extended) == {
        "max_new_turns": {"from": 8, "to": 24},
        "max_policy_calls": {"from": 16, "to": 40},
    }
    assert resume_settings_changes(old.model_dump(), old) == {}


@pytest.mark.parametrize(
    "key,value",
    [
        ("coverage_profile", "recorded-v1"),
        ("temperature", 0.0),
        ("top_p", 0.9),
        ("max_tokens", 4096),
        ("timeout_s", 90),
    ],
)
def test_resume_refuses_silent_actor_or_sampling_changes(tmp_path, key, value):
    old = config(tmp_path)
    with pytest.raises(ValueError, match="actor execution setting"):
        resume_settings_changes(old.model_dump(), old.model_copy(update={key: value}))


@pytest.mark.parametrize("key,value", [("max_new_turns", 7), ("max_policy_calls", 15), ("max_delegate_turns", 11)])
def test_resume_refuses_smaller_execution_ceilings(tmp_path, key, value):
    old = config(tmp_path)
    with pytest.raises(ValueError, match="decreases or corrupts"):
        resume_settings_changes(old.model_dump(), old.model_copy(update={key: value}))


@pytest.mark.asyncio
async def test_parent_can_finish_after_the_former_eight_response_limit(tmp_path):
    settings = config(tmp_path).model_copy(update={"max_new_turns": 12, "max_policy_calls": 20})
    source = SOURCE / "tasks/ExampleProject/pif_029.json"
    task = json.loads(source.read_text())
    count = 0

    async def provider(payload, timeout):
        nonlocal count
        count += 1
        if count < 10:
            name, arguments = "Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md", "limit": 1}
        else:
            name, arguments = (
                "Write",
                {
                    "file_path": "/workspace/logs/answer.txt",
                    "content": "Investigation draft produced after nine reads.",
                },
            )
        return 200, {
            "id": uuid4().hex,
            "model": "test",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": uuid4().hex,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                        ],
                    },
                }
            ],
        }

    root, _, _, diagnostics = await execute(
        settings, task["responses_create_params"], model="test", secret="", request_api=provider
    )
    assert diagnostics["evaluation_completed"] and diagnostics["termination"] == "first_answer_draft"
    assert count == diagnostics["model_calls"] == 10
    assert json.loads((root / "drafts.json").read_text())["/workspace/logs/answer.txt"].endswith("nine reads.")
