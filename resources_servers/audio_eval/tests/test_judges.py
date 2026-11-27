# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge selection, request construction and reply validation."""

import pytest

from nemo_gym.core import NeMoGymChatCompletion
from nemo_gym.verification import JudgeError
from resources_servers.audio_eval.judges import TEMPLATE_JUDGE, judge_kind, template_judgement, template_request


def completion(content, finish="stop", choices=1):
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "j",
            "created": 0,
            "model": "m",
            "object": "chat.completion",
            "choices": [
                {"index": i, "message": {"role": "assistant", "content": content}, "finish_reason": finish}
                for i in range(choices)
            ],
        }
    )


class Hooks:
    def __init__(self):
        self.validated = []

    def validate(self, settings):
        self.validated.append(settings)
        if settings.get("mode") == "replay":
            raise ValueError("mode must be live")


def test_judge_kind_selects_the_judge_from_the_profile():
    skills = {"METRICS_TYPE": "x"}
    hooks = Hooks()
    assert judge_kind({"skills": skills, "judge_prompt_template": {"user": "u"}}, None, {}) == TEMPLATE_JUDGE
    # A template wins over an evaluator's hooks: the hooks are not consulted.
    assert judge_kind({"skills": skills, "judge_prompt_template": {"user": "u"}}, hooks, {}) == TEMPLATE_JUDGE
    assert hooks.validated == []
    assert judge_kind({"skills": skills}, hooks, {"mode": "live"}) == "hooks"
    assert hooks.validated == [{"mode": "live"}]
    assert judge_kind({"skills": skills}, None, {}) is None


def test_judge_kind_rejects_unsupported_configurations():
    skills = {"METRICS_TYPE": "x"}
    with pytest.raises(ValueError, match="mode must be live"):
        judge_kind({"skills": skills}, Hooks(), {"mode": "replay"})
    with pytest.raises(ValueError, match="judge_prompt_template"):
        judge_kind({"skills": {"JUDGE_PIPELINE_ARGS": {"model": "m"}}}, None, {})


def test_template_request_fills_the_prompt_and_pins_sampling():
    contract = {"judge_prompt_template": {"system": "S {q}", "user": "U {q} {generation}"}}
    request = template_request({"q": "why", "generation": "because"}, contract).model_dump(exclude_unset=True)
    assert [m["content"] for m in request["messages"]] == ["S why", "U why because"]
    assert request["seed"] == 0 and request["temperature"] == 0.0 and request["top_p"] == 0.95
    assert "model" not in request
    user_only = template_request({"q": "a", "generation": "b"}, {"judge_prompt_template": {"user": "{q}"}})
    assert [m["role"] for m in user_only.messages] == ["user"]


def test_template_judgement_requires_one_complete_nonempty_choice():
    assert template_judgement(completion("CORRECTNESS: 5")) == "CORRECTNESS: 5"
    for bad in (completion("x", finish="length"), completion("x", choices=2), completion("")):
        with pytest.raises(JudgeError):
            template_judgement(bad)
