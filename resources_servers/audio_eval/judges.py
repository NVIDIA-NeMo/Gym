# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge requests and replies for the judged audio profiles.

The calls themselves go through the server's ``judge_model_server`` (see ``app.py``); this module builds the chat
requests and validates the replies. A reply that is unusable (truncated, empty) raises ``JudgeError`` so the row is
masked and retried instead of being scored as a wrong answer.

A profile is judged in one of two ways. A ``judge_prompt_template`` in its contract selects the template judge: the
template is filled from the row and the judge's text is stored as ``judgement``. A scorer whose judge call is not a
prompt template (a structured verifier, say) defines a module-level ``JUDGE`` in its evaluator module (see
:class:`JudgeHooks`); it is used for every profile that names that evaluator.
"""

from __future__ import annotations

from typing import Any, Protocol

from nemo_gym.core import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.verification import JudgeError


TEMPLATE_JUDGE = "template"


class JudgeHooks(Protocol):
    """What an evaluator module's ``JUDGE`` object provides for a judged profile."""

    def validate(self, settings: dict[str, Any]) -> None:
        """Raise ``ValueError`` when the profile's ``++eval_config.*`` settings cannot be judged."""

    def request(self, row: dict) -> NeMoGymChatCompletionCreateParamsNonStreaming:
        """The chat request for one scored row."""

    def fields(self, completion: NeMoGymChatCompletion, row: dict) -> dict[str, Any]:
        """The row fields the judge reply contributes (raise ``JudgeError`` for an unusable reply)."""


def judge_kind(contract: dict, hooks: JudgeHooks | None, settings: dict[str, Any]) -> str | None:
    """Which judge a profile needs, or None when its scoring is judge-free.

    ``hooks`` is the ``JUDGE`` object of the profile's evaluator module, if it defines one.
    """
    if "judge_prompt_template" in contract:
        return TEMPLATE_JUDGE
    if hooks is not None:
        hooks.validate(settings)
        return "hooks"
    if contract["skills"].get("JUDGE_PIPELINE_ARGS"):
        raise ValueError("A judged profile needs a judge_prompt_template")
    return None


def template_request(row: dict, contract: dict) -> NeMoGymChatCompletionCreateParamsNonStreaming:
    """The profile's judge prompt filled from the scored row, with the pinned sampling parameters."""
    template = contract["judge_prompt_template"]
    messages = []
    if "system" in template:
        messages.append({"role": "system", "content": template["system"].format(**row)})
    messages.append({"role": "user", "content": template["user"].format(**row)})
    return NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
        {
            "messages": messages,
            "seed": 0,
            "stop": None,
            "stream": False,
            "tools": None,
            "response_format": None,
            "logprobs": False,
            "top_logprobs": None,
            "max_completion_tokens": None,
            "temperature": 0.0,
            "top_p": 0.95,
        }
    )


def template_judgement(completion: NeMoGymChatCompletion) -> str:
    """The judge's text from a single, complete, non-empty choice."""
    if len(completion.choices) != 1 or completion.choices[0].finish_reason != "stop":
        raise JudgeError("Incomplete or invalid judge completion")
    judgement = completion.choices[0].message.content
    if not judgement:
        raise JudgeError("Empty judge completion")
    return judgement
