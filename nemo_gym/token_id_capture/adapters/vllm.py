# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light extraction adapter for vLLM chat completions."""

from __future__ import annotations

from typing import Any


PREFIX_IDS_FIELD = "required_prefix_token_ids"
PROMPT_IDS_FIELD = "prompt_token_ids"
ROUTED_EXPERTS_FIELD = "routed_experts"


def _message(choice: dict[str, Any]) -> dict[str, Any]:
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        raise ValueError("vLLM response choice.message must be an object")
    return message


def _single_choice(response_payload: dict[str, Any]) -> dict[str, Any]:
    choices = response_payload.get("choices") or []
    if len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError(f"token capture requires exactly one object choice, got {len(choices)}")
    return choices[0]


def extract_generation_token_info(choice: dict[str, Any]) -> tuple[list[int], list[float]]:
    """Read exact generation IDs/log probabilities from supported vLLM shapes."""
    message = _message(choice)
    if "generation_token_ids" in message and "generation_log_probs" in message:
        raw_ids = message["generation_token_ids"]
        raw_log_probs = message["generation_log_probs"]
    else:
        content_log_probs = (choice.get("logprobs") or {}).get("content")
        if content_log_probs is None:
            raise ValueError("vLLM response contained neither message token fields nor choice.logprobs.content")
        raw_ids = [item["token"] for item in content_log_probs]
        raw_log_probs = [item["logprob"] for item in content_log_probs]
    token_ids = [int(str(token_id).removeprefix("token_id:")) for token_id in raw_ids]
    log_probs = [float(value) for value in raw_log_probs]
    if len(token_ids) != len(log_probs):
        raise ValueError(f"generated token and log-probability lengths differ: {len(token_ids)} != {len(log_probs)}")
    return token_ids, log_probs


class VLLMCaptureAdapter:
    """Translate vLLM request/response payloads at the framework boundary."""

    def enter_prefix(self, request_payload: dict[str, Any], prefix_ids: list[int]) -> dict[str, Any]:
        request_payload[PREFIX_IDS_FIELD] = list(prefix_ids)
        return request_payload

    def extract_prompt_ids(self, response_payload: dict[str, Any]) -> list[int]:
        prompt_ids = response_payload.get(PROMPT_IDS_FIELD)
        if prompt_ids is None:
            prompt_ids = _message(_single_choice(response_payload)).get(PROMPT_IDS_FIELD)
        if prompt_ids is None:
            raise ValueError("vLLM response carries no prompt_token_ids")
        return [int(token_id) for token_id in prompt_ids]

    def extract_generation(self, response_payload: dict[str, Any]) -> tuple[list[int], list[float]]:
        return extract_generation_token_info(_single_choice(response_payload))

    def extract_extras(self, response_payload: dict[str, Any]) -> dict[str, Any] | None:
        routed_experts = _message(_single_choice(response_payload)).get(ROUTED_EXPERTS_FIELD)
        if routed_experts is None:
            return None
        if not isinstance(routed_experts, (str, dict, list)):
            raise ValueError("vLLM routed_experts must use a JSON-compatible envelope")
        return {ROUTED_EXPERTS_FIELD: routed_experts}
