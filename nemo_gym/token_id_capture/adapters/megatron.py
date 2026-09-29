# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light extraction adapter for Megatron Inference offloaded payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nemo_gym.token_id_capture.staging.media import (
    COMPACT_TOKEN_IDS_DELTA_FIELD,
    build_compact_token_ids_delta,
)


PREFIX_IDS_FIELD = "required_prefix_token_ids"
PROMPT_IDS_FIELD = "prompt_token_ids"
GENERATED_IDS_FIELD = "generated_token_ids"
GENERATED_LOGPROBS_FIELD = "generated_log_probs"
# Compact-space material for VLM calls. Megatron Inference's OffloadedRequestPayload
# carries the compact prompt (one token per image or video); the framework worker
# attaches ``compact_prev_len`` (the compact length of the parent chain it spliced in)
# before handing the payload to the capture core.
COMPACT_PROMPT_IDS_FIELD = "compact_prompt_token_ids"
COMPACT_PREV_LEN_FIELD = "compact_prev_len"


def _field(payload: Any, name: str) -> Any:
    """Read one field from a Megatron Inference payload object or an equivalent mapping."""
    if isinstance(payload, Mapping):
        return payload.get(name)
    return getattr(payload, name, None)


def _sequence(payload: Any, name: str) -> Sequence[Any]:
    value = _field(payload, name)
    if value is None:
        raise ValueError(f"Megatron offloaded payload carries no {name}")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Megatron offloaded payload field {name} must be a token-id sequence")
    return value


def _token_ids(payload: Any, name: str) -> list[int]:
    """Read a token-id field, rejecting anything but plain integers.

    Megatron Inference hands host-side ``list[int]`` values. A float, string, or
    bool element means the payload is malformed; ``int()`` would silently
    truncate or coerce it into a plausible-looking id.
    """
    values = _sequence(payload, name)
    if any(type(value) is not int for value in values):
        raise ValueError(f"Megatron offloaded payload field {name} must contain only integer token ids")
    return list(values)


def _log_probs(payload: Any, name: str) -> list[float]:
    """Read a log-probability field, rejecting non-numeric elements."""
    values = _sequence(payload, name)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"Megatron offloaded payload field {name} must contain only numeric log probabilities")
    return [float(value) for value in values]


class MegatronCaptureAdapter:
    """Translate Megatron inference request/response material at the framework boundary.

    Megatron inference offloads exact prompt ids, generated ids, and selected-token log
    probabilities as attributes on a payload object rather than as a chat
    completion dict. The adapter reads either shape so extraction failures
    flow through ``RolloutTokenCapture.complete_call_from_response`` and
    poison the call with ``capture_failed`` coordinates, the same outcome the
    vLLM adapter produces.
    """

    def enter_prefix(self, request_payload: dict[str, Any], prefix_ids: list[int]) -> dict[str, Any]:
        request_payload[PREFIX_IDS_FIELD] = list(prefix_ids)
        return request_payload

    def extract_prompt_ids(self, response_payload: Any) -> list[int]:
        return _token_ids(response_payload, PROMPT_IDS_FIELD)

    def extract_generation(self, response_payload: Any) -> tuple[list[int], list[float]]:
        token_ids = _token_ids(response_payload, GENERATED_IDS_FIELD)
        log_probs = _log_probs(response_payload, GENERATED_LOGPROBS_FIELD)
        if len(token_ids) != len(log_probs):
            raise ValueError(
                f"Megatron generated token and log-probability lengths differ: {len(token_ids)} != {len(log_probs)}"
            )
        return token_ids, log_probs

    def extract_extras(self, response_payload: Any) -> dict[str, Any] | None:
        """Stage the compact-space delta for a call whose engine expanded media tokens.

        Megatron Inference routed-experts rows are total_tokens - 1 long while the
        staging contract is delta-token aligned. Until that shift has a first-class
        representation the adapter stages no routing extras. Media tensors do not
        travel in extras: the framework passes them to
        ``complete_call_from_response`` as ``attachments``.
        """
        compact_prompt = _field(response_payload, COMPACT_PROMPT_IDS_FIELD)
        if compact_prompt is None:
            return None
        compact_prev_len = _field(response_payload, COMPACT_PREV_LEN_FIELD)
        if compact_prev_len is None:
            compact_prev_len = 0
        if type(compact_prev_len) is not int:
            raise ValueError("Megatron offloaded payload compact_prev_len must be an int")
        return {
            COMPACT_TOKEN_IDS_DELTA_FIELD: build_compact_token_ids_delta(
                _token_ids(response_payload, COMPACT_PROMPT_IDS_FIELD),
                _token_ids(response_payload, GENERATED_IDS_FIELD),
                compact_prev_len=compact_prev_len,
            )
        }
