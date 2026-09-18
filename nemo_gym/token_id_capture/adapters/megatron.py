# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light extraction adapter for Megatron Inference offloaded payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nemo_gym.token_id_capture.staging.media import (
    build_compact_token_ids_delta,
    build_multimodal_extras,
)


PREFIX_IDS_FIELD = "required_prefix_token_ids"
PROMPT_IDS_FIELD = "prompt_token_ids"
GENERATED_IDS_FIELD = "generated_token_ids"
GENERATED_LOGPROBS_FIELD = "generated_log_probs"
# Multimodal material. MInf's OffloadedRequestPayload carries the compact prompt; the
# framework worker attaches ``compact_prev_len`` (the compact length of the parent
# chain it spliced in) and ``media`` (the geometry it read off the engine's media
# tensors) before handing the payload to the capture core.
COMPACT_PROMPT_IDS_FIELD = "compact_prompt_token_ids"
COMPACT_PREV_LEN_FIELD = "compact_prev_len"
MEDIA_FIELD = "media"
_MEDIA_KEYS = ("modality", "imgs_sizes", "num_frames", "num_tiles")


def _field(payload: Any, name: str) -> Any:
    """Read one field from an MInf payload object or an equivalent mapping."""
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


class MegatronCaptureAdapter:
    """Translate MInf request/response material at the framework boundary.

    MInf offloads exact prompt ids, generated ids, and selected-token log
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
        return [int(token_id) for token_id in _sequence(response_payload, PROMPT_IDS_FIELD)]

    def extract_generation(self, response_payload: Any) -> tuple[list[int], list[float]]:
        token_ids = [int(token_id) for token_id in _sequence(response_payload, GENERATED_IDS_FIELD)]
        log_probs = [float(value) for value in _sequence(response_payload, GENERATED_LOGPROBS_FIELD)]
        if len(token_ids) != len(log_probs):
            raise ValueError(
                f"Megatron generated token and log-probability lengths differ: {len(token_ids)} != {len(log_probs)}"
            )
        return token_ids, log_probs

    def extract_extras(self, response_payload: Any) -> dict[str, Any] | None:
        """Stage the multimodal envelope for a call that carried media.

        MInf routed-experts rows are total_tokens - 1 long while the staging
        contract is delta-token aligned. Until that shift has a first-class
        representation the adapter stages no routing extras.
        """
        media = _field(response_payload, MEDIA_FIELD)
        if media is None:
            return None
        compact_prompt = _field(response_payload, COMPACT_PROMPT_IDS_FIELD)
        compact_delta = None
        if compact_prompt is not None:
            compact_prev_len = _field(response_payload, COMPACT_PREV_LEN_FIELD)
            if compact_prev_len is None:
                compact_prev_len = 0
            if type(compact_prev_len) is not int:
                raise ValueError("Megatron offloaded payload compact_prev_len must be an int")
            compact_delta = build_compact_token_ids_delta(
                _sequence(response_payload, COMPACT_PROMPT_IDS_FIELD),
                _sequence(response_payload, GENERATED_IDS_FIELD),
                compact_prev_len=compact_prev_len,
            )
        return build_multimodal_extras(
            compact_token_ids_delta=compact_delta,
            media=_media_mapping(media),
        )


def _media_mapping(media: Any) -> dict[str, Any]:
    """Read the media geometry (mapping or attribute object) into staging media keys."""
    if isinstance(media, Mapping):
        return {name: media[name] for name in _MEDIA_KEYS if name in media}
    return {name: getattr(media, name) for name in _MEDIA_KEYS if hasattr(media, name)}
