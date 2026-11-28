# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VoiceCodeBench scoring of one recording: the entity-verifier judge request and the per-row measurements.

The WER lattice and the verifier contract are the vendored official implementation (``official.py``). The verifier
request is sent by the resources server through its judge model server; this module only builds the request, parses
the reply and turns it into the row fields that ``VoiceCodeBenchMetrics`` aggregates.
"""

from __future__ import annotations

import json
from typing import Any

from nemo_gym.core import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.verification import JudgeError
from resources_servers.audio_eval.voice_code_bench.official import (
    DEFAULT_VERIFIER_ID,
    FORMAT_INVARIANT_WER_REFERENCE,
    build_openai_verify_request,
    compute_format_invariant_wer,
    load_verifier_config,
    parse_llm_entity_matches,
)


def verifier_request(datapoint: dict[str, Any], generation: str) -> NeMoGymChatCompletionCreateParamsNonStreaming:
    """The official verifier request; the judge model server supplies the model."""
    request = build_openai_verify_request(load_verifier_config(DEFAULT_VERIFIER_ID), datapoint, generation)
    request.pop("model")
    return NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(request)


def parse_verifier_reply(completion: NeMoGymChatCompletion, datapoint: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-entity matches from a verifier reply; an unusable verdict is a judge failure, not a wrong answer."""
    if len(completion.choices) != 1 or completion.choices[0].finish_reason != "stop":
        raise JudgeError(f"Incomplete or invalid VoiceCodeBench verifier completion for {datapoint['audio_id']}")
    try:
        content = completion.choices[0].message.content
        return parse_llm_entity_matches(json.loads(content), datapoint["entities"])
    except (IndexError, TypeError, ValueError) as error:
        raise JudgeError(f"Invalid VoiceCodeBench verifier reply for {datapoint['audio_id']}: {error}") from error


def score_recording(datapoint: dict[str, Any], generation: str, matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Row fields for one recording: WER counts and CTEM/TSR counts from the verifier's entity matches."""
    wer = compute_format_invariant_wer(datapoint, generation)
    config = load_verifier_config(DEFAULT_VERIFIER_ID)
    correct_by_type: dict[str, int] = {}
    total_by_type: dict[str, int] = {}
    for entity, match in zip(datapoint["entities"], matches, strict=True):
        total_by_type[entity["type"]] = total_by_type.get(entity["type"], 0) + 1
        correct_by_type[entity["type"]] = correct_by_type.get(entity["type"], 0) + int(match["present"])
    correct = sum(correct_by_type.values())
    total = len(datapoint["entities"])
    return {
        "voice_code_wer_reference": FORMAT_INVARIANT_WER_REFERENCE,
        "voice_code_wer_errors": wer.substitutions + wer.deletions + wer.insertions,
        "voice_code_wer_ref_words": wer.reference_words,
        "voice_code_wer_hyp_words": wer.hypothesis_words,
        "voice_code_wer_substitutions": wer.substitutions,
        "voice_code_wer_deletions": wer.deletions,
        "voice_code_wer_insertions": wer.insertions,
        "voice_code_entity_verifier_status": "complete",
        "voice_code_verifier_id": config.id,
        "voice_code_verifier_model": config.model,
        "voice_code_verifier_config_digest": config.digest,
        "voice_code_entity_matches": matches,
        "voice_code_correct_entities": correct,
        "voice_code_total_entities": total,
        "voice_code_task_success": correct == total,
        "voice_code_correct_by_type": correct_by_type,
        "voice_code_total_by_type": total_by_type,
        "is_correct": correct == total,
    }
