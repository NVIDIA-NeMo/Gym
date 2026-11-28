# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VoiceCodeBench evaluator: official WER plus the entity matches the resources server obtained from its judge.

The module-level ``JUDGE`` is how the ``audio_eval`` server finds this scorer's judge (see
``resources_servers.audio_eval.judges.JudgeHooks``): the packaged entity verifier, called once per recording.
"""

from typing import Any

from nemo_gym.core import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from resources_servers.audio_eval._scoring.evaluation.evaluator.base import BaseEvaluator
from resources_servers.audio_eval.voice_code_bench.official import DEFAULT_VERIFIER_ID
from resources_servers.audio_eval.voice_code_bench.scoring import (
    parse_verifier_reply,
    score_recording,
    verifier_request,
)


# ``++eval_config.*`` keys a profile of this evaluator may set (``cleanup_generation_loops`` is read by the server).
_SETTINGS = ("verifier_id", "cleanup_generation_loops")


class EntityVerifier:
    """The judge hooks of the ``audio_eval`` server for the packaged entity verifier."""

    def validate(self, settings: dict[str, Any]) -> None:
        unknown = sorted(set(settings) - set(_SETTINGS))
        if unknown:
            raise ValueError(f"Unrecognized VoiceCodeBench eval_config keys {unknown}; known: {list(_SETTINGS)}")
        if settings.get("verifier_id", DEFAULT_VERIFIER_ID) != DEFAULT_VERIFIER_ID:
            raise ValueError(f"The only packaged entity verifier is {DEFAULT_VERIFIER_ID}")

    def request(self, row: dict) -> NeMoGymChatCompletionCreateParamsNonStreaming:
        return verifier_request(row["voice_code_bench"], row["generation"].strip())

    def fields(self, completion: NeMoGymChatCompletion, row: dict) -> dict[str, Any]:
        return {"voice_code_entity_matches": parse_verifier_reply(completion, row["voice_code_bench"])}


JUDGE = EntityVerifier()


class VoiceCodeBenchEvaluator(BaseEvaluator):
    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        generation = str(data_point.get("generation") or "").strip()
        return score_recording(data_point["voice_code_bench"], generation, data_point["voice_code_entity_matches"])
