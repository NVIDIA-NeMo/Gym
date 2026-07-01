# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Offline unit tests for the generic VLMEvalKit audio driver.

These exercise ``verify()``'s generic scorer with canned model outputs — no
model, no audio, no ``vlmeval`` install required (the option extractor falls
back to a local implementation when ``vlmeval`` is absent).
"""

from unittest.mock import MagicMock

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.vlm_eval_kit_audio.app import (
    VlmEvalKitAudioResourcesServer,
    VlmEvalKitAudioResourcesServerConfig,
    VlmEvalKitAudioVerifyRequest,
    infer_option,
    strip_think,
)

MINIMAL_PARAMS = {"input": [{"role": "user", "content": "test"}], "parallel_tool_calls": True}


def _server() -> VlmEvalKitAudioResourcesServer:
    config = VlmEvalKitAudioResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return VlmEvalKitAudioResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_1",
                "role": "assistant",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _req(text: str, answer, dataset_type: str = "QA", choices=None) -> VlmEvalKitAudioVerifyRequest:
    return VlmEvalKitAudioVerifyRequest(
        responses_create_params=MINIMAL_PARAMS,
        response=_response(text),
        answer=answer,
        dataset_type=dataset_type,
        choices=choices or {},
        benchmark_name="MMAU_test",
    )


class TestHelpers:
    def test_strip_think_removes_reasoning(self) -> None:
        assert strip_think("<think>A or B or C</think>\nFinal: D") == "Final: D"
        assert strip_think("no reasoning here") == "no reasoning here"
        assert strip_think("") == ""

    def test_infer_option_bare_letter(self) -> None:
        assert infer_option("B", {}) == "B"

    def test_infer_option_phrase(self) -> None:
        assert infer_option("The answer is C.", {}) == "C"

    def test_infer_option_none_when_absent(self) -> None:
        assert infer_option("no option letter here at all", {}) is None


class TestVerify:
    def test_sanity(self) -> None:
        assert _server() is not None

    async def test_mcq_bare_letter_correct(self) -> None:
        result = await _server().verify(_req("B", "B"))
        assert result.reward == 1.0
        assert result.extracted == "B"

    async def test_mcq_phrase_correct(self) -> None:
        result = await _server().verify(_req("The answer is C.", "C"))
        assert result.reward == 1.0

    async def test_mcq_wrong(self) -> None:
        result = await _server().verify(_req("A", "B"))
        assert result.reward == 0.0

    async def test_think_leak_is_stripped_before_scoring(self) -> None:
        # Without strip_think the scorer would latch onto 'A' inside the reasoning.
        text = "<think>Maybe A, could be B or C</think>\nFinal answer: D"
        result = await _server().verify(_req(text, "D"))
        assert result.reward == 1.0
        assert result.extracted == "D"

    async def test_empty_response_scores_zero(self) -> None:
        result = await _server().verify(_req("", "B"))
        assert result.reward == 0.0

    async def test_yorn(self) -> None:
        assert (await _server().verify(_req("Yes, it is.", "yes", dataset_type="Y/N"))).reward == 1.0
        assert (await _server().verify(_req("No.", "yes", dataset_type="Y/N"))).reward == 0.0

    async def test_vqa_containment(self) -> None:
        result = await _server().verify(_req("The tower is in Paris.", "paris", dataset_type="VQA"))
        assert result.reward == 1.0
