# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
import asyncio
from unittest.mock import MagicMock

from nemo_gym.base_resources_server import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.math_rlvr.app import (
    MathRlvrResourcesServer,
    MathRlvrResourcesServerConfig,
    MathRlvrVerifyRequest,
)


def _make_server() -> MathRlvrResourcesServer:
    config = MathRlvrResourcesServerConfig(
        host="0.0.0.0", port=8080, entrypoint="", name=""
    )
    return MathRlvrResourcesServer(
        config=config, server_client=MagicMock(spec=ServerClient)
    )


def _make_request(
    response_content: str,
    ground_truth: str,
    verifier_type: str,
    request_id: int = 1,
) -> MathRlvrVerifyRequest:
    response = NeMoGymResponse(
        id=f"resp_test_{request_id}",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": f"msg_test_{request_id}",
                "content": [
                    {
                        "annotations": [],
                        "text": response_content,
                        "type": "output_text",
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    return MathRlvrVerifyRequest(
        responses_create_params={"input": []},
        response=response,
        ground_truth=ground_truth,
        verifier_type=verifier_type,
    )


# ---------------------------------------------------------------------------
# math (math-verify-backed)
# ---------------------------------------------------------------------------


class TestMathVerifier:
    def test_sanity(self) -> None:
        _make_server()

    def test_simple_integer_match(self) -> None:
        req = _make_request("The answer is \\boxed{42}.", "42", "math")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
        assert result.verifier_type == "math"
        assert result.verification_failed is False

    def test_simple_integer_mismatch(self) -> None:
        req = _make_request("The answer is \\boxed{17}.", "42", "math")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0

    def test_equivalent_fraction(self) -> None:
        # 2/4 == 1/2 — math-verify should treat as equivalent.
        req = _make_request(
            "Therefore the answer is $\\boxed{\\frac{2}{4}}$.",
            "\\frac{1}{2}",
            "math",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_thinking_tags_stripped(self) -> None:
        req = _make_request(
            "<think>maybe \\boxed{17}</think>\n\\boxed{42}",
            "42",
            "math",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_unclosed_think_tag_returns_zero_no_failure(self) -> None:
        req = _make_request(
            "<think>still thinking and \\boxed{42}",
            "42",
            "math",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False


# ---------------------------------------------------------------------------
# math500 (Hendrycks MATH boxed + is_equiv)
# ---------------------------------------------------------------------------


class TestMath500Verifier:
    def test_boxed_match(self) -> None:
        req = _make_request(
            "Working it out, \\boxed{120}.",
            "\\boxed{120}",
            "math500",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
        assert result.extracted_answer == "120"

    def test_boxed_mismatch(self) -> None:
        req = _make_request(
            "I think \\boxed{121}.",
            "\\boxed{120}",
            "math500",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0

    def test_no_boxed_in_response_returns_zero(self) -> None:
        req = _make_request(
            "I cannot compute this.",
            "\\boxed{120}",
            "math500",
        )
        result = asyncio.run(_make_server().verify(req))
        # last_boxed_only_string returns None on the response; remove_boxed
        # propagates None; is_equiv(None, '120') -> False.
        assert result.reward == 0.0
        assert result.verification_failed is False

    def test_is_equiv_normalizes_whitespace(self) -> None:
        # strip_string drops spaces/linebreaks before comparing.
        req = _make_request(
            "\\boxed{ \\frac{1}{2} }",
            "\\boxed{\\frac{1}{2}}",
            "math500",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_unclosed_think_tag_returns_zero_no_failure(self) -> None:
        req = _make_request(
            "<think>thinking \\boxed{120}",
            "\\boxed{120}",
            "math500",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False


# ---------------------------------------------------------------------------
# english_multichoice
# ---------------------------------------------------------------------------


class TestEnglishMultichoice:
    def test_correct_letter(self) -> None:
        req = _make_request(
            "After thinking, Answer: B",
            "B",
            "english_multichoice",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
        assert result.extracted_answer == "B"

    def test_wrong_letter(self) -> None:
        req = _make_request(
            "Answer: A",
            "B",
            "english_multichoice",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0

    def test_case_insensitive_prefix(self) -> None:
        req = _make_request(
            "answer: b",
            "B",
            "english_multichoice",
        )
        # The regex uppercases the letter via `[A-Z]` only; lowercase 'b'
        # would NOT match. RLVR's pattern is the same. Confirm the strict
        # behavior so we don't drift.
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0

    def test_no_answer_prefix(self) -> None:
        req = _make_request(
            "I think the right one is B.",
            "B",
            "english_multichoice",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.extracted_answer is None

    def test_thinking_tags_stripped(self) -> None:
        req = _make_request(
            "<think>maybe Answer: A</think>\nAnswer: B",
            "B",
            "english_multichoice",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_unclosed_think_returns_zero(self) -> None:
        req = _make_request(
            "<think>thinking Answer: B",
            "B",
            "english_multichoice",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0


# ---------------------------------------------------------------------------
# multilingual_multichoice
# ---------------------------------------------------------------------------


class TestMultilingualMultichoice:
    def test_english_answer_prefix(self) -> None:
        req = _make_request(
            "Answer: C",
            "C",
            "multilingual_multichoice",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
        assert result.extracted_answer == "C"

    def test_chinese_answer_prefix(self) -> None:
        req = _make_request(
            "经过推理，答案: C",
            "C",
            "multilingual_multichoice",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
        assert result.extracted_answer == "C"

    def test_arabic_letter_normalized_to_english(self) -> None:
        # Arabic 'ج' is the C-equivalent letter and should normalize to ' C'.
        # normalize_extracted_answer also calls .strip() so the leading space goes.
        req = _make_request(
            "الإجابة: ج",
            "C",
            "multilingual_multichoice",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
        assert result.extracted_answer == "C"

    def test_no_known_prefix_returns_zero(self) -> None:
        req = _make_request(
            "the right one is C",
            "C",
            "multilingual_multichoice",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.extracted_answer is None


# ---------------------------------------------------------------------------
# Misc / dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_unknown_verifier_type_falls_back_to_math(self) -> None:
        req = _make_request("\\boxed{42}", "42", "definitely_not_real")
        result = asyncio.run(_make_server().verify(req))
        # Falls back to math-verify which accepts \boxed{42}==42.
        assert result.reward == 1.0
        # We preserve the originally-requested verifier_type on the response.
        assert result.verifier_type == "definitely_not_real"

    def test_default_verifier_type_is_math(self) -> None:
        # When `verifier_type` is omitted, the model default kicks in.
        config = MathRlvrResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name=""
        )
        server = MathRlvrResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
        # Build the request without setting verifier_type explicitly.
        response = NeMoGymResponse(
            id="r1",
            created_at=0.0,
            model="m",
            object="response",
            output=[
                {
                    "id": "msg",
                    "content": [
                        {
                            "annotations": [],
                            "text": "\\boxed{42}",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        req = MathRlvrVerifyRequest(
            responses_create_params={"input": []},
            response=response,
            ground_truth="42",
        )
        result = asyncio.run(server.verify(req))
        assert result.verifier_type == "math"
        assert result.reward == 1.0
