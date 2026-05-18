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
from resources_servers.code_output_prediction.app import (
    CodeOutputPredictionResourcesServer,
    CodeOutputPredictionResourcesServerConfig,
    CodeOutputPredictionVerifyRequest,
)


def _make_server() -> CodeOutputPredictionResourcesServer:
    config = CodeOutputPredictionResourcesServerConfig(
        host="0.0.0.0", port=8080, entrypoint="", name=""
    )
    return CodeOutputPredictionResourcesServer(
        config=config, server_client=MagicMock(spec=ServerClient)
    )


def _make_request(
    response_content: str, ground_truth, request_id: int = 1
) -> CodeOutputPredictionVerifyRequest:
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
    return CodeOutputPredictionVerifyRequest(
        responses_create_params={"input": []},
        response=response,
        ground_truth=ground_truth,
    )


class TestApp:
    def test_sanity(self) -> None:
        _make_server()

    def test_correct_output_str(self) -> None:
        req = _make_request(
            "After running it, the output is {'output': '6'}",
            "{'ground_truth': '6'}",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
        assert result.verification_failed is False

    def test_wrong_output(self) -> None:
        req = _make_request(
            "After running it, the output is {'output': '7'}",
            "{'ground_truth': '6'}",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False

    def test_ground_truth_dict_directly(self) -> None:
        req = _make_request(
            "{'output': 'hello'}",
            {"ground_truth": "hello"},
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_only_last_brace_block_used(self) -> None:
        # The first dict is reasoning scratch; the LAST one is the answer.
        req = _make_request(
            "First I tried {'output': 'wrong'}, then I corrected to {'output': '42'}",
            "{'ground_truth': '42'}",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_no_brace_block_returns_zero(self) -> None:
        req = _make_request(
            "I cannot determine the output of this snippet.",
            "{'ground_truth': '42'}",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False

    def test_unparseable_brace_block_returns_zero(self) -> None:
        # Trailing characters break ast.literal_eval but the brace-rebalance
        # fallback should still recover the inner dict.
        req = _make_request(
            "The output is {'output': '6'}!!!",
            "{'ground_truth': '6'}",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_garbage_brace_block_returns_zero(self) -> None:
        req = _make_request(
            "Here it is: {definitely not python}",
            "{'ground_truth': '6'}",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False

    def test_non_dict_brace_block_returns_zero(self) -> None:
        # ast.literal_eval succeeds but produces a set, not a dict.
        req = _make_request(
            "{1, 2, 3}",
            "{'ground_truth': '6'}",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0

    def test_dict_missing_output_key_returns_zero(self) -> None:
        req = _make_request(
            "{'result': '6'}",
            "{'ground_truth': '6'}",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0

    def test_think_tag_stripped_before_extraction(self) -> None:
        req = _make_request(
            "<think>maybe {'output': 'wrong'}</think>\n{'output': '6'}",
            "{'ground_truth': '6'}",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_unclosed_think_returns_zero_no_failure(self) -> None:
        req = _make_request(
            "<think>still thinking about {'output': '6'}",
            "{'ground_truth': '6'}",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False

    def test_invalid_ground_truth_marks_zero(self) -> None:
        req = _make_request(
            "{'output': '6'}",
            "not-a-python-literal",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        # Note: matches RLVR semantics (returns False, not a verifier failure).
        assert result.verification_failed is False

    def test_list_output_value(self) -> None:
        req = _make_request(
            "{'output': '[1, 2, 3]'}",
            "{'ground_truth': '[1, 2, 3]'}",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
