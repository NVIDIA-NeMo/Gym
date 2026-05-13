# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
import asyncio
import json
from unittest.mock import MagicMock

from nemo_gym.base_resources_server import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.ifeval.app import (
    IFEvalResourcesServer,
    IFEvalResourcesServerConfig,
    IFEvalVerifyRequest,
)


def _make_server() -> IFEvalResourcesServer:
    config = IFEvalResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return IFEvalResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_request(response_content: str, ground_truth, request_id: int = 1) -> IFEvalVerifyRequest:
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
    return IFEvalVerifyRequest(
        responses_create_params={"input": []},
        response=response,
        ground_truth=ground_truth,
    )


class TestApp:
    def test_sanity(self) -> None:
        _make_server()

    def test_keyword_positive(self) -> None:
        req = _make_request(
            "The waves crashed against the rocks while the sun dipped below the horizon.",
            [{"func_name": "verify_keywords", "keyword_list": ["waves", "horizon"]}],
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
        assert result.follow_all_constraints is True
        assert result.follow_constraint_list == [True]
        assert result.verification_failed is False

    def test_keyword_negative(self) -> None:
        req = _make_request(
            "The mountain stood tall against the sky.",
            [{"func_name": "verify_keywords", "keyword_list": ["waves", "horizon"]}],
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.follow_all_constraints is False
        assert result.follow_constraint_list == [False]

    def test_no_commas_positive(self) -> None:
        req = _make_request(
            "A quiet morning without punctuation worth noting.",
            [{"func_name": "validate_no_commas"}],
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_no_commas_negative(self) -> None:
        req = _make_request(
            "Hello, this has commas.",
            [{"func_name": "validate_no_commas"}],
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0

    def test_multiple_constraints_all_pass(self) -> None:
        req = _make_request(
            "the waves rolled in over the horizon",
            [
                {"func_name": "verify_keywords", "keyword_list": ["waves", "horizon"]},
                {"func_name": "validate_lowercase"},
                {"func_name": "validate_no_commas"},
            ],
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
        assert result.follow_constraint_list == [True, True, True]

    def test_multiple_constraints_one_fails(self) -> None:
        req = _make_request(
            "The waves rolled in over the horizon.",
            [
                {"func_name": "verify_keywords", "keyword_list": ["waves", "horizon"]},
                {"func_name": "validate_lowercase"},
            ],
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.follow_constraint_list == [True, False]

    def test_ground_truth_as_json_string(self) -> None:
        req = _make_request(
            "the waves and horizon",
            json.dumps([{"func_name": "verify_keywords", "keyword_list": ["waves", "horizon"]}]),
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_ground_truth_single_dict(self) -> None:
        req = _make_request(
            "no commas here",
            {"func_name": "validate_no_commas"},
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_think_tag_stripped_before_verification(self) -> None:
        req = _make_request(
            "<think>let me reason, with commas</think>\nno commas here",
            [{"func_name": "validate_no_commas"}],
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_unclosed_think_tag_returns_zero_no_failure(self) -> None:
        req = _make_request(
            "<think>incomplete reasoning",
            [{"func_name": "validate_no_commas"}],
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False

    def test_invalid_json_marks_verification_failed(self) -> None:
        req = _make_request("anything", "{not valid json")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is True

    def test_unknown_func_name_returns_false(self) -> None:
        req = _make_request("anything", [{"func_name": "does_not_exist"}])
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.follow_constraint_list == [False]
        assert result.verification_failed is False
