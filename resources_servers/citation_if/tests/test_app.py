# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Server-layer tests: response extraction, the /verify contract, and the tool catch-all.

Scoring logic itself is covered by test_citation_if.py (gate semantics),
test_reward_hack_matrix.py (reward-hack fixtures) and test_residual_fuzz.py (properties).
This file covers what those cannot: that the server correctly turns a Responses API payload
into the (text, function_call_count) pair the scorer expects, and returns a well-formed
verdict.

The function_call_count path matters more than it looks. A rollout that keeps calling tools
instead of answering must score 0 via gate 0, and that only works if the server counts
function_call items — so it is asserted here rather than left implicit.
"""

from unittest.mock import MagicMock

from pytest import approx, fixture

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.citation_if.app import (
    CitationIfResourcesServer,
    CitationIfResourcesServerConfig,
    CitationIfVerifyRequest,
    extract_response_shape,
)
from resources_servers.citation_if.scorer import TERMINAL_TOOL_RESPONSE


class TestExtractResponseShape:
    """The Responses-API payload -> (text, function_call_count) contract."""

    def test_single_output_text(self) -> None:
        response = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Answer. [citation_ab12]"}],
                }
            ]
        }
        assert extract_response_shape(response) == ("Answer. [citation_ab12]", 0)

    def test_counts_function_calls(self) -> None:
        response = {
            "output": [
                {"type": "function_call", "name": "web_search", "arguments": "{}"},
                {"type": "function_call", "name": "web_search", "arguments": "{}"},
            ]
        }
        text, calls = extract_response_shape(response)
        assert text == ""
        assert calls == 2

    def test_text_and_function_call_together(self) -> None:
        response = {
            "output": [
                {"type": "function_call", "name": "web_search", "arguments": "{}"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Partial answer."}],
                },
            ]
        }
        assert extract_response_shape(response) == ("Partial answer.", 1)

    def test_ignores_non_assistant_messages(self) -> None:
        response = {
            "output": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "output_text", "text": "should be ignored"}],
                }
            ]
        }
        assert extract_response_shape(response) == ("", 0)

    def test_reasoning_items_are_not_answer_text(self) -> None:
        """Reasoning must never reach the scorer — a citation inside it earns nothing."""
        response = {
            "output": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "I should cite [citation_ab12]"}],
                }
            ]
        }
        assert extract_response_shape(response) == ("", 0)

    def test_empty_and_malformed_output(self) -> None:
        assert extract_response_shape({}) == ("", 0)
        assert extract_response_shape({"output": None}) == ("", 0)
        assert extract_response_shape({"output": ["not a dict", 42]}) == ("", 0)


class TestCitationIfServer:
    @fixture
    def config(self) -> CitationIfResourcesServerConfig:
        return CitationIfResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="citation_if",
        )

    @fixture
    def verifier(self) -> dict:
        return {
            "type": "citation_if",
            "mode": "cite",
            "grammar": "ascii_brackets",
            "id_kind": "full_source",
            "id_regex": r"citation_[a-z0-9]{4}",
            "valid_id_set": ["citation_ab12", "citation_cd34"],
            "expected_ids": ["citation_ab12"],
            "expected_slack": 1,
            "min_valid_citations": 1,
        }

    def _server(self, config: CitationIfResourcesServerConfig) -> CitationIfResourcesServer:
        return CitationIfResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _response(self, text: str = "", function_calls: int = 0) -> NeMoGymResponse:
        output = [
            {"type": "function_call", "name": "web_search", "call_id": f"c{i}", "arguments": "{}"}
            for i in range(function_calls)
        ]
        if text:
            output.append(
                NeMoGymResponseOutputMessage(
                    id="msg",
                    content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
        return NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="policy_model",
            object="response",
            output=output,
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    def _request(self, verifier: dict, **kwargs) -> CitationIfVerifyRequest:
        return CitationIfVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._response(**kwargs),
            verifier=verifier,
        )

    async def test_verify_compliant_answer_scores_one(self, config, verifier) -> None:
        server = self._server(config)
        result = await server.verify(self._request(verifier, text="Gateway sold it in 1999. [citation_ab12]"))
        assert result.reward == approx(1.0)
        assert result.match_details["cited_ids"] == ["citation_ab12"]

    async def test_verify_missing_citation_scores_zero(self, config, verifier) -> None:
        server = self._server(config)
        result = await server.verify(self._request(verifier, text="Gateway sold it in 1999."))
        assert result.reward == approx(0.0)

    async def test_verify_hallucinated_id_scores_zero(self, config, verifier) -> None:
        server = self._server(config)
        result = await server.verify(self._request(verifier, text="Gateway sold it in 1999. [citation_zzzz]"))
        assert result.reward == approx(0.0)

    async def test_verify_tool_call_scores_zero(self, config, verifier) -> None:
        """Gate 0: a rollout that calls a tool instead of answering earns nothing."""
        server = self._server(config)
        result = await server.verify(self._request(verifier, function_calls=1))
        assert result.reward == approx(0.0)

    async def test_verify_citation_without_answer_scores_zero(self, config, verifier) -> None:
        """Gate 0b: markup only, no answer text."""
        server = self._server(config)
        result = await server.verify(self._request(verifier, text="[citation_ab12]"))
        assert result.reward == approx(0.0)

    async def test_verify_echoes_request_fields(self, config, verifier) -> None:
        """The response must carry the request through, so rollouts stay re-scorable."""
        server = self._server(config)
        request = self._request(verifier, text="Answer. [citation_ab12]")
        result = await server.verify(request)
        assert result.verifier == verifier
        assert result.response == request.response

    async def test_verify_no_cite_mode(self, config, verifier) -> None:
        no_cite = {**verifier, "mode": "no_cite", "expected_ids": None}
        server = self._server(config)
        clean = await server.verify(self._request(no_cite, text="Gateway sold it in 1999."))
        cited = await server.verify(self._request(no_cite, text="Gateway sold it in 1999. [citation_ab12]"))
        assert clean.reward == approx(1.0)
        assert cited.reward == approx(0.0)

    async def test_tool_catchall_returns_terminal_response(self, config) -> None:
        """Any tool call during a rollout terminates it rather than 404-looping."""
        server = self._server(config)
        result = await server._tool_catchall("web_search", MagicMock())
        assert result.body.decode() == TERMINAL_TOOL_RESPONSE
