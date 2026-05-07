# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.skill_judge.app import (
    SkillJudgeResourcesServer,
    _extract_json_array,
    _format_tool_calls,
)
from resources_servers.skill_judge.schemas import (
    JudgeRequest,
    SkillJudgeResourcesServerConfig,
    SkillJudgeVerifyRequest,
    ToolCallLogEntry,
)


PROMPT_TEMPLATE_FPATH = str(Path(__file__).resolve().parents[1] / "prompt_templates" / "skill_judge.txt")


def _make_config(**overrides) -> SkillJudgeResourcesServerConfig:
    return SkillJudgeResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        judge_prompt_template_fpath=PROMPT_TEMPLATE_FPATH,
        **overrides,
    )


def _make_server(**overrides) -> tuple[SkillJudgeResourcesServer, MagicMock]:
    server_client = MagicMock(spec=ServerClient)
    server = SkillJudgeResourcesServer(config=_make_config(**overrides), server_client=server_client)
    return server, server_client


def _assistant_msg(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg_id",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _mock_judge_response(server_client: MagicMock, judge_text: str) -> None:
    response = NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="judge",
        object="response",
        output=[_assistant_msg(judge_text)],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )
    raw = MagicMock()
    raw.read = AsyncMock(return_value=response.model_dump_json().encode("utf-8"))
    server_client.post = AsyncMock(return_value=raw)


class TestJsonExtraction:
    def test_plain_array(self) -> None:
        assert _extract_json_array('[{"id":1}]') == [{"id": 1}]

    def test_strips_markdown_fence(self) -> None:
        text = '```json\n[{"id": 1, "satisfied": true}]\n```'
        assert _extract_json_array(text) == [{"id": 1, "satisfied": True}]

    def test_extracts_from_surrounding_prose(self) -> None:
        text = 'Here is the result: [{"id": 1}] thanks.'
        assert _extract_json_array(text) == [{"id": 1}]

    def test_returns_none_on_garbage(self) -> None:
        assert _extract_json_array("not json at all") is None

    def test_returns_none_on_non_array_json(self) -> None:
        assert _extract_json_array('{"id": 1}') is None


class TestFormatToolCalls:
    def test_empty(self) -> None:
        assert _format_tool_calls([]) == "(no tool calls)"

    def test_renders_exit_code_and_snippets(self) -> None:
        entries = [
            ToolCallLogEntry(
                name="run_bash",
                arguments='"python scripts/review.py"',
                exit_code=0,
                stdout_snippet="BLOCK httpx-usage",
                stderr_snippet="warn",
                truncated=True,
            )
        ]
        out = _format_tool_calls(entries)
        assert "run_bash" in out
        assert "exit 0" in out
        assert "BLOCK httpx-usage" in out
        assert "(output truncated)" in out


class TestJudgeEndpoint:
    @pytest.mark.asyncio
    async def test_all_assertions_satisfied_gives_full_reward(self) -> None:
        server, client = _make_server()
        _mock_judge_response(
            client,
            '[{"id":1,"satisfied":true,"evidence":"ran review.py"},'
            '{"id":2,"satisfied":true,"evidence":"httpx block reported"}]',
        )

        resp = await server.judge(
            JudgeRequest(
                prompt="Review this server",
                response="BLOCK: httpx-usage at line 42",
                tool_calls=[ToolCallLogEntry(name="run_bash", arguments='"python scripts/review.py"', exit_code=0)],
                assertions=["agent runs review.py", "httpx block reported"],
            )
        )

        assert resp.reward == 1.0
        assert len(resp.grades) == 2
        assert all(g.satisfied for g in resp.grades)
        assert resp.parse_error is None

    @pytest.mark.asyncio
    async def test_partial_satisfaction_gives_fractional_reward(self) -> None:
        server, client = _make_server()
        _mock_judge_response(
            client,
            '[{"id":1,"satisfied":true,"evidence":"x"},'
            '{"id":2,"satisfied":false,"evidence":"y"},'
            '{"id":3,"satisfied":false,"evidence":"z"},'
            '{"id":4,"satisfied":true,"evidence":"w"}]',
        )

        resp = await server.judge(
            JudgeRequest(
                prompt="p",
                response="r",
                assertions=["a1", "a2", "a3", "a4"],
            )
        )

        assert resp.reward == 0.5
        assert [g.satisfied for g in resp.grades] == [True, False, False, True]

    @pytest.mark.asyncio
    async def test_missing_grade_defaults_unsatisfied(self) -> None:
        server, client = _make_server()
        _mock_judge_response(
            client,
            '[{"id":1,"satisfied":true,"evidence":"ok"}]',  # only 1 of 2
        )

        resp = await server.judge(
            JudgeRequest(prompt="p", response="r", assertions=["a1", "a2"]),
        )

        assert len(resp.grades) == 2
        assert resp.grades[0].satisfied is True
        assert resp.grades[1].satisfied is False
        assert "did not return" in resp.grades[1].evidence
        assert resp.reward == 0.5

    @pytest.mark.asyncio
    async def test_unparseable_judge_output(self) -> None:
        server, client = _make_server()
        _mock_judge_response(client, "I refuse to output JSON")

        resp = await server.judge(JudgeRequest(prompt="p", response="r", assertions=["a1", "a2"]))

        assert resp.reward == 0.0
        assert resp.parse_error is not None
        assert all(not g.satisfied for g in resp.grades)
        assert all("not parseable" in g.evidence for g in resp.grades)

    @pytest.mark.asyncio
    async def test_markdown_fenced_output_is_parsed(self) -> None:
        server, client = _make_server()
        _mock_judge_response(
            client,
            '```json\n[{"id":1,"satisfied":true,"evidence":"yes"}]\n```',
        )
        resp = await server.judge(JudgeRequest(prompt="p", response="r", assertions=["a"]))
        assert resp.reward == 1.0

    @pytest.mark.asyncio
    async def test_evidence_truncated_to_200_chars(self) -> None:
        server, client = _make_server()
        long_evidence = "x" * 500
        _mock_judge_response(
            client,
            f'[{{"id":1,"satisfied":true,"evidence":"{long_evidence}"}}]',
        )
        resp = await server.judge(JudgeRequest(prompt="p", response="r", assertions=["a"]))
        assert len(resp.grades[0].evidence) == 200

    @pytest.mark.asyncio
    async def test_empty_assertions_returns_zero_reward_without_llm_call(self) -> None:
        server, client = _make_server()
        client.post = AsyncMock(side_effect=AssertionError("should not be called"))
        resp = await server.judge(JudgeRequest(prompt="p", response="r", assertions=[]))
        assert resp.reward == 0.0
        assert resp.grades == []

    @pytest.mark.asyncio
    async def test_malformed_raw_grade_items_are_skipped(self) -> None:
        server, client = _make_server()
        _mock_judge_response(
            client,
            '["not a dict", {"id":"not-an-int"}, {"id":1,"satisfied":true,"evidence":"ok"}]',
        )
        resp = await server.judge(JudgeRequest(prompt="p", response="r", assertions=["a1", "a2"]))
        assert resp.grades[0].satisfied is True
        assert resp.grades[1].satisfied is False

    @pytest.mark.asyncio
    async def test_judge_usage_propagates(self) -> None:
        server, client = _make_server()
        response = NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="judge",
            object="response",
            output=[_assistant_msg('[{"id":1,"satisfied":true}]')],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
            usage={
                "input_tokens": 123,
                "output_tokens": 45,
                "total_tokens": 168,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        )
        raw = MagicMock()
        raw.read = AsyncMock(return_value=response.model_dump_json().encode("utf-8"))
        client.post = AsyncMock(return_value=raw)

        resp = await server.judge(JudgeRequest(prompt="p", response="r", assertions=["a"]))
        assert resp.judge_usage.input_tokens == 123
        assert resp.judge_usage.output_tokens == 45
        assert resp.judge_usage.total_tokens == 168


class TestVerifyEndpoint:
    @pytest.mark.asyncio
    async def test_extracts_assertions_from_verifier_metadata(self) -> None:
        server, client = _make_server()
        _mock_judge_response(client, '[{"id":1,"satisfied":true,"evidence":"ok"}]')

        body = SkillJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "review this server"}]
            ),
            response=NeMoGymResponse(
                id="r",
                created_at=0.0,
                model="m",
                object="response",
                output=[_assistant_msg("BLOCK httpx-usage")],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            ),
            verifier_metadata={
                "assertions": ["httpx is flagged"],
                "tool_calls": [{"name": "run_bash", "arguments": '"ls"', "exit_code": 0}],
                "expected_output": "BLOCK: httpx-usage",
            },
        )

        resp = await server.verify(body)
        assert resp.reward == 1.0
        assert len(resp.grades) == 1

    @pytest.mark.asyncio
    async def test_verify_with_no_assertions_returns_zero(self) -> None:
        server, client = _make_server()
        client.post = AsyncMock(side_effect=AssertionError("should not be called"))

        body = SkillJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse(
                id="r",
                created_at=0.0,
                model="m",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            ),
        )
        resp = await server.verify(body)
        assert resp.reward == 0.0
        assert resp.grades == []

    @pytest.mark.asyncio
    async def test_verify_with_non_list_assertions_skipped(self) -> None:
        server, client = _make_server()
        client.post = AsyncMock(side_effect=AssertionError("should not be called"))

        body = SkillJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse(
                id="r",
                created_at=0.0,
                model="m",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            ),
            verifier_metadata={"assertions": "not a list"},
        )
        resp = await server.verify(body)
        assert resp.reward == 0.0


class TestApp:
    def test_sanity(self) -> None:
        _make_server()

    def test_setup_webserver_registers_judge_route(self) -> None:
        server, _ = _make_server()
        app = server.setup_webserver()
        paths = {r.path for r in app.routes}
        assert "/judge" in paths
        assert "/verify" in paths

    def test_config_with_unlimited_concurrency(self) -> None:
        server, _ = _make_server(judge_endpoint_max_concurrency=None)
        # nullcontext works as an async context manager under `async with`
        assert server._judge_semaphore is not None
