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
import json
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

from pytest import approx

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.swe_atlas_qna.app import (
    SweAtlasQnaConfig,
    SweAtlasQnaServer,
    SweAtlasQnaVerifyRequest,
    _apply_negative_flip,
    _canonicalize_judge_result,
    _is_scored,
    _normalize_score,
    _normalize_status,
    _parse_judge_response,
    _resolve_path,
    _score_from_status,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _make_response(text: str) -> NeMoGymResponse:
    """Build a policy response object carrying ``text`` as the candidate answer."""
    return NeMoGymResponse(
        id="mock_resp",
        created_at=0.0,
        model="mock_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="mock_msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _chat_completion_bytes(text: str) -> bytes:
    return json.dumps(
        {
            "id": "mock_chat",
            "object": "chat.completion",
            "created": 0,
            "model": "mock_judge",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    ).encode()


def _rating(status: Optional[str], score: Optional[str], statement: str = "s") -> str:
    return json.dumps(
        {"ratings": [{"rubric_statement": statement, "status": status, "score": score, "justification": "j"}]}
    )


def _rubric(rid: str, title: str, rtype: str = "positive hli verifier", importance: str = "must have") -> dict:
    return {"id": rid, "title": title, "annotations": {"type": rtype, "importance": importance}}


def _verify_request(answer: str, rubrics: list[dict], problem_statement: str = "q") -> SweAtlasQnaVerifyRequest:
    return SweAtlasQnaVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_make_response(answer),
        verifier_metadata={"rubrics": rubrics, "problem_statement": problem_statement},
    )


class TestVerdictHelpers:
    def test_normalize_status(self) -> None:
        assert _normalize_status("yes") == "YES"
        assert _normalize_status("N") == "NO"
        assert _normalize_status("1") == "YES"
        assert _normalize_status("maybe") is None
        assert _normalize_status(None) is None

    def test_normalize_score(self) -> None:
        assert _normalize_score("1.0") == "1"
        assert _normalize_score("0") == "0"
        assert _normalize_score("yes") == "1"
        assert _normalize_score("false") == "0"
        assert _normalize_score("2") is None
        assert _normalize_score(None) is None

    def test_score_from_status(self) -> None:
        assert _score_from_status("YES") == "1"
        assert _score_from_status("NO") == "0"
        assert _score_from_status(None) is None

    def test_apply_negative_flip(self) -> None:
        assert _apply_negative_flip("1", "negative hli verifier") == ("0", True)
        assert _apply_negative_flip("0", "negative hli verifier") == ("1", True)
        assert _apply_negative_flip("1", "positive hli verifier") == ("1", False)
        assert _apply_negative_flip(None, "negative") == (None, False)

    def test_canonicalize_non_dict(self) -> None:
        assert _canonicalize_judge_result("nope", "positive") is None

    def test_canonicalize_status_score_mismatch(self) -> None:
        # status=YES (->1) but score=0 => mismatch flagged, status is canonical.
        result = _canonicalize_judge_result({"status": "YES", "score": "0"}, "positive")
        assert result["score"] == "1"
        assert result["judge_status_score_mismatch"] is True

    def test_canonicalize_score_fallback_when_no_status(self) -> None:
        result = _canonicalize_judge_result({"status": "bogus", "score": "1"}, "positive")
        assert result["score"] == "1"
        assert result["status"] == "YES"

    def test_canonicalize_unscored(self) -> None:
        result = _canonicalize_judge_result({"status": "bogus", "score": "bogus"}, "positive")
        assert result["score"] is None
        assert result["status"] is None

    def test_is_scored(self) -> None:
        assert _is_scored({"score": "1"}) is True
        assert _is_scored({"score": "0"}) is True
        assert _is_scored({"score": None}) is False
        assert _is_scored(None) is False


class TestParseJudgeResponse:
    def test_plain_json(self) -> None:
        assert _parse_judge_response(_rating("YES", "1"))["status"] == "YES"

    def test_json_fence(self) -> None:
        text = "here:\n```json\n" + _rating("NO", "0") + "\n```\ntrailing"
        assert _parse_judge_response(text)["status"] == "NO"

    def test_ratings_slice_from_prose(self) -> None:
        text = 'blah blah {"ratings": [{"rubric_statement": "s", "status": "YES", "score": "1"}]} tail'
        assert _parse_judge_response(text)["score"] == "1"

    def test_empty(self) -> None:
        assert _parse_judge_response("") is None

    def test_invalid_json(self) -> None:
        assert _parse_judge_response("{not json") is None

    def test_no_ratings_key(self) -> None:
        assert _parse_judge_response('{"foo": 1}') is None

    def test_empty_ratings(self) -> None:
        assert _parse_judge_response('{"ratings": []}') is None


class TestExtraction:
    def _server(self) -> SweAtlasQnaServer:
        return _make_server()

    def test_extract_plain_answer(self) -> None:
        assert self._server()._extract_answer(_make_response("hello world")) == "hello world"

    def test_extract_final_answer_tags(self) -> None:
        wrapped = "preamble <<FINAL_ANSWER>>\nthe answer\n<<FINAL_ANSWER>> trailing"
        assert self._server()._extract_answer(_make_response(wrapped)) == "the answer"

    def test_extract_empty_output(self) -> None:
        empty = NeMoGymResponse(
            id="e",
            created_at=0.0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        assert self._server()._extract_answer(empty) == ""

    def test_extract_chat_completion_text(self) -> None:
        completion = NeMoGymChatCompletion.model_validate(json.loads(_chat_completion_bytes("verdict").decode()))
        assert SweAtlasQnaServer._extract_chat_completion_text(completion) == "verdict"

    def test_extract_chat_completion_no_choices(self) -> None:
        completion = NeMoGymChatCompletion.model_validate(
            {"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": []}
        )
        assert SweAtlasQnaServer._extract_chat_completion_text(completion) == ""


def _make_config() -> SweAtlasQnaConfig:
    return SweAtlasQnaConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="swe_atlas_qna",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_chat_completions_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(messages=[]),
    )


def _make_server(server_mock: Any = None) -> SweAtlasQnaServer:
    if server_mock is None:
        server_mock = MagicMock(spec=ServerClient)
    return SweAtlasQnaServer(config=_make_config(), server_client=server_mock)


def _mock_constant(server_mock: MagicMock, rating_text: str) -> None:
    """Every judge call returns the same rating."""

    async def _post(**kwargs: Any) -> Any:
        resp = AsyncMock()
        resp.read = AsyncMock(return_value=_chat_completion_bytes(rating_text))
        return resp

    server_mock.post = AsyncMock(side_effect=_post)


def _mock_by_title(server_mock: MagicMock, title_to_rating: dict[str, str], default: str) -> None:
    """Return a rating keyed on which rubric title appears in the user message."""

    async def _post(*, server_name: str, url_path: str, json: Any) -> Any:  # noqa: A002
        content = json.messages[1]["content"]
        rating_text = default
        for needle, text in title_to_rating.items():
            if needle in content:
                rating_text = text
                break
        resp = AsyncMock()
        resp.read = AsyncMock(return_value=_chat_completion_bytes(rating_text))
        return resp

    server_mock.post = AsyncMock(side_effect=_post)


class TestVerify:
    async def test_single_rubric_pass(self) -> None:
        server_mock = MagicMock(spec=ServerClient)
        _mock_constant(server_mock, _rating("YES", "1"))
        server = _make_server(server_mock)
        result = await server.verify(_verify_request("ans", [_rubric("r1", "1.1: states the port")]))
        assert result.reward == approx(1.0)
        assert result.passed is True
        assert result.agg_score == approx(1.0)
        assert result.num_scored == 1

    async def test_single_rubric_fail(self) -> None:
        server_mock = MagicMock(spec=ServerClient)
        _mock_constant(server_mock, _rating("NO", "0"))
        server = _make_server(server_mock)
        result = await server.verify(_verify_request("ans", [_rubric("r1", "states the port")]))
        assert result.reward == approx(0.0)
        assert result.passed is False
        assert result.agg_score == approx(0.0)

    async def test_negative_rubric_present_fails(self) -> None:
        # Judge says the (undesirable) behavior IS present (YES/1) -> flipped to 0.
        server_mock = MagicMock(spec=ServerClient)
        _mock_constant(server_mock, _rating("YES", "1"))
        server = _make_server(server_mock)
        result = await server.verify(
            _verify_request("ans", [_rubric("r1", "leaks secret", rtype="negative hli verifier")])
        )
        assert result.reward == approx(0.0)
        assert result.rubric_scores[0]["score"]["was_flipped"] is True

    async def test_negative_rubric_absent_passes(self) -> None:
        server_mock = MagicMock(spec=ServerClient)
        _mock_constant(server_mock, _rating("NO", "0"))
        server = _make_server(server_mock)
        result = await server.verify(
            _verify_request("ans", [_rubric("r1", "leaks secret", rtype="negative hli verifier")])
        )
        assert result.reward == approx(1.0)

    async def test_all_rubrics_pass(self) -> None:
        server_mock = MagicMock(spec=ServerClient)
        _mock_constant(server_mock, _rating("YES", "1"))
        server = _make_server(server_mock)
        rubrics = [_rubric(f"r{i}", f"criterion {i}") for i in range(3)]
        result = await server.verify(_verify_request("ans", rubrics))
        assert result.reward == approx(1.0)
        assert result.num_scored == 3
        assert result.agg_score == approx(1.0)

    async def test_one_rubric_fails_blocks_reward(self) -> None:
        server_mock = MagicMock(spec=ServerClient)
        _mock_by_title(
            server_mock,
            title_to_rating={"BADCRIT": _rating("NO", "0")},
            default=_rating("YES", "1"),
        )
        server = _make_server(server_mock)
        rubrics = [_rubric("r1", "good one"), _rubric("r2", "BADCRIT here"), _rubric("r3", "good two")]
        result = await server.verify(_verify_request("ans", rubrics))
        assert result.reward == approx(0.0)  # strict all-must-pass
        assert result.passed is False
        assert result.agg_score == approx(2.0 / 3.0)  # soft signal

    async def test_no_answer_scores_zero(self) -> None:
        server = _make_server()
        result = await server.verify(_verify_request("", [_rubric("r1", "criterion")]))
        assert result.reward == approx(0.0)
        assert result.num_unscored == 1
        assert result.rubric_scores == []

    async def test_no_rubrics_scores_zero(self) -> None:
        server = _make_server()
        result = await server.verify(_verify_request("ans", []))
        assert result.reward == approx(0.0)
        assert result.num_rubrics == 0

    async def test_judge_exception_yields_unscored(self) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server_mock.post = AsyncMock(side_effect=RuntimeError("judge down"))
        server = _make_server(server_mock)
        result = await server.verify(_verify_request("ans", [_rubric("r1", "criterion")]))
        assert result.reward == approx(0.0)
        assert result.num_scored == 0
        assert result.num_unscored == 1
        assert result.rubric_scores[0]["score"] is None
        # Retries exhausted: post called judge_max_retries times.
        assert server_mock.post.await_count == server.config.judge_max_retries

    async def test_invalid_then_valid_retry(self) -> None:
        server_mock = MagicMock(spec=ServerClient)
        payloads = [_chat_completion_bytes("garbage, no json"), _chat_completion_bytes(_rating("YES", "1"))]

        async def _post(**kwargs: Any) -> Any:
            resp = AsyncMock()
            resp.read = AsyncMock(return_value=payloads.pop(0))
            return resp

        server_mock.post = AsyncMock(side_effect=_post)
        server = _make_server(server_mock)
        result = await server.verify(_verify_request("ans", [_rubric("r1", "criterion")]))
        assert result.reward == approx(1.0)
        assert server_mock.post.await_count == 2

    async def test_final_answer_tags_are_graded(self) -> None:
        server_mock = MagicMock(spec=ServerClient)
        _mock_constant(server_mock, _rating("YES", "1"))
        server = _make_server(server_mock)
        wrapped = "junk <<FINAL_ANSWER>>\nreal answer\n<<FINAL_ANSWER>>"
        result = await server.verify(_verify_request(wrapped, [_rubric("r1", "criterion")]))
        # The wrapped answer reached the judge (not the junk preamble).
        sent = server_mock.post.await_args.kwargs["json"].messages[1]["content"]
        assert "real answer" in sent
        assert result.reward == approx(1.0)

    async def test_user_supplied_braces_do_not_break_judge_prompt_formatting(self) -> None:
        server_mock = MagicMock(spec=ServerClient)
        _mock_constant(server_mock, _rating("YES", "1"))
        server = _make_server(server_mock)
        problem_statement = "What does `{x: value` mean in this code sample?"
        answer = "It is probably an object literal like `{foo: bar}` or an f-string fragment `{baz`."

        result = await server.verify(_verify_request(answer, [_rubric("r1", "criterion")], problem_statement))

        sent = server_mock.post.await_args.kwargs["json"].messages[1]["content"]
        assert problem_statement in sent
        assert answer in sent
        assert result.reward == approx(1.0)


class TestMetrics:
    def test_score_fn(self) -> None:
        assert SweAtlasQnaServer._score_fn({"passed": True, "agg_score": 0.5}) == {"pass": 1.0, "agg_score": 0.5}
        assert SweAtlasQnaServer._score_fn({}) == {"pass": 0.0, "agg_score": 0.0}

    def test_compute_metrics(self) -> None:
        server = _make_server()
        tasks = [
            [{"passed": True, "agg_score": 1.0, "reward": 1.0}],
            [{"passed": False, "agg_score": 0.5, "reward": 0.0}],
        ]
        metrics = server.compute_metrics(tasks)
        assert any(k.startswith("pass@1[avg-of-") and k.endswith("/pass") for k in metrics), metrics

    def test_get_key_metrics(self) -> None:
        server = _make_server()
        agent_metrics = {
            "mean/reward": 0.5,
            "mean/input_tokens": 100.0,
            "mean/output_tokens": 200.0,
            "pass@4/pass": 0.75,
            "pass@4/no_answer": 0.01,
            "pass@1[avg-of-4]/pass": 0.5,
        }
        key = server.get_key_metrics(agent_metrics)
        assert key["mean/reward"] == 0.5
        assert "pass@4/pass" in key
        assert "pass@4/no_answer" not in key
        assert "pass@1[avg-of-4]/pass" in key


class TestModelPostInit:
    def test_prompts_loaded(self) -> None:
        server = _make_server()
        assert len(server._judge_system_prompt) > 0
        assert (
            "{problem_statement}" in server._judge_user_template or "problem_statement" in server._judge_user_template
        )


class TestResolvePath:
    def test_absolute_existing(self, tmp_path) -> None:
        f = tmp_path / "p.txt"
        f.write_text("x")
        assert _resolve_path(str(f)) == f

    def test_cwd_relative_existing(self) -> None:
        # The committed prompt path is gym-root-relative and exists from cwd.
        rel = "resources_servers/swe_atlas_qna/prompts/judge_system.txt"
        assert _resolve_path(rel).exists()

    def test_module_relative_fallback(self) -> None:
        # Path relative to the module dir (not cwd).
        assert _resolve_path("prompts/judge_system.txt").exists()

    def test_missing_returns_as_given(self) -> None:
        from pathlib import Path

        assert _resolve_path("nope/does_not_exist.txt") == Path("nope/does_not_exist.txt")
