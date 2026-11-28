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

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List
from unittest.mock import MagicMock

import pytest
import yaml
from aiohttp import ClientResponseError, RequestInfo
from fastapi.testclient import TestClient
from frontier_finance.metrics import MetricsReport
from frontier_finance.models import ItemResult, Rubric
from frontier_finance.prompts import RubricPrompt
from multidict import CIMultiDict
from pydantic import ValidationError
from yarl import URL

from benchmarks.frontier_finance import prepare as frontier_finance_prepare
from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import JUDGE_FAILED, VERIFIER_ERROR, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.finance_agent_v2.app import (
    FinanceAgentV2ResourcesServer,
    FinanceAgentV2ResourcesServerConfig,
    FinanceAgentV2RunRequest,
    FinanceAgentV2SeedSessionRequest,
    FinanceAgentV2VerifyRequest,
    cap_tool_dates,
)
from resources_servers.finance_agent_v2.frontier_finance_grading import (
    FrontierFinanceGrader,
    FrontierFinanceGraderConfig,
    FrontierFinanceJudgeConfig,
    frontier_finance_metrics,
    item_result_from_rollout,
)
from resources_servers.finance_agent_v2.task_data import FrontierFinanceTaskFields, TaskData


_SERVER_DIR = Path(__file__).resolve().parents[1]
_PROMPTS = _SERVER_DIR / "prompt_templates"
_JUDGES = ("claude", "gemini", "gpt")

# sha256 of the official prompt at the pinned frontier-finance commit. A change means
# the grader is no longer the one the published numbers used.
_PINNED_SYSTEM_SHA256 = "1b1207c4b94b2b75b2a57b57a870d8dfb8dc0b4d42a6f735041e5a4bf82c105b"  # pragma: allowlist secret
_PINNED_TEMPLATE_SHA256 = (
    "77b46b1f7f86cf7efa257a7eee82096d9841a5b07f66a8ee69d31764b3456eb7"  # pragma: allowlist secret
)


_GRADER_FIELDS = set(FrontierFinanceGraderConfig.model_fields)


def _make_server(**overrides) -> FinanceAgentV2ResourcesServer:
    grader = dict(
        judge_panel=[
            FrontierFinanceJudgeConfig(name=name, model_server=ModelServerRef(type="responses_api_models", name=name))
            for name in _JUDGES
        ],
        judge_thread_pool_size=8,
    )
    grader.update({k: overrides.pop(k) for k in list(overrides) if k in _GRADER_FIELDS})
    cfg = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="frontier_finance_test",
        sec_api_key="dummy-sec",  # pragma: allowlist secret
        edgar_search_mode="live",
        use_cache=False,
        retrieval_system_prompt_fpath=str(_PROMPTS / "finance_agent_v2_retrieval.yaml"),
        rubric_judge_prompt_template_fpath=str(_PROMPTS / "finance_agent_v2_rubric_judge.yaml"),
        frontier_finance_grader=FrontierFinanceGraderConfig(**grader),
    )
    cfg.update(overrides)
    config = FinanceAgentV2ResourcesServerConfig(**cfg)
    return FinanceAgentV2ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _request(session_id: str = "s1") -> MagicMock:
    req = MagicMock()
    req.session = {SESSION_ID_KEY: session_id}
    return req


def _call(name: str, arguments: str, i: int = 1) -> dict:
    return {
        "id": f"tc_{i}",
        "call_id": f"call_{i}",
        "name": name,
        "arguments": arguments,
        "type": "function_call",
        "status": "completed",
    }


def _submit(answer: str, i: int = 1) -> dict:
    return _call("submit_final_result", json.dumps({"final_result": answer}), i)


def _response(answer: str | None, output: List[dict] | None = None, stop_reason: str | None = None) -> NeMoGymResponse:
    if output is None:
        output = [_submit(answer)] if answer is not None else []
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="test",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        metadata={"stop_reason": stop_reason} if stop_reason else None,
    )


def _rubrics(n: int, must_have_every: int = 2) -> List[dict]:
    return [
        {
            "rubric_id": i + 1,
            "rubric_text": f"criterion {i + 1}",
            "must_have": i % must_have_every == 0,
            "rubric_type": "Factual Data Extraction",
            "data_source_type": "sec_filing",
        }
        for i in range(n)
    ]


def _verify_request(rubrics: List[dict], answer: str | None = "The answer.") -> FinanceAgentV2VerifyRequest:
    return FinanceAgentV2VerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Question:\nq?"}]},
        response=_response(answer),
        question="q?",
        query_id="q1",
        as_of_date="2025-06-23",
        use_cases=["company_research"],
        rubrics=rubrics,
    )


def _batch_rubric_texts(user_prompt: str) -> List[str]:
    block = user_prompt.split("<rubrics>\n")[1].split("\n</rubrics>")[0]
    return [m.group(1) for m in re.finditer(r"^\d+\. (.*)$", block, flags=re.MULTILINE)]


async def _no_sleep(_seconds: float) -> None:
    return None


def _http_error_response(status: int):
    url = URL("http://judge/v1/responses")
    error = ClientResponseError(
        RequestInfo(url, "POST", CIMultiDict(), url), (), status=status, message="error", headers=CIMultiDict()
    )

    async def read() -> bytes:
        return b""

    class _Response:
        ok = False
        content = SimpleNamespace(read=read)

        def raise_for_status(self) -> None:
            raise error

    return _Response()


def _labels_reply(labels: List[bool]) -> str:
    return json.dumps({str(i): {"reason": "r", "label": label} for i, label in enumerate(labels)})


class _FakeJudges:
    """Stands in for the judge model servers; records every prompt per judge."""

    def __init__(self, decide: Dict[str, Callable[[str, int], str]]):
        self.decide = decide
        self.prompts: Dict[str, List[str]] = {name: [] for name in decide}

    async def __call__(self, server, judge, create_params) -> str:
        user = create_params.input[1].content
        self.prompts[judge.name].append(user)
        return self.decide[judge.name](user, len(self.prompts[judge.name]))


def _all(label: bool) -> Callable[[str, int], str]:
    return lambda user, _call: _labels_reply([label] * len(_batch_rubric_texts(user)))


def _by_text(true_texts: set) -> Callable[[str, int], str]:
    return lambda user, _call: _labels_reply([t in true_texts for t in _batch_rubric_texts(user)])


def _raise(user: str, _call: int) -> str:
    raise RuntimeError("judge down")


@pytest.fixture
def judges(monkeypatch):
    def install(**decide) -> _FakeJudges:
        fake = _FakeJudges(decide)

        async def fake_call(self, judge, create_params):
            return await fake(self, judge, create_params)

        monkeypatch.setattr(FrontierFinanceGrader, "_judge_call", fake_call)
        return fake

    return install


class TestDateCap:
    @pytest.mark.parametrize(
        "args, expected",
        [
            ({"end_date": "2026-01-01"}, {"end_date": "2025-06-23"}),
            ({}, {"end_date": "2025-06-23"}),
            ({"end_date": ""}, {"end_date": "2025-06-23"}),
            ({"end_date": "2025-01-01"}, {"end_date": "2025-01-01"}),
            (
                {"start_date": "2025-09-01", "end_date": "2025-10-01"},
                {"start_date": "2025-06-23", "end_date": "2025-06-23"},
            ),
            ({"start_date": "2024-01-01"}, {"start_date": "2024-01-01", "end_date": "2025-06-23"}),
        ],
    )
    def test_cap_tool_dates(self, args, expected) -> None:
        assert cap_tool_dates(args, "2025-06-23") == expected

    async def test_dispatch_caps_only_dated_tools_per_session(self) -> None:
        server = _make_server()
        seen: Dict[str, dict] = {}

        def fake_tool(name):
            async def execute(args, state, logger):
                seen[name] = args
                return SimpleNamespace(output="ok")

            return SimpleNamespace(execute=execute)

        for name in ("edgar_search", "web_search", "price_history", "calculator"):
            server._tools[name] = fake_tool(name)

        await server.seed_session(_request("dated"), FinanceAgentV2SeedSessionRequest(as_of_date="2025-06-23"))
        await server.seed_session(_request("undated"), FinanceAgentV2SeedSessionRequest())

        for name in ("edgar_search", "web_search", "price_history"):
            await server._dispatch_tool(name, _request("dated"), {"query": "x", "end_date": "2026-02-01"})
            assert seen[name]["end_date"] == "2025-06-23"
        await server._dispatch_tool("calculator", _request("dated"), {"expression": "1+1"})
        assert seen["calculator"] == {"expression": "1+1"}

        await server._dispatch_tool("edgar_search", _request("undated"), {"query": "x", "end_date": "2026-02-01"})
        assert seen["edgar_search"]["end_date"] == "2026-02-01"

    async def test_tool_call_limit_blocks_all_but_submit(self) -> None:
        server = _make_server(max_tool_calls=2)
        calls: List[str] = []

        def fake_tool(name):
            async def execute(args, state, logger):
                calls.append(name)
                return SimpleNamespace(output="ok")

            return SimpleNamespace(execute=execute)

        for name in ("calculator", "submit_final_result"):
            server._tools[name] = fake_tool(name)
        await server.seed_session(_request("a"), FinanceAgentV2SeedSessionRequest())
        await server.seed_session(_request("b"), FinanceAgentV2SeedSessionRequest())

        for _ in range(2):
            await server._dispatch_tool("calculator", _request("a"), {"expression": "1+1"})
        blocked = await server._dispatch_tool("calculator", _request("a"), {"expression": "1+1"})
        assert "Tool call limit reached (2 / 2)" in blocked["results"]
        await server._dispatch_tool("submit_final_result", _request("a"), {"final_result": "x"})
        await server._dispatch_tool("calculator", _request("b"), {"expression": "1+1"})
        assert calls == ["calculator", "calculator", "submit_final_result", "calculator"]

        server._end_session("a")
        await server.seed_session(_request("a"), FinanceAgentV2SeedSessionRequest())
        await server._dispatch_tool("calculator", _request("a"), {"expression": "1+1"})
        assert calls[-1] == "calculator" and len(calls) == 5

    async def test_reseeding_a_live_session_keeps_cap_and_budgets(self) -> None:
        server = _make_server(max_tool_calls=1)
        seen: List[dict] = []

        async def execute(args, state, logger):
            seen.append(args)
            return SimpleNamespace(output="ok")

        server._tools["web_search"] = SimpleNamespace(execute=execute)
        await server.seed_session(_request("a"), FinanceAgentV2SeedSessionRequest(as_of_date="2025-06-23"))
        await server._dispatch_tool("web_search", _request("a"), {"query": "x", "end_date": "2026-01-01"})

        for reseed in (FinanceAgentV2SeedSessionRequest(), FinanceAgentV2SeedSessionRequest(as_of_date="2099-01-01")):
            await server.seed_session(_request("a"), reseed)
            assert server._as_of_dates["a"] == "2025-06-23"
            blocked = await server._dispatch_tool("web_search", _request("a"), {"query": "x"})
            assert "Tool call limit reached (1 / 1)" in blocked["results"]
        assert seen == [{"query": "x", "end_date": "2025-06-23"}]
        assert server._tool_usage("a")["tool_limit_hit"] == "max_tool_calls"

    async def test_tool_finishing_after_session_end_leaves_no_state(self) -> None:
        server = _make_server()
        release = asyncio.Event()

        async def execute(args, state, logger):
            await release.wait()
            return SimpleNamespace(output="ok")

        server._tools["web_search"] = SimpleNamespace(execute=execute)
        await server.seed_session(_request("a"), FinanceAgentV2SeedSessionRequest())
        pending = asyncio.create_task(server._dispatch_tool("web_search", _request("a"), {"query": "x"}))
        await asyncio.sleep(0)
        server._end_session("a")
        release.set()
        await pending
        for state in (
            server._tool_seconds,
            server._tool_call_counts,
            server._tools_running,
            server._tools_busy_since,
            server._data_storage,
        ):
            assert "a" not in state

    async def test_overlapping_tool_calls_count_wall_clock_time_once(self, monkeypatch) -> None:
        server = _make_server(max_tool_time_seconds=10)
        clock = {"now": 0.0}
        monkeypatch.setattr("resources_servers.finance_agent_v2.app.time.monotonic", lambda: clock["now"])
        release = asyncio.Event()

        async def execute(args, state, logger):
            await release.wait()
            return SimpleNamespace(output="ok")

        server._tools["web_search"] = SimpleNamespace(execute=execute)
        await server.seed_session(_request("a"), FinanceAgentV2SeedSessionRequest())
        calls = [
            asyncio.create_task(server._dispatch_tool("web_search", _request("a"), {"query": "x"})) for _ in range(3)
        ]
        await asyncio.sleep(0)
        clock["now"] += 4
        assert server._tool_time_used("a") == 4
        release.set()
        await asyncio.gather(*calls)
        assert server._tool_seconds["a"] == 4

    async def test_tool_time_limit_counts_only_tool_time(self, monkeypatch) -> None:
        server = _make_server(max_tool_time_seconds=10)
        clock = {"now": 0.0}
        monkeypatch.setattr("resources_servers.finance_agent_v2.app.time.monotonic", lambda: clock["now"])
        calls: List[str] = []

        def fake_tool(name, seconds):
            async def execute(args, state, logger):
                calls.append(name)
                clock["now"] += seconds
                return SimpleNamespace(output="ok")

            return SimpleNamespace(execute=execute)

        server._tools["web_search"] = fake_tool("web_search", 5)
        server._tools["submit_final_result"] = fake_tool("submit_final_result", 0)
        await server.seed_session(_request("a"), FinanceAgentV2SeedSessionRequest())

        await server._dispatch_tool("web_search", _request("a"), {"query": "x"})
        clock["now"] += 1000
        await server._dispatch_tool("web_search", _request("a"), {"query": "x"})
        blocked = await server._dispatch_tool("web_search", _request("a"), {"query": "x"})
        assert "Tool time limit reached (10s / 10s)" in blocked["results"]
        await server._dispatch_tool("submit_final_result", _request("a"), {"final_result": "x"})
        assert calls == ["web_search", "web_search", "submit_final_result"]

    @pytest.mark.parametrize("as_of_date", ["06/23/2025", "20250623", "2025-02-30", ""])
    def test_seed_rejects_malformed_date(self, as_of_date) -> None:
        with pytest.raises(ValidationError):
            FinanceAgentV2SeedSessionRequest(as_of_date=as_of_date)


class TestVerify:
    async def test_unanimous(self, judges) -> None:
        judges(claude=_by_text({"criterion 1"}), gemini=_by_text({"criterion 1"}), gpt=_by_text({"criterion 1"}))
        result = await _make_server().verify(_request(), _verify_request(_rubrics(4)))
        assert result.rubric_labels == [True, False, False, False]
        assert result.reward == pytest.approx(0.25)
        assert (result.num_must_have, result.num_must_have_qualified) == (2, 1)
        assert not result.mask_sample and result.grading_failure is None

    async def test_majority_vote(self, judges) -> None:
        judges(claude=_all(True), gemini=_all(True), gpt=_all(False))
        result = await _make_server().verify(_request(), _verify_request(_rubrics(3)))
        assert result.rubric_labels == [True, True, True]
        assert result.judge_labels["gpt"] == [False, False, False]

    async def test_tie_goes_to_first_surviving_judge(self, judges) -> None:
        judges(claude=_raise, gemini=_all(False), gpt=_all(True))
        result = await _make_server().verify(_request(), _verify_request(_rubrics(2)))
        assert result.rubric_labels == [False, False]
        assert result.judge_labels["claude"] is None
        assert result.failed_checks_by_judge == {"claude": 2}
        assert not result.mask_sample

    async def test_tie_goes_to_first_surviving_judge_when_it_says_true(self, judges) -> None:
        judges(claude=_raise, gemini=_all(True), gpt=_all(False))
        result = await _make_server().verify(_request(), _verify_request(_rubrics(2)))
        assert result.rubric_labels == [True, True]

    @pytest.mark.parametrize(
        ("error_cause", "failure_kind"),
        [("model_call_failed", "model_request_error"), ("tool_aborted", "environment_server_failed")],
    )
    async def test_policy_or_tool_outage_masks_the_sample(self, judges, error_cause, failure_kind) -> None:
        fake = judges(claude=_all(True), gemini=_all(True), gpt=_all(True))
        request = _verify_request(_rubrics(1))
        request.response = _response("A", stop_reason="error")
        request.response.metadata["error_cause"] = error_cause
        result = await _make_server().verify(_request(), request)
        assert (result.mask_sample, result.failure_kind) == (True, failure_kind)
        assert all(not prompts for prompts in fake.prompts.values())

    async def test_context_overflow_is_a_scored_failure(self, judges) -> None:
        judges(claude=_all(True), gemini=_all(True), gpt=_all(True))
        request = _verify_request(_rubrics(1))
        request.response = _response("A", stop_reason="error")
        request.response.metadata["error_cause"] = "context_overflow"
        result = await _make_server().verify(_request(), request)
        assert not result.mask_sample
        assert (result.reward, result.grading_failure) == (0.0, "no_response")

    def test_seed_tool_call_and_verify_over_http(self, judges) -> None:
        judges(claude=_all(True), gemini=_all(True), gpt=_all(False))
        server = _make_server(max_tool_calls=5)
        seen: List[dict] = []

        async def execute(args, state, logger):
            seen.append(args)
            return SimpleNamespace(output="ok")

        server._tools["web_search"] = SimpleNamespace(execute=execute)
        row = _verify_request(_rubrics(2)).model_dump(mode="json")
        with TestClient(server.setup_webserver()) as client:
            assert client.post("/seed_session", json={**row, "as_of_date": "20250623"}).status_code == 422
            assert client.post("/seed_session", json=row).status_code == 200
            assert client.post("/web_search", json={"query": "x", "end_date": "2026-01-01"}).status_code == 200
            reply = client.post("/verify", json=row)

        assert reply.status_code == 200
        assert seen == [{"query": "x", "end_date": "2025-06-23"}]
        body = reply.json()
        assert body["rubric_labels"] == [True, True]
        assert body["judge_labels"]["gpt"] == [False, False]
        assert (body["tool_calls_used"], body["tool_limit_hit"]) == (1, None)
        assert body["reward"] == 1.0

    async def test_all_judges_failing_masks_the_sample(self, judges) -> None:
        judges(claude=_raise, gemini=_raise, gpt=_raise)
        result = await _make_server().verify(_request(), _verify_request(_rubrics(2)))
        assert result.mask_sample is True
        assert result.failure_kind == JUDGE_FAILED
        assert result.grading_failure == "judge_error"
        assert result.failed_checks_by_judge == {"claude": 2, "gemini": 2, "gpt": 2}

    async def test_batches_of_max_rubrics_per_call(self, judges) -> None:
        fake = judges(claude=_all(True), gemini=_all(True), gpt=_all(True))
        result = await _make_server().verify(_request(), _verify_request(_rubrics(35)))
        assert result.num_qualified == 35
        for name in _JUDGES:
            assert [len(_batch_rubric_texts(p)) for p in fake.prompts[name]] == [30, 5]

    async def test_prompt_includes_query_and_date(self, judges) -> None:
        fake = judges(claude=_all(True), gemini=_all(True), gpt=_all(True))
        await _make_server().verify(_request(), _verify_request(_rubrics(1)))
        prompt = fake.prompts["claude"][0]
        assert "<query>\nq?\n</query>" in prompt
        assert "<date>\n2025-06-23\n</date>" in prompt
        assert "<report>\nThe answer.\n</report>" in prompt

    async def test_unparseable_reply_is_retried_with_format_reminder(self, judges) -> None:
        def flaky(user, call):
            return "not json" if call == 1 else _labels_reply([True] * len(_batch_rubric_texts(user)))

        fake = judges(claude=flaky, gemini=_all(True), gpt=_all(True))
        result = await _make_server().verify(_request(), _verify_request(_rubrics(2)))
        assert result.judge_labels["claude"] == [True, True]
        assert fake.prompts["claude"][1].endswith(RubricPrompt.JSON_FORMAT_RETRY_SUFFIX)

    async def test_missing_index_counts_as_not_met(self, judges) -> None:
        judges(
            claude=lambda u, c: json.dumps({"0": {"label": True}}),
            gemini=lambda u, c: json.dumps({"0": {"label": True}}),
            gpt=lambda u, c: json.dumps({"0": {"label": True}}),
        )
        result = await _make_server().verify(_request(), _verify_request(_rubrics(2)))
        assert result.rubric_labels == [True, False]

    async def test_errored_call_is_retried(self, judges) -> None:
        def once_down(user, call):
            if call == 1:
                raise RuntimeError("transient")
            return _labels_reply([True] * len(_batch_rubric_texts(user)))

        judges(claude=once_down, gemini=_all(True), gpt=_all(True))
        result = await _make_server(judge_max_retries=1).verify(_request(), _verify_request(_rubrics(1)))
        assert result.judge_labels["claude"] == [True]

    async def test_no_submission_is_a_scored_failure(self, judges) -> None:
        fake = judges(claude=_all(True), gemini=_all(True), gpt=_all(True))
        result = await _make_server().verify(_request(), _verify_request(_rubrics(3), answer=None))
        assert result.reward == 0.0
        assert result.grading_failure == "no_response"
        assert not result.mask_sample
        assert all(not prompts for prompts in fake.prompts.values())

    async def test_grades_the_first_submission_in_a_turn(self, judges) -> None:
        fake = judges(claude=_all(True), gemini=_all(True), gpt=_all(True))
        request = _verify_request(_rubrics(1))
        request.response = _response(None, [_submit("A", 1), _submit("B", 2)], stop_reason="done_tool")
        await _make_server().verify(_request(), request)
        assert "<report>\nA\n</report>" in fake.prompts["claude"][0]

        request.response = _response(None, [_submit("A", 1), _call("submit_final_result", "{bad", 2)])
        assert (await _make_server().verify(_request(), request)).grading_failure is None

    async def test_malformed_first_submission_is_no_response(self, judges) -> None:
        fake = judges(claude=_all(True), gemini=_all(True), gpt=_all(True))
        request = _verify_request(_rubrics(1))
        request.response = _response(None, [_call("submit_final_result", "{bad", 1), _submit("B", 2)])
        result = await _make_server().verify(_request(), request)
        assert (result.reward, result.grading_failure) == (0.0, "no_response")
        assert all(not prompts for prompts in fake.prompts.values())

    @pytest.mark.parametrize("stop_reason", ["error", "max_output_tokens", "max_time"])
    async def test_submission_not_executed_is_no_response(self, judges, stop_reason) -> None:
        fake = judges(claude=_all(True), gemini=_all(True), gpt=_all(True))
        request = _verify_request(_rubrics(1))
        output = [_call("web_search", json.dumps({"query": "q"}), 1), _submit("A", 2)]
        request.response = _response(None, output, stop_reason=stop_reason)
        result = await _make_server().verify(_request(), request)
        assert (result.reward, result.grading_failure) == (0.0, "no_response")
        assert all(not prompts for prompts in fake.prompts.values())

    async def test_judge_timeout_covers_reading_the_body(self) -> None:
        server = _make_server(judge_call_timeout=0.05)

        class _StalledResponse:
            ok = True

            async def read(self):
                await asyncio.sleep(3600)

        async def post(**kwargs):
            return _StalledResponse()

        server.server_client.post = post
        judge = server.config.frontier_finance_grader.judge_panel[0]
        params = NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=1)
        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(server._frontier_finance_grader._judge_call(judge, params), timeout=5)
        assert time.monotonic() - start < 1

    @pytest.mark.parametrize(("status", "attempts"), [(401, 1), (404, 1), (429, 3), (503, 3)])
    async def test_judge_retries_only_transient_http_errors(self, monkeypatch, status, attempts) -> None:
        server = _make_server(judge_max_retries=2)
        monkeypatch.setattr(asyncio, "sleep", _no_sleep)
        posts = []

        async def post(**kwargs):
            posts.append(kwargs)
            return _http_error_response(status)

        server.server_client.post = post
        judge = server.config.frontier_finance_grader.judge_panel[0]
        with pytest.raises(Exception) as excinfo:
            await server._frontier_finance_grader._judge_complete(judge, "system", "user")
        assert getattr(excinfo.value, "http_status", None) == status
        assert len(posts) == attempts

    async def test_no_rubrics_masks_the_sample(self, judges) -> None:
        judges(claude=_all(True), gemini=_all(True), gpt=_all(True))
        result = await _make_server().verify(_request(), _verify_request([]))
        assert result.mask_sample is True
        assert result.failure_kind == VERIFIER_ERROR

    @pytest.mark.parametrize(
        "override",
        [
            {"max_rubrics_per_call": 0},
            {"judge_call_timeout": 0},
            {"judge_max_output_tokens": 0},
            {"judge_thread_pool_size": 0},
            {"judge_max_retries": -1},
            {"max_json_parse_retries": -1},
            {"max_tool_calls": 0},
            {"max_tool_time_seconds": 0},
        ],
    )
    def test_rejects_non_positive_settings(self, override) -> None:
        with pytest.raises(ValidationError):
            _make_server(**override)

    def test_rejects_empty_or_duplicate_panel(self) -> None:
        with pytest.raises(ValueError):
            _make_server(judge_panel=[])
        dup = FrontierFinanceJudgeConfig(name="a", model_server=ModelServerRef(type="responses_api_models", name="a"))
        with pytest.raises(ValueError):
            _make_server(judge_panel=[dup, dup])


def _row(query_id: str, labels: List[bool] | None, failure: str | None = None, must_have_every: int = 2) -> dict:
    n = len(labels) if labels is not None else 4
    rubrics = _rubrics(n, must_have_every)
    return {
        "query_id": query_id,
        "rubrics": rubrics,
        "rubric_labels": labels,
        "grading_failure": failure,
        "failed_checks_by_judge": {},
        "use_cases": ["company_research"],
        "num_rubrics": n,
        "num_qualified": sum(labels or []),
    }


class TestMetrics:
    def test_matches_official_metrics_report(self) -> None:
        rows = [
            _row("a", [True, False, True, True]),
            _row("b", [False, False]),
            _row("c", None, failure="no_response"),
        ]
        expected = MetricsReport(
            [
                ItemResult(
                    query_id="a",
                    rubrics=[Rubric.from_dict(r) for r in rows[0]["rubrics"]],
                    labels=[True, False, True, True],
                ),
                ItemResult(
                    query_id="b", rubrics=[Rubric.from_dict(r) for r in rows[1]["rubrics"]], labels=[False, False]
                ),
                ItemResult(
                    query_id="c",
                    rubrics=[Rubric.from_dict(r) for r in rows[2]["rubrics"]],
                    failed=True,
                    failure_reason="no_response",
                ),
            ]
        ).compute()
        metrics = frontier_finance_metrics(rows)
        for name, value in expected.items():
            if isinstance(value, (int, float)):
                assert metrics[f"frontier_finance/{name}"] == pytest.approx(value), name
        assert metrics["frontier_finance/macro_avg_qualification_rate_on_all_queries"] == pytest.approx(0.75 / 3)
        assert metrics["frontier_finance/micro_avg_qualification_rate_on_all_queries"] == pytest.approx(3 / 10)
        assert metrics["frontier_finance/use_case/company_research"] == pytest.approx(3 / 6)
        assert metrics["frontier_finance/rubric_type/Factual Data Extraction"] == pytest.approx(3 / 6)

    def test_item_result_from_failed_rollout_has_no_labels(self) -> None:
        result = item_result_from_rollout(_row("c", None, failure="judge_error"))
        assert result.failed and result.failure_reason == "judge_error"
        assert result.labels == [] and result.num_rubrics == 4

    def test_rollout_that_never_reached_verify_is_a_failed_query(self) -> None:
        bare_row = {"reward": 0.0, "response": {"metadata": {"stop_reason": "error"}}}
        result = item_result_from_rollout(bare_row)
        assert result.failed and result.failure_reason == "no_response"
        metrics = frontier_finance_metrics([_row("a", [True, True]), bare_row])
        assert metrics["frontier_finance/num_failed_queries"] == 1
        assert metrics["frontier_finance/macro_avg_qualification_rate_on_success_queries"] == pytest.approx(1.0)

    def test_tool_limit_and_stop_reason_shares(self) -> None:
        rows = [
            {
                **_row("a", [True]),
                "tool_limit_hit": "max_tool_time_seconds",
                "response": {"metadata": {"stop_reason": "done_tool"}},
            },
            {**_row("b", [False]), "response": {"metadata": {"stop_reason": "max_turns"}}},
        ]
        metrics = frontier_finance_metrics(rows)
        assert metrics["frontier_finance/tool_limit_hit_rate/max_tool_time_seconds"] == 0.5
        assert metrics["frontier_finance/tool_limit_hit_rate/max_tool_calls"] == 0.0
        assert metrics["frontier_finance/stop_reason/done_tool"] == 0.5
        assert metrics["frontier_finance/stop_reason/max_turns"] == 0.5

    def test_key_metrics_report_failed_checks_per_judge(self) -> None:
        server = _make_server()
        row = {**_row("a", [True, True]), "failed_checks_by_judge": {"gpt": 2}}
        key = server.get_key_metrics(server.compute_metrics([[row]]))
        assert key["frontier_finance/failed_checks/gpt"] == 2
        assert key["frontier_finance/macro_avg_qualification_rate_on_all_queries"] == pytest.approx(1.0)
        assert key["frontier_finance/num_failed_queries"] == 0

    def test_repeat_metrics_are_the_headline_rates(self) -> None:
        server = _make_server()
        repeat = server.compute_repeat_metrics([[_row("a", [True, False])], [_row("b", [True, True])]])
        assert repeat["frontier_finance/macro_avg_qualification_rate_on_all_queries"] == pytest.approx(0.75)
        assert all(name.startswith("frontier_finance/") for name in repeat)


class TestPinnedGrader:
    def test_official_prompt_is_unchanged(self) -> None:
        assert hashlib.sha256(RubricPrompt.SYSTEM.encode()).hexdigest() == _PINNED_SYSTEM_SHA256
        assert hashlib.sha256(RubricPrompt._TEMPLATE.encode()).hexdigest() == _PINNED_TEMPLATE_SHA256


class TestShippedConfig:
    def test_panel_matches_official_judges_in_order(self) -> None:
        config = yaml.safe_load((_SERVER_DIR / "configs" / "frontier_finance.yaml").read_text())
        server = config["frontier_finance_resources_server"]["resources_servers"]["finance_agent_v2"]
        panel = server["frontier_finance_grader"]["judge_panel"]
        assert [j["name"] for j in panel] == ["claude-sonnet-4-6", "gemini-3.1-pro-preview", "gpt-5.4"]
        for judge in panel:
            assert judge["model_server"]["name"] in config

    def test_example_data_parses(self) -> None:
        lines = (_SERVER_DIR / "data" / "frontier_finance_example.jsonl").read_text().splitlines()
        assert len(lines) == 5
        for line in lines:
            row = json.loads(line)
            request = FinanceAgentV2VerifyRequest(**row, response=_response("x"))
            assert request.rubrics and request.as_of_date
            TaskData.model_validate({k: v for k, v in row.items() if k != "responses_create_params"})


class TestTaskData:
    def test_schema_fields_match_the_verify_request(self) -> None:
        wire = FinanceAgentV2VerifyRequest.model_fields
        for name in [*FrontierFinanceTaskFields.model_fields, "question"]:
            field = TaskData.model_fields[name]
            assert name in wire
            assert (
                wire[name].annotation,
                wire[name].is_required(),
                wire[name].get_default(call_default_factory=True),
            ) == (
                field.annotation,
                field.is_required(),
                field.get_default(call_default_factory=True),
            ), name

    def test_prepared_row_parses_as_every_server_request(self) -> None:
        source = {
            "query_id": "q1",
            "query": "What did ACME report?",
            "query_date": "20250623",
            "use_cases": ["company_research"],
            "capabilities": ["fact_lookup"],
            "rubrics": [{"rubric_id": 1, "rubric_text": "A.", "must_have": True, "rubric_type": "Factual"}],
        }
        row = frontier_finance_prepare.convert_row(source, frontier_finance_prepare.build_tools())
        run = FinanceAgentV2RunRequest.model_validate(row)
        verify = FinanceAgentV2VerifyRequest.model_validate({**row, "response": _response("x").model_dump()})
        seed = FinanceAgentV2SeedSessionRequest.model_validate(row)
        TaskData.model_validate({k: v for k, v in row.items() if k != "responses_create_params"})
        for request in (run, verify):
            assert (request.query_id, request.question, request.as_of_date) == (
                "q1",
                "What did ACME report?",
                "2025-06-23",
            )
            assert request.use_cases == ["company_research"]
            assert [(r.rubric_id, r.must_have, r.rubric_type) for r in request.rubrics] == [(1, True, "Factual")]
        assert seed.as_of_date == "2025-06-23"

    def test_rejects_malformed_rubric(self) -> None:
        with pytest.raises(ValidationError):
            TaskData.model_validate({"rubrics": [{"rubric_id": "not-an-int", "rubric_text": "A"}]})

    def test_accepts_undated_and_unrubricked_rows(self) -> None:
        TaskData.model_validate({"question": "q"})
