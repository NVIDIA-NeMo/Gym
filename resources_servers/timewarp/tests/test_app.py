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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.timewarp import app as app_module
from resources_servers.timewarp.app import (
    JUDGE_NOT_CONFIGURED,
    VERIFIER_FIXTURE,
    SiteUrls,
    TimewarpResourcesServer,
    TimewarpResourcesServerConfig,
    TimewarpVerifier,
    TimewarpVerifyRequest,
    extract_final_answer,
)
from resources_servers.timewarp.task_data import TaskData


EXAMPLE_PATH = Path(__file__).parents[1] / "data" / "example.jsonl"
SITES = SiteUrls(wiki="http://wiki.test", news="http://news.test", webshop="http://shop.test/abc")


def _message(text: str) -> dict:
    return {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _call(name: str = "observe") -> list[dict]:
    return [
        {"type": "function_call", "call_id": "c1", "name": name, "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "URL: http://wiki.test/"},
    ]


def _response(output: list[dict]) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "r",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "output": output,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def _request(answer: str, *, eval_types=("string_match",), references=None, **extra) -> TimewarpVerifyRequest:
    references = references or {"must_include": ["alaska"], "fuzzy_match": "Alaska"}
    return TimewarpVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "q"}]},
            "response": _response([*_call(), _message(answer)]).model_dump(),
            "verifier_metadata": {"eval_types": list(eval_types), "reference_answers": references},
            "intent": "Which state uses boroughs?",
            "ui_version": 3,
            "sites": ["wiki", "news"],
            **extra,
        }
    )


def _server(**config) -> TimewarpResourcesServer:
    return TimewarpResourcesServer(
        config=TimewarpResourcesServerConfig(
            name="timewarp", host="0.0.0.0", port=8080, entrypoint="app.py", site_urls={1: SITES}, **config
        ),
        server_client=MagicMock(spec=ServerClient),
    )


async def test_verifier_fixture() -> None:
    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
    )
    assert [(result.kind, result.observed_rewards) for result in results] == [
        ("full_reward", (1.0,)),
        ("zero_reward", (0.0,)),
        ("zero_reward", (0.0,)),
        ("malformed", ()),
    ]


class TestFinalAnswer:
    def test_is_the_message_after_the_last_tool_step(self):
        response = _response([_message("Let me look."), *_call(), _message("<think>Texas?</think>Alaska.")])
        assert extract_final_answer(response) == "Alaska."

    def test_is_empty_when_the_episode_ended_on_a_tool_step(self):
        assert extract_final_answer(_response([_message("Alaska, probably."), *_call()])) == ""

    def test_skips_reasoning_items(self):
        reasoning = {"id": "rs", "type": "reasoning", "summary": [{"type": "summary_text", "text": "Texas"}]}
        assert extract_final_answer(_response([*_call(), _message("Alaska"), reasoning])) == "Alaska"


class TestVerify:
    async def test_labels_version_and_site_for_the_metric_breakdowns(self):
        result = await TimewarpVerifier().verify(_request("Alaska."))
        assert (result.reward, result.extracted_answer) == (1.0, "Alaska.")
        assert (result.timewarp_version, result.timewarp_site) == ("v3", "multi")

    async def test_response_owned_fields_sent_by_a_caller_are_recomputed(self):
        result = await TimewarpVerifier().verify(_request("Texas.", reward=1.0, extracted_answer="Alaska"))
        assert (result.reward, result.extracted_answer) == (0.0, "Texas.")

    async def test_judge_task_without_a_judge_is_masked_not_failed(self):
        request = _request("Ethiopian.", eval_types=("llm_judge",), references={"fuzzy_match": "(2) Ethiopian"})
        result = await TimewarpVerifier().verify(request)
        assert result.mask_sample is True
        assert result.failure_kind == JUDGE_NOT_CONFIGURED

    async def test_na_gold_is_scored_without_calling_the_judge(self):
        request = _request("N/A", eval_types=("llm_judge",), references={"fuzzy_match": "N/A"})
        assert (await TimewarpVerifier().verify(request)).reward == 1.0

    async def test_configured_judge_receives_the_upstream_prompt(self, monkeypatch):
        judge = AsyncMock(return_value=_response([_message("correct")]))
        monkeypatch.setattr(app_module, "call_judge", judge)
        server = _server(judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"))
        request = _request("Ethiopian crash.", eval_types=("llm_judge",), references={"fuzzy_match": "(2) Ethiopian"})

        result = await TimewarpVerifier.verify(server, request)

        assert result.reward == 1.0 and not result.mask_sample
        params = judge.await_args.kwargs["json"]
        assert judge.await_args.kwargs["server_name"] == "judge_model"
        prompt = params.input[1].content
        assert "- question: Which state uses boroughs?" in prompt
        assert "- reference answer: (2) Ethiopian" in prompt
        assert "- student answer: Ethiopian crash." in prompt

    async def test_partially_correct_verdict_scores_zero(self, monkeypatch):
        monkeypatch.setattr(
            app_module, "call_judge", AsyncMock(return_value=_response([_message("partially correct")]))
        )
        server = _server(judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"))
        request = _request("Ethiopian.", eval_types=("llm_judge",), references={"fuzzy_match": "(2) Ethiopian"})
        assert (await TimewarpVerifier.verify(server, request)).reward == 0.0


class _FakeSession:
    def __init__(self, start_url: str):
        self.start_url = start_url
        self.closed = 0

    async def observe(self, part: int = 1) -> str:
        return f"URL: {self.start_url}\nPage content (part {part})"

    def is_alive(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed += 1


@pytest.fixture
def fake_sessions(monkeypatch):
    opened: list[dict] = []

    async def _open(browser, *, start_url, allowed_origins, limits):
        session = _FakeSession(start_url)
        opened.append({"session": session, "allowed_origins": allowed_origins})
        return session

    monkeypatch.setattr(app_module.BrowserSession, "open", _open)
    monkeypatch.setattr(app_module.SharedBrowser, "get", AsyncMock(return_value=None))
    return opened


class TestLifecycle:
    def test_seed_opens_the_start_site_of_the_rows_ui_version(self, fake_sessions):
        client = TestClient(_server().setup_webserver())
        client.post("/seed_session", json={"ui_version": 1, "start_site": "webshop", "intent": "q"}).raise_for_status()

        (opened,) = fake_sessions
        assert opened["session"].start_url == "http://shop.test/abc"
        assert opened["allowed_origins"] == {"http://wiki.test", "http://news.test", "http://shop.test"}
        observation = client.post("/observe", json={"part": 2})
        assert observation.headers["content-type"].startswith("text/plain")
        assert observation.text == "URL: http://shop.test/abc\nPage content (part 2)"

    def test_unconfigured_ui_version_fails_the_seed(self, fake_sessions):
        client = TestClient(_server().setup_webserver(), raise_server_exceptions=False)
        assert client.post("/seed_session", json={"ui_version": 4, "start_site": "wiki"}).status_code == 500
        assert fake_sessions == []

    def test_verify_and_reseed_release_the_browser(self, fake_sessions):
        client = TestClient(_server().setup_webserver())
        client.post("/seed_session", json={"ui_version": 1, "start_site": "wiki"}).raise_for_status()
        client.post("/seed_session", json={"ui_version": 1, "start_site": "news"}).raise_for_status()
        first, second = (entry["session"] for entry in fake_sessions)
        assert (first.closed, second.closed) == (1, 0)

        verified = client.post("/verify", json=_request("Alaska").model_dump(mode="json"))
        assert verified.json()["reward"] == 1.0
        assert second.closed == 1
        assert client.post("/observe", json={}).text.startswith("Error: this rollout has no browser session")

    def test_tool_errors_are_returned_to_the_policy(self, fake_sessions):
        client = TestClient(_server().setup_webserver())
        client.post("/seed_session", json={"ui_version": 1, "start_site": "wiki"}).raise_for_status()
        fake_sessions[0]["session"].click = AsyncMock(side_effect=RuntimeError("Timeout 5000ms exceeded.\nCall log:"))
        response = client.post("/click", json={"ref": "e5"})
        assert (response.status_code, response.text) == (200, "Error: Timeout 5000ms exceeded.")

    def test_a_browser_lost_mid_episode_masks_the_rollout(self, fake_sessions):
        client = TestClient(_server().setup_webserver())
        client.post("/seed_session", json={"ui_version": 1, "start_site": "wiki"}).raise_for_status()
        session = fake_sessions[0]["session"]
        session.click = AsyncMock(side_effect=RuntimeError("Target page, context or browser has been closed"))
        session.is_alive = lambda: False

        assert "browser session was lost" in client.post("/click", json={"ref": "e5"}).text
        verified = client.post("/verify", json=_request("Alaska").model_dump(mode="json")).json()
        assert (verified["mask_sample"], verified["failure_kind"]) == (True, "session_lost")

    def test_more_than_one_worker_is_rejected(self):
        with pytest.raises(ValueError, match="num_workers=1"):
            _server(num_workers=2)


class TestExampleData:
    ROWS = [json.loads(line) for line in EXAMPLE_PATH.read_text().splitlines()]

    def test_rows_validate_against_the_task_schema(self):
        assert len(self.ROWS) == 5
        for row in self.ROWS:
            TaskData.model_validate(
                {k: v for k, v in row.items() if k not in ("responses_create_params", "verifier_metadata")}
                | row["verifier_metadata"]
            )

    def test_every_advertised_tool_is_a_server_route_with_matching_arguments(self):
        openapi = _server().setup_webserver().openapi()
        schemas = openapi["components"]["schemas"]
        for tool in self.ROWS[0]["responses_create_params"]["tools"]:
            operation = openapi["paths"][f"/{tool['name']}"]["post"]
            body = operation.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
            accepted = set(schemas[body["$ref"].rsplit("/", 1)[1]]["properties"]) if body else set()
            assert set(tool["parameters"]["properties"]) == accepted, tool["name"]


class TestMetrics:
    def test_success_rate_is_broken_down_by_ui_version_and_site(self):
        rollout = lambda version, site, reward: {"timewarp_version": version, "timewarp_site": site, "reward": reward}  # noqa: E731
        tasks = [[rollout("v1", "wiki", 1.0)], [rollout("v1", "news", 1.0)], [rollout("v2", "wiki", 0.0)]]
        metrics = _server().compute_metrics(tasks)
        assert metrics["v1/pass@1/accuracy"] == 100.0
        assert metrics["v2/pass@1/accuracy"] == 0.0
        assert metrics["wiki/pass@1/accuracy"] == 50.0
        assert metrics["news/pass@1/accuracy"] == 100.0
