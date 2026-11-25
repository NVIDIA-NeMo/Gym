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

import asyncio
import json
import random
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.verification import JudgeError
from resources_servers.browsecomp_advanced_harness.judge_prompt import JUDGE_PROMPT_TEMPLATE
from resources_servers.opencode_search.app import (
    _JUDGE_TEMPLATE_ECHO,
    SearchResourcesServer,
    SearchResourcesServerConfig,
    SearchVerifyRequest,
    _last_assistant_text,
    parse_judge_verdict,
)
from resources_servers.opencode_search.exa_cache import ExaCache


def _text_part(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="output_text", text=text)


def _assistant_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="message", role="assistant", content=[_text_part(text)])


class TestParseJudgeVerdict:
    def test_parses_yes(self):
        is_correct, extracted, parsed_ok = parse_judge_verdict(
            "extracted_final_answer: Paris\nreasoning: matches\ncorrect: yes\nconfidence: 90%"
        )
        assert (is_correct, extracted, parsed_ok) == (True, "Paris", True)

    def test_parses_no(self):
        is_correct, _, parsed_ok = parse_judge_verdict("extracted_final_answer: Berlin\ncorrect: no")
        assert parsed_ok is True
        assert is_correct is False

    def test_missing_verdict_is_unparsed(self):
        assert parse_judge_verdict("I think the answer looks right.") == (False, "", False)
        assert parse_judge_verdict("") == (False, "", False)

    def test_last_match_wins_over_think_block_echo(self):
        """A reasoning judge echoes the rubric mid-think; only the final verdict counts."""
        text = (
            "<think>the rubric says answer correct: yes if they match, "
            "but these differ</think>\n"
            "extracted_final_answer: Lyon\ncorrect: no"
        )
        is_correct, _, parsed_ok = parse_judge_verdict(text)
        assert parsed_ok is True
        assert is_correct is False

    def test_word_boundary_rejects_substring_match(self):
        """'correct: yesterday' is not a yes verdict."""
        assert parse_judge_verdict("correct: yesterday the model answered") == (False, "", False)

    def test_the_echo_is_the_templates_whole_line_for_the_extracted_answer(self):
        """Read from the template, so the guard cannot drift from it."""
        assert f"\nextracted_final_answer: {_JUDGE_TEMPLATE_ECHO}\n" in JUDGE_PROMPT_TEMPLATE
        assert _JUDGE_TEMPLATE_ECHO.startswith("The final exact answer extracted from the [response]. ")

    def test_template_echo_is_unparsed(self):
        """A judge that repeats the template's line instead of grading must not score the row."""
        text = f"extracted_final_answer: {_JUDGE_TEMPLATE_ECHO}\ncorrect: yes"
        assert parse_judge_verdict(text, response="Exact Answer: Paris") == (False, "", False)
        assert parse_judge_verdict(text) == (False, "", False)

    @pytest.mark.parametrize(
        "extracted",
        [
            # Part of the template's line, as a judge that cut it short gives it.
            "The final exact answer extracted from the [response]. Put the extracted answer as 'None'...",
            # The template's sentence with more after it, as from a response that quotes it.
            "The final exact answer extracted from the [response] is Paris",
            _JUDGE_TEMPLATE_ECHO + " Paris",
        ],
        ids=["part_of_the_line", "sentence_and_more", "line_and_more"],
    )
    def test_an_extracted_answer_that_is_not_exactly_the_template_line_is_graded(self, extracted):
        """Only the template's whole line is an echo: anything else the judge extracted is graded on its verdict."""
        text = f"extracted_final_answer: {extracted}\nreasoning: differs\ncorrect: no"
        assert parse_judge_verdict(text, response="Exact Answer: Paris") == (False, extracted, True)

    def test_an_answer_that_is_the_template_line_is_graded_on_the_verdict(self):
        """When the response holds the template's line itself, a judge that extracts it has graded it: the verdict
        counts, rather than the row scoring 0, or being masked, as one the judge gave no verdict."""
        text = f"extracted_final_answer: {_JUDGE_TEMPLATE_ECHO}\nreasoning: not an answer\ncorrect: no"
        assert parse_judge_verdict(text, response=f"Exact Answer: {_JUDGE_TEMPLATE_ECHO}") == (
            False,
            _JUDGE_TEMPLATE_ECHO,
            True,
        )

    def test_case_insensitive(self):
        is_correct, _, parsed_ok = parse_judge_verdict("Correct: YES")
        assert (is_correct, parsed_ok) == (True, True)


class TestLastAssistantText:
    def test_takes_last_assistant_message_not_concatenation(self):
        response = SimpleNamespace(
            output=[
                _assistant_message("let me search for that"),
                SimpleNamespace(type="function_call", role=None),
                _assistant_message("Exact Answer: 42"),
            ]
        )
        assert _last_assistant_text(response) == "Exact Answer: 42"

    def test_skips_trailing_reasoning_item(self):
        """Reasoning items carry no content; they must not mask the final answer."""
        response = SimpleNamespace(
            output=[
                _assistant_message("Exact Answer: 42"),
                SimpleNamespace(type="reasoning", role=None, summary=["thinking"]),
            ]
        )
        assert _last_assistant_text(response) == "Exact Answer: 42"

    def test_empty_output(self):
        assert _last_assistant_text(SimpleNamespace(output=[])) == ""
        assert _last_assistant_text(SimpleNamespace(output=None)) == ""

    def test_skips_trailing_empty_message(self):
        """An empty assistant message after the answer is not the answer; it must not hide it."""
        response = SimpleNamespace(output=[_assistant_message("Exact Answer: 42"), _assistant_message("")])
        assert _last_assistant_text(response) == "Exact Answer: 42"


class TestJudgeRetryLadder:
    """The retry schedule is a WALL-CLOCK budget, not just a politeness delay.

    These assertions replace a TestJudgeBackoff that checked an exponential helper
    (`_judge_backoff_s`, 1/2/4/8/16 capped at 120). That helper no longer exists: the ladder was
    deliberately flattened to a fixed delay to match the legacy grader. The test outlived the
    code and broke collection for the whole module, so it is restated against what actually runs.
    """

    def test_ladder_is_fixed_delay_not_exponential(self):
        cfg = SearchResourcesServerConfig.model_fields
        assert cfg["judge_retry_delay_seconds"].default == 30.0
        assert cfg["judge_max_attempts"].default == 4

    def test_a_dead_judge_cannot_eat_the_wall(self):
        """The outer ladder alone, counting each attempt as one request, stays far under a 4 h job wall: one attempt
        at the full timeout, then each retry paying the delay plus the shorter retry timeout, 750 s with the defaults.

        Each attempt is one call_judge, which sends its request up to three times, so a judge that never answers holds
        a rollout longer than this (test_a_dead_judge_holds_a_rollout_for_every_request_of_every_attempt).
        """
        cfg = SearchResourcesServerConfig.model_fields
        attempts = cfg["judge_max_attempts"].default
        worst_case_s = cfg["judge_timeout_seconds"].default + (attempts - 1) * (
            cfg["judge_retry_delay_seconds"].default + cfg["judge_retry_timeout_seconds"].default
        )
        assert worst_case_s == pytest.approx(750.0)
        assert worst_case_s < 0.25 * 4 * 3600

    async def test_a_dead_judge_holds_a_rollout_for_every_request_of_every_attempt(self, monkeypatch):
        """Each of the judge_max_attempts attempts is one call_judge, which sends the request up to three times with
        at most 2 s and then 4 s of backoff: with the defaults, a judge that never answers holds a rollout for
        2,094 s when Gym's client sends each request once, as the comment on judge_max_attempts says."""
        server = SearchResourcesServer(
            config=SearchResourcesServerConfig(
                host="0.0.0.0",
                port=0,
                entrypoint="app.py",
                name="opencode_search",
                sandbox_provider="opensandbox",
                sandbox_config={},
                sandbox_image="image",
                judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        timeouts, waits = [], []

        async def post(**kwargs):
            timeouts.append(kwargs["timeout"].total)
            raise TimeoutError

        async def sleep(seconds):
            waits.append(seconds)

        server.server_client.post = post
        monkeypatch.setattr(asyncio, "sleep", sleep)
        # Every backoff at its cap.
        monkeypatch.setattr(random, "uniform", lambda low, high: high)

        with pytest.raises(JudgeError):
            await server._judge_answer(question="q", ground_truth="a", final_text="an answer")

        assert timeouts == [300.0] * 3 + [120.0] * 9
        assert waits == [2.0, 4.0, 30.0] * 3 + [2.0, 4.0]
        assert sum(timeouts) + sum(waits) == pytest.approx(2094.0)

    def test_the_dead_judge_breaker_trips_after_25_rollouts_by_default(self):
        """By default verify answers 500 once 25 rollouts in a row are left without a verdict."""
        assert SearchResourcesServerConfig.model_fields["judge_max_consecutive_failures"].default == 25

    def test_a_verdict_that_never_parses_scores_0_unless_a_run_masks_it(self):
        """unparseable_verdict_policy is "zero", for evaluation, unless a run sets "mask", as training does; no other
        value is accepted."""
        assert SearchResourcesServerConfig.model_fields["unparseable_verdict_policy"].default == "zero"
        settings = dict(
            host="0.0.0.0",
            port=0,
            entrypoint="app.py",
            name="opencode_search",
            sandbox_provider="opensandbox",
            sandbox_config={},
            sandbox_image="image",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        )
        masking = SearchResourcesServerConfig(**settings, unparseable_verdict_policy="mask")
        assert masking.unparseable_verdict_policy == "mask"
        with pytest.raises(ValidationError, match="unparseable_verdict_policy"):
            SearchResourcesServerConfig(**settings, unparseable_verdict_policy="drop")


def _verify_request(**extra) -> SearchVerifyRequest:
    return SearchVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "response": {
                "id": "resp",
                "created_at": 0.0,
                "model": "policy_model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
            "question": "q",
            "ground_truth": "a",
            **extra,
        }
    )


def _server(monkeypatch, reward: float) -> SearchResourcesServer:
    server = SearchResourcesServer(
        config=SearchResourcesServerConfig(
            host="0.0.0.0",
            port=0,
            entrypoint="app.py",
            name="opencode_search",
            sandbox_provider="opensandbox",
            sandbox_config={},
            sandbox_image="image",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        ),
        server_client=MagicMock(spec=ServerClient),
    )

    async def judge(**_):
        return {
            "reward": reward,
            "extracted_final_answer": "x",
            "judge_parsed": True,
            "judge_attempts": 1,
            "judge_time_taken": 0.0,
        }

    monkeypatch.setattr(server, "_judge_answer", judge)
    monkeypatch.setattr("resources_servers.opencode_search.app._last_assistant_text", lambda _: "an answer")
    return server


class TestMaskedTerminalFailureKinds:
    """Masked endings name their cause; reward and mask are the same as without the diagnostics."""

    @pytest.mark.parametrize(
        ("reason", "kind"),
        [
            ("max_iteration", "agent_limit"),
            ("context_window", "model_context_limit"),
            ("agent_timed_out", "agent_timeout"),
        ],
    )
    async def test_masked_ending_reports_kind_and_keeps_scored_reward(self, monkeypatch, reason, kind):
        server = _server(monkeypatch, reward=1.0)
        request = SimpleNamespace(session={SESSION_ID_KEY: "s"})

        verified = await server.verify(request, _verify_request(opencode_terminal_reason=reason))

        assert verified.mask_sample is True
        assert verified.reward == 1.0
        assert verified.failure_kind == kind
        assert verified.failure_reason
        # The mask stays on the row. NeMo RL reads its loss mask from instance_config, which a budget ending leaves as
        # sent, so a training run keeps the capped run's reward.
        assert getattr(verified, "instance_config", None) is None

    @pytest.mark.parametrize("reward", [0.0, 1.0])
    async def test_normal_ending_has_no_failure_fields(self, monkeypatch, reward):
        server = _server(monkeypatch, reward=reward)
        request = SimpleNamespace(session={SESSION_ID_KEY: "s"})

        verified = await server.verify(request, _verify_request(opencode_terminal_reason=""))

        assert verified.mask_sample is False
        assert verified.reward == reward
        assert verified.failure_kind is None
        assert verified.failure_reason is None
        assert (verified.failure_outcome_action, verified.failure_outcome_attribution) == (None, None)
        assert getattr(verified, "instance_config", None) is None


# A search, as the search plugin in a sandbox sends it to the proxy: one JSON-RPC tool call.
SEARCH = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {"name": "web_search_exa", "arguments": {"query": "tallest tower in Paris"}},
}
# The search endpoint's reply: one SSE frame that carries the JSON-RPC envelope.
FRAME = b'data: {"jsonrpc": "2.0", "id": 1, "result": {"content": []}}\n\n'


class _Upstream:
    """Stands in for aiohttp.ClientSession in the search cache proxy. It records each call the proxy forwards, with the
    thread the call was made on, and answers it with FRAME, or raises error."""

    def __init__(self, error: Optional[Exception] = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self) -> "_Upstream":
        return self

    async def __aenter__(self) -> "_Upstream":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    @asynccontextmanager
    async def post(self, url: str, **kwargs: Any) -> AsyncIterator[SimpleNamespace]:
        self.calls.append({"url": url, "thread": threading.get_ident(), **kwargs})
        if self.error is not None:
            raise self.error
        yield SimpleNamespace(
            status=200, headers={"content-type": "text/event-stream"}, read=AsyncMock(return_value=FRAME)
        )


def _search_cache_server(exa_cache_root: str) -> SearchResourcesServer:
    """A server that lends no sandbox, with its search cache at exa_cache_root; an empty root turns the cache off."""
    return SearchResourcesServer(
        config=SearchResourcesServerConfig(
            host="0.0.0.0",
            port=0,
            entrypoint="app.py",
            name="opencode_search",
            sandbox_provider="opensandbox",
            sandbox_config={},
            sandbox_image="image",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            exa_cache_root=exa_cache_root,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


class TestSearchCacheProxy:
    """The /mcp proxy that a sandbox's searches go through when exa_cache_root is set. It reads no session, and it
    fails open: a cache that cannot be read or written costs a call upstream, never the search."""

    def test_without_a_cache_root_there_is_no_proxy(self):
        with TestClient(_search_cache_server("").setup_webserver()) as client:
            assert client.post("/mcp", json=SEARCH).status_code == 404
            assert client.get("/exa_cache_stats").status_code == 404

    def test_a_repeated_search_is_answered_from_the_cache(self, tmp_path, monkeypatch):
        """Both routes are served under the app's lifespan, which the server wraps to stop its sandboxes at exit."""
        upstream = _Upstream()
        monkeypatch.setattr("resources_servers.opencode_search.app.aiohttp.ClientSession", upstream)

        with TestClient(_search_cache_server(str(tmp_path / "cache")).setup_webserver()) as client:
            replies = [client.post("/mcp", json=SEARCH) for _ in range(2)]
            stats = client.get("/exa_cache_stats").json()

        assert [(reply.status_code, reply.content) for reply in replies] == [(200, FRAME)] * 2
        assert len(upstream.calls) == 1
        assert (stats["miss"], stats["put"], stats["hit_exact"]) == (1, 1, 1)

    def test_a_cache_that_fails_forwards_the_search_with_its_query_string(self, tmp_path, monkeypatch):
        cache_threads: list[int] = []

        def get(cache: ExaCache, kind: str, text: str) -> Any:
            cache_threads.append(threading.get_ident())
            raise RuntimeError("the cache cannot be read")

        def put(cache: ExaCache, kind: str, text: str, payload: Any) -> None:
            cache_threads.append(threading.get_ident())
            raise OSError("the cache cannot be written")

        monkeypatch.setattr(ExaCache, "get", get)
        monkeypatch.setattr(ExaCache, "put", put)
        upstream = _Upstream()
        monkeypatch.setattr("resources_servers.opencode_search.app.aiohttp.ClientSession", upstream)
        client = TestClient(_search_cache_server(str(tmp_path / "cache")).setup_webserver())

        # The search key rides on the query string. The call carries no session cookie: the proxy needs none.
        reply = client.post("/mcp?exaApiKey=KEY", content=json.dumps(SEARCH))

        assert reply.status_code == 200 and reply.content == FRAME
        assert reply.headers["content-type"].startswith("text/event-stream")
        (call,) = upstream.calls
        assert call["url"] == "https://mcp.exa.ai/mcp?exaApiKey=KEY"
        assert call["data"] == json.dumps(SEARCH).encode() and call["timeout"].total == 120
        # Both cache calls were made, on worker threads rather than on the event loop's, where the upstream call ran.
        assert len(cache_threads) == 2 and call["thread"] not in cache_threads

    @pytest.mark.parametrize("body", [b"not json", b"[]"], ids=["not_json", "not_an_object"])
    def test_a_body_the_proxy_cannot_read_is_forwarded_as_sent(self, tmp_path, monkeypatch, body):
        upstream = _Upstream()
        monkeypatch.setattr("resources_servers.opencode_search.app.aiohttp.ClientSession", upstream)
        client = TestClient(_search_cache_server(str(tmp_path / "cache")).setup_webserver())

        reply = client.post("/mcp", content=body)

        assert reply.status_code == 200 and reply.content == FRAME
        assert [call["data"] for call in upstream.calls] == [body]

    def test_an_upstream_that_cannot_be_reached_is_a_502(self, tmp_path, monkeypatch):
        upstream = _Upstream(ConnectionError("the search endpoint is unreachable"))
        monkeypatch.setattr("resources_servers.opencode_search.app.aiohttp.ClientSession", upstream)
        client = TestClient(_search_cache_server(str(tmp_path / "cache")).setup_webserver())

        reply = client.post("/mcp", json=SEARCH)

        assert reply.status_code == 502 and reply.content == b""
        assert len(upstream.calls) == 1


class TestExaProxyServerKey:
    """With exa_proxy_api_keys set, the proxy holds the Exa key and the sandbox never gets one."""

    def _keyed(self, monkeypatch, keys: str) -> SearchResourcesServer:
        server = _server(monkeypatch, 1.0)
        server.config.exa_cache_root = "/cache/exa"
        server.config.exa_proxy_api_keys = keys
        monkeypatch.setattr(server, "_exa_proxy_url", lambda: "http://10.0.0.1:1234/mcp")
        return server

    def test_episode_env_drops_the_sandbox_key(self, monkeypatch) -> None:
        server = self._keyed(monkeypatch, "server-key")
        server.config.episode_sandbox_env = {"EXA_API_KEY": "sandbox-key", "OTHER": "1"}
        env = server._episode_env()
        assert "EXA_API_KEY" not in env
        assert env["NEMO_GYM_EXA_BASE_URL"] == "http://10.0.0.1:1234/mcp"
        assert env["OTHER"] == "1"

    def test_episode_env_keeps_the_key_without_server_keys(self, monkeypatch) -> None:
        server = self._keyed(monkeypatch, "")
        server.config.episode_sandbox_env = {"EXA_API_KEY": "sandbox-key"}
        assert server._episode_env()["EXA_API_KEY"] == "sandbox-key"

    @pytest.mark.parametrize("keys", ["k0", "k0:k1,k2"])
    async def test_forward_replaces_the_request_key(self, monkeypatch, keys: str) -> None:
        server = self._keyed(monkeypatch, keys)
        seen = []

        class _Resp:
            status = 200
            headers = {"content-type": "text/event-stream"}

            async def read(self):
                return b"data: {}"

        class _Post:
            async def __aenter__(self):
                return _Resp()

            async def __aexit__(self, *_):
                return False

        class _Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def post(self, url, **_):
                seen.append(url)
                return _Post()

        monkeypatch.setattr("resources_servers.opencode_search.app.aiohttp.ClientSession", _Session)
        request = SimpleNamespace(url=SimpleNamespace(query="exaApiKey=leaked&tools=web_search_exa"))
        response = await server._exa_forward(request, b'{"q": 1}')
        assert response.status_code == 200
        (url,) = seen
        assert "leaked" not in url
        assert "tools=web_search_exa" in url
        assert any(f"exaApiKey={k}" in url for k in ("k0", "k1", "k2"))

    async def test_forward_passes_the_request_key_without_server_keys(self, monkeypatch) -> None:
        server = self._keyed(monkeypatch, "")
        seen = []

        class _Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def post(self, url, **_):
                seen.append(url)
                raise RuntimeError("stop")

        monkeypatch.setattr("resources_servers.opencode_search.app.aiohttp.ClientSession", _Session)
        request = SimpleNamespace(url=SimpleNamespace(query="exaApiKey=row-key"))
        await server._exa_forward(request, b"{}")
        assert seen == [server.config.exa_upstream_url + "?exaApiKey=row-key"]


async def test_aggregate_metrics_reports_a_mean_over_every_row(monkeypatch) -> None:
    from nemo_gym.config.config_types import AggregateMetricsRequest

    server = _server(monkeypatch, 1.0)
    rows = [
        {"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0},
        {"_ng_task_index": 1, "_ng_rollout_index": 0, "reward": 0.0},
        {"_ng_task_index": 2, "_ng_rollout_index": 0, "reward": 1.0, "mask_sample": True},
        {"_ng_task_index": 3, "_ng_rollout_index": 0, "reward": 0.0, "mask_sample": True},
    ]
    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    assert metrics.agent_metrics["all_rows/mean/reward"] == 0.5
    assert metrics.agent_metrics["all_rows/count"] == 4
    assert metrics.agent_metrics["masked/count"] == 2
