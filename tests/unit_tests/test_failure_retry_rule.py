# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gym, not the agent, decides whether a failure is retried (infra) or final (the model's, scored 0)."""

import json
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from aiohttp import ClientConnectorError, ClientResponseError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from multidict import CIMultiDict
from omegaconf import OmegaConf

from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from nemo_gym.config_types import AgentServerRef
from nemo_gym.failure_kinds import (
    EVIDENCE,
    FAILURE_CAUSE_HEADER,
    FAILURE_KINDS,
    FAILURE_RULES,
    INFRA,
    MODEL,
    failure_cause_of,
    failure_cause_of_text,
    is_retryable,
)
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.rollout_collection import (
    NG_FAILURE_CAUSE_KEY,
    NG_FAILURE_CLASS_KEY,
    RolloutCollectionConfig,
    RolloutCollectionHelper,
    _agent_request_failure_row,
    _failure_rows_counted_as_zero,
    failure_cause_of_row,
    failures_path_for,
    is_terminal_failure,
)
from nemo_gym.server_utils import BaseServerConfig, ServerClient, SimpleServer


def _http_error(status: int, body: bytes = b"", headers: dict | None = None) -> ClientResponseError:
    request_info = SimpleNamespace(method="POST", url="http://agent/run", real_url="http://agent/run")
    error = ClientResponseError(
        request_info=request_info, history=(), status=status, message="boom", headers=CIMultiDict(headers or {})
    )
    error.response_content = body
    return error


def _refused() -> ClientConnectorError:
    return ClientConnectorError(
        SimpleNamespace(host="model", port=8000, ssl=False), OSError(111, "Connection refused")
    )


class TestEveryKindHasARule:
    def test_every_registered_kind_has_a_rule(self) -> None:
        assert FAILURE_KINDS <= FAILURE_RULES.keys()

    def test_the_ambiguous_kinds_are_split_by_evidence(self) -> None:
        for kind in (
            "agent_run_error",
            "agent_request_failed",
            "agent_timeout",
            "verifier_error",
            "agent_command_not_allowed",
        ):
            assert FAILURE_RULES[kind] == EVIDENCE
            assert is_retryable(kind, cause=INFRA)
            assert not is_retryable(kind, cause=MODEL)


class TestWhereTheFailureHappened:
    @pytest.mark.parametrize("status", [401, 403, 408, 429, 500, 502, 503, 504])
    def test_a_service_status_is_infra(self, status: int) -> None:
        assert failure_cause_of(_http_error(status)) == INFRA

    @pytest.mark.parametrize("status", [400, 404, 413, 422])
    def test_a_status_answering_what_the_model_sent_is_the_models(self, status: int) -> None:
        """A 400 for a context overflow answers the model's own request."""
        assert failure_cause_of(_http_error(status)) == MODEL

    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (422, b'{"error": {"message": "Budget has been exceeded! Current cost: 101.2, Max budget: 100.0"}}'),
            (402, b'{"error": "Payment Required"}'),
            (429, b'{"error": {"code": "insufficient_quota", "message": "You exceeded your current quota"}}'),
            (400, b'{"error": {"type": "insufficient_quota", "message": "quota exhausted for this key"}}'),
        ],
        ids=["422-budget-exceeded", "402", "429-insufficient-quota", "400-insufficient-quota"],
    )
    def test_a_spent_budget_or_quota_is_infra(self, status: int, body: bytes) -> None:
        """The endpoint refused to serve the key, whatever the model sent: retry, never score the model 0."""
        assert failure_cause_of(_http_error(status, body)) == INFRA
        # The same failure relayed as text, as the sidecar keeps it.
        assert failure_cause_of_text(f"{status}, message='Unprocessable Entity' {body.decode()}") == INFRA

    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (400, b'{"error": {"code": "malformed_authorization", "message": "Malformed authorization header"}}'),
            (400, b'{"error": {"message": "Incorrect API key provided: sk-****"}}'),
            (400, b'{"error": {"message": "Invalid API key"}}'),
            (400, b'{"error": {"message": "Missing API key. Please provide an Authorization header."}}'),
            (403, b'{"error": {"message": "This model requires approved access"}}'),
        ],
        ids=["malformed-authorization", "incorrect-key", "invalid-key", "missing-key", "requires-approved-access"],
    )
    def test_a_broken_credential_is_infra_on_any_endpoint(self, status: int, body: bytes) -> None:
        """A key or route the model does not control: retried, never the model's 0, also on the policy endpoint."""
        error = _http_error(status, body)
        error.request_info = SimpleNamespace(
            method="POST", url="http://policy/v1/responses", real_url="http://policy/v1/responses", headers={}
        )
        assert failure_cause_of(error) == INFRA
        assert failure_cause_of_text(f"{status}, message='Bad Request' {body.decode()}") == INFRA

    @pytest.mark.parametrize(
        "body",
        [b'{"error": "prompt is longer than the context window"}', b'{"error": "invalid tool schema"}', b""],
        ids=["context-overflow", "bad-request", "no-body"],
    )
    def test_other_4xx_keep_their_label(self, body: bytes) -> None:
        assert failure_cause_of(_http_error(422, body)) == MODEL
        assert failure_cause_of_text(f"422, message='Unprocessable Entity' {body.decode()}") == MODEL

    @pytest.mark.parametrize(
        "url",
        ["http://policy:8000/v1/responses", "http://judge:8000/v1/chat/completions", "http://judge/v1/completions"],
        ids=["policy-responses", "judge-chat", "completions"],
    )
    def test_a_404_from_a_model_or_judge_route_is_infra(self, url: str) -> None:
        """Model not found or a wrong route is a deployment or config error, never the model's output."""
        error = _http_error(404, b'{"error": "The model `policy` does not exist."}')
        error.request_info = SimpleNamespace(method="POST", url=url, real_url=url, headers={})
        assert failure_cause_of(error) == INFRA
        # The same failure relayed as text, as aiohttp words it and the sidecar keeps it.
        assert failure_cause_of_text(str(error)) == INFRA

    def test_a_404_from_another_call_keeps_its_label(self) -> None:
        # The agent's own /run route, and a 404 that names no route.
        assert failure_cause_of(_http_error(404)) == MODEL
        assert failure_cause_of_text("404, message='Not Found', url='http://agent/run'") == MODEL
        assert failure_cause_of_text("404, message='Not Found'") == MODEL

    def test_a_sandbox_exec_404_is_unchanged(self) -> None:
        """The sandbox rule decides an exec 404 while the sandbox is alive; the model-route 404 rule must not."""
        message = "Failed to run command: execution not found"
        relayed = f"SandboxApiException: {message} (status 404)"
        assert failure_cause_of_text(relayed) == failure_cause_of_text(f"SandboxApiException: {message}")

        # The live exception needs the optional sandbox SDK; without it only the relayed-text check runs.
        sdk = pytest.importorskip("opensandbox.exceptions", reason="opensandbox SDK is not installed")
        live = sdk.SandboxApiException(message, status_code=404)
        without_status = sdk.SandboxApiException(message)
        assert failure_cause_of(live) == failure_cause_of(without_status)

    def test_a_connection_error_wrapped_in_an_agent_error_is_infra(self) -> None:
        try:
            try:
                raise _refused()
            except ClientConnectorError as inner:
                raise RuntimeError("agent step failed") from inner
        except RuntimeError as error:
            assert failure_cause_of(error) == INFRA

    def test_an_exception_group_is_searched(self) -> None:
        assert failure_cause_of(ExceptionGroup("task group", [ValueError("x"), _refused()])) == INFRA

    def test_a_parse_error_of_the_models_output_is_the_models(self) -> None:
        try:
            json.loads("{not json")
        except json.JSONDecodeError as error:
            assert failure_cause_of(error) == MODEL

    def test_an_agents_own_time_budget_is_the_models(self) -> None:
        assert failure_cause_of(TimeoutError("max steps or time used up")) == MODEL

    def test_text_names_a_dead_peer(self) -> None:
        assert failure_cause_of_text("ClientConnectorError(ConnectionKey(host='model', port=8000))") == INFRA
        assert failure_cause_of_text("Cannot connect to host model:8000 ssl:default") == INFRA
        assert failure_cause_of_text("503, message='Service Unavailable', url='http://judge/v1'") == INFRA
        assert failure_cause_of_text("litellm.RateLimitError: judge over quota") == INFRA

    def test_text_naming_only_a_model_output_error_is_the_models(self) -> None:
        assert failure_cause_of_text("Tau2MalformedToolCallError: tool call 0 arguments is invalid JSON") == MODEL
        assert failure_cause_of_text("400, message='prompt is longer than the context window'") == MODEL


class TestAFinalFailureScoresZero:
    """A final failure is a score of 0, even when its row carries a reward."""

    def _sidecar(self, tmp_path: Path, row: dict) -> Path:
        failures = tmp_path / "out_failures.jsonl"
        failures.write_bytes(orjson.dumps({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, **row}) + b"\n")
        return failures

    def test_a_final_row_with_a_reward_counts_as_zero(self, tmp_path: Path) -> None:
        row = {NG_FAILURE_CLASS_KEY: "agent_run_error", "_ng_failure_cause": "model", "reward": 0.8}
        [counted] = _failure_rows_counted_as_zero([self._sidecar(tmp_path, row)], [], set())
        assert counted["reward"] == 0.0

    def test_a_selected_retryable_row_keeps_its_reward(self, tmp_path: Path) -> None:
        row = {NG_FAILURE_CLASS_KEY: "judge_failed", "reward": 0.8}
        [counted] = _failure_rows_counted_as_zero([self._sidecar(tmp_path, row)], ["judge_failed"], set())
        assert counted["reward"] == 0.8


class TestTheRuleOnSidecarRows:
    def test_a_forbidden_command_is_final(self) -> None:
        assert is_terminal_failure(
            {NG_FAILURE_CLASS_KEY: "agent_run_error", "failure_kind": "agent_command_not_allowed"}
        )

    def test_an_unknown_class_without_evidence_is_final(self) -> None:
        """The safe default: a failure Gym cannot place gives the model no free retry."""
        assert is_terminal_failure({NG_FAILURE_CLASS_KEY: "invented_by_an_agent", "error": "something broke"})

    def test_an_unknown_class_with_infra_evidence_is_retried(self) -> None:
        assert not is_terminal_failure({NG_FAILURE_CLASS_KEY: "invented_by_an_agent", NG_FAILURE_CAUSE_KEY: INFRA})

    def test_a_server_may_withhold_a_retry_but_not_grant_one(self) -> None:
        assert is_terminal_failure({NG_FAILURE_CLASS_KEY: "judge_failed", "_ng_failure_terminal": True})
        assert is_terminal_failure({NG_FAILURE_CLASS_KEY: "tau2_malformed_tool_call", "_ng_failure_terminal": False})


class TestAgentRunErrorBySource:
    """Killing the model server mid-run gives agent_run_error with status 500."""

    def test_a_500_carrying_a_dead_model_server_is_retried(self) -> None:
        error = _http_error(
            500, b"Hit an exception in agent calling an inner server: ClientConnectorError(model:8000 refused)"
        )
        row = _agent_request_failure_row(error, 500)

        assert row[NG_FAILURE_CLASS_KEY] == "agent_run_error"
        assert row[NG_FAILURE_CAUSE_KEY] == INFRA
        assert not is_terminal_failure(row)

    def test_a_500_from_a_model_output_parse_error_is_final(self) -> None:
        error = _http_error(500, b"JSONDecodeError('Expecting value: line 1 column 1 (char 0)')")
        row = _agent_request_failure_row(error, 500)

        assert row[NG_FAILURE_CLASS_KEY] == "agent_run_error"
        assert row[NG_FAILURE_CAUSE_KEY] == MODEL
        assert is_terminal_failure(row)

    def test_the_cause_a_gym_server_named_wins_over_its_body(self) -> None:
        error = _http_error(500, b"anything", {FAILURE_CAUSE_HEADER: INFRA})
        assert _agent_request_failure_row(error, 500)[NG_FAILURE_CAUSE_KEY] == INFRA

    def test_no_reply_at_all_is_retried(self) -> None:
        row = _agent_request_failure_row(_refused(), None)

        assert row[NG_FAILURE_CLASS_KEY] == "agent_request_failed"
        assert not is_terminal_failure(row)


def _agent(breaks_with: BaseException) -> TestClient:
    """A Gym agent server, with Gym's own exception middleware, whose /run raises ``breaks_with``."""
    app = FastAPI()

    @app.post("/run")
    async def run() -> dict:
        raise breaks_with

    server = SimpleNamespace(
        config=SimpleNamespace(name="agent"), get_session_middleware_key=lambda: "SimpleAgent___agent"
    )
    SimpleServer.setup_exception_middleware(server, app)
    return TestClient(app, raise_server_exceptions=False)


class _Relayed:
    """What legacy_agent's ServerClient hands back: the agent's reply, status and headers unchanged."""

    def __init__(self, response) -> None:
        self.status = response.status_code
        self.ok = response.is_success
        self.headers = CIMultiDict(response.headers.items())
        self._body = response.content

    async def read(self) -> bytes:
        return self._body


class _RelayClient(ServerClient):
    agent: object

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Relayed:
        return _Relayed(self.agent.post("/run", content=kwargs.get("data") or b"{}"))


def _through_legacy_agent(breaks_with: BaseException):
    """POST /run to legacy_agent, which relays to a real Gym agent server that raises ``breaks_with``."""
    client = _RelayClient.model_construct(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create({}),
        agent=_agent(breaks_with),
    )
    relay = LegacyAgentEnvironmentServer(
        config=LegacyAgentEnvironmentServerConfig(
            name="environment",
            host="environment",
            port=8002,
            entrypoint="app.py",
            agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
        ),
        server_client=client,
    )
    return TestClient(relay.setup_webserver()).post("/run", json={})


def _collector_row(response) -> dict:
    """The sidecar row rollout collection builds from the relayed reply."""
    request_info = SimpleNamespace(method="POST", url="http://environment/run", real_url="http://environment/run")
    error = ClientResponseError(
        request_info=request_info,
        history=(),
        status=response.status_code,
        message="Internal Server Error",
        headers=CIMultiDict(response.headers.items()),
    )
    error.response_content = response.content
    return _agent_request_failure_row(error, response.status_code)


class TestThroughLegacyAgent:
    """Many benchmarks run their agent behind legacy_agent, so the cause must survive the relay."""

    def test_a_dead_model_server_is_retried(self) -> None:
        response = _through_legacy_agent(_refused())

        assert response.status_code == 500
        assert response.headers[FAILURE_CAUSE_HEADER] == INFRA
        assert not is_terminal_failure(_collector_row(response))

    def test_a_judge_429_or_500_is_retried(self) -> None:
        for status in (429, 500):
            response = _through_legacy_agent(_http_error(status, b"judge overloaded"))

            assert response.headers[FAILURE_CAUSE_HEADER] == INFRA
            assert not is_terminal_failure(_collector_row(response))

    def test_a_malformed_model_tool_call_is_final(self) -> None:
        try:
            json.loads('{"path": "unterminated')
        except json.JSONDecodeError as error:
            response = _through_legacy_agent(error)

        assert response.headers[FAILURE_CAUSE_HEADER] == MODEL
        row = _collector_row(response)
        assert row[NG_FAILURE_CLASS_KEY] == "agent_run_error"
        assert is_terminal_failure(row)


class TestJudgeFailed:
    @pytest.mark.parametrize(
        "judge_error",
        [
            "ClientResponseError: 400, message='Bad Request'",
            "ClientResponseError: 404, message='Not Found'",
            "ValidationError: judge reply did not parse",
            "",
        ],
    )
    def test_judge_failed_is_retried_whatever_its_status(self, judge_error: str) -> None:
        """Gym main records no structured judge status, so judge_failed alone decides: infra, retried."""
        row = {
            NG_FAILURE_CLASS_KEY: "judge_failed",
            "failure_kind": "judge_failed",
            "_ng_failure_judge_error": judge_error,
            "_ng_failure_message": judge_error,
        }

        assert failure_cause_of_row(row) == INFRA
        assert not is_terminal_failure(row)


class TestResume:
    """End to end on the collector's resume path: a final failure is not dispatched again and scores 0."""

    async def test_a_malformed_tool_call_is_not_rerun_and_an_infra_failure_is(self, tmp_path: Path) -> None:
        output = tmp_path / "out.jsonl"
        output.write_bytes(b"")
        rows = [
            {"responses_create_params": {"input": []}, TASK_INDEX_KEY_NAME: i, ROLLOUT_INDEX_KEY_NAME: 0}
            for i in range(3)
        ]
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "in.jsonl"), output_jsonl_fpath=str(output), resume_from_cache=True
        )
        config.materialized_jsonl_fpath.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
        dead_model = _agent_request_failure_row(
            _http_error(500, b"calling an inner server: ClientConnectorError(model:8000)"), 500
        )
        failures = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "tau2_malformed_tool_call"},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, **dead_model},
        ]
        failures_path_for(output).write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in failures))

        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)

        assert [row[TASK_INDEX_KEY_NAME] for row in input_rows] == [1, 2]


class TestHarnessFailurePaths:
    """Harness failure paths, through the rule Gym applies to the sidecar row."""

    def test_tau2_malformed_tool_call_is_final_and_a_dead_policy_is_retried(self) -> None:
        malformed = {NG_FAILURE_CLASS_KEY: "tau2_malformed_tool_call", "error": "Tau2MalformedToolCallError: bad"}
        dead_policy = {
            NG_FAILURE_CLASS_KEY: "tau2_agent_error",
            "error": "APIConnectionError: litellm.APIConnectionError: Connection error.",
        }
        assert is_terminal_failure(malformed)
        assert not is_terminal_failure(dead_policy)

    def test_tau2_agent_error_without_infra_evidence_is_final(self) -> None:
        """Any model error the agent raises, such as an unusable answer, gets no free retry."""
        assert is_terminal_failure({NG_FAILURE_CLASS_KEY: "tau2_agent_error", "error": "KeyError: 'name'"})

    @pytest.mark.parametrize(
        ("failure_type", "final"),
        [
            ("NonZeroAgentExitCodeError", True),
            ("AgentTimeoutError", True),
            ("VerifierOutputParseError", True),
            # Terminal-Bench scores a trial that leaves no reward file as 0.
            ("RewardFileNotFoundError", True),
            ("EnvironmentStartTimeoutError", False),
            ("HealthcheckError", False),
            ("AgentSetupTimeoutError", False),
            ("AgentInfrastructureError", False),
            ("SeededSessionExpired", False),
        ],
    )
    def test_terminal_bench_exits_split_by_who_failed(self, failure_type: str, final: bool) -> None:
        row = {NG_FAILURE_CLASS_KEY: "infrastructure_error", "_ng_failure_type": failure_type}
        assert is_terminal_failure(row) is final
