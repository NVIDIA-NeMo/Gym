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

import asyncio
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
    MASKED_SAMPLE_FAILURE_CLASS,
    NG_FAILURE_CAUSE_KEY,
    NG_FAILURE_CLASS_KEY,
    RolloutCollectionConfig,
    RolloutCollectionHelper,
    _agent_request_failure_row,
    _CompletedRollout,
    _counted_failure_row,
    _failure_rows_counted_as_zero,
    _masked_result_names_infra,
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
        assert is_terminal_failure({NG_FAILURE_CLASS_KEY: "timeout_exceeded", "_ng_failure_terminal": False})


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
            # The agent's own time budget: a final model failure.
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "timeout_exceeded"},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, **dead_model},
        ]
        failures_path_for(output).write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in failures))

        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)

        assert [row[TASK_INDEX_KEY_NAME] for row in input_rows] == [1, 2]


class TestHarnessFailurePaths:
    """Harness failure paths, through the rule Gym applies to the sidecar row."""

    def test_tau2_reruns_a_malformed_tool_call_and_a_dead_policy(self) -> None:
        # tau2 reruns a simulation whose tool-call arguments were malformed, then leaves it out at the cap.
        malformed = {NG_FAILURE_CLASS_KEY: "tau2_malformed_tool_call", "error": "Tau2MalformedToolCallError: bad"}
        dead_policy = {
            NG_FAILURE_CLASS_KEY: "tau2_agent_error",
            "error": "APIConnectionError: litellm.APIConnectionError: Connection error.",
        }
        assert not is_terminal_failure(malformed)
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
            # Terminal-Bench scores a trial that leaves no reward file as 0, and retries a verifier start timeout.
            ("RewardFileNotFoundError", True),
            ("VerifierStartTimeoutError", False),
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


class TestMaskedRowsInTheCollector:
    async def test_a_masked_result_goes_to_the_sidecar_and_is_dispatched_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("nemo_gym.rollout_collection.get_global_config_dict", lambda *args, **kwargs: {})
        output = tmp_path / "out.jsonl"
        rows = [{"responses_create_params": {"input": []}, "agent_ref": {"name": "agent"}, "x": i} for i in range(3)]
        (tmp_path / "in.jsonl").write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                futures = []
                for row in examples:
                    future = asyncio.get_event_loop().create_future()
                    result = {
                        0: {"reward": 1.0},
                        # A provider failure: the mask names infra.
                        1: {
                            "reward": 0.0,
                            "mask_sample": True,
                            "failure_kind": "agent_request_failed",
                            "failure_reason": "APIConnectionError: Connection error.",
                        },
                        # swe_agents' resolved rollout at max_iteration: the model's budget.
                        2: {"reward": 1.0, "mask_sample": True},
                    }[row["x"]]
                    future.set_result(_CompletedRollout(row=row, result=result, rollout_latency_ms=None))
                    futures.append(future)
                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error=True):
                return None

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "in.jsonl"),
            output_jsonl_fpath=str(output),
            disable_health_check=True,
        )
        await Helper().run_from_config(config)

        scored = [orjson.loads(line) for line in output.read_bytes().splitlines()]
        failed = [orjson.loads(line) for line in failures_path_for(output).read_bytes().splitlines()]
        # The budget mask keeps the benchmark's own treatment: a scored row, masked, never retried.
        assert [(row["reward"], row.get("mask_sample", False)) for row in scored] == [(1.0, False), (1.0, True)]
        assert [row[NG_FAILURE_CLASS_KEY] for row in failed] == [MASKED_SAMPLE_FAILURE_CLASS]
        assert _failure_rows_counted_as_zero([failures_path_for(output)], [], set()) == []
        # After the cap, an infra mask counts as 0, even when the verifier left a reward on it.
        assert _counted_failure_row({**failed[0], "reward": 1.0})["reward"] == 0.0

        config.resume_from_cache = True
        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)
        assert [row["x"] for row in input_rows] == [1]


class TestBudgetMasks:
    """A result masked for the model's own budget is a failed task: eval scores count it as 0."""

    async def test_a_budget_mask_counts_as_zero_and_stays_masked_in_the_rollouts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("nemo_gym.rollout_collection.get_global_config_dict", lambda *args, **kwargs: {})
        output = tmp_path / "out.jsonl"
        rows = [{"responses_create_params": {"input": []}, "agent_ref": {"name": "agent"}, "x": i} for i in range(3)]
        (tmp_path / "in.jsonl").write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
        aggregated: list = []

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                futures = []
                for row in examples:
                    future = asyncio.get_event_loop().create_future()
                    result = {
                        0: {"reward": 1.0},
                        # A resolved rollout at max_iteration: the reward is accidental, the task unfinished.
                        1: {"reward": 1.0, "mask_sample": True, "mask_reason": "budget"},
                        # An unlabelled mask keeps its treatment: left out of the score.
                        2: {"reward": 1.0, "mask_sample": True},
                    }[row["x"]]
                    future.set_result(_CompletedRollout(row=row, result=result, rollout_latency_ms=None))
                    futures.append(future)
                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error=True):
                aggregated.extend(results)
                return None

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "in.jsonl"), output_jsonl_fpath=str(output), disable_health_check=True
        )
        await Helper().run_from_config(config)

        stored = [orjson.loads(line) for line in output.read_bytes().splitlines()]
        # The rollouts file, which training reads, keeps both masks as written.
        assert [(row["reward"], row.get("mask_sample", False)) for row in stored] == [
            (1.0, False),
            (1.0, True),
            (1.0, True),
        ]
        assert not failures_path_for(output).exists() or failures_path_for(output).read_bytes() == b""
        # The metric input scores the budget mask as an unmasked 0; the unlabelled mask stays masked.
        assert [(row["reward"], row["mask_sample"]) for row in aggregated if "mask_sample" in row] == [
            (0.0, False),
            (1.0, True),
        ]

    def test_a_budget_mask_is_not_retried_whatever_its_text(self) -> None:
        assert not _masked_result_names_infra(
            {"mask_sample": True, "mask_reason": "budget", "error": "Connection refused"}
        )


class TestTau2LeavesOutWhatItReruns:
    """tau2 reruns a malformed or empty simulation and leaves it out once its retries are spent."""

    @pytest.mark.parametrize("failure_class", ["tau2_malformed_tool_call", "tau2_empty_output"])
    def test_left_out_at_the_cap_and_counted_before_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_class: str
    ) -> None:
        monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "2")
        failures = tmp_path / "out_failures.jsonl"
        row = {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: failure_class}
        failures.write_bytes(orjson.dumps(row) + b"\n")
        assert len(_failure_rows_counted_as_zero([failures], None, set())) == 1
        failures.write_bytes(failures.read_bytes() + orjson.dumps(row) + b"\n")
        assert _failure_rows_counted_as_zero([failures], None, set()) == []

    @pytest.mark.parametrize("failure_class", ["tau2_malformed_tool_call", "tau2_empty_output"])
    def test_a_rerun_episode_is_the_policys_failure(self, failure_class: str) -> None:
        # Rerun as tau2 does, but the policy produced it: the cause is model, never infra.
        assert failure_cause_of_row({NG_FAILURE_CLASS_KEY: failure_class}) == MODEL

    def test_a_context_overflow_is_never_scored(self, tmp_path: Path) -> None:
        failures = tmp_path / "out_failures.jsonl"
        row = {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "tau2_context_overflow"}
        failures.write_bytes(orjson.dumps(row) + b"\n")
        assert is_terminal_failure(row)
        assert _failure_rows_counted_as_zero([failures], None, set()) == []


class TestScoringDefaults:
    """Every failed rollout counts as 0 by default; opting out leaves only final failures counted."""

    def _sidecar(self, tmp_path: Path) -> Path:
        failures = tmp_path / "out_failures.jsonl"
        rows = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "timeout_exceeded"},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "judge_failed"},
        ]
        failures.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
        return failures

    def test_the_default_counts_every_failure(self, tmp_path: Path) -> None:
        config = RolloutCollectionConfig(input_jsonl_fpath="in.jsonl", output_jsonl_fpath="out.jsonl")
        assert config.count_failure_classes_as_zero is None
        assert config.route_failures_to_sidecar is True

        counted = _failure_rows_counted_as_zero([self._sidecar(tmp_path)], None, set())
        assert sorted(row[TASK_INDEX_KEY_NAME] for row in counted) == [0, 1]

    def test_the_default_leaves_a_data_gap_out_of_the_score(self, tmp_path: Path) -> None:
        failures = self._sidecar(tmp_path)
        gap = {TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "reference_missing"}
        failures.write_bytes(failures.read_bytes() + orjson.dumps(gap) + b"\n")
        config = RolloutCollectionConfig(input_jsonl_fpath="in.jsonl", output_jsonl_fpath="out.jsonl")

        counted = _failure_rows_counted_as_zero([failures], config.count_failure_classes_as_zero, set())
        assert sorted(row[TASK_INDEX_KEY_NAME] for row in counted) == [0, 1]

    def test_opting_out_still_counts_a_final_failure(self, tmp_path: Path) -> None:
        counted = _failure_rows_counted_as_zero([self._sidecar(tmp_path)], [], set())
        assert [(row[TASK_INDEX_KEY_NAME], row["reward"]) for row in counted] == [(0, 0.0)]

    def test_a_class_list_counts_those_classes_and_every_final_failure(self, tmp_path: Path) -> None:
        counted = _failure_rows_counted_as_zero([self._sidecar(tmp_path)], ["judge_failed"], set())
        assert sorted(row[TASK_INDEX_KEY_NAME] for row in counted) == [0, 1]
