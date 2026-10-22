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
"""The retry rule on real failure rows from past runs.

Each fixture is a real sidecar row, trimmed to the fields the rule reads and scrubbed of paths, hosts and ids.
"""

import asyncio
from pathlib import Path

import orjson
import pytest

from nemo_gym.failure_kinds import (
    INFRA,
    MODEL,
    failure_cause_of,
    failure_cause_of_text,
)
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.rollout_collection import (
    MASKED_SAMPLE_FAILURE_CLASS,
    NG_FAILURE_CLASS_KEY,
    RolloutCollectionConfig,
    _counted_failure_row,
    _failure_rows_counted_as_zero,
    _masked_result_names_infra,
    _settled_failure_keys,
    failure_cause_of_row,
    is_terminal_failure,
    is_unscored_failure,
)
from nemo_gym.sandbox.providers.base import SandboxOperationTimeoutError


class TestMaskedRows:
    """A mask naming infra is retried; a budget mask keeps its treatment."""

    def test_a_masked_row_with_an_evidence_kind_and_infra_text_is_retried(self) -> None:
        row = {
            NG_FAILURE_CLASS_KEY: MASKED_SAMPLE_FAILURE_CLASS,
            "failure_kind": "verifier_error",
            "failure_reason": "grader timed out: Connection refused",
        }
        assert not is_terminal_failure(row)

    def test_a_budget_mask_does_not_name_infra(self) -> None:
        # swe_agents and anyswe mask a resolved rollout at max_iteration or the context window.
        assert not _masked_result_names_infra({"mask_sample": True, "reward": 1.0})

    def test_a_counted_masked_row_scores_zero_not_its_reward(self) -> None:
        row = {NG_FAILURE_CLASS_KEY: MASKED_SAMPLE_FAILURE_CLASS, "reward": 1.0, "mask_sample": True}
        counted = _counted_failure_row(row)
        assert (counted["reward"], counted["mask_sample"]) == (0.0, False)

    def test_an_osworld_infra_mask_is_retried(self) -> None:
        # An OSWorld provider failure, named with the infra cause OSWorld sets on platform evidence.
        result = {
            "mask_sample": True,
            "failure_kind": "provider_unavailable",
            "failure_reason": "environment_start_failed",
            "_ng_failure_cause": "infra",
        }
        assert _masked_result_names_infra(result)

    def test_an_osworld_provider_mask_without_a_cause_is_not_retried(self) -> None:
        # An evaluator that broke on the VM the model left behind: the producer's provider claim is no evidence.
        result = {"mask_sample": True, "failure_kind": "provider_unavailable", "failure_reason": "evaluator_error"}
        assert not _masked_result_names_infra(result)


class TestCheckCompletion:
    """Final and capped rows are complete, so require_complete does not fail the run."""

    def test_final_and_capped_rows_count_as_complete(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "2")
        failures = tmp_path / "out_failures.jsonl"
        rows = [
            # Final: the agent's own time budget.
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "timeout_exceeded"},
            # Capped: a judge failure retried up to the cap.
            {TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "judge_failed"},
            {TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "judge_failed"},
            # Retryable, attempts left: not settled.
            {TASK_INDEX_KEY_NAME: 3, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "judge_failed"},
        ]
        failures.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
        settled = _settled_failure_keys(failures)
        assert settled == {(1, 0), (2, 0)}

        config = RolloutCollectionConfig(
            input_jsonl_fpath="in.jsonl", output_jsonl_fpath=str(tmp_path / "out.jsonl"), require_complete=True
        )
        scored = [{TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0}]
        config.check_completion(expected=3, results=scored, settled_keys=settled)
        with pytest.raises(RuntimeError, match="EVAL FAILED: 3/4"):
            config.check_completion(expected=4, results=scored, settled_keys=settled)


class TestTerminalBench4Rows:
    """Terminal-Bench keeps its cause only in failure_reason and infrastructure_error."""

    @pytest.mark.parametrize(
        "name",
        [
            "SandboxCreateError",  # sandbox creation, before the agent's first command
            "VerifierStartTimeoutError",
            "ClientPayloadError",
            "APIConnectionError",  # model server down mid-run
            "OpenSandboxCreateError",  # sandbox provisioning
        ],
    )
    def test_a_tb4_infra_row_is_retried(self, name: str) -> None:
        row = {
            NG_FAILURE_CLASS_KEY: "infrastructure_error",
            "failure_reason": name,
            "infrastructure_error": name,
            "mask_sample": False,
            "reward": 0.0,
        }
        assert not is_terminal_failure(row)

    @pytest.mark.parametrize(
        "name",
        [
            "AgentInfrastructureError",
            "AgentInfrastructureError: mini-SWE produced no trajectory",
            "SandboxApiException",  # the sandbox failed while the agent's commands ran
            "SandboxTimeoutException",
        ],
    )
    def test_a_tb4_sandbox_lost_during_the_agents_run_is_a_final_zero(self, name: str) -> None:
        row = {NG_FAILURE_CLASS_KEY: "infrastructure_error", "failure_reason": name, "infrastructure_error": name}
        assert is_terminal_failure(row)

    def test_a_tb4_missing_reward_file_is_a_final_zero(self) -> None:
        # Terminal-Bench scores a trial that leaves no reward file as 0; it is not retried.
        row = {
            NG_FAILURE_CLASS_KEY: "infrastructure_error",
            "failure_reason": "RewardFileNotFoundError",
            "infrastructure_error": "RewardFileNotFoundError",
        }
        assert is_terminal_failure(row)

    def test_a_tb4_verifier_timeout_stays_final(self) -> None:
        # A verifier timeout is not on the infra lists, so it stays final.
        row = {
            NG_FAILURE_CLASS_KEY: "infrastructure_error",
            "failure_reason": "VerifierTimeoutError",
            "infrastructure_error": "VerifierTimeoutError",
        }
        assert is_terminal_failure(row)


class TestTimeouts:
    """The agent's own time budget is the model's; an infra call's timeout is infra."""

    async def test_the_agents_own_asyncio_timeout_is_the_models(self) -> None:
        with pytest.raises(TimeoutError) as caught:
            async with asyncio.timeout(0.01):
                await asyncio.sleep(1)
        assert isinstance(caught.value.__context__, asyncio.CancelledError)
        assert failure_cause_of(caught.value) == MODEL

    def test_a_timeout_raised_from_none_is_no_evidence(self) -> None:
        try:
            try:
                raise ConnectionRefusedError("hidden on purpose")
            except ConnectionRefusedError:
                raise TimeoutError("agent budget") from None
        except TimeoutError as error:
            assert failure_cause_of(error) == MODEL

    def test_a_named_sandbox_timeout_is_the_models(self) -> None:
        # The model's own command can outlive a sandbox call's deadline; only platform evidence makes it infra.
        assert failure_cause_of(SandboxOperationTimeoutError("write_file")) == MODEL

    def test_a_task_that_ran_out_of_its_own_time_budget_is_a_final_zero(self) -> None:
        row = {NG_FAILURE_CLASS_KEY: "timeout_exceeded", "_ng_failure_message": "per-task timeout exceeded"}
        assert failure_cause_of_row(row) == MODEL
        assert is_terminal_failure(row)

    def test_the_opensandbox_write_file_timeout_as_relayed_text_is_the_models(self) -> None:
        # The body a legacy_agent relay carried, before the error had a name.
        body = (
            '"SandboxOperationTimeoutError(\\"Timed out during OpenSandbox write_file(<path>) after 300s; '
            "sandbox_id='<uuid>'\\\")\""
        )
        assert failure_cause_of_text(body) == MODEL

    async def test_an_aiohttp_total_timeout_is_infra(self) -> None:
        aiohttp = pytest.importorskip("aiohttp")
        from aiohttp import web

        async def slow(request):
            await asyncio.sleep(5)
            return web.Response(text="late")

        app = web.Application()
        app.router.add_get("/", slow)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                with pytest.raises(TimeoutError) as caught:
                    async with session.get(f"http://127.0.0.1:{port}/", timeout=aiohttp.ClientTimeout(total=0.2)):
                        pass
        finally:
            await runner.cleanup()
        assert failure_cause_of(caught.value) == INFRA

    @pytest.mark.parametrize(
        ("text", "cause"),
        [
            # Ray could not carry the task's exception back.
            ("UnserializableException: Failed to deserialize exception.", INFRA),
            ("ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.", INFRA),
            ("ray.exceptions.WorkerCrashedError: The worker died unexpectedly", INFRA),
            # A RayTaskError takes the cause of what it wraps; a config ImportError stays final.
            ("RayTaskError(ImportError): cannot import name 'DEFAULT_TOOLS' from 'stirrup.tools'", MODEL),
            ("RayTaskError(ClientConnectorError): Cannot connect to host", INFRA),
        ],
    )
    def test_ray_failures(self, text: str, cause: str) -> None:
        assert failure_cause_of_text(text) == cause


class TestEnvironmentServerPath:
    """The server records the cause from the live error; terminal=False is a retry."""

    def _episode(self, failure: dict, header_cause: str | None = None) -> dict:
        from nemo_gym.rollout_collection import NG_FAILURE_CAUSE_KEY, _episode_record

        reply = {"task_id": {"taskset": "t", "task_id": "a"}, "failure": failure}
        if header_cause is not None:
            reply[NG_FAILURE_CAUSE_KEY] = header_cause  # what the collector adds from the response header
        return _episode_record(reply)

    def test_an_empty_timeout_reason_the_server_marks_not_terminal_is_retried(self) -> None:
        # single_agent_turn: str(TimeoutError()) is empty.
        row = self._episode({"failure_reason": "", "terminal": False, "stage": "agent"})
        assert not is_terminal_failure(row)

    def test_a_client_payload_error_with_a_400_in_its_text_is_retried(self) -> None:
        reason = "ClientPayloadError: 400, message='Can not decode content-encoding: br'"
        row = self._episode({"failure_reason": reason, "terminal": False, "stage": "agent"})
        assert not is_terminal_failure(row)

    def test_the_cause_the_server_recorded_wins_over_the_text(self) -> None:
        # An agent 500 that the agent's middleware named "model": a malformed tool call.
        reason = "500, message='Internal Server Error'"
        row = self._episode({"failure_reason": reason, "terminal": False, "stage": "agent"}, header_cause=MODEL)
        assert is_terminal_failure(row)

    @pytest.mark.parametrize("cause", [MODEL, INFRA, None])
    def test_the_cause_travels_in_the_header_and_the_failure_record_is_unchanged(self, cause) -> None:
        """The versioned failure record is unchanged: the cause of a handled failure travels in the header."""
        from unittest.mock import MagicMock

        from fastapi.testclient import TestClient

        from nemo_gym.base_environment_server import (
            BaseEnvironmentServer,
            BaseEnvironmentServerConfig,
            HandledEpisodeError,
        )
        from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure
        from nemo_gym.failure_kinds import FAILURE_CAUSE_HEADER
        from nemo_gym.rollout_collection import NG_FAILURE_CAUSE_KEY, _episode_record
        from nemo_gym.server_utils import ServerClient

        class Server(BaseEnvironmentServer[BaseEpisodeRequest, BaseEpisodeResponse]):
            request_model = BaseEpisodeRequest
            response_model = BaseEpisodeResponse

            async def run(self, request, cleanup):
                failure = EpisodeFailure(failure_reason="500, message='Internal Server Error'", terminal=False)
                raise HandledEpisodeError(failure, cause=cause)

            async def aggregate_metrics(self, body):
                raise NotImplementedError

        config = BaseEnvironmentServerConfig(
            name="env",
            host="127.0.0.1",
            port=1,
            entrypoint="app.py",
            default_episode_timeout_seconds=1,
            cleanup_timeout_seconds=0.01,
        )
        app = Server(config=config, server_client=MagicMock(spec=ServerClient)).setup_webserver()
        task = {"task_id": {"taskset": "t:default", "task_id": "a"}, "task_input": {}}
        reply = TestClient(app).post("/run", json={"episode_id": {"rollout_id": "r", "attempt": 1}, "task": task})

        body = reply.json()
        # The failure record carries only its version-1 fields.
        assert set(body["failure"]) == {"failure_reason", "terminal"}
        assert reply.headers.get(FAILURE_CAUSE_HEADER) == cause
        if cause is not None:
            body[NG_FAILURE_CAUSE_KEY] = cause  # what the collector adds from the header
        # A 500 with no cause in the header is infra by its text; a cause the server read wins over the text.
        assert _episode_record(body)[NG_FAILURE_CAUSE_KEY] == (cause or INFRA)

    def test_episode_failure_cause_reads_the_live_error(self) -> None:
        from types import SimpleNamespace

        from aiohttp import ClientPayloadError, ClientResponseError
        from multidict import CIMultiDict

        from nemo_gym.failure_kinds import FAILURE_CAUSE_HEADER, episode_failure_cause

        assert episode_failure_cause(TimeoutError(), terminal=False) == INFRA
        assert episode_failure_cause(ClientPayloadError("400, message='x'"), terminal=True) == INFRA
        named_model = ClientResponseError(
            request_info=SimpleNamespace(real_url="http://agent/run"),
            history=(),
            status=500,
            headers=CIMultiDict({FAILURE_CAUSE_HEADER: MODEL}),
        )
        assert episode_failure_cause(named_model, terminal=False) == MODEL
        assert episode_failure_cause(ValueError("bad seed row"), terminal=True) == MODEL


# A real Stirrup harness row from a past run: Ray could not carry the task's exception back, and the exception
# it wraps is the policy model server's empty 500. Scrubbed: paths, pid and ip; the traceback is cut.
_STIRRUP_RAY_ERROR_MESSAGE = (
    "UnserializableException: Failed to deserialize exception. Refer to <url> for more information.\n"
    "Original exception:\n"
    "ray.exceptions.RayTaskError: ray::run_stirrup_agent_remote() (pid=<pid>, ip=<ip>)\n"
    '  File "<gym>/responses_api_agents/stirrup_agent/app.py", line 469, in run_stirrup_agent_remote\n'
    "    return asyncio.run(_run_stirrup_agent(**params))\n"
    "    ...<cut>...\n"
    '  File "<site-packages>/openai/_base_client.py", line 1716, in request\n'
    "    raise self._make_status_error_from_response(err.response) from None\n"
    "openai.InternalServerError: Error code: 500 - Hit an exception in VLLMModel___policy_model calling an inner "
    "server: b''"
)


class TestDataGaps:
    """GDPval's own data and judge-eligibility gaps are not the model's: never retried, never scored."""

    @pytest.mark.parametrize("failure_class", ["reference_missing", "eval_missing", "transport_ineligible"])
    def test_a_data_gap_is_final_unscored_and_complete(self, failure_class: str, tmp_path: Path) -> None:
        # The row as GDPval's verify writes it: a masked, unstamped failure with a namespaced kind.
        row = {
            TASK_INDEX_KEY_NAME: 1,
            ROLLOUT_INDEX_KEY_NAME: 0,
            NG_FAILURE_CLASS_KEY: failure_class,
            "failure_kind": f"gdpval:{failure_class}",
            "mask_sample": True,
            "reward": 0.0,
        }
        assert is_terminal_failure(row)
        assert is_unscored_failure(row)

        failures = tmp_path / "out_failures.jsonl"
        failures.write_bytes(orjson.dumps(row) + b"\n")
        for selected in (None, [], [failure_class]):
            assert _failure_rows_counted_as_zero([failures], selected, set()) == []
        assert _settled_failure_keys(failures) == {(1, 0)}

    @pytest.mark.parametrize(("failure_class", "fails"), [("reference_missing", False), ("judge_failed", True)])
    async def test_a_run_with_nothing_left_to_retry_finishes(
        self, failure_class: str, fails: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every row a data gap: no score, but a resume has nothing to do, so the run ends without an error."""
        from nemo_gym.rollout_collection import RolloutCollectionHelper, _CompletedRollout

        monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "3")
        monkeypatch.setattr("nemo_gym.rollout_collection.get_global_config_dict", lambda *args, **kwargs: {})
        row = {"responses_create_params": {"input": []}, "agent_ref": {"name": "agent"}}
        (tmp_path / "in.jsonl").write_bytes(orjson.dumps(row) + b"\n")

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                futures = []
                for example in examples:
                    future = asyncio.get_event_loop().create_future()
                    result = {NG_FAILURE_CLASS_KEY: failure_class, "failure_kind": f"gdpval:{failure_class}"}
                    future.set_result(_CompletedRollout(row=example, result=result, rollout_latency_ms=None))
                    futures.append(future)
                return futures

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "in.jsonl"),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            disable_health_check=True,
            require_complete=True,
            count_failure_classes_as_zero=[],
        )
        if fails:
            # A judge failure is retried: a resume still has work to do, so the run must not pass as done.
            with pytest.raises(RuntimeError, match="produced a result"):
                await Helper().run_from_config(config)
        else:
            await Helper().run_from_config(config)
            assert _settled_failure_keys(tmp_path / "out_failures.jsonl") == {(0, 0)}

    def test_a_scored_class_is_not_a_data_gap(self) -> None:
        assert not is_unscored_failure({NG_FAILURE_CLASS_KEY: "tau2_malformed_tool_call"})
        assert not is_unscored_failure({NG_FAILURE_CLASS_KEY: "judge_failed"})


class TestStirrupErrorMessage:
    """Stirrup rows from older Gym versions keep their error text only in ``error_message``."""

    def test_a_stirrup_harness_crash_is_retried(self) -> None:
        row = {
            NG_FAILURE_CLASS_KEY: "legitimate",
            "error_class": "legitimate",
            "error_message": _STIRRUP_RAY_ERROR_MESSAGE,
            "reward": 0.0,
        }
        assert failure_cause_of_row(row) == INFRA
        assert not is_terminal_failure(row)

    def test_a_stirrup_row_whose_wrapped_error_is_the_tasks_own_stays_final(self) -> None:
        # A RayTaskError takes the cause of what it wraps: a config ImportError is no infra evidence.
        row = {
            NG_FAILURE_CLASS_KEY: "legitimate",
            "error_class": "legitimate",
            "error_message": "RayTaskError(ImportError): cannot import name 'DEFAULT_TOOLS' from 'stirrup.tools'",
        }
        assert is_terminal_failure(row)
