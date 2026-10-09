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
import json
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch, fixture, mark

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    AgentPhaseObservation,
    ContextCompactionObservation,
    SandboxObservation,
    ToolCallObservation,
)
from nemo_gym.sandbox import SandboxHandle
from nemo_gym.sandbox.utils import CPU_CAP_ENV_VARS
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.opencode_sandboxed_agent import app as app_module
from responses_api_agents.opencode_sandboxed_agent.app import (
    SQLITE_SNAPSHOT_SCRIPT,
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
    _probe_sandbox_clock,
    parse_opencode_observations,
)


class TestOpenCodeSandboxedAgent:
    def test_import_does_not_load_standalone_opencode_agent(self) -> None:
        code = (
            "import sys; import responses_api_agents.opencode_sandboxed_agent.app; "
            "assert not any(name == 'responses_api_agents.opencode_agent' "
            "or name.startswith('responses_api_agents.opencode_agent.') for name in sys.modules)"
        )
        subprocess.run([sys.executable, "-c", code], check=True, timeout=30)

    def _create_config(self) -> OpenCodeSandboxedAgentConfig:
        return OpenCodeSandboxedAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(type="resources_servers", name=""),
            model_server=ModelServerRef(type="responses_api_models", name=""),
            opencode_version="",
            sandbox_provider="",
            sandbox_config=dict(),
            sandbox_timeout=0,
            opencode_max_context_window=0,
            token_id_capture=True,
        )

    async def test_start_sandbox_derives_cpu_cap_env_from_cpu_limit(self, monkeypatch: MonkeyPatch) -> None:
        sandbox = MagicMock()
        sandbox.start = AsyncMock()
        sandbox.pty = AsyncMock()
        monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {})
        monkeypatch.setattr(app_module, "create_provider", lambda *_: MagicMock())
        monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: MagicMock())
        monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
        monkeypatch.setattr(app_module, "AsyncSandbox", MagicMock(return_value=sandbox))

        async def created_spec(sandbox_config: Dict[str, Any]) -> Any:
            config = self._create_config()
            config.sandbox_config = sandbox_config
            server = OpenCodeSandboxedAgent(config=config, server_client=MagicMock(spec=ServerClient))
            await server._start_sandbox()
            return sandbox.start.await_args.args[0]

        # Floored to whole cores; every cap gets the same value.
        spec = await created_spec({"resources": {"cpu": 2.7, "memory_mib": 8192}})
        assert spec.env == {name: "2" for name in CPU_CAP_ENV_VARS}
        spec = await created_spec({"resources": {"cpu": 0.5}})
        assert spec.env["OMP_NUM_THREADS"] == "1"

        # Opt-out and no-cpu-limit paths inject nothing.
        assert (await created_spec({"resources": {"cpu": 2}, "derive_cpu_env": False})).env == {}
        assert (await created_spec({"resources": {"memory_mib": 8192}})).env == {}

        # Explicit sandbox_config.env is passed through and wins over the derived caps.
        spec = await created_spec(
            {"resources": {"cpu": 2}, "env": {"EXECD_API_GRACE_SHUTDOWN": "50ms", "OMP_NUM_THREADS": "4"}}
        )
        assert spec.env["EXECD_API_GRACE_SHUTDOWN"] == "50ms"
        assert spec.env["OMP_NUM_THREADS"] == "4"
        assert all(spec.env[name] == "2" for name in CPU_CAP_ENV_VARS if name != "OMP_NUM_THREADS")

    @fixture
    def opencode_export_test_data(self) -> Dict[str, Any]:
        test_data_path = Path(__file__).parent / "opencode_export_test_data.json"
        return json.loads(test_data_path.read_text())

    def test_opencode_export_to_output_items(
        self, opencode_export_test_data: Dict[str, Any], monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", MagicMock(return_value=MagicMock(hex="")))

        actual_output_items = OpenCodeSandboxedAgent._opencode_export_to_output_items(None, opencode_export_test_data)
        expected_output_items = [
            NeMoGymEasyInputMessage(content=[{"text": "hello", "type": "input_text"}], role="user", type="message"),
            NeMoGymResponseOutputMessage(
                id="msg_",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[], text="Hello! How can I help you today?", type="output_text", logprobs=None
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
            NeMoGymResponseReasoningItem(
                id="rs_",
                summary=[
                    NeMoGymSummary(
                        text="Let me look at the main implementation of `separability_matrix` in `separable.py` and the `_calculate_separability_matrix` method in `core.py`.",
                        type="summary_text",
                    )
                ],
                type="reasoning",
                encrypted_content=None,
            ),
            NeMoGymResponseFunctionToolCall(
                arguments='{"filePath": "/testbed/astropy/modeling/separable.py"}',
                call_id="chatcmpl-tool-944dd9d62f6ccf66",
                name="read",
                type="function_call",
                id=None,
                status=None,
            ),
            NeMoGymFunctionCallOutput(
                call_id="chatcmpl-tool-944dd9d62f6ccf66",
                output="<path>/testbed/astropy/modeling/separable.py</path>\n<type>file</type>\n<content>\n...(End of file - total 317 lines)\n</content>",
                type="function_call_output",
                id=None,
                status=None,
            ),
        ]

        assert expected_output_items == actual_output_items

    def test_opencode_export_to_usages(self, opencode_export_test_data: Dict[str, Any]) -> None:
        actual_usages = OpenCodeSandboxedAgent._opencode_export_to_usages(None, opencode_export_test_data)
        expected_usages = [
            NeMoGymResponseUsage(
                input_tokens=55,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=7808),
                output_tokens=10,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=7873,
            ),
            NeMoGymResponseUsage(
                input_tokens=8692,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=71,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=8763,
            ),
        ]

        assert expected_usages == actual_usages

    async def test_responses_sanity(self, opencode_export_test_data: Dict[str, Any], monkeypatch: MonkeyPatch) -> None:
        config = self._create_config()
        server = OpenCodeSandboxedAgent(config=config, server_client=MagicMock(spec=ServerClient))

        sandbox_mock = MagicMock()
        sandbox_mock.exec = AsyncMock(
            side_effect=[
                SimpleNamespace(
                    stdout="Shell: /bin/bash\nOpenCode run finished", stderr="", return_code=0, error_type=None
                ),
                SimpleNamespace(stdout='[{"id": "session-id"}]', stderr="", return_code=0, error_type=None),
                SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None),
            ]
        )
        sandbox_mock.download = AsyncMock()
        monkeypatch.setattr(server, "_sandbox_id_to_sandbox", {"": sandbox_mock})
        monkeypatch.setattr(server, "_create_opencode_config", AsyncMock(return_value=dict()))

        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.Path.exists",
            lambda self: True,
        )
        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.Path.read_text",
            lambda self: json.dumps(opencode_export_test_data),
        )
        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.uuid4", MagicMock(return_value=MagicMock(hex=""))
        )
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", MagicMock(return_value=MagicMock(hex="")))
        monkeypatch.setattr("responses_api_agents.opencode_sandboxed_agent.app.time", MagicMock(return_value=0.0))

        actual_response = await server.responses(
            request=MagicMock(
                session={SESSION_ID_KEY: "my session"},
                cookies={"sandbox_id": ""},
                path_params={"rollout_id": "direct-call"},
            ),
            body=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "hello"}],
            ),
        )
        expected_response = NeMoGymResponse(
            id="resp_",
            created_at=0.0,
            error=None,
            incomplete_details=None,
            instructions=None,
            metadata=None,
            model="",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_",
                    content=[
                        NeMoGymResponseOutputText(
                            annotations=[], text="Hello! How can I help you today?", type="output_text", logprobs=None
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
                NeMoGymResponseReasoningItem(
                    id="rs_",
                    summary=[
                        NeMoGymSummary(
                            text="Let me look at the main implementation of `separability_matrix` in `separable.py` and the `_calculate_separability_matrix` method in `core.py`.",
                            type="summary_text",
                        )
                    ],
                    type="reasoning",
                    encrypted_content=None,
                ),
                NeMoGymResponseFunctionToolCall(
                    arguments='{"filePath": "/testbed/astropy/modeling/separable.py"}',
                    call_id="chatcmpl-tool-944dd9d62f6ccf66",
                    name="read",
                    type="function_call",
                    id=None,
                    status=None,
                ),
                NeMoGymFunctionCallOutput(
                    call_id="chatcmpl-tool-944dd9d62f6ccf66",
                    output="<path>/testbed/astropy/modeling/separable.py</path>\n<type>file</type>\n<content>\n...(End of file - total 317 lines)\n</content>",
                    type="function_call_output",
                    id=None,
                    status=None,
                ),
            ],
            parallel_tool_calls=True,
            temperature=None,
            tool_choice="auto",
            tools=[],
            top_p=None,
            background=None,
            conversation=None,
            max_output_tokens=None,
            max_tool_calls=None,
            previous_response_id=None,
            prompt=None,
            prompt_cache_key=None,
            reasoning=None,
            safety_identifier=None,
            service_tier=None,
            status=None,
            text=None,
            top_logprobs=None,
            truncation=None,
            usage=NeMoGymResponseUsage(
                input_tokens=8747,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=7808),
                output_tokens=81,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=16636,
            ),
            user=None,
        )

        assert expected_response == actual_response
        assert not any(key.startswith("_ng_") for key in server._sandbox_id_to_run_result[""])
        assert "XDG_DATA_HOME" not in sandbox_mock.exec.await_args_list[0].kwargs["command"]

    def test_agent_sandbox_observation_classifies_timeout_errors(self) -> None:
        server = OpenCodeSandboxedAgent(
            config=self._create_config(),
            server_client=MagicMock(spec=ServerClient),
        )
        sandbox = MagicMock()
        sandbox._handle = SandboxHandle(sandbox_id="connected-sandbox", provider_name="opensandbox", raw=None)

        observation = server._agent_sandbox_observation(
            sandbox=sandbox,
            return_code=125,
            error_type="TimeoutError",
            finished=False,
        )

        assert observation.outcome == "timeout"
        assert observation.exit_code is None
        assert observation.sandbox_id == "connected-sandbox"
        assert observation.provider == "opensandbox"

        observation = server._agent_sandbox_observation(
            sandbox=sandbox,
            return_code=137,
            error_type="OutOfMemoryError",
            finished=False,
        )
        assert observation.outcome == "sandbox_error"
        assert observation.exit_code is None

    @mark.parametrize(
        ("observability_enabled", "token_capture_enabled", "expected_base_url"),
        [
            (False, False, "http://model-server/v1"),
            (True, False, "http://model-server/ng-rollout/7-2/v1"),
            (False, True, "http://model-server/ng-rollout/7-2/training-token-capture/v1"),
            (True, True, "http://model-server/ng-rollout/7-2/training-token-capture/v1"),
        ],
        ids=("disabled", "observability-only", "token-capture-only", "both"),
    )
    async def test_create_opencode_config_routes_each_capture_state(
        self,
        monkeypatch: MonkeyPatch,
        observability_enabled: bool,
        token_capture_enabled: bool,
        expected_base_url: str,
    ) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {
            "observability_enabled": observability_enabled,
            "token_id_capture": {"enabled": token_capture_enabled, "all_agents": False},
        }
        server = OpenCodeSandboxedAgent(config=self._create_config(), server_client=server_client)
        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.get_server_url",
            lambda _name: "http://model-server",
        )
        request = MagicMock()
        request.json = AsyncMock(
            return_value={
                "responses_create_params": {"input": "solve"},
                "_ng_task_index": 7,
                "_ng_rollout_index": 2,
            }
        )

        config = await server._create_opencode_config(request)

        assert config["provider"]["nemo_gym"]["options"]["baseURL"] == expected_base_url

    @mark.parametrize("latency_metrics", [False, True])
    async def test_run_builds_observations_from_live_wal_snapshot(
        self,
        tmp_path: Path,
        opencode_export_test_data: Dict[str, Any],
        monkeypatch: MonkeyPatch,
        latency_metrics: bool,
    ) -> None:
        class Response:
            ok = True

            def __init__(self, payload: dict[str, Any], cookies: dict[str, str] | None = None):
                self.payload = payload
                self.cookies = cookies or {}

            async def json(self) -> dict[str, Any]:
                return self.payload

            async def read(self) -> bytes:
                return json.dumps(self.payload).encode()

        class RunRequest:
            def __init__(self) -> None:
                self._cookies: dict[str, str] = {}
                self.session = {SESSION_ID_KEY: "session-1"}
                self.state = SimpleNamespace()

            @property
            def cookies(self) -> dict[str, str]:
                return self._cookies

        db_path = tmp_path / "source.db"
        connection = sqlite3.connect(db_path)
        connection.execute("pragma journal_mode=wal")
        connection.execute("create table session (id text, parent_id text, time_created integer)")
        connection.execute("create table message (id text, session_id text, data text, time_created integer)")
        connection.execute(
            "create table part (id text, message_id text, session_id text, data text, time_created integer)"
        )
        connection.commit()
        connection.execute("pragma wal_checkpoint(truncate)")
        connection.execute("insert into session values ('root', null, 0)")
        connection.execute(
            "insert into message values (?, ?, ?, ?)",
            ("m1", "root", json.dumps({"role": "assistant", "time": {"created": 1, "completed": 3}}), 1),
        )
        connection.execute(
            "insert into part values (?, ?, ?, ?, ?)",
            (
                "p1",
                "m1",
                "root",
                json.dumps(
                    {
                        "type": "tool",
                        "tool": "bash",
                        "callID": "call-1",
                        "state": {
                            "status": "completed",
                            "input": {"command": "true"},
                            "output": "",
                            "time": {"start": 1_000, "end": 2_000},
                        },
                    }
                ),
                1,
            ),
        )
        connection.commit()
        assert db_path.with_name(f"{db_path.name}-wal").stat().st_size > 0
        main_only_path = tmp_path / "main-only.db"
        main_only_path.write_bytes(db_path.read_bytes())
        with sqlite3.connect(main_only_path) as main_only:
            assert main_only.execute("select count(*) from session").fetchone() == (0,)

        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {
            "observability_enabled": True,
            "token_id_capture": {"enabled": False, "all_agents": False},
        }
        config = self._create_config()
        config.latency_metrics = latency_metrics
        server = OpenCodeSandboxedAgent(config=config, server_client=server_client)
        server._create_opencode_config = AsyncMock(return_value={})

        run_stdout = "Shell: /bin/bash\nOpenCode run finished"
        if latency_metrics:
            run_stdout = "Shell: /bin/bash\nNG_T_LAUNCH=1700000000100000000\nNG_T_EXIT=1700000000900000000\nOpenCode run finished"
        sandbox = MagicMock()
        sandbox._handle = SandboxHandle(sandbox_id="connected-sandbox", provider_name="opensandbox", raw=None)
        sandbox.exec = AsyncMock(
            side_effect=[
                # clock probe at sandbox start (date +%s.%N)
                SimpleNamespace(stdout="1700000000.250000000\n", stderr="", return_code=0, error_type=None),
                SimpleNamespace(stdout=run_stdout, stderr="", return_code=0, error_type=None),
                SimpleNamespace(stdout='[{"id": "session-id"}]', stderr="", return_code=0, error_type=None),
                SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None),
                SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None),
            ]
        )
        snapshot_path = tmp_path / "snapshot.db"

        def local_quote(value: str) -> str:
            if value.endswith("/opencode/opencode.db"):
                value = str(db_path)
            elif value.endswith("/opencode/nemo-gym-observations.db"):
                value = str(snapshot_path)
            return shlex.quote(value)

        monkeypatch.setattr("responses_api_agents.opencode_sandboxed_agent.app.quote", local_quote)

        async def download(remote_path: str, local_path: Path) -> None:
            if remote_path == "/tmp/opencode_export.json":
                local_path.write_text(json.dumps(opencode_export_test_data))
            else:
                assert remote_path.endswith("/opencode/nemo-gym-observations.db")
                subprocess.run(shlex.split(sandbox.exec.await_args_list[-1].kwargs["command"]), check=True)
                local_path.write_bytes(snapshot_path.read_bytes())

        sandbox.download = AsyncMock(side_effect=download)
        sandbox.stop = AsyncMock(side_effect=RuntimeError("resource server already stopped the sandbox"))
        server._start_sandbox = AsyncMock(return_value=sandbox)
        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.__file__",
            str(tmp_path / "app.py"),
        )

        verifier_sandbox = SandboxObservation(
            role="verifier",
            provider="opensandbox",
            sandbox_id="verify-sandbox",
            outcome="completed",
            wall_time_s=2.0,
        )

        async def post(server_name, url_path, json=None, cookies=None):
            if url_path == "/seed_session":
                return Response({"sandbox_handle": "seed-sandbox"})
            assert url_path == "/verify"
            return Response(
                json
                | {
                    "reward": 1.0,
                    "verifier_sandbox_observation": verifier_sandbox.model_dump(mode="json"),
                }
            )

        server_client.post = AsyncMock(side_effect=post)
        request = RunRequest()
        body = OpenCodeSandboxedAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": [{"role": "user", "content": "solve"}]},
                "_ng_task_index": 7,
                "_ng_rollout_index": 2,
            }
        )

        try:
            result = await server.run(request, body)
        finally:
            connection.close()

        assert result.ng_agent_observations is not None
        [invocation] = [
            record for record in result.ng_agent_observations.records if isinstance(record, AgentInvocation)
        ]
        assert invocation.invocation_id == "root"
        assert invocation.status == "completed"
        assert (invocation.started_at, invocation.completed_at) == (0.001, 0.003)
        [tool] = [record for record in result.ng_agent_observations.records if isinstance(record, ToolCallObservation)]
        assert tool.tool_call_id == "call-1"
        assert tool.sandbox_id == "connected-sandbox"
        # The tool starts when its part was persisted (the request), not at the state start OpenCode rewrites.
        assert (tool.started_at, tool.completed_at, tool.duration_ms) == (0.001, 2.0, 1_999.0)
        assert tool.tool_name == "bash"
        assert tool.operation == "true"
        sandbox_records = [
            record for record in result.ng_agent_observations.records if isinstance(record, SandboxObservation)
        ]
        assert [(record.role, record.sandbox_id) for record in sandbox_records] == [
            ("agent", "connected-sandbox"),
            ("verifier", "verify-sandbox"),
        ]
        assert sandbox_records[0].provider == "opensandbox"
        assert sandbox_records[0].outcome == "completed"
        assert sandbox_records[0].wall_time_s is None
        assert sandbox_records[0].clock_offset_s is not None
        assert sandbox_records[0].clock_offset_uncertainty_s is not None
        gap_codes = {gap.code for gap in result.ng_agent_observations.gaps}
        assert "model_call_ownership_unavailable" not in gap_codes
        assert "sandbox_lifecycle_timing_unavailable" in gap_codes
        assert "sandbox_cleanup_failed" not in gap_codes
        assert sandbox.exec.await_args_list[0].kwargs["command"] == "date +%s.%N"
        session_list_env = sandbox.exec.await_args_list[2].kwargs["env"]
        export_env = sandbox.exec.await_args_list[3].kwargs["env"]
        remote_data_home = session_list_env["XDG_DATA_HOME"]
        assert remote_data_home.startswith("/tmp/nemo-gym-opencode-")
        assert f"XDG_DATA_HOME={remote_data_home}" in sandbox.exec.await_args_list[1].kwargs["command"]
        assert export_env["XDG_DATA_HOME"] == remote_data_home
        assert (
            "opencode export session-id > /tmp/opencode_export.json"
            in sandbox.exec.await_args_list[3].kwargs["command"]
        )
        assert not hasattr(request.state, "_ng_observation_invocation_id")
        assert server._sandbox_id_to_run_result == {}
        assert not (tmp_path / "results" / "session-1" / "opencode.db").exists()

        run_command = sandbox.exec.await_args_list[1].kwargs["command"]
        phases = {r.name: r for r in result.ng_agent_observations.records if isinstance(r, AgentPhaseObservation)}
        assert set(phases) >= {
            "seed_session",
            "start_sandbox",
            "agent_sandbox_exec",
            "agent_export",
            "verify",
            "sandbox_stop",
        }
        assert phases["sandbox_stop"].status == "failed" and all(p.duration_ms is not None for p in phases.values())
        assert phases["seed_session"].completed_at <= phases["start_sandbox"].started_at
        assert (
            tool.clock_domain == "sandbox:connected-sandbox" and invocation.clock_domain == "sandbox:connected-sandbox"
        )
        # The harness's view of each step rides along as a trajectory record the collector merges with the capture.
        assert result.ng_trajectory is not None and result.ng_trajectory.rollout_id == "7-2"
        [step] = result.ng_trajectory.model_calls
        assert (step.invocation_id, step.completed_at, step.processed_at) == ("root", 0.001, 0.003)
        if latency_metrics:
            assert "NG_T_LAUNCH=" in run_command and ".config/opencode/plugins" in run_command
            assert (invocation.process_started_at, invocation.process_exited_at) == (1700000000.1, 1700000000.9)
        else:
            assert "NG_T_LAUNCH=" not in run_command and "plugins" not in run_command
            assert invocation.process_started_at is None


class TestSqliteSnapshotScript:
    """The snapshot runs inside the instance sandbox, on whatever ``python3`` that image ships.

    SWE-bench pins django <= 3.2, scikit-learn <= 0.22 and astropy 1.3 to Python 3.6, which has
    no ``sqlite3.Connection.backup``. Those instances lost their whole trajectory, so the script
    must not depend on it.
    """

    @staticmethod
    def _seed_database(path: Path) -> sqlite3.Connection:
        """Seed a WAL database and leave it open, as OpenCode holds it when the snapshot runs."""
        connection = sqlite3.connect(path)
        connection.execute("pragma journal_mode=wal")
        connection.execute("create table session (id text, parent_id text, time_created integer)")
        connection.execute("create table part (id text, session_id text, data text, blob blob)")
        connection.execute("create index part_session on part (session_id)")
        connection.commit()
        connection.execute("pragma wal_checkpoint(truncate)")
        connection.execute("insert into session values ('root', null, 0)")
        connection.execute(
            "insert into part values (?, ?, ?, ?)",
            ("p1", "root", json.dumps({"tool": "bash", "input": {"command": "ls -la 'quoted'"}}), b"\x00\xff"),
        )
        connection.commit()
        return connection

    @staticmethod
    def _contents(path: Path) -> tuple[Any, ...]:
        connection = sqlite3.connect(path)
        try:
            return (
                connection.execute("select * from session").fetchall(),
                connection.execute("select * from part").fetchall(),
                sorted(row[0] for row in connection.execute("select name from sqlite_master where type='index'")),
            )
        finally:
            connection.close()

    def _run(self, script: str, source: Path, destination: Path) -> None:
        result = subprocess.run(
            [sys.executable, "-c", script, str(source), str(destination)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_snapshot_preserves_rows_written_only_to_the_wal(self, tmp_path: Path) -> None:
        source = tmp_path / "opencode.db"
        connection = self._seed_database(source)
        try:
            # The rows are in the -wal, not the main file: a plain copy would lose them.
            assert source.with_name(f"{source.name}-wal").stat().st_size > 0
            main_only = tmp_path / "main-only.db"
            main_only.write_bytes(source.read_bytes())
            assert self._contents(main_only)[0] == []

            destination = tmp_path / "snapshot.db"
            self._run(SQLITE_SNAPSHOT_SCRIPT, source, destination)
        finally:
            connection.close()

        sessions, parts, indexes = self._contents(destination)
        assert sessions == [("root", None, 0)]
        assert json.loads(parts[0][2])["input"]["command"] == "ls -la 'quoted'"
        assert parts[0][3] == b"\x00\xff"
        assert "part_session" in indexes

    def test_snapshot_without_connection_backup_matches_backup(self, tmp_path: Path) -> None:
        source = tmp_path / "opencode.db"
        connection = self._seed_database(source)

        # Python 3.6 reaches the same branch by not having ``Connection.backup`` at all; forcing
        # the guard false is the only way to exercise it from a modern interpreter.
        fallback_script = SQLITE_SNAPSHOT_SCRIPT.replace("hasattr(source,'backup')", "False")
        assert fallback_script != SQLITE_SNAPSHOT_SCRIPT
        assert "iterdump" in fallback_script

        with_backup = tmp_path / "with_backup.db"
        without_backup = tmp_path / "without_backup.db"
        try:
            self._run(SQLITE_SNAPSHOT_SCRIPT, source, with_backup)
            self._run(fallback_script, source, without_backup)
        finally:
            connection.close()

        assert self._contents(without_backup) == self._contents(with_backup)
        assert self._contents(with_backup)[0] == [("root", None, 0)]


class TestInvocationSpans:
    """agent_start/agent_end, and the same bounds for each subagent session.

    Without these the analysis cannot compute e2e_agent_time, pre/post-agent time, or any
    subagent span. OpenCode records them on its own in-sandbox clock, the same clock its tool
    timings come from.
    """

    @staticmethod
    def _database(tmp_path: Path, sessions: list[tuple], messages: list[tuple]) -> Path:
        path = tmp_path / "opencode.db"
        connection = sqlite3.connect(path)
        try:
            connection.execute("create table session (id text, parent_id text, time_created integer)")
            connection.execute("create table message (id text, session_id text, data text, time_created integer)")
            connection.execute(
                "create table part (id text, message_id text, session_id text, data text, time_created integer)"
            )
            connection.executemany("insert into session values (?, ?, ?)", sessions)
            connection.executemany("insert into message values (?, ?, ?, ?)", messages)
            connection.commit()
        finally:
            connection.close()
        return path

    @staticmethod
    def _invocations(bundle: AgentObservationBundle) -> dict[str, AgentInvocation]:
        return {r.invocation_id: r for r in bundle.records if isinstance(r, AgentInvocation)}

    def test_span_runs_from_the_first_message_to_the_last_assistant_completion(self, tmp_path: Path) -> None:
        messages = [
            ("m1", "root", json.dumps({"role": "user", "time": {"created": 1_000}}), 1),
            ("m2", "root", json.dumps({"role": "assistant", "time": {"created": 1_500, "completed": 2_000}}), 2),
            ("m3", "root", json.dumps({"role": "assistant", "time": {"created": 2_500, "completed": 9_000}}), 3),
        ]
        bundle = parse_opencode_observations(self._database(tmp_path, [("root", None, 500)], messages), "fallback")
        invocation = self._invocations(bundle)["root"]
        assert (invocation.started_at, invocation.completed_at) == (1.0, 9.0)
        assert invocation.status == "completed"

    def test_a_subagent_session_carries_its_own_span(self, tmp_path: Path) -> None:
        sessions = [("root", None, 500), ("child", "root", 3_000)]
        messages = [
            ("m1", "root", json.dumps({"role": "user", "time": {"created": 1_000}}), 1),
            ("m2", "root", json.dumps({"role": "assistant", "time": {"created": 1_500, "completed": 9_000}}), 2),
            ("m3", "child", json.dumps({"role": "user", "time": {"created": 3_100}}), 3),
            ("m4", "child", json.dumps({"role": "assistant", "time": {"created": 3_200, "completed": 4_000}}), 4),
        ]
        invocations = self._invocations(parse_opencode_observations(self._database(tmp_path, sessions, messages), "f"))
        assert (invocations["root"].started_at, invocations["root"].completed_at) == (1.0, 9.0)
        assert (invocations["child"].started_at, invocations["child"].completed_at) == (3.1, 4.0)
        assert invocations["child"].parent_invocation_id == "root"

    def test_session_creation_time_is_the_fallback_start(self, tmp_path: Path) -> None:
        messages = [("m1", "root", json.dumps({"role": "user"}), 1)]
        bundle = parse_opencode_observations(self._database(tmp_path, [("root", None, 2_500)], messages), "fallback")
        invocation = self._invocations(bundle)["root"]
        assert invocation.started_at == 2.5
        assert invocation.completed_at is None

    def test_inconsistent_artifact_timestamps_are_dropped_not_raised(self, tmp_path: Path) -> None:
        # A completion before the first message would fail AgentInvocation validation and cost
        # the entire bundle, so it is reported as a gap instead.
        messages = [
            ("m1", "root", json.dumps({"role": "user", "time": {"created": 9_000}}), 1),
            ("m2", "root", json.dumps({"role": "assistant", "time": {"created": 9_100, "completed": 1_000}}), 2),
        ]
        bundle = parse_opencode_observations(self._database(tmp_path, [("root", None, 8_000)], messages), "fallback")
        invocation = self._invocations(bundle)["root"]
        assert invocation.started_at == 9.0
        assert invocation.completed_at is None
        assert "agent_span_timing_inconsistent" in {gap.code for gap in bundle.gaps}

    def test_tool_call_requested_at_is_the_part_creation_time(self, tmp_path: Path) -> None:
        # OpenCode writes the tool part when the model emits the call and stamps state.time.start
        # only when it runs; the row's creation time is therefore tool_call_requested, and the
        # gap to started_at is the dispatch delay the analysis wants.
        db = self._database(
            tmp_path,
            [("root", None, 500)],
            [("m1", "root", json.dumps({"role": "assistant", "time": {"created": 1_000, "completed": 9_000}}), 1)],
        )
        connection = sqlite3.connect(db)
        try:
            connection.execute(
                "insert into part values (?, ?, ?, ?, ?)",
                (
                    "p1",
                    "m1",
                    "root",
                    json.dumps(
                        {
                            "type": "tool",
                            "tool": "bash",
                            "callID": "call-1",
                            "state": {
                                "status": "completed",
                                "input": {"command": "pytest -x"},
                                "output": "",
                                "time": {"start": 2_500, "end": 4_000},
                            },
                        }
                    ),
                    2_000,
                ),
            )
            connection.commit()
        finally:
            connection.close()
        bundle = parse_opencode_observations(db, "fallback")
        [tool] = [r for r in bundle.records if isinstance(r, ToolCallObservation)]
        # started_at is the request too: OpenCode rewrites the state start while a streaming tool runs.
        assert (tool.requested_at, tool.started_at, tool.completed_at) == (2.0, 2.0, 4.0)
        # No later assistant message: nothing bounds when the result reached the next prompt.
        assert tool.response_received_at is None
        assert "tool_response_boundary_approximate" not in {g.code for g in bundle.gaps}

    def test_response_received_is_bounded_by_the_next_model_call_and_flagged(self, tmp_path: Path) -> None:
        db = self._database(
            tmp_path,
            [("root", None, 500)],
            [
                ("m1", "root", json.dumps({"role": "assistant", "time": {"created": 1_000, "completed": 2_200}}), 1),
                ("m2", "root", json.dumps({"role": "assistant", "time": {"created": 5_000, "completed": 6_000}}), 2),
            ],
        )
        connection = sqlite3.connect(db)
        try:
            connection.execute(
                "insert into part values (?, ?, ?, ?, ?)",
                (
                    "p1",
                    "m1",
                    "root",
                    json.dumps(
                        {
                            "type": "tool",
                            "tool": "bash",
                            "callID": "call-1",
                            "state": {
                                "status": "completed",
                                "input": {"command": "ls"},
                                "output": "",
                                "time": {"start": 2_500, "end": 4_000},
                            },
                        }
                    ),
                    2_000,
                ),
            )
            connection.commit()
        finally:
            connection.close()
        bundle = parse_opencode_observations(db, "fallback")
        [tool] = [r for r in bundle.records if isinstance(r, ToolCallObservation)]
        assert tool.response_received_at == 5.0
        gaps = {(g.code, g.detail) for g in bundle.gaps}
        assert ("tool_response_boundary_approximate", "next_assistant_message_created") in gaps


class TestSandboxClockProbe:
    """OpenCode times tools and messages on the sandbox clock, model calls are on the harness
    clock; one round trip at start gives the offset and how far it can be trusted."""

    @staticmethod
    def _sandbox(stdout: str, return_code: int = 0, delay_s: float = 0.0):
        async def exec(command: str):
            assert command == "date +%s.%N"
            await asyncio.sleep(delay_s)
            return SimpleNamespace(stdout=stdout, stderr="", return_code=return_code, error_type=None)

        return SimpleNamespace(exec=exec)

    async def test_offset_is_measured_against_the_round_trip_midpoint(self, monkeypatch: MonkeyPatch) -> None:
        clock = iter([1000.0, 1000.2])  # sent, received
        monkeypatch.setattr("responses_api_agents.opencode_sandboxed_agent.app.time", lambda: next(clock))
        offset, uncertainty = await _probe_sandbox_clock(self._sandbox("1003.100000000"))
        assert offset == pytest.approx(1003.1 - 1000.1)
        assert uncertainty == pytest.approx(0.1)

    async def test_a_failed_probe_reports_nothing_rather_than_a_guess(self) -> None:
        assert await _probe_sandbox_clock(self._sandbox("", return_code=127)) == (None, None)
        assert await _probe_sandbox_clock(self._sandbox("date: not found")) == (None, None)

    async def test_a_raising_sandbox_does_not_break_the_rollout(self) -> None:
        async def exec(command: str):
            raise RuntimeError("sandbox gone")

        assert await _probe_sandbox_clock(SimpleNamespace(exec=exec)) == (None, None)


class TestLatencyEvidence:
    """What OpenCode's own artifacts say about steps, tools, compactions and sub-agents, in the schema."""

    @staticmethod
    def _ms(seconds: float) -> int:
        return int(round(seconds * 1000))

    def _database(self, tmp_path: Path) -> Path:
        ms = self._ms
        path = tmp_path / "opencode.db"
        con = sqlite3.connect(path)
        con.execute("create table session (id text, parent_id text, time_created integer)")
        con.execute("create table message (id text, session_id text, data text, time_created integer)")
        con.execute("create table part (id text, message_id text, session_id text, data text, time_created integer)")

        def message(id_, session_id, data, t):
            con.execute("insert into message values (?,?,?,?)", (id_, session_id, json.dumps(data), ms(t)))

        def part(id_, message_id, session_id, data, t):
            con.execute("insert into part values (?,?,?,?,?)", (id_, message_id, session_id, json.dumps(data), ms(t)))

        def tokens(input_, output, reasoning, cached=0):
            return {"input": input_, "output": output, "reasoning": reasoning, "cache": {"read": cached, "write": 0}}

        con.execute("insert into session values ('root', null, ?)", (ms(100.0),))
        message("u1", "root", {"role": "user", "time": {"created": ms(100.05)}}, 100.05)
        # step 1: reasoning, a bash tool whose state start was rewritten while it streamed, a task tool spawning
        # a sub-agent, then step-finish with the step's token counts
        message(
            "a1",
            "root",
            {
                "role": "assistant",
                "parentID": "u1",
                "time": {"created": ms(100.1), "completed": ms(130.0)},
                "tokens": tokens(1000, 50, 20, 300),
            },
            100.1,
        )
        part("ss1", "a1", "root", {"type": "step-start"}, 100.1)
        part(
            "r1",
            "a1",
            "root",
            {"type": "reasoning", "text": "think", "time": {"start": ms(105.0), "end": ms(105.02)}},
            105.0,
        )
        part(
            "t1",
            "a1",
            "root",
            {
                "type": "tool",
                "tool": "bash",
                "callID": "chatcmpl-tool-1",
                "state": {
                    "status": "completed",
                    "input": {"command": "pytest -q"},
                    "output": "ok",
                    "title": "pytest",
                    "time": {"start": ms(161.0), "end": ms(161.01)},
                },
            },
            105.05,
        )
        part(
            "t2",
            "a1",
            "root",
            {
                "type": "tool",
                "tool": "task",
                "callID": "chatcmpl-tool-2",
                "state": {
                    "status": "error",
                    "input": {"description": "explore"},
                    "error": "boom",
                    "title": "explore",
                    "metadata": {"sessionId": "child"},
                    "time": {"start": ms(106.5), "end": ms(129.0)},
                },
            },
            105.06,
        )
        part(
            "sf1",
            "a1",
            "root",
            {"type": "step-finish", "reason": "tool-calls", "tokens": tokens(1000, 50, 20, 300)},
            130.0,
        )
        con.execute("insert into session values ('child', 'root', ?)", (ms(106.6),))
        message("cu1", "child", {"role": "user", "time": {"created": ms(106.7)}}, 106.7)
        message(
            "ca1",
            "child",
            {
                "role": "assistant",
                "parentID": "cu1",
                "time": {"created": ms(106.8), "completed": ms(128.0)},
                "tokens": tokens(300, 10, 5),
            },
            106.8,
        )
        part("css", "ca1", "child", {"type": "step-start"}, 106.8)
        part(
            "ctx",
            "ca1",
            "child",
            {"type": "text", "text": "found it", "time": {"start": ms(120.0), "end": ms(128.0)}},
            120.0,
        )
        part("csf", "ca1", "child", {"type": "step-finish", "reason": "stop", "tokens": tokens(300, 10, 5)}, 128.0)
        # compaction after step 1, then the summary message, then the next agent step
        message("u2", "root", {"role": "user", "time": {"created": ms(131.0)}}, 131.0)
        part("cp", "u2", "root", {"type": "compaction", "auto": True}, 131.0)
        message(
            "s1",
            "root",
            {
                "role": "assistant",
                "parentID": "u2",
                "summary": True,
                "time": {"created": ms(131.1), "completed": ms(140.0)},
                "tokens": tokens(1100, 200, 0),
            },
            131.1,
        )
        part(
            "st",
            "s1",
            "root",
            {"type": "text", "text": "summary", "time": {"start": ms(135.0), "end": ms(140.0)}},
            135.0,
        )
        message(
            "a2",
            "root",
            {
                "role": "assistant",
                "parentID": "u2",
                "time": {"created": ms(141.0), "completed": ms(150.0)},
                "tokens": tokens(400, 30, 10),
            },
            141.0,
        )
        part("ss2", "a2", "root", {"type": "step-start"}, 141.0)
        part(
            "tx2",
            "a2",
            "root",
            {"type": "text", "text": "done", "time": {"start": ms(145.0), "end": ms(150.0)}},
            145.0,
        )
        part("sf2", "a2", "root", {"type": "step-finish", "reason": "stop", "tokens": tokens(400, 30, 10)}, 150.0)
        con.commit()
        con.close()
        return path

    def test_steps_become_harness_model_calls(self, tmp_path: Path) -> None:
        steps = app_module.parse_opencode_steps(self._database(tmp_path), sandbox_id="sb-1")
        assert [(s.invocation_id, s.model_call_purpose) for s in steps] == [
            ("root", "agent_step"),
            ("child", "subagent_step"),
            ("root", "compaction_summary"),
            ("root", "agent_step"),
        ]
        first = steps[0]
        assert first.started_at is None  # the plugin's send time comes from the capture record
        assert (first.completed_at, first.processed_at) == (105.0, 130.0)  # first part arrival; step finished
        assert first.clock_domain == "sandbox:sb-1"
        assert (first.token_stats.prompt_tokens, first.token_stats.completion_tokens) == (1000, 50)
        assert (first.token_stats.reasoning_tokens, first.token_stats.cached_tokens) == (20, 300)
        assert steps[2].completed_at == 135.0 and steps[2].processed_at == 140.0

    def test_tool_starts_when_requested_and_statuses_map(self, tmp_path: Path) -> None:
        bundle = parse_opencode_observations(self._database(tmp_path), "fallback")
        tools = {t.tool_call_id: t for t in bundle.records if isinstance(t, ToolCallObservation)}
        bash = tools["chatcmpl-tool-1"]
        # OpenCode rewrites the state's start while a streaming tool runs: a 56 s pytest looked like a 10 ms tool.
        assert bash.started_at == bash.requested_at == 105.05 and bash.completed_at == 161.01
        assert bash.duration_ms == pytest.approx(55_960.0, abs=1.0) and bash.status == "completed"
        assert tools["chatcmpl-tool-2"].status == "failed"

    def test_compaction_gets_its_end_and_token_counts_from_the_messages(self, tmp_path: Path) -> None:
        bundle = parse_opencode_observations(self._database(tmp_path), "fallback")
        [compaction] = [r for r in bundle.records if isinstance(r, ContextCompactionObservation)]
        assert (compaction.observed_at, compaction.completed_at) == (131.0, 140.0)
        assert compaction.trigger == "automatic" and compaction.outcome == "completed"
        assert compaction.tokens_before == 1000 + 50 + 20 and compaction.tokens_after == 400
        assert "compaction_token_counts_unavailable" not in {g.code for g in bundle.gaps}

    def test_subagent_carries_requested_and_observed_times(self, tmp_path: Path) -> None:
        bundle = parse_opencode_observations(self._database(tmp_path), "fallback")
        child = next(r for r in bundle.records if isinstance(r, AgentInvocation) and r.invocation_id == "child")
        assert child.spawned_by_tool_call_id == "chatcmpl-tool-2"
        assert (child.requested_at, child.response_observed_at) == (105.06, 129.0)
        assert (child.started_at, child.completed_at) == (106.7, 128.0)

    def test_process_marks_plugin_command_and_exec_outcome(self) -> None:
        wrapped = app_module.wrap_run_command("false")
        completed = subprocess.run(
            ["bash", "-c", wrapped + ' && echo "OpenCode run finished"'], capture_output=True, text=True
        )
        assert completed.returncode == 1 and "OpenCode run finished" not in completed.stdout
        launch, exit_ = app_module.parse_marks(completed.stdout)
        assert launch is not None and exit_ is not None and exit_ >= launch
        assert app_module.parse_marks("garbage") == (None, None)
        install = app_module.plugin_install_command()
        assert '"$HOME/.config/opencode/plugins"' in install and "/testbed" not in install
        for header in ("x-session-id", "x-ng-agent", "x-ng-client-call-id", "x-ng-client-sent-at"):
            assert header in app_module.PLUGIN_SOURCE
        assert app_module._exec_outcome(0, None, True) == "completed"
        assert app_module._exec_outcome(-1, "timeout", False) == "timeout"
        assert app_module._exec_outcome(1, None, False) == "failed"
        assert app_module._exec_outcome(None, "SandboxError", False) == "sandbox_error"
