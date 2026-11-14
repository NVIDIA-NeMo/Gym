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
import shutil
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

from pytest import MonkeyPatch, fixture, mark, skip

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
    SandboxObservation,
    ToolCallObservation,
    TrajectoryRecord,
)
from nemo_gym.sandbox import SandboxHandle
from nemo_gym.sandbox.utils import CPU_CAP_ENV_VARS
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.opencode_sandboxed_agent import app as app_module
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
)


class TestOpenCodeSandboxedAgent:
    def test_import_only_loads_shared_opencode_observability(self) -> None:
        code = (
            "import sys; import responses_api_agents.opencode_sandboxed_agent.app; "
            "assert {name for name in sys.modules if name == 'responses_api_agents.opencode_agent' "
            "or name.startswith('responses_api_agents.opencode_agent.')} == "
            "{'responses_api_agents.opencode_agent', 'responses_api_agents.opencode_agent.observability'}"
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

    async def test_start_sandbox_connects_in_the_seeded_workdir(self, monkeypatch: MonkeyPatch) -> None:
        async_sandbox = MagicMock()
        async_sandbox.connect = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {})
        monkeypatch.setattr(app_module, "create_provider", lambda *_: MagicMock())
        monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: MagicMock())
        monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
        monkeypatch.setattr(app_module, "AsyncSandbox", async_sandbox)
        server = OpenCodeSandboxedAgent(config=self._create_config(), server_client=MagicMock(spec=ServerClient))

        await server._start_sandbox(sandbox_id="sb-1", workdir="/workspace/repo")
        assert async_sandbox.connect.await_args.args[0] == {"sandbox_id": "sb-1", "workdir": "/workspace/repo"}

        await server._start_sandbox(sandbox_id="sb-2")
        assert async_sandbox.connect.await_args.args[0] == {"sandbox_id": "sb-2", "workdir": None}

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

    async def test_data_home_in_tmp_sets_xdg_for_run_and_session_commands(
        self, opencode_export_test_data: Dict[str, Any], monkeypatch: MonkeyPatch
    ) -> None:
        config = self._create_config().model_copy(update={"opencode_data_home_in_tmp": True})
        server = OpenCodeSandboxedAgent(config=config, server_client=MagicMock(spec=ServerClient))
        sandbox_mock = MagicMock()
        sandbox_mock.exec = AsyncMock(
            side_effect=[
                SimpleNamespace(stdout="OpenCode run finished", stderr="", return_code=0, error_type=None),
                SimpleNamespace(stdout='[{"id": "session-id"}]', stderr="", return_code=0, error_type=None),
                SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None),
            ]
        )
        sandbox_mock.download = AsyncMock()
        monkeypatch.setattr(server, "_sandbox_id_to_sandbox", {"": sandbox_mock})
        monkeypatch.setattr(server, "_create_opencode_config", AsyncMock(return_value=dict()))
        monkeypatch.setattr("responses_api_agents.opencode_sandboxed_agent.app.Path.exists", lambda self: True)
        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.Path.read_text",
            lambda self: json.dumps(opencode_export_test_data),
        )
        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.uuid4", MagicMock(return_value=MagicMock(hex="abc"))
        )

        await server.responses(
            request=MagicMock(
                session={SESSION_ID_KEY: "my session"},
                cookies={"sandbox_id": ""},
                path_params={"rollout_id": "direct-call"},
            ),
            body=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hello"}]),
        )

        run_call, session_list_call, export_call = sandbox_mock.exec.await_args_list[:3]
        # data, state, cache and config all move off $HOME (pyautogui's image failed on ~/.local/state otherwise)
        expected = {
            "XDG_DATA_HOME": "/tmp/nemo-gym-opencode-abc",
            "XDG_STATE_HOME": "/tmp/nemo-gym-opencode-abc-state",
            "XDG_CACHE_HOME": "/tmp/nemo-gym-opencode-abc-cache",
            "XDG_CONFIG_HOME": "/tmp/nemo-gym-opencode-abc-config",
        }
        for key, value in expected.items():
            assert f"{key}={value}" in run_call.kwargs["command"]
        # the session list/export must read the same dirs the run wrote to
        assert session_list_call.kwargs["env"] == expected
        assert export_call.kwargs["env"] == expected

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

    async def test_run_builds_observations_from_live_wal_snapshot(
        self,
        tmp_path: Path,
        opencode_export_test_data: Dict[str, Any],
        monkeypatch: MonkeyPatch,
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
        for part_id, kind in [("start", "step-start"), ("finish", "step-finish")]:
            connection.execute(
                "insert into part values (?, 'm1', 'root', ?, 2)", (part_id, json.dumps({"type": kind}))
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
        server = OpenCodeSandboxedAgent(config=self._create_config(), server_client=server_client)
        server._create_opencode_config = AsyncMock(return_value={})

        sandbox = MagicMock()
        sandbox._handle = SandboxHandle(sandbox_id="connected-sandbox", provider_name="opensandbox", raw=None)
        sandbox.exec = AsyncMock(
            side_effect=[
                SimpleNamespace(
                    stdout="Shell: /bin/bash\nOpenCode run finished", stderr="", return_code=0, error_type=None
                ),
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
        [turn] = TrajectoryRecord.model_validate(result.ng_trajectory).turns
        assert (turn.task_id, turn.rollout_id, turn.invocation_id) == ("7", "7-2", "root")
        assert turn.answer[0]["call_id"] == "call-1"
        assert not turn.model_calls
        [invocation] = [
            record for record in result.ng_agent_observations.records if isinstance(record, AgentInvocation)
        ]
        assert invocation.invocation_id == "root"
        assert invocation.status == "completed"
        [tool] = [record for record in result.ng_agent_observations.records if isinstance(record, ToolCallObservation)]
        assert tool.tool_call_id == "call-1"
        assert tool.sandbox_id == "connected-sandbox"
        assert tool.duration_ms == 1_000
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
        gap_codes = {gap.code for gap in result.ng_agent_observations.gaps}
        assert "model_call_ownership_unavailable" not in gap_codes
        assert "sandbox_lifecycle_timing_unavailable" in gap_codes
        assert "sandbox_cleanup_failed" not in gap_codes
        session_list_env = sandbox.exec.await_args_list[1].kwargs["env"]
        export_env = sandbox.exec.await_args_list[2].kwargs["env"]
        remote_data_home = session_list_env["XDG_DATA_HOME"]
        assert remote_data_home.startswith("/tmp/nemo-gym-opencode-")
        assert f"XDG_DATA_HOME={remote_data_home}" in sandbox.exec.await_args_list[0].kwargs["command"]
        assert export_env["XDG_DATA_HOME"] == remote_data_home
        assert (
            "opencode export session-id > /tmp/opencode_export.json"
            in sandbox.exec.await_args_list[2].kwargs["command"]
        )
        assert not hasattr(request.state, "_ng_observation_invocation_id")
        assert server._sandbox_id_to_run_result == {}
        assert not (tmp_path / "results" / "session-1" / "opencode.db").exists()


def _export(last_parts, error=None, finish=None) -> Dict[str, Any]:
    info = {"role": "assistant"}
    if error:
        info["error"] = error
    if finish:
        info["finish"] = finish
    return {
        "messages": [
            {"info": {"role": "user"}, "parts": [{"type": "text", "text": "build it"}]},
            {"info": {"role": "assistant"}, "parts": [{"type": "tool", "tool": "read"}]},
            {"info": info, "parts": last_parts},
        ]
    }


def test_idle_reason_follows_the_programbench_idle_nudge_rule() -> None:
    reason = app_module._idle_reason
    # any stop without a tool call is idle: reasoning only, empty, or text that just stops
    assert reason(_export([{"type": "reasoning", "text": "plan ... from .util"}])) == "idle"
    assert reason(_export([])) == "idle"
    assert reason(_export([{"type": "text", "text": "I'll implement the parser next."}], finish="stop")) == "idle"
    # unless the model says the agreed sentinel
    assert reason(_export([{"type": "text", "text": "All done. TASK_COMPLETE"}]), "TASK_COMPLETE") is None
    # a turn cut off at the output token limit (the 2026-11-02 live case, partial text and all)
    cut = [{"type": "text", "text": "Let me set up a task plan"}, {"type": "step-finish", "reason": "length"}]
    assert reason(_export(cut, finish="length"), "TASK_COMPLETE") == "length"
    assert reason(_export(cut), "TASK_COMPLETE") == "length"
    # not continued: a tool call, an API error, no assistant turn
    assert reason(_export([{"type": "tool", "tool": "bash"}], finish="length")) is None
    assert reason(_export([{"type": "reasoning", "text": "x"}], error={"name": "APIError"})) is None
    assert reason({"messages": []}) is None


class TestIdleNudge:
    TEXT = "Continue working now. If nothing is left to do, reply with exactly TASK_COMPLETE."

    def _server(self, nudges: int) -> Any:
        config = (
            TestOpenCodeSandboxedAgent()
            ._create_config()
            .model_copy(
                update={
                    "idle_nudge": app_module.IdleNudgeConfig(
                        text=self.TEXT, max_nudges=nudges, stop_on_text="TASK_COMPLETE"
                    )
                }
            )
        )
        return OpenCodeSandboxedAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def _sandbox(self, exports: list, stuck_marker: str = "") -> MagicMock:
        """exec answers by command; each download writes the next session export."""

        async def exec_(command: str, **kwargs: Any) -> SimpleNamespace:
            if "nemo_gym_stuck/stuck.json" in command:
                return SimpleNamespace(stdout=stuck_marker, stderr="", return_code=0, error_type=None)
            if "session list" in command:
                return SimpleNamespace(stdout='[{"id": "ses_1"}]', stderr="", return_code=0, error_type=None)
            if "opencode export" in command:
                return SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None)
            if "opencode run --session" in command:
                return SimpleNamespace(
                    stdout="working\nOpenCode run finished", stderr="", return_code=0, error_type=None
                )
            raise AssertionError(command)

        remaining = list(exports)

        async def download(remote: str, local: Path) -> None:
            Path(local).write_text(json.dumps(remaining.pop(0)))

        sandbox = MagicMock()
        sandbox.exec = AsyncMock(side_effect=exec_)
        sandbox.download = AsyncMock(side_effect=download)
        return sandbox

    async def _run(
        self, server: Any, sandbox: Any, first: Any, seconds_left: float = 3600, config: str = "{}"
    ) -> tuple:
        return await server._nudge_idle_turns(
            sandbox=sandbox,
            first_result=first,
            config_content=config,
            env_suffix="OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1",
            run_flags="--thinking",
            session_env={"XDG_DATA_HOME": "/tmp/x"},
            deadline=app_module.time() + seconds_left,
        )

    FIRST = SimpleNamespace(
        stdout="Shell: /bin/bash\nOpenCode run finished", stderr="", return_code=0, error_type=None
    )

    def _resumes(self, sandbox: Any) -> list:
        return [
            c.kwargs["command"]
            for c in sandbox.exec.await_args_list
            if "opencode run --session" in c.kwargs["command"]
        ]

    async def test_continues_until_the_model_says_task_complete(self) -> None:
        stopped = _export([{"type": "text", "text": "Next I will write the tests."}], finish="stop")
        done = _export([{"type": "text", "text": "TASK_COMPLETE"}], finish="stop")
        sandbox = self._sandbox([stopped, done])
        result, nudges = await self._run(self._server(3), sandbox, self.FIRST)
        assert nudges == 1
        [resume] = self._resumes(sandbox)
        assert "--session ses_1" in resume and shlex.quote(self.TEXT) in resume
        assert "OPENCODE_CONFIG_CONTENT='{}'" in resume and "--thinking" in resume
        assert "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1" in resume
        assert result.stdout.endswith("OpenCode run finished")  # opencode_finished still detected after the resume

    async def test_length_cutoff_gets_the_length_note_first(self) -> None:
        cut = _export([{"type": "text", "text": "Let me set up"}], finish="length")
        done = _export([{"type": "text", "text": "TASK_COMPLETE"}], finish="stop")
        sandbox = self._sandbox([cut, done])
        _, nudges = await self._run(self._server(3), sandbox, self.FIRST)
        assert nudges == 1
        [resume] = self._resumes(sandbox)
        assert shlex.quote(f"{app_module.LENGTH_NOTE}\n{self.TEXT}") in resume

    async def test_stops_at_the_limit_after_an_unclean_exit_and_when_time_is_short(self) -> None:
        idle = _export([{"type": "reasoning", "text": "cut"}])
        _, nudges = await self._run(self._server(2), self._sandbox([idle, idle, idle]), self.FIRST)
        assert nudges == 2
        crashed = SimpleNamespace(stdout="Shell: x\nError: boom", stderr="", return_code=1, error_type=None)
        _, nudges = await self._run(self._server(2), self._sandbox([idle]), crashed)
        assert nudges == 0
        _, nudges = await self._run(self._server(2), self._sandbox([idle]), self.FIRST, seconds_left=30)
        assert nudges == 0

    def _server_with(self, **nudge) -> Any:
        config = (
            TestOpenCodeSandboxedAgent()
            ._create_config()
            .model_copy(
                update={
                    "idle_nudge": app_module.IdleNudgeConfig(text=self.TEXT, stop_on_text="TASK_COMPLETE", **nudge)
                }
            )
        )
        return OpenCodeSandboxedAgent(config=config, server_client=MagicMock(spec=ServerClient))

    @staticmethod
    def _with_steps(export: Dict[str, Any], steps: int) -> Dict[str, Any]:
        msgs = [{"info": {"role": "assistant"}, "parts": [{"type": "step-finish"}]} for _ in range(steps)]
        return {"messages": msgs + export["messages"]}

    async def test_step_cap_is_one_budget_across_resumes(self) -> None:
        # OpenHands' max_iterations: the session used 120 of 500 steps, so the resume may take 380 more
        idle = self._with_steps(_export([{"type": "reasoning", "text": "x"}]), 119)  # +1 step-finish-free last turn
        done = _export([{"type": "text", "text": "TASK_COMPLETE"}])
        sandbox = self._sandbox([idle, done])
        config = json.dumps({"agent": {"build": {"steps": 500}}, "permission": {"*": "allow"}})
        _, nudges = await self._run(self._server_with(max_nudges=100), sandbox, self.FIRST, config=config)
        assert nudges == 1
        [resume] = self._resumes(sandbox)
        sent = json.loads(shlex.split(resume.split("OPENCODE_CONFIG_CONTENT=", 1)[1])[0])
        assert sent["agent"]["build"]["steps"] == 500 - 119 and sent["permission"] == {"*": "allow"}

    async def test_no_resume_once_the_step_budget_is_spent(self) -> None:
        idle = self._with_steps(_export([{"type": "reasoning", "text": "x"}]), 500)
        sandbox = self._sandbox([idle])
        config = json.dumps({"agent": {"build": {"steps": 500}}})
        _, nudges = await self._run(self._server_with(max_nudges=100), sandbox, self.FIRST, config=config)
        assert nudges == 0 and self._resumes(sandbox) == []

    async def test_length_note_off_sends_the_text_alone(self) -> None:
        cut = _export([{"type": "text", "text": "Let me"}], finish="length")
        done = _export([{"type": "text", "text": "TASK_COMPLETE"}])
        sandbox = self._sandbox([cut, done])
        _, nudges = await self._run(self._server_with(length_note=False), sandbox, self.FIRST)
        [resume] = self._resumes(sandbox)
        assert nudges == 1 and app_module.LENGTH_NOTE not in resume and shlex.quote(self.TEXT) in resume

    # OpenHands' StuckDetector: the plugin's marker, and the monologue check made here
    def _stuck_server(self) -> Any:
        server = self._server_with(max_nudges=100)
        server.config = server.config.model_copy(update={"stuck_detector": True})
        return server

    async def test_no_nudge_after_the_plugin_ended_a_stuck_run(self) -> None:
        idle = _export([{"type": "text", "text": "hm"}])
        sandbox = self._sandbox([idle], stuck_marker='{"scenario": "repeating_action_observation"}\n')
        result, nudges = await self._run(self._stuck_server(), sandbox, self.FIRST)
        assert nudges == 0 and self._resumes(sandbox) == [] and result.stuck == "repeating_action_observation"

    async def test_monologue_ends_the_session(self) -> None:
        def said(*texts: str) -> Dict[str, Any]:
            msgs = [{"info": {"role": "user"}, "parts": [{"type": "text", "text": "build it"}]}]
            for t in texts:
                msgs.append({"info": {"role": "assistant"}, "parts": [{"type": "step-start"}, {"type": "text", "text": t}, {"type": "step-finish"}]})
                msgs.append({"info": {"role": "user"}, "parts": [{"type": "text", "text": "continue"}]})
            return {"messages": msgs[:-1]}

        sandbox = self._sandbox([said("done"), said("done", "done"), said("done", "done", "done")])
        result, nudges = await self._run(self._stuck_server(), sandbox, self.FIRST)
        assert nudges == 2 and result.stuck == "monologue"


def _step(*parts: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"type": "step-start"}, *parts, {"type": "step-finish"}]


def test_monologue_stuck_follows_openhands() -> None:
    text = lambda t: {"type": "text", "text": t}  # noqa: E731
    tool = {"type": "tool", "tool": "bash"}
    user = {"info": {"role": "user"}, "parts": [text("continue")]}
    agent = lambda *steps: {"info": {"role": "assistant"}, "parts": [p for s in steps for p in s]}  # noqa: E731
    same = {"messages": [agent(_step(text("ok"))), user, agent(_step(text("ok"))), user, agent(_step(text("ok")))]}
    assert app_module._monologue_stuck(same)
    differ = {"messages": [agent(_step(text("ok"))), user, agent(_step(text("ok!"))), user, agent(_step(text("ok")))]}
    assert not app_module._monologue_stuck(differ)
    # a tool call between them is an observation: not a monologue
    worked = {"messages": [agent(_step(text("ok"))), user, agent(_step(text("x"), tool), _step(text("ok"))), user, agent(_step(text("ok")))]}
    assert not app_module._monologue_stuck(worked)
    assert not app_module._monologue_stuck({"messages": [agent(_step(text("ok"))), user, agent(_step(text("ok")))]})


def test_stuck_detector_plugin_scenarios() -> None:
    node = shutil.which("node")
    if node is None:
        skip("node not installed")
    here = Path(__file__).parent
    out = subprocess.run([node, str(here / "stuck_detector_check.mjs")], capture_output=True, text=True, cwd=here)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr



def _session(sid: str, tokens: List[int], children: List[str]) -> Dict[str, Any]:
    """An export whose assistant messages used `tokens` input tokens each and whose task calls started `children`."""
    msgs: List[Dict[str, Any]] = [{"info": {"role": "user"}, "parts": [{"type": "text", "text": "go"}]}]
    for n in tokens:
        msgs.append(
            {
                "info": {"role": "assistant", "tokens": {"input": n, "output": 1, "reasoning": 0, "cache": {"read": 2, "write": 0}}},
                "parts": [{"type": "text", "text": "ok"}],
            }
        )
    for child in children:
        msgs.append(
            {
                "info": {"role": "assistant"},
                "parts": [
                    {
                        "type": "tool",
                        "tool": "task",
                        "callID": f"call_{child}",
                        "state": {"status": "completed", "input": {"prompt": "x"}, "metadata": {"sessionId": child},
                                  "output": f'<task id="{child}" state="completed">done</task>'},
                    }
                ],
            }
        )
    return {"info": {"id": sid}, "messages": msgs}


def test_child_session_ids_and_token_breakdown() -> None:
    main = _session("ses_main", [10, 20], ["ses_a", "ses_b"])
    # a resumed child is named by task_id; an errored call with no session names none
    main["messages"].append(
        {"info": {"role": "assistant"}, "parts": [
            {"type": "tool", "tool": "task", "state": {"status": "completed", "input": {"task_id": "ses_a"}, "output": ""}},
            {"type": "tool", "tool": "task", "state": {"status": "error", "input": {"prompt": "y"}, "error": "boom"}},
            {"type": "tool", "tool": "bash", "state": {"status": "completed", "input": {"command": "echo ses_zzz"}}},
        ]}
    )
    assert app_module._child_session_ids(main) == ["ses_a", "ses_b"]
    out = app_module._token_breakdown(main, {"ses_a": _session("ses_a", [5], []), "ses_b": _session("ses_b", [7, 1], [])})
    assert out["main"]["input"] == 30 and out["main"]["calls"] == 2 and out["main"]["cache_read"] == 4
    assert out["subagents"]["input"] == 13 and out["subagents"]["calls"] == 3
    assert out["per_subagent"]["ses_b"]["input"] == 8


async def test_subagent_sessions_are_exported_recursively(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    exports = {
        "ses_main": _session("ses_main", [100], ["ses_a"]),
        "ses_a": _session("ses_a", [40], ["ses_grand"]),
        "ses_grand": _session("ses_grand", [3], []),
    }
    written: Dict[str, str] = {}

    async def exec_(command: str, **kwargs: Any) -> SimpleNamespace:
        ok = lambda out="": SimpleNamespace(stdout=out, stderr="", return_code=0, error_type=None)  # noqa: E731
        if "session list" in command:
            return ok('[{"id": "ses_main"}]')
        if "opencode export" in command:
            sid = command.split("opencode export ", 1)[1].split()[0]
            written[command.rsplit("> ", 1)[1].strip()] = json.dumps(exports[sid])
            return ok()
        return ok("Shell: /bin/bash\nOpenCode run finished")

    async def download(remote: str, local: Path) -> None:
        Path(local).write_text(written[remote])

    sandbox = MagicMock()
    sandbox.exec = AsyncMock(side_effect=exec_)
    sandbox.download = AsyncMock(side_effect=download)
    config = TestOpenCodeSandboxedAgent()._create_config().model_copy(update={"opencode_results_dir": str(tmp_path)})
    server = OpenCodeSandboxedAgent(config=config, server_client=MagicMock(spec=ServerClient))
    monkeypatch.setattr(server, "_sandbox_id_to_sandbox", {"": sandbox})
    monkeypatch.setattr(server, "_create_opencode_config", AsyncMock(return_value=dict()))
    await server.responses(
        request=MagicMock(session={SESSION_ID_KEY: "s1"}, cookies={"sandbox_id": ""}, path_params={"rollout_id": "r"}),
        body=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hello"}]),
    )
    row = server._sandbox_id_to_run_result[""]
    assert row["opencode_subagent_sessions"] == ["ses_a", "ses_grand"]
    assert row["opencode_tokens"]["main"]["input"] == 100 and row["opencode_tokens"]["subagents"]["input"] == 43
    assert (tmp_path / "s1" / "export.json").exists()
    assert sorted(p.name for p in (tmp_path / "s1" / "children").iterdir()) == ["ses_a.json", "ses_grand.json"]


async def test_agent_phase_is_cut_off_at_sandbox_timeout(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """A run that never returns (provider timeouts not holding) is stopped at sandbox_timeout + wall_grace_s:
    OpenCode is killed in the sandbox, the session is still exported, and the row says opencode_wall_exceeded."""
    commands: List[str] = []
    written: Dict[str, str] = {}

    async def exec_(command: str, **kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        ok = lambda out="": SimpleNamespace(stdout=out, stderr="", return_code=0, error_type=None)  # noqa: E731
        if "opencode run" in command and "pkill" not in command:
            await asyncio.sleep(3600)  # hangs, as the 14 h rollouts did
        if "session list" in command:
            return ok('[{"id": "ses_main"}]')
        if "opencode export" in command:
            written[command.rsplit("> ", 1)[1].strip()] = json.dumps(_session("ses_main", [10], []))
        return ok()

    async def download(remote: str, local: Path) -> None:
        Path(local).write_text(written[remote])

    sandbox = MagicMock()
    sandbox.exec = AsyncMock(side_effect=exec_)
    sandbox.download = AsyncMock(side_effect=download)
    config = TestOpenCodeSandboxedAgent()._create_config().model_copy(
        update={"opencode_results_dir": str(tmp_path), "sandbox_timeout": 0.2, "wall_grace_s": 0.0}
    )
    server = OpenCodeSandboxedAgent(config=config, server_client=MagicMock(spec=ServerClient))
    monkeypatch.setattr(server, "_sandbox_id_to_sandbox", {"": sandbox})
    monkeypatch.setattr(server, "_create_opencode_config", AsyncMock(return_value=dict()))
    await asyncio.wait_for(
        server.responses(
            request=MagicMock(session={SESSION_ID_KEY: "s1"}, cookies={"sandbox_id": ""}, path_params={"rollout_id": "r"}),
            body=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hello"}]),
        ),
        timeout=10,
    )
    row = server._sandbox_id_to_run_result[""]
    assert row["opencode_wall_exceeded"] is True and row["opencode_finished"] is False
    assert any("pkill -KILL -f 'opencode run'" in c for c in commands)
    assert row["opencode_tokens"]["main"]["input"] == 10  # exported after the cut-off
