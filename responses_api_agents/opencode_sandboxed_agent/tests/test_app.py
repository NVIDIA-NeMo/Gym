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
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError
from pytest import MonkeyPatch, fixture, mark, raises

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
)
from nemo_gym.sandbox import SandboxHandle
from nemo_gym.sandbox.utils import CPU_CAP_ENV_VARS
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.opencode_sandboxed_agent import app as app_module
from responses_api_agents.opencode_sandboxed_agent.app import (
    _AGENT_HOME_RESET,
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
    _agent_home_prefix,
)


# The exact `session list` and `export` commands issued with no identity (image default); the identity-bearing
# variants must be these strings with `_AGENT_HOME_RESET` prepended and nothing else changed.
SESSION_LIST_COMMAND = "export PATH=$HOME/.opencode/bin:$PATH && opencode session list --format json"
EXPORT_COMMAND = "export PATH=$HOME/.opencode/bin:$PATH && opencode export session-id > /tmp/opencode_export.json"


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
        assert "sandbox_lifecycle_timing_unavailable" in {gap.code for gap in result.ng_agent_observations.gaps}
        assert "sandbox_cleanup_failed" not in {gap.code for gap in result.ng_agent_observations.gaps}
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


class RecordingSandbox:
    """Fake AsyncSandbox that records every ``exec(command, **kwargs)`` and serves scripted results in order.

    ``script`` maps a command marker to the result returned when the recorded command contains that marker; the
    identity check commands are served from ``check_results`` so a test can script a violation.
    """

    def __init__(self, *, root_uid: str = "0", agent_output: str = "1000\n1000", agent_return_code: int = 0):
        self.calls: list[tuple[str, Any]] = []
        self.kwargs: list[dict[str, Any]] = []
        self.download = AsyncMock()
        self.stop = AsyncMock()
        self._handle = SandboxHandle(sandbox_id="connected-sandbox", provider_name="opensandbox", raw=None)
        self._root_uid = root_uid
        self._agent_output = agent_output
        self._agent_return_code = agent_return_code

    async def exec(self, command: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((command, kwargs.get("user")))
        self.kwargs.append(kwargs)
        if command == "id -u":
            return SimpleNamespace(stdout=self._root_uid, stderr="", return_code=0, error_type=None)
        if command == "id -u && id -g":
            return SimpleNamespace(
                stdout=self._agent_output, stderr="", return_code=self._agent_return_code, error_type=None
            )
        if "opencode run" in command:
            return SimpleNamespace(
                stdout="Shell: /bin/bash\nOpenCode run finished", stderr="", return_code=0, error_type=None
            )
        if "opencode session list" in command:
            return SimpleNamespace(stdout='[{"id": "session-id"}]', stderr="", return_code=0, error_type=None)
        return SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None)


def _marker(command: str) -> str:
    """Reduce a recorded exec command to the marker the assertions compare against."""
    if command in ("id -u", "id -u && id -g"):
        return command
    if "opencode run" in command:
        return "install+run"
    if "opencode session list" in command:
        return "session list"
    if "opencode export" in command:
        return "export"
    if command.startswith("python3 -c "):
        return "snapshot"
    return command


class TestOpenCodeSandboxedAgentUser:
    @fixture
    def opencode_export_test_data(self) -> Dict[str, Any]:
        return json.loads((Path(__file__).parent / "opencode_export_test_data.json").read_text())

    def _create_config(self, **overrides: Any) -> OpenCodeSandboxedAgentConfig:
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
            **overrides,
        )

    @mark.parametrize(
        ("value", "expected"),
        [(None, None), ("agent", "agent"), ("1000", 1000), (1000, 1000), ("root", "root"), (0, 0)],
    )
    def test_config_normalizes_agent_user(self, value: Any, expected: Any) -> None:
        config = self._create_config(agent_user=value)
        assert config.agent_user == expected
        assert type(config.agent_user) is type(expected)

    def test_config_defaults_agent_user_to_none(self) -> None:
        assert self._create_config().agent_user is None

    @mark.parametrize("value", [True, False, "", "-m", 1.5])
    def test_config_rejects_malformed_agent_user(self, value: Any) -> None:
        with raises(ValidationError):
            self._create_config(agent_user=value)

    @mark.parametrize(
        ("agent_user", "expected"),
        [(None, ""), ("root", ""), (0, ""), ("agent", _AGENT_HOME_RESET), (1000, _AGENT_HOME_RESET)],
        ids=("none", "root", "uid-0", "name", "uid"),
    )
    def test_agent_home_prefix(self, agent_user: Any, expected: str) -> None:
        assert _agent_home_prefix(agent_user) == expected
        # POSIX sh, not bash: the provider's `su -s /bin/sh -c` and execd's uid path both run plain `sh`.
        assert "getent passwd" in _AGENT_HOME_RESET and _AGENT_HOME_RESET.endswith(" && ")
        assert "${ng_home:-$HOME}" in _AGENT_HOME_RESET, "an empty getent result must fall back to the current HOME"

    def _patch_responses_io(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, opencode_export_test_data: Dict[str, Any]
    ) -> None:
        monkeypatch.setattr("responses_api_agents.opencode_sandboxed_agent.app.__file__", str(tmp_path / "app.py"))
        monkeypatch.setattr("responses_api_agents.opencode_sandboxed_agent.app.Path.exists", lambda self: True)
        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.Path.read_text",
            lambda self: json.dumps(opencode_export_test_data),
        )
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", MagicMock(return_value=MagicMock(hex="")))

    def _request(self, *, observations: bool) -> MagicMock:
        state = SimpleNamespace(_ng_observation_invocation_id="rollout-1") if observations else SimpleNamespace()
        return MagicMock(
            session={SESSION_ID_KEY: "my session"},
            cookies={"sandbox_id": "session-key"},
            path_params={"rollout_id": "direct-call"},
            state=state,
        )

    @mark.parametrize("observations", [False, True], ids=("no-observations", "observations"))
    async def test_responses_runs_check_first_then_every_exec_as_agent_user(
        self,
        tmp_path: Path,
        opencode_export_test_data: Dict[str, Any],
        monkeypatch: MonkeyPatch,
        observations: bool,
    ) -> None:
        self._patch_responses_io(monkeypatch, tmp_path, opencode_export_test_data)
        server = OpenCodeSandboxedAgent(config=self._create_config(), server_client=MagicMock(spec=ServerClient))
        server._create_opencode_config = AsyncMock(return_value={})
        sandbox = RecordingSandbox()
        server._sandbox_id_to_sandbox["session-key"] = sandbox
        server._sandbox_id_to_agent_user["session-key"] = "agent"

        response = await server.responses(
            request=self._request(observations=observations),
            body=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hello"}]),
        )

        expected = [
            ("id -u", "root"),
            ("id -u && id -g", "agent"),
            ("install+run", "agent"),
            ("session list", "agent"),
            ("export", "agent"),
        ]
        if observations:
            expected.append(("snapshot", "agent"))
        assert [(_marker(command), user) for command, user in sandbox.calls] == expected
        # The check ran before anything model-controlled: the install+run exec is strictly after both id probes.
        assert _marker(sandbox.calls[2][0]) == "install+run"
        assert "opencode run" in sandbox.calls[2][0] and "$HOME/.opencode/bin" in sandbox.calls[2][0]
        assert sandbox.kwargs[2]["timeout_s"] == server.config.sandbox_timeout
        # Under a non-root identity the three opencode commands pin $HOME first (the uid path skips `su`, so it
        # would otherwise inherit /root); the snapshot addresses the store by absolute path and needs no reset.
        assert sandbox.calls[2][0].startswith(_AGENT_HOME_RESET)
        assert sandbox.calls[3][0] == _AGENT_HOME_RESET + SESSION_LIST_COMMAND
        assert sandbox.calls[4][0] == _AGENT_HOME_RESET + EXPORT_COMMAND
        if observations:
            assert "getent" not in sandbox.calls[5][0]
        assert isinstance(response, NeMoGymResponse)
        assert server._sandbox_id_to_run_result["session-key"]["opencode_finished"] is True

    @mark.parametrize("observations", [False, True], ids=("no-observations", "observations"))
    async def test_responses_without_identity_runs_no_check_and_every_exec_as_image_default(
        self,
        tmp_path: Path,
        opencode_export_test_data: Dict[str, Any],
        monkeypatch: MonkeyPatch,
        observations: bool,
    ) -> None:
        self._patch_responses_io(monkeypatch, tmp_path, opencode_export_test_data)
        server = OpenCodeSandboxedAgent(
            config=self._create_config(agent_user=None), server_client=MagicMock(spec=ServerClient)
        )
        server._create_opencode_config = AsyncMock(return_value={})
        sandbox = RecordingSandbox()
        server._sandbox_id_to_sandbox["session-key"] = sandbox
        # No per-session entry and a lane default of None: the lookup must fall back to "image default", not KeyError.

        await server.responses(
            request=self._request(observations=observations),
            body=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hello"}]),
        )

        expected = [("install+run", None), ("session list", None), ("export", None)]
        if observations:
            expected.append(("snapshot", None))
        assert [(_marker(command), user) for command, user in sandbox.calls] == expected
        assert all(kwargs["user"] is None for kwargs in sandbox.kwargs)
        # Byte-for-byte the pre-`agent_user` commands: no HOME reset anywhere without an identity.
        assert sandbox.calls[0][0].startswith('\n        echo "Shell: $SHELL"')
        assert sandbox.calls[1][0] == SESSION_LIST_COMMAND
        assert sandbox.calls[2][0] == EXPORT_COMMAND
        assert not any("getent" in command for command, _ in sandbox.calls)

    @mark.parametrize("observations", [False, True], ids=("no-observations", "observations"))
    @mark.parametrize("agent_user", ["agent", 1000], ids=("name", "uid"))
    async def test_responses_identity_only_prepends_home_reset_to_opencode_commands(
        self,
        tmp_path: Path,
        opencode_export_test_data: Dict[str, Any],
        monkeypatch: MonkeyPatch,
        observations: bool,
        agent_user: Any,
    ) -> None:
        """The identity-bearing commands are exactly the image-default commands with the HOME reset prepended."""
        self._patch_responses_io(monkeypatch, tmp_path, opencode_export_test_data)
        # Pin the per-rollout XDG_DATA_HOME so the two runs compose the same install+run command.
        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.uuid4", MagicMock(return_value=MagicMock(hex="fixed"))
        )
        commands: dict[Any, list[str]] = {}
        for identity in (None, agent_user):
            server = OpenCodeSandboxedAgent(
                config=self._create_config(agent_user=None), server_client=MagicMock(spec=ServerClient)
            )
            server._create_opencode_config = AsyncMock(return_value={})
            sandbox = RecordingSandbox()
            server._sandbox_id_to_sandbox["session-key"] = sandbox
            server._sandbox_id_to_agent_user["session-key"] = identity
            await server.responses(
                request=self._request(observations=observations),
                body=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hello"}]),
            )
            # Drop the two identity-check probes so both lists line up as install+run, session list, export[, snapshot].
            commands[identity] = [command for command, _ in sandbox.calls if not command.startswith("id -u")]

        baseline, with_identity = commands[None], commands[agent_user]
        assert len(baseline) == len(with_identity) == (4 if observations else 3)
        assert with_identity[0] == _AGENT_HOME_RESET + baseline[0]
        assert with_identity[1] == _AGENT_HOME_RESET + baseline[1]
        assert with_identity[2] == _AGENT_HOME_RESET + baseline[2]
        if observations:
            assert with_identity[3] == baseline[3]

    async def test_responses_without_session_entry_falls_back_to_lane_agent_user(
        self, tmp_path: Path, opencode_export_test_data: Dict[str, Any], monkeypatch: MonkeyPatch
    ) -> None:
        self._patch_responses_io(monkeypatch, tmp_path, opencode_export_test_data)
        # No per-session entry (responses() reached without run()), lane default "agent": the lane default must
        # not be lost, so the check runs and every exec carries "agent".
        server = OpenCodeSandboxedAgent(
            config=self._create_config(agent_user="agent"), server_client=MagicMock(spec=ServerClient)
        )
        server._create_opencode_config = AsyncMock(return_value={})
        sandbox = RecordingSandbox()
        server._sandbox_id_to_sandbox["session-key"] = sandbox
        assert "session-key" not in server._sandbox_id_to_agent_user

        await server.responses(
            request=self._request(observations=False),
            body=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hello"}]),
        )

        assert [(_marker(command), user) for command, user in sandbox.calls] == [
            ("id -u", "root"),
            ("id -u && id -g", "agent"),
            ("install+run", "agent"),
            ("session list", "agent"),
            ("export", "agent"),
        ]
        assert sandbox.calls[3][0] == _AGENT_HOME_RESET + SESSION_LIST_COMMAND

    async def test_responses_row_root_skips_check_and_passes_root_through(
        self, tmp_path: Path, opencode_export_test_data: Dict[str, Any], monkeypatch: MonkeyPatch
    ) -> None:
        self._patch_responses_io(monkeypatch, tmp_path, opencode_export_test_data)
        # Lane says "agent"; the per-session (row) identity says "root": the escape hatch wins and no check runs.
        server = OpenCodeSandboxedAgent(
            config=self._create_config(agent_user="agent"), server_client=MagicMock(spec=ServerClient)
        )
        server._create_opencode_config = AsyncMock(return_value={})
        sandbox = RecordingSandbox()
        server._sandbox_id_to_sandbox["session-key"] = sandbox
        server._sandbox_id_to_agent_user["session-key"] = "root"

        await server.responses(
            request=self._request(observations=False),
            body=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hello"}]),
        )

        assert [(_marker(command), user) for command, user in sandbox.calls] == [
            ("install+run", "root"),
            ("session list", "root"),
            ("export", "root"),
        ]

    @mark.parametrize(
        ("sandbox_kwargs", "condition"),
        [
            ({"agent_output": "0\n0"}, "still resolve to uid 0"),
            ({"agent_output": "", "agent_return_code": 1}, "could not run a command as 'agent'"),
            ({"root_uid": "1000"}, "image default user must be root"),
        ],
        ids=("agent-uid-0", "agent-exec-rc-1", "image-default-not-root"),
    )
    async def test_responses_fails_closed_before_install(
        self,
        tmp_path: Path,
        opencode_export_test_data: Dict[str, Any],
        monkeypatch: MonkeyPatch,
        sandbox_kwargs: Dict[str, Any],
        condition: str,
    ) -> None:
        self._patch_responses_io(monkeypatch, tmp_path, opencode_export_test_data)
        # The failing check must stop everything before the install command is built or run.
        server = OpenCodeSandboxedAgent(
            config=self._create_config(),
            server_client=MagicMock(spec=ServerClient),
        )
        server._create_opencode_config = AsyncMock(return_value={})
        sandbox = RecordingSandbox(**sandbox_kwargs)
        server._sandbox_id_to_sandbox["session-key"] = sandbox
        server._sandbox_id_to_agent_user["session-key"] = "agent"

        with raises(RuntimeError, match=condition) as excinfo:
            await server.responses(
                request=self._request(observations=False),
                body=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hello"}]),
            )

        assert "agent_user='agent'" in str(excinfo.value)
        markers = [_marker(command) for command, _ in sandbox.calls]
        assert "install+run" not in markers and "session list" not in markers and "export" not in markers
        assert markers == (["id -u"] if "root_uid" in sandbox_kwargs else ["id -u", "id -u && id -g"])
        server._create_opencode_config.assert_not_awaited()
        assert "session-key" not in server._sandbox_id_to_run_result

    def _run_harness(
        self, monkeypatch: MonkeyPatch, *, lane_agent_user: Any, seed_payload: Dict[str, Any]
    ) -> tuple[Any, ...]:
        """Wire a server whose ``responses`` records the identity it sees and whose ``/verify`` post is captured."""

        class Response:
            ok = True

            def __init__(self, payload: dict[str, Any]):
                self.payload = payload
                self.cookies: dict[str, str] = {}

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

        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {
            "observability_enabled": False,
            "token_id_capture": {"enabled": False, "all_agents": False},
        }
        server = OpenCodeSandboxedAgent(
            config=self._create_config(agent_user=lane_agent_user), server_client=server_client
        )
        sandbox = RecordingSandbox()
        server._start_sandbox = AsyncMock(return_value=sandbox)
        seen: dict[str, Any] = {}
        posted: dict[str, Any] = {}

        async def fake_responses(agent: OpenCodeSandboxedAgent, request: Any, body: Any) -> NeMoGymResponse:
            assert agent is server
            session_key = request.cookies["sandbox_id"]
            seen["agent_user"] = server._sandbox_id_to_agent_user[session_key]
            seen["sandbox"] = server._sandbox_id_to_sandbox[session_key]
            server._sandbox_id_to_run_result[session_key] = {
                "opencode_results_fpath": "",
                "opencode_run_stdout": "",
                "opencode_run_stderr": "",
                "opencode_export_found": False,
                "opencode_finished": True,
            }
            return NeMoGymResponse(
                id="resp_x",
                created_at=0,
                model="",
                object="response",
                output=[],
                parallel_tool_calls=True,
                tool_choice="auto",
                tools=[],
            )

        monkeypatch.setattr(OpenCodeSandboxedAgent, "responses", fake_responses)

        async def post(server_name, url_path, json=None, cookies=None):
            if url_path == "/seed_session":
                return Response(seed_payload)
            assert url_path == "/verify"
            posted.update(json)
            return Response(json | {"reward": 1.0})

        server_client.post = AsyncMock(side_effect=post)
        body = OpenCodeSandboxedAgentRunRequest.model_validate(
            {"responses_create_params": {"input": [{"role": "user", "content": "solve"}]}}
        )
        return server, sandbox, RunRequest(), body, seen, posted

    @mark.parametrize(
        ("lane_agent_user", "seed_payload", "expected"),
        [
            (None, {"sandbox_handle": "seed-sandbox", "agent_user": "agent"}, "agent"),
            (None, {"sandbox_handle": "seed-sandbox", "agent_user": "1000"}, 1000),
            ("agent", {"sandbox_handle": "seed-sandbox"}, "agent"),
            ("agent", {"sandbox_handle": "seed-sandbox", "agent_user": None}, "agent"),
            ("agent", {"sandbox_handle": "seed-sandbox", "agent_user": "root"}, "root"),
            ("agent", {"sandbox_handle": "seed-sandbox", "agent_user": 0}, 0),
            (None, {"sandbox_handle": "seed-sandbox"}, None),
        ],
        ids=(
            "row-wins",
            "row-digit-string-normalized",
            "lane-fallback",
            "row-null-lane",
            "row-root-beats-lane",
            "row-0-beats-lane",
            "neither",
        ),
    )
    async def test_run_precedence_and_verify_payload(
        self, monkeypatch: MonkeyPatch, lane_agent_user: Any, seed_payload: Dict[str, Any], expected: Any
    ) -> None:
        server, sandbox, request, body, seen, posted = self._run_harness(
            monkeypatch, lane_agent_user=lane_agent_user, seed_payload=seed_payload
        )

        result = await server.run(request, body)

        assert seen["agent_user"] == expected
        assert type(seen["agent_user"]) is type(expected)
        assert seen["sandbox"] is sandbox
        assert "agent_user" in posted and posted["agent_user"] == expected
        assert posted["response"]["id"] == "resp_x"
        assert result.reward == 1.0
        sandbox.stop.assert_awaited_once()
        assert server._sandbox_id_to_sandbox == {}
        assert server._sandbox_id_to_agent_user == {}
        assert server._sandbox_id_to_run_result == {}

    async def test_run_rejects_malformed_row_echo_and_stops_sandbox(self, monkeypatch: MonkeyPatch) -> None:
        server, sandbox, request, body, seen, posted = self._run_harness(
            monkeypatch, lane_agent_user=None, seed_payload={"sandbox_handle": "seed-sandbox", "agent_user": True}
        )

        with raises(ValueError, match="agent_user must be an account name"):
            await server.run(request, body)

        assert seen == {} and posted == {}
        sandbox.stop.assert_awaited_once()
        assert server._sandbox_id_to_sandbox == {}
        assert server._sandbox_id_to_agent_user == {}

    @mark.parametrize("stop_raises", [False, True], ids=("stop-ok", "stop-raises"))
    async def test_run_cleans_up_when_responses_raises(self, monkeypatch: MonkeyPatch, stop_raises: bool) -> None:
        server, sandbox, request, body, seen, posted = self._run_harness(
            monkeypatch, lane_agent_user="agent", seed_payload={"sandbox_handle": "seed-sandbox"}
        )
        if stop_raises:
            sandbox.stop = AsyncMock(side_effect=ConnectionError("provider gone"))

        async def failing_responses(agent: OpenCodeSandboxedAgent, request: Any, body: Any) -> NeMoGymResponse:
            raise RuntimeError("agent_user='agent' identity check failed")

        monkeypatch.setattr(OpenCodeSandboxedAgent, "responses", failing_responses)

        with raises(RuntimeError, match="identity check failed"):
            await server.run(request, body)

        sandbox.stop.assert_awaited_once()
        assert server._sandbox_id_to_sandbox == {}
        assert server._sandbox_id_to_agent_user == {}
        assert server._sandbox_id_to_run_result == {}
        assert posted == {}, "/verify must not be posted when the agent phase failed"
        assert not hasattr(request.state, "_ng_observation_invocation_id")
