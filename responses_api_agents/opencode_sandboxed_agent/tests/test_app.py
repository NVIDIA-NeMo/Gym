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
    SandboxObservation,
    ToolCallObservation,
)
from nemo_gym.sandbox import SandboxHandle
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
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
                SimpleNamespace(stdout="OpenCode run finished", stderr="", return_code=0, error_type=None),
                SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None),
                SimpleNamespace(stdout="my dir"),
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
        assert "XDG_DATA_HOME" not in sandbox_mock.exec.await_args_list[0].kwargs["env"]
        assert sandbox_mock.exec.await_args_list[1].kwargs["env"] is None

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
                SimpleNamespace(stdout="OpenCode run finished", stderr="", return_code=0, error_type=None),
                SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None),
                SimpleNamespace(stdout="/workspace\n"),
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
            if remote_path == "/workspace/export.json":
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
        run_env = sandbox.exec.await_args_list[0].kwargs["env"]
        export_env = sandbox.exec.await_args_list[1].kwargs["env"]
        assert run_env["XDG_DATA_HOME"].startswith("/tmp/nemo-gym-opencode-")
        assert export_env["XDG_DATA_HOME"] == run_env["XDG_DATA_HOME"]
        assert not hasattr(request.state, "_ng_observation_invocation_id")
        assert server._sandbox_id_to_run_result == {}
        assert not (tmp_path / "results" / "session-1" / "opencode.db").exists()
