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
import os
import subprocess
from pathlib import Path
from shlex import quote
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

from pytest import MonkeyPatch, fixture

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
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    _build_remote_opencode_install_command,
    _extract_opencode_session_id,
)


def _run_remote_install_command(
    tmp_path: Path, ldd_output: str, *, include_musl_binary: bool = True
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ldd = bin_dir / "ldd"
    ldd.write_text(f"#!/bin/sh\nprintf '%s\\n' {ldd_output!r}\n")
    ldd.chmod(0o755)

    binary = tmp_path / "opencode binary"
    binary.write_text("#!/bin/sh\necho glibc\n")
    musl_binary = tmp_path / "opencode musl binary"
    musl_binary.write_text("#!/bin/sh\necho musl\n")
    installer = Path(__file__).parent.parent / "install_cached_opencode.sh"

    if include_musl_binary:
        command = _build_remote_opencode_install_command(
            install_script_path=str(installer),
            binary_path=str(binary),
            musl_binary_path=str(musl_binary),
        )
    else:
        command = f"bash {quote(str(installer))} --glibc-binary {quote(str(binary))}"
    return subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=tmp_path,
        env=os.environ
        | {
            "HOME": str(tmp_path),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_remote_opencode_install_command_uses_mounted_binary_on_glibc(tmp_path: Path) -> None:
    result = _run_remote_install_command(tmp_path, "ldd (GNU libc) 2.40")

    assert result.returncode == 0, result.stderr
    assert "S3-cached OpenCode glibc binary" in result.stdout
    installed = tmp_path / ".opencode" / "bin" / "opencode"
    assert subprocess.run([installed], capture_output=True, text=True, check=True).stdout.strip() == "glibc"


def test_remote_opencode_install_command_uses_mounted_musl_binary_on_musl(tmp_path: Path) -> None:
    result = _run_remote_install_command(tmp_path, "musl libc (x86_64)")

    assert result.returncode == 0, result.stderr
    assert "S3-cached OpenCode musl binary" in result.stdout
    installed = tmp_path / ".opencode" / "bin" / "opencode"
    assert subprocess.run([installed], capture_output=True, text=True, check=True).stdout.strip() == "musl"


def test_cached_opencode_installer_allows_omitted_musl_binary_on_glibc(tmp_path: Path) -> None:
    result = _run_remote_install_command(tmp_path, "ldd (GNU libc) 2.40", include_musl_binary=False)

    assert result.returncode == 0, result.stderr
    assert "S3-cached OpenCode glibc binary" in result.stdout


def test_cached_opencode_installer_requires_musl_binary_on_musl(tmp_path: Path) -> None:
    result = _run_remote_install_command(tmp_path, "musl libc (x86_64)", include_musl_binary=False)

    assert result.returncode == 2
    assert "--musl-binary is required on musl systems" in result.stderr


def test_extract_opencode_session_id_uses_first_session() -> None:
    assert _extract_opencode_session_id('[{"id": "session-new"}, {"id": "session-old"}]') == "session-new"


class TestOpenCodeSandboxedAgent:
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

        sandbox_mock = AsyncMock()
        monkeypatch.setattr(server, "_sandbox_id_to_sandbox", {"": sandbox_mock})
        monkeypatch.setattr(server, "_create_opencode_config", AsyncMock(return_value={}))

        sandbox_mock.exec.side_effect = [
            MagicMock(stdout="OpenCode run finished", stderr="", return_code=0),
            MagicMock(stdout='[{"id": "session-id"}]', stderr="", return_code=0),
            MagicMock(stdout="", stderr="", return_code=0),
        ]

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
            request=MagicMock(session={SESSION_ID_KEY: "my session"}, cookies={"sandbox_id": ""}),
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
        export_command = sandbox_mock.exec.await_args_list[2].kwargs["command"]
        # Must land outside the sandbox workdir, which is the git repo for SWE-bench-style
        # environments: a transcript written there is swept into the extracted model patch.
        assert "opencode export session-id > /tmp/opencode_export.json" in export_command
        assert "> export.json" not in export_command
        assert "apt-get" not in export_command
        assert "jq" not in export_command
        assert sandbox_mock.download.await_args.args[0] == "/tmp/opencode_export.json"
