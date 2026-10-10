# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shlex
import subprocess
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.opencode_sandboxed_agent import app as app_module
from responses_api_agents.opencode_sandboxed_agent.agent_user import (
    agent_home_prefix,
    check_agent_user,
    is_root_agent_user,
    normalize_agent_user,
)
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
)


# Quotes, a backslash, $, backticks, a command substitution, a tab and newlines: any shell or CLI quoting changes it.
TASK = (
    'Write "42" to /app/answer.txt.\n\nQuoting check: \'single\', a backslash \\ and \\"escaped\\", $HOME, `id`,'
    " $(echo not-run), a\ttab and 100% literal.\n"
)


def config(**overrides: Any) -> OpenCodeSandboxedAgentConfig:
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
        **overrides,
    )


class RecordingSandbox:
    """Answers each command the way a root-default image with an `agent` account (uid 1000) would."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def exec(self, command: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((command, kwargs.get("user")))
        if command == "id -u":
            stdout = "0\n"
        elif command == "id -u && id -g":
            stdout = "1000\n1000\n"
        elif "opencode session list" in command:
            stdout = '[{"id": "ses_1"}]'
        elif "opencode run" in command:
            stdout = "Shell: /bin/sh\nOpenCode run finished"
        else:
            stdout = ""
        return SimpleNamespace(stdout=stdout, stderr="", return_code=0, error_type=None)

    async def upload(self, *args: Any) -> None:
        pass

    async def download(self, *args: Any) -> None:
        raise FileNotFoundError("no export in this test")


async def run_responses(server: OpenCodeSandboxedAgent, sandbox: RecordingSandbox, tmp_path, task: str = TASK):
    server._sandbox_id_to_sandbox["session-key"] = sandbox
    server._create_opencode_config = AsyncMock(return_value={})
    server.config.artifacts_dir = str(tmp_path)
    request = MagicMock(session={SESSION_ID_KEY: "s"}, cookies={"sandbox_id": "session-key"}, state=SimpleNamespace())
    await server.responses(
        request=request, body=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": task}])
    )


def command_of(sandbox: RecordingSandbox, needle: str) -> tuple[str, Any]:
    [match] = [(command, user) for command, user in sandbox.calls if needle in command]
    return match


class TestTaskOnStandardInput:
    async def test_the_task_reaches_opencode_byte_for_byte(self, tmp_path) -> None:
        server = OpenCodeSandboxedAgent(config=config(), server_client=MagicMock(spec=ServerClient))
        sandbox = RecordingSandbox()
        await run_responses(server, sandbox, tmp_path)

        run_command, _ = command_of(sandbox, "opencode run")
        piped = f"printf '%s' {shlex.quote(TASK)} | "
        assert piped in run_command
        # No positional message: OpenCode would wrap it in quotes (cli/cmd/run.ts) before the model sees it.
        assert shlex.quote(TASK) not in run_command.split("opencode run", 1)[1]
        delivered = subprocess.run(["sh", "-c", piped + "cat"], capture_output=True, check=True).stdout.decode()
        assert delivered == TASK


class TestAgentIdentity:
    @pytest.mark.parametrize("agent_user", [None, "root", 0])
    async def test_the_image_default_needs_no_check_and_no_home_reset(self, tmp_path, agent_user) -> None:
        server = OpenCodeSandboxedAgent(
            config=config(agent_user=agent_user), server_client=MagicMock(spec=ServerClient)
        )
        sandbox = RecordingSandbox()
        await run_responses(server, sandbox, tmp_path)

        assert not any(command in ("id -u", "id -u && id -g") for command, _ in sandbox.calls)
        run_command, run_user = command_of(sandbox, "opencode run")
        assert run_user == agent_user and "getent passwd" not in run_command

    async def test_a_non_root_identity_is_checked_first_and_runs_every_opencode_command(self, tmp_path) -> None:
        server = OpenCodeSandboxedAgent(config=config(agent_user="agent"), server_client=MagicMock(spec=ServerClient))
        sandbox = RecordingSandbox()
        await run_responses(server, sandbox, tmp_path)

        assert sandbox.calls[:2] == [("id -u", "root"), ("id -u && id -g", "agent")]
        for needle in ("opencode run", "opencode session list", "opencode export", "opencode db"):
            command, user = command_of(sandbox, needle)
            assert user == "agent", needle
            assert command.lstrip().startswith(agent_home_prefix("agent")), needle

    async def test_a_missing_account_stops_before_the_install(self, tmp_path) -> None:
        server = OpenCodeSandboxedAgent(config=config(agent_user="ghost"), server_client=MagicMock(spec=ServerClient))
        sandbox = RecordingSandbox()

        async def no_such_account(command: str, **kwargs: Any) -> SimpleNamespace:
            sandbox.calls.append((command, kwargs.get("user")))
            failed = command == "id -u && id -g"
            return SimpleNamespace(
                stdout="" if failed else "0\n",
                stderr="su: user ghost does not exist",
                return_code=int(failed),
                error_type=None,
            )

        sandbox.exec = no_such_account
        with pytest.raises(RuntimeError, match="does the account exist"):
            await run_responses(server, sandbox, tmp_path)
        assert not any("opencode" in command for command, _ in sandbox.calls)

    async def test_the_row_identity_wins_and_reaches_the_verifier(self, monkeypatch: MonkeyPatch) -> None:
        class Seed:
            ok = True
            cookies: dict[str, str] = {}

            async def json(self) -> dict[str, Any]:
                return {"sandbox_handle": "sb", "agent_user": "1000"}

        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=Seed())
        server = OpenCodeSandboxedAgent(config=config(agent_user="agent"), server_client=server_client)
        sandbox = MagicMock()
        sandbox.stop = AsyncMock()
        server._start_sandbox = AsyncMock(return_value=sandbox)
        seen: dict[str, Any] = {}

        async def responses(self, request, params):
            seen["identity"] = self._sandbox_id_to_agent_user["session-1"]
            self._sandbox_id_to_run_result["session-1"] = {
                "opencode_results_fpath": "",
                "opencode_run_stdout": "",
                "opencode_run_stderr": "",
                "opencode_finished": True,
                "opencode_export_found": False,
            }
            return NeMoGymResponse(
                id="r",
                created_at=0,
                model="m",
                object="response",
                output=[],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=False,
            )

        async def verify(client, server_ref, body, response, cookies, *, force_zero_reward):
            seen["verified_as"] = body.model_dump().get("agent_user")
            return {
                "reward": 1.0,
                "response": response.model_dump(mode="json"),
                "responses_create_params": body.responses_create_params.model_dump(mode="json"),
            }

        monkeypatch.setattr(OpenCodeSandboxedAgent, "responses", responses)
        monkeypatch.setattr(app_module, "verify_agent_response", verify)
        monkeypatch.setattr(OpenCodeSandboxedAgent, "_seed_tool_servers", AsyncMock(return_value={}))
        request = SimpleNamespace(cookies={}, session={SESSION_ID_KEY: "session-1"}, state=SimpleNamespace())
        body = OpenCodeSandboxedAgentRunRequest.model_validate(
            {"responses_create_params": {"input": [{"role": "user", "content": "solve"}]}}
        )
        result = await server.run(request, body)

        assert result.reward == 1.0
        assert seen == {"identity": 1000, "verified_as": 1000}
        assert server._sandbox_id_to_agent_user == {}
        sandbox.stop.assert_awaited_once()


class TestIdentityHelpers:
    @pytest.mark.parametrize(
        ("value", "expected"), [("1000", 1000), ("0", 0), ("agent", "agent"), (1000, 1000), (None, None), ("²", "²")]
    )
    def test_normalize(self, value, expected) -> None:
        assert normalize_agent_user(value) == expected

    @pytest.mark.parametrize("value", [True, False, "", "-m", "--login"])
    def test_rejected(self, value) -> None:
        with pytest.raises(ValueError, match="agent_user must be"):
            normalize_agent_user(value)

    def test_root_values(self) -> None:
        assert all(is_root_agent_user(value) for value in (None, "root", 0))
        assert not any(is_root_agent_user(value) for value in ("agent", 1000))

    async def test_check_refuses_an_identity_that_is_still_root(self) -> None:
        executor = MagicMock()
        executor.exec = AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="0\n", stderr=""),
                SimpleNamespace(return_code=0, stdout="0\n0\n", stderr=""),
            ]
        )
        with pytest.raises(RuntimeError, match="still resolve to uid 0"):
            await check_agent_user(executor, "agent")
