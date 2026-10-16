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
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import yaml
from pytest import MonkeyPatch, fixture, mark, raises

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.failure_kinds import AGENT_RUN_ERROR
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.rollout_observability import AgentInvocation, SandboxObservation
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.claude_code_sandboxed_agent import app as app_module
from responses_api_agents.claude_code_sandboxed_agent.app import (
    FINISHED_MARKER,
    ClaudeCodeRunPaths,
    ClaudeCodeSandboxedAgent,
    ClaudeCodeSandboxedAgentConfig,
    ClaudeCodeSandboxedAgentRunRequest,
    context_overflow,
    invocation_outcome,
    parse_claude_code_stream,
)


AGENT_DIR = Path(app_module.__file__).parent
VERSION = "2.1.287"
QUERY = "Fix the bug.\nIt's in `parse()`: don't use $(rm -rf /) or \"quotes\" \\ wrongly."
BODY = {
    "responses_create_params": {"input": [{"role": "user", "content": QUERY}]},
    "_ng_task_index": 7,
    "_ng_rollout_index": 2,
}
CAPTURE_CONFIG = {"token_id_capture": {"enabled": True, "all_agents": False}}
UNKNOWN_TOOL = (
    "<tool_use_error>Error: No such tool available: read. Tool names are case-sensitive: call Read instead."
    "</tool_use_error>"
)


def _config(**overrides: Any) -> ClaudeCodeSandboxedAgentConfig:
    values: Dict[str, Any] = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="claude_code_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="task_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        claude_code_version=VERSION,
        sandbox_provider="sandbox",
        sandbox_config={},
        sandbox_timeout=600,
        token_id_capture=True,
    )
    values.update(overrides)
    return ClaudeCodeSandboxedAgentConfig(**values)


def _agent(global_config: Dict[str, Any] | None = None, **overrides: Any) -> ClaudeCodeSandboxedAgent:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = global_config or {}
    return ClaudeCodeSandboxedAgent(config=_config(**overrides), server_client=server_client)


def _event(event_type: str, **fields: Any) -> Dict[str, Any]:
    return {"type": event_type, "session_id": "s1", "parent_tool_use_id": None, **fields}


def _assistant(message_id: str, block: Dict[str, Any], usage: Dict[str, int], **fields: Any) -> Dict[str, Any]:
    message = {"id": message_id, "model": "policy", "role": "assistant", "content": [block], "usage": usage}
    return _event("assistant", message=message, **fields)


def _tool_result(call_id: str, content: Any, **fields: Any) -> Dict[str, Any]:
    is_error = fields.pop("is_error", None)
    block = {"type": "tool_result", "tool_use_id": call_id, "content": content}
    if is_error is not None:
        block["is_error"] = is_error
    return _event("user", message={"role": "user", "content": [block]}, **fields)


MSG1_USAGE = {"input_tokens": 10, "output_tokens": 5}
STREAM_EVENTS: List[Dict[str, Any]] = [
    _event("system", subtype="init", model="policy", tools=["Bash", "Task"]),
    _assistant("msg_1", {"type": "thinking", "thinking": "I should look.", "signature": ""}, MSG1_USAGE),
    _assistant("msg_1", {"type": "text", "text": "\n\nLet me check. "}, MSG1_USAGE),
    _assistant("msg_1", {"type": "tool_use", "id": "call_1", "name": "Bash", "input": {"command": "ls"}}, MSG1_USAGE),
    _tool_result("call_1", "a.py\nb.py"),
    _assistant(
        "msg_2",
        {"type": "tool_use", "id": "call_2", "name": "Task", "input": {"prompt": "explore", "description": "x"}},
        {"input_tokens": 20, "output_tokens": 3},
    ),
    _assistant(
        "msg_sub",
        {"type": "text", "text": "subagent text"},
        {"input_tokens": 7, "output_tokens": 1},
        parent_tool_use_id="call_2",
    ),
    _tool_result("call_sub", "subagent tool output", parent_tool_use_id="call_2"),
    _tool_result("call_2", [{"type": "text", "text": "sub done"}, {"type": "text", "text": "really"}]),
    _event("system", subtype="status", status="compacting"),
    _event("system", subtype="status", status=None, compact_result="success"),
    _event(
        "assistant",
        message={
            "id": "msg_synthetic",
            "model": "<synthetic>",
            "role": "assistant",
            "content": [{"type": "text", "text": "API Error: boom"}],
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    ),
    _assistant("msg_3", {"type": "text", "text": "Fixed."}, {"input_tokens": 30, "output_tokens": 2}),
]
RESULT_EVENT = _event(
    "result",
    subtype="success",
    is_error=False,
    duration_ms=1234,
    num_turns=3,
    result="Fixed.",
    usage={"input_tokens": 100, "cache_read_input_tokens": 50, "cache_creation_input_tokens": 10, "output_tokens": 20},
)
OVERFLOW_EVENT = _event("result", subtype="success", is_error=True, duration_ms=99000, result="Prompt is too long")


def _stream_text(events: List[Dict[str, Any]], *extra_lines: str) -> str:
    return "\n".join([json.dumps(event) for event in events] + list(extra_lines)) + "\n"


EXPECTED_OUTPUT = [
    NeMoGymResponseReasoningItem(
        id="rs_x", summary=[NeMoGymSummary(text="I should look.", type="summary_text")], type="reasoning"
    ),
    NeMoGymResponseOutputMessage(
        id="msg_x",
        content=[NeMoGymResponseOutputText(text="\n\nLet me check. ", annotations=[], type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    ),
    NeMoGymResponseFunctionToolCall(
        arguments='{"command": "ls"}', call_id="call_1", name="Bash", type="function_call", status="completed"
    ),
    NeMoGymFunctionCallOutput(call_id="call_1", output="a.py\nb.py", type="function_call_output"),
    NeMoGymResponseFunctionToolCall(
        arguments='{"prompt": "explore", "description": "x"}',
        call_id="call_2",
        name="Task",
        type="function_call",
        status="completed",
    ),
    NeMoGymFunctionCallOutput(call_id="call_2", output="sub done\nreally", type="function_call_output"),
    NeMoGymResponseOutputMessage(
        id="msg_x",
        content=[NeMoGymResponseOutputText(text="Fixed.", annotations=[], type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    ),
]


@fixture
def fixed_uuid(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "uuid4", lambda: SimpleNamespace(hex="x"))


@fixture
def model_server_url(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "sandbox_server_url", lambda _name, require_reachable=False: "http://model-server")


class TestParseStream:
    def test_main_conversation_only_and_session_usage(self, fixed_uuid: None) -> None:
        stream = parse_claude_code_stream(_stream_text([*STREAM_EVENTS, RESULT_EVENT], "not json").splitlines())

        assert stream.output == EXPECTED_OUTPUT
        assert stream.usage == NeMoGymResponseUsage(
            input_tokens=160,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=50),
            output_tokens=20,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=180,
        )
        assert stream.result is not None and stream.result["subtype"] == "success"
        assert stream.compaction_attempts == []
        assert stream.compaction_boundaries == 0
        assert stream.invalid_lines == 1
        # msg_1, msg_2 and msg_3 are the main conversation; the synthetic message is nobody's turn.
        assert (stream.main_turns, stream.subagent_turns, stream.subagent_calls) == (3, 1, 1)
        assert (stream.tool_calls, stream.tool_errors, stream.unknown_tool_calls) == (2, 0, 0)

    def test_without_result_counts_each_message_once(self, fixed_uuid: None) -> None:
        stream = parse_claude_code_stream(_stream_text(STREAM_EVENTS).splitlines())

        assert stream.result is None
        assert stream.output == EXPECTED_OUTPUT
        # msg_1 appears in three events but is counted once; the synthetic message not at all.
        assert stream.usage is not None
        assert (stream.usage.input_tokens, stream.usage.output_tokens) == (10 + 20 + 7 + 30, 5 + 3 + 1 + 2)

    def test_compaction_is_reported(self) -> None:
        events = [
            _event("system", subtype="status", status="compacting"),
            _event("system", subtype="compact_boundary"),
            _event("system", subtype="status", status="compacting", session_id="s2"),
            _event("system", subtype="status", status=None, compact_result="failed", session_id="s2"),
            _event("system", subtype="status", status="compacting", session_id="s3"),
        ]
        stream = parse_claude_code_stream(_stream_text(events).splitlines())

        assert stream.compaction_boundaries == 1
        assert sorted(stream.compaction_attempts, key=lambda a: a["invocation_id"]) == [
            {"invocation_id": "s1", "outcome": "unknown"},
            {"invocation_id": "s2", "outcome": "failed"},
            {"invocation_id": "s3", "outcome": "unknown"},
        ]
        assert stream.output == [] and stream.usage is None

    def test_refused_tool_calls_are_counted_in_every_conversation(self, fixed_uuid: None) -> None:
        events = [
            _assistant("msg_1", {"type": "tool_use", "id": "c1", "name": "read", "input": {"path": "a"}}, MSG1_USAGE),
            _tool_result("c1", UNKNOWN_TOOL),
            _assistant(
                "msg_sub",
                {"type": "tool_use", "id": "c2", "name": "Bash", "input": {"command": "false"}},
                MSG1_USAGE,
                parent_tool_use_id="c0",
            ),
            _tool_result("c2", "Exit code 1", is_error=True, parent_tool_use_id="c0"),
            _tool_result("c3", "fine", parent_tool_use_id="c0"),
        ]
        stream = parse_claude_code_stream(_stream_text(events).splitlines())

        assert (stream.tool_calls, stream.tool_errors, stream.unknown_tool_calls) == (2, 2, 1)
        assert (stream.main_turns, stream.subagent_turns, stream.subagent_calls) == (1, 1, 0)
        assert [item.type for item in stream.output] == ["function_call", "function_call_output"]

    @mark.parametrize(
        ("result", "expected"),
        [
            (None, ("incomplete", "result_missing")),
            ({"subtype": "success", "is_error": False}, ("completed", None)),
            ({"subtype": "error_max_turns", "is_error": True}, ("incomplete", "error_max_turns")),
            ({"subtype": "error_during_execution", "is_error": True}, ("failed", "error_during_execution")),
            ({"subtype": "success", "is_error": True, "result": "API Error: 500"}, ("failed", "agent_error")),
            ({"subtype": "success", "is_error": True, "result": "Prompt is too long"}, ("failed", "context_overflow")),
        ],
    )
    def test_invocation_outcome(self, result: Any, expected: Any) -> None:
        assert invocation_outcome(result) == expected

    def test_context_overflow_needs_an_error_result(self) -> None:
        assert context_overflow(None) is False
        assert context_overflow({"subtype": "success", "is_error": False, "result": "Prompt is too long"}) is False
        assert context_overflow({"subtype": "success", "is_error": True, "result": " Prompt is too long"}) is True


class TestCommand:
    def test_training_settings(self) -> None:
        agent = _agent(
            auto_compact=False,
            remote_claude_code_binary_path="/mnt/s3/claude",
            max_output_tokens=128000,
            api_timeout_ms=3600000,
            stream_first_byte_timeout_ms=1800000,
            disallowed_tools="WebFetch,WebSearch",
        )
        paths = ClaudeCodeRunPaths(root="/tmp/run")
        env = agent._env("http://model:5/ng-rollout/7-2/training-token-capture", paths)

        assert env["ANTHROPIC_BASE_URL"] == "http://model:5/ng-rollout/7-2/training-token-capture"
        assert env["CLAUDE_CONFIG_DIR"] == "/tmp/run/config"
        for name in (
            "ANTHROPIC_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "CLAUDE_CODE_SUBAGENT_MODEL",
        ):
            assert env[name] == "policy"
        assert env["DISABLE_COMPACT"] == env["DISABLE_AUTO_COMPACT"] == "1"
        assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
        assert env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
        assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "128000"
        assert env["API_TIMEOUT_MS"] == "3600000"
        assert env["CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS"] == "1800000"
        assert agent._settings() == {"autoCompactEnabled": False, "precomputeCompactionEnabled": False}

        args = agent._claude_args(paths, QUERY)
        assert args[:5] == ["-p", "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions"]
        assert args[args.index("--settings") + 1] == "/tmp/run/settings.json"
        assert args[args.index("--disallowedTools") + 1] == "WebFetch,WebSearch"
        assert args[args.index("--setting-sources") + 1] == "user"
        assert "--bare" not in args and "--max-turns" not in args
        assert args[-2:] == ["--", QUERY]

    def test_bare_and_setting_sources_are_optional(self) -> None:
        paths = ClaudeCodeRunPaths(root="/tmp/run")
        args = _agent(bare=True, setting_sources=None, max_turns=50)._claude_args(paths, QUERY)
        assert "--bare" in args and "--setting-sources" not in args
        assert args[args.index("--max-turns") + 1] == "50"
        assert args[-2:] == ["--", QUERY]

    def test_defaults_leave_compaction_on(self) -> None:
        agent = _agent()
        env = agent._env("http://model:5", ClaudeCodeRunPaths(root="/tmp/run"))

        assert "DISABLE_COMPACT" not in env and "DISABLE_AUTO_COMPACT" not in env
        assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in env and "API_TIMEOUT_MS" not in env
        assert agent._settings() == {}

    def test_overrides_are_layered_last(self) -> None:
        agent = _agent(
            auto_compact=False,
            claude_code_env={"DISABLE_COMPACT": "0", "EXTRA": "1"},
            claude_code_settings={"autoCompactEnabled": True, "includeCoAuthoredBy": False},
        )
        env = agent._env("http://model:5", ClaudeCodeRunPaths(root="/tmp/run"))

        assert (env["DISABLE_COMPACT"], env["EXTRA"]) == ("0", "1")
        assert agent._settings() == {
            "autoCompactEnabled": True,
            "precomputeCompactionEnabled": False,
            "includeCoAuthoredBy": False,
        }

    def test_without_staged_binary_runs_the_official_installer(self) -> None:
        command = _agent().build_command("http://model:5", QUERY, ClaudeCodeRunPaths(root="/tmp/run"))
        assert "curl -fsSL https://claude.ai/install.sh -o /tmp/run/install.sh" in command
        assert f"bash /tmp/run/install.sh {VERSION}" in command

    def test_musl_sandbox_without_musl_binary_fails_clearly(self) -> None:
        agent = _agent(remote_claude_code_binary_path="/mnt/s3/claude")
        command = agent.build_command("http://model:5", QUERY, ClaudeCodeRunPaths(root="/tmp/run"))
        assert "remote_claude_code_musl_binary_path is not set" in command
        agent = _agent(remote_claude_code_binary_path="/mnt/a", remote_claude_code_musl_binary_path="/mnt/b")
        command = agent.build_command("http://model:5", QUERY, ClaudeCodeRunPaths(root="/tmp/run"))
        assert "then src=/mnt/b; else src=/mnt/a; fi" in command

    @mark.parametrize(
        ("observability_enabled", "token_capture_enabled", "expected_base_url"),
        [
            (False, False, "http://model-server"),
            (True, False, "http://model-server/ng-rollout/7-2"),
            (False, True, "http://model-server/ng-rollout/7-2/training-token-capture"),
        ],
        ids=("disabled", "observability-only", "token-capture"),
    )
    async def test_model_base_url_routes_model_calls(
        self,
        model_server_url: None,
        observability_enabled: bool,
        token_capture_enabled: bool,
        expected_base_url: str,
    ) -> None:
        agent = _agent(
            {
                "observability_enabled": observability_enabled,
                "token_id_capture": {"enabled": token_capture_enabled, "all_agents": False},
            }
        )
        request = MagicMock()
        request.json = AsyncMock(return_value=BODY)

        assert await agent._model_base_url(request) == expected_base_url


FAKE_CLAUDE = """#!/bin/sh
if [ "$1" = "--version" ]; then echo "{version} (Claude Code)"; exit 0; fi
run_dir=$(dirname "$CLAUDE_CONFIG_DIR")
printf '%s\\0' "$@" > "$run_dir/argv.bin"
env > "$run_dir/env.txt"
readlink /proc/self/fd/0 > "$run_dir/stdin.txt"
pwd > "$run_dir/cwd.txt"
echo '{{"type": "system", "subtype": "init", "session_id": "s1"}}'
echo '{{"type": "result", "subtype": "success", "is_error": false, "session_id": "s1"}}'
exit {exit_code}
"""


@mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
class TestCommandRuns:
    """Run the generated command in a local shell against a staged fake binary."""

    def _run(self, tmp_path: Path, *, reported_version: str = VERSION, exit_code: int = 0, **overrides: Any):
        staged = tmp_path / "staged" / "claude"
        staged.parent.mkdir()
        staged.write_text(FAKE_CLAUDE.format(version=reported_version, exit_code=exit_code))
        staged.chmod(0o644)  # bucket mounts keep no executable bit
        repo = tmp_path / "repo"
        repo.mkdir()
        agent = _agent(
            auto_compact=False,
            remote_claude_code_binary_path=str(staged),
            remote_work_dir=str(tmp_path / "work"),
            **overrides,
        )
        paths = agent._new_run_paths()
        command = agent.build_command("http://model:5/ng-rollout/7-2/training-token-capture", QUERY, paths)
        result = subprocess.run(
            ["sh", "-c", command], cwd=repo, capture_output=True, text=True, timeout=60, stdin=subprocess.PIPE
        )
        return agent, paths, result, repo

    def test_installs_and_runs_claude(self, tmp_path: Path) -> None:
        agent, paths, result, repo = self._run(tmp_path)

        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[-1] == FINISHED_MARKER
        assert result.stdout.splitlines()[0].startswith("Shell: ")
        root = Path(paths.root)
        assert (root / "bin" / "claude").stat().st_mode & 0o111
        argv = (root / "argv.bin").read_bytes().split(b"\0")[:-1]
        assert [arg.decode() for arg in argv] == agent._claude_args(paths, QUERY)
        env = dict(line.split("=", 1) for line in (root / "env.txt").read_text().splitlines() if "=" in line)
        assert env["ANTHROPIC_BASE_URL"] == "http://model:5/ng-rollout/7-2/training-token-capture"
        assert env["DISABLE_COMPACT"] == "1" and env["IS_SANDBOX"] == "1"
        assert (root / "stdin.txt").read_text().strip() == "/dev/null"
        assert Path((root / "cwd.txt").read_text().strip()).resolve() == repo.resolve()
        assert json.loads((root / "settings.json").read_text()) == agent._settings()
        stream = parse_claude_code_stream((root / "stream.jsonl").read_text().splitlines())
        assert stream.result is not None and stream.result["subtype"] == "success"
        assert list(repo.iterdir()) == []  # nothing lands in the repository

    def test_version_mismatch_stops_before_the_run(self, tmp_path: Path) -> None:
        agent, paths, result, _ = self._run(tmp_path, reported_version="2.1.286")

        assert result.returncode != 0
        assert FINISHED_MARKER not in result.stdout
        assert f"does not report version {VERSION}" in result.stderr
        assert not Path(paths.stream).exists()

    def test_failed_run_prints_no_marker(self, tmp_path: Path) -> None:
        _, _, result, _ = self._run(tmp_path, exit_code=1)

        assert result.returncode == 1
        assert FINISHED_MARKER not in result.stdout

    @mark.skipif(shutil.which("timeout") is None, reason="needs coreutils timeout")
    def test_cli_runs_under_a_deadline_before_the_sandbox_timeout(self, tmp_path: Path) -> None:
        agent, paths, result, _ = self._run(tmp_path, sandbox_timeout=130, harness_timeout_margin_s=120)

        assert "limit='timeout -k 30 10'" in agent.build_command("http://m", QUERY, paths)
        assert result.returncode == 0, result.stderr

    def test_no_deadline_when_the_margin_exceeds_the_timeout(self) -> None:
        agent = _agent(sandbox_timeout=60, harness_timeout_margin_s=120)
        command = agent.build_command("http://m", QUERY, ClaudeCodeRunPaths(root="/tmp/run"))
        assert "limit='' && " in command and "timeout -k" not in command


def _exec_result(*, return_code: int = 0, finished: bool = True, error_type: str | None = None) -> SimpleNamespace:
    marker = f"\n{FINISHED_MARKER}" if finished else ""
    return SimpleNamespace(
        stdout=f"Shell: /bin/bash\nInstalled Claude Code{marker}\n",
        stderr="",
        return_code=return_code,
        error_type=error_type,
    )


class _FakeSandbox:
    """Runs the command, then serves the stream file and transcripts archive like AsyncSandbox."""

    def __init__(
        self,
        stream: str | None,
        transcripts: bytes | None,
        run_result: SimpleNamespace | None = None,
        tar_return_code: int = 0,
    ) -> None:
        self.stream = stream
        self.transcripts = transcripts
        self.run_result = run_result or _exec_result()
        self.tar_result = SimpleNamespace(stdout="", stderr="", return_code=tar_return_code, error_type=None)
        self.exec = AsyncMock(side_effect=self._exec)
        self.download = AsyncMock(side_effect=self._download)
        self.stop = AsyncMock()

    async def _exec(self, command: str, timeout_s: float | None = None) -> SimpleNamespace:
        # The run command ends by printing the finished marker; the only other command archives transcripts.
        return self.run_result if FINISHED_MARKER in command else self.tar_result

    async def _download(self, remote_path: str, local_path: Path) -> None:
        if remote_path.endswith("/stream.jsonl") and self.stream is not None:
            Path(local_path).write_text(self.stream)
        elif remote_path.endswith("/transcripts.tar") and self.transcripts is not None:
            Path(local_path).write_bytes(self.transcripts)
        else:
            raise FileNotFoundError(remote_path)


def _transcripts_tar() -> bytes:
    lines = [
        {
            "type": "assistant",
            "sessionId": "s1",
            "uuid": "u1",
            "timestamp": "2026-10-01T00:00:00Z",
            "message": {"id": "msg_3", "role": "assistant", "content": [{"type": "text", "text": "Fixed."}]},
        }
    ]
    data = ("\n".join(json.dumps(line) for line in lines) + "\n").encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("projects/-repo/s1.jsonl")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _request(observe: bool = True, cookies: Dict[str, str] | None = None) -> Any:
    state = SimpleNamespace(_ng_observation_invocation_id="7-2") if observe else SimpleNamespace()
    return SimpleNamespace(
        cookies={"sandbox_id": "session-1"} if cookies is None else cookies,
        session={SESSION_ID_KEY: "session-1"},
        state=state,
        json=AsyncMock(return_value=BODY),
    )


def _params() -> Any:
    return ClaudeCodeSandboxedAgentRunRequest.model_validate(BODY).responses_create_params


class TestResponses:
    @fixture(autouse=True)
    def _results_in_tmp(self, tmp_path: Path, monkeypatch: MonkeyPatch, model_server_url: None) -> None:
        monkeypatch.setattr(app_module, "__file__", str(tmp_path / "app.py"))

    def _agent_with(self, sandbox: _FakeSandbox, **overrides: Any) -> ClaudeCodeSandboxedAgent:
        agent = _agent(
            CAPTURE_CONFIG, auto_compact=False, remote_claude_code_binary_path="/mnt/s3/claude", **overrides
        )
        agent._sandbox_id_to_sandbox["session-1"] = sandbox
        return agent

    async def test_transcript_receipt_and_observations(self, tmp_path: Path, fixed_uuid: None) -> None:
        sandbox = _FakeSandbox(_stream_text([*STREAM_EVENTS, RESULT_EVENT]), _transcripts_tar())
        agent = self._agent_with(sandbox)

        response = await agent.responses(_request(), _params())

        assert response.output == EXPECTED_OUTPUT
        assert response.usage is not None and response.usage.total_tokens == 180
        command = sandbox.exec.await_args_list[0].kwargs["command"]
        assert "ANTHROPIC_BASE_URL=http://model-server/ng-rollout/7-2/training-token-capture " in command
        assert sandbox.exec.await_args_list[0].kwargs["timeout_s"] == 600
        assert sandbox.exec.await_args_list[1].kwargs["command"].startswith("tar -C ")

        run_result = agent._sandbox_id_to_run_result["session-1"]
        observations = run_result.pop("_ng_agent_observations")
        results_dir = tmp_path / "results" / "session-1"
        assert run_result == {
            "claude_code_failed": False,
            "claude_code_exit_code": 0,
            "claude_code_error_type": None,
            "claude_code_results_fpath": str(results_dir / "stream.jsonl"),
            "claude_code_run_stdout": sandbox.run_result.stdout,
            "claude_code_run_stderr": "",
            "claude_code_export_found": True,
            "claude_code_finished": True,
            "claude_code_result_success": True,
            "claude_code_result_error": False,
            "claude_code_result_missing": False,
            "claude_code_context_overflow": False,
            "claude_code_duration_s": 1.234,
            "claude_code_main_turns": 3,
            "claude_code_subagent_turns": 1,
            "claude_code_subagent_calls": 1,
            "claude_code_tool_calls": 2,
            "claude_code_tool_errors": 0,
            "claude_code_unknown_tool_calls": 0,
        }
        assert (results_dir / "stream.jsonl").read_text() == _stream_text([*STREAM_EVENTS, RESULT_EVENT])
        receipt = json.loads((results_dir / "generation.json").read_text())
        assert receipt["execution"]["claude_code_finished"] is True
        assert [item["type"] for item in receipt["response"]["output"]] == [item.type for item in EXPECTED_OUTPUT]
        [root] = [record for record in observations.records if isinstance(record, AgentInvocation)]
        assert (root.invocation_id, root.status, root.duration_ms) == ("s1", "completed", 1234)
        [sandbox_record] = [record for record in observations.records if isinstance(record, SandboxObservation)]
        assert (sandbox_record.role, sandbox_record.outcome, sandbox_record.exit_code) == ("agent", "completed", 0)

    async def test_missing_stream_is_a_failed_run(self, tmp_path: Path) -> None:
        sandbox = _FakeSandbox(None, None, run_result=_exec_result(return_code=1, finished=False), tar_return_code=2)
        agent = self._agent_with(sandbox)

        response = await agent.responses(_request(), _params())

        assert (response.output, response.usage) == ([], None)
        run_result = agent._sandbox_id_to_run_result["session-1"]
        observations = run_result.pop("_ng_agent_observations")
        assert run_result["claude_code_failed"] is True
        assert run_result["claude_code_exit_code"] == 1
        assert run_result["claude_code_export_found"] is False
        assert run_result["claude_code_results_fpath"] == ""
        assert run_result["claude_code_result_missing"] is True
        assert run_result["claude_code_main_turns"] == 0
        assert {gap.code for gap in observations.gaps} >= {"observation_capture_failed"}
        [root] = [record for record in observations.records if isinstance(record, AgentInvocation)]
        assert root.invocation_id == "7-2"
        assert not (tmp_path / "results" / "session-1" / "stream.jsonl").exists()
        assert (
            json.loads((tmp_path / "results" / "session-1" / "generation.json").read_text())["response"]["output"]
            == []
        )

    async def test_deadline_kill_leaves_no_result(self) -> None:
        # `timeout` ends the CLI with 124 before it writes its result event; the patch so far is still graded.
        sandbox = _FakeSandbox(
            _stream_text(STREAM_EVENTS), None, run_result=_exec_result(return_code=124, finished=False)
        )
        agent = self._agent_with(sandbox)

        response = await agent.responses(_request(observe=False), _params())

        assert len(response.output) == len(EXPECTED_OUTPUT)
        run_result = agent._sandbox_id_to_run_result["session-1"]
        assert "_ng_agent_observations" not in run_result
        assert (run_result["claude_code_failed"], run_result["claude_code_exit_code"]) == (True, 124)
        assert (run_result["claude_code_result_missing"], run_result["claude_code_result_success"]) == (True, False)
        assert run_result["claude_code_error_type"] is None
        assert run_result["claude_code_duration_s"] is None
        sandbox.exec.assert_awaited_once()

    async def test_context_overflow_result(self) -> None:
        sandbox = _FakeSandbox(
            _stream_text([*STREAM_EVENTS, OVERFLOW_EVENT]),
            None,
            run_result=_exec_result(return_code=1, finished=False),
        )
        agent = self._agent_with(sandbox)

        await agent.responses(_request(observe=False), _params())

        run_result = agent._sandbox_id_to_run_result["session-1"]
        assert (run_result["claude_code_context_overflow"], run_result["claude_code_result_error"]) == (True, True)
        assert (run_result["claude_code_failed"], run_result["claude_code_error_type"]) == (True, "context_overflow")
        assert run_result["claude_code_duration_s"] == 99.0

    async def test_exec_error_is_reported(self) -> None:
        sandbox = _FakeSandbox(None, None)
        sandbox.exec = AsyncMock(side_effect=TimeoutError("sandbox gone"))
        agent = self._agent_with(sandbox)

        response = await agent.responses(_request(observe=False), _params())

        assert response.output == []
        run_result = agent._sandbox_id_to_run_result["session-1"]
        assert (run_result["claude_code_failed"], run_result["claude_code_error_type"]) == (True, "TimeoutError")
        assert run_result["claude_code_exit_code"] is None

    async def test_requires_a_seeded_sandbox(self) -> None:
        with raises(ValueError, match="Use /run"):
            await _agent().responses(_request(cookies={}), _params())


class _Response:
    ok = True

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload
        self.cookies: Dict[str, str] = {}

    async def json(self) -> Dict[str, Any]:
        return self.payload

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class _RunRequest:
    def __init__(self, body: Dict[str, Any]) -> None:
        self._cookies: Dict[str, str] = {}
        self._body = body
        self.session = {SESSION_ID_KEY: "session-1"}
        self.state = SimpleNamespace()

    @property
    def cookies(self) -> Dict[str, str]:
        return self._cookies

    async def json(self) -> Dict[str, Any]:
        return self._body


def _server(agent: ClaudeCodeSandboxedAgent, *, reward_for_output: Any) -> AsyncMock:
    async def post(server_name: str, url_path: str, json: Any = None, cookies: Any = None) -> _Response:
        assert server_name == "task_server"
        if url_path == "/seed_session":
            return _Response({"sandbox_handle": "task-sandbox", "workdir": "/repo"})
        assert url_path == "/verify"
        return _Response(json | {"reward": reward_for_output(json["response"]["output"])})

    agent.server_client.post = AsyncMock(side_effect=post)
    return agent.server_client.post


class TestRun:
    @fixture(autouse=True)
    def _results_in_tmp(self, tmp_path: Path, monkeypatch: MonkeyPatch, model_server_url: None) -> None:
        monkeypatch.setattr(app_module, "__file__", str(tmp_path / "app.py"))

    async def test_run_end_to_end(self) -> None:
        agent = _agent(CAPTURE_CONFIG, auto_compact=False, remote_claude_code_binary_path="/mnt/s3/claude")
        sandbox = _FakeSandbox(_stream_text([*STREAM_EVENTS, RESULT_EVENT]), _transcripts_tar())
        agent._start_sandbox = AsyncMock(return_value=sandbox)
        post = _server(agent, reward_for_output=lambda output: 1.0 if output else 0.0)

        result = await agent.run(_RunRequest(BODY), ClaudeCodeSandboxedAgentRunRequest.model_validate(BODY))

        agent._start_sandbox.assert_awaited_once_with(sandbox_id="task-sandbox", workdir="/repo")
        assert [call.kwargs["url_path"] for call in post.await_args_list] == ["/seed_session", "/verify"]
        assert post.await_args_list[1].kwargs["cookies"]["sandbox_id"] == "session-1"
        dumped = result.model_dump(mode="json")
        assert dumped["reward"] == 1.0
        assert dumped["claude_code_finished"] is True and dumped["claude_code_failed"] is False
        assert dumped["claude_code_result_success"] is True
        assert dumped["claude_code_export_found"] is True
        assert dumped["claude_code_results_fpath"].endswith("results/session-1/stream.jsonl")
        assert [item["type"] for item in dumped["response"]["output"]][-1] == "message"
        assert dumped["ng_agent_observations"]["source"] == "claude_code"
        assert dumped.get("failure_kind") is None
        sandbox.stop.assert_awaited_once()
        assert agent._sandbox_id_to_sandbox == {} and agent._sandbox_id_to_run_result == {}

    async def test_execution_failure_reward_zero_grades_an_empty_response(self) -> None:
        agent = _agent(
            CAPTURE_CONFIG, remote_claude_code_binary_path="/mnt/s3/claude", execution_failure_reward_zero=True
        )
        sandbox = _FakeSandbox(
            _stream_text(STREAM_EVENTS), None, run_result=_exec_result(return_code=124, finished=False)
        )
        agent._start_sandbox = AsyncMock(return_value=sandbox)
        post = _server(agent, reward_for_output=lambda output: 1.0 if output else 0.0)

        result = await agent.run(_RunRequest(BODY), ClaudeCodeSandboxedAgentRunRequest.model_validate(BODY))

        assert post.await_args_list[1].kwargs["json"]["response"]["output"] == []
        dumped = result.model_dump(mode="json")
        assert (dumped["reward"], dumped["failure_kind"]) == (0.0, AGENT_RUN_ERROR)
        assert dumped["claude_code_failed"] is True and dumped["claude_code_exit_code"] == 124
        # The response keeps what the model generated, for inspection.
        assert len(dumped["response"]["output"]) == len(EXPECTED_OUTPUT)
        sandbox.stop.assert_awaited_once()

    async def test_sandbox_is_stopped_when_the_run_fails(self) -> None:
        agent = _agent(CAPTURE_CONFIG, remote_claude_code_binary_path="/mnt/s3/claude")
        sandbox = _FakeSandbox(None, None)
        sandbox.exec = AsyncMock(side_effect=RuntimeError("exec failed"))
        sandbox.download = AsyncMock(side_effect=RuntimeError("download failed"))
        agent._start_sandbox = AsyncMock(return_value=sandbox)

        async def post(server_name: str, url_path: str, json: Any = None, cookies: Any = None) -> _Response:
            if url_path == "/seed_session":
                return _Response({"sandbox_handle": "task-sandbox"})
            raise RuntimeError("verifier down")

        agent.server_client.post = AsyncMock(side_effect=post)

        with raises(RuntimeError, match="verifier down"):
            await agent.run(_RunRequest(BODY), ClaudeCodeSandboxedAgentRunRequest.model_validate(BODY))

        sandbox.stop.assert_awaited_once()
        assert agent._sandbox_id_to_sandbox == {} and agent._sandbox_id_to_run_result == {}


def test_agent_config_file_validates() -> None:
    config = yaml.safe_load((AGENT_DIR / "configs" / "claude_code_sandboxed_agent.yaml").read_text())
    block = config["claude_code_sandboxed_agent"]["responses_api_agents"]["claude_code_sandboxed_agent"]
    block |= {
        "host": "0.0.0.0",
        "port": 1,
        "name": "x",
        "resources_server": {"type": "resources_servers", "name": "r"},
    }

    parsed = ClaudeCodeSandboxedAgentConfig.model_validate(block)

    assert parsed.claude_code_version == VERSION
    assert parsed.token_id_capture is True
    assert parsed.stream_first_byte_timeout_ms == 1800000
    assert parsed.auto_compact is True  # training recipes turn it off
    assert (parsed.bare, parsed.auto_memory, parsed.setting_sources) == (False, False, "user")
    assert (parsed.execution_failure_reward_zero, parsed.artifacts_dir) == (False, None)


@mark.parametrize("server", ["swe_rebench", "scale_swe", "swemer_v1", "swemer_v2", "swe_next"])
def test_swe_wiring_configs_mirror_opencode(server: str) -> None:
    configs = PARENT_DIR / "resources_servers" / server / "configs"
    claude_code = yaml.safe_load((configs / f"{server}_claude_code.yaml").read_text())
    opencode = yaml.safe_load((configs / f"{server}_opencode.yaml").read_text())

    assert claude_code["config_paths"] == [
        "responses_api_agents/claude_code_sandboxed_agent/configs/claude_code_sandboxed_agent.yaml",
        f"resources_servers/{server}/configs/{server}.yaml",
    ]
    agent_name = f"{server}_claude_code_sandboxed_agent"
    agent = claude_code[agent_name]
    assert agent["_inherit_from"] == "claude_code_sandboxed_agent"
    block = agent["responses_api_agents"]["claude_code_sandboxed_agent"]
    opencode_block = opencode[f"{server}_opencode_sandboxed_agent"]["responses_api_agents"]["opencode_sandboxed_agent"]
    assert block == opencode_block
    # Rollout collection dispatches through an environment server that names the agent; a run that
    # loads both wiring configs must not have the two harnesses' servers collide.
    environment = claude_code[f"{server}_claude_code_environment_server"]["environment_servers"]["legacy_agent"]
    assert environment["agent_server"] == {"type": "responses_api_agents", "name": agent_name}
    assert set(claude_code) & set(opencode) == {"config_paths"}
    # Rollout collection refuses rows whose agent type the resources server does not accept.
    resources = yaml.safe_load((configs / f"{server}.yaml").read_text())
    accepted = resources[f"{server}_resources_server"]["resources_servers"][server]["allowed_agents"]
    assert {"opencode_sandboxed_agent", "claude_code_sandboxed_agent"} <= set(accepted)
