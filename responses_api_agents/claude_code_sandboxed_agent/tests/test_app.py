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
from pytest import MonkeyPatch, fixture, mark

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
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
from nemo_gym.rollout_observability import AgentInvocation
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.claude_code_sandboxed_agent import app as app_module
from responses_api_agents.claude_code_sandboxed_agent.app import (
    ClaudeCodeRunPaths,
    ClaudeCodeSandboxedAgent,
    ClaudeCodeSandboxedAgentConfig,
    ClaudeCodeSandboxedAgentRunRequest,
    invocation_outcome,
    parse_claude_code_stream,
)


AGENT_DIR = Path(app_module.__file__).parent
VERSION = "2.1.287"
QUERY = "Fix the bug.\nIt's in `parse()`: don't use $(rm -rf /) or \"quotes\" \\ wrongly."


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
    block = {"type": "tool_result", "tool_use_id": call_id, "content": content}
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

    @mark.parametrize(
        ("result", "expected"),
        [
            (None, ("incomplete", "result_missing")),
            ({"subtype": "success", "is_error": False}, ("completed", None)),
            ({"subtype": "error_max_turns", "is_error": True}, ("incomplete", "error_max_turns")),
            ({"subtype": "error_during_execution", "is_error": True}, ("failed", "error_during_execution")),
            ({"subtype": "success", "is_error": True}, ("failed", "success")),
        ],
    )
    def test_invocation_outcome(self, result: Any, expected: Any) -> None:
        assert invocation_outcome(result) == expected


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
    async def test_harness_command_routes_model_calls(
        self, monkeypatch: MonkeyPatch, observability_enabled: bool, token_capture_enabled: bool, expected_base_url: str
    ) -> None:
        agent = _agent(
            {
                "observability_enabled": observability_enabled,
                "token_id_capture": {"enabled": token_capture_enabled, "all_agents": False},
            }
        )
        monkeypatch.setattr(app_module, "get_server_url", lambda _name: "http://model-server")
        request = MagicMock()
        request.json = AsyncMock(
            return_value={"responses_create_params": {"input": "x"}, "_ng_task_index": 7, "_ng_rollout_index": 2}
        )

        command, paths = await agent._harness_command(request, QUERY, collect_observations=False)

        assert f"ANTHROPIC_BASE_URL={expected_base_url} " in command
        assert paths.root.startswith("/tmp/nemo-gym-claude-code-")


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
        assert result.stdout.splitlines()[-1] == agent.finished_marker
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
        assert agent.finished_marker not in result.stdout
        assert f"does not report version {VERSION}" in result.stderr
        assert not Path(paths.stream).exists()

    def test_failed_run_prints_no_marker(self, tmp_path: Path) -> None:
        agent, _, result, _ = self._run(tmp_path, exit_code=1)

        assert result.returncode == 1
        assert agent.finished_marker not in result.stdout

    @mark.skipif(shutil.which("timeout") is None, reason="needs coreutils timeout")
    def test_cli_runs_under_a_deadline_before_the_sandbox_timeout(self, tmp_path: Path) -> None:
        agent, paths, result, _ = self._run(tmp_path, sandbox_timeout=130, harness_timeout_margin_s=120)

        assert "limit='timeout -k 30 10'" in agent.build_command("http://m", QUERY, paths)
        assert result.returncode == 0, result.stderr

    def test_no_deadline_when_the_margin_exceeds_the_timeout(self) -> None:
        agent = _agent(sandbox_timeout=60, harness_timeout_margin_s=120)
        command = agent.build_command("http://m", QUERY, ClaudeCodeRunPaths(root="/tmp/run"))
        assert "limit='' && " in command and "timeout -k" not in command


class _FakeSandbox:
    """Serves a stream file and a transcripts archive the way AsyncSandbox.download does."""

    def __init__(self, stream: str | None, transcripts: bytes | None, tar_return_code: int = 0) -> None:
        self.stream = stream
        self.transcripts = transcripts
        self.exec = AsyncMock(
            return_value=SimpleNamespace(stdout="", stderr="", return_code=tar_return_code, error_type=None)
        )
        self.download = AsyncMock(side_effect=self._download)

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


class TestCollect:
    @fixture(autouse=True)
    def _results_in_tmp(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(app_module, "__file__", str(tmp_path / "app.py"))

    def _request(self) -> Any:
        return SimpleNamespace(session={SESSION_ID_KEY: "session-1"})

    async def test_transcript_and_observations(self, tmp_path: Path, fixed_uuid: None) -> None:
        agent = _agent()
        sandbox = _FakeSandbox(_stream_text([*STREAM_EVENTS, RESULT_EVENT]), _transcripts_tar())
        paths = ClaudeCodeRunPaths(root="/tmp/run")

        transcript = await agent._harness_collect(self._request(), sandbox, paths, True, "7-2")

        assert transcript.output == EXPECTED_OUTPUT
        assert transcript.usage is not None and transcript.usage.total_tokens == 180
        assert transcript.results_fpath == tmp_path / "results" / "session-1" / "stream.jsonl"
        assert transcript.results_fpath.read_text() == _stream_text([*STREAM_EVENTS, RESULT_EVENT])
        assert sandbox.exec.await_args.kwargs["command"] == "tar -C /tmp/run/config -cf /tmp/run/transcripts.tar projects"
        [root] = [record for record in transcript.observations.records if isinstance(record, AgentInvocation)]
        assert (root.invocation_id, root.status, root.duration_ms) == ("s1", "completed", 1234)

    async def test_missing_stream_and_transcripts(self, tmp_path: Path) -> None:
        agent = _agent()
        sandbox = _FakeSandbox(None, None, tar_return_code=2)

        transcript = await agent._harness_collect(self._request(), sandbox, ClaudeCodeRunPaths(root="/r"), True, "7-2")

        assert (transcript.output, transcript.usage, transcript.results_fpath) == ([], None, None)
        assert {gap.code for gap in transcript.observations.gaps} >= {"observation_capture_failed"}
        [root] = transcript.observations.records
        assert root.invocation_id == "7-2"

    async def test_no_observations_unless_requested(self) -> None:
        agent = _agent()
        sandbox = _FakeSandbox(_stream_text(STREAM_EVENTS), None)

        transcript = await agent._harness_collect(self._request(), sandbox, ClaudeCodeRunPaths(root="/r"), False, None)

        assert transcript.observations is None
        sandbox.exec.assert_not_awaited()


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


async def test_run_end_to_end(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "__file__", str(tmp_path / "app.py"))
    monkeypatch.setattr(app_module, "get_server_url", lambda _name: "http://model-server")
    agent = _agent(
        {"token_id_capture": {"enabled": True, "all_agents": False}},
        auto_compact=False,
        remote_claude_code_binary_path="/mnt/s3/claude",
    )
    sandbox = _FakeSandbox(_stream_text([*STREAM_EVENTS, RESULT_EVENT]), _transcripts_tar())
    finished = SimpleNamespace(
        stdout=f"Shell: /bin/bash\nInstalled Claude Code\n{agent.finished_marker}\n",
        stderr="",
        return_code=0,
        error_type=None,
    )
    archived = SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None)
    sandbox.exec = AsyncMock(side_effect=[finished, archived])
    sandbox.stop = AsyncMock()
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    async def post(server_name: str, url_path: str, json: Any = None, cookies: Any = None) -> _Response:
        if url_path == "/seed_session":
            return _Response({"sandbox_handle": "task-sandbox"})
        assert url_path == "/verify"
        return _Response(json | {"reward": 1.0})

    agent.server_client.post = AsyncMock(side_effect=post)
    body = {"responses_create_params": {"input": [{"role": "user", "content": QUERY}]}, "_ng_task_index": 7}
    body["_ng_rollout_index"] = 2

    result = await agent.run(_RunRequest(body), ClaudeCodeSandboxedAgentRunRequest.model_validate(body))

    command = sandbox.exec.await_args_list[0].kwargs["command"]
    assert "ANTHROPIC_BASE_URL=http://model-server/ng-rollout/7-2/training-token-capture " in command
    assert sandbox.exec.await_args_list[0].kwargs["timeout_s"] == 600
    dumped = result.model_dump(mode="json")
    assert dumped["reward"] == 1.0
    assert dumped["harness_finished"] is True and dumped["claude_code_finished"] is True
    assert dumped["claude_code_export_found"] is True
    assert dumped["claude_code_results_fpath"].endswith("results/session-1/stream.jsonl")
    assert [item["type"] for item in dumped["response"]["output"]][-1] == "message"
    assert dumped["ng_agent_observations"]["source"] == "claude_code"
    sandbox.stop.assert_awaited_once()


def test_agent_config_file_validates() -> None:
    config = yaml.safe_load((AGENT_DIR / "configs" / "claude_code_sandboxed_agent.yaml").read_text())
    block = config["claude_code_sandboxed_agent"]["responses_api_agents"]["claude_code_sandboxed_agent"]
    block |= {"host": "0.0.0.0", "port": 1, "name": "x", "resources_server": {"type": "resources_servers", "name": "r"}}

    parsed = ClaudeCodeSandboxedAgentConfig.model_validate(block)

    assert parsed.claude_code_version == VERSION
    assert parsed.token_id_capture is True
    assert parsed.stream_first_byte_timeout_ms == 1800000
    assert parsed.auto_compact is True  # training recipes turn it off
    assert (parsed.bare, parsed.auto_memory, parsed.setting_sources) == (False, False, "user")


@mark.parametrize("server", ["swe_rebench", "scale_swe", "swemer_v1", "swemer_v2", "swe_next"])
def test_swe_wiring_configs_mirror_opencode(server: str) -> None:
    configs = PARENT_DIR / "resources_servers" / server / "configs"
    claude_code = yaml.safe_load((configs / f"{server}_claude_code.yaml").read_text())
    opencode = yaml.safe_load((configs / f"{server}_opencode.yaml").read_text())

    assert claude_code["config_paths"] == [
        "responses_api_agents/claude_code_sandboxed_agent/configs/claude_code_sandboxed_agent.yaml",
        f"resources_servers/{server}/configs/{server}.yaml",
    ]
    agent = claude_code[f"{server}_claude_code_sandboxed_agent"]
    assert agent["_inherit_from"] == "claude_code_sandboxed_agent"
    block = agent["responses_api_agents"]["claude_code_sandboxed_agent"]
    opencode_block = opencode[f"{server}_opencode_sandboxed_agent"]["responses_api_agents"]["opencode_sandboxed_agent"]
    assert block == opencode_block
    # Rollout collection refuses rows whose agent type the resources server does not accept.
    resources = yaml.safe_load((configs / f"{server}.yaml").read_text())
    accepted = resources[f"{server}_resources_server"]["resources_servers"][server]["allowed_agents"]
    assert {"opencode_sandboxed_agent", "claude_code_sandboxed_agent"} <= set(accepted)
