# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Claude's staged activation with the shared bash runner, as the sandbox does."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.agents import HARNESS_RUNNER_FILE, HARNESS_RUNNER_SCRIPT
from responses_api_agents.claude_code_agent import sandbox as adapter
from responses_api_agents.claude_code_agent.parsing import _invocation_outcome, parse_stream_json
from responses_api_agents.claude_code_agent.tests.test_sandbox import seed


pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")


def stage(tmp_path: Path, command: list[str], *, access_env: dict[str, str] | None = None) -> list[str]:
    """Stage ``command`` through stage_activation into a local session directory, and return the runner's argv.

    ``access_env`` is what the owner of the borrowed sandbox states for the harness in ``SandboxAccess.env``.
    """

    async def upload(source, target):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    shutil.copyfile(HARNESS_RUNNER_SCRIPT, tmp_path / HARNESS_RUNNER_FILE)
    request = seed()
    request.sandbox_access.env = access_env or {}
    state = adapter.ClaudeCodeSandboxSession(
        request=request,
        runtime="/runtime",
        session=SimpleNamespace(session_dir=str(tmp_path), sandbox=SimpleNamespace(upload=upload)),
    )
    payload = {
        "command": command,
        "cwd": str(tmp_path),
        "env": {"GYM_VALUE": "it's set"},
        "settings": {},
        "mcp_config": None,
        "skills_path": None,
    }
    return asyncio.run(state.stage_activation(payload)).argv


def runner_command(tmp_path: Path, code: str) -> list[str]:
    """Stage a stub CLI through stage_activation into a local session directory."""
    child = tmp_path / "child.py"
    child.write_text(code)
    return stage(tmp_path, [sys.executable, str(child)])


def test_large_json_stdout_cannot_be_corrupted_by_interleaved_stderr(tmp_path):
    command = runner_command(
        tmp_path,
        """import json,os,sys
text="x"*300000
line=json.dumps({"type":"assistant","message":{"content":[{"type":"text","text":text}]}}).encode()+b"\\n"
os.write(1,line[:150000])
os.write(2,b'{"type":"result","subtype":"error_fake"}\\n'+b'diagnostic\\xff\\n')
os.write(1,line[150000:])
os.write(2,(os.environ["GYM_VALUE"]+"|"+repr(sys.stdin.read())).encode())
os.write(1,b'{"type":"result","subtype":"success","is_error":false}\\n')
""",
    )
    result = subprocess.run(command, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    raw = (tmp_path / "events.jsonl").read_text()
    assert "error_fake" not in raw
    output, metadata = parse_stream_json(raw)
    assert output[0].content[0].text == "x" * 300000
    assert _invocation_outcome(metadata, result.returncode) == ("completed", None)
    assert (tmp_path / "stderr.log").read_bytes().endswith(b"diagnostic\xff\nit's set|''")
    runtime = adapter.parse_runtime_info(json.loads((tmp_path / "runtime.json").read_text()))
    assert runtime is not None and runtime.hostname and runtime.pid > 0


@pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX termination")
def test_termination_preserves_partial_events(tmp_path):
    command = runner_command(
        tmp_path,
        """import os,time
os.write(1,b'{"type":"assistant","message":{"content":[{"type":"text","text":"partial"}]}}\\n')
time.sleep(60)
""",
    )
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "events.jsonl").exists() or (tmp_path / "events.jsonl").stat().st_size == 0:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        process.terminate()
        process.communicate(timeout=5)
        output, metadata = parse_stream_json((tmp_path / "events.jsonl").read_text())
        assert output[0].content[0].text == "partial"
        assert "subtype" not in metadata
        assert process.returncode != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_sandbox_access_env_reaches_claude_except_reserved_prefixes(tmp_path):
    access_env = {
        "TASK_API_TOKEN": "task value",
        "LANG": "C",
        "PATH": f"/task-bin:{os.environ['PATH']}",
        "CLAUDE_PROJECT_DIR": "/task-project",
        "GYM_VALUE": "must-not-apply",
        "ANTHROPIC_BASE_URL": "http://must-not-apply",
        "CLAUDE_CODE_USE_BEDROCK": "must-not-apply",
        "NEMO_GYM_SUPERVISOR": "must-not-apply",
    }
    report = "".join(f'printf "%s|" "${{{name}-unset}}"; ' for name in access_env)
    command = stage(tmp_path, ["bash", "-c", report], access_env=access_env)
    # The sandbox's own environment; the process supervisor sets the marker for the harness.
    env = {key: value for key, value in os.environ.items() if key not in access_env}
    env |= {"LANG": "C.UTF-8", "PATH": os.environ["PATH"], "NEMO_GYM_SUPERVISOR": "marker"}
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    # An entry replaces the sandbox's value of the same name, and the adapter's own variable keeps its value.
    path = f"/task-bin:{os.environ['PATH']}"
    assert (tmp_path / "events.jsonl").read_text() == f"task value|C|{path}|/task-project|it's set|unset|unset|marker|"


@pytest.mark.parametrize("name", ["command", "cwd", "env", "stdout"])
def test_sandbox_access_env_entry_cannot_replace_a_runner_parameter(tmp_path, name):
    value = str(tmp_path / "elsewhere")
    report = f'printf "%s|" "$PWD" "$GYM_VALUE" "${name}"'
    command = stage(tmp_path, ["bash", "-c", report], access_env={name: value})
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    # The runner started the staged command in its working directory, with the adapter's variable and the staged
    # output file.
    assert (tmp_path / "events.jsonl").read_text() == f"{tmp_path}|it's set|{value}|"


def test_sandbox_access_env_pwd_entry_leaves_pwd_at_the_workdir(tmp_path):
    # printenv shows the variable as Claude Code receives it; a shell would set PWD again when it starts.
    command = stage(tmp_path, ["printenv", "PWD"], access_env={"PWD": "/nowhere"})
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "events.jsonl").read_text() == f"{tmp_path}\n"


def test_sandbox_access_env_path_entry_does_not_hide_the_staged_command(tmp_path):
    # The CLI is staged by its absolute path, so a PATH entry that holds only a task directory still starts it.
    printenv = shutil.which("printenv")
    command = stage(tmp_path, [printenv, "PATH"], access_env={"PATH": "/task-bin"})
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "events.jsonl").read_text() == "/task-bin\n"


def test_sandbox_access_env_entry_with_an_invalid_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid environment variable names"):
        stage(tmp_path, ["true"], access_env={"-i": "value"})


@pytest.mark.parametrize("access_env", [None, {"ANTHROPIC_BASE_URL": "http://must-not-apply"}])
def test_seed_without_forwarded_env_stages_the_command_unchanged(tmp_path, access_env):
    stage(tmp_path, ["claude", "-p"], access_env=access_env)
    assert "command=(claude -p)\n" in (tmp_path / "params.sh").read_text()
