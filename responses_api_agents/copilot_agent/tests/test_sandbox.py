# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shutil
import signal
import subprocess
import time

import pytest

from nemo_gym.agents import HARNESS_RUNNER_SCRIPT, AgentSeedSessionRequest, SandboxSession
from nemo_gym.core import EpisodeId, TaskId
from responses_api_agents.copilot_agent.app import _events, parse_copilot_events
from responses_api_agents.copilot_agent.sandbox import (
    CopilotSandboxSession,
    parse_compaction_state,
    parse_runtime_info,
)


class LocalFiles:
    """Provider double whose uploads land on the local filesystem."""

    async def upload(self, source, destination):
        shutil.copy(source, destination)


@pytest.mark.skipif(shutil.which("bash") is None, reason="The shared harness runner requires bash")
def test_shared_runner_records_events_the_adapter_reads(tmp_path):
    """harness_runner.sh, given the adapter's params.sh, records Copilot's JSON events with stdin closed."""
    child_pid = tmp_path / "tool.pid"
    copilot = tmp_path / "copilot"
    events = [
        {"type": "assistant.message_delta", "data": {"deltaContent": "Fi"}},
        {"type": "assistant.message", "data": {"content": "Fixed", "toolRequests": [], "apiCallId": "chatcmpl-1"}},
        {"type": "result", "sessionId": "copilot-run-1", "exitCode": 0},
    ]
    copilot.write_text(
        "#!/bin/bash\n"
        '[ -z "$(cat)" ] || exit 9\n'
        'echo "not json"\n'
        + "".join(f"printf '%s\\n' {json.dumps(json.dumps(event))}\n" for event in events)
        # A tool that stays behind with Copilot's stdout must not hold the run open.
        + f"sleep 60 & echo $! > {child_pid}\n"
        + 'echo "model: $COPILOT_MODEL" >&2\n'
        # The final event has no trailing newline.
        + 'printf \'{"type": "session.shutdown", "data": {}}\'\n'
        + "exit 4\n"
    )
    copilot.chmod(0o755)
    shutil.copy(HARNESS_RUNNER_SCRIPT, tmp_path / HARNESS_RUNNER_SCRIPT.name)
    seed = AgentSeedSessionRequest(
        agent_session_id="copilot-session",
        episode_id=EpisodeId(rollout_id="copilot-smoke", attempt=0),
        task_id=TaskId(taskset="terminal_bench_2_1", task_id="task"),
    )
    state = CopilotSandboxSession(
        request=seed,
        session=SandboxSession(
            sandbox=LocalFiles(), session_dir=str(tmp_path), workdir=str(tmp_path), harness="Copilot"
        ),
        invocation_id="copilot-invocation",
    )
    command = asyncio.run(state.stage_activation(command=[str(copilot)], env={"COPILOT_MODEL": "m"}))
    started = time.monotonic()
    try:
        finished = subprocess.run(command.argv, stdin=subprocess.DEVNULL, timeout=30)
        elapsed = time.monotonic() - started
    finally:
        if child_pid.exists():
            try:
                os.kill(int(child_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
    assert finished.returncode == 4
    assert elapsed < 20, f"runner waited {elapsed:.1f}s for a detached tool"
    recorded = _events((tmp_path / "events.jsonl").read_text())
    assert [event["type"] for _, event in recorded] == [
        "assistant.message_delta",
        "assistant.message",
        "result",
        "session.shutdown",
    ]
    transcript = parse_copilot_events(recorded, "copilot-invocation")
    assert [item.type for item in transcript.output] == ["message"]
    assert transcript.response_ids == ["chatcmpl-1"] and transcript.session_id == "copilot-run-1"
    assert (tmp_path / "stderr.log").read_text() == "model: m\n"
    info = parse_runtime_info(json.loads((tmp_path / "runtime.json").read_text()))
    assert info is not None and info.pid > 0


@pytest.mark.parametrize(
    ("logs", "expected"),
    [
        ([], None),
        ([""], None),
        (['{"type":"session.start"}\n'], False),
        (['{"type":"session.start"}\n{"type":"session.compaction_start"}\n'], True),
        (['{"type":"session.compaction_complete","data":{"success":false}}\n'], True),
        (['{"type":"session.start"}\n', '{"type":"session.compaction_start"}\n'], True),
        (['{"type":"session.start"}\n{"truncated":'], None),
        (['["not an event"]\n'], None),
    ],
)
def test_persisted_compaction_state(logs, expected):
    assert parse_compaction_state(logs) is expected
