# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in integration test for the Terminus 2 ``agent_user`` containment split.

Runs the REAL ``Terminus2Agent._execute`` (Harbor Terminus 2 + tmux) with a scripted LLM inside a
sandbox created by the REAL ``TerminalBench21ResourcesServer._create_sandbox``, then runs the root
verifier phase through the real ``_upload_folder``. Skipped unless ``OPENSANDBOX_DOMAIN``,
``OPENSANDBOX_API_KEY`` and ``TERMINUS2_AGENT_USER_TEST_IMAGE`` are set (no private image tag lives
in the repo). Optional knobs: ``TERMINUS2_AGENT_USER_TEST_USER`` (default ``agent``),
``TERMINUS2_AGENT_USER_TEST_UID`` (default ``1000``), ``TERMINUS2_AGENT_USER_TEST_WORKSPACE``
(default ``/app``), ``TERMINUS2_AGENT_USER_TEST_CPU`` (1), ``_MEMORY_MIB`` (2048), ``_DISK_GIB`` (30),
``_TTL_S`` (1200), ``_READY_TIMEOUT_S`` (900).
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from harbor.llms.base import BaseLLM, LLMResponse

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminal_bench_2_1 import app as rs_module
from resources_servers.terminal_bench_2_1.app import (
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
    TerminalBench21SeedSessionRequest,
)
from responses_api_agents.terminus_2_sandboxed_agent import app as app_module
from responses_api_agents.terminus_2_sandboxed_agent.app import NeMoGymTerminus2, Terminus2Agent, Terminus2AgentConfig


PROBE_DIR = "/tmp/ng_agent_probe"
PROBE_DONE_MARKER = "NG_PROBE_DONE"
CONFIRMATION_TEXT = "Are you sure you want to mark the task as complete"
TMUX_SESSION_NAME = NeMoGymTerminus2.name()
# Fixed by Harbor: TmuxSession pipe-pane target (EnvironmentPaths.agent_dir / "terminus_2.pane").
TMUX_PANE_LOG = "/logs/agent/terminus_2.pane"
# /proc walk instead of ps/pgrep, which slim images lack: one "<uid> <cmdline>" line per tmux process.
TMUX_PROCESSES_COMMAND = (
    "for c in /proc/[0-9]*/comm; do "
    'case "$(cat "$c" 2>/dev/null)" in tmux*) p=${c%/comm}; '
    'echo "$(stat -c %u "$p") $(tr \'\\0\' \' \' < "$p/cmdline")";; esac; done'
)


def _integration_settings() -> SimpleNamespace:
    domain = os.environ.get("OPENSANDBOX_DOMAIN")
    api_key = os.environ.get("OPENSANDBOX_API_KEY")
    image = os.environ.get("TERMINUS2_AGENT_USER_TEST_IMAGE")
    if not domain or not api_key or not image:
        pytest.skip(
            "set OPENSANDBOX_DOMAIN, OPENSANDBOX_API_KEY and TERMINUS2_AGENT_USER_TEST_IMAGE "
            "(a root-default image with a non-root agent account) to run the agent_user integration tests"
        )
    return SimpleNamespace(
        domain=domain,
        api_key=api_key,
        protocol=os.environ.get("OPENSANDBOX_PROTOCOL", "http"),
        image=image,
        user=os.environ.get("TERMINUS2_AGENT_USER_TEST_USER", "agent"),
        uid=os.environ.get("TERMINUS2_AGENT_USER_TEST_UID", "1000"),
        workspace=os.environ.get("TERMINUS2_AGENT_USER_TEST_WORKSPACE", "/app"),
        cpu=float(os.environ.get("TERMINUS2_AGENT_USER_TEST_CPU", "1")),
        memory_mib=int(os.environ.get("TERMINUS2_AGENT_USER_TEST_MEMORY_MIB", "2048")),
        disk_gib=int(os.environ.get("TERMINUS2_AGENT_USER_TEST_DISK_GIB", "30")),
        ttl_s=int(os.environ.get("TERMINUS2_AGENT_USER_TEST_TTL_S", "1200")),
        ready_timeout_s=int(os.environ.get("TERMINUS2_AGENT_USER_TEST_READY_TIMEOUT_S", "900")),
    )


def _global_config(settings: SimpleNamespace) -> dict[str, Any]:
    # Exact env.yaml shape of the `sandbox` block that `sandbox_provider: sandbox` references.
    return {
        "sandbox": {
            "opensandbox": {
                "connection": {
                    "domain": settings.domain,
                    "api_key": settings.api_key,
                    "protocol": settings.protocol,
                    "use_server_proxy": True,
                    "request_timeout_s": 300,
                },
                "create": {
                    "timeout_s": 1500,
                    "request_timeout_s": 1200,
                    "skip_health_check": False,
                    "retries": 3,
                },
            }
        }
    }


def _resources_server(settings: SimpleNamespace) -> TerminalBench21ResourcesServer:
    config = TerminalBench21ResourcesServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="terminus_2_agent_user_integration",
        sandbox_provider="sandbox",
        sandbox_config={
            "ttl_s": settings.ttl_s,
            "ready_timeout_s": settings.ready_timeout_s,
            "resources": {"cpu": settings.cpu, "memory_mib": settings.memory_mib, "disk_gib": settings.disk_gib},
            "provider_options": {
                "resource_requests": {"cpu": 0.5, "memory_mib": 512, "disk_gib": settings.disk_gib},
            },
        },
        evaluation_timeout=600,
    )
    return TerminalBench21ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _agent() -> Terminus2Agent:
    config = Terminus2AgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="terminus_2_sandboxed_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="terminal_bench_2_1_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_turns=4,
        parser_name="json",
        enable_summarize=False,
        proactive_summarization_threshold=8000,
        tmux_pane_width=160,
        tmux_pane_height=40,
        dump_trajectory=False,
        debug=True,
        sandbox_provider="sandbox",
        sandbox_timeout=600,
        remote_tmux_binary_path=None,
    )
    return Terminus2Agent(config=config, server_client=MagicMock(spec=ServerClient))


class ScriptedTerminusLLM(BaseLLM):
    """Replays a fixed list of Terminus JSON replies and records every prompt it was asked."""

    def __init__(self, replies: list[str], prompts: list[str], **_kwargs: Any):
        super().__init__()
        self._replies = list(replies)
        self.prompts = prompts
        self.trajectory: list[Any] = []
        self._times_spent: list[float] = []
        self._model_calls_gt_10min = 0

    async def call(self, prompt: str, **kwargs: Any) -> LLMResponse:
        # Harbor's Chat passes exactly these keyword arguments.
        assert set(kwargs) <= {"message_history", "logging_path", "previous_response_id"}, sorted(kwargs)
        self.prompts.append(prompt)
        assert self._replies, f"scripted LLM exhausted after {len(self.prompts) - 1} replies; prompt: {prompt[-500:]}"
        self._times_spent.append(0.0)
        return LLMResponse(content=self._replies.pop(0), usage=None)

    def get_model_context_limit(self) -> int:
        return 1_000_000

    def get_model_output_limit(self) -> int | None:
        return None


def _terminus_reply(commands: list[dict[str, Any]], task_complete: bool) -> str:
    # Field order analysis -> plan -> commands is what the Harbor JSON parser expects (required fields).
    return json.dumps(
        {
            "analysis": "Scripted integration probe.",
            "plan": "Record the agent identity, then mark the task complete.",
            "commands": commands,
            "task_complete": task_complete,
        }
    )


def _probe_command(workspace: str) -> str:
    # Independent statements (`;`) so one denied step cannot hide the others; every rc lands in its own file.
    return (
        f"P={PROBE_DIR}; mkdir -p $P; "
        "id -u > $P/uid; id -un > $P/user; id -g > $P/gid; "
        "mkdir /tests 2>/dev/null; echo $? > $P/mkdir_tests_rc; "
        "mkdir -p /logs/verifier 2>/dev/null; echo $? > $P/mkdir_verifier_rc; "
        "touch /logs/verifier/ng-agent-write 2>/dev/null; echo $? > $P/touch_verifier_rc; "
        "sudo -n id -u >/dev/null 2>&1; echo $? > $P/sudo_rc; "
        f"touch {workspace}/ng-agent-probe; echo $? > $P/touch_workspace_rc; "
        f"echo {PROBE_DONE_MARKER}\n"
    )


def _install_scripted_llm(monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> list[str]:
    prompts: list[str] = []
    monkeypatch.setattr(app_module, "NeMoGymLLM", lambda **kwargs: ScriptedTerminusLLM(replies, prompts, **kwargs))
    monkeypatch.setattr(Terminus2Agent, "base_url_for_run", lambda *_args, **_kwargs: "http://model")
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")
    return prompts


def _request() -> SimpleNamespace:
    async def request_json() -> dict[str, Any]:
        return {}

    return SimpleNamespace(json=request_json, session={SESSION_ID_KEY: "it"})


async def _exec_ok(sandbox: Any, command: str, **kwargs: Any) -> str:
    result = await sandbox.exec(command, **kwargs)
    assert result.return_code == 0, (command, result)
    return (result.stdout or "").strip()


async def _download_text(sandbox: Any, remote_path: str, local_dir: Path) -> str:
    local_path = local_dir / Path(remote_path).name
    await sandbox.download(remote_path, local_path)
    return local_path.read_text().strip()


async def _create_sandbox(monkeypatch: pytest.MonkeyPatch, settings: SimpleNamespace, task_folder: Path) -> Any:
    monkeypatch.setattr(rs_module, "get_global_config_dict", lambda: _global_config(settings))
    resources_server = _resources_server(settings)
    seed_request = TerminalBench21SeedSessionRequest(
        task_name="terminus-2-agent-user-integration",
        docker_image=settings.image,
        task_folder=str(task_folder),
        agent_user=settings.user,
    )
    sandbox = await resources_server._create_sandbox(seed_request)
    return resources_server, sandbox


@pytest.mark.asyncio
async def test_agent_user_runs_terminus_unprivileged_and_verifier_as_image_default(monkeypatch, tmp_path):
    settings = _integration_settings()
    resources_server, sandbox = await _create_sandbox(monkeypatch, settings, tmp_path)
    try:
        # The verifier runs as the image default, so the image default must be root.
        assert await _exec_ok(sandbox, "id -u") == "0"
        # Some fixture images ship a root-owned /logs/verifier (mode 0700) already; `mkdir -p` on it returns 0
        # as the agent even though the agent cannot write inside it, so record the image default beforehand.
        verifier_dir_preexists = (await sandbox.exec("test -d /logs/verifier")).return_code == 0

        # --- Agent phase: real Terminus 2 driven by the scripted LLM, as the non-root agent identity.
        prompts = _install_scripted_llm(
            monkeypatch,
            [
                _terminus_reply([{"keystrokes": _probe_command(settings.workspace), "duration": 5.0}], False),
                _terminus_reply([], True),
                _terminus_reply([], True),  # Terminus asks for confirmation once; confirm.
            ],
        )
        _response, metrics = await _agent()._execute(
            _request(),
            NeMoGymResponseCreateParamsNonStreaming(input="Record who you are, then mark the task complete."),
            sandbox,
            agent_user=settings.user,
        )

        # The probe files below are the authoritative evidence that the command ran; the pane capture in
        # prompts[1] is timing-dependent and is not asserted on.
        assert metrics["terminus2_completed"] is True
        assert len(prompts) == 3
        assert CONFIRMATION_TEXT not in prompts[1]
        assert CONFIRMATION_TEXT in prompts[2]

        # --- Read the probes as the image default.
        probe_output = await _exec_ok(
            sandbox,
            "for f in uid user gid mkdir_tests_rc mkdir_verifier_rc touch_verifier_rc sudo_rc touch_workspace_rc; do "
            f'printf "%s=%s\\n" "$f" "$(cat {PROBE_DIR}/$f)"; done',
        )
        probes = dict(line.split("=", 1) for line in probe_output.splitlines())
        assert probes["uid"] == settings.uid, probes
        assert probes["user"] == settings.user, probes
        assert probes["gid"] != "0", probes
        assert probes["mkdir_tests_rc"] != "0", probes
        # Writing inside /logs/verifier is always denied; creating it is denied only when it did not already exist.
        assert probes["touch_verifier_rc"] != "0", probes
        if not verifier_dir_preexists:
            assert probes["mkdir_verifier_rc"] != "0", probes
        assert probes["sudo_rc"] != "0", probes
        assert probes["touch_workspace_rc"] == "0", probes
        assert await _exec_ok(sandbox, f"stat -c %u {settings.workspace}/ng-agent-probe") == settings.uid
        assert await _exec_ok(sandbox, f"stat -c %u {TMUX_PANE_LOG}") == settings.uid

        # The tmux server belongs to the agent identity and is still alive (no teardown before verification).
        tmux_lines = [line for line in (await _exec_ok(sandbox, TMUX_PROCESSES_COMMAND)).splitlines() if line.strip()]
        assert tmux_lines, "no tmux process found via /proc"
        assert any(line.startswith(f"{settings.uid} ") for line in tmux_lines), tmux_lines
        assert not any(line.startswith("0 ") for line in tmux_lines), tmux_lines
        has_session = await sandbox.exec(f"tmux has-session -t {TMUX_SESSION_NAME}", user=settings.user)
        assert has_session.return_code == 0, has_session

        # --- Verifier phase as the image default: real _upload_folder, then bash /tests/test.sh.
        await _exec_ok(sandbox, "mkdir -p /tests /logs/verifier")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test.sh").write_text(
            "#!/bin/bash\nid -u > /logs/verifier/uid.txt\necho 1 > /logs/verifier/reward.txt\n"
        )
        await resources_server._upload_folder(sandbox, tests_dir, "/tests", {}, task_name="terminus-2-agent-user")
        await _exec_ok(sandbox, "bash /tests/test.sh")
        assert float(await _download_text(sandbox, "/logs/verifier/reward.txt", tmp_path)) == 1.0
        assert int(await _download_text(sandbox, "/logs/verifier/uid.txt", tmp_path)) == 0

        # The agent identity cannot write the verifier surfaces.
        for path in ("/tests/agent-write", "/logs/verifier/agent-write"):
            denied = await sandbox.exec(f"touch {path}", user=settings.user)
            assert denied.return_code != 0, (path, denied)
    finally:
        await sandbox.stop()


@pytest.mark.asyncio
async def test_agent_user_unknown_account_fails_closed_before_terminus_setup(monkeypatch, tmp_path):
    settings = _integration_settings()
    _, sandbox = await _create_sandbox(monkeypatch, settings, tmp_path)
    try:
        prompts = _install_scripted_llm(monkeypatch, [_terminus_reply([], True)])

        with pytest.raises(RuntimeError, match="no-such-user-ng"):
            await _agent()._execute(
                _request(),
                NeMoGymResponseCreateParamsNonStreaming(input="never runs"),
                sandbox,
                agent_user="no-such-user-ng",
            )

        # The LLM was never asked and no tmux session exists (tmux may not even be installed: rc 127).
        assert prompts == []
        has_session = await sandbox.exec(f"tmux has-session -t {TMUX_SESSION_NAME}")
        assert has_session.return_code != 0, has_session
    finally:
        await sandbox.stop()
