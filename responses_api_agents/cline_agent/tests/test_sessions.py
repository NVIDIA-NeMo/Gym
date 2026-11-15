# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions: Cline in the task sandbox, behind the seed, activation and close contract."""

import asyncio
import json
import os
import shlex
import shutil
import signal
import stat
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request
from omegaconf import OmegaConf

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AGENT_SESSION_COOKIE_KEY,
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
    ROLLOUT_STOP_METADATA_KEY,
    AgentSeedSessionRequest,
    SandboxSession,
    agent_outcome_policy,
    check_agent_seed_rejected,
)
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming, NgFailureEvent, RequestFailureLog
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.trajectory import AgentInvocation, SandboxObservation, ToolCallObservation
from responses_api_agents.cline_agent import app
from responses_api_agents.cline_agent import sandbox as adapter
from responses_api_agents.cline_agent.observability import build_cline_observations


MODEL = ModelServerRef(type="responses_api_models", name="policy_model")


def seed(*, borrowed=True, workdir="/app", tool_accesses=(), descriptor=None):
    return AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id={"rollout_id": "rollout"},
        task_id={"taskset": "fixture", "task_id": "task"},
        tool_accesses=list(tool_accesses),
        sandbox_access={
            "connection": {"provider_config_ref": "sandbox", "descriptor": descriptor or {"id": "borrowed"}},
            "workdir": workdir,
        }
        if borrowed
        else None,
    )


def agent(**kwargs):
    kwargs.setdefault("model_server", MODEL)
    kwargs.setdefault("model", "policy-model")
    kwargs.setdefault("cline_version", "3.0.55")
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = OmegaConf.create(
        {
            "sandbox": {"local": {}},
            "policy_model": {"responses_api_models": {"vllm_model": {"host": "10.0.0.5", "port": 9000}}},
        }
    )
    instance = app.ClineAgent(
        config=app.ClineAgentConfig(name="cline", host="127.0.0.1", port=8080, entrypoint="app.py", **kwargs),
        server_client=server_client,
    )
    server_client._build_server_base_url = lambda config: f"http://{config['host']}:{config['port']}"
    return instance


def result(stdout="", *, code=0):
    return SimpleNamespace(stdout=stdout, stderr="diagnostic" if code else "", return_code=code, error_type=None)


def request(session_id="session", rollout="rollout"):
    return Request(
        {"type": "http", "session": {AGENT_SESSION_COOKIE_KEY: session_id}, "path_params": {"rollout_id": rollout}}
    )


def stream(*records) -> str:
    stamped = [{"ts": f"2026-11-03T00:00:{index:02d}.000Z", **record} for index, record in enumerate(records)]
    return "\n".join(json.dumps(record) for record in stamped) + "\n"


def tool(kind, call_id, **fields):
    return {"type": "agent_event", "event": {"type": kind, "contentType": "tool", "toolCallId": call_id, **fields}}


def text(value):
    return {"type": "agent_event", "event": {"type": "content_end", "contentType": "text", "text": value}}


def error_event(name, message, error_class="unknown"):
    return {
        "type": "agent_event",
        "event": {"type": "error", "error": {"name": name, "message": message}, "errorClass": error_class},
    }


def run_result(finish, **fields):
    return {"type": "run_result", "finishReason": finish, "durationMs": 1200, **fields}


def done(reason):
    return {"type": "agent_event", "event": {"type": "done", "reason": reason, "iterations": 1}}


COMPLETED = stream(
    tool("content_start", "call-1", toolName="run_commands", input={"commands": ["ls"]}),
    tool("content_end", "call-1", toolName="run_commands", output=[{"query": "ls", "result": "a", "success": True}]),
    text("Done."),
    run_result("completed", aggregateUsage={"inputTokens": 100, "outputTokens": 7}),
)


class FakeSession:
    """The SandboxSession surface the adapter uses, with a scripted execute()."""

    def __init__(self, *, artifacts=None, cleanup=None, raises=None):
        self.session_dir = "/tmp/nemo-gym-cline-sessions/id"
        self.workdir = "/app"
        self.sandbox = SimpleNamespace(sandbox_id="sandbox-1")
        self.closing = False
        self.expired = False
        self.artifacts = None
        self.cleanup = None
        self._result = (artifacts, cleanup, raises)
        self.staged = None
        self.close = AsyncMock()

    async def execute(self, *, stage_activation, collect, timeout, close_timeout):
        self.timeout = timeout
        self.staged = await stage_activation()
        self.artifacts, self.cleanup, raises = self._result
        if raises is not None:
            raise raises
        return self.artifacts


def state_with(session, *, borrowed=True):
    state = adapter.ClineSandboxSession(request=seed(borrowed=borrowed), session=session, runtime="/runtime")
    state.upload_json = AsyncMock()
    state.stage_activation = AsyncMock(return_value="command")
    return state


def cleanup(*, code=0, timed_out=False):
    return {"cleanup_confirmed": True, "return_code": code, "timed_out": timed_out, "error": None}


class LocalSandbox:
    """A sandbox whose files and commands are this machine's: uploads copy, exec runs bash."""

    sandbox_id = "local-1"
    uploads_independent_of_execs = True

    def __init__(self):
        self.disconnected = False

    async def upload(self, source, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    async def download(self, source, destination):
        shutil.copyfile(source, destination)

    async def disconnect(self):
        self.disconnected = True

    async def exec(self, command, *, cwd=None, timeout_s=30, user=None):
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = await asyncio.wait_for(process.communicate(), timeout_s)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            raise
        return SandboxExecResult(out.decode(errors="replace"), err.decode(errors="replace"), process.returncode)


def local_state(tmp_path):
    """A borrowed session over a real SandboxSession, with the directories seed would have prepared."""
    session_dir = tmp_path / "session"
    (session_dir / "config").mkdir(parents=True)
    (session_dir / "data" / "settings").mkdir(parents=True)
    shutil.copyfile(HARNESS_RUNNER_SCRIPT, session_dir / HARNESS_RUNNER_FILE)
    workdir = tmp_path / "task"
    workdir.mkdir()
    sandbox = LocalSandbox()
    state = adapter.ClineSandboxSession(
        request=seed(workdir=str(workdir)),
        session=SandboxSession(sandbox=sandbox, session_dir=str(session_dir), workdir=str(workdir), harness="Cline"),
        runtime=str(tmp_path / "runtime"),
    )
    return state, sandbox, workdir


# Seed


@pytest.mark.parametrize(
    ("config", "unsupported", "reason"),
    [
        (
            {},
            {"tool_accesses": [{"kind": "direct_http", "name": "tools", "base_url": "http://r", "required": True}]},
            "Cline sandbox sessions use Cline's built-in tools; required tool grants are unsupported: tools",
        ),
        ({"cline_version": None}, {}, "Cline sandbox sessions require an exact cline_version, for example 3.0.55"),
        ({"cline_version": "3.0"}, {}, "Cline sandbox sessions require an exact cline_version, for example 3.0.55"),
        ({"cline_version": "9.9.9"}, {}, "Cline 9.9.9 has no pinned runtime checksums; supported: 3.0.55"),
        ({"model_server": None}, {}, "Cline sandbox sessions require a Gym model_server and its model name"),
        ({"model": None}, {}, "Cline sandbox sessions require a Gym model_server and its model name"),
        (
            {"env": {"A": "b"}, "extra_args": ["--verbose"], "command": "npx cline"},
            {},
            "Cline sandbox sessions do not support these host settings: env, extra_args, command",
        ),
        (
            {"repo_dir": "/repo", "openai_api_key": "key", "openai_base_url": "http://x"},  # pragma: allowlist secret
            {},
            "Cline sandbox sessions do not support these host settings: repo_dir, openai_api_key, openai_base_url",
        ),
        ({}, {"borrowed": False}, "Cline requires sandbox_access or a configured sandbox_provider"),
        *(
            ({}, {"workdir": workdir}, "Cline sandbox workdir must be absolute and separate from its session storage")
            for workdir in (
                "/tmp",
                "app",
                "/app/../x",
                "/tmp/nemo-gym-cline-sessions/x",
                "/tmp/nemo-gym-cline-runtime-3.0.55/x",
            )
        ),
    ],
)
def test_unsupported_seed_is_refused_with_a_final_422(monkeypatch, config, unsupported, reason):
    connect = AsyncMock()
    monkeypatch.setattr(app.AsyncSandbox, "connect", connect)
    instance = agent(**config)
    assert check_agent_seed_rejected(instance.setup_webserver(), seed(**unsupported)) == reason
    assert not instance._session_records
    connect.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [True, False])
async def test_seed_installs_pinned_runtime_inside_the_sandbox(monkeypatch, borrowed):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock(), start=AsyncMock())
    constructor = MagicMock(return_value=remote)
    constructor.connect = AsyncMock(return_value=remote)
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    provider = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    instance = agent(sandbox_provider="sandbox", sandbox_config={"workdir": "/app"})

    state = await instance._seed_agent_session_state(seed(borrowed=borrowed))

    commands = [call.args[0] for call in remote.exec.await_args_list]
    digests = adapter.CLINE_RUNTIME_SHA512["3.0.55"]
    assert state.runtime == "/tmp/nemo-gym-cline-runtime-3.0.55"
    assert state.session.session_dir.startswith("/tmp/nemo-gym-cline-sessions/")
    assert state.session.owns_sandbox is not borrowed
    install = commands[-1]
    assert install.endswith(f"{state.runtime} 3.0.55 {digests['x64']} {digests['arm64']}")
    assert install.startswith(f"bash {state.session.session_dir}/install_cline_runtime.sh")
    uploaded = [call.args[1] for call in remote.upload.await_args_list]
    assert uploaded == [
        f"{state.session.session_dir}/install_cline_runtime.sh",
        f"{state.session.session_dir}/harness_runner.sh",
    ]
    assert remote.start.await_count == (0 if borrowed else 1)
    if borrowed:
        constructor.connect.assert_awaited_once_with({"id": "borrowed"}, provider=provider)
        # The Resources Server prepared the borrowed task; seed only checks that its workdir exists.
        assert not any(command.startswith("mkdir -p -- /app") for command in commands)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sandbox_config", "ttl_s"),
    [
        # Provider TTLs outlive the agent server: installation, the run and ten minutes to close.
        ({"metadata": {"team": "gym"}}, 600 + 900 + 600),
        ({"metadata": {"team": "gym"}, "ttl_s": 42}, 42),
    ],
)
async def test_owned_sandbox_gets_a_ttl_and_episode_labels(monkeypatch, sandbox_config, ttl_s):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock(), start=AsyncMock())
    monkeypatch.setattr(app, "AsyncSandbox", MagicMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: SimpleNamespace(aclose=AsyncMock()))
    instance = agent(sandbox_provider="sandbox", sandbox_config=sandbox_config)
    instance.server_client.global_config_dict.sandbox.default_metadata = {"site": "lab", "team": "x"}
    await instance._seed_agent_session_state(seed(borrowed=False))
    spec = remote.start.await_args.args[0]
    assert spec.ttl_s == ttl_s
    # Provider defaults, then sandbox_config, then the episode's labels.
    assert spec.metadata == {
        "site": "lab",
        "team": "gym",
        "gym-agent": "cline",
        "gym-session": "session",
        "gym-rollout": "rollout",
    }
    assert spec.workdir == "/app"


@pytest.mark.parametrize(
    "provider_name,model_host,options,reason",
    [
        ("docker", "10.0.0.5", {}, "OpenSandbox network-policy provider"),
        ("opensandbox", "127.0.0.1", {}, "sandbox-reachable host"),
        ("opensandbox", "0.0.0.0", {}, "sandbox-reachable host"),
        ("opensandbox", "localhost", {}, "sandbox-reachable host"),
        ("opensandbox", "10.0.0.5", {"network_policy": {"defaultAction": "allow"}}, "not both"),
    ],
)
def test_restricted_seed_refuses_unenforceable_policy_before_allocation(
    monkeypatch, provider_name, model_host, options, reason
):
    provider = SimpleNamespace(name=provider_name, aclose=AsyncMock())
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    constructor = MagicMock()
    constructor.connect = AsyncMock()
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    instance = agent(
        sandbox_provider="sandbox", network_access="model_only", sandbox_config={"provider_options": options}
    )
    instance.server_client.global_config_dict.policy_model.responses_api_models.vllm_model.host = model_host
    assert reason in check_agent_seed_rejected(instance.setup_webserver(), seed(borrowed=False))
    constructor.assert_not_called()
    constructor.connect.assert_not_awaited()
    provider.aclose.assert_awaited_once()


def test_borrowed_sandbox_refuses_network_changes_before_connect(monkeypatch):
    provider = MagicMock()
    monkeypatch.setattr(app, "create_provider", provider)
    instance = agent(network_access="model_only")
    assert "not a borrowed one" in check_agent_seed_rejected(instance.setup_webserver(), seed())
    provider.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("restricted", [False, True])
async def test_owned_network_policy_preserves_options_and_uses_offline_setup(monkeypatch, restricted):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), start=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr(app, "AsyncSandbox", MagicMock(return_value=remote))
    provider = SimpleNamespace(name="opensandbox", aclose=AsyncMock())
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    prepare, install = AsyncMock(), AsyncMock()
    monkeypatch.setattr(adapter.ClineSandboxSession, "prepare", prepare)
    monkeypatch.setattr(adapter.ClineSandboxSession, "install_runtime", install)
    options = {"platform": "linux/amd64"}
    if not restricted:
        options["network_policy"] = {"defaultAction": "allow"}
    instance = agent(
        sandbox_provider="sandbox",
        sandbox_config={"provider_options": options},
        network_access="model_only" if restricted else "inherit",
    )
    state = await instance._seed_agent_session_state(seed(borrowed=False))
    spec = remote.start.await_args.args[0]
    assert spec.provider_options == {
        **options,
        "network_policy": {"defaultAction": "deny", "egress": [{"action": "allow", "target": "10.0.0.5"}]}
        if restricted
        else {"defaultAction": "allow"},
    }
    assert spec.ttl_s == 2100 and spec.metadata["gym-rollout"] == "rollout"
    prepare.assert_awaited_once_with(timeout=600, allow_package_install=not restricted, create_workdir=restricted)
    install.assert_awaited_once_with(version="3.0.55", timeout=600, download_on_host=restricted)
    await instance._close_agent_session_state(state)
    remote.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_policy_refusal_survives_provider_cleanup_failure(monkeypatch):
    provider = SimpleNamespace(name="docker", aclose=AsyncMock(side_effect=RuntimeError("release failed")))
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    constructor = MagicMock()
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    with RequestFailureLog.capture() as log:
        with pytest.raises(app.AgentSeedRejectedError, match="OpenSandbox network-policy provider"):
            await agent(sandbox_provider="sandbox", network_access="model_only")._seed_agent_session_state(
                seed(borrowed=False)
            )
    constructor.assert_not_called()
    assert any(event.failure_kind == "session_release_failed" for event in log.report.events.values())


@pytest.mark.asyncio
async def test_restricted_setup_refusal_stops_owned_sandbox(monkeypatch):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), start=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr(app, "AsyncSandbox", MagicMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: SimpleNamespace(name="opensandbox", aclose=AsyncMock()))
    refused = app.AgentSeedRejectedError("Cline model_only requires bash in the sandbox image")
    monkeypatch.setattr(adapter.ClineSandboxSession, "prepare", AsyncMock(side_effect=refused))
    instance = agent(sandbox_provider="sandbox", network_access="model_only")
    with pytest.raises(app.AgentSeedRejectedError) as error:
        await instance._seed_agent_session_state(seed(borrowed=False))
    assert error.value is refused
    remote.start.assert_awaited_once()
    remote.stop.assert_awaited_once()


def test_restricted_seed_refuses_missing_mkdir_before_trying_to_create_workdir(monkeypatch, tmp_path):
    # Execute the actual prerequisite script with an image-like PATH that has no mkdir.
    commands = tmp_path / "bin"
    commands.mkdir()
    for name in ("bash", "uname", "getconf"):
        (commands / name).symlink_to(shutil.which(name))
    workdir = tmp_path / "task"

    async def execute(command, **kwargs):
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            env={"PATH": str(commands)},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
        return SandboxExecResult(stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode)

    remote = SimpleNamespace(
        exec=AsyncMock(side_effect=execute), start=AsyncMock(), stop=AsyncMock(), upload=AsyncMock()
    )
    monkeypatch.setattr(app, "AsyncSandbox", MagicMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: SimpleNamespace(name="opensandbox", aclose=AsyncMock()))
    instance = agent(sandbox_provider="sandbox", network_access="model_only", sandbox_config={"workdir": str(workdir)})
    reason = check_agent_seed_rejected(instance.setup_webserver(), seed(borrowed=False))
    assert "requires mkdir" in reason and "preinstall Bash and coreutils" in reason
    # Owned-session cleanup may execute after the prerequisite probe refuses setup.
    assert remote.exec.await_args_list[0].args[0].startswith(adapter._OFFLINE_PREREQUISITES)
    remote.upload.assert_not_awaited()
    remote.stop.assert_awaited_once()
    assert not workdir.exists()


@pytest.mark.asyncio
async def test_failed_install_releases_and_failed_release_retains_state(monkeypatch):
    remote = SimpleNamespace(exec=AsyncMock(side_effect=[result(), result("no network", code=6)]), upload=AsyncMock())
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: object())
    close = AsyncMock()
    monkeypatch.setattr(adapter.ClineSandboxSession, "close", close)
    instance = agent()
    with pytest.raises(RuntimeError, match=r"(?s)Cline sandbox installation failed .*no network"):
        await instance._seed_agent_session_state(seed())
    close.assert_awaited_once()

    remote.exec = AsyncMock(side_effect=[result(), result(code=1)])
    close.side_effect = RuntimeError("release failed")
    with pytest.raises(app.AgentSessionSetupError) as error:
        await instance._seed_agent_session_state(seed())
    # The unreleased state goes to close, and the installation error stays the cause.
    assert isinstance(error.value.state, adapter.ClineSandboxSession)
    assert error.value.state.session.sandbox is remote
    assert isinstance(error.value.error, RuntimeError)
    assert str(error.value.error).startswith("Cline sandbox installation failed (exit 1")


# Activation


@pytest.mark.asyncio
async def test_activation_stages_isolated_settings_and_maps_a_completed_run():
    instance = agent(system_prompt="Configured", thinking="low")
    session = FakeSession(artifacts=adapter.ClineArtifacts(stdout=COMPLETED, stderr=""), cleanup=cleanup())
    state = state_with(session)
    state.runtime_info = adapter.HarnessProcessInfo(hostname="sandbox", pid=7)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "system", "content": "System"}, {"role": "user", "content": "Question"}],
        metadata={"agent_timeout_sec": "30"},
    )

    response = await instance._sandbox_response(state, instance._validate_session_request(body))

    staged = state.stage_activation.await_args.kwargs
    command = staged["command"]
    assert command[0] == "/runtime/cline"
    assert command[-2:] == ["--", "Configured\n\nSystem\n\nQuestion"]
    conversation = next(record for record in state.observations.records if isinstance(record, AgentInvocation))
    assert [(item.role, item.content) for item in conversation.conversation[:1]] == [
        ("user", "Configured\n\nSystem\n\nQuestion")
    ]
    assert "--timeout" not in command
    assert command[command.index("--config") + 1] == f"{session.session_dir}/config"
    assert command[command.index("--data-dir") + 1] == f"{session.session_dir}/data"
    assert command[command.index("-m") + 1] == "policy-model"
    assert command[command.index("--thinking") + 1] == "low"
    settings = staged["providers"]["providers"]["openai-compatible"]["settings"]
    assert settings["baseUrl"] == "http://10.0.0.5:9000/ng-rollout/rollout/v1"
    assert settings["headers"] == {"x-session-id": "rollout"}
    assert staged["global_settings"] == {"telemetryOptOut": True, "autoUpdateEnabled": False}
    env = staged["env"]
    assert env["CLINE_DATA_DIR"] == f"{session.session_dir}/data"
    assert env["CLINE_NO_AUTO_UPDATE"] == "1"
    # Tool shells keep the task's HOME and XDG directories.
    assert not {"HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"} & set(env)
    assert session.timeout == 30

    assert response.status == "completed"
    assert response.output[-1].content[0].text == "Done."
    assert response.usage.input_tokens == 100 and response.usage.output_tokens == 7
    assert response.metadata["harness_execution"] == "sandbox"
    assert response.metadata["harness_hostname"] == "sandbox"
    records = state.observations.records
    invocation = next(record for record in records if isinstance(record, AgentInvocation))
    assert (invocation.invocation_id, invocation.status, invocation.duration_ms) == ("rollout", "completed", 1200)
    sandbox = next(record for record in records if isinstance(record, SandboxObservation))
    assert (sandbox.outcome, sandbox.sandbox_id, sandbox.exit_code) == ("completed", "sandbox-1", 0)
    tool_call = next(record for record in records if isinstance(record, ToolCallObservation))
    assert (tool_call.status, tool_call.sandbox_id) == ("completed", "sandbox-1")


# Stands in for the Cline binary: records what it was started with, then leaves a detached process holding its
# stdout open, as a tool that starts a background service does.
FAKE_CLINE = r"""
import json, os, subprocess, sys
seen = {
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "home": os.environ.get("HOME"),
    "cline_vars": sorted(name for name in os.environ if name.startswith("CLINE_")),
    "openai": sorted(name for name in os.environ if name.startswith("OPENAI_")),
    "providers": json.load(open(os.environ["CLINE_PROVIDER_SETTINGS_PATH"])),
    "stdin": sys.stdin.read(),
}
json.dump(seen, open(os.path.join(os.environ["PROBE_OUT"], "seen.json"), "w"))
child = subprocess.Popen(["sleep", "30"], start_new_session=True, stdout=sys.stdout, stderr=sys.stderr)
open(os.path.join(os.environ["PROBE_OUT"], "child.pid"), "w").write(str(child.pid))
print(json.dumps({"type": "agent_event", "event": {"type": "content_end", "contentType": "text", "text": "Done."}}))
print(json.dumps({"type": "run_result", "finishReason": "completed", "aggregateUsage": {"inputTokens": 3, "outputTokens": 2}}))
print("deprecation warning", file=sys.stderr)
"""


@pytest.mark.asyncio
async def test_staged_runner_isolates_cline_and_captures_its_output(tmp_path, monkeypatch):
    instance = agent(command_permissions={"allow": ["ls *"]})
    state, _, workdir = local_state(tmp_path)
    Path(state.runtime).mkdir()
    script = tmp_path / "fake_cline.py"
    script.write_text(FAKE_CLINE)
    binary = Path(state.executable)
    # A wrapper rather than a shebang naming the interpreter, whose path can exceed the kernel's shebang limit.
    binary.write_text(f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(script))} "$@"\n')
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    probe_out = tmp_path / "probe"
    probe_out.mkdir()
    # Variables an image can carry; none of them may reach Cline.
    monkeypatch.setenv("CLINE_DATA_DIR", "/root/.cline")
    monkeypatch.setenv("CLINE_IMAGE_SETTING", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "image-key")  # pragma: allowlist secret
    monkeypatch.setenv("OPENAI_BASE_URL", "http://image")
    monkeypatch.setenv("PROBE_OUT", str(probe_out))

    async def execute(*, stage_activation, collect, timeout, close_timeout):
        # The process supervisor needs Linux; run the staged runner command as it would.
        command = await stage_activation()
        process = await asyncio.create_subprocess_exec(
            *command.argv, cwd=str(workdir), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await asyncio.wait_for(process.communicate(), 20)
        state.session.cleanup = cleanup(code=process.returncode)
        state.session.artifacts = await collect()
        return state.session.artifacts

    state.session.execute = execute
    try:
        response = await asyncio.wait_for(
            instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Fix the bug")), 30
        )
    finally:
        if (probe_out / "child.pid").exists():
            try:
                os.kill(int((probe_out / "child.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass

    seen = json.loads((probe_out / "seen.json").read_text())
    session_dir = state.session.session_dir
    assert response.status == "completed"
    assert response.output[-1].content[0].text == "Done."
    assert (response.usage.input_tokens, response.usage.output_tokens) == (3, 2)
    assert Path(seen["cwd"]).resolve() == workdir.resolve()
    assert seen["home"] == os.environ["HOME"]
    assert seen["cline_vars"] == sorted([*app._cline_state_env(state.data_dir), "CLINE_COMMAND_PERMISSIONS"])
    assert seen["openai"] == []
    assert seen["stdin"] == ""
    assert seen["argv"][-2:] == ["--", "Fix the bug"]
    settings = seen["providers"]["providers"]["openai-compatible"]["settings"]
    assert settings["headers"] == {"x-session-id": "rollout"}
    assert settings["baseUrl"] == "http://10.0.0.5:9000/ng-rollout/rollout/v1"
    global_settings = json.loads(Path(f"{session_dir}/data/settings/global-settings.json").read_text())
    assert global_settings == {"telemetryOptOut": True, "autoUpdateEnabled": False}
    assert state.session.artifacts.stderr.strip() == "deprecation warning"
    assert state.runtime_info is not None and state.runtime_info.pid > 0
    assert not list(workdir.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("records", "receipt", "status", "failure_kind"),
    [
        # Cline's own error: a failed response the Environment Server classifies.
        (
            [error_event("Error", "Cannot connect to API: Unable to connect"), run_result("error")],
            cleanup(code=1),
            "failed",
            "model_transport_error",
        ),
        ([error_event("Error", "stub boom"), run_result("error")], cleanup(code=1), "failed", "unknown"),
        ([error_event("Error", "400 Invalid request"), run_result("error")], cleanup(code=1), "failed", "unknown"),
        ([error_event("Error", "429 Rate limited"), run_result("error")], cleanup(code=1), "failed", "unknown"),
        ([error_event("Error", "503 Unavailable"), run_result("error")], cleanup(code=1), "failed", "unknown"),
        # A context overflow and an abort leave gradable partial work.
        (
            [error_event("ContextWindowOverflowError", "too large", "context_window_exceeded"), run_result("error")],
            cleanup(code=1),
            "incomplete",
            None,
        ),
        (
            [
                error_event("Error", "Model reached the maximum output token limit before completing the turn"),
                run_result("error"),
            ],
            cleanup(code=1),
            "incomplete",
            None,
        ),
        # Cline 3.0.55 exits 0 on an abort too.
        ([text("partial"), run_result("aborted")], cleanup(code=0), "incomplete", None),
        # The supervisor stopped Cline at its deadline.
        ([text("partial")], cleanup(code=None, timed_out=True), "incomplete", None),
        ([], cleanup(code=None, timed_out=True), "failed", "unknown"),
        ([{"type": "run_start"}], cleanup(code=None, timed_out=True), "failed", "unknown"),
    ],
)
async def test_activation_status_follows_clines_exit_rule(records, receipt, status, failure_kind):
    instance = agent()
    session = FakeSession(artifacts=adapter.ClineArtifacts(stdout=stream(*records), stderr=""), cleanup=receipt)
    state = state_with(session)
    response = await instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert response.status == status
    outcome = (response.metadata or {}).get(AGENT_OUTCOME_METADATA_KEY)
    assert (json.loads(outcome)["failure_kind"] if outcome else None) == failure_kind
    if failure_kind == "unknown":
        assert json.loads(outcome).get("status_code") is None
        assert agent_outcome_policy(response) == "retry"
    invocation = next(record for record in state.observations.records if isinstance(record, AgentInvocation))
    assert invocation.status == status
    if receipt["timed_out"]:
        assert invocation.stop_reason == "timeout"
        if status == "failed":
            assert response.error.message == "Cline's deadline passed before any model output was recorded"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stdout", "code"),
    [
        ("", 0),
        ("", 125),
        ("", 127),
        # Only the final run_result is a result; a done event alone is not.
        (stream(text("Done."), done("completed")), 0),
    ],
)
async def test_run_without_a_result_fails_the_activation(stdout, code):
    instance = agent()
    session = FakeSession(
        artifacts=adapter.ClineArtifacts(stdout=stdout, stderr="cline: crashed"), cleanup=cleanup(code=code)
    )
    state = state_with(session)
    with pytest.raises(RuntimeError, match="Cline produced no run result .*crashed"):
        await instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert state.observations is not None


@pytest.mark.asyncio
async def test_interrupted_activation_keeps_observations_and_reraises():
    instance = agent()
    session = FakeSession(raises=RuntimeError("provider lost the sandbox"))
    state = state_with(session)
    with pytest.raises(RuntimeError, match="provider lost the sandbox"):
        await instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert "agent_artifact_unavailable" in [gap.code for gap in state.observations.gaps]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_original_event", [False, True])
async def test_provider_timeout_after_a_confirmed_deadline_is_still_a_timeout(with_original_event):
    instance = agent()
    failure = TimeoutError("exec wait expired")
    original = NgFailureEvent(
        boundary="sandbox", failure_kind="transport_timeout", failure_reason="provider wait", stage="agent"
    )
    if with_original_event:
        failure.__cause__ = original
    session = FakeSession(
        artifacts=adapter.ClineArtifacts(stdout=stream(text("partial")), stderr=""),
        cleanup=cleanup(code=None, timed_out=True),
        raises=failure,
    )
    state = state_with(session)
    with RequestFailureLog.capture() as activation_log:
        response = await instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert response.status == "incomplete"
    assert AGENT_OUTCOME_METADATA_KEY not in response.metadata
    sandbox = next(record for record in state.observations.records if isinstance(record, SandboxObservation))
    assert sandbox.outcome == "timeout"
    report = activation_log.reply_report(None)
    assert report.failure_id is None  # Retained evidence does not become the terminal episode outcome.
    wait = report.events[state.provider_wait_failure.event_id]
    assert (wait.boundary, wait.operation, wait.failure_kind) == ("sandbox", "exec_wait", "transport_timeout")
    assert wait.foreign_error.exception_type == "TimeoutError"
    assert wait.foreign_error.message == "exec wait expired"
    if with_original_event:
        assert report.events[original.event_id].stage == "agent"
        assert wait.cause_id == original.event_id
    # Close replays the same IDs even when the original activation waiter disconnected.
    with RequestFailureLog.capture() as close_log:
        await instance._close_agent_session_state(state)
    assert close_log.reply_report(None) == report


@pytest.mark.asyncio
async def test_retry_shares_one_activation_and_rejects_another():
    instance = agent()
    state = state_with(SimpleNamespace(closing=False))
    async with instance._locked_agent_session("session") as record:
        record.state = state
    response = MagicMock()
    response.model_copy.return_value = response
    execute = AsyncMock(return_value=response)
    instance._sandbox_response = execute
    body = NeMoGymResponseCreateParamsNonStreaming(input="Question")
    assert await instance.responses(request(), body) is response
    assert await instance.responses(request(), body) is response
    assert execute.await_count == 1
    for changed, rollout in ((NeMoGymResponseCreateParamsNonStreaming(input="Changed"), "rollout"), (body, "other")):
        with pytest.raises(HTTPException) as error:
            await instance.responses(request(rollout=rollout), changed)
        assert error.value.status_code == 409
    assert execute.await_count == 1


@pytest.mark.asyncio
async def test_closing_session_refuses_an_activation():
    instance = agent()
    session = FakeSession()
    session.closing = True
    state = state_with(session)
    async with instance._locked_agent_session("session") as record:
        record.state = state
    instance._sandbox_response = AsyncMock()
    with pytest.raises(HTTPException) as error:
        await instance.responses(request(), NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert error.value.status_code == 409
    instance._sandbox_response.assert_not_called()
    assert state.task is None


@pytest.mark.asyncio
async def test_unknown_session_never_falls_back_to_the_host():
    instance = agent()
    instance._run_cline = AsyncMock()
    with pytest.raises(HTTPException) as error:
        await instance.responses(request("unknown"), NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert error.value.status_code == 409
    instance._run_cline.assert_not_awaited()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"input": "Q", "temperature": 0.5}, "do not support request field temperature"),
        ({"input": "Q", "max_output_tokens": 10}, "cannot enforce request max_output_tokens"),
        ({"input": "Q", "metadata": {"other": "x"}}, "do not support request field metadata"),
        ({"input": "Q", "model": "another-model"}, "selected by the agent configuration"),
        (
            {"input": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
            "one text user prompt",
        ),
        (
            {
                "input": [
                    {"role": "user", "content": [{"type": "input_image", "image_url": "data:x", "detail": "auto"}]}
                ]
            },
            "only supports text input",
        ),
    ],
)
def test_session_request_rejects_controls_cline_cannot_apply(body, message):
    with pytest.raises(HTTPException, match=message) as error:
        agent()._validate_session_request(NeMoGymResponseCreateParamsNonStreaming(**body))
    assert error.value.status_code == 422


def test_session_request_accepts_the_policy_model_and_its_server_name():
    instance = agent()
    for model in ("policy-model", "policy_model", None):
        instance._validate_session_request(NeMoGymResponseCreateParamsNonStreaming(input="Q", model=model))


def test_session_request_takes_a_developer_message_as_the_system_message():
    instance = agent()
    body = instance._validate_session_request(
        NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "developer", "content": "Dev"}, {"role": "user", "content": "Q"}], metadata={}
        )
    )
    assert instance._session_prompt(body) == "Dev\n\nQ"


@pytest.mark.parametrize(
    "prompt",
    [
        "x " * (app.MAX_PROMPT_BYTES // 2 + 1),
        # Within the limit until Cline's trailing space pads the single word.
        "x" * app.MAX_PROMPT_BYTES,
    ],
)
def test_prompt_beyond_the_argv_limit_is_rejected(prompt):
    instance = agent()
    body = NeMoGymResponseCreateParamsNonStreaming(input=prompt)
    with pytest.raises(HTTPException, match="argv limit") as error:
        instance._session_prompt(body)
    assert error.value.status_code == 422


# Close


@pytest.mark.asyncio
async def test_close_without_activation_returns_a_gap_and_keeps_the_borrowed_sandbox():
    instance = agent()
    session = FakeSession()
    state = state_with(session)
    closed = await instance._close_agent_session_state(state)
    session.close.assert_awaited_once()
    assert [gap.code for gap in closed.agent_observations.gaps] == ["observation_capture_failed"]
    assert closed.resources_cookies is None


@pytest.mark.asyncio
async def test_close_returns_observations_of_a_finished_activation():
    instance = agent()
    session = FakeSession(artifacts=adapter.ClineArtifacts(stdout=COMPLETED, stderr=""), cleanup=cleanup())
    state = state_with(session)
    state.task = asyncio.create_task(
        instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    )
    await state.task
    closed = await instance._close_agent_session_state(state)
    assert closed.agent_observations is state.observations
    assert instance._rollout_sandbox_session(state) is session


@pytest.mark.asyncio
async def test_close_cancels_an_activation_still_staging():
    session = FakeSession()
    state = state_with(session)
    started = asyncio.Event()

    async def stuck():
        started.set()
        await asyncio.sleep(60)

    state.task = asyncio.create_task(stuck())
    await asyncio.wait_for(started.wait(), 1)
    await asyncio.wait_for(state.close(1), 5)
    assert state.task.cancelled()


# Rollout stop


@pytest.mark.asyncio
async def test_rollout_stop_before_launch_is_graded_as_a_stop(tmp_path):
    instance = agent()
    state, sandbox, _ = local_state(tmp_path)
    # What POST /rollouts/<id>/stop does to a seeded session whose Cline has not launched.
    assert await state.session.expire(timeout=5, reason="budget")
    response = await asyncio.wait_for(
        instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question")), 30
    )
    assert response.status == "incomplete"
    assert json.loads(response.metadata[ROLLOUT_STOP_METADATA_KEY]) == {"reason": "budget"}
    assert state.session.expired_before_launch
    await state.close(5)
    assert sandbox.disconnected


@pytest.mark.asyncio
async def test_rollout_stop_after_launch_marks_the_response():
    instance = agent()
    session = FakeSession(
        artifacts=adapter.ClineArtifacts(stdout=stream(text("partial")), stderr=""),
        cleanup=cleanup(code=None, timed_out=True),
    )
    session.expired, session.expire_reason = True, "operator"
    state = state_with(session)
    response = await instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert response.status == "incomplete"
    assert json.loads(response.metadata[ROLLOUT_STOP_METADATA_KEY]) == {"reason": "operator"}


@pytest.mark.asyncio
def stopped_state(sandbox, *, return_code):
    """A Cline session that a rollout stop ended, with the supervisor's receipt for that stop."""
    session = SandboxSession(
        sandbox=sandbox, session_dir="/tmp/nemo-gym-cline-sessions/id", workdir="/app", harness="Cline"
    )
    session._expire_requested = True
    session.cleanup = {"cleanup_confirmed": True, "return_code": return_code, "timed_out": True, "error": None}
    return adapter.ClineSandboxSession(request=seed(), session=session, runtime="/runtime")


@pytest.mark.asyncio
async def test_rollout_stop_before_launch_collects_nothing_from_the_sandbox():
    unreachable = AsyncMock(side_effect=ConnectionError("sandbox unreachable"))
    state = stopped_state(SimpleNamespace(exec=unreachable, download=unreachable), return_code=None)
    assert state.session.expired_before_launch
    assert await state.collect_artifacts() == adapter.ClineArtifacts(stdout="", stderr="")
    unreachable.assert_not_awaited()


@pytest.mark.asyncio
async def test_rollout_stop_before_cline_opened_its_output_keeps_an_empty_transcript():
    commands = []

    async def exec_(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(return_code=0, error_type=None, stdout="absent\n", stderr="")

    async def download(source, destination):
        raise FileNotFoundError(source)

    # The stop ended the runner after launch, before it redirected Cline's stdout.
    state = stopped_state(SimpleNamespace(exec=exec_, download=download), return_code=143)
    assert (await state.collect_artifacts()).stdout == ""
    assert "/tmp/nemo-gym-cline-sessions/id/stdout.jsonl" in commands[-1]
    # Without the stop, missing output is still a failure.
    state.session._expire_requested = False
    with pytest.raises(RuntimeError, match="no readable output"):
        await state.collect_artifacts()


# Host path and outcome classification


@pytest.mark.asyncio
async def test_sessionless_calls_keep_the_host_cli_path(monkeypatch):
    instance = agent()
    instance._run_cline = AsyncMock(return_value=([], {}, "policy-model"))
    plain = SimpleNamespace(path_params={}, url=SimpleNamespace(path=""))  # what harness_agent's runner passes
    response = await instance.responses(plain, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    instance._run_cline.assert_awaited_once_with("Question", None, None)
    assert response.output[0].type == "message"


@pytest.mark.asyncio
async def test_run_requires_resources_and_refuses_sessions():
    with pytest.raises(HTTPException) as error:
        await agent().run(request(), MagicMock())
    assert error.value.status_code == 409
    plain = Request({"type": "http", "session": {}, "path_params": {}})
    with pytest.raises(HTTPException) as error:
        await agent().run(plain, MagicMock())
    assert error.value.status_code == 422
    agent(resources_server=ResourcesServerRef(type="resources_servers", name="math"))  # the host config stays valid


@pytest.mark.asyncio
async def test_host_install_is_lazy_shared_and_retryable(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def install(version):
        entered.set()
        release.wait(timeout=5)

    installer = MagicMock(side_effect=install)
    monkeypatch.setattr(app, "ensure_cline", installer)
    instance = agent()
    installer.assert_not_called()
    waiters = [asyncio.create_task(instance._ensure_local_cline()) for _ in range(2)]
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        waiters[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiters[0]
        assert not instance._local_setup_task.cancelled()
    finally:
        release.set()
        await asyncio.gather(*waiters, return_exceptions=True)
    assert installer.call_count == 1
    instance = agent()
    installer.side_effect = RuntimeError("install failed")
    with pytest.raises(RuntimeError, match="install failed"):
        await instance._ensure_local_cline()
    installer.side_effect = None
    await instance._ensure_local_cline()
    assert installer.call_count == 3


@pytest.mark.parametrize(
    ("terminal", "kind"),
    [
        ({"error": {"name": "ContextWindowOverflowError", "message": "x"}}, "model_context_limit"),
        ({"error": {"name": "Error", "message": "x"}, "errorClass": "context_window_exceeded"}, "model_context_limit"),
        ({"error": {"name": "Error", "message": "The operation timed out."}}, "model_transport_error"),
        ({"error": {"name": "Error", "message": "Cannot connect to API: refused"}}, "model_transport_error"),
        ({"error": {"name": "Error", "message": "Model returned empty response"}}, "agent_invalid_output"),
        ({"error": {"name": "Error", "message": "Rate limit exceeded"}}, "unknown"),
        ({"error": {"name": "Error", "message": "400 Invalid request"}}, "unknown"),
        ({"error": {"name": "Error", "message": "429 Rate limit exceeded"}}, "unknown"),
        ({"error": {"name": "Error", "message": "503 Service unavailable"}}, "unknown"),
        ({}, "unknown"),
    ],
)
def test_classify_outcome_reads_clines_terminal_error(terminal, kind):
    assert agent().classify_outcome(terminal).failure_kind == kind


# Observations


def test_observations_time_tools_and_mark_failed_and_unfinished_calls():
    stdout = (
        stream(
            tool("content_start", "ok", toolName="run_commands"),
            tool("content_end", "ok", toolName="run_commands", output=[{"success": True}]),
            tool("content_start", "bad", toolName="run_commands"),
            tool("content_end", "bad", toolName="run_commands", output=[{"success": True}, {"success": False}]),
            tool("content_start", "err", toolName="editor"),
            tool("content_end", "err", toolName="editor", error="no such file"),
            tool("content_start", "plain", toolName="read_files"),
            tool("content_end", "plain", toolName="read_files", output="contents"),
            tool("content_start", "cut", toolName="run_commands"),
        )
        + "not json\n"
    )
    bundle = build_cline_observations(
        stdout, "rollout", status="incomplete", error_type=None, conversation=[], unfinished_tool_status="timeout"
    )
    tools = {record.tool_call_id: record for record in bundle.records if isinstance(record, ToolCallObservation)}
    assert {call: record.status for call, record in tools.items()} == {
        "ok": "completed",
        "bad": "failed",
        "err": "failed",
        "plain": "unknown",
        "cut": "timeout",
    }
    # The cut-off call gets an output that says it returned none, in the observed conversation only.
    invocation = next(record for record in bundle.records if isinstance(record, AgentInvocation))
    assert [(item.call_id, item.output) for item in invocation.conversation] == [
        ("cut", "[no output: Cline stopped before this tool call returned (timeout)]")
    ]
    assert tools["ok"].duration_ms == 1000 and tools["ok"].timing_source == "harness"
    gaps = [(gap.code, gap.detail) for gap in bundle.gaps]
    assert ("tool_outcome_unavailable", "plain") in gaps
    assert ("tool_timing_unavailable", "cut") in gaps
    assert ("subagent_hierarchy_unavailable", None) in gaps


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("records", "expected", "gap"),
    [
        # openai-compatible counts cache reads inside inputTokens; they are a detail, not an addition.
        (
            [
                run_result(
                    "completed", aggregateUsage={"inputTokens": 37432, "cacheReadTokens": 13152, "outputTokens": 9}
                )
            ],
            (37432, 13152, 9, None),
            "reasoning_token_usage_unavailable",
        ),
        # A run the deadline stopped has only running totals, which can lag answered calls: usage stays unknown.
        (
            [
                {
                    "type": "agent_event",
                    "event": {
                        "type": "usage",
                        "totalInputTokens": 50,
                        "totalCacheReadTokens": 0,
                        "totalOutputTokens": 4,
                    },
                }
            ],
            None,
            "model_call_usage_unavailable",
        ),
        ([text("no usage")], None, "model_call_usage_unavailable"),
    ],
)
async def test_session_usage_counts_cache_reads_once(records, expected, gap):
    instance = agent()
    session = FakeSession(
        artifacts=adapter.ClineArtifacts(stdout=stream(*records), stderr=""), cleanup=cleanup(timed_out=True)
    )
    state = state_with(session)
    response = await instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    usage = response.usage
    observed = (
        None
        if usage is None
        else (
            usage.input_tokens,
            usage.input_tokens_details.cached_tokens,
            usage.output_tokens,
            usage.output_tokens_details.reasoning_tokens,
        )
    )
    assert observed == expected
    assert gap in [item.code for item in state.observations.gaps]


@pytest.mark.asyncio
async def test_deadline_marks_the_running_tool_call_as_timed_out():
    instance = agent()
    records = [tool("content_start", "call-1", toolName="run_commands", input={"commands": ["sleep 999"]})]
    session = FakeSession(
        artifacts=adapter.ClineArtifacts(stdout=stream(*records), stderr=""),
        cleanup=cleanup(code=None, timed_out=True),
    )
    state = state_with(session)
    response = await instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert response.status == "incomplete"
    # The response keeps Cline's own items; only the observed conversation records the missing output.
    assert [item.type for item in response.output] == ["function_call"]
    tool_call = next(record for record in state.observations.records if isinstance(record, ToolCallObservation))
    assert (tool_call.tool_call_id, tool_call.status) == ("call-1", "timeout")


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("ensure_ascii", [False, True])
@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
async def test_jsonl_preserves_unicode_text_final_status_usage_and_tool_evidence(separator, ensure_ascii, line_ending):
    value = f"before{separator}after"
    records = [
        tool("content_start", "unicode-call", toolName="run_commands", input={"commands": [value]}),
        tool("content_end", "unicode-call", toolName="run_commands", output=value),
        text(value),
        run_result("completed", model={"id": value}, aggregateUsage={"inputTokens": 12, "outputTokens": 3}),
    ]
    stdout = (
        line_ending.join(
            json.dumps({"ts": f"2026-11-03T00:00:{i:02d}.000Z", **record}, ensure_ascii=ensure_ascii)
            for i, record in enumerate(records)
        )
        + line_ending
    )
    instance = agent()
    state = state_with(FakeSession(artifacts=adapter.ClineArtifacts(stdout=stdout, stderr=""), cleanup=cleanup()))
    response = await instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert response.status == "completed"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 15
    assert any(item.type == "message" and item.content[0].text == value for item in response.output)
    assert any(item.type == "function_call_output" and item.output == value for item in response.output)
    invocation = next(item for item in state.observations.records if isinstance(item, AgentInvocation))
    observed_tool = next(item for item in state.observations.records if isinstance(item, ToolCallObservation))
    assert invocation.stop_reason == "completed" and invocation.duration_ms == 1200
    assert observed_tool.tool_call_id == "unicode-call" and observed_tool.duration_ms == 1000


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_jsonl_preserves_terminal_error_with_unicode_separator(separator):
    record = error_event("ApiError", f"bad{separator}request", error_class="api")
    assert app._terminal_error(json.dumps(record, ensure_ascii=False) + "\n") == {
        "error": record["event"]["error"],
        "errorClass": "api",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("typed_cleanup", [False, True])
async def test_failed_connect_keeps_original_failure_when_provider_cleanup_also_fails(monkeypatch, typed_cleanup):
    original = NgFailureEvent(
        boundary="sandbox", failure_kind="transport_timeout", failure_reason="connect timed out", stage="agent"
    )
    cleanup = (
        NgFailureEvent(
            boundary="sandbox", failure_kind="session_release_failed", failure_reason="close failed", stage="cleanup"
        )
        if typed_cleanup
        else RuntimeError("close failed")
    )
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(side_effect=original))
    provider = SimpleNamespace(aclose=AsyncMock(side_effect=cleanup))
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    with RequestFailureLog.capture() as log:
        with pytest.raises(NgFailureEvent) as raised:
            await agent()._seed_agent_session_state(seed())
    assert raised.value is original
    report = log.reply_report(original)
    assert report.failure_id == original.event_id
    assert report.events[original.event_id].stage == "agent"
    releases = [event for event in report.events.values() if event.failure_kind == "session_release_failed"]
    assert len(releases) == 1 and releases[0].stage == "cleanup"
    if typed_cleanup:
        assert releases[0].event_id == cleanup.event_id
    provider.aclose.assert_awaited_once()


@pytest.mark.parametrize("workdir", [123, ["/app"], None])
def test_owned_workdir_type_is_a_final_refusal(monkeypatch, workdir):
    provider = MagicMock()
    monkeypatch.setattr(app, "create_provider", provider)
    instance = agent(sandbox_provider="sandbox", sandbox_config={"workdir": workdir})
    assert "workdir" in check_agent_seed_rejected(instance.setup_webserver(), seed(borrowed=False))
    provider.assert_not_called()


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("block", [{"unknown_provider": {}}, {"local": 17}])
def test_invalid_provider_blocks_are_final_refusals_before_connect(monkeypatch, borrowed, block):
    constructor = MagicMock()
    constructor.connect = AsyncMock()
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    instance = agent(sandbox_provider="sandbox")
    instance.server_client.global_config_dict = {"sandbox": block}
    check_agent_seed_rejected(instance.setup_webserver(), seed(borrowed=borrowed))
    constructor.assert_not_called()
    constructor.connect.assert_not_awaited()
