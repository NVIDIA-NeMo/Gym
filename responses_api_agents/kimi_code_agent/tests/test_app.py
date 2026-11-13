# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi Code agent sessions: config rendering, output conversion, seed refusals, and real supervised runs of a fake
Kimi CLI."""

import asyncio
import hashlib
import io
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    SandboxSession,
    check_agent_seed_rejected,
)
from nemo_gym.agents.base_responses_api_agent import ModelEndpoint
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.kimi_code_agent.app import KimiCodeAgent, KimiCodeAgentConfig
from responses_api_agents.kimi_code_agent.conversion import max_concurrency, parse_session, to_toml
from responses_api_agents.kimi_code_agent.sandbox import INSTALLER, KIMI_VERSION, KimiSession


FAKE_KIMI = r"""
import json, os, subprocess, sys, time
from pathlib import Path
if sys.argv[1:] == ["--version"]:
    print("__VERSION__")
    sys.exit(0)
home = Path(os.environ["KIMI_CODE_HOME"])
mode = os.environ.get("TEST_MODE", "normal")
agents_md = home / "AGENTS.md"
(home / "seen.json").write_text(json.dumps({
    "argv": sys.argv[1:], "cwd": os.getcwd(), "config": (home / "config.toml").read_text(),
    "agents_md": agents_md.read_text() if agents_md.exists() else None,
    "env": {k: v for k, v in os.environ.items() if k.startswith(("KIMI_", "OPENAI_"))},
}))
if mode == "hang":
    time.sleep(60)
subprocess.run(["bash", "-c", 'printf "%s\n%s\n" "$HOME" "$TASK_TEST" > proof.txt'], check=True)
def usage(tokens):
    return {"type": "usage.record", "usage": {"inputOther": tokens, "output": 1, "inputCacheRead": 2}}
wires = {"main": [usage(100), {"type": "subagent.spawned", "subagentId": "agent-0", "parentAgentId": "main",
                                "parentToolCallId": "call-1", "subagentName": "coder"},
                  {"type": "subagent.started", "subagentId": "agent-0", "time": 1000},
                  {"type": "subagent.completed", "subagentId": "agent-0", "time": 3000}, usage(200)],
         "agent-0": [usage(10)]}
for agent, events in wires.items():
    wire = home / "sessions" / "wd_0" / "session_1" / "agents" / agent / "wire.jsonl"
    wire.parent.mkdir(parents=True, exist_ok=True)
    wire.write_text("\n".join(json.dumps(e) for e in events))
# A kimi process the agent started itself has its own session, which must not be counted.
other = home / "sessions" / "wd_0" / "session_2" / "agents" / "main" / "wire.jsonl"
other.parent.mkdir(parents=True, exist_ok=True)
other.write_text(json.dumps(usage(999)))
messages = [
    {"role": "meta", "type": "system.version", "version": "__VERSION__"},
    {"role": "assistant", "tool_calls": [{"type": "function", "id": "call-1",
     "function": {"name": "Agent", "arguments": "{\"prompt\": \"fix\"}"}}]},
    {"role": "tool", "tool_call_id": "call-1", "content": "fixed"},
]
if mode != "silent":
    messages.append({"role": "assistant", "content": "<think>check</think>Done."})
for message in messages:
    print(json.dumps(message), flush=True)
if mode == "sleep":
    time.sleep(60)
causes = {"overflow": "context.overflow: Context overflow", "auth": "provider.auth_error: 401 Unauthorized"}
if mode in causes:
    sys.stderr.write("error: failed to run prompt: " + causes[mode] + "\n")
    sys.exit(1)
print(json.dumps({"role": "meta", "type": "session.resume_hint", "session_id": "session_1"}))
sys.exit(1 if mode == "crash" else 0)
""".replace("__VERSION__", KIMI_VERSION)


class LocalSandbox:
    """Run every sandbox command on this host, except the installer, which writes the fake Kimi."""

    def __init__(self):
        self.events = []
        self.disconnected = False

    async def upload(self, source, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    async def download(self, source, destination):
        self.events.append(f"read {Path(source).name}")
        shutil.copyfile(source, destination)

    async def exec(self, command, *, cwd=None, timeout_s=30, env=None):
        argv = shlex.split(command)
        if len(argv) > 1 and argv[1].endswith("/install_kimi_runtime.sh"):
            assert argv[3] == KIMI_VERSION
            binary = Path(argv[2]) / "kimi"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text(f"#!{sys.executable}\n" + FAKE_KIMI)
            binary.chmod(0o755)
            return SandboxExecResult("", "", 0)
        process = await asyncio.create_subprocess_exec(
            "sh", "-c", command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        return SandboxExecResult(stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode)

    async def disconnect(self):
        self.events.append("disconnect")
        self.disconnected = True

    async def stop(self):
        pytest.fail("A borrowed sandbox must not be stopped")


def agent(**kwargs) -> KimiCodeAgent:
    return KimiCodeAgent(
        config=KimiCodeAgentConfig(
            host="localhost",
            port=8080,
            name="kimi_code_agent",
            entrypoint="app.py",
            model="policy",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            **kwargs,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def seed(*, workdir="/repo", tools=None, user=None, borrowed=True) -> AgentSeedSessionRequest:
    access = {
        "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"id": "task"}},
        "workdir": workdir,
    }
    if user is not None:
        access["user"] = user
    return AgentSeedSessionRequest(
        agent_session_id="session-1",
        episode_id={"rollout_id": "rollout-1"},
        task_id={"taskset": "test", "task_id": "task"},
        tool_accesses=tools or [],
        sandbox_access=access if borrowed else None,
    )


def body(**fields) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Fix the bug."}], **fields
    )


def test_to_toml_round_trips_nested_tables_rule_arrays_and_unicode():
    data = {
        "default_model": "policy",
        "models": {"a.b": {"capabilities": ["tool_use"], "max_context_size": 8}},
        "permission": {"rules": [{"decision": "deny", "pattern": "Bash(echo \U0001f600)"}]},
        "loop_control": {"max_steps_per_turn": None},
        "note": 'quote " and \x7f',
    }
    assert tomllib.loads(to_toml(data)) == data | {"loop_control": {}}
    with pytest.raises(ValueError, match="disabled holds None"):
        to_toml({"tools": {"disabled": ["Shell", None]}})


def test_unknown_subagent_events_do_not_end_a_child(tmp_path):
    wire = tmp_path / "agents" / "main" / "wire.jsonl"
    wire.parent.mkdir(parents=True)
    events = [
        {"type": "subagent.spawned", "subagentId": "agent-0", "parentAgentId": "main"},
        {"type": "subagent.started", "subagentId": "agent-0", "time": 1000},
        {"type": "subagent.progress", "subagentId": "agent-0", "time": 1500},
    ]
    wire.write_text("\n".join(json.dumps(e) for e in events))
    _, children = parse_session(tmp_path)
    assert children[0]["status"] == "incomplete" and "ended" not in children[0]
    # A cancelled child is incomplete, and an all-zero usage record leaves usage unknown.
    events += [
        {"type": "subagent.cancelled", "subagentId": "agent-0", "time": 2000},
        {"type": "usage.record", "usage": {"inputOther": 0, "output": 0}},
    ]
    wire.write_text("\n".join(json.dumps(e) for e in events))
    usage, children = parse_session(tmp_path)
    assert usage is None and children[0]["status"] == "incomplete" and children[0]["ended"] == 2000
    assert (
        max_concurrency([{"started": 0, "ended": 10}, {"started": 5, "ended": 6}, {"started": 6, "ended": 7}, {}]) == 2
    )


def test_config_routes_every_model_through_the_rollout_and_keeps_its_providers():
    server = agent(
        subagent_model_server={"type": "responses_api_models", "name": "lightning"},
        subagent_model_force=True,
        subagents=False,
        kimi_config={
            "providers": {"policy": {"base_url": "http://elsewhere"}},
            "extra": {"x": 1},
            "loop_control": {"max_steps_per_turn": 5},
        },
    )
    state = MagicMock(request=seed())

    def endpoint(*, model_server, rollout_id):
        return ModelEndpoint(base_url=f"http://{model_server.name}/ng-rollout/{rollout_id}/v1")

    with patch.object(KimiCodeAgent, "model_endpoint", side_effect=endpoint):
        config = tomllib.loads(to_toml(server.kimi_config(state)))
    assert config["providers"]["policy"]["base_url"] == "http://policy_model/ng-rollout/rollout-1/v1"
    assert config["providers"]["subagent"]["base_url"] == "http://lightning/ng-rollout/rollout-1/v1"
    assert config["secondary_model"] == {"default_model": "subagent", "force": True}
    # Web tools stay off by default.
    assert {"Agent", "AgentSwarm", "WebSearch", "FetchURL"} <= set(config["tools"]["disabled"])
    assert config["extra"] == {"x": 1}
    # Unset typed fields leave the user's kimi_config alone.
    assert config["loop_control"] == {"max_steps_per_turn": 5}
    assert config["providers"]["subagent"]["custom_headers"] == {"x-session-id": "rollout-1"}


def test_token_id_capture_refuses_a_subagent_server():
    server = agent(subagent_model_server={"type": "responses_api_models", "name": "lightning"}, token_id_capture=True)
    server.server_client.global_config_dict = {"token_id_capture": {"enabled": True}}
    # Capture would record the subagent server's calls under the policy rollout.
    reason = "Kimi Code token capture would record subagent_model_server calls as policy"
    assert check_agent_seed_rejected(server.setup_webserver(), seed()) == reason


def test_subagent_settings_require_a_subagent_server():
    with pytest.raises(ValueError, match="require subagent_model_server"):
        agent(subagent_model="child")


@pytest.mark.parametrize("capture", [True, False])
def test_token_id_capture_routes_root_calls_to_capture(capture):
    server = agent(token_id_capture=capture)
    server.server_client.global_config_dict = {"token_id_capture": {"enabled": True}}
    server.server_client._build_server_base_url.side_effect = lambda config: f"http://{config['name']}:9000"
    with patch(
        "nemo_gym.agents.base_responses_api_agent.get_first_server_config_dict",
        side_effect=lambda _, name: {"name": name},
    ):
        providers = server.kimi_config(MagicMock(request=seed()))["providers"]
    capture_path = "/training-token-capture" if capture else ""
    assert providers["policy"]["base_url"] == f"http://policy_model:9000/ng-rollout/rollout-1{capture_path}/v1"
    assert "subagent" not in providers


@pytest.mark.parametrize(
    ("request_seed", "reason"),
    [
        (
            seed(
                tools=[
                    {
                        "kind": "mcp",
                        "name": "task_tools",
                        "required": True,
                        "connection": {"transport": "streamable_http", "url": "http://r/mcp"},
                    }
                ]
            ),
            "Kimi Code sessions use Kimi's own tools; required tool grants are unsupported",
        ),
        (seed(user="nobody"), "Kimi Code does not support a SandboxAccess user override"),
        (seed(workdir="repo"), "Kimi Code task workdir must be absolute and separate from its session storage"),
        (seed(borrowed=False), "Kimi Code requires sandbox_access or a configured sandbox_provider"),
    ],
)
def test_unsupported_seeds_are_final_refusals(request_seed, reason):
    server = agent()
    assert check_agent_seed_rejected(server.setup_webserver(), request_seed) == reason
    assert not server._session_records


@pytest.mark.parametrize(
    ("network_access", "borrowed", "reason"),
    [
        ("model_only", True, "Kimi Code network_access applies only to a sandbox it creates, not a borrowed one"),
        ("model_and_tools", False, "Kimi Code uses no tool grants; set network_access to model_only or inherit"),
    ],
)
def test_network_access_refusals(network_access, borrowed, reason):
    server = agent(network_access=network_access, sandbox_provider="sandbox")
    assert check_agent_seed_rejected(server.setup_webserver(), seed(borrowed=borrowed)) == reason


@pytest.fixture
def owned_sandbox(monkeypatch):
    import responses_api_agents.kimi_code_agent.app as module

    remote = MagicMock(
        sandbox_id="owned", start=AsyncMock(), exec=AsyncMock(return_value=SandboxExecResult("", "", 0))
    )
    monkeypatch.setattr(module, "AsyncSandbox", MagicMock(return_value=remote))
    monkeypatch.setattr(
        module, "create_provider", lambda config: SimpleNamespace(name="opensandbox", aclose=AsyncMock())
    )
    monkeypatch.setattr(module, "resolve_provider_config", lambda *args: {})
    monkeypatch.setattr(module, "resolve_provider_metadata", lambda *args: {})
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(KimiSession, "install", AsyncMock())
    hosts = {"policy_model": "model.example", "lightning": "model.example", "elsewhere": "other.example"}
    monkeypatch.setattr(
        KimiCodeAgent, "resolve_model_base_url", lambda self, name, **kwargs: f"http://{hosts[name]}:9000/v1"
    )
    return remote


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("network_access", "subagent_server", "policy"),
    [
        ("inherit", "elsewhere", None),
        ("model_only", None, {"defaultAction": "deny", "egress": [{"action": "allow", "target": "model.example"}]}),
        # A subagent server on the same host as the root's needs no extra rule.
        (
            "model_only",
            "lightning",
            {"defaultAction": "deny", "egress": [{"action": "allow", "target": "model.example"}]},
        ),
    ],
)
async def test_owned_sandbox_egress_allows_only_the_model_host(owned_sandbox, network_access, subagent_server, policy):
    subagent = (
        {"subagent_model_server": {"type": "responses_api_models", "name": subagent_server}, "subagent_model": "child"}
        if subagent_server
        else {}
    )
    server = agent(network_access=network_access, sandbox_provider="sandbox", **subagent)
    await server._seed_agent_session_state(seed(borrowed=False))
    assert owned_sandbox.start.await_args.args[0].provider_options.get("network_policy") == policy


@pytest.mark.asyncio
async def test_model_only_refuses_a_subagent_server_on_another_host(owned_sandbox):
    subagent = {"type": "responses_api_models", "name": "elsewhere"}
    server = agent(
        network_access="model_only", sandbox_provider="sandbox", subagent_model="child", subagent_model_server=subagent
    )
    with pytest.raises(AgentSeedRejectedError, match="subagent_model_server is on another"):
        await server._seed_agent_session_state(seed(borrowed=False))
    owned_sandbox.start.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("uname", "installer", "error", "message"),
    [
        # A failed probe is a retryable setup error, not a refusal of the episode.
        (SandboxExecResult("", "", None, error_type="timeout"), None, RuntimeError, "Cannot probe"),
        (SandboxExecResult("riscv64\n", "", 0), None, AgentSeedRejectedError, "architecture 'riscv64'"),
        (SandboxExecResult("x86_64\n", "", 0), SandboxExecResult("", "", 3), AgentSeedRejectedError, "glibc"),
    ],
)
async def test_install_refuses_unsupported_images_and_retries_probe_failures(
    tmp_path, uname, installer, error, message
):
    ok = SandboxExecResult("", "", 0)
    sandbox = MagicMock(upload=AsyncMock(), exec=AsyncMock(side_effect=[ok, uname, installer]))
    state = KimiSession(
        request=seed(),
        provider_name="local",
        sandbox_id="task",
        session=SandboxSession(sandbox=sandbox, session_dir=str(tmp_path), workdir="/repo", harness="Kimi Code"),
    )
    with patch(
        "responses_api_agents.miniswe_sandboxed_agent.bootstrap.cached_download", AsyncMock(return_value=tmp_path)
    ):
        with pytest.raises(error, match=message):
            await state.install(timeout=10)


@pytest.mark.skipif(sys.platform != "linux", reason="The installer needs a glibc host")
def test_installer_installs_the_checked_binary_and_reuses_it(tmp_path):
    binary = tmp_path / "kimi-code-linux-x64"
    binary.write_text(f"#!/bin/sh\necho {KIMI_VERSION}\n")
    sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    runtime = tmp_path / "runtime"
    installer = Path(__file__).parents[1] / INSTALLER
    for _ in range(2):  # The second run reuses the installed binary.
        command = ["bash", installer, runtime, KIMI_VERSION, "linux-x64", sha256, binary]
        subprocess.run(command, check=True, capture_output=True)
    assert (runtime / "kimi").read_bytes() == binary.read_bytes()
    bad = ["bash", installer, tmp_path / "other", KIMI_VERSION, "linux-x64", "0" * 64, binary]
    assert subprocess.run(bad, capture_output=True).returncode != 0


def test_session_usage_without_a_resume_hint_reads_a_single_session(tmp_path):
    wire = tmp_path / "sessions" / "wd_0" / "session_1" / "agents" / "main" / "wire.jsonl"
    wire.parent.mkdir(parents=True)
    wire.write_text(json.dumps({"type": "usage.record", "usage": {"inputOther": 5, "output": 1}}))
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        tar.add(tmp_path / "sessions", arcname="sessions")
    usage, _ = KimiCodeAgent._read_sessions(archive.getvalue(), None)
    assert (usage.input_tokens, usage.output_tokens) == (5, 1)


def test_request_mapping_and_refusals():
    prompt, instructions = agent()._conversation(body(instructions="Answer in English."))
    assert (prompt, instructions) == ("Fix the bug.", "Answer in English.\n\nUse tools.")
    for fields in ({"temperature": 0.5}, {"max_output_tokens": 10}, {"model": "other"}):
        with pytest.raises(HTTPException) as error:
            agent()._conversation(body(**fields))
        assert error.value.status_code == 422
    with pytest.raises(HTTPException, match="byte argument limit"):
        agent()._conversation(NeMoGymResponseCreateParamsNonStreaming(input="x" * (128 * 1024)))


async def activate(tmp_path: Path, mode: str, *, timeout: float = 20, stop_before_launch: bool = False):
    workdir = tmp_path / "task repo"
    workdir.mkdir()
    request = seed(workdir=str(workdir))
    request.sandbox_access.env = {
        "TEST_MODE": mode,
        "TASK_TEST": "task-value",
        "HOME": str(tmp_path / "task-home"),
        "KIMI_CODE_HOME": "must-not-inherit",
    }
    sandbox = LocalSandbox()
    state = KimiSession(
        request=request,
        provider_name="local",
        sandbox_id="task",
        runtime=str(tmp_path / "runtime"),
        session=SandboxSession(
            sandbox=sandbox, session_dir=str(tmp_path / "kimi session"), workdir=str(workdir), harness="Kimi Code"
        ),
    )
    server = agent(timeout=timeout, swarm_max_concurrency=4)
    binary = tmp_path / "release"
    binary.write_bytes(b"release")
    # The agent server downloads the release; the fake installer stands in for its contents.
    with patch(
        "responses_api_agents.miniswe_sandboxed_agent.bootstrap.cached_download", AsyncMock(return_value=binary)
    ) as download:
        await state.install(timeout=10)
    assert download.await_args.args[0].startswith("https://code.kimi.com/kimi-code/binaries/2.1.1/kimi-code-linux-")
    state.activation_request = body()
    if stop_before_launch:
        assert await state.session.expire(timeout=10)
    endpoint = ModelEndpoint(base_url="http://model.invalid/ng-rollout/rollout-1/v1")
    with patch.object(KimiCodeAgent, "model_endpoint", return_value=endpoint):
        try:
            response = await asyncio.wait_for(server._activate(state, state.activation_request), timeout=60)
        except Exception as error:
            response = error
    seen_file = Path(state.home) / "seen.json"
    seen = json.loads(seen_file.read_text()) if seen_file.exists() else None
    close = await asyncio.wait_for(server._close_agent_session_state(state), timeout=60)
    return response, close, seen, state, sandbox


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_real_run_maps_output_usage_and_subagents(tmp_path):
    response, close, seen, state, sandbox = await activate(tmp_path, "normal")
    assert response.status == "completed"
    assert [item.type for item in response.output] == ["function_call", "function_call_output", "reasoning", "message"]
    assert response.output[-1].content[0].text == "Done."
    # Usage counts this run's session (root and child), not another kimi process's.
    assert (response.usage.input_tokens, response.usage.output_tokens) == (102 + 202 + 12, 3)
    assert response.metadata["kimi_num_subagents"] == "1"
    assert seen["argv"] == ["-p", "Fix the bug.", "--output-format", "stream-json"]
    assert seen["cwd"] == state.session.workdir
    assert seen["agents_md"] == "Use tools."
    assert tomllib.loads(seen["config"])["providers"]["policy"]["base_url"].endswith("/ng-rollout/rollout-1/v1")
    assert seen["env"]["KIMI_CODE_HOME"] == state.home
    assert seen["env"]["KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY"] == "4"
    assert seen["env"]["KIMI_CODE_CACHE_DIR"] == f"{state.runtime}/cache"
    assert (Path(state.session.workdir) / "proof.txt").read_text().splitlines() == [
        str(tmp_path / "task-home"),
        "task-value",
    ]
    records = close.agent_observations.records
    assert records[1].parent_invocation_id == "rollout-1" and records[1].status == "completed"
    gaps = {gap.code for gap in close.agent_observations.gaps}
    # Child calls carry the root's session id, so their ownership is unknown.
    assert "usage_unavailable" not in gaps and "model_call_ownership_unavailable" in gaps
    assert close.cleanup_confirmed and sandbox.disconnected and not Path(state.session.session_dir).exists()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
@pytest.mark.parametrize(
    ("mode", "status", "failure_kind"),
    [
        ("crash", "failed", "unknown"),
        ("silent", "failed", "agent_invalid_output"),
        ("auth", "failed", "model_request_error"),
    ],
)
async def test_failed_runs_report_an_outcome(tmp_path, mode, status, failure_kind):
    response, close, _, _, _ = await activate(tmp_path, mode)
    assert response.status == status
    outcome = json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY])
    assert outcome["failure_kind"] == failure_kind and outcome.get("status_code") == (401 if mode == "auth" else None)
    assert close.cleanup_confirmed


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
@pytest.mark.parametrize("mode", ["sleep", "hang", "overflow"])
async def test_deadlines_and_native_limits_are_incomplete(tmp_path, mode):
    response, close, _, state, _ = await activate(tmp_path, mode, timeout=3)
    assert response.status == "incomplete" and AGENT_OUTCOME_METADATA_KEY not in response.metadata
    # A deadline before Kimi printed anything is still graded.
    assert response.output[-1].content[0].text == "Done." if mode != "hang" else response.output == []
    assert state.session.cleanup["timed_out"] == (mode != "overflow")


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_rollout_stop_before_launch_is_incomplete_without_running_kimi(tmp_path):
    response, close, seen, state, _ = await activate(tmp_path, "normal", stop_before_launch=True)
    assert response.status == "incomplete" and response.output == [] and seen is None
    assert state.session.expired_before_launch and close.cleanup_confirmed
