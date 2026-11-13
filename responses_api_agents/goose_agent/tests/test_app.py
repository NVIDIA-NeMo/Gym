# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goose agent sessions: seed refusals, request mapping, and real supervised runs of a fake Goose CLI."""

import asyncio
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentSeedSessionRequest,
    SandboxSession,
    check_agent_seed_rejected,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.goose_agent.app import GooseAgent, GooseAgentConfig
from responses_api_agents.goose_agent.sandbox import GOOSE_VERSION, GooseSession


FAKE_GOOSE = r"""
import json, os, sqlite3, subprocess, sys, time
from pathlib import Path
if sys.argv[1:] == ["--version"]:
    print(" __VERSION__")
    sys.exit(0)
mode = os.environ.get("TEST_MODE", "normal")
root = Path(os.environ["GOOSE_PATH_ROOT"])
prompt = sys.stdin.read()
if mode == "hang":
    time.sleep(60)  # No session store or output before the deadline.
system = next((a[len("--system="):] for a in sys.argv if a.startswith("--system=")), "")
(root / "seen.json").write_text(json.dumps({
    "argv": sys.argv[1:], "prompt": prompt, "system": system, "cwd": os.getcwd(),
    "config": (root / "config" / "config.yaml").read_text(),
    "env": {k: v for k, v in os.environ.items() if k.startswith(("GOOSE_", "OPENAI_")) or k == "SSL_CERT_FILE"},
    "ca_certificates": len(__import__("ssl").create_default_context(cafile=os.environ["SSL_CERT_FILE"]).get_ca_certs()),
}))
# A shell tool call inherits Goose's environment, which keeps the task's HOME.
subprocess.run(["bash", "-c", 'printf "%s\n%s\n" "$HOME" "$TASK_TEST" > proof.txt'], check=True)
db = root / "data" / "sessions" / "sessions.db"
db.parent.mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(db)
con.execute("create table sessions (id text primary key, parent_session_id text, created_at timestamp)")
con.execute("create table messages (id integer primary key autoincrement, message_id text, session_id text, role text, "
            "content_json text, created_timestamp integer, metadata_json text)")
con.execute("create table usage_ledger (session_id text, input_tokens integer, output_tokens integer, "
            "cache_read_tokens integer)")
con.execute("insert into sessions values ('s1', null, '2026-11-03 00:00:00')")
call = {"type": "toolRequest", "id": "c1", "toolCall": {"status": "success",
        "value": {"name": "shell", "arguments": {"command": "ls"}}}}
out = {"type": "toolResponse", "id": "c1", "toolResult": {"status": "success",
       "value": {"content": [{"type": "text", "text": "a.txt"}], "isError": False}}}
rows = [("m0", "user", [{"type": "text", "text": prompt}]), ("m1", "assistant", [call]), ("m2", "user", [out]),
        ("m3", "assistant", [{"type": "text", "text": "Done."}])]
con.executemany("insert into messages (message_id, session_id, role, content_json, created_timestamp) "
                "values (?, 's1', ?, ?, 1)", [(i, r, json.dumps(c)) for i, r, c in rows])
con.commit()
con.close()
print("    goose is ready")
print(json.dumps({"type": "message", "message": {"id": "m3", "role": "assistant",
                  "content": [{"type": "text", "text": "Done."}]}}), flush=True)
notices = {"max_turns": "I've reached the maximum number of actions I can do without user input.",
           "model_server": "Ran into this error: Server error: Server error (500 Internal Server Error) at x: boom."}
if mode in notices:
    print(json.dumps({"type": "message", "message": {"id": "notice", "role": "assistant",
                      "content": [{"type": "text", "text": notices[mode]}]}}))
if mode == "sleep":
    time.sleep(60)
print(json.dumps({"type": "complete"}))
sys.exit({"crash": 3, "killed": 137}.get(mode, 0))
""".replace("__VERSION__", GOOSE_VERSION)


class LocalSandbox:
    """Run every sandbox command on this host, except the installer, which writes the fake Goose."""

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
        if len(argv) > 1 and argv[1].endswith("/install_goose_runtime.sh"):
            assert argv[3:6] == [GOOSE_VERSION, argv[4], argv[5]]
            binary = Path(argv[2]) / "goose"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text(f"#!{sys.executable}\n" + FAKE_GOOSE)
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


def agent(**kwargs) -> GooseAgent:
    return GooseAgent(
        config=GooseAgentConfig(
            host="localhost",
            port=8080,
            name="goose_agent",
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
            "Goose sessions use Goose's own tools; required tool grants are unsupported",
        ),
        (seed(user="nobody"), "Goose does not support a SandboxAccess user override"),
        (seed(workdir="repo"), "Goose task workdir must be absolute and separate from its session storage"),
        (seed(workdir="/tmp"), "Goose task workdir must be absolute and separate from its session storage"),
        (
            seed(workdir="/tmp/nemo-gym-goose-sessions/x"),
            "Goose task workdir must be absolute and separate from its session storage",
        ),
        (seed(borrowed=False), "Goose requires sandbox_access or a configured sandbox_provider"),
    ],
)
def test_unsupported_seeds_are_final_refusals(request_seed, reason):
    server = agent()
    assert check_agent_seed_rejected(server.setup_webserver(), request_seed) == reason
    assert not server._session_records


@pytest.mark.parametrize(
    ("network_access", "borrowed", "reason"),
    [
        ("model_only", True, "Goose network_access applies only to a sandbox it creates, not a borrowed one"),
        ("model_and_tools", False, "Goose uses no tool grants; set network_access to model_only or inherit"),
    ],
)
def test_network_access_refusals(network_access, borrowed, reason):
    server = agent(network_access=network_access, sandbox_provider="sandbox")
    assert check_agent_seed_rejected(server.setup_webserver(), seed(borrowed=borrowed)) == reason


@pytest.mark.asyncio
@pytest.mark.parametrize("network_access", ["inherit", "model_only"])
async def test_owned_sandbox_egress_allows_only_the_model_server(monkeypatch, network_access):
    import responses_api_agents.goose_agent.app as module

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
    monkeypatch.setattr(GooseSession, "install", AsyncMock())
    server = agent(network_access=network_access, sandbox_provider="sandbox")
    monkeypatch.setattr(GooseAgent, "resolve_model_base_url", lambda *args, **kwargs: "http://model.example:9000/v1")
    await server._seed_agent_session_state(seed(borrowed=False))
    policy = remote.start.await_args.args[0].provider_options.get("network_policy")
    if network_access == "inherit":
        assert policy is None
    else:
        assert policy == {"defaultAction": "deny", "egress": [{"action": "allow", "target": "model.example"}]}


def test_request_mapping_and_refusals():
    server = agent(system_prompt="Be brief.")
    prompt, system = server._conversation(body(instructions="Answer in English.", temperature=0.6))
    assert prompt == "Fix the bug."
    assert system == "Be brief.\n\nAnswer in English.\n\nUse tools."
    for fields, detail in (
        ({"max_output_tokens": 10}, "max_output_tokens"),
        ({"top_p": 0.5}, "top_p"),
        ({"model": "other"}, "model"),
    ):
        with pytest.raises(HTTPException) as error:
            server._conversation(body(**fields))
        assert error.value.status_code == 422 and detail in error.value.detail
    with pytest.raises(HTTPException, match="one user message"):
        server._conversation(NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "assistant", "content": "x"}]))
    with pytest.raises(HTTPException, match="byte argument limit"):
        agent(system_prompt="x" * (128 * 1024))._conversation(body())


def test_sessionless_requests_and_run_are_refused():
    server = agent()
    request = Request({"type": "http", "session": {}, "path_params": {}})
    with pytest.raises(HTTPException) as error:
        asyncio.run(server.responses(request, body()))
    assert error.value.status_code == 422
    with pytest.raises(HTTPException) as error:
        asyncio.run(server.run())
    assert error.value.status_code == 422


def session(tmp_path: Path, mode: str) -> tuple[GooseSession, LocalSandbox]:
    workdir = tmp_path / "task repo"
    workdir.mkdir()
    request = seed(workdir=str(workdir))
    request.sandbox_access.env = {
        "TEST_MODE": mode,
        "TASK_TEST": "task-value",
        "HOME": str(tmp_path / "task-home"),
        "OPENAI_BASE_URL": "http://must-not-inherit",
        "GOOSE_PROVIDER": "must-not-inherit",
    }
    sandbox = LocalSandbox()
    state = GooseSession(
        request=request,
        provider_name="local",
        sandbox_id="task",
        runtime=str(tmp_path / "runtime"),
        session=SandboxSession(
            sandbox=sandbox, session_dir=str(tmp_path / "goose session"), workdir=str(workdir), harness="Goose"
        ),
    )
    return state, sandbox


async def activate(tmp_path: Path, mode: str, *, timeout: float = 20):
    server = agent(timeout=timeout, delegate=False, context_limit=262144)
    state, sandbox = session(tmp_path, mode)
    archive = tmp_path / "release.tar.gz"
    archive.write_bytes(b"release")
    # The agent server downloads the release; the fake installer stands in for its contents.
    with patch(
        "responses_api_agents.miniswe_sandboxed_agent.bootstrap.cached_download", AsyncMock(return_value=archive)
    ) as download:
        await state.install(timeout=10)
    assert download.await_args.args[0].endswith(f"/goose-{os.uname().machine}-unknown-linux-musl.tar.gz")
    request = body(temperature=0.6)
    state.activation_request = request
    endpoint = SimpleNamespace(base_url="http://model.invalid/ng-rollout/rollout-1/v1")
    with patch.object(GooseAgent, "model_endpoint", return_value=endpoint):
        try:
            response = await asyncio.wait_for(server._activate(state, request), timeout=60)
        except Exception as error:
            response = error
    seen_path = Path(state.root) / "seen.json"
    seen = json.loads(seen_path.read_text()) if seen_path.exists() else None
    close = await asyncio.wait_for(server._close_agent_session_state(state), timeout=60)
    return response, close, seen, state, sandbox


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_real_run_maps_goose_output_and_keeps_the_task_environment(tmp_path):
    response, close, seen, state, sandbox = await activate(tmp_path, "normal")
    assert response.status == "completed"
    assert [item.type for item in response.output] == ["function_call", "function_call_output", "message"]
    assert response.output[-1].content[0].text == "Done."
    assert response.usage is None
    assert response.metadata["goose_exit_code"] == "0"
    assert seen["prompt"] == "Fix the bug."
    assert seen["system"] == "Use tools."
    assert seen["cwd"] == state.session.workdir
    assert seen["argv"][:3] == ["run", "--name", "gym"]
    assert json.loads(seen["config"]) == {
        "extensions": {"summon": {"enabled": False}, "extensionmanager": {"enabled": False}}
    }
    assert seen["env"]["OPENAI_BASE_URL"] == "http://model.invalid/ng-rollout/rollout-1/v1"
    assert seen["env"]["GOOSE_PROVIDER"] == "openai"
    assert seen["env"]["GOOSE_MODEL"] == seen["env"]["GOOSE_SUBAGENT_MODEL"] == "policy"
    assert seen["env"]["GOOSE_SUBAGENT_PROVIDER"] == "openai"
    assert seen["env"]["OPENAI_CUSTOM_HEADERS"] == "x-session-id=rollout-1"
    assert seen["env"]["GOOSE_TEMPERATURE"] == "0.6"
    assert seen["env"]["GOOSE_CONTEXT_LIMIT"] == "262144"
    assert seen["env"]["GOOSE_PATH_ROOT"] == state.root
    assert seen["env"]["SSL_CERT_FILE"] == f"{state.root}/ca-certificates.pem"
    assert seen["ca_certificates"] > 0
    # The tool shell kept the task's HOME and environment; Goose's own files stayed in the session directory.
    assert (Path(state.session.workdir) / "proof.txt").read_text().splitlines() == [
        str(tmp_path / "task-home"),
        "task-value",
    ]
    assert sorted(path.name for path in Path(state.session.workdir).iterdir()) == ["proof.txt"]
    assert close.cleanup_confirmed
    invocation = close.agent_observations.records[0]
    assert invocation.status == "completed" and invocation.conversation[0].content == "Fix the bug."
    assert "model_call_ownership_unavailable" not in {gap.code for gap in close.agent_observations.gaps}
    assert sandbox.disconnected and not Path(state.session.session_dir).exists()
    assert sandbox.events.index("read cleanup.json") < sandbox.events.index("read stdout.jsonl")


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
@pytest.mark.parametrize(
    ("mode", "status", "outcome"),
    [
        ("max_turns", "incomplete", None),
        ("model_server", "failed", {"failure_kind": "model_request_error", "status_code": 500}),
        ("crash", "failed", {"failure_kind": "unknown"}),
        ("killed", "failed", {"failure_kind": "unknown"}),
    ],
)
async def test_goose_stops_map_to_status_and_outcome(tmp_path, mode, status, outcome):
    response, close, _, _, _ = await activate(tmp_path, mode)
    assert response.status == status
    reported = (response.metadata or {}).get(AGENT_OUTCOME_METADATA_KEY)
    assert (json.loads(reported) if reported else None) == outcome
    assert close.cleanup_confirmed


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
@pytest.mark.parametrize(("mode", "output"), [("sleep", 3), ("hang", 0)])
async def test_timeout_is_incomplete_with_or_without_output(tmp_path, mode, output):
    response, close, _, state, _ = await activate(tmp_path, mode, timeout=3)
    assert state.session.cleanup["timed_out"]
    assert response.status == "incomplete"
    assert len(response.output) == output
    assert close.agent_observations.records[1].outcome == "timeout"


@pytest.mark.parametrize("capture", [True, False])
def test_token_id_capture_routes_goose_model_calls_to_capture(capture):
    server = agent(token_id_capture=capture)
    server.server_client.global_config_dict = {"token_id_capture": {"enabled": True}}
    server.server_client._build_server_base_url.return_value = "http://model:9000"
    with patch("nemo_gym.agents.base_responses_api_agent.get_first_server_config_dict", return_value={}):
        # The same call that sets Goose's OPENAI_BASE_URL for an activation.
        url = server.model_endpoint(
            model_server=server.config.model_server, rollout_id=seed().episode_id.capture_key
        ).base_url
    capture_path = "/training-token-capture" if capture else ""
    assert url == f"http://model:9000/ng-rollout/rollout-1{capture_path}/v1"
