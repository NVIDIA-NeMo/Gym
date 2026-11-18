# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen Code agent sessions: output conversion, settings, seed refusals, and real supervised runs of a fake CLI."""

import asyncio
import io
import json
import shlex
import shutil
import sys
import tarfile
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
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.qwen_code_agent.app import QwenCodeAgent, QwenCodeAgentConfig
from responses_api_agents.qwen_code_agent.conversion import parse_stream, summarize_subagents
from responses_api_agents.qwen_code_agent.sandbox import CLI, QWEN_VERSION, QwenSession


ROOT_EVENTS = [
    {"type": "system", "subtype": "init", "session_id": "s1"},
    {
        "type": "assistant",
        "parent_tool_use_id": None,
        "message": {"content": [{"type": "thinking", "thinking": "plan"}]},
    },
    {
        "type": "assistant",
        "parent_tool_use_id": None,
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "tool_call",
                    "input": {"name": "agent", "arguments": {"prompt": "x"}},
                }
            ]
        },
    },
    # Child events stay out of the root trajectory.
    {"type": "assistant", "parent_tool_use_id": "call-1", "message": {"content": [{"type": "text", "text": "child"}]}},
    {
        "type": "user",
        "parent_tool_use_id": None,
        "message": {"content": [{"type": "tool_result", "tool_use_id": "call-1", "content": [{"text": "42"}]}]},
    },
    {
        "type": "assistant",
        "parent_tool_use_id": None,
        "message": {"content": [{"type": "text", "text": "<think>check</think>answer 42"}]},
    },
    {"type": "result", "is_error": False, "result": "answer 42", "usage": {"input_tokens": 100, "output_tokens": 7}},
]


def archive(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data.encode())
            tar.addfile(info, io.BytesIO(data.encode()))
    return buffer.getvalue()


FAKE_QWEN = r"""
import json, os, subprocess, sys, time
from pathlib import Path
args = sys.argv[2:]
if args == ["--version"]:
    print("__VERSION__")
    sys.exit(0)
home = Path(os.environ["QWEN_HOME"])
mode = os.environ.get("TEST_MODE", "normal")
(home / "seen.json").write_text(json.dumps({
    "cli": sys.argv[1], "args": args, "prompt": sys.stdin.read(), "cwd": os.getcwd(),
    "settings": json.loads(Path(os.environ["QWEN_CODE_SYSTEM_SETTINGS_PATH"]).read_text()),
    "env": {k: v for k, v in os.environ.items() if k.startswith(("QWEN_", "NEMO_GYM_"))},
}))
if mode == "hang":
    time.sleep(60)
subprocess.run(["bash", "-c", 'printf "%s\n%s\n" "$HOME" "$TASK_TEST" > proof.txt'], check=True)
meta = home / "projects" / "p" / "subagents" / "s1" / "agent-0.meta.json"
meta.parent.mkdir(parents=True, exist_ok=True)
meta.write_text(json.dumps({"depth": 0}))
events = __EVENTS__
if mode == "silent":
    events = [e for e in events if not (e["type"] == "assistant" and "answer" in json.dumps(e))]
for event in events[:-1]:
    print(json.dumps(event), flush=True)
if mode == "sleep":
    time.sleep(60)
failed = mode in ("error", "api_error", "loop")
result = dict(events[-1], is_error=failed, subtype="error_during_execution" if failed else "success")
if mode == "api_error":
    result["error"] = {"message": "[API Error: 400 maximum context length exceeded]"}
if mode == "loop":
    result["error"] = {"message": "Loop detection halted the run."}
print(json.dumps(result))
sys.exit({"limit": 55, "loop": 1}.get(mode, 0))
""".replace("__VERSION__", QWEN_VERSION).replace("__EVENTS__", repr(ROOT_EVENTS).replace("None", "None"))


class LocalSandbox:
    """Run every sandbox command on this host, except the installer, which writes the fake runtime."""

    def __init__(self):
        self.disconnected = False

    async def upload(self, source, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    async def download(self, source, destination):
        shutil.copyfile(source, destination)

    async def exec(self, command, *, cwd=None, timeout_s=30, env=None):
        argv = shlex.split(command)
        if len(argv) > 1 and argv[1].endswith("/install_qwen_runtime.sh"):
            assert argv[3] == QWEN_VERSION
            node = Path(argv[2]) / "bin" / "node"
            node.parent.mkdir(parents=True, exist_ok=True)
            node.write_text(f"#!{sys.executable}\n" + FAKE_QWEN)
            node.chmod(0o755)
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
        self.disconnected = True

    async def stop(self):
        pytest.fail("A borrowed sandbox must not be stopped")


def agent(**kwargs) -> QwenCodeAgent:
    return QwenCodeAgent(
        config=QwenCodeAgentConfig(
            host="localhost",
            port=8080,
            name="qwen_code_agent",
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
        input=[{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Compute 6*7"}], **fields
    )


def test_parse_stream_keeps_root_trajectory_and_result():
    output, result = parse_stream("\n".join(json.dumps(e) for e in ROOT_EVENTS) + "\nnot json")
    assert [item.type for item in output] == [
        "reasoning",
        "function_call",
        "function_call_output",
        "reasoning",
        "message",
    ]
    assert output[1].name == "agent" and json.loads(output[1].arguments) == {"prompt": "x"}
    assert output[2].call_id == "call-1" and output[2].output == "42"
    assert output[4].content[0].text == "answer 42"
    assert result["usage"]["input_tokens"] == 100


def test_summarize_subagents_skips_truncated_and_corrupt_files():
    good = json.dumps({"usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 4}})
    files = {
        "projects/p/subagents/s1/agent-0.meta.json": json.dumps({"depth": 0, "isBackgrounded": True}),
        "projects/p/subagents/s1/agent-1.meta.json": '{"depth": 1, "isBackgr',
        "projects/p/subagents/s1/agent-0.jsonl": good + "\n" + '{"usageMetadata": {"promptTok',
        "projects/p/chats/s1.jsonl": json.dumps({"usageMetadata": {"promptTokenCount": 999}}),
    }
    assert summarize_subagents(archive(files)) == {
        "qwen_subagents": 1,
        "qwen_subagents_background": 1,
        "qwen_subagent_max_depth": 1,
        "qwen_subagent_input_tokens": 30,
        "qwen_subagent_output_tokens": 4,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"subagents": False, "agent_team": True},
        {"subagent_model_server": {"type": "responses_api_models", "name": "w"}},
        {"subagent_model": "policy"},
        {"subagent_model": "worker"},
        {"network_access": "model_only", "sandbox_provider": "sandbox"},
    ],
)
def test_invalid_subagent_settings(overrides):
    with pytest.raises(ValueError):
        agent(**overrides)


def test_settings_route_every_model_through_the_rollout():
    server = agent(
        subagent_model="worker",
        subagent_model_server={"type": "responses_api_models", "name": "worker_model"},
        settings={"modelProviders": {"openai": []}, "ui": {"theme": "x"}, "model": {"generationConfig": {"x": 1}}},
    )
    state = MagicMock(request=seed())

    def endpoint(*, model_server, rollout_id):
        return SimpleNamespace(base_url=f"http://{model_server.name}/ng-rollout/{rollout_id}/v1")

    with patch.object(QwenCodeAgent, "model_endpoint", side_effect=endpoint):
        settings = server.qwen_settings(state, body(temperature=0.0, top_p=0.9))
    assert [route["baseUrl"] for route in settings["modelProviders"]["openai"]] == [
        "http://policy_model/ng-rollout/rollout-1/v1",
        "http://worker_model/ng-rollout/rollout-1/v1",
    ]
    # Qwen Code applies only the selected route's generationConfig.
    assert [route["generationConfig"] for route in settings["modelProviders"]["openai"]] == [
        {"x": 1, "samplingParams": {"temperature": 0.0, "top_p": 0.9}}
    ] * 2
    assert settings["experimental"]["cron"] is False
    assert settings["ui"] == {"theme": "x"}


def test_token_id_capture_refuses_a_subagent_server():
    server = agent(
        subagent_model="worker",
        subagent_model_server={"type": "responses_api_models", "name": "worker_model"},
        token_id_capture=True,
    )
    server.server_client.global_config_dict = {"token_id_capture": {"enabled": True}}
    reason = "Qwen Code token capture is unsupported with subagent_model_server"
    assert check_agent_seed_rejected(server.setup_webserver(), seed()) == reason


@pytest.mark.parametrize("capture", [True, False])
def test_token_id_capture_routes_root_calls_to_capture(capture):
    server = agent(token_id_capture=capture)
    server.server_client.global_config_dict = {"token_id_capture": {"enabled": True}}
    server.server_client._build_server_base_url.side_effect = lambda config: f"http://{config['name']}:9000"
    with patch(
        "nemo_gym.agents.base_responses_api_agent.get_first_server_config_dict",
        side_effect=lambda _, name: {"name": name},
    ):
        settings = server.qwen_settings(MagicMock(request=seed()), body())
    capture_path = "/training-token-capture" if capture else ""
    assert [route["baseUrl"] for route in settings["modelProviders"]["openai"]] == [
        f"http://policy_model:9000/ng-rollout/rollout-1{capture_path}/v1"
    ]


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
            "Qwen Code sessions use Qwen Code's own tools; required tool grants are unsupported",
        ),
        (seed(user="nobody"), "Qwen Code does not support a SandboxAccess user override"),
        (seed(workdir="/tmp"), "Qwen Code task workdir must be absolute and separate from its session storage"),
        (seed(borrowed=False), "Qwen Code requires sandbox_access or a configured sandbox_provider"),
    ],
)
def test_unsupported_seeds_are_final_refusals(request_seed, reason):
    server = agent()
    assert check_agent_seed_rejected(server.setup_webserver(), request_seed) == reason
    assert not server._session_records


@pytest.mark.parametrize(
    ("network_access", "borrowed", "reason"),
    [
        ("model_only", True, "Qwen Code network_access applies only to a sandbox it creates, not a borrowed one"),
        ("model_and_tools", False, "Qwen Code uses no tool grants; set network_access to model_only or inherit"),
    ],
)
def test_network_access_refusals(network_access, borrowed, reason):
    server = agent(network_access=network_access, sandbox_provider="sandbox", runtime_archive="runtime.tar.gz")
    assert check_agent_seed_rejected(server.setup_webserver(), seed(borrowed=borrowed)) == reason


@pytest.fixture
def owned_sandbox(monkeypatch):
    import responses_api_agents.qwen_code_agent.app as module

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
    monkeypatch.setattr(QwenSession, "install", AsyncMock())
    hosts = {"policy_model": "model.example", "lightning": "model.example", "elsewhere": "other.example"}
    monkeypatch.setattr(
        QwenCodeAgent, "resolve_model_base_url", lambda self, name, **kwargs: f"http://{hosts[name]}:9000/v1"
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
        {"subagent_model_server": {"type": "responses_api_models", "name": subagent_server}} if subagent_server else {}
    )
    if subagent_server:
        subagent["subagent_model"] = "child"
    server = agent(network_access=network_access, sandbox_provider="sandbox", runtime_archive="r.tar.gz", **subagent)
    await server._seed_agent_session_state(seed(borrowed=False))
    assert owned_sandbox.start.await_args.args[0].provider_options.get("network_policy") == policy


@pytest.mark.asyncio
async def test_model_only_refuses_a_subagent_server_on_another_host(owned_sandbox):
    subagent = {"type": "responses_api_models", "name": "elsewhere"}
    server = agent(
        network_access="model_only",
        sandbox_provider="sandbox",
        runtime_archive="r.tar.gz",
        subagent_model="child",
        subagent_model_server=subagent,
    )
    with pytest.raises(AgentSeedRejectedError, match="subagent_model_server is on another"):
        await server._seed_agent_session_state(seed(borrowed=False))
    owned_sandbox.start.assert_not_awaited()


def test_request_mapping_and_refusals():
    server = agent(system_prompt="Show work.")
    assert server._conversation(body(instructions="Use tools.")) == (
        "Compute 6*7",
        "Show work.\n\nUse tools.\n\nBe terse.",
    )
    for fields in (
        {"max_output_tokens": 10},
        {"model": "other"},
        {"tools": [{"type": "function", "name": "f", "parameters": {}, "strict": True}]},
    ):
        with pytest.raises(HTTPException) as error:
            server._conversation(body(**fields))
        assert error.value.status_code == 422
    with pytest.raises(HTTPException, match="one user message"):
        server._conversation(
            NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("probe", "error", "message"),
    [
        # A failed probe is a retryable setup error, not a refusal of the episode.
        (SandboxExecResult("", "", -1, error_type="timeout"), RuntimeError, "Cannot probe"),
        (SandboxExecResult("riscv64\nglibc 2.39\n/usr/bin/flock\n", "", 0), AgentSeedRejectedError, "'riscv64'"),
        (SandboxExecResult("x86_64\n\n/usr/bin/flock\n", "", 0), AgentSeedRejectedError, "glibc 2.28"),
        (SandboxExecResult("x86_64\nglibc 2.39\n\n", "", 0), AgentSeedRejectedError, "needs flock"),
    ],
)
async def test_install_refuses_unsupported_images_and_retries_probe_failures(tmp_path, probe, error, message):
    missing, ok = SandboxExecResult("", "", 1), SandboxExecResult("", "", 0)
    sandbox = MagicMock(upload=AsyncMock(), exec=AsyncMock(side_effect=[ok, missing, probe]))
    state = QwenSession(
        request=seed(),
        provider_name="local",
        sandbox_id="task",
        session=SandboxSession(sandbox=sandbox, session_dir=str(tmp_path), workdir="/repo", harness="Qwen Code"),
    )
    with pytest.raises(error, match=message):
        await state.install(runtime_archive=None, timeout=10)


async def activate(tmp_path: Path, mode: str, *, timeout: float = 20):
    workdir = tmp_path / "task repo"
    workdir.mkdir()
    request = seed(workdir=str(workdir))
    request.sandbox_access.env = {
        "TEST_MODE": mode,
        "TASK_TEST": "task-value",
        "HOME": str(tmp_path / "task-home"),
        "QWEN_HOME": "must-not-inherit",
    }
    sandbox = LocalSandbox()
    state = QwenSession(
        request=request,
        provider_name="local",
        sandbox_id="task",
        runtime=str(tmp_path / "runtime"),
        session=SandboxSession(
            sandbox=sandbox, session_dir=str(tmp_path / "qwen session"), workdir=str(workdir), harness="Qwen Code"
        ),
    )
    server = agent(timeout=timeout, subagents=False, max_session_turns=50)
    node = tmp_path / "node.tar.gz"
    node.write_bytes(b"node")
    # The agent server downloads Node.js; the fake installer stands in for the runtime.
    with patch(
        "responses_api_agents.miniswe_sandboxed_agent.bootstrap.cached_download", AsyncMock(return_value=node)
    ) as download:
        await state.install(runtime_archive=None, timeout=10)
    assert download.await_args.args[0].startswith("https://nodejs.org/dist/v22.23.2/node-v22.23.2-linux-")
    state.activation_request = body(max_tool_calls=30)
    endpoint = SimpleNamespace(base_url="http://model.invalid/ng-rollout/rollout-1/v1")
    with patch.object(QwenCodeAgent, "model_endpoint", return_value=endpoint):
        response = await asyncio.wait_for(server._activate(state, state.activation_request), timeout=60)
    seen = json.loads((Path(state.qwen_home) / "seen.json").read_text())
    close = await asyncio.wait_for(server._close_agent_session_state(state), timeout=60)
    return response, close, seen, state, sandbox


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_real_run_maps_output_usage_and_keeps_the_task_environment(tmp_path):
    response, close, seen, state, sandbox = await activate(tmp_path, "normal")
    assert response.status == "completed"
    assert response.output[-1].content[0].text == "answer 42"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (100, 7)
    assert response.metadata["qwen_subagents"] == "1"
    assert seen["cli"] == f"{state.runtime}/{CLI}"
    assert seen["prompt"] == "Compute 6*7" and seen["cwd"] == state.session.workdir
    args = seen["args"]
    assert args[args.index("--append-system-prompt") + 1] == "Be terse."
    assert args[args.index("--max-tool-calls") + 1] == "30" and args[args.index("--max-session-turns") + 1] == "50"
    # Web tools stay off by default.
    excluded = {args[i + 1] for i, arg in enumerate(args) if arg == "--exclude-tools"}
    assert {"agent", "list_agents", "web_fetch", "web_search"} <= excluded
    assert seen["settings"]["modelProviders"]["openai"][0]["baseUrl"] == "http://model.invalid/ng-rollout/rollout-1/v1"
    assert seen["env"]["QWEN_HOME"] == state.qwen_home
    assert (Path(state.session.workdir) / "proof.txt").read_text().splitlines() == [
        str(tmp_path / "task-home"),
        "task-value",
    ]
    assert "usage_unavailable" not in {gap.code for gap in close.agent_observations.gaps}
    assert close.cleanup_confirmed and sandbox.disconnected and not Path(state.session.session_dir).exists()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
@pytest.mark.parametrize(
    ("mode", "outcome"),
    [
        ("error", {"failure_kind": "unknown"}),
        ("silent", {"failure_kind": "agent_invalid_output"}),
        ("api_error", {"failure_kind": "model_request_error", "status_code": 400}),
    ],
)
async def test_failed_runs_report_an_outcome(tmp_path, mode, outcome):
    response, close, _, _, _ = await activate(tmp_path, mode)
    assert response.status == "failed"
    assert json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY]) == outcome
    assert close.cleanup_confirmed


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
@pytest.mark.parametrize("mode", ["limit", "loop"])
async def test_qwen_limit_stops_are_graded_as_incomplete(tmp_path, mode):
    response, _, _, _, _ = await activate(tmp_path, mode)
    assert response.status == "incomplete" and AGENT_OUTCOME_METADATA_KEY not in response.metadata


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
@pytest.mark.parametrize("mode", ["sleep", "hang"])
async def test_timeout_is_incomplete_with_or_without_output(tmp_path, mode):
    response, close, _, state, _ = await activate(tmp_path, mode, timeout=3)
    assert state.session.cleanup["timed_out"] and response.status == "incomplete"
    assert response.usage is None
    assert close.agent_observations.records[1].outcome == "timeout"
