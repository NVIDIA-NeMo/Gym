# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prime Agent sessions: seed refusals, request mapping, and real supervised runs of a fake Prime Agent CLI."""

import asyncio
import hashlib
import io
import json
import os
import shlex
import shutil
import sys
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    ROLLOUT_STOP_METADATA_KEY,
    AgentSeedSessionRequest,
    SandboxSession,
    check_agent_seed_rejected,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.prime_agent.app import PrimeAgent, PrimeAgentConfig
from responses_api_agents.prime_agent.sandbox import CLI, PRIME_AGENT_VERSION, PRIME_RELEASE_PACKAGES, PrimeSession


# The Prime dependencies of the 0.7.0 release manifests, written out so a stale constant in sandbox.py fails here.
R2_RELEASE = "https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/releases/v0.7.0"
RELEASE_DEPENDENCIES = {
    "prime-agent": {
        "@earendil-works/pi-agent-core": f"{R2_RELEASE}/prime-agent-core-0.7.0.tgz",
        "@earendil-works/pi-ai": f"{R2_RELEASE}/prime-agent-ai-0.7.0.tgz",
        "@earendil-works/pi-tui": f"{R2_RELEASE}/prime-agent-tui-0.7.0.tgz",
        "chalk": "^5.5.0",
    },
    "prime-agent-ai": {"chalk": "^5.6.2"},
    "prime-agent-core": {"@earendil-works/pi-ai": f"{R2_RELEASE}/prime-agent-ai-0.7.0.tgz", "typebox": "^1.1.24"},
    "prime-agent-tui": {"chalk": "^5.5.0"},
}


def release_archive(path: Path, name: str) -> Path:
    content = json.dumps({"name": name, "dependencies": RELEASE_DEPENDENCIES[name]}).encode()
    member = tarfile.TarInfo("package/package.json")
    member.size = len(content)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(member, io.BytesIO(content))
    return path


def release_dependencies(path: str) -> dict[str, str]:
    with tarfile.open(path) as archive:
        return json.load(archive.extractfile("package/package.json"))["dependencies"]


FAKE_PRIME = r"""
import json, os, subprocess, sys, time
from pathlib import Path
args = sys.argv[2:]
if args == ["--version"]:
    print("__VERSION__")
    sys.exit(0)
agent_dir = Path(os.environ["PRIME_AGENT_CODING_AGENT_DIR"])
mode = os.environ.get("TEST_MODE", "normal")
(agent_dir / "seen.json").write_text(json.dumps({
    "cli": sys.argv[1], "args": args, "cwd": os.getcwd(),
    "models": json.loads((agent_dir / "models.json").read_text()),
    "env": {k: v for k, v in os.environ.items() if k.startswith(("PRIME_AGENT_", "PI_"))},
}))
subprocess.run(["bash", "-c", 'printf "%s\n%s\n" "$HOME" "$TASK_TEST" > proof.txt'], check=True)
if mode == "silent":
    time.sleep(60)
def end(message):
    print(json.dumps({"type": "message_end", "message": message}), flush=True)
usage = {"input": 10, "output": 2, "cacheRead": 1}
end({"role": "assistant", "content": [{"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "ls"}}],
     "usage": usage, "stopReason": "toolUse"})
end({"role": "toolResult", "toolCallId": "c1", "content": [{"type": "text", "text": "a.txt"}]})
if mode == "sleep":
    time.sleep(60)
if mode in ("error", "retried", "pending"):
    end({"role": "assistant", "content": [], "usage": usage, "stopReason": "error",
         "errorMessage": "503 Service Unavailable"})
if mode == "pending":
    time.sleep(60)
if mode != "error":
    end({"role": "assistant", "content": [{"type": "text", "text": "Done."}], "usage": usage, "stopReason": "stop"})
if mode == "lost":
    os.unlink(agent_dir.parent / "stdout.jsonl")
""".replace("__VERSION__", PRIME_AGENT_VERSION)


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
        if len(argv) > 1 and argv[1].endswith("/install_prime_runtime.sh"):
            assert argv[3] == PRIME_AGENT_VERSION and argv[8].endswith("/python.tar.gz")
            assert len(argv[11:]) % 2 == 0
            for path, checksum in [(argv[6], argv[7]), *zip(argv[11::2], argv[12::2])]:
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == checksum
            # npm installs the uploaded archives the rewritten manifests name, including core's transitive pi-ai.
            staged = {Path(path).name: path for path in [argv[6], *argv[11::2]]}
            assert set(staged) == {
                "prime-agent.tgz",
                "prime-agent-ai.tgz",
                "prime-agent-core.tgz",
                "prime-agent-tui.tgz",
                "runtime-package.json",
                "runtime-package-lock.json",
                "runtime-python-constraints.txt",
                "runtime-python-build-constraints.txt",
                "runtime-profile.txt",
                "uv.tar.gz",
            }
            manifest = json.loads(Path(staged["runtime-package.json"]).read_text())
            assert manifest["dependencies"] == {"prime-agent": f"file:{staged['prime-agent.tgz']}"}
            locked = json.loads(Path(staged["runtime-package-lock.json"]).read_text())
            assert locked["lockfileVersion"] == 3
            assert "node_modules/prime-agent/node_modules/@earendil-works/pi-ai" in locked["packages"]
            assert len(Path(staged["runtime-profile.txt"]).read_text().strip()) == 64
            assert release_dependencies(staged["prime-agent.tgz"]) == {
                "@earendil-works/pi-agent-core": f"file:{staged['prime-agent-core.tgz']}",
                "@earendil-works/pi-ai": f"file:{staged['prime-agent-ai.tgz']}",
                "@earendil-works/pi-tui": f"file:{staged['prime-agent-tui.tgz']}",
                "chalk": "^5.5.0",
            }
            assert release_dependencies(staged["prime-agent-core.tgz"]) == {
                "@earendil-works/pi-ai": f"file:{staged['prime-agent-ai.tgz']}",
                "typebox": "^1.1.24",
            }
            node = Path(argv[2]) / "bin" / "node"
            node.parent.mkdir(parents=True, exist_ok=True)
            node.write_text(f"#!{sys.executable}\n" + FAKE_PRIME)
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

    async def disconnect(self, **_):
        self.disconnected = True

    async def stop(self):
        pytest.fail("A borrowed sandbox must not be stopped")


def agent(**kwargs) -> PrimeAgent:
    kwargs.setdefault("model_server", {"type": "responses_api_models", "name": "policy_model"})
    return PrimeAgent(
        config=PrimeAgentConfig(
            host="localhost", port=8080, name="prime_agent", entrypoint="app.py", model="policy", **kwargs
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
    fields.setdefault(
        "input", [{"role": "system", "content": "Use bash."}, {"role": "user", "content": "Fix the bug."}]
    )
    return NeMoGymResponseCreateParamsNonStreaming(**fields)


@pytest.mark.parametrize(
    ("server", "request_seed", "reason"),
    [
        (
            {},
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
            "Prime Agent sessions use Prime Agent's own tools; required tool grants are unsupported",
        ),
        ({"model_server": None}, seed(), "Prime Agent sessions require a sandbox-reachable Gym model_server"),
        ({"extra_args": ["--x"]}, seed(), "Prime Agent sessions do not support command, extra_args, or env overrides"),
        ({}, seed(user="nobody"), "Prime Agent does not support a SandboxAccess user override"),
        ({}, seed(workdir="/tmp"), "Prime Agent task workdir must be absolute and separate from its session storage"),
        ({}, seed(borrowed=False), "Prime Agent requires sandbox_access or a configured sandbox_provider"),
    ],
)
def test_unsupported_seeds_are_final_refusals(server, request_seed, reason):
    prime = agent(**server)
    assert check_agent_seed_rejected(prime.setup_webserver(), request_seed) == reason
    assert not prime._session_records


def test_session_request_mapping_and_refusals():
    prime = agent(system_prompt="Be brief.")
    assert prime._session_conversation(body(instructions="Answer in English.")) == (
        "Fix the bug.",
        "Be brief.\n\nAnswer in English.\n\nUse bash.",
    )
    text = [{"type": "input_text", "text": "Fix "}, {"type": "input_text", "text": "it."}]
    assert prime._session_conversation(body(input=[{"role": "user", "content": text}])) == ("Fix it.", "Be brief.")
    image = {"type": "input_image", "image_url": "https://example.test/image.png", "detail": "auto"}
    for fields in (
        {"temperature": 0.5},
        {"max_output_tokens": 10},
        {"model": "other"},
        {"input": [{"role": "user", "content": [text[0], image]}]},
    ):
        with pytest.raises(HTTPException) as error:
            prime._session_conversation(body(**fields))
        assert error.value.status_code == 422


def test_run_without_resources_server_is_refused():
    with pytest.raises(HTTPException) as error:
        asyncio.run(agent().run(MagicMock(cookies={}), MagicMock()))
    assert error.value.status_code == 422


async def activate(tmp_path: Path, mode: str, *, timeout: float = 20, expire: bool = False):
    workdir = tmp_path / "task repo"
    workdir.mkdir()
    request = seed(workdir=str(workdir))
    request.sandbox_access.env = {
        "TEST_MODE": mode,
        "TASK_TEST": "task-value",
        "HOME": str(tmp_path / "task-home"),
        "PRIME_AGENT_CODING_AGENT_DIR": "must-not-inherit",
    }
    sandbox = LocalSandbox()
    state = PrimeSession(
        request=request,
        provider_name="local",
        sandbox_id="task",
        runtime=str(tmp_path / "runtime"),
        session=SandboxSession(
            sandbox=sandbox, session_dir=str(tmp_path / "prime session"), workdir=str(workdir), harness="Prime Agent"
        ),
    )
    # A host-only provider must not be uploaded into the sandbox.
    host_provider = {"apiKey": "host-only"}  # pragma: allowlist secret
    prime = agent(timeout=timeout, thinking="high", models_config={"providers": {"private": host_provider}})
    asset = tmp_path / "asset.tgz"
    asset.write_bytes(b"asset")
    releases = {f"{name}-0.7.0.tgz": release_archive(tmp_path / f"{name}.tgz", name) for name in RELEASE_DEPENDENCIES}

    async def fetch(url, **_):
        return releases.get(url.rsplit("/", 1)[1], asset)

    # The agent server downloads Node.js, the four release archives and the kernel's Python and uv; the fake
    # installer stands in for the runtime they would install.
    with (
        patch(
            "responses_api_agents.miniswe_sandboxed_agent.bootstrap.cached_download", AsyncMock(side_effect=fetch)
        ) as download,
        patch.dict(
            "responses_api_agents.prime_agent.sandbox.UV_SHA256",
            {arch: hashlib.sha256(asset.read_bytes()).hexdigest() for arch in ("x86_64", "aarch64")},
        ),
    ):
        await state.install(timeout=10)
    arch = os.uname().machine
    assert [call.args[0].rsplit("/", 1)[1] for call in download.await_args_list] == [
        f"node-v22.23.2-linux-{'arm64' if arch == 'aarch64' else 'x64'}.tar.gz",
        f"uv-{arch}-unknown-linux-musl.tar.gz",
        f"cpython-3.13.12%2B20260310-{arch}-unknown-linux-gnu-install_only_stripped.tar.gz",
        "prime-agent-0.7.0.tgz",
        "prime-agent-ai-0.7.0.tgz",
        "prime-agent-core-0.7.0.tgz",
        "prime-agent-tui-0.7.0.tgz",
    ]
    # The release archives come from GitHub and are checked against their pinned digests before the rewrite.
    assert [(call.args[0], call.kwargs["sha256"]) for call in download.await_args_list[3:]] == [
        (f"https://github.com/PrimeIntellect-ai/prime-agent/releases/download/v0.7.0/{name}-0.7.0.tgz", checksum)
        for name, checksum in PRIME_RELEASE_PACKAGES.items()
    ]
    if expire:
        assert await state.session.expire(timeout=10, reason="stopped")
    state.activation_request = body()
    with patch.object(
        PrimeAgent, "resolve_model_base_url", return_value="http://model.invalid/ng-rollout/rollout-1/v1"
    ):
        response = await asyncio.wait_for(prime._activate_session(state, state.activation_request), timeout=60)
    seen = None if expire else json.loads((Path(state.agent_dir) / "seen.json").read_text())
    close = await asyncio.wait_for(prime._close_agent_session_state(state), timeout=60)
    return response, close, seen, state, sandbox


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_real_session_run_maps_output_and_keeps_the_task_environment(tmp_path):
    response, close, seen, state, sandbox = await activate(tmp_path, "normal")
    assert response.status == "completed"
    assert [item.type for item in response.output] == ["function_call", "function_call_output", "message"]
    assert (response.usage.input_tokens, response.usage.output_tokens) == (22, 4)
    assert seen["cli"] == f"{state.runtime}/{CLI}"
    args = seen["args"]
    assert args[:5] == ["--print", "--mode", "json", "--no-session", "--daemon-socket"]
    assert args[args.index("--provider") + 1] == "nemo" and args[args.index("--model") + 1] == "policy"
    assert args[args.index("--append-system-prompt") + 1] == "Use bash." and args[-1] == "Fix the bug."
    assert list(seen["models"]["providers"]) == ["nemo"]
    assert seen["models"]["providers"]["nemo"]["baseUrl"] == "http://model.invalid/ng-rollout/rollout-1/v1"
    assert seen["env"]["PRIME_AGENT_CODING_AGENT_DIR"] == state.agent_dir
    assert seen["env"]["PRIME_AGENT_KERNEL_PYTHON"] == f"{state.runtime}/kernel/bin/python3"
    assert (Path(state.session.workdir) / "proof.txt").read_text().splitlines() == [
        str(tmp_path / "task-home"),
        "task-value",
    ]
    assert close.cleanup_confirmed and sandbox.disconnected and not Path(state.session.session_dir).exists()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_model_error_reports_the_upstream_status(tmp_path):
    response, close, _, _, _ = await activate(tmp_path, "error")
    assert response.status == "failed"
    outcome = json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY])
    assert outcome == {"failure_kind": "model_request_error", "status_code": 503}
    assert close.cleanup_confirmed


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_timeout_keeps_partial_work_as_incomplete(tmp_path):
    response, close, _, state, _ = await activate(tmp_path, "sleep", timeout=3)
    assert state.session.cleanup["timed_out"] and response.status == "incomplete"
    assert close.agent_observations.records[1].outcome == "timeout"


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_model_error_that_prime_agent_retried_is_completed(tmp_path):
    response, _, _, _, _ = await activate(tmp_path, "retried")
    assert response.status == "completed"
    assert [item.type for item in response.output] == ["function_call", "function_call_output", "message"]
    assert response.usage.output_tokens == 6


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_timeout_during_a_pending_retry_keeps_partial_work(tmp_path):
    response, _, _, _, _ = await activate(tmp_path, "pending", timeout=3)
    assert response.status == "incomplete"
    assert [item.type for item in response.output] == ["function_call", "function_call_output"]


@pytest.mark.asyncio
async def test_stop_before_the_runner_opens_stdout_collects_no_events(tmp_path):
    session = SandboxSession(
        sandbox=LocalSandbox(), session_dir=str(tmp_path / "session"), workdir=str(tmp_path), harness="Prime Agent"
    )
    state = PrimeSession(
        request=seed(workdir=str(tmp_path)), provider_name="local", sandbox_id="task", session=session
    )
    await state.stage(command=["prime-agent"], env={}, models="{}")
    assert (await state.collect()).stdout == ""


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_unreadable_stdout_raises_instead_of_scoring_an_empty_answer(tmp_path):
    with pytest.raises(FileNotFoundError):
        await activate(tmp_path, "lost")


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_rollout_stop_before_any_model_reply_is_incomplete(tmp_path):
    response, close, _, _, _ = await activate(tmp_path, "normal", expire=True)
    assert response.status == "incomplete" and response.output == []
    assert json.loads(response.metadata[ROLLOUT_STOP_METADATA_KEY]) == {"reason": "stopped"}
    assert close.cleanup_confirmed


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_timeout_before_any_model_reply_is_a_classified_failure(tmp_path):
    response, _, _, _, _ = await activate(tmp_path, "silent", timeout=3)
    assert response.status == "failed"
    assert json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY]) == {"failure_kind": "unknown"}


@pytest.mark.asyncio
async def test_install_rejects_an_unverified_uv_archive_before_installing(tmp_path: Path) -> None:
    workdir = tmp_path / "task"
    workdir.mkdir()
    runtime = tmp_path / "runtime"
    state = PrimeSession(
        request=seed(workdir=str(workdir)),
        provider_name="local",
        sandbox_id="task",
        runtime=str(runtime),
        session=SandboxSession(
            sandbox=LocalSandbox(), session_dir=str(tmp_path / "session"), workdir=str(workdir), harness="Prime Agent"
        ),
    )
    corrupt = tmp_path / "unverified-uv.tar.gz"
    corrupt.write_bytes(b"an archive that does not match the official release checksum")
    with patch(
        "responses_api_agents.miniswe_sandboxed_agent.bootstrap.cached_download", AsyncMock(return_value=corrupt)
    ) as download:
        with pytest.raises(RuntimeError, match="uv.*checksum mismatch"):
            await state.install(timeout=5)
    assert not runtime.exists()
    assert all("PrimeIntellect-ai/prime-agent" not in call.args[0] for call in download.await_args_list)
