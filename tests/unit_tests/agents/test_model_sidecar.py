# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path

import aiohttp
import psutil
import pytest
from aiohttp import web
from omegaconf import OmegaConf

from nemo_gym.agents.utils.model_sidecar import (
    ModelSidecarConfig,
    ModelSidecarError,
    _serve,
    model_only_config,
    start_model_sidecar,
)
from nemo_gym.config.global_config import NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME
from nemo_gym.sandbox.api import AsyncSandboxSidecar
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle
from nemo_gym.server_runtime.supervisor import server_supervisor


def test_resolved_sampling_is_preserved():
    model = {
        "model": "fake",
        "sampling_overrides": {"temperature": 0.61, "max_tokens": 900},
        "uses_reasoning_parser": True,
        "endpoint_connection_retries": 4,
    }
    source = {"policy_model": {"responses_api_models": {"vllm_model": model}}, "agent": {"unused": True}}
    config = ModelSidecarConfig(image="local-only")
    result = model_only_config(source, config)
    assert set(result) == {"head_server", "policy_model"}
    actual = result["policy_model"]["responses_api_models"]["vllm_model"]
    assert actual["sampling_overrides"] == model["sampling_overrides"]
    actual["sampling_overrides"]["temperature"] = 1
    assert model["sampling_overrides"]["temperature"] == 0.61
    assert result["head_server"]["host"] == "127.0.0.1"
    assert config.sidecars()[0].ports == (config.port,)


def test_task_endpoint_replaces_framework_auth_without_replacing_sampling():
    resolved = OmegaConf.create(
        {
            "framework_endpoint": "https://framework.invalid/v1",
            "temperature": 0.42,
            "unrelated": "${unset_on_purpose}",
            "policy_model": {
                "responses_api_models": {
                    "vllm_model": {
                        "model": "same-model",
                        "base_url": "${framework_endpoint}",
                        "api_key": "synthetic-framework-key",
                        "default_headers": {"X-API-Key": "synthetic-framework-header"},
                        "sampling_overrides": {"temperature": "${temperature}"},
                        "uses_reasoning_parser": True,
                        "propagate_context_overflow_errors": True,
                    }
                }
            },
        }
    )
    config = ModelSidecarConfig(image="test/gym:sidecar", upstream_base_url="https://task-upstream.invalid/v1")

    model = model_only_config(resolved, config)["policy_model"]["responses_api_models"]["vllm_model"]

    assert model["base_url"] == config.upstream_base_url
    assert model["api_key"] == "gym" and model["default_headers"] == {}
    assert model["model"] == "same-model"
    assert model["sampling_overrides"] == {"temperature": 0.42}
    assert model["uses_reasoning_parser"] and model["propagate_context_overflow_errors"]
    assert resolved.policy_model.responses_api_models.vllm_model.api_key == "synthetic-framework-key"


def test_reject_unresolved_config():
    with pytest.raises(ValueError, match="resolved"):
        model_only_config(
            {"policy_model": {"responses_api_models": {"vllm_model": {"model": "${name}"}}}},
            ModelSidecarConfig(image="local-only"),
        )


class LocalSidecarSandbox:
    """Run provider execs as local subprocesses with a separate capture directory."""

    def __init__(self, *, config, hold_readiness=False):
        self.config = config
        self.hold_readiness = hold_readiness
        self.ready = asyncio.Event()
        self.model_spawned = asyncio.Event()
        self.launch = None
        self.child = None
        self.results = []

    def sidecar(self, name):
        return AsyncSandboxSidecar(self, SandboxHandle("local", "test", None), name, ports=(self.config.port,))

    def sidecar_host(self, handle, sidecar):
        return "127.0.0.1"

    async def upload_sidecar_file(self, handle, sidecar, local, remote):
        assert sidecar == self.config.name
        assert Path(remote).parent.stat().st_mode & 0o777 == 0o700
        shutil.copyfile(local, remote)

    async def download_sidecar_file(self, handle, sidecar, remote, local):
        assert sidecar == "gym-model"
        shutil.copyfile(remote, local)

    async def sidecar_exec(self, handle, sidecar, argv, *, timeout_s, on_stdout=None):
        assert sidecar == "gym-model"
        assert "fake-test-secret" not in str(argv)
        env = os.environ | {NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME: "poisoned: true"}
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=Path(__file__).resolve().parents[3],
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if on_stdout:
            self.launch = process

        async def read_stdout():
            lines = []
            while line := await process.stdout.readline():
                text = line.decode()
                lines.append(text)
                if text.startswith("Waiting for head server to spin up"):
                    children = psutil.Process(process.pid).children()
                    assert len(children) == 1
                    self.child = children[0]
                    assert self.child.cmdline() == [os.path.abspath(sys.executable), "app.py"]
                    self.model_spawned.set()
                if text.startswith("NEMO_GYM_MODEL_SIDECAR_READY"):
                    self.ready.set()
                    if self.hold_readiness:
                        continue
                if on_stdout:
                    # A streaming provider may split a marker at any byte.
                    on_stdout(text[:10])
                    on_stdout(text[10:])
            return "".join(lines)

        try:
            async with asyncio.timeout(timeout_s):
                stdout, stderr = await asyncio.gather(read_stdout(), process.stderr.read())
                await process.wait()
            result = SandboxExecResult(stdout, stderr.decode(), process.returncode)
            self.results.append(result)
            return result
        finally:
            if process.returncode is None:
                # A lost exec waiter does not prove the provider killed the process.
                await process.wait()


def _local_config(tmp_path, unused_tcp_port_factory, *, start_timeout_s=12):
    return ModelSidecarConfig(
        image="preinstalled-test-image",
        python=sys.executable,
        remote_dir=str(tmp_path / "runtime"),
        port=unused_tcp_port_factory(),
        registry_port=unused_tcp_port_factory(),
        start_timeout_s=start_timeout_s,
        shutdown_timeout_s=9,
    )


@pytest.mark.asyncio
async def test_bootstrap_forwards_resolved_sampling_for_responses_and_stops(tmp_path, unused_tcp_port_factory):
    calls = []

    async def complete(request):
        assert request.headers["Authorization"] == "Bearer gym"
        calls.append(await request.json())
        if len(calls) == 2:
            return web.json_response({"error": {"message": "synthetic rejection"}}, status=400)
        return web.json_response(
            {
                "id": "fake-completion",
                "object": "chat.completion",
                "created": 1,
                "model": "actual-model",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    upstream_port = runner.addresses[0][1]
    config = _local_config(tmp_path, unused_tcp_port_factory).model_copy(
        update={"upstream_base_url": f"http://127.0.0.1:{upstream_port}/v1"}
    )
    sandbox = LocalSidecarSandbox(config=config)
    staged = {
        "policy_model": {
            "responses_api_models": {
                "vllm_model": {
                    "api_key": "fake-test-secret",
                    "base_url": "http://127.0.0.1:1/v1",
                    "model": "actual-model",
                    "sampling_overrides": {"temperature": 0.61, "top_p": 0.72, "max_tokens": 19},
                    "return_token_id_information": False,
                    "uses_reasoning_parser": False,
                    "max_num_tries": 1,
                }
            }
        },
        "ignored": {"responses_api_agents": {"not_a_server": {"entrypoint": "absent.py"}}},
    }
    runtime = None
    try:
        runtime = await start_model_sidecar(sandbox, staged, config, rollout_id="ep.2", run_timeout_s=20)
        assert not calls
        async with aiohttp.ClientSession() as client:
            async with client.get(f"http://127.0.0.1:{runtime.registry_port}/server_instances") as response:
                (instance,) = await response.json()
                assert instance["process_name"] == "policy_model"
                assert instance["name"] == "vllm_model"
                assert instance["pid"] == sandbox.child.pid
            async with client.get(f"http://127.0.0.1:{runtime.registry_port}/global_config_dict_yaml") as response:
                assert "fake-test-secret" not in await response.text()
            body = {
                "model": "caller-model",
                "input": "hello",
                "temperature": 0.11,
                "top_p": 0.12,
                "max_output_tokens": 50,
            }
            async with client.post(
                f"{runtime.base_url}/responses", json=body, headers={"x-session-id": "episode-session"}
            ) as response:
                assert response.status == 200, await response.text()
                assert (await response.json())["output"][0]["content"][0]["text"] == "hello"
            async with client.post(
                f"{runtime.base_url}/responses", json=body, headers={"x-session-id": "episode-session"}
            ) as response:
                assert response.status >= 400
        assert len(calls) == 2
        for sent in calls:
            assert sent["model"] == "actual-model"
            assert (sent["temperature"], sent["top_p"], sent["max_tokens"]) == (0.61, 0.72, 19)
        assert not list((tmp_path / "runtime").glob("*/config.json"))
    finally:
        if runtime is not None:
            await asyncio.gather(runtime.shutdown(), runtime.shutdown())
        await runner.cleanup()

    assert sandbox.launch.returncode == 0, sandbox.results
    assert not sandbox.child.is_running()
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", runtime.registry_port)) != 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["invalid-config", "cancelled-start", "readiness-timeout"])
async def test_start_failure_stops_local_supervisor(tmp_path, unused_tcp_port_factory, failure):
    config = _local_config(tmp_path, unused_tcp_port_factory)
    sandbox = LocalSidecarSandbox(config=config, hold_readiness=True)
    model = {
        "base_url": "http://127.0.0.1:1/v1",
        "model": "fake",
        "return_token_id_information": False,
        "uses_reasoning_parser": False,
    }
    if failure != "invalid-config":
        model["api_key"] = "fake-test-secret"
    task = asyncio.create_task(
        start_model_sidecar(
            sandbox,
            {"policy_model": {"responses_api_models": {"vllm_model": model}}},
            config,
            rollout_id="ep",
            run_timeout_s=15,
        )
    )
    if failure == "cancelled-start":
        try:
            await asyncio.wait_for(sandbox.model_spawned.wait(), timeout=10)
        finally:
            task.cancel()
        error = asyncio.CancelledError
    elif failure == "readiness-timeout":
        error = TimeoutError
    else:
        error = ModelSidecarError
    with pytest.raises(error):
        await task
    if failure == "readiness-timeout":
        assert sandbox.ready.is_set(), "the test must withhold a real readiness announcement"
    assert sandbox.launch.returncode is not None, sandbox.results
    assert sandbox.child is not None and not sandbox.child.is_running()
    assert not list((tmp_path / "runtime").glob("*/config.json"))


@pytest.mark.parametrize("force_shutdown", [False, True], ids=["graceful", "forced"])
def test_stop_during_model_spawn_reaps_child_and_preserves_shutdown_result(
    tmp_path, unused_tcp_port_factory, monkeypatch, force_shutdown
):
    config = _local_config(tmp_path, unused_tcp_port_factory)
    model = {"model": "fake", "base_url": "http://127.0.0.1:1/v1", "api_key": "gym"}
    staged = model_only_config({"policy_model": {"responses_api_models": {"vllm_model": model}}}, config)
    (tmp_path / "config.json").write_text(json.dumps(staged))
    child = None

    def spawn(*args, **kwargs):
        nonlocal child
        child = subprocess.Popen(["sleep", "60"], stderr=kwargs["stderr_target"], start_new_session=True)
        if force_shutdown:
            child.send_signal(signal.SIGSTOP)
            os.waitid(os.P_PID, child.pid, os.WSTOPPED)
        # The stop arrives while the new process is still unregistered.
        (tmp_path / "stop").touch()
        return child

    monkeypatch.setattr(server_supervisor, "run_command", spawn)
    monkeypatch.setenv("NEMO_GYM_OTEL_ENABLED", "0")
    try:
        code = _serve(tmp_path, "test", start_timeout_s=5, grace_s=0.5, run_timeout_s=5)
        assert child is not None and child.poll() == (-signal.SIGKILL if force_shutdown else -signal.SIGINT)
        assert code == int(force_shutdown)
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)


@pytest.mark.asyncio
async def test_forced_model_shutdown_is_reported(tmp_path, unused_tcp_port_factory):
    config = _local_config(tmp_path, unused_tcp_port_factory)
    sandbox = LocalSidecarSandbox(config=config)
    model = {
        "base_url": "http://127.0.0.1:1/v1",
        "model": "fake",
        "api_key": "fake-test-secret",
        "return_token_id_information": False,
        "uses_reasoning_parser": False,
    }
    runtime = await start_model_sidecar(
        sandbox,
        {"policy_model": {"responses_api_models": {"vllm_model": model}}},
        config,
        rollout_id="ep",
        run_timeout_s=20,
    )
    sandbox.child.send_signal(signal.SIGSTOP)
    with pytest.raises(ModelSidecarError, match="shutdown failed"):
        await runtime.shutdown()
    assert not sandbox.child.is_running()
    assert sandbox.launch.returncode == 1
