# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CoreWeave provider contracts; fake remote operations never allocate capacity."""

import asyncio
import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxCreateError, SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.cwsandbox import provider as module
from nemo_gym.sandbox.providers.cwsandbox.provider import CWSandboxConnectionConfig, CWSandboxProvider
from nemo_gym.sandbox.providers.registry import get_provider_class


pytestmark = pytest.mark.sandbox


class SDKTimeout(Exception):
    pass


class SDKNotFound(Exception):
    pass


@pytest.fixture
def backend(monkeypatch):
    sandbox = SimpleNamespace(
        sandbox_id="cw-test",
        status="running",
        start=AsyncMock(),
        wait=Mock(),
        exec=AsyncMock(return_value=SimpleNamespace(stdout="nemo-gym-ready", stderr="", returncode=0)),
        stop=AsyncMock(),
        write_file=AsyncMock(),
        read_file=AsyncMock(return_value=b"\x00\xffbinary"),
        get_status=Mock(return_value="running"),
    )
    constructor = Mock(return_value=sandbox)
    constructor.delete = AsyncMock()
    sdk = SimpleNamespace(
        Sandbox=constructor,
        NetworkOptions=lambda **kwargs: kwargs,
        SecurityContext=lambda **kwargs: kwargs,
        AuthHeaders=lambda **kwargs: kwargs,
        SandboxTimeoutError=SDKTimeout,
        SandboxNotFoundError=SDKNotFound,
    )
    monkeypatch.setattr(module, "import_module", lambda name: sdk)
    return sandbox, sdk, CWSandboxProvider()


@pytest.mark.asyncio
async def test_facade_lifecycle_and_resource_mapping(backend):
    sandbox, sdk, _ = backend
    spec = SandboxSpec(
        image="docker://registry/task:v1",
        ttl_s=1200,
        ready_timeout_s=56,
        workdir="/workspace",
        env={"PATH_EXTRA": "a b"},
        metadata={"harbor-task": "test"},
        resources={"cpu": 2, "memory_mib": 1024, "gpu": 1, "gpu_type": "H100"},
        provider_options={"resource_requests": {"cpu": 0.5}, "deny_egress": True},
    )
    async with AsyncSandbox({"cwsandbox": {"connection": {"api_key": "test-key"}}}, spec) as session:
        await session.start()
        assert await session.status() == SandboxStatus.RUNNING
    kwargs = sdk.Sandbox.call_args.kwargs
    assert kwargs["container_image"] == "registry/task:v1"
    assert kwargs["resources"] == {
        "requests": {"cpu": "0.5", "memory": "1024Mi"},
        "limits": {"cpu": "2.0", "memory": "1024Mi"},
        "gpu": {"count": 1, "type": "H100"},
    }
    assert kwargs["max_lifetime_seconds"] == 1200
    assert kwargs["environment_variables"] == spec.env
    assert kwargs["annotations"] == spec.metadata
    assert kwargs["working_dir"] == "/workspace"
    assert kwargs["network"] == {"deny_egress": True}
    assert kwargs["auth"]["headers"] == {"Authorization": "Bearer test-key"}
    assert "max_timeout_seconds" not in kwargs
    sandbox.wait.assert_called_once_with(timeout=56)
    sandbox.stop.assert_awaited_once_with(missing_ok=True, wait_for_ready=False)
    sdk.Sandbox.delete.assert_awaited_once_with(
        "cw-test", base_url=None, auth=kwargs["auth"], timeout_seconds=3700, missing_ok=True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["start", "wait", "probe", "terminal"])
async def test_failed_start_reclaims_sandbox(backend, failure):
    sandbox, sdk, provider = backend
    if failure == "start":
        sandbox.start.side_effect = RuntimeError("allocation failed")
    elif failure == "wait":
        sandbox.wait.side_effect = TimeoutError("not ready")
    elif failure == "probe":
        sandbox.exec.return_value.returncode = 1
    else:
        sandbox.status = "completed"
    with pytest.raises((RuntimeError, TimeoutError, SandboxCreateError)):
        await provider.create(SandboxSpec(image="task:v1"))
    sdk.Sandbox.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_start_waits_for_allocation_id_then_deletes(backend):
    sandbox, sdk, provider = backend
    accepted, release = asyncio.Event(), asyncio.Event()
    sandbox.sandbox_id = None

    async def start():
        accepted.set()
        await release.wait()
        sandbox.sandbox_id = "cw-late"

    sandbox.start.side_effect = start
    task = asyncio.create_task(provider.create(SandboxSpec(image="task:v1")))
    await accepted.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sdk.Sandbox.delete.call_args.args == ("cw-late",)


@pytest.mark.asyncio
async def test_exec_preserves_exit_output_and_quotes_env_and_user(backend):
    sandbox, _, provider = backend
    sandbox.exec.return_value = SimpleNamespace(stdout="output", stderr="failure", returncode=7)
    handle = SandboxHandle("cw-test", "cwsandbox", sandbox)
    result = await provider.exec(
        handle,
        "printf '%s' \"$VALUE\"",
        cwd="/work space",
        env={"VALUE": "x'; touch /bad"},
        user="worker",
        timeout_s=42,
    )
    assert (result.stdout, result.stderr, result.return_code) == ("output", "failure", 7)
    args, kwargs = sandbox.exec.call_args
    assert args[0][:2] == ["bash", "-c"]
    wrapped = shlex.split(args[0][2])
    assert wrapped[:4] == ["su", "worker", "-s", "/bin/bash"]
    assert shlex.split(wrapped[-1])[1] == "VALUE=x'; touch /bad"
    assert kwargs == {"cwd": "/work space", "timeout_seconds": 42}


@pytest.mark.asyncio
async def test_numeric_user_and_default_timeout(backend):
    sandbox, _, provider = backend
    sandbox.exec.side_effect = [
        SimpleNamespace(stdout="worker:x:1000:1000::/home/worker:/bin/bash", returncode=0),
        SimpleNamespace(stdout="ok", stderr="", returncode=0),
    ]
    await provider.exec(SandboxHandle("cw-test", "cwsandbox", sandbox), "id", user=1000)
    assert sandbox.exec.call_args_list[0].args == (["getent", "passwd", "1000"],)
    assert sandbox.exec.call_args.kwargs["timeout_seconds"] == 1800
    assert "su worker" in sandbox.exec.call_args.args[0][-1]


@pytest.mark.asyncio
async def test_exec_timeout_is_not_retried(backend):
    sandbox, _, provider = backend
    sandbox.exec.side_effect = SDKTimeout()
    result = await provider.exec(SandboxHandle("cw-test", "cwsandbox", sandbox), "sleep 60", timeout_s=1)
    assert result.return_code == 124
    assert result.error_type == "timeout"
    sandbox.exec.assert_awaited_once()


@pytest.mark.asyncio
async def test_binary_transfer(backend, tmp_path):
    sandbox, _, provider = backend
    handle = SandboxHandle("cw-test", "cwsandbox", sandbox)
    source = tmp_path / "input"
    source.write_bytes(b"\x00\xffbinary")
    await provider.upload_file(handle, source, "/tmp/remote")
    sandbox.write_file.assert_awaited_once_with("/tmp/remote", source.read_bytes(), timeout_seconds=3700)
    target = tmp_path / "nested/output"
    await provider.download_file(handle, "/tmp/remote", target)
    assert target.read_bytes() == source.read_bytes()


@pytest.mark.asyncio
async def test_delete_attempted_when_stop_fails(backend):
    sandbox, sdk, provider = backend
    sandbox.stop.side_effect = RuntimeError("stop failed")
    with pytest.raises(RuntimeError, match="stop failed"):
        await provider.close(SandboxHandle("cw-test", "cwsandbox", sandbox))
    sdk.Sandbox.delete.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state, expected",
    [
        ("pending", SandboxStatus.STARTING),
        ("creating", SandboxStatus.STARTING),
        ("preparing", SandboxStatus.STARTING),
        ("running", SandboxStatus.RUNNING),
        ("paused", SandboxStatus.PAUSED),
        ("completed", SandboxStatus.STOPPED),
        ("terminated", SandboxStatus.STOPPED),
        ("failed", SandboxStatus.ERROR),
        ("unspecified", SandboxStatus.UNKNOWN),
    ],
)
async def test_status_mapping(backend, state, expected):
    sandbox, _, provider = backend
    sandbox.get_status.return_value = state
    assert await provider.status(SandboxHandle("cw-test", "cwsandbox", sandbox)) == expected


@pytest.mark.asyncio
async def test_missing_sandbox_is_stopped(backend):
    sandbox, _, provider = backend
    sandbox.get_status.side_effect = SDKNotFound()
    assert await provider.status(SandboxHandle("cw-test", "cwsandbox", sandbox)) == SandboxStatus.STOPPED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spec, message",
    [
        (SandboxSpec(), "prebuilt"),
        (SandboxSpec(image="task", ports=[8080]), "ports"),
        (SandboxSpec(image="task", provider_options={"unknown": True}), "Unknown"),
        (SandboxSpec(image="task", provider_options={"deny_egress": "false"}), "boolean"),
        (SandboxSpec(image="task", resources={"cpu": -1}), "positive"),
        (SandboxSpec(image="task", provider_options={"resource_requests": {"disk": 1}}), "only cpu"),
        (
            SandboxSpec(image="task", resources={"cpu": 1}, provider_options={"resource_requests": {"cpu": 2}}),
            "exceeds",
        ),
        (SandboxSpec(image="task", ttl_s=0), "positive"),
        (SandboxSpec(image="task", ready_timeout_s=float("inf")), "positive"),
        (SandboxSpec(image="task", entrypoint=[]), "nonempty"),
    ],
)
async def test_reject_invalid_specs_before_allocating(backend, spec, message):
    _, sdk, provider = backend
    with pytest.raises(ValueError, match=message):
        await provider.create(spec)
    sdk.Sandbox.assert_not_called()


@pytest.mark.asyncio
async def test_disk_quota_warning_and_strict_mode(backend, caplog):
    _, sdk, provider = backend
    spec = SandboxSpec(image="task", resources={"disk_gib": 10})
    handle = await provider.create(spec)
    assert "cannot enforce disk_gib" in caplog.text
    await provider.close(handle)
    sdk.Sandbox.reset_mock()
    strict = CWSandboxProvider(create={"strict_resources": True})
    with pytest.raises(ValueError, match="disk_gib"):
        await strict.create(spec)
    sdk.Sandbox.assert_not_called()


def test_registration_and_credentials_repr():
    assert get_provider_class("cwsandbox") is CWSandboxProvider
    assert "secret-key" not in repr(CWSandboxConnectionConfig(api_key="secret-key"))


@pytest.mark.asyncio
async def test_installed_sdk_constructor_contract(backend, monkeypatch):
    # Validate our kwargs against a real SDK constructor, without start/network.
    sdk = pytest.importorskip("cwsandbox")
    sandbox, fake, provider = backend

    def construct(**kwargs):
        real = sdk.Sandbox(**kwargs)
        assert real.sandbox_id is None
        return sandbox

    fake.Sandbox.side_effect = construct
    fake.NetworkOptions = sdk.NetworkOptions
    fake.SecurityContext = sdk.SecurityContext
    provider = CWSandboxProvider(create={"runner_ids": ["d51a01c2-fd42-4be9-812d-d4150e083ea4"]})
    handle = await provider.create(SandboxSpec(image="ubuntu:24.04", resources={"cpu": 1, "memory_mib": 512}))
    await provider.close(handle)


@pytest.mark.asyncio
async def test_harbor_adapter_uses_coreweave(backend, tmp_path):
    pytest.importorskip("harbor")
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    from responses_api_agents.harbor_agent.custom_envs.nemo_gym_sandbox.environment import NemoGymSandboxEnvironment

    sandbox, sdk, _ = backend
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    environment = NemoGymSandboxEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="task",
        session_id="task__trial",
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=EnvironmentConfig(docker_image="ubuntu:24.04", cpus=2, memory_mb=1024),
        sandbox_provider={"cwsandbox": {}},
    )
    await environment.start(force_build=False)
    sandbox.exec.return_value = SimpleNamespace(stdout="agent-output", stderr="verifier-stderr", returncode=3)
    result = await environment.exec("echo agent-output", timeout_sec=600)
    assert (result.stdout, result.stderr, result.return_code) == ("agent-output", "verifier-stderr", 3)
    assert sandbox.exec.call_args.kwargs["timeout_seconds"] == 600
    kwargs = sdk.Sandbox.call_args.kwargs
    assert kwargs["annotations"]["harbor-session"] == "task__trial"
    assert kwargs["resources"]["limits"] == {"cpu": "2.0", "memory": "1024Mi"}
    await environment.stop(delete=False)
    sdk.Sandbox.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_readiness_budget_includes_start_acceptance(backend):
    sandbox, sdk, provider = backend

    async def slow_start():
        await asyncio.sleep(0.02)

    sandbox.start.side_effect = slow_start
    with pytest.raises(TimeoutError):
        await provider.create(SandboxSpec(image="task", ready_timeout_s=0.001))
    sdk.Sandbox.delete.assert_awaited_once()
    sandbox.wait.assert_not_called()


def test_missing_sdk_gives_install_instruction(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("No module named cwsandbox", name="cwsandbox")

    monkeypatch.setattr(module, "import_module", missing)
    with pytest.raises(ModuleNotFoundError, match=r"nemo-gym\[cwsandbox\]"):
        CWSandboxProvider()
