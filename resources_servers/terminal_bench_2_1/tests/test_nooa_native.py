# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminal_bench_2_1.nooa_app import (
    NOOATerminalBenchConfig,
    NOOATerminalBenchResourcesServer,
    NOOATerminalBenchVerifyRequest,
)
from resources_servers.terminal_bench_2_1.nooa_task_metadata import read_image_startup
from resources_servers.terminal_bench_2_1.tests.test_app import _verify_request as _base_verify_request


def _verify_request(path):
    return NOOATerminalBenchVerifyRequest.model_validate(_base_verify_request(path).model_dump())


@pytest.fixture
def native(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test.sh").write_text("#!/bin/sh\n")
    server = NOOATerminalBenchResourcesServer(
        config=NOOATerminalBenchConfig(
            name="terminal", host="localhost", port=0, entrypoint="", sandbox_provider="sandbox", sandbox_config={}
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    sandbox = AsyncMock()
    sandbox.exec.return_value = SimpleNamespace(stdout="/workspace\n", stderr="", return_code=0)
    sandbox.serialize.return_value = {"provider": "docker", "sandbox_id": "owned-container"}

    def replace_reward(remote_path, local_path):
        replacement = Path(local_path).with_suffix(".replacement")
        replacement.write_text("1")
        replacement.replace(local_path)

    sandbox.download.side_effect = replace_reward
    server._create_sandbox = AsyncMock(return_value=sandbox)
    server._upload_folder = AsyncMock()
    request = SimpleNamespace(session={})
    task = _verify_request(tmp_path)
    body = ResourcesSeedSessionRequest.model_validate(
        {
            "resources_session_id": "resources-1",
            "episode_id": {"rollout_id": "episode-1"},
            "task_id": {"taskset": "tb21", "task_id": task.task_name},
            "task_data": {
                "task_name": task.task_name,
                "task_folder": str(tmp_path),
                "docker_image": task.docker_image,
                "verifier_timeout_seconds": 37,
            },
        }
    )
    task.verifier_timeout_seconds = 37
    close = ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    return SimpleNamespace(server=server, sandbox=sandbox, request=request, seed=body, close=close, verify=task)


async def test_native_seed_is_idempotent_and_borrowed(native) -> None:
    first, second = await asyncio.gather(
        native.server.seed_session(native.request, native.seed),
        native.server.seed_session(native.request, native.seed),
    )
    assert first == second
    assert first.sandbox_access.workdir == "/workspace"
    assert first.sandbox_access.connection.provider_config_ref == "sandbox"
    assert first.sandbox_access.connection.descriptor["sandbox_id"] == "owned-container"
    assert native.request.session[SESSION_ID_KEY] == "resources-1"
    native.server._create_sandbox.assert_awaited_once()
    native.sandbox.upload.assert_not_awaited()
    native.server._upload_folder.assert_not_awaited()  # No verifier or golden solution exposed before grading.
    native.sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("mismatch", [None, "image", "command", "source"])
async def test_sandbox_launch_uses_only_source_bound_startup(native, monkeypatch, tmp_path: Path, mismatch) -> None:
    import resources_servers.terminal_bench_2_1.nooa_app as app

    (tmp_path / "environment").mkdir()
    dockerfile = tmp_path / "environment/Dockerfile"
    dockerfile.write_text('FROM ubuntu:24.04\nCMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]\n')
    (tmp_path / "task.toml").write_text('[environment]\ndocker_image = "unused-test-image"\n')
    task = native.verify
    task.image_startup = read_image_startup(tmp_path)
    if mismatch == "image":
        task.docker_image = "another-image"
    elif mismatch == "command":
        task.image_startup.command = ["different-command"]
    elif mismatch == "source":
        dockerfile.write_text(dockerfile.read_text() + "# source changed\n")
    created = AsyncMock()
    monkeypatch.setattr(app, "AsyncSandbox", lambda _: created)
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app, "resolve_provider_config", lambda *_: {})
    monkeypatch.setattr(app, "resolve_provider_metadata", lambda *_: {})
    if mismatch:
        with pytest.raises(ValueError, match="startup metadata"):
            await app.NOOATerminalBenchResourcesServer._create_sandbox(native.server, task)
        created.start_with_setup.assert_not_awaited()
    else:
        await app.NOOATerminalBenchResourcesServer._create_sandbox(native.server, task)
        spec = created.start_with_setup.call_args.args[0]
        assert spec.image == "unused-test-image"
        assert spec.entrypoint == ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
        assert spec.workdir is None  # Retain the image WORKDIR, not an agent override.


@pytest.mark.parametrize("change", ["episode", "task_data"])
async def test_native_seed_rejects_conflicting_retry(native, change: str) -> None:
    await native.server.seed_session(native.request, native.seed)
    conflict = native.seed.model_copy(deep=True)
    if change == "episode":
        conflict.episode_id = conflict.episode_id.model_copy(update={"rollout_id": "other"})
    else:
        conflict.task_data["docker_image"] = "other-image"
    with pytest.raises(HTTPException, match="different request"):
        await native.server.seed_session(native.request, conflict)
    native.server._create_sandbox.assert_awaited_once()


async def test_native_close_fences_late_seed_and_is_idempotent(native) -> None:
    await native.server.close_resources_session(native.close)
    await native.server.close_resources_session(native.close)
    with pytest.raises(HTTPException, match="already closed"):
        await native.server.seed_session(native.request, native.seed)
    native.server._create_sandbox.assert_not_awaited()
    conflict = native.close.model_copy(deep=True)
    conflict.episode_id = conflict.episode_id.model_copy(update={"rollout_id": "other"})
    with pytest.raises(HTTPException, match="episode_id"):
        await native.server.close_resources_session(conflict)


async def test_native_close_retries_failed_stop_without_losing_owner(native) -> None:
    await native.server.seed_session(native.request, native.seed)
    native.sandbox.stop.side_effect = [RuntimeError("provider busy"), None]
    with pytest.raises(RuntimeError, match="provider busy"):
        await native.server.close_resources_session(native.close)
    assert native.server._session_id_to_sandbox["resources-1"] is native.sandbox
    await native.server.close_resources_session(native.close)
    await native.server.close_resources_session(native.close)
    assert native.sandbox.stop.await_count == 2
    assert not native.server._session_id_to_sandbox


@pytest.mark.parametrize("failure", [RuntimeError("descriptor failed"), asyncio.CancelledError()])
async def test_native_seed_failure_cleans_created_sandbox(native, failure: BaseException) -> None:
    native.sandbox.serialize.side_effect = failure
    with pytest.raises(type(failure)):
        await native.server.seed_session(native.request, native.seed)
    native.sandbox.stop.assert_awaited_once()
    await native.server.close_resources_session(native.close)
    assert not native.server._session_id_to_sandbox


@pytest.mark.parametrize("problem", ["missing_tests", "wrong_identity", "golden_mode", "relative_workdir"])
async def test_native_seed_rejects_invalid_setup(native, tmp_path: Path, problem: str) -> None:
    if problem == "missing_tests":
        (tmp_path / "tests/test.sh").unlink()
    elif problem == "wrong_identity":
        native.seed.task_id = native.seed.task_id.model_copy(update={"task_id": "wrong"})
    elif problem == "golden_mode":
        native.server.config.is_verifying_golden_patch = True
    else:
        native.sandbox.exec.return_value.stdout = "relative/path"
    with pytest.raises((HTTPException, RuntimeError)):
        await native.server.seed_session(native.request, native.seed)
    if problem == "relative_workdir":
        native.sandbox.stop.assert_awaited_once()
    else:
        native.server._create_sandbox.assert_not_awaited()


async def test_native_verification_reads_replaced_file_and_keeps_owner_until_close(native) -> None:
    await native.server.seed_session(native.request, native.seed)
    result = await native.server.verify(native.request, native.verify)
    assert result.reward == 1
    assert result.evaluation_completed and not result.mask_sample
    native.sandbox.exec.assert_awaited_with("bash /tests/test.sh", timeout_s=37)
    native.sandbox.stop.assert_not_awaited()
    native.server._upload_folder.assert_awaited_once()
    assert await native.server.verify(native.request, native.verify.model_copy(deep=True)) == result
    native.server._upload_folder.assert_awaited_once()
    native.sandbox.download.assert_awaited_once()
    changed = native.verify.model_copy(
        update={"response": native.verify.response.model_copy(update={"id": "changed"})}
    )
    with pytest.raises(HTTPException, match="Verification request changed"):
        await native.server.verify(native.request, changed)
    with pytest.raises(HTTPException, match="no longer available"):
        await native.server.seed_session(native.request, native.seed)
    await native.server.close_resources_session(native.close)
    native.sandbox.stop.assert_awaited_once()


async def test_native_verify_rejects_other_task_or_budget(native) -> None:
    await native.server.seed_session(native.request, native.seed)
    for update in ({"docker_image": "other"}, {"verifier_timeout_seconds": 999}):
        with pytest.raises(HTTPException, match="does not match"):
            await native.server.verify(native.request, native.verify.model_copy(update=update))
    native.server._upload_folder.assert_not_awaited()
    await native.server.close_resources_session(native.close)
    with pytest.raises(HTTPException, match="already closed"):
        await native.server.verify(native.request, native.verify)


@pytest.mark.parametrize("reward", ["0", "1", "nan", "inf", "-1", "2", "garbage"])
async def test_invalid_rewards_are_masked_not_scored(native, reward: str) -> None:
    await native.server.seed_session(native.request, native.seed)
    native.sandbox.download.side_effect = lambda remote, local: Path(local).write_text(reward)
    result = await native.server.verify(native.request, native.verify)
    valid = reward in {"0", "1"}
    assert result.evaluation_completed is valid
    assert result.mask_sample is not valid
    assert result.failure_kind == (None if valid else "verifier_error")
    assert result.reward == (float(reward) if valid else 0)


async def test_native_verification_cancellation_propagates_and_close_still_owns_sandbox(native) -> None:
    await native.server.seed_session(native.request, native.seed)
    native.sandbox.exec.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await native.server.verify(native.request, native.verify)
    await native.server.close_resources_session(native.close)
    native.sandbox.stop.assert_awaited_once()


async def test_native_missing_reward_is_incomplete(native) -> None:
    await native.server.seed_session(native.request, native.seed)
    native.sandbox.download.side_effect = FileNotFoundError("no reward")
    result = await native.server.verify(native.request, native.verify)
    assert result.mask_sample and not result.evaluation_completed
    assert result.failure_kind == "verifier_error"


async def test_verifier_exception_is_masked_and_owner_is_retained(native) -> None:
    await native.server.seed_session(native.request, native.seed)
    native.server.config.debug = True
    native.sandbox.exec.side_effect = TimeoutError("test command timed out")
    result = await native.server.verify(native.request, native.verify)
    assert result.mask_sample and not result.evaluation_completed
    native.sandbox.download.assert_not_awaited()
    await native.server.close_resources_session(native.close)
    native.sandbox.stop.assert_awaited_once()


async def test_failed_seed_stop_remains_owned_for_close(native) -> None:
    native.sandbox.serialize.side_effect = RuntimeError("serialize failed")
    native.sandbox.stop.side_effect = [RuntimeError("stop failed"), None]
    with pytest.raises(RuntimeError, match="serialize failed"):
        await native.server.seed_session(native.request, native.seed)
    assert native.server._session_id_to_sandbox["resources-1"] is native.sandbox
    await native.server.close_resources_session(native.close)
    assert native.sandbox.stop.await_count == 2


async def test_upload_preserves_nested_assets_and_scopes_shell_repairs(native, tmp_path: Path) -> None:
    task = tmp_path / "assets"
    (task / "nested").mkdir(parents=True)
    (task / "test.sh").write_text("old-package\n")
    (task / "nested/fixture.txt").write_text("old-package\n")
    uploads = {}

    async def upload(*, local_path, remote_path):
        uploads[remote_path] = Path(local_path).read_text()

    native.sandbox.upload.side_effect = upload
    await NOOATerminalBenchResourcesServer._upload_folder(
        native.server, native.sandbox, task, "/tests", {"task": [("old-package", "new-package")]}, "task"
    )
    assert uploads == {"/tests/test.sh": "new-package\n", "/tests/nested/fixture.txt": "old-package\n"}
    assert (task / "test.sh").read_text() == "old-package\n"


async def test_process_exit_closes_abandoned_native_sandbox(native) -> None:
    app = native.server.setup_webserver()
    async with app.router.lifespan_context(app):
        await native.server.seed_session(native.request, native.seed)
    native.sandbox.stop.assert_awaited_once()


def test_native_sessions_require_one_resources_worker() -> None:
    with pytest.raises(ValueError, match="num_workers=1"):
        NOOATerminalBenchResourcesServer(
            config=NOOATerminalBenchConfig(
                name="tb",
                host="localhost",
                port=0,
                entrypoint="",
                sandbox_provider="",
                sandbox_config={},
                num_workers=2,
            ),
            server_client=MagicMock(spec=ServerClient),
        )


@pytest.mark.parametrize("failed_stops", [1, 2])
async def test_failed_setup_retains_owner_until_provider_stop_succeeds(native, monkeypatch, failed_stops: int) -> None:
    import resources_servers.terminal_bench_2_1.nooa_app as app
    from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult

    provider = SimpleNamespace(
        name="test",
        create=AsyncMock(return_value=SimpleNamespace(sandbox_id="owned-during-setup")),
        exec=AsyncMock(return_value=SandboxExecResult("", "setup failed", 1)),
        close=AsyncMock(side_effect=[RuntimeError("stop failed")] * failed_stops + [None]),
        aclose=AsyncMock(),
    )
    sandbox = AsyncSandbox(provider)
    monkeypatch.setattr(app, "AsyncSandbox", lambda _: sandbox)
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app, "resolve_provider_config", lambda *_: {})
    monkeypatch.setattr(app, "resolve_provider_metadata", lambda *_: {})
    native.server._create_sandbox = NOOATerminalBenchResourcesServer._create_sandbox.__get__(native.server)

    with pytest.raises(RuntimeError, match="stop failed"):
        await native.server.seed_session(native.request, native.seed)

    # start_with_setup attempts teardown; the native owner then retries it.
    assert provider.close.await_count == 2
    assert ("resources-1" in native.server._session_id_to_sandbox) is (failed_stops == 2)
    assert sandbox._stopped is (failed_stops == 1)
    await native.server.close_resources_session(native.close)
    await native.server.close_resources_session(native.close)
    assert provider.close.await_count == failed_stops + 1
    assert sandbox._stopped
    assert not native.server._session_id_to_sandbox
