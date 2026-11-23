# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual Fargate exec server over loopback; no AWS provisioning."""

import asyncio
import contextlib
import json
import shlex
import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox.providers import SandboxHandle
from nemo_gym.sandbox.providers.ecs_fargate import EcsFargateProvider, engine


@contextlib.contextmanager
def exec_server(identity: str | None, *, port: int = 0) -> Iterator[int]:
    namespace = {}
    exec(compile(engine.EXEC_SERVER_SCRIPT, "<exec_server>", "exec"), namespace)
    namespace["_IDENTITY"] = identity or ""
    if identity is None:
        # Legacy servers dispatch only unqualified /exec, /upload and /download.
        namespace["_H"]._route = lambda self, path: path
    server = ThreadingHTTPServer(("127.0.0.1", port), namespace["_H"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("replacement_identity", ["b" * 32, None], ids=["another-task", "legacy-server"])
async def test_borrower_identity_survives_round_trip_and_rejects_reused_port(
    tmp_path: Path, monkeypatch, replacement_identity: str | None
) -> None:
    owner = engine.EcsFargateSandbox(
        engine.SandboxSpec(image="test-image", workdir=str(tmp_path)),
        ecs_config=engine.EcsFargateConfig(cluster="test-cluster", region="us-east-1"),
    )
    provider = EcsFargateProvider(cluster="test-cluster", region="us-east-1")
    # Only AWS client construction is replaced. Descriptor serialization,
    # attachment, HTTP dispatch, commands, and file transfers are real.
    monkeypatch.setattr(engine.EcsFargateSandbox, "_init_aws_clients", lambda self: None)
    owner._started = True
    marker = tmp_path / "marker"
    marker.write_text("SAFE")
    with exec_server(owner.exec_identity) as port:
        owner._ssh_tunnel = SimpleNamespace(local_port=port)
        owner._exec_client = engine.ExecClient(port=port, sandbox_identity=owner.exec_identity)
        handle = SandboxHandle(sandbox_id="owner-task", provider_name="ecs_fargate", raw=owner)
        descriptor = json.loads(json.dumps(await provider.serialize_handle(handle)))
        borrowed = await provider.connect(descriptor)
        try:
            result = await provider.exec(borrowed, "printf owner-task")
            assert result.stdout == "owner-task" and result.return_code == 0
            await borrowed.raw.upload(marker, str(tmp_path / "copy"))
            assert await borrowed.raw.exec_client.download(str(tmp_path / "copy")) == b"SAFE"
        finally:
            await owner.exec_client.close()

    try:
        # Another rollout now answers at the exact same host/port. Neither a
        # late attachment nor an already attached borrower may act on it.
        with exec_server(replacement_identity, port=port):
            error = "identity mismatch" if replacement_identity else "HTTP 404"
            with pytest.raises(RuntimeError, match=error):
                await provider.connect(descriptor)
            assert provider._borrowed == [borrowed.raw]
            with pytest.raises(RuntimeError, match=error):
                await provider.exec(borrowed, f"printf WRONG > {shlex.quote(str(marker))}")
            with pytest.raises(RuntimeError, match=error):
                await borrowed.raw.exec_client.upload(str(marker), b"WRONG", max_retries=1)
            with pytest.raises(RuntimeError, match=error):
                await borrowed.raw.exec_client.download(str(marker), max_retries=1)
            assert marker.read_text() == "SAFE"
    finally:
        await provider.disconnect_handle(borrowed)
    assert not provider._borrowed


@pytest.mark.parametrize("failure", ["mismatch", "cancel"])
async def test_failed_identity_check_releases_client(monkeypatch, failure: str) -> None:
    client = engine.ExecClient(port=1, sandbox_identity="a" * 32)
    entered = asyncio.Event()

    async def verify():
        await client._ensure_session()
        entered.set()
        if failure == "mismatch":
            raise RuntimeError("identity mismatch")
        await asyncio.Future()

    monkeypatch.setattr(client, "verify_identity", verify)
    monkeypatch.setattr(engine, "ExecClient", lambda **kwargs: client)
    monkeypatch.setattr(engine.EcsFargateSandbox, "_init_aws_clients", lambda self: None)
    task = asyncio.create_task(
        engine.EcsFargateSandbox.attach_exec_server(
            engine.SandboxSpec(image="test"),
            ecs_config=engine.EcsFargateConfig(),
            task_arn="task",
            port=1,
            sandbox_identity="a" * 32,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    if failure == "cancel":
        task.cancel()
    with pytest.raises(asyncio.CancelledError if failure == "cancel" else RuntimeError):
        await task
    assert client._session is None


async def test_borrower_rejects_health_without_identity(monkeypatch) -> None:
    client = engine.ExecClient(port=1, sandbox_identity="a" * 32)
    monkeypatch.setattr(client, "_request", AsyncMock(return_value=b'{"ok": true}'))
    with pytest.raises(RuntimeError, match="identity mismatch"):
        await client.verify_identity()


def test_identity_is_per_task_override_not_cached_task_definition() -> None:
    cfg = engine.EcsFargateConfig(ssh_sidecar=engine.SshSidecarConfig(exec_server_port=19542))
    owners = [engine.EcsFargateSandbox(engine.SandboxSpec(image="same-image"), ecs_config=cfg) for _ in range(2)]
    assert owners[0].exec_identity != owners[1].exec_identity
    for owner in owners:
        stable, runtime = owner._split_env(owner._build_env_vars())
        assert "NEMO_GYM_EXEC_IDENTITY" not in stable
        assert runtime["NEMO_GYM_EXEC_IDENTITY"] == owner.exec_identity
        assert owner.exec_identity not in str(owner._build_container_command(cfg.ssh_sidecar))
