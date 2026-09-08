# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin the provider to the real Modal SDK surface it relies on.

These tests import the *installed* ``modal`` (no network, no account) and assert
that every keyword the provider forwards is actually accepted by the SDK, and
that every ``.aio`` coroutine the provider awaits exists. A Modal release that
renames ``unencrypted_ports`` or drops ``Sandbox.from_id.aio`` fails here, in CI,
instead of at the first live ``create()``.
"""

from __future__ import annotations

import inspect

import pytest


pytestmark = pytest.mark.sandbox
modal = pytest.importorskip("modal")


def _params(fn) -> set[str]:
    return set(inspect.signature(fn).parameters)


# Keywords ModalProvider.create() may place in Sandbox.create(**kwargs).
_CREATE_KWARGS = {
    "app",
    "image",
    "timeout",
    "idle_timeout",
    "workdir",
    "env",
    "cpu",
    "memory",
    "gpu",
    "cloud",
    "region",
    "block_network",
    "secrets",
    "volumes",
    "tags",
    "name",
    "unencrypted_ports",
    "encrypted_ports",
    "h2_ports",
}

# Keywords ModalProvider.exec()/file streams may place in sandbox.exec(**kwargs).
_EXEC_KWARGS = {"workdir", "env", "timeout", "text"}


def test_sandbox_create_accepts_every_kwarg_the_provider_forwards():
    missing = _CREATE_KWARGS - _params(modal.Sandbox.create)
    assert not missing, f"modal.Sandbox.create no longer accepts: {sorted(missing)}"


def test_sandbox_exec_accepts_every_kwarg_the_provider_forwards():
    missing = _EXEC_KWARGS - _params(modal.Sandbox.exec)
    assert not missing, f"modal.Sandbox.exec no longer accepts: {sorted(missing)}"


@pytest.mark.parametrize(
    "path",
    [
        "Sandbox.create",
        "Sandbox.from_id",
        "Sandbox.exec",
        "Sandbox.poll",
        "Sandbox.terminate",
        "Sandbox.detach",
        "Sandbox.tunnels",
        "App.lookup",
    ],
)
def test_async_surface_exists(path: str):
    obj = modal
    for part in path.split("."):
        obj = getattr(obj, part)
    assert hasattr(obj, "aio"), f"modal.{path} has no .aio coroutine"


def test_process_and_file_async_surface_exists():
    from modal.container_process import ContainerProcess
    from modal.io_streams import StreamReader, StreamWriter
    from modal.sandbox_fs import SandboxFilesystem

    assert hasattr(ContainerProcess.wait, "aio")
    assert hasattr(StreamReader.read, "aio")
    assert hasattr(StreamWriter.drain, "aio")
    # write / write_eof are sync on the writer (buffered until drain).
    assert callable(StreamWriter.write) and callable(StreamWriter.write_eof)
    for name in ("copy_from_local", "copy_to_local", "read_bytes", "write_bytes"):
        assert hasattr(getattr(SandboxFilesystem, name), "aio")
    assert "wait" in _params(modal.Sandbox.terminate)


def test_exception_classes_the_provider_classifies_exist():
    names = {
        "NotFoundError",
        "AuthError",
        "InvalidError",
        "PermissionDeniedError",
        "ResourceExhaustedError",
        "ExecTimeoutError",
        "SandboxTimeoutError",
        "SandboxTerminatedError",
        "SandboxFilesystemError",
        "SandboxFilesystemNotFoundError",
    }
    missing = {n for n in names if not isinstance(getattr(modal.exception, n, None), type)}
    assert not missing, f"modal.exception is missing: {sorted(missing)}"


def test_helpers_the_provider_calls_exist():
    assert callable(modal.Image.from_registry)
    assert "secret" in _params(modal.Image.from_registry)
    assert {"environment_name"} <= _params(modal.Secret.from_name)
    assert {"environment_name"} <= _params(modal.Volume.from_name)
    assert {"environment_name", "create_if_missing"} <= _params(modal.App.lookup)
    assert "timeout" in _params(modal.Sandbox.tunnels)
