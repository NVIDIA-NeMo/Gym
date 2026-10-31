# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from responses_api_agents.nooa_agent import offline_runtime


@pytest.fixture(params=["1000:1001", "0:0"])
def builder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest) -> AsyncMock:
    from nemo_gym import server_utils
    from responses_api_agents.miniswe_sandboxed_agent import bootstrap

    python_archive = tmp_path / "python.tar.gz"
    python_archive.write_bytes(b"portable Python")
    monkeypatch.setitem(
        offline_runtime._PYTHON_SHA256, "x86_64/gnu", hashlib.sha256(python_archive.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(offline_runtime, "_runtime_archive", lambda root: (b"source", "source-fingerprint"))
    monkeypatch.setattr(offline_runtime.shutil, "which", lambda tool: "/bin/docker")
    monkeypatch.setattr(bootstrap, "cached_download", AsyncMock(return_value=python_archive))
    monkeypatch.setattr(server_utils, "is_global_aiohttp_client_setup", lambda: True)

    async def command(*arguments: str, **kwargs: object) -> str:
        if arguments[1:3] == ("image", "inspect"):
            return "sha256:builder\n"
        if arguments[1] == "run":
            if arguments[-4:] == ("stat", "-c", "%u:%g", "/build"):
                return f"{request.param}\n"
            assert arguments[arguments.index("--user") + 1] == request.param
            work = Path(arguments[arguments.index("--volume") + 1].removesuffix(":/build"))
            assert (work / "source.tar.gz").read_bytes() == b"source"
            assert (work / "bundle/python.tar.gz").read_bytes() == b"portable Python"
            assert "sha256:builder" in arguments
            assert "linux/amd64" in arguments
            (work / "architecture.txt").write_text("x86_64\n")
            (work / "libc.txt").write_text("glibc 2.28\n")
            (work / "wheels").mkdir()
            (work / "wheels/fixture-1-py3-none-any.whl").write_bytes(b"fixture wheel")
            return "wheel build completed\n"
        return ""

    execute = AsyncMock(side_effect=command)
    monkeypatch.setattr(offline_runtime, "_command", execute)
    return execute


@pytest.mark.asyncio
async def test_builder_publishes_validated_bundle_and_reuses_without_docker(
    tmp_path: Path, builder: AsyncMock
) -> None:
    output = tmp_path / "bundle"
    assert await offline_runtime.prepare_runtime_wheelhouse(output, target="x86_64/gnu") == output
    manifest = offline_runtime.validate_runtime_wheelhouse(output, target="x86_64/gnu")
    assert manifest["builder_image"] == "sha256:builder"
    # A pure fixture wheel leaves the interpreter's own floor, not the builder's glibc 2.28.
    assert manifest["glibc"] == manifest["libc_version"] == "2.17" and manifest["builder_libc_version"] == "2.28"
    assert (output / "build.log").read_text() == "wheel build completed\n"
    assert [call.args[1] for call in builder.await_args_list] == ["pull", "image", "run", "run", "rm"]
    owner_probe, build, cleanup = builder.await_args_list[2:]
    assert owner_probe.args[:9] == build.args[:9]
    assert owner_probe.args[9] == build.args[11] == "sha256:builder"
    assert cleanup.args[2] == "--force"
    assert cleanup.args[-1] == build.args[build.args.index("--name") + 1]
    assert not list(tmp_path.glob(".nooa-wheelhouse-*"))
    builder.reset_mock()
    assert await offline_runtime.prepare_runtime_wheelhouse(output, target="x86_64/gnu") == output
    builder.assert_not_awaited()
    (output / "wheels.tar.gz").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        await offline_runtime.prepare_runtime_wheelhouse(output, target="x86_64/gnu")
    builder.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("build failed"), asyncio.CancelledError()])
@pytest.mark.parametrize("stage", ["owner", "build"])
async def test_builder_failure_removes_owned_container_without_publishing(
    tmp_path: Path, builder: AsyncMock, failure: BaseException, stage: str
) -> None:
    builder.side_effect = ["", "sha256:builder", *(["1000:1001"] if stage == "build" else []), failure, ""]
    output = tmp_path / "bundle"
    with pytest.raises(type(failure)):
        await offline_runtime.prepare_runtime_wheelhouse(output, target="x86_64/gnu")
    assert builder.await_args_list[-1].args[1:3] == ("rm", "--force")
    assert not output.exists()
    assert not list(tmp_path.glob(".nooa-wheelhouse-*"))


@pytest.mark.asyncio
async def test_cleanup_failure_is_not_accepted_as_success(tmp_path: Path, builder: AsyncMock) -> None:
    create = builder.side_effect

    async def fail_cleanup(*arguments: str, **kwargs: object) -> str:
        if arguments[1] == "rm":
            raise RuntimeError("Docker daemon disconnected")
        return await create(*arguments, **kwargs)

    builder.side_effect = fail_cleanup
    output = tmp_path / "bundle"
    with pytest.raises(RuntimeError, match="daemon disconnected"):
        await offline_runtime.prepare_runtime_wheelhouse(output, target="x86_64/gnu")
    assert not output.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["source", "glibc", "empty", "nested", "symlink"])
async def test_loader_rejects_incompatible_manifest_or_unsafe_wheels(
    tmp_path: Path, builder: AsyncMock, case: str
) -> None:
    output = await offline_runtime.prepare_runtime_wheelhouse(tmp_path / "bundle", target="x86_64/gnu")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if case == "source":
        manifest["source_fingerprint"] = "other-source"
    elif case == "glibc":
        manifest["glibc"] = manifest["libc_version"] = "unknown"
    else:
        with tarfile.open(output / "wheels.tar.gz", "w:gz") as archive:
            if case != "empty":
                member = tarfile.TarInfo("../fixture.whl" if case == "nested" else "fixture.whl")
                if case == "symlink":
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/outside"
                archive.addfile(member, io.BytesIO())
        manifest["wheels_sha256"] = offline_runtime.file_sha256(output / "wheels.tar.gz")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="NOOA wheelhouse"):
        offline_runtime.validate_runtime_wheelhouse(output, target="x86_64/gnu")


@pytest.mark.asyncio
async def test_missing_docker_fails_before_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(offline_runtime, "_runtime_archive", lambda root: (b"source", "fingerprint"))
    monkeypatch.setattr(offline_runtime.shutil, "which", lambda tool: None)
    with pytest.raises(RuntimeError, match="requires Docker"):
        await offline_runtime.prepare_runtime_wheelhouse(tmp_path / "bundle", target="x86_64/gnu")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.asyncio
async def test_command_decodes_non_utf8_and_reports_failure() -> None:
    assert await offline_runtime._command(sys.executable, "-c", "import os; os.write(1, b'\\xff')") == "\ufffd"
    with pytest.raises(RuntimeError, match="exit 7"):
        await offline_runtime._command(sys.executable, "-c", "raise SystemExit(7)")


@pytest.mark.asyncio
async def test_command_timeout_kills_and_joins_process(monkeypatch: pytest.MonkeyPatch) -> None:
    async def blocked() -> tuple[bytes, bytes]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    process = Mock(returncode=None, communicate=blocked, kill=Mock(), wait=AsyncMock())
    monkeypatch.setattr(offline_runtime.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    with pytest.raises(TimeoutError):
        await offline_runtime._command("fixture", timeout=0)
    process.kill.assert_called_once()
    process.wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_builder_accepts_empty_output_directory(tmp_path: Path, builder: AsyncMock) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    await offline_runtime.prepare_runtime_wheelhouse(output, target="x86_64/gnu")
    assert (output / "manifest.json").is_file()
    assert builder.await_args_list[0].args[-1] == "quay.io/pypa/manylinux2014_x86_64:latest"


@pytest.mark.asyncio
async def test_builder_does_not_overwrite_nonempty_directory(tmp_path: Path, builder: AsyncMock) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "owned.txt").write_text("preserve")
    with pytest.raises(ValueError, match="Prepare the NOOA runtime wheelhouse first"):
        await offline_runtime.prepare_runtime_wheelhouse(output, target="x86_64/gnu")
    assert (output / "owned.txt").read_text() == "preserve"
    builder.assert_not_awaited()


@pytest.mark.asyncio
async def test_validate_only_expands_tilde(
    tmp_path: Path, builder: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    await offline_runtime.prepare_runtime_wheelhouse(Path("~/bundle"), target="x86_64/gnu")
    assert offline_runtime.validate_runtime_wheelhouse(Path("~/bundle"), target="x86_64/gnu")["target"] == "x86_64/gnu"


@pytest.mark.asyncio
async def test_builder_publishes_a_musl_bundle_and_the_loader_keeps_targets_apart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, builder: AsyncMock
) -> None:
    monkeypatch.setitem(offline_runtime._PYTHON_SHA256, "x86_64/musl", offline_runtime._PYTHON_SHA256["x86_64/gnu"])
    original = builder.side_effect

    async def command(*arguments: str, **kwargs: object) -> str:
        if arguments[1] == "run" and "stat" not in arguments:
            work = Path(arguments[arguments.index("--volume") + 1].removesuffix(":/build"))
            result = await original(*arguments, **kwargs)
            (work / "libc.txt").write_text("musl 1.2.5\n")
            return result
        return await original(*arguments, **kwargs)

    builder.side_effect = command
    output = await offline_runtime.prepare_runtime_wheelhouse(tmp_path / "musl", target="x86_64/musl")
    pull = builder.await_args_list[0].args
    assert pull[1] == "pull" and pull[-1] == "quay.io/pypa/musllinux_1_2_x86_64:latest"
    manifest = offline_runtime.validate_runtime_wheelhouse(output, target="x86_64/musl")
    assert manifest["libc"] == "musl" and manifest["libc_version"] == "1.2.3" and "glibc" not in manifest
    assert manifest["builder_libc_version"] == "1.2.5"
    with pytest.raises(ValueError, match="incompatible with this source/platform"):
        offline_runtime.validate_runtime_wheelhouse(output, target="x86_64/gnu")


@pytest.mark.parametrize(
    ("wheels", "libc", "builder", "expected"),
    [
        (["a-1-py3-none-any.whl"], "musl", "1.2.5", "1.2.3"),
        (["a-1-cp313-cp313-musllinux_1_2_x86_64.whl", "b-1-py3-none-any.whl"], "musl", "1.2.5", "1.2.3"),
        (["a-1-cp313-cp313-musllinux_1_1_x86_64.whl"], "musl", "1.2.5", "1.2.3"),
        (["a-1-cp313-cp313-manylinux_2_28_x86_64.musllinux_1_2_x86_64.whl"], "musl", "1.2.5", "1.2.3"),
        (["a-1-cp313-cp313-linux_x86_64.whl"], "musl", "1.2.5", "1.2.5"),
        (["a-1-cp313-cp313-musllinux_1_3_x86_64.whl"], "musl", "1.2.5", "1.3"),
        (["a-1-py3-none-any.whl"], "gnu", "2.28", "2.17"),
        (["a-1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"], "gnu", "2.28", "2.17"),
        (["a-1-cp313-cp313-manylinux_2_28_x86_64.whl"], "gnu", "2.28", "2.28"),
        (["a-1-cp313-cp313-manylinux_2_28_x86_64.musllinux_1_2_x86_64.whl"], "gnu", "2.17", "2.28"),
        (["a-1-cp313-cp313-linux_x86_64.whl"], "gnu", "2.28", "2.28"),
    ],
)
def test_libc_floor_comes_from_wheel_tags_and_the_interpreter_not_the_builder(
    wheels: list[str], libc: str, builder: str, expected: str
) -> None:
    assert offline_runtime.libc_floor(wheels, libc=libc, builder_version=builder) == expected
