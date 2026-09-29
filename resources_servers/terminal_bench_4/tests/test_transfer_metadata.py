# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from resources_servers.terminal_bench_4.collection import collect
from resources_servers.terminal_bench_4.task import TaskSettings
from resources_servers.terminal_bench_4.tests.test_collection_verifier import environment
from resources_servers.terminal_bench_4.transfers import (
    artifact_metadata_path,
    download_dir,
    download_file,
    upload_dir,
    upload_file,
)
from resources_servers.terminal_bench_4.verifier import restore


@pytest.mark.parametrize("mask", [0o022, 0o077])
async def test_round_trip_preserves_directory_and_file_modes_independent_of_host_umask(tmp_path, mask):
    cfg = TaskSettings.model_validate(
        {
            "environment": {"docker_image": "agent"},
            "verifier": {"environment": {"docker_image": "verifier"}},
            "artifacts": [{"source": "/app/project", "exclude": ["*.tmp"]}, "/app/tool.sh"],
        }
    )
    agent, source, _ = environment(tmp_path / "agent", cfg)
    verifier, target, _ = environment(tmp_path / "verifier", cfg)
    project = source.path("/app/project")
    (project / "empty").mkdir(parents=True)
    (project / "tool").write_bytes(b"#!/bin/sh\nexit 0\n")
    (project / "omit.tmp").write_text("not transferred")
    (project / "link").symlink_to("tool")
    os.link(project / "tool", project / "hardlink")
    source.path("/app/tool.sh").write_text("#!/bin/sh\nexit 0\n")
    modes = {"/app/project": 0o775, "/app/project/empty": 0o750, "/app/project/tool": 0o775, "/app/tool.sh": 0o751}
    for name, mode in modes.items():
        source.path(name).chmod(mode)
    old_mask = os.umask(mask)
    try:
        diagnostics = []
        await collect(agent, tmp_path / "artifacts", diagnostics)
        assert diagnostics == []
        await restore(verifier, tmp_path / "artifacts")
    finally:
        os.umask(old_mask)
    for name, mode in modes.items():
        observed = target.path(name).stat()
        original = source.path(name).stat()
        assert observed.st_mode & 0o777 == mode
        assert (observed.st_uid, observed.st_gid) == (original.st_uid, original.st_gid)
    assert target.path("/app/project/link").is_symlink()
    assert target.path("/app/project/link").read_bytes() == (project / "tool").read_bytes()
    assert target.path("/app/project/tool").stat().st_ino == target.path("/app/project/hardlink").stat().st_ino
    assert not target.path("/app/project/omit.tmp").exists()
    assert not target.path("/app/project/manifest.json").exists()
    assert all(kwargs.get("user") == "root" for cmd, kwargs in target.commands if "--same-owner" in cmd)


def synthetic_archive(path, *, unsafe=None):
    with tarfile.open(path, "w:gz") as tar:
        for name, mode, kind in [
            (".", 0o775, tarfile.DIRTYPE),
            ("nested", 0o750, tarfile.DIRTYPE),
            ("nested/tool", 0o6775, tarfile.REGTYPE),
        ]:
            member = tarfile.TarInfo(name)
            member.uid, member.gid = 1001, 2002
            member.uname, member.gname = "container-user", "container-group"
            member.mode, member.type = mode, kind
            data = b"payload" if kind == tarfile.REGTYPE else b""
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
        if unsafe:
            member = tarfile.TarInfo(unsafe)
            if unsafe == "escape-link":
                member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
            tar.addfile(member)


@pytest.mark.parametrize("mask", [0o022, 0o077])
async def test_tar_headers_preserve_container_ids_not_host_ids_and_strip_setid(tmp_path, mask):
    archive = tmp_path / "original.tar.gz"
    synthetic_archive(archive)
    uploaded = tmp_path / "uploaded.tar.gz"
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0)))

    async def download(source, destination):
        destination.write_bytes(archive.read_bytes())

    async def upload(source, destination):
        uploaded.write_bytes(source.read_bytes())

    sandbox.download, sandbox.upload = download, upload
    snapshot = tmp_path / "artifacts/project"
    metadata = artifact_metadata_path(tmp_path / "artifacts", Path("project"))
    old_mask = os.umask(mask)
    try:
        await download_dir(sandbox, "/app/project", snapshot, metadata_path=metadata)
        await upload_dir(sandbox, snapshot, "/app/project", metadata_path=metadata)
    finally:
        os.umask(old_mask)
    assert snapshot.stat().st_uid == os.getuid()  # No privileged host chown needed.
    assert json.loads(metadata.read_text())["."] == [1001, 2002, 0o775]
    with tarfile.open(uploaded) as tar:
        members = {Path(info.name).as_posix(): info for info in tar.getmembers()}
        assert set(members) == {".", "nested", "nested/tool"}
        for info in members.values():
            assert (info.uid, info.gid, info.uname, info.gname) == (1001, 2002, "", "")
        assert members["."].mode == 0o775
        assert members["nested"].mode == 0o750
        assert members["nested/tool"].mode == 0o775
        assert tar.extractfile("./nested/tool").read() == b"payload"


@pytest.mark.parametrize("unsafe", ["../escaped", "escape-link"])
async def test_metadata_collection_keeps_archive_traversal_and_link_checks(tmp_path, unsafe):
    archive = tmp_path / "unsafe.tar.gz"
    synthetic_archive(archive, unsafe=unsafe)
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0)))

    async def download(source, destination):
        destination.write_bytes(archive.read_bytes())

    sandbox.download = download
    metadata = tmp_path / "metadata.json"
    with pytest.raises(tarfile.FilterError):
        await download_dir(sandbox, "/app", tmp_path / "view", metadata_path=metadata)
    assert not metadata.exists()
    assert not (tmp_path / "escaped").exists()


async def test_artifact_transfer_fails_instead_of_losing_metadata_without_tar(tmp_path):
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=1, stderr="no tar")), upload=AsyncMock()
    )
    metadata = tmp_path / "metadata.json"
    with pytest.raises(RuntimeError, match="archive"):
        await download_dir(sandbox, "/app", tmp_path / "view", metadata_path=metadata)
    metadata.write_text(json.dumps({".": [os.getuid(), os.getgid(), 0o755]}))
    with pytest.raises(RuntimeError, match="with its metadata"):
        await upload_dir(sandbox, tmp_path / "view", "/app", metadata_path=metadata)
    assert not any(command.args[0].startswith("find ") for command in sandbox.exec.await_args_list)


async def test_single_file_restores_numeric_metadata_not_sdk_upload_defaults(tmp_path):
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="1001 2002 6751\n")),
        upload=AsyncMock(),
    )

    async def download(source, destination):
        destination.write_bytes(b"#!/bin/sh\n")

    sandbox.download = download
    metadata = tmp_path / "metadata.json"
    local = tmp_path / "tool"
    await download_file(sandbox, "/app/tool", local, metadata_path=metadata)
    assert json.loads(metadata.read_text()) == {".": [1001, 2002, 0o751]}
    await upload_file(sandbox, local, "/app/tool", metadata_path=metadata)
    command = sandbox.exec.await_args
    assert "chown 1001:2002 -- /app/tool && chmod 751 -- /app/tool" in command.args[0]
    assert command.kwargs["user"] == "root"
    sandbox.exec.return_value = SimpleNamespace(return_code=1, stderr="chown failed")
    with pytest.raises(RuntimeError, match="restore artifact metadata"):
        await upload_file(sandbox, local, "/app/tool", metadata_path=metadata)
