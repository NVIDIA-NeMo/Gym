# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import shutil
import stat
import tarfile

import pytest

from environment_servers.harbor.sandbox_io import _exec_as_root, _unpack, upload_dir
from nemo_gym.sandbox.providers.base import SandboxExecResult


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [False, True])
async def test_root_fallback_when_identity_switch_is_unavailable(exception):
    class Sandbox:
        def __init__(self):
            self.users = []

        async def exec(self, command, *, cwd, timeout_s, user=None):
            self.users.append(user)
            if user == "root":
                if exception:
                    raise RuntimeError("identity switch requires CAP_SETUID")
                return SandboxExecResult(stdout="", stderr="identity switch requires CAP_SETUID", return_code=1)
            return SandboxExecResult(stdout="ready", stderr="", return_code=0)

    sandbox = Sandbox()
    result = await _exec_as_root(sandbox, "mkdir -p /tests", timeout_s=1)
    assert result.return_code == 0 and sandbox.users == ["root", None]


@pytest.mark.asyncio
async def test_root_command_failure_does_not_retry_as_default_user():
    calls = []

    class Sandbox:
        async def exec(self, command, **kwargs):
            calls.append(kwargs["user"])
            return SandboxExecResult(stdout="", stderr="disk full", return_code=1)

    result = await _exec_as_root(Sandbox(), "mkdir -p /tests", timeout_s=1)
    assert result.return_code == 1 and calls == ["root"]


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("tar") is None, reason="GNU tar is required")
async def test_transfer_preserves_target_permissions_and_task_file_modes(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir(mode=0o700)
    target.mkdir()
    target.chmod(0o777)
    script = source / "test.sh"
    script.write_text("#!/bin/sh\nprintf works")
    script.chmod(0o751)
    secret = source / "private.txt"
    secret.write_text("private")
    secret.chmod(0o600)

    class LocalSandbox:
        async def upload(self, local, remote):
            # A top-level '.' would restore source metadata onto the receiving directory.
            with tarfile.open(local) as archive:
                assert "." not in archive.getnames()
            shutil.copyfile(local, remote)

        async def exec(self, command, *, cwd, timeout_s, user):
            assert user == "root"
            process = await asyncio.create_subprocess_exec(
                "bash", "-c", command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
            return SandboxExecResult(
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                return_code=process.returncode,
            )

    await upload_dir(LocalSandbox(), source, str(target))
    assert stat.S_IMODE(target.stat().st_mode) == 0o777
    assert stat.S_IMODE((target / "test.sh").stat().st_mode) == 0o751
    assert stat.S_IMODE((target / "private.txt").stat().st_mode) == 0o600
    process = await asyncio.create_subprocess_exec(str(target / "test.sh"), stdout=asyncio.subprocess.PIPE)
    stdout, _ = await process.communicate()
    assert process.returncode == 0 and stdout == b"works"


def _archive(path, members):
    """``members`` are ``(name, link)`` for a file (``None``) or symlink, or ``(name, link, "hard")``."""
    with tarfile.open(path, "w:gz") as tar:
        for name, link, *kind in members:
            info = tarfile.TarInfo(name)
            if link is None:
                info.size = 2
                tar.addfile(info, io.BytesIO(b"ok"))
            else:
                info.type = tarfile.LNKTYPE if kind == ["hard"] else tarfile.SYMTYPE
                info.linkname = link
                tar.addfile(info)


def test_unpack_keeps_artifact_links_but_never_lets_a_member_escape(tmp_path):
    venv = tmp_path / "venv.tar.gz"
    _archive(venv, [("./report.txt", None), ("./bin/python", "/usr/bin/python3"), ("./up", "../../elsewhere")])
    # Verifier output read on this host: a link to an absolute target is refused outright.
    with pytest.raises(tarfile.FilterError):
        _unpack(venv, tmp_path / "data")
    # Artifacts on their way into another sandbox: links stay links, wherever they point (they are
    # re-archived, never followed, on this host).
    _unpack(venv, tmp_path / "keep", links="keep")
    python = tmp_path / "keep" / "bin" / "python"
    assert (tmp_path / "keep" / "report.txt").read_bytes() == b"ok"
    assert python.is_symlink() and str(python.readlink()) == "/usr/bin/python3"
    assert (tmp_path / "keep" / "up").is_symlink()
    # A member whose own path lands outside the target is refused in both modes.
    escaping = tmp_path / "escape.tar.gz"
    _archive(escaping, [("../outside.txt", None)])
    for links in ("data", "keep"):
        with pytest.raises(tarfile.FilterError):
            _unpack(escaping, tmp_path / f"escape-{links}", links=links)


def test_hard_links_are_created_on_this_host_so_their_target_must_stay_inside(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("host secret")
    secret.chmod(0o600)
    archive = tmp_path / "hard.tar.gz"
    _archive(archive, [("./report.txt", "../secret.txt", "hard")])
    for links in ("data", "keep"):
        with pytest.raises(tarfile.FilterError):
            _unpack(archive, tmp_path / f"hard-{links}", links=links)
        assert not (tmp_path / f"hard-{links}" / "report.txt").exists()
    # A hard link inside the tree (a virtualenv's duplicated files) is fine.
    inside = tmp_path / "inside.tar.gz"
    _archive(inside, [("./a", None), ("./b", "a", "hard")])
    _unpack(inside, tmp_path / "inside", links="keep")
    assert (tmp_path / "inside" / "b").read_bytes() == b"ok"


def test_kept_artifacts_stay_readable_by_their_owner(tmp_path):
    """A tree the agent left unreadable can still be re-archived for the verifier sandbox."""
    archive = tmp_path / "modes.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        folder = tarfile.TarInfo("./locked")
        folder.type = tarfile.DIRTYPE
        folder.mode = 0o000
        tar.addfile(folder)
        hidden = tarfile.TarInfo("./locked/secret.bin")
        hidden.mode = 0o000
        hidden.size = 2
        tar.addfile(hidden, io.BytesIO(b"ok"))
        script = tarfile.TarInfo("./run.sh")
        script.mode = 0o100
        script.size = 2
        tar.addfile(script, io.BytesIO(b"ok"))
    _unpack(archive, tmp_path / "keep", links="keep")
    assert (tmp_path / "keep" / "locked" / "secret.bin").read_bytes() == b"ok"
    mode = (tmp_path / "keep" / "run.sh").stat().st_mode
    assert mode & 0o600 == 0o600 and mode & 0o100  # readable, and still executable
