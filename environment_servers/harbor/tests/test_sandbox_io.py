# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import os
import shutil
import stat
import tarfile

import pytest

from environment_servers.harbor.sandbox_io import (
    SandboxProbeError,
    SandboxTransferError,
    SandboxUnavailableError,
    _exec_as_root,
    _unpack,
    download_path,
    remote_kind,
    tar_extract_command,
    upload_dir,
    upload_path,
)
from nemo_gym.sandbox.providers.base import SandboxExecResult


class LocalSandbox:
    """Runs the transfer commands on this host, so tar, find and chmod do what they do in a sandbox."""

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        process = await asyncio.create_subprocess_exec(
            "bash", "-c", command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        return SandboxExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
        )

    async def upload(self, local, remote):
        shutil.copyfile(local, remote)

    async def download(self, remote, local):
        shutil.copyfile(remote, local)


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


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("busybox") is None, reason="BusyBox is required")
async def test_transfer_works_with_busybox_tar_which_has_no_no_overwrite_dir_option(tmp_path):
    source, target, bin_dir = tmp_path / "source", tmp_path / "target", tmp_path / "bin"
    source.mkdir()
    script = source / "test.sh"
    script.write_text("#!/bin/sh\nprintf works")
    script.chmod(0o751)
    target.mkdir()
    target.chmod(0o777)
    # `tar` resolves to the BusyBox applet, as in an Alpine image.
    bin_dir.mkdir()
    (bin_dir / "tar").symlink_to(shutil.which("busybox"))
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}

    async def busybox_sh(command: str) -> SandboxExecResult:
        process = await asyncio.create_subprocess_exec(
            "busybox", "sh", "-c", command, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 60)
        return SandboxExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
        )

    class BusyBoxSandbox:
        async def upload(self, local, remote):
            shutil.copyfile(local, remote)

        async def exec(self, command, *, cwd, timeout_s, user):
            return await busybox_sh(command)

    probe = await busybox_sh("tar --no-overwrite-dir --help >/dev/null 2>&1 && echo gnu || echo busybox")
    assert probe.stdout.strip() == "busybox", "the symlinked tar must be BusyBox's for this test to mean anything"
    assert "--no-overwrite-dir" in tar_extract_command("/tmp/x.tar.gz", str(target))

    await upload_dir(BusyBoxSandbox(), source, str(target))
    assert (target / "test.sh").read_text() == "#!/bin/sh\nprintf works"
    assert stat.S_IMODE((target / "test.sh").stat().st_mode) == 0o751
    assert stat.S_IMODE(target.stat().st_mode) == 0o777


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


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_restored_directory_replaces_what_the_target_held(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "kept.txt").write_text("agent output")
    target.mkdir()
    (target / "stale.txt").write_text("shipped by the verifier image, deleted by the agent")
    (target / "sub").mkdir()
    (target / "sub" / "old.txt").write_text("old")

    await upload_path(LocalSandbox(), source, str(target))

    assert sorted(p.name for p in target.iterdir()) == ["kept.txt"]
    assert stat.S_IMODE(target.stat().st_mode) == 0o777


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_single_file_artifacts_keep_their_mode_bits_both_ways(tmp_path):
    sandbox, agent, host, verifier = LocalSandbox(), tmp_path / "agent", tmp_path / "host", tmp_path / "verifier"
    agent.mkdir()
    tool = agent / "tool.sh"
    tool.write_text("#!/bin/sh\nprintf built")
    tool.chmod(0o755)

    kind = await download_path(sandbox, str(tool), host / "renamed.sh", links="keep")
    assert kind == "file" and stat.S_IMODE((host / "renamed.sh").stat().st_mode) == 0o755
    assert sorted(p.name for p in host.iterdir()) == ["renamed.sh"]

    await upload_path(sandbox, host / "renamed.sh", str(verifier / "bin" / "tool.sh"))
    restored = verifier / "bin" / "tool.sh"
    assert stat.S_IMODE(restored.stat().st_mode) == 0o755 and restored.read_text().endswith("printf built")
    assert stat.S_IMODE((verifier / "bin").stat().st_mode) == 0o777
    process = await asyncio.create_subprocess_exec(str(restored), stdout=asyncio.subprocess.PIPE)
    stdout, _ = await process.communicate()
    assert process.returncode == 0 and stdout == b"built"


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_remote_kind_answers_or_raises(tmp_path):
    sandbox = LocalSandbox()
    (tmp_path / "dir").mkdir()
    (tmp_path / "file").write_text("x")
    assert await remote_kind(sandbox, str(tmp_path / "dir")) == "dir"
    assert await remote_kind(sandbox, str(tmp_path / "file")) == "file"
    assert await remote_kind(sandbox, str(tmp_path / "absent")) is None

    class Outage:
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
            return SandboxExecResult(stdout="", stderr="container gone", return_code=125, error_type="sandbox")

    class NoShell:
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
            return SandboxExecResult(stdout="", stderr="sh: not found", return_code=127)

    # A provider outage and a probe that could not run are both raised, and neither is a transfer failure.
    with pytest.raises(SandboxUnavailableError, match="sandbox: container gone") as info:
        await remote_kind(Outage(), str(tmp_path / "absent"))
    assert not isinstance(info.value, SandboxTransferError)
    with pytest.raises(SandboxProbeError, match="Could not probe .*absent: 127: sh: not found") as info:
        await remote_kind(NoShell(), str(tmp_path / "absent"))
    assert not isinstance(info.value, SandboxTransferError)


async def test_a_link_at_the_artifact_root_is_never_followed_on_restore(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "secret.txt").write_text("host file")
    link = tmp_path / "artifact"
    link.symlink_to(elsewhere, target_is_directory=True)

    class Untouched:
        async def exec(self, *args, **kwargs):
            raise AssertionError("nothing may run in the sandbox")

        upload = exec

    with pytest.raises(SandboxTransferError, match="symbolic link"):
        await upload_path(Untouched(), link, "/app/output")
    with pytest.raises(SandboxTransferError, match="symbolic link"):
        await upload_dir(Untouched(), link, "/app/output")


async def test_provider_failures_are_not_transfer_failures(tmp_path):
    """A command the provider could not run (error_type set) is an outage; a non-zero exit is a transfer failure."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "solve.sh").write_text("#!/bin/sh\n")

    class Fake:
        def __init__(self, result):
            self.result, self.commands = result, []

        async def upload(self, local, remote):
            pass

        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
            self.commands.append(command)
            return self.result

    lost = Fake(SandboxExecResult(stdout="", stderr="", return_code=125, error_type="timeout"))
    with pytest.raises(SandboxUnavailableError, match="could not run .*: timeout") as info:
        await upload_dir(lost, source, "/solution")
    assert not isinstance(info.value, SandboxTransferError)

    refused = Fake(SandboxExecResult(stdout="", stderr="tar: /solution: Permission denied", return_code=2))
    with pytest.raises(SandboxTransferError, match="Permission denied") as info:
        await upload_dir(refused, source, "/solution")
    assert not isinstance(info.value, SandboxUnavailableError)
    with pytest.raises(SandboxUnavailableError):
        await _exec_as_root(lost, "true", timeout_s=5)


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_a_file_restore_accepts_a_writable_parent_as_it_is(tmp_path):
    """An existing parent is widened with `a+rwx`: /tmp keeps its sticky bit, a 0755 parent ends up 0777."""
    host = tmp_path / "host"
    host.mkdir()
    (host / "out.json").write_text("{}")
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o1777)

    await upload_path(LocalSandbox(), host / "out.json", str(shared / "out.json"))

    assert (shared / "out.json").read_text() == "{}" and stat.S_IMODE(shared.stat().st_mode) == 0o1777
    # A parent the image ships as 0755 is widened, as Harbor does, so a verifier that drops privileges can
    # write next to the restored file.
    narrow = tmp_path / "app" / "output"
    narrow.mkdir(parents=True)
    narrow.chmod(0o755)
    await upload_path(LocalSandbox(), host / "out.json", str(narrow / "out.json"))
    assert (narrow / "out.json").read_text() == "{}" and stat.S_IMODE(narrow.stat().st_mode) == 0o777
    # A parent that does not exist yet is created world-writable, as before.
    await upload_path(LocalSandbox(), host / "out.json", str(tmp_path / "fresh" / "out.json"))
    assert stat.S_IMODE((tmp_path / "fresh").stat().st_mode) == 0o777


async def test_a_file_restore_into_a_parent_the_user_cannot_chmod_succeeds(tmp_path):
    """The before/after case: a non-root image where `chmod 777 /tmp` fails but /tmp is writable."""
    host = tmp_path / "host"
    host.mkdir()
    (host / "out.json").write_text("{}")

    class NonRootImage:
        def __init__(self):
            self.commands, self.uploads = [], []

        async def upload(self, local, remote):
            self.uploads.append(remote)

        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
            self.commands.append(command)
            if user == "root":
                return SandboxExecResult(stdout="", stderr="exec: operation not permitted", return_code=1)
            if "chmod" in command and "/tmp" in command and "if [ -d" not in command:
                return SandboxExecResult(stdout="", stderr="chmod: /tmp: Operation not permitted", return_code=1)
            return SandboxExecResult(stdout="", stderr="", return_code=0)

    sandbox = NonRootImage()
    await upload_path(sandbox, host / "out.json", "/tmp/out.json")
    prepare = next(c for c in sandbox.commands if c.startswith("if [ -d /tmp ]"))
    assert prepare == (
        "if [ -d /tmp ]; then chmod a+rwx /tmp 2>/dev/null || test -w /tmp; else mkdir -p /tmp && chmod 777 /tmp; fi"
    )
    assert sandbox.uploads and any("tar -xzf" in c and "-C /tmp;" in c for c in sandbox.commands)
