# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import os
import shutil
import stat
import tarfile

import pytest

from nemo_gym.sandbox import (
    SandboxExecResult,
    SandboxProbeError,
    SandboxTransferError,
    SandboxUnavailableError,
    download_dir,
    download_path,
    exec_as_root,
    remote_kind,
    upload_archive,
    upload_dir,
    upload_path,
)
from nemo_gym.sandbox.transfer import _pack, _unpack, tar_extract_command


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
@pytest.mark.parametrize(
    "message",
    [
        "identity switch requires CAP_SETUID",
        # Verbatim OpenSandbox stderr when the pod cannot switch identity.
        "fork/exec /usr/bin/bash: operation not permitted (switching to uid=0 requires CAP_SETUID)",
    ],
)
@pytest.mark.parametrize("exception", [False, True])
async def test_root_fallback_when_identity_switch_is_unavailable(exception, message):
    class Sandbox:
        def __init__(self):
            self.users = []

        async def exec(self, command, *, cwd, timeout_s, user=None):
            self.users.append(user)
            if user == "root":
                if exception:
                    raise RuntimeError(message)
                return SandboxExecResult(stdout="", stderr=message, return_code=1)
            return SandboxExecResult(stdout="ready", stderr="", return_code=0)

    sandbox = Sandbox()
    result = await exec_as_root(sandbox, "mkdir -p /tests", timeout_s=1)
    assert result.return_code == 0 and sandbox.users == ["root", None]


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [False, True])
async def test_root_fallback_with_a_provider_error_propagates_as_unavailable(exception):
    """A refusal to switch identity (raised or returned) followed by a provider failure on the default-user
    retry is a lost sandbox in both shapes, not a transfer result the caller scores as an oracle failure."""
    message = "identity switch requires CAP_SETUID"

    class Sandbox:
        def __init__(self):
            self.users = []

        async def exec(self, command, *, cwd, timeout_s, user=None):
            self.users.append(user)
            if user == "root":
                if exception:
                    raise RuntimeError(message)
                return SandboxExecResult(stdout="", stderr=message, return_code=1)
            return SandboxExecResult(stdout="", stderr="", return_code=125, error_type="timeout")

    sandbox = Sandbox()
    with pytest.raises(SandboxUnavailableError, match="could not run .*: timeout") as info:
        await exec_as_root(sandbox, "mkdir -p /logs/agent", timeout_s=1)
    assert not isinstance(info.value, SandboxTransferError)
    assert sandbox.users == ["root", None]


@pytest.mark.asyncio
async def test_root_command_failure_does_not_retry_as_default_user():
    calls = []

    class Sandbox:
        async def exec(self, command, **kwargs):
            calls.append(kwargs["user"])
            return SandboxExecResult(stdout="", stderr="disk full", return_code=1)

    result = await exec_as_root(Sandbox(), "mkdir -p /tests", timeout_s=1)
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


def test_pack_leaves_out_the_names_the_task_digest_skips_when_asked(tmp_path):
    """Review (round 14): ``upload_dir`` shipped a task's ``tests/`` or ``environment/`` whole, ``helper.pyc`` and
    ``__pycache__/`` included, which ``content_hash`` skips: importable bytes outside the digest. With ``ignore``
    (the digest's own rule) the archive holds exactly what the digest covers, a folder left out taking its
    subtree with it; without it (artifacts restored into the verifier) everything travels as before."""
    from nemo_gym.tasks.harbor.digest import is_ignored

    source = tmp_path / "tests"
    (source / "__pycache__").mkdir(parents=True)
    (source / "sub").mkdir()
    (source / "test.sh").write_text("echo 1\n")
    (source / "helper.pyc").write_bytes(b"\x00")
    (source / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (source / "sub" / "data.txt").write_text("d")
    (source / "sub" / "cached.pyc").write_bytes(b"\x00")
    archive = tmp_path / "upload.tar.gz"
    _pack(source, archive, ignore=is_ignored)
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == ["sub", "sub/data.txt", "test.sh"]
    _pack(source, archive)
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == [
            "__pycache__",
            "__pycache__/x.pyc",
            "helper.pyc",
            "sub",
            "sub/cached.pyc",
            "sub/data.txt",
            "test.sh",
        ]


@pytest.mark.asyncio
async def test_upload_dir_with_the_digest_rule_ships_what_the_digest_covers(tmp_path):
    from nemo_gym.tasks.harbor.digest import is_ignored

    source, target = tmp_path / "tests", tmp_path / "target"
    (source / "__pycache__").mkdir(parents=True)
    target.mkdir()
    (source / "test.sh").write_text("echo 1\n")
    (source / "helper.pyc").write_bytes(b"\x00")
    (source / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    await upload_dir(LocalSandbox(), source, str(target), ignore=is_ignored)
    assert sorted(path.name for path in target.iterdir()) == ["test.sh"]


async def busybox_sandbox(bin_dir):
    """A sandbox whose shell and ``tar`` are BusyBox's, as in an Alpine image: ``tar`` in ``bin_dir`` resolves to
    the applet and every command runs under ``busybox sh`` with that directory first on the PATH."""
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
    return BusyBoxSandbox()


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("busybox") is None, reason="BusyBox is required")
async def test_transfer_works_with_busybox_tar_which_has_no_no_overwrite_dir_option(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    script = source / "test.sh"
    script.write_text("#!/bin/sh\nprintf works")
    script.chmod(0o751)
    target.mkdir()
    target.chmod(0o777)
    sandbox = await busybox_sandbox(tmp_path / "bin")
    assert "--no-overwrite-dir" in tar_extract_command("/tmp/x.tar.gz", str(target))

    await upload_dir(sandbox, source, str(target))
    assert (target / "test.sh").read_text() == "#!/bin/sh\nprintf works"
    assert stat.S_IMODE((target / "test.sh").stat().st_mode) == 0o751
    assert stat.S_IMODE(target.stat().st_mode) == 0o777


def _plant_links(target, elsewhere):
    """What a sandbox user who owns ``target`` can leave at the names a later upload ships: a link at a top-level
    directory name, one at a nested directory name and one at a file name, all pointing into ``elsewhere``."""
    for path in (target / "fixtures", target / "sub" / "deep", target / "setup.sh"):
        if path.exists() or path.is_symlink():
            shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
    (target / "sub").mkdir(exist_ok=True)
    (target / "fixtures").symlink_to(elsewhere)
    (target / "sub" / "deep").symlink_to(elsewhere)
    (target / "setup.sh").symlink_to(elsewhere / "setup.sh")
    for leftover in elsewhere.iterdir():
        leftover.unlink()


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("busybox") is None, reason="BusyBox is required")
async def test_a_merge_lands_where_its_names_say_over_links_planted_at_member_paths_on_either_tar(tmp_path):
    """A `workdir/` merged into a directory the agent owns meets whatever the agent planted at the shipped names. GNU
    tar replaces a link at a directory member's path with the directory; BusyBox tar keeps the link and writes the
    member's children through it, so the files would land, as root, wherever the agent pointed the link. With
    `replace_links` the extraction removes a link at every member's path first, and both tars land the merge at its
    own names with nothing written through; without it BusyBox tar is diverted (what the option is for)."""
    source, target, elsewhere = tmp_path / "workdir", tmp_path / "app", tmp_path / "elsewhere"
    for relative, text in (("fixtures/input.txt", "data\n"), ("sub/deep/x.txt", "x\n"), ("setup.sh", "true\n")):
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        (source / relative).write_text(text)
    target.mkdir()
    elsewhere.mkdir()
    busybox = await busybox_sandbox(tmp_path / "bin")

    def landed():
        return (
            sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file() or p.is_symlink()),
            sorted(p.name for p in elsewhere.iterdir()),
            [str(p.relative_to(target)) for p in target.rglob("*") if p.is_symlink()],
        )

    # The hazard, on BusyBox tar: the directory links stay and the shipped files are written through them (the file
    # link is replaced, as a regular file member unlinks what is at its path).
    _plant_links(target, elsewhere)
    await upload_dir(busybox, source, str(target))
    files, diverted, links = landed()
    assert diverted == ["input.txt", "x.txt"] and links == ["fixtures", "sub/deep"]
    assert "setup.sh" in files and not (target / "setup.sh").is_symlink()

    # With replace_links, on BusyBox tar and on GNU tar alike: the links are gone, the merge is at its own names.
    for sandbox in (busybox, LocalSandbox()):
        _plant_links(target, elsewhere)
        await upload_dir(sandbox, source, str(target), replace_links=True)
        files, diverted, links = landed()
        assert (files, diverted, links) == (["fixtures/input.txt", "setup.sh", "sub/deep/x.txt"], [], [])
        assert (target / "fixtures" / "input.txt").read_text() == "data\n"
    # The plain extraction is unchanged: no listing pass, the shell line as before.
    plain = tar_extract_command("/tmp/x.tar.gz", str(target))
    assert "tar -tzf" not in plain and plain.endswith(f"-C {target}")
    clearing = tar_extract_command("/tmp/x.tar.gz", str(target), replace_links=True)
    assert clearing.index("tar -tzf /tmp/x.tar.gz | while IFS= read -r m") < clearing.index("tar -xzf")


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


@pytest.mark.asyncio
@pytest.mark.parametrize("tar", ["busybox", "gnu"])
async def test_reward_collection_by_name_works_with_busybox_tar(tmp_path, tar):
    """M35 D7: the ``names`` filter listed the names NUL-separated to ``tar --null -T -``; ``--null`` is GNU-only and
    BusyBox tar (Alpine task images) answers ``tar: unrecognized option: null``, exit 1, so the reward of every
    hil-bench task on such an image was lost (19/100) although ``test.sh`` had written it. The names are fixed,
    plain file names: the ones present are passed to ``tar`` as arguments, which both tars take; none present
    gives the empty archive without ``tar``. A source that is gone fails the collection as before, and never
    archives those names from the exec's own working directory (repair round 1: ``cd X && names=''; for ...``
    guarded only the assignment, so a missing ``/logs/verifier`` collected ``/reward.txt`` with exit 0)."""
    import os

    busybox = shutil.which("busybox")
    if tar == "busybox" and busybox is None:
        pytest.skip("busybox is not installed")
    source, target, bin_dir = tmp_path / "logs", tmp_path / "collected", tmp_path / "bin"
    source.mkdir()
    (source / "reward.txt").write_text("1")
    (source / "test-stdout.txt").write_text("not collected by name")
    bin_dir.mkdir()
    if tar == "busybox":
        (bin_dir / "tar").symlink_to(busybox)
    # The exec's own working directory holds a decoy reward, as `/` does in a sandbox whose agent planted one.
    decoy = tmp_path / "exec-cwd"
    decoy.mkdir()
    (decoy / "reward.txt").write_text("planted")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}
    commands = []

    class Sandbox(LocalSandbox):
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
            commands.append(command)
            argv = ["busybox", "sh", "-c", command] if tar == "busybox" else ["bash", "-c", command]
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd or decoy,
                env=sandbox_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
            return SandboxExecResult(stdout.decode(), stderr.decode(), process.returncode)

    sandbox_env = env
    if tar == "busybox":
        probe = await Sandbox().exec("tar --null --help >/dev/null 2>&1 && echo gnu || echo busybox", timeout_s=30)
        assert probe.stdout.strip() == "busybox", "the symlinked tar must be BusyBox's for this test to mean anything"
    await download_dir(Sandbox(), str(source), target, names=("reward.json", "reward.txt", "exit-record.txt"))
    assert (target / "reward.txt").read_text() == "1"
    assert not (target / "test-stdout.txt").exists()
    (archive,) = [command for command in commands if command.startswith("cd ")]
    assert "--null" not in archive and " -T " not in archive
    # No name present: an empty archive, as a verifier that wrote no reward.
    empty = tmp_path / "empty-collected"
    await download_dir(Sandbox(), str(tmp_path / "bin"), empty, names=("reward.json", "reward.txt"))
    assert empty.is_dir() and list(empty.iterdir()) == []
    # The source is gone (removed by test.sh, a sidecar or an agent process left running): a failure, not the
    # decoy from the exec's working directory and not a measured zero.
    gone = tmp_path / "gone-collected"
    with pytest.raises(SandboxTransferError, match="Failed to archive"):
        await download_dir(Sandbox(), str(tmp_path / "logs" / "missing"), gone, names=("reward.json", "reward.txt"))
    assert list(gone.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tar", ["busybox", "gnu"])
async def test_a_single_file_artifact_whose_name_has_whitespace_or_a_leading_dash_is_collected(tmp_path, tar):
    """Repair round 3: ``download_path`` collects a single file through the ``names`` branch, and a task's
    ``[[artifacts]]`` source is any name. The BusyBox-safe form of the branch collected the names in one string and
    expanded it unquoted, so ``my report.txt`` became two arguments (``tar: report.txt: Cannot stat``, the artifact
    logged as not collected, and a decoy ``my`` collected in its place). The names are the shell's positional
    parameters now, passed after ``--``, so whitespace and a leading ``-`` travel as the name."""
    import os

    busybox = shutil.which("busybox")
    if tar == "busybox" and busybox is None:
        pytest.skip("busybox is not installed")
    out, host, bin_dir = tmp_path / "app" / "out", tmp_path / "host", tmp_path / "bin"
    out.mkdir(parents=True)
    (out / "my report.txt").write_text("the report")
    (out / "my").write_text("decoy")
    (out / "-dash.txt").write_text("dash")
    bin_dir.mkdir()
    if tar == "busybox":
        (bin_dir / "tar").symlink_to(busybox)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}

    class Sandbox(LocalSandbox):
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
            argv = ["busybox", "sh", "-c", command] if tar == "busybox" else ["bash", "-c", command]
            process = await asyncio.create_subprocess_exec(
                *argv, cwd=cwd, env=sandbox_env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
            return SandboxExecResult(stdout.decode(), stderr.decode(), process.returncode)

    sandbox_env = env
    if tar == "busybox":
        probe = await Sandbox().exec("tar --null --help >/dev/null 2>&1 && echo gnu || echo busybox", timeout_s=30)
        assert probe.stdout.strip() == "busybox", "the symlinked tar must be BusyBox's for this test to mean anything"
    assert await download_path(Sandbox(), str(out / "my report.txt"), host / "my report.txt") == "file"
    assert (host / "my report.txt").read_text() == "the report"
    assert sorted(p.name for p in host.iterdir()) == ["my report.txt"]
    assert await download_path(Sandbox(), str(out / "-dash.txt"), host / "renamed.txt") == "file"
    assert (host / "renamed.txt").read_text() == "dash"
    assert sorted(p.name for p in host.iterdir()) == ["my report.txt", "renamed.txt"]


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


def test_keep_mode_refuses_device_nodes_and_fifos_like_data_mode(tmp_path, caplog):
    """`tar_filter` alone would create an agent's device node or FIFO on this host (a character device raised
    PermissionError out of the per-artifact handler, a FIFO was created silently): both are refused in both
    modes, and skipped with a warning where the caller asks for that."""
    archive = tmp_path / "special.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        device = tarfile.TarInfo("./dev-null")
        device.type, device.devmajor, device.devminor, device.mode = tarfile.CHRTYPE, 1, 3, 0o666
        tar.addfile(device)
        fifo = tarfile.TarInfo("./pipe")
        fifo.type = tarfile.FIFOTYPE
        tar.addfile(fifo)
        plain = tarfile.TarInfo("./report.txt")
        plain.size = 2
        tar.addfile(plain, io.BytesIO(b"ok"))
    for links in ("keep", "data"):
        with pytest.raises(tarfile.SpecialFileError, match="special file"):
            _unpack(archive, tmp_path / f"refused-{links}", links=links)
        assert not (tmp_path / f"refused-{links}" / "pipe").exists()
    with caplog.at_level("WARNING"):
        _unpack(archive, tmp_path / "skipped", links="keep", skip_unsafe=True)
    assert sorted(p.name for p in (tmp_path / "skipped").iterdir()) == ["report.txt"]
    assert "Skipping unsafe diagnostic archive member './dev-null'" in caplog.text
    assert "Skipping unsafe diagnostic archive member './pipe'" in caplog.text


async def test_a_refused_archive_leaves_no_archive_copy_behind(tmp_path):
    """The archive kept for the verifier exists only when the host's filtered copy was accepted."""
    payload = tmp_path / "payload.tar.gz"
    _archive(payload, [("../outside.txt", None)])

    class Fake:
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
            return SandboxExecResult(stdout="", stderr="", return_code=0)

        async def download(self, remote, local):
            shutil.copyfile(payload, local)

    kept = tmp_path / "artifacts" / "00.tar.gz"
    with pytest.raises(tarfile.FilterError):
        await download_dir(Fake(), "/app/output", tmp_path / "artifacts" / "00", links="keep", archive_copy=kept)
    assert not kept.exists()


def test_kept_artifacts_stay_readable_by_their_owner(tmp_path):
    """The host staging copy of a tree the agent left unreadable can still be re-archived; the host copy only,
    the verifier receives the archive itself (see the mode round trip below)."""
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


class RootLikeSandbox(LocalSandbox):
    """A LocalSandbox whose tar extracts as root's does: root ignores the umask, this user's tar applies it."""

    async def exec(self, command, **kwargs):
        return await super().exec(f"umask 0; {command}", **kwargs)


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_the_verifier_receives_the_modes_the_agent_left_while_the_host_copy_is_widened(tmp_path):
    """The reviewer's case: a 0444 artifact came back 0644 in the verifier because the host staging copy's widened
    mode (tar_filter plus `| 0o600`) was re-archived. The verifier now receives the agent sandbox's own archive, so
    read-only and world-writable modes survive; the host copy keeps its safe, widened modes."""
    sandbox, agent, host, verifier = RootLikeSandbox(), tmp_path / "agent", tmp_path / "host", tmp_path / "verifier"
    output = agent / "output"
    output.mkdir(parents=True)
    (output / "ro.txt").write_text("read-only")
    (output / "ro.txt").chmod(0o444)
    (output / "ww.txt").write_text("world-writable")
    (output / "ww.txt").chmod(0o666)
    (output / "wwdir").mkdir()
    (output / "wwdir" / "x.sh").write_text("#!/bin/sh\nprintf built")
    (output / "wwdir" / "x.sh").chmod(0o755)
    (output / "wwdir").chmod(0o777)
    archive = tmp_path / "artifacts" / "00.tar.gz"

    kind = await download_path(sandbox, str(output), host / "output", links="keep", archive_copy=archive)
    assert kind == "dir" and archive.is_file()
    staged = {name: stat.S_IMODE((host / "output" / name).stat().st_mode) for name in ("ro.txt", "ww.txt", "wwdir")}
    assert staged == {"ro.txt": 0o644, "ww.txt": 0o644, "wwdir": 0o755}

    (verifier / "app" / "output").mkdir(parents=True)
    (verifier / "app" / "output" / "stale").write_text("shipped by the verifier image")
    await upload_archive(sandbox, archive, kind, str(verifier / "app" / "output"))
    restored = verifier / "app" / "output"
    assert sorted(p.name for p in restored.iterdir()) == ["ro.txt", "ww.txt", "wwdir"]
    got = {name: stat.S_IMODE((restored / name).stat().st_mode) for name in ("ro.txt", "ww.txt", "wwdir")}
    assert got == {"ro.txt": 0o444, "ww.txt": 0o666, "wwdir": 0o777}
    assert stat.S_IMODE((restored / "wwdir" / "x.sh").stat().st_mode) == 0o755
    # The restore target itself is prepared as before: cleared and world-writable, its own mode not overwritten
    # by the archive's `./` entry.
    assert stat.S_IMODE(restored.stat().st_mode) == 0o777

    # A single read-only file travels the same way.
    single = tmp_path / "artifacts" / "01.tar.gz"
    kind = await download_path(sandbox, str(output / "ro.txt"), host / "ro.txt", links="keep", archive_copy=single)
    assert kind == "file" and stat.S_IMODE((host / "ro.txt").stat().st_mode) == 0o644
    await upload_archive(sandbox, single, kind, str(verifier / "etc" / "ro.txt"))
    assert stat.S_IMODE((verifier / "etc" / "ro.txt").stat().st_mode) == 0o444
    assert (verifier / "etc" / "ro.txt").read_text() == "read-only"
    assert stat.S_IMODE((verifier / "etc").stat().st_mode) == 0o777


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_an_unreadable_file_round_trips_as_the_agent_left_it(tmp_path):
    """A 0000 file (which only root's tar can pack in the agent sandbox, so the archive is built here): the
    host copy is readable by its owner, the verifier's copy is 0000 again."""
    archive = tmp_path / "artifacts" / "00.tar.gz"
    archive.parent.mkdir()
    with tarfile.open(archive, "w:gz") as tar:
        hidden = tarfile.TarInfo("./secret.bin")
        hidden.mode = 0o000
        hidden.size = 2
        tar.addfile(hidden, io.BytesIO(b"ok"))
    _unpack(archive, tmp_path / "host", links="keep")
    assert stat.S_IMODE((tmp_path / "host" / "secret.bin").stat().st_mode) == 0o600
    await upload_archive(RootLikeSandbox(), archive, "dir", str(tmp_path / "verifier" / "out"))
    assert stat.S_IMODE((tmp_path / "verifier" / "out" / "secret.bin").stat().st_mode) == 0o000


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_keep_out_refuses_a_restore_that_a_link_diverts_onto_a_kept_directory(tmp_path):
    """``keep_out`` judges where the archive physically lands, not what its target is called: a target that is a
    link into a kept directory, or above one, is refused before anything is extracted; a target that is where its
    name says is extracted even when that name overlaps a kept path (the caller judged the name); without
    ``keep_out`` no probe runs; a sandbox that answers the probe with nothing is taken at the name, and one that
    cannot run it is an outage."""

    class Recording(RootLikeSandbox):
        def __init__(self, answer=None):
            self.commands, self.answer = [], answer

        async def exec(self, command, **kwargs):
            self.commands.append(command)
            if self.answer is not None and command.endswith("pwd -P"):
                return self.answer
            return await super().exec(command, **kwargs)

    archive = tmp_path / "00.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("./test.sh")
        member.size = 2
        tar.addfile(member, io.BytesIO(b"ok"))
    verifier = tmp_path / "verifier"
    tests, logs, app = verifier / "tests", verifier / "logs", verifier / "app"
    tests.mkdir(parents=True)
    (tests / "test.sh").write_text("trusted")
    app.mkdir()
    (app / "out").symlink_to(tests, target_is_directory=True)
    (app / "root").symlink_to(verifier, target_is_directory=True)
    keep_out = (str(tests), str(logs / "verifier"))

    sandbox = Recording()
    # Into a kept directory through a link, as a directory and as a file whose parent is the link.
    assert await upload_archive(sandbox, archive, "dir", str(app / "out"), keep_out=keep_out) is False
    assert await upload_archive(sandbox, archive, "file", str(app / "out" / "test.sh"), keep_out=keep_out) is False
    # Above a kept directory through a link.
    assert await upload_archive(sandbox, archive, "dir", str(app / "root"), keep_out=keep_out) is False
    # The same places spelled with two leading slashes, which bash keeps for such a cwd and PurePosixPath keeps
    # apart from one, on the target and on a kept path.
    assert await upload_archive(sandbox, archive, "dir", "/" + str(app / "out"), keep_out=keep_out) is False
    assert await upload_archive(sandbox, archive, "dir", str(app / "out"), keep_out=("/" + str(tests),)) is False
    assert (tests / "test.sh").read_text() == "trusted" and (app / "out").is_symlink()
    assert not any("tar -xzf" in command for command in sandbox.commands)
    assert sum(command.endswith("pwd -P") for command in sandbox.commands) == 5
    # The named path itself, above a kept directory: the caller's call, extracted.
    (logs / "verifier").mkdir(parents=True)
    assert await upload_archive(sandbox, archive, "dir", str(logs), keep_out=keep_out) is True
    assert (logs / "test.sh").read_text() == "ok"
    # Without keep_out the target is not resolved at all.
    plain = Recording()
    assert await upload_archive(plain, archive, "dir", str(app / "elsewhere")) is True
    assert not any(command.endswith("pwd -P") for command in plain.commands)
    # A sandbox that answers nothing leaves the name as the only judge; one that cannot answer raises.
    mute = Recording(SandboxExecResult(stdout="", stderr="", return_code=0))
    assert await upload_archive(mute, archive, "dir", str(app / "out"), keep_out=keep_out) is True
    assert (tests / "test.sh").read_text() == "ok"
    broken = Recording(SandboxExecResult(stdout="", stderr="cd: no such directory", return_code=1))
    with pytest.raises(SandboxTransferError, match="Could not resolve"):
        await upload_archive(broken, archive, "dir", str(app / "out"), keep_out=keep_out)


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_keep_out_is_judged_before_the_target_is_prepared_so_a_refusal_touches_nothing(tmp_path):
    """The reviewer's case: with ``/app/out -> /tests``, restoring ``/app/out/suite`` or ``/app/out/`` used to be
    prepared first (``mkdir -p``, ``find -delete``, ``chmod``) and refused only afterwards, so the trusted grader
    files were gone by the time the refusal was logged. The physical destination is now resolved before anything
    touches the target: a target under a linked parent, one spelled with a trailing slash, a file whose parent is
    under the link, and a target that does not exist yet below the link are all refused with the kept directory
    byte-identical and no destructive command run, while a target that merely does not exist yet is extracted."""

    class Recording(RootLikeSandbox):
        def __init__(self):
            self.commands = []

        async def exec(self, command, **kwargs):
            self.commands.append(command)
            return await super().exec(command, **kwargs)

    def snapshot(root):
        return {
            str(path.relative_to(root)): (path.read_bytes() if path.is_file() else None, path.stat().st_mode)
            for path in sorted(root.rglob("*"))
        }

    archive = tmp_path / "00.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("./helper.sh")
        member.size = 7
        tar.addfile(member, io.BytesIO(b"planted"))
    verifier = tmp_path / "verifier"
    tests, logs, app = verifier / "tests", verifier / "logs", verifier / "app"
    (tests / "suite").mkdir(parents=True)
    (tests / "test.sh").write_text("trusted")
    (tests / "suite" / "helper.sh").write_text("trusted helper")
    (tests / "suite" / "helper.sh").chmod(0o500)
    app.mkdir()
    (app / "out").symlink_to(tests, target_is_directory=True)
    keep_out = (str(tests), str(logs / "verifier"))
    trusted = snapshot(tests)

    sandbox = Recording()
    # A directory under the linked parent, the same with a trailing slash, the link itself with a trailing slash,
    # a file whose parent is under the link, and a directory that does not exist yet under the link.
    for kind, target in (
        ("dir", str(app / "out" / "suite")),
        ("dir", str(app / "out" / "suite") + "/"),
        ("dir", str(app / "out") + "/"),
        ("file", str(app / "out" / "suite" / "helper.sh")),
        ("dir", str(app / "out" / "new" / "deeper")),
        ("file", str(app / "out" / "new" / "helper.sh")),
    ):
        assert await upload_archive(sandbox, archive, kind, target, keep_out=keep_out) is False, target
        assert snapshot(tests) == trusted, target
    assert (app / "out").is_symlink() and not (tests / "new").exists()
    # Every command the refusals ran was the read-only probe: nothing was created, emptied, opened or extracted.
    assert len(sandbox.commands) == 6 and all(command.endswith("pwd -P") for command in sandbox.commands)
    assert not any(word in command for command in sandbox.commands for word in ("mkdir", "find ", "chmod", "tar "))
    # A target that does not exist yet and is not diverted is prepared and extracted as before, so the probe's
    # walk up to the nearest existing ancestor keeps a legitimate restore working.
    fresh = app / "fresh" / "sub"
    assert await upload_archive(sandbox, archive, "dir", str(fresh), keep_out=keep_out) is True
    assert (fresh / "helper.sh").read_text() == "planted" and snapshot(tests) == trusted
    assert sandbox.commands[6].endswith("pwd -P") and "mkdir -p" in sandbox.commands[7]


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_exclude_patterns_reach_tar_unanchored_as_harbor_passes_them(tmp_path):
    """Harbor passes each artifact `exclude` pattern to `tar --exclude` verbatim, so `node_modules` leaves out
    every directory of that name; the anchored form kept `sub/node_modules`. Asserted on the extracted tree,
    which checking the generated flags alone misses."""
    source = tmp_path / "build"
    for relative in ("node_modules/x", "sub/node_modules/y", "sub/k", "cache/c", "a.log", "sub/b.log"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    target = tmp_path / "host"

    await download_dir(LocalSandbox(), str(source), target, exclude=("node_modules", "./cache", "*.log"))

    files = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file())
    assert files == ["sub/k"]
    # `./cache` names the top-level folder only (member names start with `./`): a nested cache would stay.
    (source / "sub" / "cache").mkdir()
    (source / "sub" / "cache" / "n").write_text("nested")
    nested = tmp_path / "host2"
    await download_dir(LocalSandbox(), str(source), nested, exclude=("node_modules", "./cache", "*.log"))
    assert sorted(str(p.relative_to(nested)) for p in nested.rglob("*") if p.is_file()) == ["sub/cache/n", "sub/k"]


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
        await exec_as_root(lost, "true", timeout_s=5)


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
