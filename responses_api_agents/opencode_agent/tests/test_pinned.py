# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned files: the staging steps a sandbox shell runs, and the archive check on the agent server."""

import hashlib
import io
import json
import os
import posixpath
import shutil
import subprocess
import sys
import tarfile

import pytest

from responses_api_agents.opencode_agent.pinned import (
    COPY_PINNED,
    REJECTED,
    STAGE_EXECUTABLE,
    UNPACK_ARCHIVE,
    check_archive,
)


pytestmark = pytest.mark.skipif(shutil.which("sha256sum") is None, reason="the staging steps need sha256sum")
linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux reports a missing interpreter as 127")


def run(script, *args, env=None):
    """Run a staging step as a sandbox's ``sh -c`` does, with its positional parameters."""
    return subprocess.run(
        ["/bin/sh", "-c", script, "sh", *map(str, args)], capture_output=True, text=True, timeout=60, env=env
    )


def pinned(path, data):
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def stage(tmp_path, source, digest):
    target = tmp_path / "session/cache/opencode/bin/rg"
    return run(STAGE_EXECUTABLE, source, digest, target.with_name(".pinned-rg"), target), target


def test_stage_executable_installs_a_checked_copy_that_runs(tmp_path):
    source = tmp_path / "mounted-rg"
    digest = pinned(source, b"#!/bin/sh\necho 'ripgrep 15.1.0'\n")
    source.chmod(0o444)
    staged, target = stage(tmp_path, source, digest)
    assert staged.returncode == 0, staged.stderr
    assert target.read_bytes() == source.read_bytes()
    assert os.access(target, os.X_OK)
    assert [path.name for path in target.parent.iterdir()] == ["rg"]


@pytest.mark.parametrize(
    ("script", "digest", "message"),
    [
        (b"#!/bin/sh\necho 'ripgrep 15.1.0'\n", "0" * 64, "SHA-256 mismatch: expected " + "0" * 64),
        (b"#!/bin/sh\nexit 4\n", None, "--version exited 4"),
        # Bytes the kernel refuses to run, which the shell reports as 126.
        (b"\x7fELF\x02\x01\x01\x00\x00\x00not really a binary\n", None, "--version exited 126"),
    ],
)
def test_stage_executable_rejects_wrong_or_broken_bytes_without_installing(tmp_path, script, digest, message):
    source = tmp_path / "rg"
    actual = pinned(source, script)
    staged, target = stage(tmp_path, source, digest or actual)
    assert staged.returncode == REJECTED, staged.stderr
    assert message in staged.stderr
    assert list(target.parent.iterdir()) == []


def test_a_missing_source_or_tool_stays_retryable(tmp_path):
    staged, _ = stage(tmp_path, tmp_path / "absent", "0" * 64)
    assert staged.returncode == 1 and "is missing" in staged.stderr
    source = tmp_path / "rg"
    digest = pinned(source, b"#!/bin/sh\necho ok\n")
    empty = tmp_path / "empty-path"
    empty.mkdir()
    staged = run(
        STAGE_EXECUTABLE, source, digest, tmp_path / ".pinned-rg", tmp_path / "rg-out", env={"PATH": str(empty)}
    )
    assert staged.returncode == 1 and "requires sha256sum" in staged.stderr


def test_version_check_killed_by_a_signal_stays_retryable(tmp_path):
    source = tmp_path / "rg"
    digest = pinned(source, b"#!/bin/sh\nkill -KILL $$\n")
    staged, target = stage(tmp_path, source, digest)
    assert staged.returncode == 1, staged.stderr
    assert "--version exited 137" in staged.stderr
    assert list(target.parent.iterdir()) == []


@linux_only
def test_a_missing_interpreter_stays_retryable(tmp_path):
    source = tmp_path / "rg"
    digest = pinned(source, b"#!/nonexistent/interpreter\n")
    staged, target = stage(tmp_path, source, digest)
    assert staged.returncode == 1, staged.stderr
    assert "--version exited 127" in staged.stderr
    assert list(target.parent.iterdir()) == []


def plugin_archive(tmp_path, *members):
    """A config-directory bundle as tar builds it from an online run: files, folders and relative links.

    Each member is ``(name, kind, data)``, or ``(name, kind, data, mode)``.
    """
    archive = tmp_path / "plugin-dependencies.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, kind, data, *mode in members:
            info = tarfile.TarInfo(name)
            info.mode = mode[0] if mode else 0o755
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                bundle.addfile(info)
            elif kind in ("symlink", "hardlink"):
                info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                info.linkname = data
                bundle.addfile(info)
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
                bundle.addfile(info)
            else:
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def unpack(tmp_path, archive, digest):
    """Stage an archive as the adapter does: copy and check it in the sandbox, check its members, then unpack it."""
    target = tmp_path / "session/config/opencode"
    copy = target.with_name(".pinned-opencode")
    copied = run(COPY_PINNED, archive, digest, copy)
    assert copied.returncode == 0, copied.stderr
    check_archive(copy, digest)
    unpacked = run(UNPACK_ARCHIVE, copy, target)
    assert unpacked.returncode == 0, unpacked.stderr
    return target


# The real archive's links (npm's node_modules/.bin), plus a scoped and a nested package.
NPM_LINKS = [
    ("node_modules/.bin/uuid", "../uuid/dist-node/bin/uuid"),
    ("node_modules/.bin/yaml", "../yaml/bin.mjs"),
    ("node_modules/.bin/node-which", "../which/bin/node-which"),
    ("node_modules/.bin/plugin-tool", "../@opencode-ai/plugin/bin/tool.js"),
    ("node_modules/zod/node_modules/.bin/semver", "../semver/bin/semver.js"),
]


def test_plugin_dependencies_unpack_with_their_links(tmp_path):
    targets = [posixpath.normpath(posixpath.join(posixpath.dirname(name), link)) for name, link in NPM_LINKS]
    archive, digest = plugin_archive(
        tmp_path,
        ("package.json", "file", b'{"dependencies": {"@opencode-ai/plugin": "1.17.11"}}'),
        ("package-lock.json", "file", b'{"packages": {"": {"dependencies": {"@opencode-ai/plugin": "1.17.11"}}}}'),
        ("node_modules/@opencode-ai/plugin/package.json", "file", b'{"version": "1.17.11"}'),
        *((path, "file", b"#!/usr/bin/env node\n") for path in targets),
        ("node_modules/.bin", "dir", b""),
        *((name, "symlink", link) for name, link in NPM_LINKS),
        ("node_modules/.package-lock.json", "hardlink", "package-lock.json"),
    )
    target = unpack(tmp_path, archive, digest)
    assert json.loads((target / "node_modules/@opencode-ai/plugin/package.json").read_text()) == {"version": "1.17.11"}
    for (name, _), linked in zip(NPM_LINKS, targets):
        assert (target / name).is_symlink()
        assert (target / name).resolve() == (target / linked).resolve()
    assert (target / "node_modules/.package-lock.json").read_bytes() == (target / "package-lock.json").read_bytes()
    assert not list((tmp_path / "session/config").glob(".pinned-*"))


def test_an_archive_copy_with_the_wrong_digest_is_rejected_and_removed(tmp_path):
    archive, _ = plugin_archive(tmp_path, ("package.json", "file", b"{}"))
    copy = tmp_path / "session/config/.pinned-opencode"
    copied = run(COPY_PINNED, archive, "0" * 64, copy)
    assert copied.returncode == REJECTED and "SHA-256 mismatch" in copied.stderr
    assert not copy.exists()


@pytest.mark.parametrize(
    "member",
    [
        ("../outside.json", "file", b"{}"),
        ("/etc/opencode.json", "file", b"{}"),
        ("node_modules/escape", "symlink", "../../../../outside"),
        ("node_modules/absolute", "symlink", "/etc/passwd"),
        ("node_modules/hard", "hardlink", "../outside.json"),
    ],
)
def test_members_that_leave_the_config_directory_are_refused(tmp_path, member):
    archive, digest = plugin_archive(tmp_path, ("package.json", "file", b"{}"), member)
    with pytest.raises(ValueError, match="leaves the target directory"):
        check_archive(archive, digest)


# Each stays inside the target lexically; through the links it is written to, or leads to, the target's parent.
LINK_CHAINS = {
    "member through links": [
        ("d", "dir", b""),
        ("d/l", "symlink", ".."),
        ("d/l/l2", "symlink", ".."),
        ("d/l/l2/escaped.txt", "file", b"outside"),
    ],
    "dot-dot after a link": [("d", "dir", b""), ("d/l", "symlink", ".."), ("d/l/../escaped.txt", "file", b"outside")],
    # Checked before d/x is a link, y lexically leads to d/escaped.txt; writing y follows d/x first.
    "link target through a later link": [
        ("y", "symlink", "d/x/../escaped.txt"),
        ("d", "dir", b""),
        ("d/x", "symlink", ".."),
        ("y", "file", b"outside"),
    ],
    "hard link through a link": [
        ("d", "dir", b""),
        ("d/l", "symlink", ".."),
        ("h", "hardlink", "d/l/../escaped.txt"),
    ],
    # Unpacking cannot turn the folder d/l into a link afterwards; refusing it up front makes it a verdict.
    "member under a later link": [("d/l/f.txt", "file", b"inside"), ("d/l", "symlink", ".")],
}


@pytest.mark.parametrize("chain", LINK_CHAINS.values(), ids=LINK_CHAINS.keys())
def test_no_member_path_or_link_target_may_pass_through_a_link(tmp_path, chain):
    archive, digest = plugin_archive(tmp_path, *chain)
    with pytest.raises(ValueError, match="passes through the link 'd/"):
        check_archive(archive, digest)


@pytest.mark.parametrize(
    "member",
    [
        ("bin/tool", "file", b"#!/bin/sh\n", 0o4755),
        ("bin/tool", "file", b"#!/bin/sh\n", 0o2755),
        ("tmp", "dir", b"", 0o1777),
    ],
    ids=["setuid", "setgid", "sticky"],
)
def test_members_with_setuid_setgid_or_sticky_bits_are_refused(tmp_path, member):
    archive, digest = plugin_archive(tmp_path, member)
    with pytest.raises(ValueError, match="setuid, setgid or sticky bits"):
        check_archive(archive, digest)


def test_other_member_types_and_other_bytes_are_refused(tmp_path):
    archive, digest = plugin_archive(tmp_path, ("package.json", "file", b"{}"), ("pipe", "fifo", b""))
    with pytest.raises(ValueError, match="unsupported archive member 'pipe'"):
        check_archive(archive, digest)
    other = tmp_path / "not-an-archive"
    with pytest.raises(ValueError, match="not a usable tar archive"):
        check_archive(other, pinned(other, b"plain text"))


def test_a_download_unlike_the_pin_faults_the_transfer_not_the_file(tmp_path):
    archive, _ = plugin_archive(tmp_path, ("package.json", "file", b"{}"))
    with pytest.raises(RuntimeError, match="not the pinned " + "0" * 64):
        check_archive(archive, "0" * 64)
