# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Installer requires Linux flock and atomic mv -T")
INSTALLER = Path(__file__).resolve().parents[1] / "install_nooa_runtime.sh"
FINGERPRINT = "a" * 64
# Run the actual shell/checksum/lock/publication code; only package installation is injected.
PRIVATE_PYTHON = r"""
import json, os, pathlib, sys, time
args = [arg for arg in sys.argv[1:] if arg not in {"-I", "-B"}]
if args[:2] == ["-m", "pip"]:
    root = pathlib.Path(__file__).resolve().parents[2]
    assert (root / "source/cache").is_dir(), "Gym egg_base must exist before pip builds source"
    record = {"argv": sys.argv[1:], "entry": str(pathlib.Path(__file__).resolve()),
              "home": os.environ["HOME"], "cwd": os.getcwd(),
              "pythonhome": os.environ.get("PYTHONHOME"),
              "pythonpath": os.environ.get("PYTHONPATH"),
              "virtual_env": os.environ.get("VIRTUAL_ENV")}
    with open(os.environ["NOOA_TEST_EVENTS"], "a") as stream:
        stream.write(json.dumps(record) + "\n")
    gate = os.environ.get("NOOA_TEST_GATE")
    if gate:
        pathlib.Path(gate + ".started").touch()
        deadline = time.monotonic() + 30
        while not pathlib.Path(gate).exists():
            if time.monotonic() > deadline:
                sys.exit(99)
            time.sleep(0.01)
    if os.environ.get("NOOA_TEST_PIP_FAIL"):
        sys.exit(42)
    (root / "pip-installed").write_text("complete")
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
"""


def archive_bytes(files: dict[str, tuple[bytes, int]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, (data, mode) in files.items():
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), mode
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


@dataclass
class InstallerCall:
    script: Path
    runtime: Path
    source_sha: str
    python_sha: str
    env: dict[str, str]

    @property
    def scratch(self) -> Path:
        return self.script.with_suffix("")

    def command(self, mode: str = "install", health: str = "pass") -> list[str]:
        args = ["bash", str(self.script), mode, str(self.runtime), FINGERPRINT]
        if mode == "install":
            args.extend([self.source_sha, self.python_sha])
        return [*args, health]

    def run(self, mode: str = "install", health: str = "pass", **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.run(
            self.command(mode, health), env=self.env, text=True, capture_output=True, timeout=20, **kwargs
        )

    def start(self) -> subprocess.Popen:
        return subprocess.Popen(
            self.command(),
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )


def make_call(directory: Path, *, runtime: Path | None = None, events: Path | None = None) -> InstallerCall:
    directory.mkdir(parents=True)
    script = directory / "setup.sh"
    shutil.copyfile(INSTALLER, script)
    scratch = script.with_suffix("")
    scratch.mkdir()
    source = archive_bytes({"runtime-requirements.txt": (b"# fixture\n", 0o644)})
    python = archive_bytes({"python/bin/python3": (f"#!{sys.executable}\n{PRIVATE_PYTHON}".encode(), 0o755)})
    (scratch / "source.tar.gz").write_bytes(source)
    (scratch / "python.tar.gz").write_bytes(python)
    env = {**os.environ, "NOOA_TEST_EVENTS": str(events or directory / "events.jsonl")}
    return InstallerCall(
        script,
        runtime or directory / "runtimes" / FINGERPRINT,
        hashlib.sha256(source).hexdigest(),
        hashlib.sha256(python).hexdigest(),
        env,
    )


def wait_for(path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Installer never reached {path}")
        time.sleep(0.01)


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("archive", ["source", "python"])
def test_rejects_changed_upload_before_extracting_either_archive(tmp_path: Path, archive: str) -> None:
    call = make_call(tmp_path / "call")
    with (call.scratch / f"{archive}.tar.gz").open("ab") as stream:
        stream.write(b"changed after expected checksum was established")
    result = call.run()
    assert result.returncode != 0
    assert "FAILED" in result.stdout
    assert not call.runtime.exists()
    assert not (call.runtime.parent / ".builds").exists()
    assert not Path(call.env["NOOA_TEST_EVENTS"]).exists()
    assert (call.scratch / "install.exit-code").read_text().strip() != "0"
    assert "sha256sum" in (call.scratch / "install.log").read_text()


@pytest.mark.parametrize("failure,exit_code", [("pip", 42), ("health", 43)])
def test_failed_install_is_unpublished_and_a_fresh_call_recovers(tmp_path: Path, failure: str, exit_code: int) -> None:
    first = make_call(tmp_path / "first")
    if failure == "pip":
        first.env["NOOA_TEST_PIP_FAIL"] = "1"
    result = first.run(health="raise SystemExit(43)" if failure == "health" else "pass")
    assert result.returncode == exit_code
    assert not first.runtime.exists()
    (failed_generation,) = (first.runtime.parent / ".builds").iterdir()
    assert not (failed_generation / "ready").exists()
    assert (first.scratch / "install.exit-code").read_text().strip() == str(exit_code)
    assert "command failed" in (first.scratch / "install.log").read_text()
    retry = make_call(tmp_path / "retry", runtime=first.runtime)
    result = retry.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert first.runtime.is_symlink()
    assert first.runtime.resolve() != failed_generation
    assert (first.runtime / "ready").read_text().strip() == FINGERPRINT
    assert failed_generation.exists()


def test_check_reports_a_runtime_root_the_user_cannot_install_into() -> None:
    # Not tmp_path: when the suite runs as root the check runs as uid 65534, which must reach
    # these files, and other pytest sessions reset tmp_path's parents to 0700.
    base = Path(tempfile.mkdtemp(prefix="nooa-check-", dir="/tmp"))
    try:
        base.chmod(0o755)
        root = base / "opt"
        root.mkdir()
        call = make_call(base / "call", runtime=root / "nemo-gym-nooa" / FINGERPRINT)
        shutil.rmtree(call.scratch)
        call.script.parent.chmod(0o777)
        kwargs = {"user": 65534, "group": 65534, "extra_groups": []} if os.geteuid() == 0 else {}
        root.chmod(0o555)
        result = call.run("check", **kwargs)
        assert result.returncode == 73, result.stdout + result.stderr
        assert f"{root} is not a writable directory" in result.stdout + result.stderr
        root.chmod(0o777)
        result = call.run("check", **kwargs)
        assert result.returncode == 1, result.stdout + result.stderr
        # Neither answer creates anything where the runtime would go.
        assert list(root.iterdir()) == []
        # An install needs a directory there: a file in its place cannot be used either.
        (root / "nemo-gym-nooa").write_text("")
        assert call.run("check", **kwargs).returncode == 73
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_private_root_belongs_to_this_user_alone() -> None:
    # Not tmp_path, for the same reason as the check above.
    base = Path(tempfile.mkdtemp(prefix="nooa-private-", dir="/tmp"))
    try:
        base.chmod(0o1777)  # Shared like /tmp: the sticky bit keeps each user's entries in place.
        script = base / "setup.sh"
        shutil.copyfile(INSTALLER, script)
        script.chmod(0o755)
        as_root = os.geteuid() == 0
        uid = 65534 if as_root else os.geteuid()
        kwargs = {"user": 65534, "group": 65534, "extra_groups": []} if as_root else {}

        def claim(name: str) -> subprocess.CompletedProcess:
            command = ["bash", str(script), "private-root", str(base / name)]
            return subprocess.run(command, text=True, capture_output=True, timeout=20, **kwargs)

        root = base / f"runtime-{uid}"
        for _ in range(2):  # Created on the first call, recognised on the next.
            result = claim("runtime")
            assert (result.returncode, result.stdout) == (0, f"{root}\n"), result.stderr
            assert (root.stat().st_uid, stat.S_IMODE(root.stat().st_mode)) == (uid, 0o700)

        # A runtime that another user could have placed is never handed out.
        writable = base / f"writable-{uid}"
        writable.mkdir()
        os.chown(writable, uid, -1)
        writable.chmod(0o777)
        (base / f"link-{uid}").symlink_to(root)
        (base / f"file-{uid}").write_text("")
        refused = ["writable", "link", "file"]
        if as_root:
            (base / f"foreign-{uid}").mkdir(mode=0o755)
            refused.append("foreign")
        for name in refused:
            result = claim(name)
            assert (result.returncode, result.stdout) == (1, ""), name

        # The parent's own name must be fixed too, even when it leads to a directory that is safe now.
        (base / "linked").symlink_to(base)
        result = claim("linked/runtime")
        assert (result.returncode, result.stdout) == (1, "")
        assert f"{base / 'linked'} is a symbolic link" in result.stderr
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.mark.parametrize("sticky", [True, False])
def test_private_root_needs_a_parent_where_others_cannot_replace_it(sticky: bool) -> None:
    base = Path(tempfile.mkdtemp(prefix="nooa-parent-", dir="/tmp"))
    try:
        base.chmod(0o755)
        as_root = os.geteuid() == 0
        uid = 65534 if as_root else os.geteuid()
        kwargs = {"user": 65534, "group": 65534, "extra_groups": []} if as_root else {}
        call = base / "call"
        call.mkdir()
        call.chmod(0o777)
        script = call / "setup.sh"
        shutil.copyfile(INSTALLER, script)
        script.chmod(0o755)
        shared = base / "shared"
        shared.mkdir()
        # A runtime that passes the health check is already in place, in a directory that is
        # itself private to this user. Its interpreter leaves a mark whenever it is run.
        root = shared / f"runtime-{uid}"
        runtime = root / FINGERPRINT
        python = runtime / "python/bin/python3"
        python.parent.mkdir(parents=True)
        (runtime / "ready").write_text(FINGERPRINT + "\n")
        mark = root / "executed"
        python.write_text(f"#!/bin/sh\ntouch {mark}\n")
        python.chmod(0o755)
        root.chmod(0o700)
        for path in [root, *root.rglob("*")]:
            os.chown(path, uid, -1)
        shared.chmod(0o1777 if sticky else 0o777)

        def run(*arguments: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["bash", str(script), *arguments], text=True, capture_output=True, timeout=20, **kwargs
            )

        claim = run("private-root", str(shared / "runtime"))
        if sticky:
            assert (claim.returncode, claim.stdout) == (0, f"{root}\n"), claim.stderr
            # The caller checks a runtime only below a root that was handed out.
            check = run("check", f"{claim.stdout.strip()}/{FINGERPRINT}", FINGERPRINT, "pass")
            assert check.returncode == 0, check.stdout + check.stderr
            assert mark.exists()
        else:
            # Another user could swap the directory after any check of it, so nothing in it is run.
            assert (claim.returncode, claim.stdout) == (1, "")
            assert f"other users can write {shared} and it has no sticky bit" in claim.stderr
            assert not mark.exists()
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.mark.skipif(os.geteuid() != 0, reason="Acting as two different users needs root")
def test_another_user_cannot_replace_a_private_root_that_was_handed_out() -> None:
    owner, other = ({"user": uid, "group": uid, "extra_groups": []} for uid in (65534, 65533))
    base = Path(tempfile.mkdtemp(prefix="nooa-two-users-", dir="/tmp"))
    try:
        base.chmod(0o755)
        script = base / "setup.sh"
        shutil.copyfile(INSTALLER, script)
        script.chmod(0o755)
        shared = base / "shared"
        shared.mkdir()
        root = shared / "runtime-65534"
        claim = ["bash", str(script), "private-root", str(shared / "runtime")]
        replace = ["sh", "-c", 'mv "$0" "$0.moved" && mkdir "$0"', str(root)]

        shared.chmod(0o1777)
        result = subprocess.run(claim, text=True, capture_output=True, timeout=20, **owner)
        assert (result.returncode, result.stdout) == (0, f"{root}\n"), result.stderr
        assert subprocess.run(replace, capture_output=True, timeout=20, **other).returncode != 0
        assert root.stat().st_uid == 65534

        # Without the sticky bit the same replacement succeeds, which is why the root is refused there.
        shared.chmod(0o777)
        assert subprocess.run(replace, capture_output=True, timeout=20, **other).returncode == 0
        assert root.stat().st_uid == 65533
        result = subprocess.run(claim, text=True, capture_output=True, timeout=20, **owner)
        assert (result.returncode, result.stdout) == (1, "")
        assert "no sticky bit" in result.stderr

        # The sticky bit does not bind the parent's owner, so only root or this user may own it.
        shared.chmod(0o1777)
        os.chown(shared, 65533, -1)
        result = subprocess.run(claim, text=True, capture_output=True, timeout=20, **owner)
        assert (result.returncode, result.stdout) == (1, "")
        assert f"{shared} belongs to uid 65533" in result.stderr
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_concurrent_callers_publish_one_complete_generation(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    first = make_call(tmp_path / "first", events=events)
    second = make_call(tmp_path / "second", runtime=first.runtime, events=events)
    gate = tmp_path / "release"
    first.env["NOOA_TEST_GATE"] = str(gate)
    one, two = first.start(), None
    try:
        wait_for(Path(str(gate) + ".started"))
        two = second.start()
        wait_for(second.scratch / "install.log")
        time.sleep(0.05)
        assert two.poll() is None
        assert len(records(events)) == 1
        assert not first.runtime.exists()
        gate.touch()
        out_one = one.communicate(timeout=20)
        out_two = two.communicate(timeout=20)
        assert one.returncode == two.returncode == 0, (out_one, out_two)
        assert len(records(events)) == 1
        assert len(list((first.runtime.parent / ".builds").iterdir())) == 1
        assert (first.runtime / "ready").read_text().strip() == FINGERPRINT
    finally:
        gate.touch()
        for process in (one, two):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()


def test_killed_installer_does_not_leave_a_pip_owned_lock_or_publish_partial_runtime(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    first = make_call(tmp_path / "first", events=events)
    gate = tmp_path / "release"
    first.env["NOOA_TEST_GATE"] = str(gate)
    process = first.start()
    try:
        wait_for(Path(str(gate) + ".started"))
        (partial,) = (first.runtime.parent / ".builds").iterdir()
        process.kill()
        process.wait(timeout=5)
        retry = make_call(tmp_path / "retry", runtime=first.runtime, events=events)
        result = retry.run()
        assert result.returncode == 0, result.stdout + result.stderr
        assert first.runtime.resolve() != partial
        assert not (partial / "ready").exists()
        assert len(records(events)) == 2
    finally:
        gate.touch()
        process.communicate(timeout=10)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_private_python_ignores_task_python_and_ambient_environment(tmp_path: Path) -> None:
    call = make_call(tmp_path / "call")
    task_bin = tmp_path / "task-bin"
    task_bin.mkdir()
    touched = tmp_path / "task-python-was-called"
    for name in ["python", "python3", "pip"]:
        path = task_bin / name
        path.write_text(f"#!/bin/sh\ntouch '{touched}'\nexit 77\n")
        path.chmod(0o755)
    call.env.update(
        {
            "PATH": f"{task_bin}:{call.env['PATH']}",
            "PYTHONHOME": "/task/python",
            "PYTHONPATH": "/task/modules",
            "VIRTUAL_ENV": "/task/venv",
        }
    )
    original = {p.name: p.read_bytes() for p in task_bin.iterdir()}
    result = call.run(
        health="import os; assert not any(os.getenv(k) for k in ['PYTHONHOME','PYTHONPATH','VIRTUAL_ENV'])"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not touched.exists()
    assert original == {p.name: p.read_bytes() for p in task_bin.iterdir()}
    (record,) = records(Path(call.env["NOOA_TEST_EVENTS"]))
    assert record["entry"] == str(call.runtime.resolve() / "python/bin/python3")
    assert record["home"] == str(call.scratch / "home")
    assert record["cwd"] == str(call.scratch)
    assert all(record[key] is None for key in ["pythonhome", "pythonpath", "virtual_env"])
    assert record["argv"][:7] == ["-I", "-B", "-m", "pip", "--isolated", "install", "--no-cache-dir"]


def test_warm_check_uses_private_scratch_without_writing_read_only_runtime(tmp_path: Path) -> None:
    call = make_call(tmp_path / "cold")
    result = call.run()
    assert result.returncode == 0, result.stdout + result.stderr
    warm = make_call(tmp_path / "warm", runtime=call.runtime)
    shutil.rmtree(warm.scratch)
    for parent in [tmp_path, *tmp_path.parents]:
        if parent == Path("/tmp") or parent == Path("/"):
            break
        parent.chmod(0o755)
    warm.script.parent.chmod(0o777)
    paths = [call.runtime.parent, *call.runtime.parent.rglob("*")]
    for path in paths:
        if not path.is_symlink():
            path.chmod(0o555 if path.is_dir() or path.name == "python3" else 0o444)
    snapshot = {str(p): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None) for p in paths}
    kwargs = {"user": 65534, "group": 65534, "extra_groups": []} if os.geteuid() == 0 else {}
    try:
        result = warm.run(
            "check", health="import pathlib; pathlib.Path.home().joinpath('health-write').touch()", **kwargs
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert (warm.scratch / "home/health-write").exists()
        assert snapshot == {str(p): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None) for p in paths}
        assert not Path(warm.env["NOOA_TEST_EVENTS"]).exists()
    finally:
        for path in paths:
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() or path.name == "python3" else 0o644)


@pytest.mark.parametrize("corrupt", [False, True])
def test_offline_install_verifies_wheels_and_never_resolves_remote_requirements(tmp_path: Path, corrupt: bool) -> None:
    call = make_call(tmp_path / "offline")
    wheels = archive_bytes({"package-1-py3-none-any.whl": (b"fixture", 0o644)})
    (call.scratch / "wheels.tar.gz").write_bytes(wheels + (b"corruption" if corrupt else b""))
    result = subprocess.run(
        [*call.command(), hashlib.sha256(wheels).hexdigest()],
        env=call.env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    if corrupt:
        assert result.returncode != 0
        assert not (call.runtime.parent / ".builds").exists()
        assert not Path(call.env["NOOA_TEST_EVENTS"]).exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        (record,) = records(Path(call.env["NOOA_TEST_EVENTS"]))
        assert "--no-index" in record["argv"]
        assert "--no-deps" in record["argv"]
        assert "--only-binary=:all:" in record["argv"]
        assert not any(arg.startswith(("http:", "https:")) for arg in record["argv"])
        assert "-r" not in record["argv"]
        assert record["argv"][-1].endswith("/wheels/package-1-py3-none-any.whl")
        hook = call.runtime / "python/lib/python3.13/site-packages/nooa_offline_runtime.pth"
        assert hook.read_text() == 'import os; os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"\n'
        assert call.runtime.is_symlink()
