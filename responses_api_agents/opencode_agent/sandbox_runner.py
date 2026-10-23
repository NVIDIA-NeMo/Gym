# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture OpenCode output; the supervisor bounds execution; the sandbox owner stops task services."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import posixpath
import re
import signal
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import ExitStack, closing
from pathlib import Path
from typing import TypedDict


# OpenCode session IDs become file names in the owner's export directory.
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]+")
_PRIVATE_DIRECTORIES = ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME")
# Exit status for an owner-supplied input that cannot work (pinned bytes, an export): retrying cannot help.
REJECTED = 3
# Seconds a pinned executable gets to answer ``--version``.
_VERSION_CHECK_TIMEOUT = 60


class Rejected(Exception):
    """An owner-supplied input failed its checks."""


def _child_environment(params: dict, *, plugins: bool = True) -> dict:
    """The CLI's environment: the task's, then the adapter's (OpenCode's private directories and, when set,
    ``PWD``), with the ``PATH`` prefix.

    Run, export and import put the configured ``path_prefix`` ahead of the sandbox's ``PATH``, so each finds the
    same tools. Export and import run without the run's configuration, so no plugin loads.
    """
    env = {**os.environ, **params["env"]}
    if params.get("path_prefix"):
        env["PATH"] = os.pathsep.join([*params["path_prefix"], os.environ.get("PATH", "")])
    if not plugins:
        env.pop("OPENCODE_CONFIG_CONTENT", None)
    return env


class RunnerInput(TypedDict):
    """Harness input, independent of supervisor cleanup and runtime metadata."""

    directory: str
    prompt: str
    command: list[str]
    cwd: str
    env: dict[str, str]
    path_prefix: list[str]
    harness_session_id: str | None
    previous_message_ids: list[str]


def run(params: RunnerInput) -> int:
    """Run the CLI with isolated files and wait for it before returning captured output."""
    directory = Path(params["directory"])
    (directory / "runtime.json").write_text(
        json.dumps({"hostname": os.uname().nodename, "pid": os.getpid(), "python": sys.executable})
    )
    # Never accept an initialization receipt left by an earlier invocation.
    (directory / "task-environment.ready").unlink(missing_ok=True)
    # Save only task-directory variables for the shell.env hook, before CLI overrides.
    task_home = os.environ.get("HOME") or str(Path.home())
    task_env = {"HOME": task_home}
    for key, suffix in (
        ("XDG_DATA_HOME", ".local/share"),
        ("XDG_CONFIG_HOME", ".config"),
        ("XDG_CACHE_HOME", ".cache"),
        ("XDG_STATE_HOME", ".local/state"),
    ):
        task_env[key] = os.environ.get(key) or str(Path(task_home) / suffix)
    (directory / "task-env.json").write_text(json.dumps(task_env))
    for key in _PRIVATE_DIRECTORIES:
        if key in params["env"]:
            Path(params["env"][key]).mkdir(parents=True, exist_ok=True)
    (directory / "prompt.txt").write_text(params["prompt"])
    stopping = False

    def interrupt(*_: object) -> None:
        nonlocal stopping
        # Defer TERM until Popen returns so a signal cannot lose the child handle.
        stopping = True

    signal.signal(signal.SIGTERM, interrupt)
    with ExitStack() as stack:
        stdout = stack.enter_context((directory / "stdout.jsonl").open("wb"))
        stderr = stack.enter_context((directory / "stderr.log").open("wb"))
        stdin = stack.enter_context((directory / "prompt.txt").open("rb"))
        process = subprocess.Popen(
            params["command"],
            cwd=params["cwd"],
            env=_child_environment(params),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )
        try:
            while True:
                if stopping:
                    process.terminate()
                    stopping = False
                try:
                    return process.wait(timeout=0.05)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # An I/O/capture error must not leave the CLI changing the task.
            # Only stop our direct CLI; task services belong to the sandbox owner.
            if process.poll() is None:
                process.kill()
            process.wait()


def snapshot(directory: Path, *, session_directory: Path | None = None) -> None:
    """Export root messages and retain the full session tree for observations."""
    session_directory = session_directory or directory
    database = session_directory / "data/opencode/opencode.db"
    input_path = directory / "input.json"
    params = json.loads(input_path.read_text()) if input_path.exists() else {}
    previous_ids = set(params.get("previous_message_ids", []))
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as source:
        with sqlite3.connect(directory / "observations.db") as destination:
            source.backup(destination)
        source.row_factory = sqlite3.Row
        sessions = source.execute(
            "select id from session where parent_id is null order by time_created, id"
        ).fetchall()
        if len(sessions) != 1:
            raise RuntimeError(f"Expected one OpenCode root session, found {len(sessions)}")
        rows = source.execute(
            "select id, data from message where session_id=? order by time_created, id", (sessions[0]["id"],)
        ).fetchall()
        session_id = sessions[0]["id"]
        expected_id = params.get("harness_session_id")
        if expected_id is not None and expected_id != session_id:
            raise RuntimeError(f"OpenCode resumed unexpected session {session_id!r}")
        messages = []
        for row in rows:
            if row["id"] in previous_ids:
                continue
            info = json.loads(row["data"])
            parts = [
                json.loads(part[0])
                for part in source.execute(
                    "select data from part where message_id=? order by time_created, id", (row["id"],)
                )
            ]
            messages.append({"id": row["id"], "info": info, "parts": parts})
        all_messages = source.execute("select id, data from message order by time_created, id").fetchall()
        usage_messages = [json.loads(row["data"]) for row in all_messages if row["id"] not in previous_ids]
        (directory / "export.json").write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "message_ids": [row["id"] for row in all_messages],
                    "activation_message_ids": [row["id"] for row in all_messages if row["id"] not in previous_ids],
                    "messages": messages,
                    "usage_messages": usage_messages,
                }
            )
        )


def _replace(path: Path, data: bytes) -> None:
    """Publish a complete file or none: the owner never reads a partial export."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


class ExportStatus(TypedDict):
    """Outcome of ``--export``, printed as one JSON line on stdout.

    Providers differ in what they keep of a command's stderr (OpenSandbox's background exec reports only the
    exit status there), so the outcome never depends on it.
    """

    # Every stored session's ID, mapped to "ok" or why its ``<id>.json`` is missing.
    sessions: dict[str, str]
    # "ok", "absent" when the run left no event log, "not attempted", or why it was not written.
    event_log: str
    # Whether the deadline stopped an export or kept one from starting.
    deadline_exhausted: bool
    # Why the runner stopped before finishing, such as an unreadable session store.
    error: str | None
    budget_seconds: float
    elapsed_seconds: float


def export_failures(status: ExportStatus) -> list[str]:
    """Each problem ``status`` records; none means every export was written."""
    failures = [f"{session_id}: {outcome}" for session_id, outcome in status["sessions"].items() if outcome != "ok"]
    if status["event_log"] not in ("ok", "absent"):
        failures.append(f"event log: {status['event_log']}")
    if status["error"] is not None:
        failures.append(status["error"])
    return failures


def export_sessions(*, directory: Path, session_directory: Path, export_dir: Path, timeout: float) -> ExportStatus:
    """Write every session's ``opencode export`` and this run's event log for the sandbox owner.

    ``sessions.json`` lists the stored session tree first, so the owner can tell a failed export from
    an absent session. The CLI runs with
    the run's private HOME/XDG directories and working directory but without its OpenCode configuration,
    so no plugin loads again. Failures are recorded in the returned status, not raised.
    """
    started = time.monotonic()
    deadline = started + timeout
    status: ExportStatus = {
        "sessions": {},
        "event_log": "not attempted",
        "deadline_exhausted": False,
        "error": None,
        "budget_seconds": round(timeout, 3),
        "elapsed_seconds": 0.0,
    }
    # The runner's process boundary: whatever stops it is reported in the status, not as a traceback.
    try:
        params = json.loads((directory / "input.json").read_text())
        export_dir.mkdir(parents=True, exist_ok=True)
        # Never leave an earlier activation's inventory in place of this one.
        (export_dir / "sessions.json").unlink(missing_ok=True)
        database = session_directory / "data/opencode/opencode.db"
        with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as source:
            rows = source.execute("select id, parent_id from session order by time_created, id").fetchall()
        sessions = [{"id": session_id, "parent_id": parent_id} for session_id, parent_id in rows]
        _replace(export_dir / "sessions.json", json.dumps(sessions).encode())
        env = _child_environment(params, plugins=False)
        outcomes = status["sessions"]
        for session in sessions:
            session_id = session["id"]
            if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
                outcomes[str(session_id)] = "unsafe session ID"
                continue
            target = export_dir / f"{session_id}.json"
            target.unlink(missing_ok=True)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status["deadline_exhausted"] = True
                outcomes[session_id] = "export deadline exceeded"
                continue
            # OpenCode (Bun) writes stdout to a pipe asynchronously and can exit before the pipe drains, which cut
            # exports to their first 8 KB; it writes a regular file synchronously.
            partial = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            try:
                with partial.open("wb") as output:
                    exported = subprocess.run(
                        [params["command"][0], "export", session_id],
                        cwd=params["cwd"],
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=subprocess.PIPE,
                        timeout=remaining,
                    )
                if exported.returncode != 0:
                    stderr = exported.stderr.decode(errors="replace").strip()[-1000:]
                    raise RuntimeError(f"exit {exported.returncode}: {stderr}")
                data = partial.read_bytes()
                try:
                    if not isinstance(json.loads(data), dict):
                        raise ValueError("not an object")
                except ValueError as error:
                    raise ValueError(f"export ({len(data)} bytes) is not a JSON object: {error}") from None
                os.replace(partial, target)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                if isinstance(error, subprocess.TimeoutExpired):
                    status["deadline_exhausted"] = True
                outcomes[session_id] = str(error)
            else:
                outcomes[session_id] = "ok"
            finally:
                partial.unlink(missing_ok=True)
        events = directory / "stdout.jsonl"
        roots = [session["id"] for session in sessions if session["parent_id"] is None]
        if not events.is_file():
            status["event_log"] = "absent"
        elif len(roots) == 1 and isinstance(roots[0], str) and _SESSION_ID.fullmatch(roots[0]):
            activation = params.get("activation_id")
            suffix = "events.jsonl" if activation is None else f"activation-{int(activation)}.events.jsonl"
            _replace(export_dir / f"{roots[0]}.{suffix}", events.read_bytes())
            status["event_log"] = "ok"
        else:
            status["event_log"] = f"no single root session to belong to: {roots!r}"
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
    status["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return status


def _private_copy(source: Path, expected: str, directory: Path) -> Path:
    """Copy ``source`` into ``directory`` and check the copy, so the verified bytes are the ones used."""
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pinned-", dir=str(directory))
    copy = Path(name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as writer, source.open("rb") as reader:
            for chunk in iter(lambda: reader.read(1 << 20), b""):
                digest.update(chunk)
                writer.write(chunk)
        if digest.hexdigest() != expected:
            raise Rejected(f"SHA-256 mismatch: expected {expected}, found {digest.hexdigest()} in {source}")
    except BaseException:
        copy.unlink(missing_ok=True)
        raise
    return copy


def stage_executable(source: Path, expected: str, target: Path) -> None:
    """Install a pinned executable at ``target`` once its copy matches ``expected`` and answers ``--version``.

    Only a verdict on the bytes is ``Rejected``: a digest mismatch, a format the kernel refuses (ENOEXEC,
    EACCES) or a non-zero exit. A signal such as the OOM killer's, a timeout or another OS error is raised as
    is, so the setup stays retryable.
    """
    copy = _private_copy(source, expected, target.parent)
    try:
        copy.chmod(0o755)
        try:
            checked = subprocess.run(
                [str(copy), "--version"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=_VERSION_CHECK_TIMEOUT,
            )
        except OSError as error:
            if error.errno in (errno.ENOEXEC, errno.EACCES):
                raise Rejected(f"{source} does not run: {error}") from error
            raise
        if checked.returncode < 0:
            raise RuntimeError(f"{source} --version was killed by signal {-checked.returncode}")
        if checked.returncode != 0:
            stderr = checked.stderr.decode(errors="replace")[-1000:]
            raise Rejected(f"{source} --version exited {checked.returncode}: {stderr}")
        os.replace(copy, target)
    finally:
        copy.unlink(missing_ok=True)


def _archive_path(folder: str, path: str, links: frozenset[str]) -> str:
    """Where ``path``, read from the archive folder ``folder``, leads in the extraction directory.

    The walk is lexical, so it is exact only while no folder on the way is a link: a path that passes through
    one of ``links`` goes wherever that link points. Raises ValueError for such a path and for one that leaves.
    """
    if posixpath.isabs(path):
        raise ValueError("leaves the target directory")
    parts = [part for part in folder.split("/") if part]
    for component in path.split("/"):
        if component in ("", "."):
            continue
        if "/".join(parts) in links:
            raise ValueError(f"passes through the link {'/'.join(parts)!r}")
        if component != "..":
            parts.append(component)
        elif parts:
            parts.pop()
        else:
            raise ValueError("leaves the target directory")
    return "/".join(parts)


def stage_archive(source: Path, expected: str, target: Path) -> None:
    """Unpack a pinned tar archive into ``target``; members must stay inside it and be files, folders or links.

    Python 3.12+ (and patched older releases) also apply tarfile's ``data`` filter, which resolves links on
    disk. Without it, no member path or link target may pass through a link member, wherever that link points.
    """
    copy = _private_copy(source, expected, target.parent)
    try:
        try:
            with tarfile.open(copy) as archive:
                members = archive.getmembers()
                filtered = hasattr(tarfile, "data_filter")
                links = frozenset(
                    posixpath.normpath(member.name)
                    for member in members
                    if not filtered and (member.issym() or member.islnk())
                )
                for member in members:
                    if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                        raise Rejected(f"unsupported archive member {member.name!r}")
                    try:
                        name = _archive_path("", member.name, links)
                    except ValueError as error:
                        raise Rejected(f"archive member {member.name!r} {error}") from None
                    if member.issym() or member.islnk():
                        # A symbolic link is relative to its own folder; a hard link names another member.
                        folder = posixpath.dirname(name) if member.issym() else ""
                        try:
                            _archive_path(folder, member.linkname, links)
                        except ValueError as error:
                            raise Rejected(f"archive member {member.name!r} -> {member.linkname!r} {error}") from None
                target.mkdir(parents=True, exist_ok=True)
                archive.extractall(target, members=members, **({"filter": "data"} if filtered else {}))
        except tarfile.TarError as error:
            # Includes the safety filter's refusals; I/O errors such as a full disk stay retryable.
            raise Rejected(f"{source} is not a usable tar archive: {error}") from error
    finally:
        copy.unlink(missing_ok=True)


class ImportedSession(TypedDict):
    """The imported root session and every message ID in the private store."""

    session_id: str
    message_ids: list[str]


def import_session(params: dict, export: Path, *, timeout: float) -> ImportedSession | None:
    """Import an owner-staged ``opencode export`` file into the private store; return its session and messages.

    Returns None when nothing is staged at ``export``. The CLI runs from the task directory, with the ``PWD`` the
    adapter sets to it, because OpenCode binds an imported session to the importing process's directory. Only an import that
    exits with an error, or that leaves other than one root session, is ``Rejected``; a signal (such as the OOM
    killer's), a timeout or an OS error is raised as is, so the setup stays retryable.
    """
    if not export.is_file():
        return None
    for key in _PRIVATE_DIRECTORIES:
        if key in params["env"]:
            Path(params["env"][key]).mkdir(parents=True, exist_ok=True)
    imported = subprocess.run(
        [params["command"][0], "import", str(export)],
        cwd=params["cwd"],
        env=_child_environment(params, plugins=False),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
    )
    stderr = imported.stderr.decode(errors="replace")[-2000:]
    if imported.returncode < 0:
        raise RuntimeError(f"opencode import was killed by signal {-imported.returncode}: {stderr}")
    if imported.returncode != 0:
        raise Rejected(f"opencode import exited {imported.returncode}: {stderr}")
    database = Path(params["env"]["XDG_DATA_HOME"]) / "opencode/opencode.db"
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as source:
        roots = [row[0] for row in source.execute("select id from session where parent_id is null")]
        if len(roots) != 1:
            raise Rejected(f"Expected one imported root session, found {len(roots)}")
        message_ids = [row[0] for row in source.execute("select id from message order by time_created, id")]
    return {"session_id": roots[0], "message_ids": message_ids}


def main() -> None:
    if sys.argv[1] == "--snapshot":
        snapshot(Path(sys.argv[2]), session_directory=Path(sys.argv[3]) if len(sys.argv) > 3 else None)
        return
    if sys.argv[1] == "--export":
        status = export_sessions(
            directory=Path(sys.argv[2]),
            session_directory=Path(sys.argv[3]),
            export_dir=Path(sys.argv[4]),
            timeout=float(sys.argv[5]),
        )
        failures = export_failures(status)
        if failures:
            print("OpenCode session export failed: " + "; ".join(failures), file=sys.stderr, flush=True)
        # Last, as one line on stdout, which every provider returns; some drop stderr or merge it into stdout.
        print(json.dumps(status), flush=True)
        if failures:
            raise SystemExit(1)
        return
    if sys.argv[1] in ("--stage-executable", "--stage-archive", "--import"):
        try:
            if sys.argv[1] == "--import":
                params = json.loads(Path(sys.argv[2]).read_text())
                print(json.dumps(import_session(params, Path(sys.argv[3]), timeout=float(sys.argv[4]))))
            else:
                stage = stage_executable if sys.argv[1] == "--stage-executable" else stage_archive
                stage(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
        except Rejected as error:
            print(f"Rejected: {error}", file=sys.stderr)
            raise SystemExit(REJECTED)
        return
    params = json.loads(Path(sys.argv[1]).read_text())
    raise SystemExit(run(params))


if __name__ == "__main__":
    main()
