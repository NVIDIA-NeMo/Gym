# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade one converted SWE-rebench-V2 task in Harbor's separate verifier sandbox (a fresh sandbox of the row's image).

Adapted from ``resources_servers/scale_swe/harbor/task-template/tests/nemo_gym/grade.py``: everything but this
docstring and ``grade()`` is copied unchanged (the reward protection, the bounded capture reader, the audit hook and
the mapping check; a test keeps the two files in step). ``grade()`` repeats what ``SWERebenchResourcesServer._verify``
does for an episode session once the patch is captured (``resources_servers/swe_rebench/app.py``), calling the
server's own modules, vendored unchanged beside this file:

1. read the capture the ``[[verifier.collect]]`` hook left in ``/logs/artifacts/nemo_gym`` (``capture.py``). The
   hook ran in the agent's sandbox, where the agent is root and can make it report anything, so every outcome but a
   readable patch (within its size bound, a regular file, valid JSON of the expected shape) is a completed 0, never a
   mask;
2. build ``VerificationInputs`` as ``app._inputs`` does (test files dropped from the patch, install and test
   commands and the log parser name from ``install_config``);
3. resolve the row's upstream log parser with the server's ``log_parsers.resolve_parser`` from the vendored
   SWE-rebench-V2 tree at the server's pinned commit (``parser_cache/``), so no clone is needed; an unknown parser
   masks, as at the server. A missing repository directory masks too (the image's, before any agent code runs);
4. write ``verification_files`` (eval script, Maven and Gradle mirror files, patches) into this sandbox and run
   ``run_verification``, whose one ``sandbox.exec`` is served by ``local_sandbox.LocalSandbox``, bounded by the
   server's ``evaluation_timeout``;
5. write ``reward.json``: ``reward`` is 1.0 when the instance resolves, the other keys are metrics. Every run without
   a verdict is a completed 0 (``no_verdict_after_tests``): the eval script's install block already runs the agent's
   code (its build files, its package hooks) before ``TEST_OUTPUT_BEGIN``, so nothing in that output is trusted.

Reward safety is the Scale-SWE task's: the eval script runs the agent's patched code as uid 0 without the
capabilities that reach the reward folder, which is handed to ``REWARD_OWNER_UID`` first; the files mapped by this
process and by the processes above it that relay test.sh's exit status are handed to that uid without write bits; once
the eval script starts, an audit hook refuses every import, load, spawn and unlisted open; and the outputs are written
as that uid. Anything that goes wrong after the tests started exits 1 without a reward (a measured 0). See the
Scale-SWE grader's docstring for the full argument.

Standard library only. ``tests/test.sh`` runs it with the image's Python 3.10+ or the pinned standalone Python
(isolated mode).
"""

from __future__ import annotations

import asyncio
import codecs
import concurrent.futures.thread  # noqa: F401  (asyncio.to_thread's module; imported before the tests start)
import ctypes
import errno
import importlib
import importlib.util
import io
import json
import os
import stat
import sys
import time
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARTIFACTS = Path("/logs/artifacts/nemo_gym")
LOGS = Path("/logs/verifier")
# EX_TEMPFAIL: Harbor masks a separately run verifier that exits 75 without a reward.
NO_VERDICT = 75
# Exit code after the tests started when the grader cannot vouch for its own state: no reward, a measured 0.
FAILED_CLOSED = 1
# A uid no process in this sandbox runs as: the owner of every directory on the path to the reward file.
REWARD_OWNER_UID = 47311
PROTECTED_DIRS = (Path("/"), Path("/logs"), LOGS)
# Capabilities the eval script keeps (linux/capability.h). None reaches a file or process another uid owns, or the
# uid itself: CAP_FSETID 4, CAP_KILL 5, CAP_NET_BIND_SERVICE 10, CAP_NET_RAW 13, CAP_AUDIT_WRITE 29.
KEPT_CAPABILITIES = frozenset({4, 5, 10, 13, 29})
_PR_CAPBSET_DROP = 24
_PR_SET_NO_NEW_PRIVS = 38
_PR_CAP_AMBIENT = 47
_PR_CAP_AMBIENT_CLEAR_ALL = 4
_LINUX_CAPABILITY_VERSION_3 = 0x20080522
# Bounds on what the agent's sandbox left behind (capture.py writes a few KiB of JSON).
CAPTURE_MAX_BYTES = 1 << 20
PATCH_MAX_BYTES = 64 << 20
CAPTURE_TEXT_MAX_CHARS = 4000
CAPTURE_WARNINGS_MAX = 100
PROC_MAPS = "/proc/self/maps"
# Imported before the tests start, so that nothing the verdict needs is loaded afterwards.
WARM_IMPORTS = (
    "encodings.utf_8",
    "encodings.ascii",
    "encodings.latin_1",
    "xml.parsers.expat",
    "xml.etree.ElementTree",
    "selectors",
    "signal",
    "subprocess",
)
# Audit events refused once the tests have started (the first subprocess.Popen is the eval script).
REFUSED_AFTER_TESTS = frozenset(
    {
        "import",
        "exec",
        "ctypes.dlopen",
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.spawn",
        "os.system",
        "subprocess.Popen",
    }
)
_AFTER_TESTS = {"started": False, "refused": None, "allowed": frozenset()}


class _FailClosed(BaseException):
    """The grader can no longer vouch for its own state after the tests started: exit without a reward.

    A ``BaseException``, so no ``except Exception`` in the vendored code can swallow it.
    """


class _Unreadable(Exception):
    """An agent-written artifact that is not a regular file within its size bound."""


class _CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapData(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]


def _privilege_dropper():
    """The function the eval script's process runs before ``exec``: root without the capabilities that reach the
    reward. libc and the capability count are looked up here, in this process, not after the fork.

    An error in it makes ``subprocess`` raise in this process after the tests count as started, which fails closed.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    last_cap = int(Path("/proc/sys/kernel/cap_last_cap").read_text())
    header, data = _CapHeader(_LINUX_CAPABILITY_VERSION_3, 0), (_CapData * 2)()

    def drop() -> None:
        for cap in range(last_cap + 1):
            if cap not in KEPT_CAPABILITIES and libc.prctl(_PR_CAPBSET_DROP, cap, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), f"prctl(PR_CAPBSET_DROP, {cap}) failed")
        # A root execve gains inheritable | bounding: clear the inheritable and ambient sets too.
        if libc.capget(ctypes.byref(header), data) != 0:
            raise OSError(ctypes.get_errno(), "capget failed")
        for half in data:
            half.inheritable = 0
        if libc.capset(ctypes.byref(header), data) != 0:
            raise OSError(ctypes.get_errno(), "capset failed")
        if libc.prctl(_PR_CAP_AMBIENT, _PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_CAP_AMBIENT_CLEAR_ALL) failed")
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")

    return drop


def _give_root_ownership(top: str = "/") -> int:
    """Make uid 0 the owner of every entry on the root filesystem that another uid owns; returns how many.

    At the server the eval script's root overrides file permissions, so a repository an image built as another user
    (``testuser`` owns ``test/rsrc`` in the beets image) is as writable as root's own files. Without
    ``CAP_DAC_OVERRIDE`` only ownership gives that, so ownership moves to root; the group and mode stay.

    The walk keeps its own list of folders instead of recursing (``os.walk`` recurses before Python 3.12), so a
    folder tree as deep as the agent's restored artifacts can make it cannot exhaust the interpreter's stack.
    """
    root_device = os.lstat(top).st_dev
    changed = 0
    pending = [top]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                names = [entry.name for entry in entries]
        except OSError:
            continue  # as os.walk: a folder that cannot be listed is skipped
        for name in names:
            path = os.path.join(directory, name)
            try:
                info = os.lstat(path)
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(info.st_mode):
                # Stay on the root filesystem (no /proc, /sys, /dev or other mounts), as `find -xdev` does.
                if info.st_dev != root_device:
                    continue
                pending.append(path)
            if info.st_uid != 0:
                try:
                    os.lchown(path, 0, -1)
                except FileNotFoundError:
                    continue
                changed += 1
    return changed


def _protect_reward_path() -> None:
    """Hand the reward file's directories to ``REWARD_OWNER_UID`` and drop any reward file already there.

    No exit record is written: the harbor server takes test.sh's exit status from its run exec and reads nothing
    written in this sandbox after test.sh but the reward. An older harbor server wrapped test.sh in a shell that
    wrote ``.ng-exit`` after it, and masked a sample whose record was missing or unreadable, so there a test process
    that kills that shell or spoils its record masks the sample."""
    for name in ("reward.json", "reward.txt"):
        (LOGS / name).unlink(missing_ok=True)
    for directory in PROTECTED_DIRS:
        os.chown(directory, REWARD_OWNER_UID, -1)
        os.chmod(directory, 0o755)


def _write_output(name: str, text: str) -> None:
    """Create ``/logs/verifier/<name>`` as ``REWARD_OWNER_UID``, so a process without full capabilities cannot
    change it. Before ``_protect_reward_path`` the folder is Harbor's (mode 777), which that uid can write too.

    The text is written as UTF-8 bytes with ``os.write``: no text layer, so no codec or locale is looked up."""
    os.seteuid(REWARD_OWNER_UID)
    try:
        path = str(LOGS / name)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
        try:
            data = memoryview(text.encode("utf-8", errors="replace"))
            while data:
                data = data[os.write(descriptor, data) :]
        finally:
            os.close(descriptor)
    finally:
        os.seteuid(0)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _no_verdict(reason: str, details: dict) -> int:
    """Leave no reward and exit 75, so Harbor masks the sample as the server would."""
    _write_output("grading.json", json.dumps({**details, "masked": True, "reason": reason}, indent=2, default=str))
    print(f"[nemo_gym] no verdict, masking the sample: {reason}", flush=True)
    return NO_VERDICT


def _verdict(reward: float, metrics: dict, details: dict) -> int:
    _write_output("grading.json", json.dumps({**details, "masked": False}, indent=2, default=str))
    _write_output("reward.json", json.dumps({"reward": reward, **metrics}))
    print(json.dumps({"reward": reward, "error": details.get("error")}), flush=True)
    return 0


def _capture_failed(status: object, error: object, details: dict) -> int:
    """No patch came out of the agent's sandbox: a completed 0 with no test run.

    ``git_error`` and ``runner_failed`` are the server's ``_UnreadablePatchError`` (a measured 0 there too). Anything
    else (``infra_error``, no or unreadable ``capture.json`` or patch) the server would mask, but here it comes from
    the agent's sandbox, whose hook files and outputs the agent owns, so it is scored like an unreadable patch.
    """
    unreadable = status in ("git_error", "runner_failed")
    details["error"] = f"Failed to extract model patch ({status}): {error}"
    metrics = {
        "evaluation_completed": 1.0,
        "resolved": 0.0,
        "patch_unreadable": float(unreadable),
        "capture_failed": float(not unreadable),
    }
    return _verdict(0.0, metrics, details)


def _fail_closed(error: BaseException) -> int:
    """After the tests started: one line on stderr and no reward, which Harbor measures as 0 (exit 1, not 75)."""
    line = f"[nemo_gym] failing closed after the tests started, no reward: {type(error).__name__}: {error}\n"
    os.write(2, line.encode("utf-8", errors="replace"))
    return FAILED_CLOSED


def _read_agent_file(path: Path, limit: int) -> bytes:
    """The bytes of an agent-written regular file of at most ``limit`` bytes; raises ``FileNotFoundError`` when it
    is absent, ``_Unreadable`` or ``OSError`` otherwise. A symlink, FIFO, device, socket or directory is refused
    without reading it, and at most ``limit + 1`` bytes are read (procfs reports size 0)."""
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise _Unreadable(f"{path.name} is not a regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise _Unreadable(f"{path.name} is not a regular file")
        if info.st_size > limit:
            raise _Unreadable(f"{path.name} is larger than {limit} bytes")
        chunks, total = [], 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, limit + 1 - total))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise _Unreadable(f"{path.name} is larger than {limit} bytes")
    finally:
        os.close(descriptor)


def _capture_summary(capture: dict) -> dict:
    """``capture.json``'s top-level scalars for ``grading.json``; anything nested is replaced by its type name."""

    def scalar(value: object) -> object:
        if isinstance(value, str):
            return value[:CAPTURE_TEXT_MAX_CHARS]
        if value is None or isinstance(value, (bool, float)):
            return value
        if isinstance(value, int) and -(1 << 63) <= value < (1 << 63):
            return value
        return type(value).__name__

    summary = {}
    for key, value in capture.items():
        if key == "warnings" and isinstance(value, list) and len(value) <= CAPTURE_WARNINGS_MAX:
            if all(isinstance(item, str) for item in value):
                summary[key] = [item[:CAPTURE_TEXT_MAX_CHARS] for item in value]
                continue
        summary[key[:CAPTURE_TEXT_MAX_CHARS]] = scalar(value)
    return summary


def _read_capture() -> tuple[dict | None, str | None, tuple[object, str] | None]:
    """``(capture summary, patch text, None)``, or a ``(status, error)`` failure as the last item.

    Every way the agent's sandbox can leave these files (absent, a symlink, a FIFO, oversized, not UTF-8, not JSON,
    nested too deep, the wrong shape) is a failure, which the caller scores as a completed 0, never a mask.
    """
    try:
        data = _read_agent_file(ARTIFACTS / "capture.json", CAPTURE_MAX_BYTES)
    except FileNotFoundError:
        return None, None, ("missing", "the patch capture hook left no capture.json")
    except (OSError, _Unreadable) as error:
        return None, None, ("unreadable", f"capture.json could not be read: {error}")
    try:
        capture = json.loads(data.decode("utf-8", errors="replace"))
    except (ValueError, RecursionError) as error:
        return None, None, ("unreadable", f"capture.json is not JSON: {type(error).__name__}: {str(error)[:200]}")
    if not isinstance(capture, dict) or not isinstance(capture.get("status"), str):
        return None, None, ("unreadable", "capture.json is not a JSON object with a string status")
    summary = _capture_summary(capture)
    if capture["status"] != "ok":
        return summary, None, (summary["status"], str(summary.get("error")))
    try:
        patch = _read_agent_file(ARTIFACTS / "model_patch.diff", PATCH_MAX_BYTES)
    except FileNotFoundError:
        return summary, None, ("missing", "capture.json says ok but model_patch.diff is missing")
    except (OSError, _Unreadable) as error:
        return summary, None, ("unreadable", f"model_patch.diff could not be read: {error}")
    return summary, patch.decode("utf-8", errors="replace"), None


async def _inline_to_thread(func, /, *args, **kwargs):
    """``asyncio.to_thread`` for this process: the same result, called inline, so no thread is ever started (a
    thread's start or exit can load code after the tests)."""
    return func(*args, **kwargs)


async def _warm_event_loop() -> None:
    await asyncio.to_thread(int)


def _read_mappings(maps: str = PROC_MAPS) -> set[tuple[str, str, str]]:
    """``(path, device, inode)`` of every file-backed mapping listed in ``maps`` (this process's by default)."""
    descriptor = os.open(maps, os.O_RDONLY | os.O_CLOEXEC)
    try:
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    entries = set()
    for line in b"".join(chunks).decode("utf-8", errors="replace").splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith("/") and fields[4] != "0":
            entries.add((fields[5], fields[3], fields[4]))
    return entries


def _mapped_paths(maps: str = PROC_MAPS) -> list[str]:
    """The files a process has mapped that still have a path (``maps`` is this process's by default)."""
    return sorted({path for path, _, _ in _read_mappings(maps) if not path.endswith(" (deleted)")})


def _ancestors() -> list[tuple[int, str | None]]:
    """``(pid, name)`` of every process above this one, nearest first.

    They all wait for this process and outlive the tests: ``timeout`` and the shell the harbor server runs test.sh in
    relay its exit status, and whatever started them (``su``, the provider's daemon) relays it further. No shape is
    assumed (an older harbor server's shell has more to do after test.sh, a provider may add or remove a level): the
    walk follows the parent links until a parent outside this PID namespace (pid 0), one that is gone, or one whose
    ``stat`` this process may not read (listed with the name ``None``; its parent is unknown, so the walk ends).
    """
    found: list[tuple[int, str | None]] = []
    pid = os.getppid()
    while pid > 0 and all(pid != seen for seen, _ in found):
        try:
            with open(f"/proc/{pid}/stat", "rb") as handle:
                line = handle.read()
        except (FileNotFoundError, ProcessLookupError):
            break
        except PermissionError:
            found.append((pid, None))
            break
        # The name is in parentheses and may hold any byte: the fields after the last ")" are state, then ppid.
        head, _, tail = line.rpartition(b")")
        found.append((pid, head.partition(b"(")[2].decode("utf-8", errors="replace")))
        pid = int(tail.split()[1])
    return found


def _mapping_ids() -> set[tuple[str, str]]:
    return {(device, inode) for _, device, inode in _read_mappings()}


def _protected_mode(mode: int) -> int:
    """No write bits anywhere; group and other get at least the owner's read and execute."""
    owner_rx = (mode >> 6) & 0o5
    return (mode & 0o555) | (owner_rx << 3) | owner_rx


def _file_state(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_uid, info.st_mode, info.st_size, info.st_mtime_ns


def _protect_mappings() -> dict:
    """Hand every regular file mapped by this process or one of its ``_ancestors`` to ``REWARD_OWNER_UID`` without
    write bits, keeping an open descriptor and the state of each, so the tests cannot rewrite in place the code that
    computes the reward or relays the exit status. A read-only filesystem counts as protected; an ancestor that is
    gone, and a mapped path that does not exist in this mount namespace, leave nothing the tests could write. An
    ancestor whose maps this process may not read is skipped and listed in ``unreadable_ancestors``. Any other error
    raises (before the tests: the caller masks)."""
    started = time.monotonic()
    ancestors = _ancestors()
    paths = set(_mapped_paths())
    unreadable = []
    for pid, name in ancestors:
        try:
            paths.update(_mapped_paths(f"/proc/{pid}/maps"))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # A process this one may not inspect started with more privilege than the exec that runs the grader: in
            # practice the provider's own service or the sandbox's init. Whether test code can change it is the
            # provider's boundary, not something the grader can protect. The relays started by that same exec
            # (timeout, the server's shell, su) stay readable and are protected.
            unreadable.append({"pid": pid, "name": name})
    kept = {}
    for path in sorted(paths):
        try:
            info = os.stat(path)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        try:
            os.chmod(path, _protected_mode(info.st_mode))
            os.chown(path, REWARD_OWNER_UID, -1)
        except OSError as error:
            if error.errno != errno.EROFS:
                raise
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        kept[path] = (descriptor, _file_state(os.fstat(descriptor)))
    return {
        "mappings": _mapping_ids(),
        "kept": kept,
        "ancestors": [name for _, name in ancestors],
        "unreadable_ancestors": unreadable,
        "seconds": round(time.monotonic() - started, 3),
    }


def _prepare_after_tests() -> dict:
    """Load everything the verdict needs, make ``asyncio.to_thread`` inline, then protect the mapped files."""
    for name in WARM_IMPORTS:
        try:
            importlib.import_module(name)
        except ImportError:
            pass
    # The encoding text files open with by default (the vendored code writes test_output.log with it).
    codecs.lookup(io.TextIOWrapper(io.BytesIO()).encoding)
    asyncio.to_thread = _inline_to_thread
    asyncio.run(_warm_event_loop())
    return _protect_mappings()


def _after_tests_hook(event: str, args: tuple) -> None:
    state = _AFTER_TESTS
    if not state["started"]:
        if event == "subprocess.Popen":
            state["started"] = True
        return
    refusal = None
    if event in REFUSED_AFTER_TESTS:
        refusal = f"{event} {str(args[:1])[:200]} after the tests started"
    elif event == "open" and not isinstance(args[0], int) and os.fsdecode(args[0]) not in state["allowed"]:
        refusal = f"open of {os.fsdecode(args[0])[:200]!r} after the tests started"
    if refusal is not None:
        if state["refused"] is None:
            state["refused"] = refusal
        raise _FailClosed(refusal)


def _arm_after_tests() -> None:
    """Install the audit hook (it cannot be removed): the first ``subprocess.Popen`` starts the tests; after it only
    the outputs and ``/proc/self/maps`` may be opened, and nothing may be imported, loaded, executed or spawned."""
    names = ("reward.json", "grading.json", "test_output.log")
    _AFTER_TESTS["allowed"] = frozenset({str(LOGS / name) for name in names} | {PROC_MAPS})
    sys.addaudithook(_after_tests_hook)


def _check_mappings(record: dict) -> None:
    """After the tests: no new file mapping, and every protected file has the owner, mode, size and mtime it had."""
    if _AFTER_TESTS["refused"] is not None:
        raise _FailClosed(_AFTER_TESTS["refused"])
    new = _mapping_ids() - record["mappings"]
    if new:
        raise _FailClosed(f"{len(new)} new file mapping(s) after the tests started")
    for path, (descriptor, before) in record["kept"].items():
        if _file_state(os.fstat(descriptor)) != before:
            raise _FailClosed(f"a file the grader has mapped changed during the tests: {path}")


def grade() -> int:
    sys.path.insert(0, str(HERE))
    from local_sandbox import LocalSandbox

    task = json.loads((HERE / "task.json").read_text())
    row = task["row"]
    details: dict = {"instance_id": row["instance_id"], "language": row.get("language", "")}

    capture, patch, failure = _read_capture()
    if capture is not None:
        details["capture"] = capture
    if failure is not None:
        return _capture_failed(*failure, details)

    v = _load("nemo_gym_swe_rebench_verification", HERE / "resources_servers/swe_rebench/verification.py")
    parsers = _load("nemo_gym_swe_rebench_log_parsers", HERE / "resources_servers/swe_rebench/log_parsers.py")
    try:
        model_patch = v.drop_test_patch_files(patch, row.get("test_patch") or "")
    except MemoryError:
        raise
    except Exception as error:
        # The agent's text through the server's regular expressions: its problem, a completed 0.
        return _capture_failed("unreadable", f"the patch could not be split into files: {error!r}"[:500], details)
    # app.SWERebenchResourcesServer._inputs
    install_config = row.get("install_config") or {}
    inputs = v.VerificationInputs(
        instance_id=row["instance_id"],
        repo=row["repo"],
        base_commit=row["base_commit"],
        patch=model_patch,
        test_patch=row.get("test_patch") or "",
        install=v.as_command_list(install_config.get("install")),
        test_cmd=v.as_command_list(install_config.get("test_cmd")),
        log_parser=str(install_config.get("log_parser") or ""),
        fail_to_pass=list(row.get("FAIL_TO_PASS") or []),
        pass_to_pass=list(row.get("PASS_TO_PASS") or []),
    )
    # app._verify resolves the parser before spending a sandbox; an unknown name masks the sample there too.
    try:
        parser = parsers.resolve_parser(inputs.log_parser, HERE / "parser_cache")
    except Exception as error:
        return _no_verdict(f"log parser {inputs.log_parser!r} could not be resolved: {error}", details)
    repo_directory = v.repo_directory(inputs.repo)
    if not os.path.isdir(repo_directory):
        # The image's own layout, before any agent code runs (the eval script would exit 97 here).
        return _no_verdict(f"repo directory {repo_directory} not present in the image", details)
    files = v.verification_files(inputs, overwrite_maven_settings=task["overwrite_maven_settings"])
    for sandbox_path, content in files.items():
        target = Path(sandbox_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    try:
        details["ownership_moved_to_root"] = _give_root_ownership()
        _protect_reward_path()
        drop_privileges = _privilege_dropper()
    except OSError as error:
        # Nothing of the agent's has run yet: this sandbox cannot keep the reward from the tests.
        return _no_verdict(f"could not protect the reward folder: {error}", details)
    try:
        record = _prepare_after_tests()
    except OSError as error:
        return _no_verdict(f"could not protect the files the grader has mapped: {error}", details)
    details["protected_mappings"] = {
        "files": len(record["kept"]),
        "ancestors": record["ancestors"],
        "unreadable_ancestors": record["unreadable_ancestors"],
        "seconds": record["seconds"],
    }
    _arm_after_tests()
    result = asyncio.run(
        v.run_verification(
            sandbox=LocalSandbox(preexec_fn=drop_privileges),
            inputs=inputs,
            parser=parser,
            timeout_s=task["evaluation_timeout"],
            log_dir=LOGS,
        )
    )
    _check_mappings(record)
    details.update(
        {
            "error": result.error,
            "test_results": result.test_results,
            "test_patch_failed": result.test_patch_failed,
        }
    )
    # The server masks a run without a verdict (exit 97, the test-patch marker, an interrupted run, a parser that
    # raised). Here the agent's code ran before any of them could be observed, so each is a completed 0.
    no_verdict = not result.completed
    metrics = {
        "evaluation_completed": 1.0,
        "resolved": float(bool(result.resolved) and not no_verdict),
        "patch_applied": float(bool(result.patch_applied)),
        "test_patch_failed": float(bool(result.test_patch_failed)),
        "model_patch_bytes": float(len(patch.encode("utf-8", errors="replace"))),
        "tests_observed": float((result.test_results or {}).get("tests_observed", 0)),
        "no_verdict_after_tests": float(no_verdict),
    }
    return _verdict(metrics["resolved"], metrics, details)


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        return grade()
    except _FailClosed as error:
        return _fail_closed(error)
    except Exception as error:
        if _AFTER_TESTS["started"]:
            # The tests could have caused it (a flood of output, a refused import): never a mask.
            return _fail_closed(error)
        # The server masks a verification that raised (VERIFIER_ERROR).
        traceback.print_exc()
        return _no_verdict("the grader raised", {"traceback": traceback.format_exc()[-4000:]})


if __name__ == "__main__":
    sys.exit(main())
