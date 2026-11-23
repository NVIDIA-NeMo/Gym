# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade one converted Scale-SWE task in Harbor's separate verifier sandbox (a fresh sandbox of the row's image).

Adapted from the SWE rows-to-Harbor converter PR (#5789), ``scripts/swe_rows_to_harbor/grade.py``, with its known
gaps fixed: a verification without a verdict exits 75 (masked) instead of scoring 0, as the Scale-SWE server masks it
on its episode-session path, a lost patch is not graded, and the agent's code cannot write the reward.

It repeats what ``ScaleSWEResourcesServer._verify`` does for an episode session once the patch is captured
(``resources_servers/scale_swe/app.py``), calling the server's own ``verification.py``, vendored unchanged beside this
file:

1. read the capture the ``[[verifier.collect]]`` hook left in ``/logs/artifacts/nemo_gym`` (``capture.py``). The
   hook ran in the agent's sandbox, where the agent is root and can make it report anything, so every outcome but a
   readable patch is a completed 0, never a mask (see "Reward safety");
2. build ``VerificationInputs`` as ``app._inputs`` does (test files dropped from the patch);
3. write ``verification_files`` into this sandbox and run ``run_verification``, whose one ``sandbox.exec`` is served
   by ``local_sandbox.LocalSandbox``, bounded by the server's ``evaluation_timeout``;
4. a run without a verdict (``not completed``, or ``infrastructure_failure``) exits 75; otherwise write
   ``reward.json``: ``reward`` is 1.0 when the instance resolves, the other keys are metrics.

Reward safety. The eval script runs the agent's patched code (its tests, a ``conftest.py`` it added) as root in this
sandbox, and Harbor reads ``/logs/verifier/reward.json`` whatever test.sh's exit code. At the server the verdict is
computed outside the sandbox; here it is protected inside it:

- before the eval script runs, every file another uid owns becomes root's (``_give_root_ownership``: root keeps
  write access to the repository without overriding permissions), then ``/``, ``/logs`` and ``/logs/verifier`` are
  handed to a uid nothing runs as (``REWARD_OWNER_UID``), so a root process without ``CAP_DAC_OVERRIDE`` or
  ``CAP_FOWNER`` cannot create, rename or replace anything on the path to the reward file;
- the eval script runs as root (uid 0, as at the server) with its capability bounding set cut down to
  ``KEPT_CAPABILITIES`` and ``no_new_privs`` set: it keeps root's ownership of root's files, but cannot override file
  permissions, change owners, switch uid, trace this process or create device nodes;
- this process writes its outputs with its effective uid set to ``REWARD_OWNER_UID``, so a test process left running
  cannot reopen ``reward.json`` and change it.

The agent's code can still kill this process (same uid): then no reward is written, which Harbor scores 0 (or masks,
when the wrapper around test.sh was killed too), never a planted 1. The eval script keeps root's ownership of root's
files, so in principle it could rewrite in place a file this process has already loaded (a shared library) and so
reach this process: the protection is best effort against code running beside the grader, and closing that gap needs
the verdict computed outside the sandbox (a harbor server change).

Standard library only. ``tests/test.sh`` runs it with the image's Python (3.9 or newer, isolated mode).
"""

from __future__ import annotations

import asyncio
import concurrent.futures.thread  # noqa: F401  (asyncio.to_thread imports it lazily, after the agent's code ran)
import ctypes
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARTIFACTS = Path("/logs/artifacts/nemo_gym")
LOGS = Path("/logs/verifier")
# EX_TEMPFAIL: Harbor masks a separately run verifier that exits 75 without a reward.
NO_VERDICT = 75
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


class _CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapData(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]


def _privilege_dropper():
    """The function the eval script's process runs before ``exec``: root without the capabilities that reach the
    reward. libc and the capability count are looked up here, in this process, not after the fork.

    An error in it makes ``subprocess`` raise in this process, which masks the sample before any agent code ran.
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


def _give_root_ownership() -> int:
    """Make uid 0 the owner of every entry on the root filesystem that another uid owns; returns how many.

    At the server the eval script's root overrides file permissions, so a repository an image built as another user
    (``testuser`` owns ``test/rsrc`` in the beets image) is as writable as root's own files. Without
    ``CAP_DAC_OVERRIDE`` only ownership gives that, so ownership moves to root; the group and mode stay.
    """
    root_device = os.lstat("/").st_dev

    def on_root_device(path: str) -> bool:
        try:
            return os.lstat(path).st_dev == root_device
        except FileNotFoundError:
            return False

    changed = 0
    for directory, subdirs, files in os.walk("/"):
        # Stay on the root filesystem (no /proc, /sys, /dev or other mounts), as `find -xdev` does.
        subdirs[:] = [name for name in subdirs if on_root_device(os.path.join(directory, name))]
        for name in subdirs + files:
            path = os.path.join(directory, name)
            try:
                if os.lstat(path).st_uid != 0:
                    os.lchown(path, 0, -1)
                    changed += 1
            except FileNotFoundError:
                continue
    return changed


def _protect_reward_path() -> None:
    """Hand the reward file's directories to ``REWARD_OWNER_UID`` and drop any reward file already there."""
    for name in ("reward.json", "reward.txt"):
        (LOGS / name).unlink(missing_ok=True)
    for directory in PROTECTED_DIRS:
        os.chown(directory, REWARD_OWNER_UID, -1)
        os.chmod(directory, 0o755)


def _write_output(name: str, text: str) -> None:
    """Create ``/logs/verifier/<name>`` as ``REWARD_OWNER_UID``, so a process without full capabilities cannot
    change it. Before ``_protect_reward_path`` the folder is Harbor's (mode 777), which that uid can write too."""
    os.seteuid(REWARD_OWNER_UID)
    try:
        path = LOGS / name
        path.unlink(missing_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
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
    else (``infra_error``, no or unreadable ``capture.json``) the server would mask, but here it comes from the agent's
    sandbox, whose hook files and outputs the agent owns, so it is scored like an unreadable patch.
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


def grade() -> int:
    sys.path.insert(0, str(HERE))
    from local_sandbox import LocalSandbox

    task = json.loads((HERE / "task.json").read_text())
    row = task["row"]
    details: dict = {"instance_id": row["instance_id"]}

    capture_path = ARTIFACTS / "capture.json"
    if not capture_path.is_file():
        return _capture_failed("missing", "the patch capture hook left no capture.json", details)
    try:
        capture = json.loads(capture_path.read_text(errors="replace"))
    except ValueError as error:
        return _capture_failed("unreadable", f"capture.json is not JSON: {error}", details)
    if not isinstance(capture, dict):
        return _capture_failed("unreadable", "capture.json is not a JSON object", details)
    details["capture"] = capture
    status = capture.get("status")
    patch_path = ARTIFACTS / "model_patch.diff"
    if status != "ok":
        return _capture_failed(status, capture.get("error"), details)
    if not patch_path.is_file():
        return _capture_failed("missing", "capture.json says ok but model_patch.diff is missing", details)
    patch = patch_path.read_text(errors="replace")

    v = _load("nemo_gym_scale_swe_verification", HERE / "resources_servers/scale_swe/verification.py")
    # app.ScaleSWEResourcesServer._inputs
    inputs = v.VerificationInputs(
        instance_id=row["instance_id"],
        workdir=row["workdir"],
        patch=v.drop_test_patch_files(patch, row.get("f2p_patch") or ""),
        pre_commands=row.get("pre_commands") or "",
        f2p_patch=row.get("f2p_patch") or "",
        f2p_script=row.get("f2p_script") or "",
        fail_to_pass=v.as_id_list(row.get("FAIL_TO_PASS")),
        pass_to_pass=v.as_id_list(row.get("PASS_TO_PASS")),
    )
    for sandbox_path, content in v.verification_files(inputs).items():
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
    modules_before = set(sys.modules)
    result = asyncio.run(
        v.run_verification(
            sandbox=LocalSandbox(preexec_fn=drop_privileges),
            inputs=inputs,
            timeout_s=task["evaluation_timeout"],
            log_dir=LOGS,
        )
    )
    reason = v.infrastructure_failure(result)
    details.update(
        {
            "error": reason or result.error,
            "test_results": result.test_results,
            "exec_error_type": result.exec_error_type,
            "test_patch_failed": result.test_patch_failed,
            # Modules loaded after the agent's code ran, which can rewrite files root owns: expected empty.
            "modules_imported_after_tests": sorted(set(sys.modules) - modules_before),
        }
    )
    if reason is not None or not result.completed:
        # app._verify: no verdict on an episode session is masked (the server first retries once on a fresh sandbox).
        return _no_verdict(reason or result.error or "No verdict", details)
    metrics = {
        "evaluation_completed": 1.0,
        "resolved": float(bool(result.resolved)),
        "patch_applied": float(bool(result.patch_applied)),
        "test_patch_failed": float(bool(result.test_patch_failed)),
        "model_patch_bytes": float(len(patch.encode("utf-8", errors="replace"))),
        "tests_observed": float((result.test_results or {}).get("tests_observed", 0)),
        "verifier_timed_out": float(result.exec_error_type == "timeout"),
    }
    return _verdict(1.0 if result.resolved else 0.0, metrics, details)


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        return grade()
    except Exception:
        # The server masks a verification that raised (VERIFIER_ERROR).
        traceback.print_exc()
        return _no_verdict("the grader raised", {"traceback": traceback.format_exc()[-4000:]})


if __name__ == "__main__":
    sys.exit(main())
