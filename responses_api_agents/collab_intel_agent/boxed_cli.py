# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The upstream harness CLI, with the python subprocesses that run the policy's code sent to the agent's sandbox.

Run by the agent in place of ``python -m collab_intel.harness.cli``, in the upstream repo's venv, so this file uses the
standard library only. subprocess.run of a command that starts with this interpreter (pytest in the policy's workspace,
the probes of a few checkers) is sent to CI_EXEC_URL instead (boxed_exec.py): the workspace goes there, and the output
and the files the command changed come back. Any other way of starting a python interpreter (Popen, check_output, a
bare "python", os.system, ...) is refused, so a launch form a new harness commit adds fails the episode instead of
running the code here. A failure of the sandbox ends the process with exit code 86 and CI_EXEC_ERROR on stderr, so it
can never be scored as a policy failure.
"""

import base64
import functools
import importlib.metadata
import io
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import threading
import urllib.error
import urllib.request
from pathlib import Path


EXIT_CODE = 86
DEFAULT_TIMEOUT_S = 600
# the box has no network and a bare python: pytest and its dependencies come from this venv (pure python)
RUNTIME_DISTRIBUTION = "pytest"
SKIP = {"__pycache__", ".pytest_cache"}
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

_run = subprocess.run
_Popen = subprocess.Popen
_lock = threading.Lock()
_runtime_sent = False
_PYTHON_NAME = re.compile(r"python[0-9.]*$")
# the ways of starting a process that bypass subprocess.Popen (os.popen goes through it); the pinned harness uses none
# of them. Not os.posix_spawn: subprocess.Popen itself may use it.
_OS_LAUNCHERS = (
    *("system", "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe"),
    *("execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe"),
)


def _die(message: str) -> None:
    print(f"CI_EXEC_ERROR: {message}", file=sys.stderr, flush=True)
    os._exit(EXIT_CODE)


def _tar(root: Path) -> bytes:
    buf, size = io.BytesIO(), 0
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if SKIP & set(rel.parts) or p.is_symlink() or not p.is_file():
                continue
            size += p.stat().st_size
            if size > MAX_UPLOAD_BYTES:
                _die(f"the workspace is larger than {MAX_UPLOAD_BYTES} bytes")
            tar.add(p, arcname=rel.as_posix())
    return buf.getvalue()


def _requirements(name: str, found: dict[str, importlib.metadata.Distribution]) -> None:
    """The installed distribution `name` and, recursively, what it requires (optional extras left out)."""
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return  # a requirement this interpreter does not need, e.g. one gated on an older python
    if dist.metadata["Name"] in found:
        return
    found[dist.metadata["Name"]] = dist
    for requirement in dist.requires or []:
        if "extra ==" not in requirement:
            _requirements(re.split(r"[ ;<>=!~(\[]", requirement, maxsplit=1)[0], found)


def _runtime_tar() -> bytes:
    """pytest and its dependencies as the files this venv installed for them (no caches, scripts or metadata)."""
    found: dict[str, importlib.metadata.Distribution] = {}
    _requirements(RUNTIME_DISTRIBUTION, found)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dist in found.values():
            for f in dist.files or []:
                path = Path(dist.locate_file(f))
                parts = Path(str(f)).parts
                if parts[0] == ".." or SKIP & set(parts) or parts[0].endswith(".dist-info") or not path.is_file():
                    continue
                tar.add(path, arcname=str(f))
    return buf.getvalue()


def _post(payload: dict, timeout_s: float) -> dict:
    request = urllib.request.Request(
        os.environ["CI_EXEC_URL"], data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        _die(f"the sandbox refused a command (HTTP {exc.code}): {exc.read()[-500:].decode(errors='replace')}")
    except Exception as exc:
        _die(f"the sandbox did not answer ({type(exc).__name__})")
    return {}


def _send(payload: dict, timeout_s: float) -> dict:
    """The first command of the process carries the runtime; the others wait until it has been unpacked."""
    global _runtime_sent
    if not _runtime_sent:
        with _lock:
            if not _runtime_sent:
                payload["runtime"] = base64.b64encode(_runtime_tar()).decode()
                result = _post(payload, timeout_s)
                _runtime_sent = True
                return result
    return _post(payload, timeout_s)


def _apply(cwd: Path, result: dict) -> None:
    """What the command created, changed or deleted in the sandbox, in the workspace here. Deletions go first, and a
    directory where a file now is goes too, so a file replaced by a directory (or the reverse) comes back as well."""
    root = cwd.resolve()
    for name in result.get("deleted", []):
        target = (root / name).resolve()
        if root in target.parents:
            target.unlink(missing_ok=True)
    for name, data in result.get("changed", {}).items():
        target = (root / name).resolve()
        if root not in target.parents:
            continue
        if target.is_dir():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(data))


def _is_python(program) -> bool:
    """This interpreter, or any other python found by name ("python", "python3", "/usr/bin/python3.12", ...)."""
    program = str(program)
    return program == sys.executable or bool(_PYTHON_NAME.match(os.path.basename(program)))


def run(args, *popenargs, **kwargs):
    """subprocess.run, with a command that starts with this interpreter sent to the sandbox."""
    if not (isinstance(args, (list, tuple)) and args and _is_python(args[0])):
        if isinstance(args, str) and re.search(r"\bpython", args):
            _die("a python command given as a string was refused: it would run on the agent's host")
        return _run(args, *popenargs, **kwargs)
    cwd = kwargs.get("cwd")
    if cwd is None:
        _die("a python command without a cwd cannot be sent to the sandbox")
    host_dirs = {os.path.abspath(str(cwd)), os.path.realpath(str(cwd))}

    def mapped(value: str) -> str:
        for d in sorted(host_dirs, key=len, reverse=True):
            value = value.replace(d, "@dir")
        return value

    timeout = kwargs.get("timeout") or DEFAULT_TIMEOUT_S
    env = {k: mapped(str(v)) for k, v in (kwargs.get("env") or {}).items() if k != "PATH"}
    env.setdefault("HOME", "@dir")
    payload = {
        "argv": ["@python", *(mapped(str(a)) for a in args[1:])],
        "env": env,
        "timeout": timeout,
        "files": base64.b64encode(_tar(Path(cwd))).decode(),
    }
    result = _send(payload, timeout + 600)
    _apply(Path(cwd), result)
    if result.get("rc") is None:
        raise subprocess.TimeoutExpired(args, timeout)
    capture = kwargs.get("capture_output") or kwargs.get("stdout") == subprocess.PIPE
    text = bool(kwargs.get("text") or kwargs.get("universal_newlines") or kwargs.get("encoding"))
    out, err = result.get("stdout", ""), result.get("stderr", "")
    if not text:
        out, err = out.encode(), err.encode()
    completed = subprocess.CompletedProcess(args, result["rc"], out if capture else None, err if capture else None)
    if kwargs.get("check"):
        completed.check_returncode()
    return completed


class GuardedPopen(_Popen):
    """Every other way of starting a python interpreter (Popen, check_output, ...) would run the code on this host."""

    def __init__(self, args, *popenargs, **kwargs):
        if (isinstance(args, (list, tuple)) and args and _is_python(args[0])) or (
            isinstance(args, str) and re.search(r"\bpython", args)
        ):
            _die("a python command outside subprocess.run was refused: it would run on the agent's host")
        super().__init__(args, *popenargs, **kwargs)


def _refuse(name: str, *args, **kwargs):
    _die(f"os.{name} was refused: a process it starts would run on the agent's host")


def main() -> None:
    if "CI_EXEC_URL" not in os.environ:
        _die("CI_EXEC_URL is not set")
    subprocess.run = run
    subprocess.Popen = GuardedPopen
    for name in _OS_LAUNCHERS:
        if hasattr(os, name):
            setattr(os, name, functools.partial(_refuse, name))
    runpy.run_module("collab_intel.harness.cli", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
