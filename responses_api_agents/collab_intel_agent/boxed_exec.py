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
"""Where the policy's code runs in the coding scenarios: a sandbox, never this host.

The upstream harness runs the policy's code in python subprocesses (pytest in the policy's workspace, and the probes of a
few checkers). The agent starts the harness through boxed_cli.py, which sends each of those subprocesses to the agent's
/ci_exec route instead. There the workspace goes into one sandbox per phase, the command runs in it, and its output and
the files it changed come back, so the harness sees the directory as if the command had run in it.

The sandbox must have no network: a probe checks it when the sandbox starts. A provider failure, a sandbox with egress or
a run that changes too much is a SandboxExecError, which fails the episode (an evaluation failure, not a policy
failure). The code never falls back to this host.
"""

import asyncio
import base64
import json
import logging
import shlex
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import AsyncSandbox, SandboxProvider, SandboxSpec, create_provider


logger = logging.getLogger(__name__)

RUNTIME_DIR = "_rt"
RUNS_DIR = "runs"
# the changed files of one run that come back, in bytes
MAX_CHANGED_BYTES = 8 * 1024 * 1024
# a connect() to a public address, on a port that is not DNS (a deny policy still lets port 53 through): "egress" if it
# succeeds
EGRESS_PROBE = (
    "import socket\n"
    "try:\n"
    "    socket.create_connection(('1.1.1.1', 443), 3).close()\n"
    "    print('egress')\n"
    "except OSError:\n"
    "    print('blocked')\n"
)
# runs in the sandbox for one command ("@python" is its interpreter, "@dir" the unpacked workspace): unpack the
# workspace, run, report the output and the files the run changed
WRAPPER = """import base64, hashlib, json, os, shutil, subprocess, sys, tarfile
a = json.loads(base64.b64decode(sys.argv[1]))
d = a["dir"]
os.makedirs(d)
with tarfile.open(a["tar"]) as t:
    t.extractall(d, filter="data")
os.remove(a["tar"])


def snapshot():
    out = {}
    for root, dirs, names in os.walk(d):
        dirs[:] = [x for x in dirs if x not in ("__pycache__", ".pytest_cache")]
        for n in names:
            p = os.path.join(root, n)
            if os.path.isfile(p) and not os.path.islink(p):
                with open(p, "rb") as f:
                    out[os.path.relpath(p, d)] = hashlib.sha256(f.read()).hexdigest()
    return out


before = snapshot()
argv = [sys.executable if x == "@python" else x.replace("@dir", d) for x in a["argv"]]
env = {k: v.replace("@dir", d) for k, v in a["env"].items()}
env |= {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": a["pythonpath"]}
try:
    r = subprocess.run(argv, cwd=d, capture_output=True, text=True, errors="replace", timeout=a["timeout"], env=env,
                       check=False)
    res = {"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
except subprocess.TimeoutExpired:
    res = {"rc": None, "stdout": "", "stderr": ""}
after = snapshot()
changed, size = {}, 0
for n, h in after.items():
    if before.get(n) != h:
        with open(os.path.join(d, n), "rb") as f:
            data = f.read()
        size += len(data)
        if size > a["max_bytes"]:
            res["too_large"] = True
            break
        changed[n] = base64.b64encode(data).decode()
res["changed"] = changed
res["deleted"] = sorted(set(before) - set(after))
shutil.rmtree(d, ignore_errors=True)
print(json.dumps(res))
"""


class SandboxExecError(RuntimeError):
    """The code could not run in the sandbox: a provider failure, a sandbox with egress, or a request that does not fit.
    Never a policy outcome."""


def _runtime_unpack(tar: str, target: str) -> str:
    return (
        f"python -c \"import sys, tarfile; tarfile.open(sys.argv[1]).extractall(sys.argv[2], filter='data')\" "
        f"{shlex.quote(tar)} {shlex.quote(target)}"
    )


class BoxedExec:
    """One sandbox per phase, created when the phase's first command arrives and stopped when the phase ends."""

    def __init__(
        self, provider: Mapping[str, Any] | SandboxProvider, spec: SandboxSpec, *, require_no_egress: bool = True
    ) -> None:
        self._provider = create_provider(provider) if isinstance(provider, Mapping) else provider
        self._spec = spec
        self._require_no_egress = require_no_egress
        self._workdir = (spec.workdir or "/tmp").rstrip("/")
        self._boxes: dict[str, AsyncSandbox] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _put(self, box: AsyncSandbox, data: bytes, remote: str) -> None:
        with tempfile.TemporaryDirectory(prefix="ci-box-upload-") as tmp:
            local = Path(tmp) / "payload"
            local.write_bytes(data)
            await box.upload(local, remote)

    async def _sh(self, box: AsyncSandbox, command: str, *, timeout_s: float = 120) -> str:
        result = await box.exec(command, cwd=self._workdir, timeout_s=timeout_s)
        if result.error_type is not None or result.return_code != 0:
            raise SandboxExecError(f"sandbox command failed ({result.error_type or result.return_code})")
        return result.stdout or ""

    async def _start(self, phase: str, runtime: bytes | None) -> AsyncSandbox:
        """The phase's sandbox, with the runtime (pytest and its pure-python dependencies) unpacked in it."""
        async with self._locks.setdefault(phase, asyncio.Lock()):
            if phase in self._boxes:
                return self._boxes[phase]
            if runtime is None:
                raise SandboxExecError("the first command of a phase must carry the runtime")
            box = AsyncSandbox(self._provider, owns_provider=False)
            try:
                await box.start(self._spec)
            except Exception as exc:
                raise SandboxExecError(f"sandbox did not start ({type(exc).__name__})") from None
            try:
                probe = (await self._sh(box, f"python -c {shlex.quote(EGRESS_PROBE)}")).strip()
                if probe not in ("blocked", "egress"):
                    raise SandboxExecError("the sandbox has no working python")
                if probe == "egress" and self._require_no_egress:
                    raise SandboxExecError(
                        "the sandbox can open connections to other hosts: set a network policy that blocks them in "
                        "sandbox_spec.provider_options, or require_no_egress: false to run with egress"
                    )
                if probe == "egress":
                    logger.warning("collab_intel sandbox has network egress (require_no_egress is false)")
                await self._put(box, runtime, f"{self._workdir}/{RUNTIME_DIR}.tar")
                await self._sh(box, f"mkdir -p {RUNTIME_DIR} {RUNS_DIR}")
                await self._sh(
                    box, _runtime_unpack(f"{self._workdir}/{RUNTIME_DIR}.tar", f"{self._workdir}/{RUNTIME_DIR}")
                )
                await self._put(box, WRAPPER.encode(), f"{self._workdir}/wrap.py")
            except BaseException as exc:
                await self._stop(box)
                if isinstance(exc, (SandboxExecError, asyncio.CancelledError)):
                    raise
                raise SandboxExecError(f"sandbox setup failed ({type(exc).__name__})") from None
            self._boxes[phase] = box
            return box

    async def run(self, phase: str, request: dict[str, Any]) -> dict[str, Any]:
        """One command of the harness: request has argv ("@python" for the interpreter), env, timeout, a tar of the
        workspace (files), and on the phase's first command the runtime tar. Returns the exit code (None on a
        timeout), stdout, stderr, the changed files by relative path (base64) and the deleted paths."""
        try:
            runtime = base64.b64decode(request["runtime"]) if request.get("runtime") else None
            box = await self._start(phase, runtime)
            run_dir = f"{self._workdir}/{RUNS_DIR}/{uuid.uuid4().hex}"
            await self._put(box, base64.b64decode(request["files"]), f"{run_dir}.tar")
            args = {
                "dir": run_dir,
                "tar": f"{run_dir}.tar",
                "argv": request["argv"],
                "env": request["env"],
                "timeout": request["timeout"],
                "pythonpath": f"{self._workdir}/{RUNTIME_DIR}",
                "max_bytes": MAX_CHANGED_BYTES,
            }
            encoded = base64.b64encode(json.dumps(args).encode()).decode()
            out = await self._sh(
                box,
                f"python {shlex.quote(self._workdir + '/wrap.py')} {shlex.quote(encoded)}",
                timeout_s=request["timeout"] + 120,
            )
            result = json.loads(out.strip().splitlines()[-1])
        except SandboxExecError:
            raise
        except (KeyError, ValueError, TypeError) as exc:
            raise SandboxExecError(f"bad sandbox run ({type(exc).__name__})") from None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise SandboxExecError(f"sandbox run failed ({type(exc).__name__})") from None
        if result.pop("too_large", False):
            raise SandboxExecError(f"the command changed more than {MAX_CHANGED_BYTES} bytes of files")
        return result

    async def _stop(self, box: AsyncSandbox) -> None:
        try:
            await box.stop()
        except Exception as exc:  # the spec's ttl is the backstop
            logger.warning("collab_intel sandbox stop failed (%s)", type(exc).__name__)

    async def release(self, phase: str) -> None:
        box = self._boxes.pop(phase, None)
        self._locks.pop(phase, None)
        if box is not None:
            await self._stop(box)

    async def close(self) -> None:
        for phase in list(self._boxes):
            await self.release(phase)
        try:
            await self._provider.aclose()
        except Exception as exc:
            logger.warning("sandbox provider close failed (%s)", type(exc).__name__)
