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
"""Code runs in remote sandboxes through Gym's provider-neutral sandbox API (`nemo_gym.sandbox`, for example OpenSandbox).

Same interface as ``arc_sandbox.SandboxPool`` (``run``, ``predict_answer``, ``close``), so the agent does not change. The
pool holds ``num_workers`` remote sandboxes, each created on first use. A sandbox gets the vendored sandbox server and
helper files at creation (``SandboxSpec.files``) and installs numpy, scipy and networkx once with ``setup_command``. Every
code run uploads one request and runs ``sandbox_server.py`` on it: the server forks a child with the same resource limits
and restricted builtins as locally, replays the episode's earlier code blocks and runs the new one, so any sandbox can serve
any rollout and nothing outlives a run. A sandbox whose run fails is stopped and recreated on its next use; the failure
comes back to the model as an error reply.
"""

import asyncio
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from responses_api_agents.arc_code_agent.arc_sandbox import TOOLS_DIR, SandboxPool


LOGGER = logging.getLogger(__name__)
# What the model sees when a remote sandbox fails. Provider details (endpoints, exception text) go to the log only, so
# they never enter a trajectory or the exported training data.
UNAVAILABLE_REPLY = "sandbox unavailable; the code was not run, try again"
RESTARTED_REPLY = "sandbox restarted; the code was not run, try again"


# Flat under /tmp: it exists in every image, and file writes need not create parent directories.
REMOTE_DIR = "/tmp"
DEFAULT_SETUP_COMMAND = "python3 -m pip install --quiet --disable-pip-version-check numpy scipy networkx"
CHECK_COMMAND = "python3 -c 'import numpy, scipy, scipy.ndimage, networkx'"


def sandbox_files() -> Dict[str, str]:
    """The sandbox server and its helper modules, keyed by their path inside the sandbox."""
    return {f"{REMOTE_DIR}/{p.name}": p.read_text() for p in sorted(TOOLS_DIR.glob("*.py"))}


def build_spec(sandbox_spec: Dict[str, Any]):
    """A ``SandboxSpec`` from the YAML mapping, with the sandbox files added. Unknown keys fail loudly."""
    from nemo_gym.sandbox import SandboxResources, SandboxSpec

    spec = dict(sandbox_spec or {})
    files = {**dict(spec.pop("files", {})), **sandbox_files()}
    built = SandboxSpec(
        image=spec.pop("image", None),
        ttl_s=spec.pop("ttl_s", None),
        ready_timeout_s=spec.pop("ready_timeout_s", None),
        workdir=spec.pop("workdir", None),
        env=dict(spec.pop("env", {})),
        files=files,
        metadata=dict(spec.pop("metadata", {})),
        resources=SandboxResources.from_mapping(spec.pop("resources", {})),
        entrypoint=spec.pop("entrypoint", None),
        provider_options=dict(spec.pop("provider_options", {})),
    )
    if spec:
        raise ValueError(f"Unknown sandbox_spec keys: {', '.join(sorted(spec))}")
    return built


def parse_reply(stdout: str) -> Optional[Dict[str, Any]]:
    """The server's reply: the last JSON line of its stdout that is not the readiness line."""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            reply = json.loads(line)
        except ValueError:
            continue
        if isinstance(reply, dict) and "ready" not in reply:
            reply.setdefault("output", "")
            return reply
    return None


class _ApiServer:
    """One remote sandbox. Created on first use; stopped and recreated after a failed run."""

    def __init__(self, provider: Dict[str, Any], spec_config: Dict[str, Any], setup_command: Optional[str]) -> None:
        self.provider = provider
        self.spec_config = spec_config
        self.setup_command = setup_command
        self.sandbox = None

    async def start(self) -> None:
        from nemo_gym.sandbox import AsyncSandbox

        sandbox = await AsyncSandbox(self.provider, build_spec(self.spec_config)).start()
        try:
            if self.setup_command:
                setup = await sandbox.exec(self.setup_command, timeout_s=900)
                if setup.return_code != 0:
                    raise RuntimeError(f"setup failed ({setup.return_code}): {(setup.stderr or setup.stdout)[-300:]}")
            check = await sandbox.exec(CHECK_COMMAND, timeout_s=300)
            if check.return_code != 0:
                raise RuntimeError(f"numpy/scipy/networkx missing: {(check.stderr or check.stdout)[-300:]}")
        except BaseException:
            await sandbox.stop()
            raise
        self.sandbox = sandbox

    async def call(self, req: Dict[str, Any], wait_s: float) -> Dict[str, Any]:
        if self.sandbox is None:
            try:
                await self.start()
            except Exception as exc:
                LOGGER.warning("sandbox start failed: %s: %s", type(exc).__name__, exc)
                return {"output": "", "timed_out": False, "error": UNAVAILABLE_REPLY}
        remote = f"{REMOTE_DIR}/arc-req-{uuid.uuid4().hex}.json"
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
                handle.write(json.dumps(req) + "\n")
                local = Path(handle.name)
            try:
                await self.sandbox.upload(local, remote)
            finally:
                local.unlink(missing_ok=True)
            # The server execs the helpers, answers the one request on stdin and exits at end of input.
            result = await self.sandbox.exec(
                f"python3 -I {REMOTE_DIR}/sandbox_server.py {REMOTE_DIR} < {remote}; rm -f {remote}",
                timeout_s=wait_s + 60,
            )
            reply = parse_reply(result.stdout)
            if reply is None:
                raise RuntimeError(f"no reply (exit {result.return_code}): {(result.stderr or '')[-300:]}")
            return reply
        except Exception as exc:  # recreate the sandbox on next use and report to the model
            LOGGER.warning("sandbox run failed, restarting: %s: %s", type(exc).__name__, exc)
            await self.close()
            return {"output": "", "timed_out": False, "error": RESTARTED_REPLY}

    async def close(self) -> None:
        if self.sandbox is not None:
            sandbox, self.sandbox = self.sandbox, None
            try:
                await sandbox.stop()
            except Exception:
                pass


class SandboxApiPool(SandboxPool):
    """``SandboxPool`` whose servers are remote sandboxes; ``run`` takes one for the duration of a code run."""

    def __init__(
        self,
        provider: Dict[str, Any],
        sandbox_spec: Dict[str, Any],
        setup_command: Optional[str] = DEFAULT_SETUP_COMMAND,
        num_workers: int = 8,
        timeout_s: float = 30.0,
        out_chars: int = 4000,
        mem_mb: int = 6144,
        extra_imports: Optional[List[str]] = None,
        recursion_limit: int = 0,
    ) -> None:
        super().__init__(
            num_workers=num_workers,
            timeout_s=timeout_s,
            out_chars=out_chars,
            mem_mb=mem_mb,
            extra_imports=extra_imports,
            recursion_limit=recursion_limit,
        )
        if not provider:
            raise ValueError("sandbox_backend: sandbox_api needs sandbox_provider")
        build_spec(sandbox_spec)  # validate the spec now, not at the first code run
        self.provider = provider
        self.sandbox_spec = sandbox_spec
        self.setup_command = setup_command

    async def _ensure_started(self) -> asyncio.Queue:
        if self._idle is None:
            # Sandboxes are created lazily by their first run, so creates follow the load instead of bursting at start.
            idle: asyncio.Queue = asyncio.Queue()
            self._servers = [
                _ApiServer(self.provider, self.sandbox_spec, self.setup_command) for _ in range(self.num_workers)
            ]
            for server in self._servers:
                idle.put_nowait(server)
            self._idle = idle
        return self._idle
