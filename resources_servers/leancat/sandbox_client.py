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

"""Client for the Lean 4 compilation sandbox.

Talks to the same NeMo-Skills sandbox ``math_formal_lean`` uses:
- Server: https://github.com/NVIDIA-NeMo/NeMo-Skills/tree/main/nemo_skills/code_execution/local_sandbox
- Dockerfile: https://github.com/NVIDIA-NeMo/NeMo-Skills/blob/main/dockerfiles/Dockerfile.sandbox

Unlike that server's client this one goes through ``nemo_gym.server_utils.request`` --
Gym's shared aiohttp client -- rather than httpx, per ``AGENTS.md``.

The sandbox must be built against Mathlib v4.19.0. LeanCat statements are written against
that release's ``CategoryTheory`` API, and a sandbox on a different Mathlib will fail
tasks for reasons that have nothing to do with the model.
"""

import asyncio
import base64
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

import aiohttp

from nemo_gym.server_utils import request


if TYPE_CHECKING:
    from nemo_gym.sandbox import SandboxSpec


LOG = logging.getLogger(__name__)


class Lean4SandboxClient:
    """Compiles a complete Lean 4 file and reports what the compiler said."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6000,
        max_output_characters: int = 4000,
    ):
        self.host = host
        self.port = port
        self.max_output_characters = max_output_characters

    @property
    def execute_url(self) -> str:
        return f"http://{self.host}:{self.port}/execute"

    async def execute_lean4(self, code: str, timeout: float = 300.0) -> Dict[str, Any]:
        """Compile ``code`` and return ``{process_status, stdout, stderr}``.

        Never raises: a sandbox that is down or wedged has to score as a failed proof
        with a diagnosable status, not take the rollout down with it.
        """
        payload = {
            "generated_code": code,
            "language": "lean4",
            "timeout": timeout,
            "max_output_characters": self.max_output_characters,
        }

        try:
            response = await request(
                "POST",
                self.execute_url,
                json=payload,
                # Buffer over the compiler's own timeout so the sandbox, not the client,
                # is the one that reports a genuine compile timeout.
                timeout=aiohttp.ClientTimeout(total=timeout + 30.0),
            )
            if response.status != 200:
                body = (await response.text())[: self.max_output_characters]
                LOG.warning("Sandbox returned HTTP %d", response.status)
                return {
                    "process_status": "error",
                    "stdout": "",
                    "stderr": f"Sandbox HTTP {response.status}: {body}",
                }
            # content_type=None: the sandbox has been seen to answer with text/plain.
            return await response.json(content_type=None)

        except asyncio.TimeoutError:
            LOG.warning("Sandbox request timed out after %.1fs", timeout)
            return {"process_status": "timeout", "stdout": "", "stderr": "Client timed out"}
        except aiohttp.ClientError as exc:
            LOG.error("Error talking to Lean sandbox: %s", exc)
            return {"process_status": "error", "stdout": "", "stderr": str(exc)}
        except json.JSONDecodeError as exc:
            LOG.error("Malformed sandbox response: %s", exc)
            return {"process_status": "error", "stdout": "", "stderr": "Invalid JSON response"}

    async def health_check(self, timeout: float = 5.0) -> bool:
        try:
            response = await request(
                "GET",
                f"http://{self.host}:{self.port}/health",
                timeout=aiohttp.ClientTimeout(total=timeout),
            )
            return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False


class GymSandboxLean4Client:
    """Compile Lean 4 inside a Gym-managed sandbox instead of an HTTP sandbox server.

    Same ``execute_lean4`` contract as :class:`Lean4SandboxClient`, so ``app.py`` picks one
    or the other and nothing else changes.

    This backend exists because the HTTP route needs a sandbox process that somebody else
    started and kept alive at a known host:port -- which on Slurm means a second
    ``srun --overlap`` outside Gym's control, since ``ServiceConfig`` models only ``vllm``
    and ``ray``. Going through ``nemo_gym.sandbox`` instead lets the resources server own
    its own sandbox, the way ``swebench``/``deepswe``/``litmus_agent`` already do, and makes
    the enroot provider (the HPC-native one) usable without any extra launch step.

    It also runs the same command upstream does -- ``lake env lean <file>`` -- rather than
    NeMo-Skills' Flask wrapper around it.

    One sandbox is created lazily and shared: verification is stateless, and the provider's
    own ``exec.concurrency`` bounds parallel compiles.
    """

    def __init__(
        self,
        provider: Dict[str, Any],
        spec: Dict[str, Any],
        lean_project_dir: str = "/lean4/my_project",
        max_output_characters: int = 4000,
    ):
        self.provider = provider
        self.spec = spec
        self.lean_project_dir = lean_project_dir
        self.max_output_characters = max_output_characters
        self._sandbox: Any = None
        self._lock: Optional[asyncio.Lock] = None

    def _build_spec(self) -> "SandboxSpec":
        from nemo_gym.sandbox import SandboxResources, SandboxSpec

        spec = dict(self.spec)
        known = SandboxSpec(
            image=spec.pop("image", None),
            ttl_s=spec.pop("ttl_s", None),
            ready_timeout_s=spec.pop("ready_timeout_s", None),
            workdir=spec.pop("workdir", None),
            env=dict(spec.pop("env", {})),
            files=dict(spec.pop("files", {})),
            metadata=dict(spec.pop("metadata", {})),
            resources=SandboxResources.from_mapping(spec.pop("resources", {})),
            entrypoint=spec.pop("entrypoint", None),
            provider_options=dict(spec.pop("provider_options", {})),
        )
        if spec:
            raise ValueError(f"Unknown sandbox_spec keys: {', '.join(sorted(spec))}")
        return known

    async def _ensure_sandbox(self) -> Any:
        from nemo_gym.sandbox import AsyncSandbox

        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._sandbox is None:
                LOG.info("Starting Lean sandbox via nemo_gym.sandbox provider")
                self._sandbox = await AsyncSandbox(self.provider, self._build_spec()).start()
        return self._sandbox

    async def execute_lean4(self, code: str, timeout: float = 300.0) -> Dict[str, Any]:
        """Compile ``code`` with ``lake env lean`` and report what the compiler said.

        The file is shipped in base64 and decoded inside the sandbox rather than
        interpolated into the shell command. Lean sources are full of quotes, backslashes
        and unicode, and a heredoc delimiter can appear inside a proof -- base64 removes
        every quoting question at once.
        """
        try:
            sandbox = await self._ensure_sandbox()
        except Exception as exc:  # pragma: no cover - provider-specific failures
            LOG.error("Could not start Lean sandbox: %s", exc)
            return {"process_status": "error", "stdout": "", "stderr": f"Sandbox start failed: {exc}"}

        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        path = f"/tmp/leancat_{uuid.uuid4().hex}.lean"
        command = (
            f"printf %s {encoded} | base64 -d > {path} && "
            f"cd {self.lean_project_dir} && lake env lean {path}; "
            # Preserve lean's status across the cleanup so a failed compile is not masked.
            f"status=$?; rm -f {path}; exit $status"
        )

        try:
            result = await sandbox.exec(command, timeout_s=timeout)
        except Exception as exc:  # pragma: no cover - provider-specific failures
            LOG.error("Lean sandbox exec failed: %s", exc)
            return {"process_status": "error", "stdout": "", "stderr": f"Sandbox exec failed: {exc}"}

        stdout = (result.stdout or "")[: self.max_output_characters]
        stderr = (result.stderr or "")[: self.max_output_characters]

        if getattr(result, "error_type", None) == "timeout":
            return {"process_status": "timeout", "stdout": stdout, "stderr": stderr, "return_code": None}
        if getattr(result, "error_type", None):
            return {"process_status": "error", "stdout": stdout, "stderr": stderr, "return_code": None}

        # `completed` means the command ran, not that Lean accepted the file; `return_code`
        # carries that, and determine_proof_status treats non-zero as a compile error.
        return {
            "process_status": "completed",
            "stdout": stdout,
            "stderr": stderr,
            "return_code": result.return_code,
        }

    async def health_check(self, timeout: float = 5.0) -> bool:
        try:
            sandbox = await self._ensure_sandbox()
            result = await sandbox.exec("lake --version", timeout_s=timeout)
            return result.return_code == 0
        except Exception:
            return False
