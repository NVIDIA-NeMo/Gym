# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""HTTP client for communicating with Lean4 sandbox container.

Reference sandbox implementation:
- Server: https://github.com/NVIDIA-NeMo/NeMo-Skills/tree/main/nemo_skills/code_execution/local_sandbox
- Dockerfile: https://github.com/NVIDIA-NeMo/NeMo-Skills/blob/main/dockerfiles/Dockerfile.sandbox
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from typing import Any, Dict

import httpx

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, await_cleanup


LOG = logging.getLogger(__name__)


class Lean4SandboxClient:
    """Async HTTP client for Lean4 proof compilation sandbox."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6000,
        max_output_characters: int = 1000,
    ):
        """Initialize sandbox client.

        Args:
            host: Sandbox server hostname
            port: Sandbox server port
            max_output_characters: Maximum characters in output
        """
        self.host = host
        self.port = port
        self.max_output_characters = max_output_characters
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=100, max_connections=100),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_execute_url(self) -> str:
        """Get the sandbox execute endpoint URL."""
        return f"http://{self.host}:{self.port}/execute"

    async def execute_lean4(
        self,
        code: str,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Execute Lean4 code in the sandbox.

        Args:
            code: Complete Lean4 code to compile
            timeout: Compilation timeout in seconds

        Returns:
            Dictionary with process_status, stdout, stderr
        """
        request_data = {
            "generated_code": code,
            "language": "lean4",
            "timeout": timeout,
            "max_output_characters": self.max_output_characters,
        }

        client = await self._get_client()

        try:
            response = await client.post(
                url=self._get_execute_url(),
                content=json.dumps(request_data),
                timeout=timeout + 5.0,  # Add buffer for network overhead
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 502:
                LOG.warning("Sandbox returned 502 error")
                return {"process_status": "error", "stdout": "", "stderr": "Sandbox 502 error"}

            return response.json()

        except httpx.TimeoutException:
            LOG.warning("Sandbox request timed out after %.1f seconds", timeout)
            return {"process_status": "timeout", "stdout": "", "stderr": "Client timed out"}

        except httpx.HTTPError as e:
            LOG.error("HTTP error communicating with sandbox: %s", e)
            return {"process_status": "error", "stdout": "", "stderr": str(e)}

        except json.JSONDecodeError as e:
            LOG.error("Failed to parse sandbox response: %s", e)
            return {"process_status": "error", "stdout": "", "stderr": "Invalid JSON response"}

    async def health_check(self, timeout: float = 5.0) -> bool:
        """Check if sandbox is healthy.

        Args:
            timeout: Timeout for health check

        Returns:
            True if sandbox is healthy, False otherwise
        """
        url = f"http://{self.host}:{self.port}/health"
        client = await self._get_client()

        try:
            response = await client.get(url=url, timeout=timeout)
            return response.status_code == 200
        except httpx.HTTPError:
            return False


class GymSandboxLean4Client:
    """Lean4 compilation on per-verify OpenSandbox pods via provider exec.

    Runs the same Lean command and preserves the NS server's result contract. A
    configured warm pool reuses prepared sandboxes and replaces failed leases;
    otherwise each verification gets a fresh sandbox.
    """

    def __init__(
        self,
        provider: Dict[str, Any],
        image: str,
        project_dir: str = "/lean4/my_project",
        max_concurrent: int = 8,
        acquire_timeout_s: float = 120.0,
        create_ttl_s: float = 3600.0,
        resources: Dict[str, Any] | None = None,
        max_output_characters: int = 1000,
        pool_size: int = 0,
        prefetch_paths: str = "/root/.elan /lean4",
        pool_ref: str = "",
    ):
        if not image:
            raise ValueError("sandbox_backend=gym_sandbox requires a non-empty image")
        connection = (provider.get("opensandbox") or {}).get("connection", {}) if provider else {}
        if not connection.get("domain") or not connection.get("api_key"):
            raise ValueError(
                "sandbox_backend=gym_sandbox requires provider connection domain/api_key — "
                "set OPENSANDBOX_BASE_URL / OPENSANDBOX_API_KEY"
            )
        self._provider = provider
        self._image = image
        self._project_dir = project_dir.rstrip("/")
        self._create_ttl_s = create_ttl_s
        self._resources = dict(resources or {})
        self.max_output_characters = max_output_characters
        self._acquire_timeout_s = acquire_timeout_s
        self._semaphore_size = int(max_concurrent)
        self._semaphore = asyncio.Semaphore(self._semaphore_size)
        self._pool_size = int(pool_size)
        self._prefetch_paths = prefetch_paths
        self._pool_ref = pool_ref or ""
        self._pool: asyncio.Queue[AsyncSandbox] | None = None
        self._pool_sandboxes: set[AsyncSandbox] = set()
        self._fill_tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    def _new_sandbox(self, files: Dict[str, str] | None = None, use_pool: bool = True) -> AsyncSandbox:
        # pool_ref claims a prewarmed pod from a server-side Pool whose template
        # has already warmed the lean toolchain, so the prepare-time prefetch
        # degrades to a fast cache hit.
        provider_options = {"extensions": {"poolRef": self._pool_ref}} if (self._pool_ref and use_pool) else {}
        return AsyncSandbox(
            provider=dict(self._provider),
            spec=SandboxSpec(
                image=self._image,
                entrypoint=["sleep", "infinity"],
                ttl_s=self._create_ttl_s,
                files=files or {},
                resources=self._resources,
                metadata={"purpose": "math-formal-lean-verify"},
                provider_options=provider_options,
            ),
        )

    async def _start_sandbox(self, files: Dict[str, str] | None = None) -> AsyncSandbox:
        """Start a sandbox, degrading a failed pool claim to a direct create."""
        sandbox = self._new_sandbox(files)
        try:
            await sandbox.start()
            return sandbox
        except Exception as exc:
            if not self._pool_ref:
                raise
            LOG.warning("lean pool '%s' claim failed (%s); falling back to a direct create", self._pool_ref, exc)
            sandbox = self._new_sandbox(files, use_pool=False)
            await sandbox.start()
            return sandbox

    async def _stop_sandbox(self, sandbox: AsyncSandbox) -> None:
        self._pool_sandboxes.discard(sandbox)
        try:
            await sandbox.stop()
        except Exception as exc:
            LOG.warning("lean sandbox teardown failed (TTL will reap): %s", exc)

    async def _create_pool_pod(self) -> AsyncSandbox:
        """Create + warm one pool pod: a single bulk tar read pulls the olean tree at
        line rate (concurrent chunk fetches) instead of the compile's serial faults."""
        sandbox = await self._start_sandbox()
        try:
            # NOT `tar cf /dev/null`: GNU tar detects the null sink and skips reading
            # file contents, silently defeating the prefetch.
            await sandbox.exec(
                f"find {self._prefetch_paths} -type f -exec cat {{}} + > /dev/null 2>&1; true", timeout_s=1800
            )
        except asyncio.CancelledError:
            await self._stop_sandbox(sandbox)
            raise
        except Exception:
            pass  # prefetch is an optimization; the first compile warms the rest
        return sandbox

    def start_pool(self) -> None:
        """Kick the pool fill early (call from server lifespan startup) so the first
        verify does not pay pool warmup inside its admission window."""
        if self._pool_size <= 0 or self._pool is not None or self._closed:
            return
        self._pool = asyncio.Queue(maxsize=self._pool_size)
        for _ in range(self._pool_size):
            self._schedule_fill()

    def _schedule_fill(self) -> None:
        if self._closed:
            return
        task = asyncio.create_task(self._fill_one())
        self._fill_tasks.add(task)
        task.add_done_callback(self._fill_tasks.discard)

    async def _fill_one(self) -> None:
        retry_s = 1.0
        while not self._closed:
            try:
                sandbox = await self._create_pool_pod()
            except Exception as exc:
                LOG.error("lean pool pod create failed; retrying in %.0fs: %s", retry_s, exc)
                await asyncio.sleep(retry_s)
                retry_s = min(retry_s * 2, 30.0)
                continue

            if self._closed:
                await self._stop_sandbox(sandbox)
            else:
                self._pool_sandboxes.add(sandbox)
                self._pool.put_nowait(sandbox)
            return

    async def _execute_pooled(self, code: str, timeout: float) -> Dict[str, Any]:
        self.start_pool()
        if self._pool is None:
            raise RuntimeError("Lean sandbox pool is closed")
        pool = self._pool
        for attempt in (1, 2):
            try:
                sandbox = await asyncio.wait_for(pool.get(), timeout=self._acquire_timeout_s)
            except asyncio.TimeoutError:
                LOG.warning("Lean pool admission timed out after %.0fs", self._acquire_timeout_s)
                return {"process_status": "timeout", "stdout": "", "stderr": "Client timed out"}
            proof_name = f"proof_{uuid.uuid4().hex}.lean"
            try:
                with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as fh:
                    fh.write(code)
                try:
                    await sandbox.upload(fh.name, f"{self._project_dir}/{proof_name}")
                finally:
                    os.unlink(fh.name)
                # rc must survive the cleanup rm: 124/137 keep mapping to timeout.
                command = (
                    f"cd {self._project_dir} && timeout -s KILL {timeout} "
                    f"lake env --dir {self._project_dir} lean {proof_name}; "
                    f"rc=$?; rm -f {proof_name}; exit $rc"
                )
                result = await sandbox.exec(command, timeout_s=timeout + 60)
            except asyncio.CancelledError:
                await self._stop_sandbox(sandbox)
                self._schedule_fill()
                raise
            except Exception as exc:
                # Pod is suspect (TTL expiry, node loss): replace it, retry once elsewhere.
                LOG.warning("lean pool pod failed (attempt %d), replacing: %s", attempt, exc)
                await self._stop_sandbox(sandbox)
                self._schedule_fill()
                if attempt == 1:
                    continue
                return {"process_status": "error", "stdout": "", "stderr": str(exc)}
            if self._closed:
                await self._stop_sandbox(sandbox)
            else:
                pool.put_nowait(sandbox)
            return self._map_result(result, timeout)
        return {"process_status": "error", "stdout": "", "stderr": "lean pool exhausted"}

    def _map_result(self, result, timeout: float) -> Dict[str, Any]:
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if result.return_code == 0:
            process_status = "completed"
        elif result.return_code in (124, 137, -9):
            process_status = "timeout"
            stderr += f"Execution timed out after {timeout} seconds\n"
        else:
            process_status = "failed"
        if len(stdout) > self.max_output_characters:
            stdout = stdout[: self.max_output_characters] + "<output cut>"
        if len(stderr) > self.max_output_characters:
            stderr = stderr[: self.max_output_characters] + "<output cut>"
        return {"process_status": process_status, "stdout": stdout, "stderr": stderr}

    async def execute_lean4(self, code: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Same signature and return contract as Lean4SandboxClient.execute_lean4."""
        if self._pool_size > 0:
            return await self._execute_pooled(code, timeout)

        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._acquire_timeout_s)
        except asyncio.TimeoutError:
            LOG.warning("Lean sandbox admission timed out after %.0fs", self._acquire_timeout_s)
            return {"process_status": "timeout", "stdout": "", "stderr": "Client timed out"}

        proof_name = f"proof_{uuid.uuid4().hex}.lean"
        sandbox = None
        try:
            sandbox = await self._start_sandbox(files={f"{self._project_dir}/{proof_name}": code})
            # In-sandbox `timeout -s KILL` must always fire before the provider deadline so the
            # partial-stdout + "Execution timed out..." contract is preserved (never a raw exec kill).
            command = (
                f"cd {self._project_dir} && timeout -s KILL {timeout} "
                f"lake env --dir {self._project_dir} lean {proof_name}"
            )
            result = await sandbox.exec(command, timeout_s=timeout + 60)
            return self._map_result(result, timeout)
        except Exception as e:  # infra failure -> degrade, never raise into verify()
            LOG.error("OpenSandbox lean4 execution failed: %s", e)
            return {"process_status": "error", "stdout": "", "stderr": str(e)}
        finally:
            self._semaphore.release()
            if sandbox is not None:
                await self._stop_sandbox(sandbox)

    async def close(self) -> None:
        """Stop pool maintenance and every warm sandbox owned by this client."""
        if self._close_task is None:
            self._closed = True

            async def cleanup() -> None:
                tasks = tuple(self._fill_tasks)
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                sandboxes = tuple(self._pool_sandboxes)
                if sandboxes:
                    await asyncio.gather(*(self._stop_sandbox(sandbox) for sandbox in sandboxes))
                self._pool = None

            self._close_task = asyncio.create_task(cleanup())
        await await_cleanup(self._close_task)
