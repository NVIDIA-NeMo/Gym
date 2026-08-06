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

import json
import logging
import os
from typing import Any, Dict

import httpx


LOG = logging.getLogger(__name__)


class Lean4SandboxClient:
    """Async HTTP client for Lean4 proof compilation sandbox."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6000,
        max_output_characters: int = 1000,
        base_url: str | None = None,
        extra_headers: Dict[str, str] | None = None,
    ):
        """Initialize sandbox client.

        Args:
            host: Sandbox server hostname
            port: Sandbox server port
            max_output_characters: Maximum characters in output
            base_url: Full base URL override (e.g. an OpenSandbox proxied endpoint);
                when set, host/port are ignored.
            extra_headers: Headers added to every request (e.g. proxy auth).
        """
        self.host = host
        self.port = port
        self.max_output_characters = max_output_characters
        self.base_url = base_url.rstrip("/") if base_url else None
        self.extra_headers = dict(extra_headers or {})
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
        if self.base_url:
            return f"{self.base_url}/execute"
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
                headers={"Content-Type": "application/json", **self.extra_headers},
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
        base = self.base_url if self.base_url else f"http://{self.host}:{self.port}"
        url = f"{base}/health"
        client = await self._get_client()

        try:
            response = await client.get(url=url, timeout=timeout, headers=self.extra_headers)
            return response.status_code == 200
        except httpx.HTTPError:
            return False


class OpenSandboxLean4Client:
    """Lean4 compilation on per-verify OpenSandbox pods via provider exec.

    Reimplements the NS server's lean4 invocation exactly (reference frozen at
    nemo_skills local_sandbox_server.py:631-685 @ da85a881): the proof lands in
    /lean4/my_project, `lake env --dir /lean4/my_project lean <file>` runs with an
    in-sandbox `timeout -s KILL`, and the exit code maps to the same
    process_status/stdout/stderr contract as `Lean4SandboxClient.execute_lean4`.
    Long compiles never hold an HTTP connection open (background/short exec
    requests), so proxy read-timeout ceilings do not apply.

    With ``pool_size=0`` every verify gets a fresh pod (created under a bounded
    semaphore, destroyed in finally). With ``pool_size=N`` a warm pool of N pods is
    built lazily and reused across verifies: a fresh pod's first `import Mathlib`
    lazy-pulls ~5GB of olean files through nydus one FUSE fault at a time (measured
    902s measured on a production cluster), while a warmed pod compiles in ~4s — so pool pods bulk-prefetch
    the olean tree once at prepare and then serve verifies back-to-back. A pod that
    hits an infra error (incl. TTL expiry) is killed and replaced in place; the
    verify retries once on a second pod before degrading. Infra failures degrade to
    the client's existing error/timeout shapes, never raise into verify().
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
    ):
        if not image:
            raise ValueError("sandbox_backend=opensandbox requires a non-empty image")
        connection = (next(iter(provider.values()), {}) or {}).get("connection", {}) if provider else {}
        if not connection.get("domain") or not connection.get("api_key"):
            raise ValueError(
                "sandbox_backend=opensandbox requires provider connection domain/api_key — "
                "set OPENSANDBOX_BASE_URL / OPENSANDBOX_API_KEY"
            )
        self._provider = provider
        self._image = image
        self._project_dir = project_dir.rstrip("/")
        self._create_ttl_s = create_ttl_s
        self._resources = dict(resources or {})
        self.max_output_characters = max_output_characters
        self._acquire_timeout_s = acquire_timeout_s
        self._semaphore_size = max_concurrent
        self._semaphore: Any = None  # bound lazily to the serving event loop
        self._pool_size = int(pool_size)
        self._prefetch_paths = prefetch_paths
        self._pool: Any = None  # asyncio.Queue of warm AsyncSandbox pods, filled lazily
        self._pool_started = False

    def _get_semaphore(self):
        import asyncio

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._semaphore_size)
        return self._semaphore

    def _new_sandbox(self, files: Dict[str, str] | None = None):
        from nemo_gym.sandbox.api import AsyncSandbox
        from nemo_gym.sandbox.providers.base import SandboxSpec

        return AsyncSandbox(
            provider=dict(self._provider),
            spec=SandboxSpec(
                image=self._image,
                entrypoint=["sleep", "infinity"],
                ttl_s=self._create_ttl_s,
                files=files or {},
                resources=self._resources,
                metadata={"purpose": "math-formal-lean-verify"},
            ),
        )

    async def _create_pool_pod(self):
        """Create + warm one pool pod: a single bulk tar read pulls the olean tree at
        line rate (concurrent chunk fetches) instead of the compile's serial faults."""
        sandbox = self._new_sandbox()
        await sandbox.start()
        try:
            # NOT `tar cf /dev/null`: GNU tar detects the null sink and skips reading
            # file contents, silently defeating the prefetch.
            await sandbox.exec(
                f"find {self._prefetch_paths} -type f -exec cat {{}} + > /dev/null 2>&1; true", timeout_s=1800
            )
        except Exception:
            pass  # prefetch is an optimization; the first compile warms the rest
        return sandbox

    def start_pool(self) -> None:
        """Kick the pool fill early (call from server lifespan startup) so the first
        verify does not pay pool warmup inside its admission window."""
        if self._pool_size > 0:
            self._pool_queue()

    @property
    def pool_ready_count(self) -> int:
        return self._pool.qsize() if self._pool is not None else 0

    def _pool_queue(self):
        import asyncio

        if self._pool is None:
            self._pool = asyncio.Queue()
            for _ in range(self._pool_size):
                asyncio.get_running_loop().create_task(self._fill_one())
        return self._pool

    async def _fill_one(self):
        try:
            self._pool.put_nowait(await self._create_pool_pod())
        except Exception as e:
            LOG.error("lean pool pod create failed (capacity reduced until next heal): %s", e)

    async def _execute_pooled(self, code: str, timeout: float) -> Dict[str, Any]:
        import asyncio
        import uuid

        pool = self._pool_queue()
        for attempt in (1, 2):
            try:
                sandbox = await asyncio.wait_for(pool.get(), timeout=self._acquire_timeout_s)
            except asyncio.TimeoutError:
                LOG.warning("Lean pool admission timed out after %.0fs", self._acquire_timeout_s)
                return {"process_status": "timeout", "stdout": "", "stderr": "Client timed out"}
            proof_name = f"proof_{uuid.uuid4().hex}.lean"
            try:
                import tempfile

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
            except Exception as e:
                # Pod is suspect (TTL expiry, node loss): replace it, retry once elsewhere.
                LOG.warning("lean pool pod failed (attempt %d), replacing: %s", attempt, e)
                try:
                    await sandbox.stop()
                except Exception:
                    pass
                asyncio.get_running_loop().create_task(self._fill_one())
                if attempt == 1:
                    continue
                return {"process_status": "error", "stdout": "", "stderr": str(e)}
            pool.put_nowait(sandbox)
            return self._map_result(result, timeout)
        return {"process_status": "error", "stdout": "", "stderr": "lean pool exhausted"}

    def _map_result(self, result, timeout: float) -> Dict[str, Any]:
        stdout = (result.stdout or "")[: self.max_output_characters]
        stderr = (result.stderr or "")[: self.max_output_characters]
        if result.return_code == 0:
            return {"process_status": "completed", "stdout": stdout, "stderr": stderr}
        if result.return_code in (124, 137, -9):
            return {
                "process_status": "timeout",
                "stdout": stdout,
                "stderr": stderr + f"Execution timed out after {timeout} seconds\n",
            }
        return {"process_status": "failed", "stdout": stdout, "stderr": stderr}

    async def execute_lean4(self, code: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Same signature and return contract as Lean4SandboxClient.execute_lean4."""
        import asyncio
        import uuid

        if self._pool_size > 0:
            return await self._execute_pooled(code, timeout)

        try:
            await asyncio.wait_for(self._get_semaphore().acquire(), timeout=self._acquire_timeout_s)
        except asyncio.TimeoutError:
            LOG.warning("Lean sandbox admission timed out after %.0fs", self._acquire_timeout_s)
            return {"process_status": "timeout", "stdout": "", "stderr": "Client timed out"}

        proof_name = f"proof_{uuid.uuid4().hex}.lean"
        sandbox = self._new_sandbox(files={f"{self._project_dir}/{proof_name}": code})
        try:
            await sandbox.start()
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
            self._get_semaphore().release()
            try:
                await sandbox.stop()
            except Exception as exc:
                LOG.warning("lean sandbox teardown failed (TTL will reap): %s", exc)
