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

"""HTTP client for communicating with Lean4 sandbox container."""

import json
import logging
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
