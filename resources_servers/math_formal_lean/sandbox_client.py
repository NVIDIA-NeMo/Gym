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

Uses Gym's shared aiohttp client (``nemo_gym.server_utils.request``) rather than httpx, as
``AGENTS.md`` requires.
"""

import asyncio
import json
import logging
from typing import Any, Dict

import aiohttp

from nemo_gym.server_utils import request


LOG = logging.getLogger(__name__)


class Lean4SandboxClient:
    """Async HTTP client for Lean4 proof compilation sandbox."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6000,
        max_output_characters: int = 1000,
        timeout_buffer: float = 5.0,
    ):
        """Initialize sandbox client.

        Args:
            host: Sandbox server hostname
            port: Sandbox server port
            max_output_characters: Maximum characters in output
            timeout_buffer: Seconds the HTTP client waits beyond the compile timeout, so the
                sandbox rather than the client reports a genuine compile timeout.
        """
        self.host = host
        self.port = port
        self.max_output_characters = max_output_characters
        self.timeout_buffer = timeout_buffer

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

        try:
            response = await request(
                "POST",
                self._get_execute_url(),
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=timeout + self.timeout_buffer),
            )

            if response.status != 200:
                body = (await response.text())[: self.max_output_characters]
                LOG.warning("Sandbox returned HTTP %d", response.status)
                return {"process_status": "error", "stdout": "", "stderr": f"Sandbox HTTP {response.status}: {body}"}

            # content_type=None: the sandbox has been seen to answer with text/plain.
            return await response.json(content_type=None)

        except asyncio.TimeoutError:
            LOG.warning("Sandbox request timed out after %.1f seconds", timeout)
            return {"process_status": "timeout", "stdout": "", "stderr": "Client timed out"}

        except aiohttp.ClientError as e:
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
        try:
            response = await request(
                "GET",
                f"http://{self.host}:{self.port}/health",
                timeout=aiohttp.ClientTimeout(total=timeout),
            )
            return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False
