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

"""Shared client for the Lean 4 compilation sandbox.

Talks to the NeMo-Skills sandbox (``POST /execute``). Shared by every Lean benchmark in the
repo so that a fix lands once rather than three times.

Deliberately **not** a copy of ``math_formal_lean``'s client: that one is built on
``httpx.AsyncClient``, which ``AGENTS.md`` rules out -- httpx/httpcore has O(n^2) connection
pooling that hangs at Gym's concurrency (16k+ requests, and a single Lean benchmark run is
already 12k). This goes through ``nemo_gym.server_utils.request``, Gym's shared aiohttp client.

The sandbox is pinned to one Mathlib per container: ``/execute`` takes no project parameter and
NeMo-Skills hardcodes the project path, so the Mathlib version is chosen when the container
starts, not per request. Callers should verify it -- see ``toolchain.py``.
"""

import asyncio
import json
import logging
from typing import Any, Dict

import aiohttp

from nemo_gym.server_utils import request


LOG = logging.getLogger(__name__)


class Lean4SandboxClient:
    """Compiles a complete Lean 4 file and reports what the compiler said."""

    def __init__(self, host: str = "127.0.0.1", port: int = 6000, max_output_characters: int = 4000):
        self.host = host
        self.port = port
        self.max_output_characters = max_output_characters

    @property
    def execute_url(self) -> str:
        return f"http://{self.host}:{self.port}/execute"

    async def execute_lean4(self, code: str, timeout: float = 300.0) -> Dict[str, Any]:
        """Compile ``code`` and return ``{process_status, stdout, stderr}``.

        Never raises: a sandbox that is down or wedged has to score as a failed proof with a
        diagnosable status, not take the whole rollout down with it.
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
                # Buffer past the compiler's own timeout so the sandbox, not the client, is
                # the one that reports a genuine compile timeout.
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
                "GET", f"http://{self.host}:{self.port}/health", timeout=aiohttp.ClientTimeout(total=timeout)
            )
            return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False
