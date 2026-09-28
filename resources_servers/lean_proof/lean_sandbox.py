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

"""A Lean compile sandbox, shared by the Lean benchmarks.

Any Lean server needs the same thing: a sandbox holding a Mathlib build, and a way to compile
one file in it. That is all this is. The image comes from ``lean_image/`` -- one per Mathlib
version, because compiled oleans do not carry across versions.

One sandbox per server process, created on first use and reused. A sandbox per rollout is not
viable: pod allocation costs minutes and a run is thousands of rollouts. There is no server
shutdown hook, so ``sandbox_config.ttl_s`` is what reclaims it if the process dies.
"""

import asyncio
import logging
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

from pydantic import BaseModel

from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.utils import cpu_cap_env


logger = logging.getLogger(__name__)


class CompilerOutput(BaseModel):
    """What the Lean toolchain said, carried on a verify response for debugging a rollout.

    ``process_status`` holds this library's ``proof_status`` vocabulary.
    """

    process_status: str
    stdout: str
    stderr: str


# Where lean_image/ puts the prebuilt Mathlib project. `lake env lean` runs here so imports
# resolve against it.
DEFAULT_LEAN_PROJECT_DIR = "/opt/mathlib"

# OpenSandbox requires an entry process when creating from an image; the image's own CMD is
# `sleep infinity`, and this is the same thing made explicit for providers that need it.
DEFAULT_ENTRYPOINT = ["sleep", "infinity"]

# Reports the Lean version Mathlib was built against.
TOOLCHAIN_PROBE = "import Mathlib\n#eval Lean.versionString"


class LeanSandbox:
    """Lazily-started sandbox that compiles Lean files.

    ``sandbox_config`` takes ``image``, ``resources``, ``ttl_s``, ``provider_options`` and the
    rest, as in the shipped provider configs. ``server_name`` lands in sandbox metadata, so a
    stray sandbox is attributable.
    """

    def __init__(
        self,
        sandbox_provider: str,
        sandbox_config: Dict[str, Any],
        project_dir: str = DEFAULT_LEAN_PROJECT_DIR,
        server_name: str = "lean",
    ) -> None:
        self._provider = sandbox_provider
        self._config = dict(sandbox_config)
        self._project_dir = project_dir
        self._server_name = server_name
        self._sandbox: Optional[AsyncSandbox] = None
        self._lock = asyncio.Lock()

    async def start(self) -> AsyncSandbox:
        """Return the running sandbox, creating it on first call."""
        if self._sandbox is not None:
            return self._sandbox

        async with self._lock:
            if self._sandbox is not None:
                return self._sandbox

            global_config = get_global_config_dict()
            provider = resolve_provider_config(self._provider, global_config)
            default_metadata = resolve_provider_metadata(self._provider, global_config)

            resources = SandboxResources.from_mapping(self._config.get("resources", {}))
            env = dict(self._config.get("env", {}))
            if self._config.get("derive_cpu_env", True):
                # `lake` sizes its worker pool from the host core count, not this sandbox's
                # limit, on clusters without LXCFS. Explicit keys win.
                env = cpu_cap_env(resources.cpu) | env

            sandbox = AsyncSandbox(provider)
            await sandbox.start(
                SandboxSpec(
                    image=self._config.get("image"),
                    ttl_s=self._config.get("ttl_s"),
                    ready_timeout_s=self._config.get("ready_timeout_s"),
                    workdir=self._project_dir,
                    env=env,
                    metadata=default_metadata
                    | self._config.get("metadata", {})
                    | {"nemo_gym_agent": self._server_name},
                    resources=resources,
                    entrypoint=self._config.get("entrypoint", DEFAULT_ENTRYPOINT),
                    provider_options=self._config.get("provider_options", {}),
                )
            )
            self._sandbox = sandbox
            return sandbox

    async def compile(self, code: str, timeout_s: float) -> SandboxExecResult:
        """Compile one Lean file and return the raw exec result.

        Written through a heredoc rather than interpolated into the command, so quotes,
        backslashes and unicode in a proof need no escaping. One sandbox serves many concurrent
        verifies, hence the unique filename.
        """
        sandbox = await self.start()
        path = f"attempt_{uuid.uuid4().hex}.lean"
        delimiter = f"LEAN_EOF_{uuid.uuid4().hex}"
        command = (
            f"cat > {path} <<'{delimiter}'\n{code}\n{delimiter}\n"
            f"lake env lean {path}; status=$?; rm -f {path}; exit $status"
        )
        return await sandbox.exec(
            command,
            cwd=self._project_dir,
            # Let the sandbox, not the client, report the timeout.
            timeout_s=timeout_s + 30,
        )

    async def check_toolchain(
        self,
        expected: Optional[str],
        compile_fn: Optional[Callable[[str, float], Awaitable[Any]]] = None,
    ) -> Optional[str]:
        """Log an error unless the sandbox's Lean matches ``expected``. Returns what it found.

        A wrong Mathlib fails statements with ordinary compile errors, so the score looks
        plausible and is meaningless.
        """
        from resources_servers.lean_proof.toolchain import normalize_version, parse_lean_version

        # The probe must take the same path a real verify does, so callers that wrap
        # `compile` pass their wrapper.
        run = compile_fn or self.compile
        result = await run(TOOLCHAIN_PROBE, 600)
        found = parse_lean_version({"stdout": result.stdout or "", "stderr": result.stderr or ""})
        want = normalize_version(expected)

        if found is None:
            logger.error(
                "LEAN VERSION UNKNOWN: could not determine the sandbox's Lean version. If "
                "`import Mathlib` does not compile, every task fails for reasons unrelated to "
                "the model."
            )
        elif want and found != want:
            logger.error(
                "MATHLIB MISMATCH: sandbox is Lean %s but the rows are written against %s. "
                "Statements will fail with ordinary compile errors and the score will be "
                "meaningless but plausible.",
                found,
                want,
            )
        return found
