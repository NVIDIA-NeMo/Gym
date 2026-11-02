# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run harness processes inside task sandboxes under a shared supervisor.

The harness adapter installs its runtime, stages activation input, and parses
harness output. SandboxSession uploads the supervisor, runs and stops the harness
process, collects artifacts, and releases the sandbox. The supervisor client
implements the shell protocol over session control files. All three run on the
agent server.

Only process_supervisor.sh and harness_runner.sh are copied from this package into
the task sandbox; they need bash, not Python. The supervisor enforces deadlines, stops
the harness process group and writes its termination receipt. A forced stop also kills the
harness's descendants in other groups and processes that inherited its marker; processes
that had left the harness before the stop, and detached processes after a natural exit,
remain for the sandbox owner to stop. Core sandbox APIs and providers live in
:mod:`nemo_gym.sandbox`.
"""

# Public imports for harness adapters; core modules import their defining submodules.
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
    from nemo_gym.agent_utils.task_timeout import resolve_agent_timeout
    from nemo_gym.agent_utils.uv_bootstrap import DEFAULT_UV_CACHE_DIR, uv_for_sandbox


_PUBLIC_API = {
    "DEFAULT_UV_CACHE_DIR": "uv_bootstrap",
    "SandboxCommand": "sandbox_session",
    "SandboxSession": "sandbox_session",
    "resolve_agent_timeout": "task_timeout",
    "uv_for_sandbox": "uv_bootstrap",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
