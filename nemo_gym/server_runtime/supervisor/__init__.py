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
"""Starting the servers of a run and the virtual environments they run in.

This code runs in the process that launches servers, never inside a server, so nothing else in ``nemo_gym.server_runtime``
imports it. ``server_supervisor`` starts and stops the servers, ``server_venv`` builds their environments,
``_locked_env`` and ``_prepare_worker`` are the scripts those environments run, and ``h2_ping_sidecar`` runs the
HTTP/2 PING proxy for model endpoints with idle timeouts.
"""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.server_runtime.supervisor._locked_env import sync_environment
    from nemo_gym.server_runtime.supervisor.server_venv import get_venv_path, run_command, setup_env_command


# What code outside the library, such as servers and benchmarks, imports from this package. Each name's submodule
# loads on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "get_venv_path": "server_venv",
    "run_command": "server_venv",
    "setup_env_command": "server_venv",
    "sync_environment": "_locked_env",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
