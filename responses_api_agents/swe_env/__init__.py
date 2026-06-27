# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provider-neutral SWE environment library.

Decouples SWE environment infrastructure (sandbox provisioning, exec, and
verification recipes) from agent harnesses. Built entirely on
``nemo_gym.sandbox``. Any agent imports this to provision and drive its own
working container and to grade a patch inline via ``verify_task`` (the harness
recipes plus grading score the patch in a fresh sandbox) — no separate verifier
server is required.
"""

from responses_api_agents.swe_env.harness import (
    EvalArtifacts,
    SweEvalReport,
    SweTask,
    SweTaskHarness,
    compute_resolved,
    get_harness,
    list_harnesses,
    register_harness,
    reward_from_report,
)
from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment


__all__ = [
    "AsyncSweEnvironment",
    "EvalArtifacts",
    "SweEvalReport",
    "SweTask",
    "SweTaskHarness",
    "compute_resolved",
    "reward_from_report",
    "get_harness",
    "list_harnesses",
    "register_harness",
]
