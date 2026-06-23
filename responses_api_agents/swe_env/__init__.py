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
working container; the separate ``resources_servers/swe_env`` verifier imports
the harness recipes and grading to score a patch in a fresh sandbox.
"""

from responses_api_agents.swe_env.environment import AsyncSweEnvironment
from responses_api_agents.swe_env.grading import compute_resolved, reward_from_report
from responses_api_agents.swe_env.harness import (
    EvalArtifacts,
    SweEvalReport,
    SweTask,
    SweTaskHarness,
)
from responses_api_agents.swe_env.registry import (
    get_harness,
    list_harnesses,
    register_harness,
)


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
