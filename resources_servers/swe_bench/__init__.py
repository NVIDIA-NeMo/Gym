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

"""SWE-bench Environment resources server modules.

Grading harnesses, parsing, and verify_task implement the Environment MDP
authority. Agent servers connect via HTTP ``seed_session`` / ``verify`` only.
"""

from resources_servers.swe_bench.harness import (
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
from resources_servers.swe_bench.sandbox import AsyncSweEnvironment


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
