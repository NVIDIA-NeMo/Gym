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
"""Shared machine-readable failure kinds.

The constants are stable grouping keys for rollout and verification artifacts.
They do not decide whether a sample is masked or whether a run may continue.
Retryability, delivery state, and policy attribution belong to each failure record.
Environments may add namespaced values such as ``my_server:quota_exhausted``.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


RegistryStatus = Literal["active", "reserved"]


@dataclass(frozen=True)
class FailureKindRegistration:
    """Registration state and provenance for one shared failure kind."""

    status: RegistryStatus
    source: str


# Active kinds have producers on main.
JUDGE_FAILED = "judge_failed"
KILL_SHAPED = "kill_shaped"
TIMEOUT_EXCEEDED = "timeout_exceeded"
SKIPPED = "skipped"
TRANSIENT = "transient"
LEGITIMATE = "legitimate"
REMOTE_AGENT_ERROR = "remote_agent_error"
EVAL_TIMEOUT = "eval_timeout"
SANDBOX = "sandbox"
OOM_KILLED = "oom_killed"

# Reserved kinds belong to changes that have not landed.
SESSION_LOST = "session_lost"
AGENT_REQUEST_FAILED = "agent_request_failed"
VERIFY_REQUEST_FAILED = "verify_request_failed"
UNREACHABLE = "unreachable"
RESOURCE = "resource"
PEER_DROP = "peer_drop"
TIMEOUT = "timeout"
FATAL = "fatal"
COHORT_TIMEOUT = "cohort_timeout"
NO_COMPARISONS = "no_comparisons"
AGGREGATION_FAILED = "aggregation_failed"


FAILURE_KIND_REGISTRY: Mapping[str, FailureKindRegistration] = MappingProxyType(
    {
        JUDGE_FAILED: FailureKindRegistration(
            status="active",
            source="nemo_gym.judge",
        ),
        KILL_SHAPED: FailureKindRegistration(
            status="active",
            source="responses_api_agents.stirrup_agent",
        ),
        TIMEOUT_EXCEEDED: FailureKindRegistration(
            status="active",
            source="responses_api_agents.stirrup_agent",
        ),
        SKIPPED: FailureKindRegistration(
            status="active",
            source="responses_api_agents.stirrup_agent",
        ),
        TRANSIENT: FailureKindRegistration(
            status="active",
            source="responses_api_agents.stirrup_agent",
        ),
        LEGITIMATE: FailureKindRegistration(
            status="active",
            source="responses_api_agents.stirrup_agent",
        ),
        REMOTE_AGENT_ERROR: FailureKindRegistration(
            status="active",
            source="responses_api_agents.remote_agent",
        ),
        EVAL_TIMEOUT: FailureKindRegistration(
            status="active",
            source="responses_api_agents.anyswe_agent",
        ),
        SANDBOX: FailureKindRegistration(
            status="active",
            source="responses_api_agents.anyswe_agent",
        ),
        OOM_KILLED: FailureKindRegistration(
            status="active",
            source="responses_api_agents.swe_agents",
        ),
        SESSION_LOST: FailureKindRegistration(
            status="reserved",
            source="#2611",
        ),
        AGENT_REQUEST_FAILED: FailureKindRegistration(
            status="reserved",
            source="#2363",
        ),
        VERIFY_REQUEST_FAILED: FailureKindRegistration(
            status="reserved",
            source="#2363",
        ),
        UNREACHABLE: FailureKindRegistration(
            status="reserved",
            source="#2361",
        ),
        RESOURCE: FailureKindRegistration(
            status="reserved",
            source="#2361",
        ),
        PEER_DROP: FailureKindRegistration(
            status="reserved",
            source="#2361",
        ),
        TIMEOUT: FailureKindRegistration(
            status="reserved",
            source="#2361",
        ),
        FATAL: FailureKindRegistration(
            status="reserved",
            source="#2361",
        ),
        COHORT_TIMEOUT: FailureKindRegistration(
            status="reserved",
            source="#2385",
        ),
        NO_COMPARISONS: FailureKindRegistration(
            status="reserved",
            source="#2385",
        ),
        AGGREGATION_FAILED: FailureKindRegistration(
            status="reserved",
            source="#2385",
        ),
    }
)

ACTIVE_FAILURE_KINDS = frozenset(
    kind for kind, metadata in FAILURE_KIND_REGISTRY.items() if metadata.status == "active"
)
RESERVED_FAILURE_KINDS = frozenset(
    kind for kind, metadata in FAILURE_KIND_REGISTRY.items() if metadata.status == "reserved"
)
