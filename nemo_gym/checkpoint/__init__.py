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
"""Checkpoint control plane shared by every Gym server.

Partial rollout checkpointing pauses, drains, commits, and restores Gym
servers in lockstep with the NeMo-RL training checkpoint. The pieces in this
package are the server-side mechanisms that make those control calls safe:

- ``control``: the ``/ng-control/v1`` capability declaration, checkpoint-id
  fencing, phase machine, and deadline plumbing every control route uses.
- ``admission``: the admission limiter that drains a server's data plane to
  a quiescent point and refuses work that can safely be re-issued.
- ``coordinator``: the service-level coordinator that closes every worker's
  limiter and aggregates worker acknowledgements and in-flight counts.
"""

from nemo_gym.checkpoint.admission import (
    GATED_MODEL_ROUTE_SUFFIXES,
    PLANE_HEADER,
    AdmissionLimiter,
    AdmissionMiddleware,
    AdmissionParkedError,
    AdmissionTicket,
    StaleAttemptError,
)
from nemo_gym.checkpoint.agent import (
    AGENT_CHECKPOINT_SCHEMA_VERSION,
    AGENT_CHECKPOINT_URL_PREFIX,
    AGENT_MANIFEST_NAME,
    AGENT_STATE_SUBDIR,
    AgentAdmissionClosedError,
    AgentBoundaryRecord,
    AgentCheckpointError,
    AgentCheckpointParticipant,
    AgentExecutionState,
    DuplicateExecutionError,
    commit_agent_state,
    install_agent_checkpoint,
    restore_agent_state,
)
from nemo_gym.checkpoint.control import (
    CONTROL_SCHEMA_VERSION,
    CONTROL_URL_PREFIX,
    AdmissionState,
    CheckpointConflictError,
    CheckpointControlRequest,
    CheckpointPhase,
    ControlCapabilities,
    ControlError,
    ControlFence,
    Deadline,
    InvalidPhaseError,
    MultiProcessCapability,
    StaleCheckpointError,
    install_control_plane,
    multi_process_capability_from_num_workers,
)
from nemo_gym.checkpoint.coordinator import (
    AdmissionCoordinator,
    MissingWorkersError,
    WorkerAdmissionAgent,
    build_coordinator_control_app,
)
from nemo_gym.checkpoint.ledger import (
    LEDGER_MANIFEST_NAME,
    MODEL_CHECKPOINT_URL_PREFIX,
    MODEL_LEDGER_SUBDIR,
    CaptureLedgerCheckpointer,
    CheckpointableCaptureLedger,
    LedgerMismatchError,
    LedgerNotCheckpointableError,
    LedgerNotQuiescentError,
    install_model_checkpoint,
)
from nemo_gym.checkpoint.model_admission import (
    NotPolicyInstanceError,
    install_model_admission,
)
from nemo_gym.checkpoint.model_control_contracts import (
    MODEL_ADMISSION_URL_PREFIX,
    ModelAbortInflightRequest,
    ModelAdmissionPauseRequest,
    ModelAdmissionResumeRequest,
)
from nemo_gym.checkpoint.resources import (
    RESOURCE_STATE_REVISION_HEADER,
    RESOURCES_CHECKPOINT_SCHEMA_VERSION,
    RESOURCES_CHECKPOINT_URL_PREFIX,
    RESOURCES_MANIFEST_NAME,
    RESOURCES_STATE_SUBDIR,
    ResourcesAdmissionClosedError,
    ResourcesCheckpointError,
    ResourcesCheckpointParticipant,
    ResourceSnapshot,
    ResourcesSessionMiddleware,
    commit_resources_state,
    install_resources_checkpoint,
    load_resources_state,
)


__all__ = [
    "AGENT_CHECKPOINT_SCHEMA_VERSION",
    "AGENT_CHECKPOINT_URL_PREFIX",
    "AGENT_MANIFEST_NAME",
    "AGENT_STATE_SUBDIR",
    "CONTROL_SCHEMA_VERSION",
    "CONTROL_URL_PREFIX",
    "GATED_MODEL_ROUTE_SUFFIXES",
    "LEDGER_MANIFEST_NAME",
    "MODEL_ADMISSION_URL_PREFIX",
    "MODEL_CHECKPOINT_URL_PREFIX",
    "MODEL_LEDGER_SUBDIR",
    "PLANE_HEADER",
    "RESOURCES_CHECKPOINT_SCHEMA_VERSION",
    "RESOURCES_CHECKPOINT_URL_PREFIX",
    "RESOURCES_MANIFEST_NAME",
    "RESOURCES_STATE_SUBDIR",
    "RESOURCE_STATE_REVISION_HEADER",
    "AdmissionLimiter",
    "AdmissionMiddleware",
    "AdmissionCoordinator",
    "AdmissionParkedError",
    "AdmissionState",
    "AdmissionTicket",
    "AgentAdmissionClosedError",
    "AgentBoundaryRecord",
    "AgentCheckpointError",
    "AgentCheckpointParticipant",
    "AgentExecutionState",
    "MissingWorkersError",
    "ModelAbortInflightRequest",
    "ModelAdmissionPauseRequest",
    "ModelAdmissionResumeRequest",
    "WorkerAdmissionAgent",
    "build_coordinator_control_app",
    "commit_agent_state",
    "commit_resources_state",
    "CheckpointConflictError",
    "CheckpointControlRequest",
    "CheckpointPhase",
    "ControlCapabilities",
    "ControlError",
    "ControlFence",
    "Deadline",
    "DuplicateExecutionError",
    "CaptureLedgerCheckpointer",
    "CheckpointableCaptureLedger",
    "InvalidPhaseError",
    "LedgerMismatchError",
    "LedgerNotCheckpointableError",
    "LedgerNotQuiescentError",
    "MultiProcessCapability",
    "NotPolicyInstanceError",
    "ResourceSnapshot",
    "ResourcesAdmissionClosedError",
    "ResourcesCheckpointError",
    "ResourcesCheckpointParticipant",
    "ResourcesSessionMiddleware",
    "StaleAttemptError",
    "StaleCheckpointError",
    "install_agent_checkpoint",
    "install_control_plane",
    "install_model_admission",
    "install_model_checkpoint",
    "install_resources_checkpoint",
    "load_resources_state",
    "multi_process_capability_from_num_workers",
    "restore_agent_state",
]
