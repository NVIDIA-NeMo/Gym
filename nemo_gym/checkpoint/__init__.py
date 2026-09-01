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
"""

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


__all__ = [
    "CONTROL_SCHEMA_VERSION",
    "CONTROL_URL_PREFIX",
    "AdmissionState",
    "CheckpointConflictError",
    "CheckpointControlRequest",
    "CheckpointPhase",
    "ControlCapabilities",
    "ControlError",
    "ControlFence",
    "Deadline",
    "InvalidPhaseError",
    "MultiProcessCapability",
    "StaleCheckpointError",
    "install_control_plane",
    "multi_process_capability_from_num_workers",
]
