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
"""The trajectory data model: the records of what happened in a rollout, and their ATIF export and import."""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.trajectory.atif_export import AtifExportError, encode_atif_trajectory, transcript_to_atif
    from nemo_gym.trajectory.trajectory_record import (
        NG_AGENT_OBSERVATIONS_KEY,
        NG_TRAJECTORY_KEY,
        NG_VERIFIER_INVOCATIONS_KEY,
        AgentEpisode,
        AgentInvocation,
        AgentObservationBundle,
        ContextCompactionObservation,
        ModelCallRef,
        ObservationGap,
        SandboxObservation,
        ToolCallObservation,
        TrajectoryRecord,
        TrajectoryToolCall,
        TrajectoryTurn,
        VerifierInvocation,
    )


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "AgentEpisode": "trajectory_record",
    "AgentInvocation": "trajectory_record",
    "AgentObservationBundle": "trajectory_record",
    "AtifExportError": "atif_export",
    "ContextCompactionObservation": "trajectory_record",
    "ModelCallRef": "trajectory_record",
    "NG_AGENT_OBSERVATIONS_KEY": "trajectory_record",
    "NG_TRAJECTORY_KEY": "trajectory_record",
    "NG_VERIFIER_INVOCATIONS_KEY": "trajectory_record",
    "ObservationGap": "trajectory_record",
    "SandboxObservation": "trajectory_record",
    "ToolCallObservation": "trajectory_record",
    "TrajectoryRecord": "trajectory_record",
    "TrajectoryToolCall": "trajectory_record",
    "TrajectoryTurn": "trajectory_record",
    "VerifierInvocation": "trajectory_record",
    "encode_atif_trajectory": "atif_export",
    "transcript_to_atif": "atif_export",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
