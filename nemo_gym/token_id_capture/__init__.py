# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Training-token capture: produce, store, read, and source ``TokenEntry`` records.

This is the per-model-call training data path, kept separate from evaluation
capture. The capture middleware sets a per-request token sink; the model server
records a ``TokenEntry`` from its complete response; a trainer reads a rollout's
entries through a ``TokenSource`` and stitches them into a trajectory.

**This package is a leaf.** Importing it must not pull in fastapi, ray, uvicorn,
aiohttp, requests, or torch, because a training framework's inference worker
imports the record, the protocols, and the capture core to write into its own
data plane (see ``protocols.py``). The HTTP read route and the HTTP reader do
need Gym's server stack, so they are deliberately *not* re-exported here --
import ``nemo_gym.token_id_capture.routes`` / ``.reader`` directly from server
code.
"""

from nemo_gym.token_id_capture.builder import (
    BuildOutput,
    Chain,
    Trajectory,
    assert_nemo_rl_contiguity,
    build_trajectories,
    classify_side_calls,
    per_request,
    prefix_merging,
    project_main_chain_response,
)
from nemo_gym.token_id_capture.config import TokenIdCaptureConfig
from nemo_gym.token_id_capture.consumer import (
    clear_token_captures_for_rollouts,
    token_id_capture_dirs_from_config,
    trajectories_for_rollout,
    trajectories_from_source,
)
from nemo_gym.token_id_capture.lineage import (
    LineageIndex,
    LineageNode,
    RolloutLineage,
    assistant_fingerprint,
    canonicalize_tool_arguments,
)
from nemo_gym.token_id_capture.protocols import (
    TokenSink,
    TokenSource,
    install_token_sink,
    installed_token_sink,
)
from nemo_gym.token_id_capture.records import (
    TOKEN_FIELDS,
    TokenEntry,
    compute_digest,
    cumulative_tokens,
    encode_token_ids,
    extract_token_fields,
    stamp_lineage,
)
from nemo_gym.token_id_capture.sink import (
    CaptureContext,
    TokenCaptureContext,
    capture_tokens,
    current_capture_context,
    lineage_index,
    reset_token_sink,
    set_token_sink,
)
from nemo_gym.token_id_capture.source import CaptureTokenSource
from nemo_gym.token_id_capture.store import TokenCaptureStore, validate_rollout_id


__all__ = [
    "TokenIdCaptureConfig",
    "TokenEntry",
    "TOKEN_FIELDS",
    "extract_token_fields",
    "compute_digest",
    "encode_token_ids",
    "cumulative_tokens",
    "stamp_lineage",
    "TokenCaptureStore",
    "validate_rollout_id",
    "TokenSink",
    "TokenSource",
    "install_token_sink",
    "installed_token_sink",
    "CaptureContext",
    "TokenCaptureContext",
    "set_token_sink",
    "reset_token_sink",
    "capture_tokens",
    "current_capture_context",
    "lineage_index",
    "CaptureTokenSource",
    "LineageIndex",
    "LineageNode",
    "RolloutLineage",
    "assistant_fingerprint",
    "canonicalize_tool_arguments",
    "build_trajectories",
    "classify_side_calls",
    "per_request",
    "prefix_merging",
    "project_main_chain_response",
    "assert_nemo_rl_contiguity",
    "Trajectory",
    "Chain",
    "BuildOutput",
    "trajectories_for_rollout",
    "clear_token_captures_for_rollouts",
    "trajectories_from_source",
    "token_id_capture_dirs_from_config",
]
