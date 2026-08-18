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

"""Provide the core training-token capture interfaces.

Training capture is separate from evaluation capture.
Middleware sets a request-scoped token sink.
The model server records a ``TokenEntry`` from its complete response.
Consumers call ``TokenSource.freeze`` for an atomic snapshot.
The snapshot includes entries and incomplete state.
Its ``snapshot_id`` identifies the exact frozen state.
``TokenCaptureStore`` is Gym's local sink and source implementation.
Framework transports may provide their own sink and source.
There is no HTTP token reader.
This leaf package avoids imports from Gym's server stack.
The rollout-record finalizer needs Gym's server stack.
It is deliberately not re-exported here.
Import ``nemo_gym.token_id_capture.delivery`` from server-side code.
The incomplete state prevents training on a rollout that lost a model call.
Finalization freezes and rebuilds the rollout.
Finalization does not retire the snapshot.
The caller retires it only after durable handoff.
Retirement uses the frozen ``snapshot_id`` and version.
Failed or masked builds retain their capture evidence.
"""

from nemo_gym.token_id_capture.builder import (
    Chain,
    assert_prefix_contiguity,
    per_request,
    prefix_merging,
    project_chain_to_output_items,
    project_main_chain_response,
    run_builder,
)
from nemo_gym.token_id_capture.config import TokenIdCaptureConfig
from nemo_gym.token_id_capture.consumer import (
    clear_token_captures_for_rollouts,
    token_id_capture_dirs_from_config,
    trajectories_for_rollout,
    trajectories_from_source,
)
from nemo_gym.token_id_capture.lineage import (
    FileLineageStore,
    InMemoryLineageStore,
    LineageIndex,
    RolloutLineage,
    assistant_fingerprint,
    canonicalize_tool_arguments,
)
from nemo_gym.token_id_capture.protocols import (
    LineageMatch,
    LineageStore,
    TokenCaptureSnapshot,
    TokenSink,
    TokenSource,
    install_lineage_store,
    install_token_sink,
    install_token_source,
    installed_lineage_store,
    installed_token_sink,
    installed_token_source,
)
from nemo_gym.token_id_capture.records import (
    TOKEN_ENTRY_RECORD_SCHEMA_VERSION,
    TOKEN_FIELDS,
    TokenEntry,
    compute_digest,
    cumulative_tokens,
    extract_token_fields,
    stamp_lineage,
)
from nemo_gym.token_id_capture.sink import (
    CaptureContext,
    capture_tokens,
    commit_entry,
    current_capture_context,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
)
from nemo_gym.token_id_capture.store import TokenCaptureStore, make_token_store, validate_rollout_id


__all__ = [
    "Chain",
    "TokenIdCaptureConfig",
    "TokenEntry",
    "TOKEN_ENTRY_RECORD_SCHEMA_VERSION",
    "TOKEN_FIELDS",
    "extract_token_fields",
    "compute_digest",
    "cumulative_tokens",
    "stamp_lineage",
    "TokenCaptureStore",
    "validate_rollout_id",
    "make_token_store",
    "TokenSink",
    "TokenSource",
    "LineageMatch",
    "LineageStore",
    "install_lineage_store",
    "installed_lineage_store",
    "TokenCaptureSnapshot",
    "install_token_sink",
    "install_token_source",
    "installed_token_sink",
    "installed_token_source",
    "CaptureContext",
    "set_token_sink",
    "reset_token_sink",
    "resolve_parent",
    "capture_tokens",
    "commit_entry",
    "current_capture_context",
    "FileLineageStore",
    "InMemoryLineageStore",
    "LineageIndex",
    "RolloutLineage",
    "assistant_fingerprint",
    "per_request",
    "prefix_merging",
    "project_chain_to_output_items",
    "project_main_chain_response",
    "run_builder",
    "assert_prefix_contiguity",
    "trajectories_for_rollout",
    "clear_token_captures_for_rollouts",
    "trajectories_from_source",
    "token_id_capture_dirs_from_config",
]
