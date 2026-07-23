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
entries through a ``TokenSource``.
"""

from nemo_gym.token_id_capture.builder import (
    BuildOutput,
    Chain,
    Trajectory,
    assert_nemo_rl_contiguity,
    build_trajectories,
    per_request,
    prefix_merging,
    project_main_chain_response,
)
from nemo_gym.token_id_capture.config import TokenIdCaptureConfig
from nemo_gym.token_id_capture.consumer import (
    token_id_capture_dirs_from_config,
    trajectories_for_rollout,
    trajectories_from_source,
)
from nemo_gym.token_id_capture.reader import HttpTokenReader, LocalTokenReader, TokenReader
from nemo_gym.token_id_capture.records import TOKEN_FIELDS, TokenEntry, extract_token_fields
from nemo_gym.token_id_capture.routes import install_token_capture_routes, make_token_store
from nemo_gym.token_id_capture.sink import TokenSink, capture_tokens, reset_token_sink, set_token_sink
from nemo_gym.token_id_capture.source import CaptureTokenSource, TokenSource
from nemo_gym.token_id_capture.store import TokenCaptureStore, validate_rollout_id


__all__ = [
    "TokenIdCaptureConfig",
    "TokenEntry",
    "TOKEN_FIELDS",
    "extract_token_fields",
    "TokenCaptureStore",
    "validate_rollout_id",
    "TokenSink",
    "set_token_sink",
    "reset_token_sink",
    "capture_tokens",
    "TokenReader",
    "LocalTokenReader",
    "HttpTokenReader",
    "TokenSource",
    "CaptureTokenSource",
    "make_token_store",
    "install_token_capture_routes",
    "build_trajectories",
    "per_request",
    "prefix_merging",
    "project_main_chain_response",
    "assert_nemo_rl_contiguity",
    "Trajectory",
    "Chain",
    "BuildOutput",
    "trajectories_for_rollout",
    "trajectories_from_source",
    "token_id_capture_dirs_from_config",
]
