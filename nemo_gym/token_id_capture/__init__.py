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
data plane (see ``protocols.py``).

Records are read back through a ``TokenSource``. ``TokenCaptureStore`` is one,
and is what a reader sitting alongside the store uses. A framework staging
records through its own transport supplies its own source, which lives wherever
that transport does. What any source owes is an honest ``is_incomplete``: it is
how a consumer learns a rollout lost a call, and one that always answers False
trains on an incomplete rollout without knowing.
"""

from nemo_gym.token_id_capture.config import TokenIdCaptureConfig
from nemo_gym.token_id_capture.protocols import (
    TokenSink,
    TokenSource,
    install_token_sink,
    installed_token_sink,
)
from nemo_gym.token_id_capture.records import (
    TOKEN_ENTRY_RECORD_SCHEMA_VERSION,
    TOKEN_FIELDS,
    TokenEntry,
    extract_token_fields,
)
from nemo_gym.token_id_capture.sink import (
    CaptureContext,
    capture_tokens,
    commit_entry,
    reset_token_sink,
    set_token_sink,
)
from nemo_gym.token_id_capture.store import TokenCaptureStore, make_token_store, validate_rollout_id


__all__ = [
    "TokenIdCaptureConfig",
    "TokenEntry",
    "TOKEN_ENTRY_RECORD_SCHEMA_VERSION",
    "TOKEN_FIELDS",
    "extract_token_fields",
    "TokenCaptureStore",
    "validate_rollout_id",
    "make_token_store",
    "TokenSink",
    "TokenSource",
    "install_token_sink",
    "installed_token_sink",
    "CaptureContext",
    "set_token_sink",
    "reset_token_sink",
    "capture_tokens",
    "commit_entry",
]
