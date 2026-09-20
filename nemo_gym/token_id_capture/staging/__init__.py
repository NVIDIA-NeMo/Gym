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

"""Portable wire and integrity contracts for framework-owned token staging.

This namespace is separate from Gym's complete-record ``TokenSink`` and
``TokenSource`` APIs. It carries per-call deltas between inference workers and
framework storage without importing serving or framework dependencies.
"""

from nemo_gym.token_id_capture.staging.capture import (
    ActiveCall,
    CaptureError,
    CaptureHost,
    RolloutTokenCapture,
    StreamingUnsupportedError,
)
from nemo_gym.token_id_capture.staging.conformance import assert_golden_vectors, load_golden_vectors
from nemo_gym.token_id_capture.staging.digest import (
    EMPTY_EXTRAS_DIGEST,
    EXTRAS_DIGEST_VERSION,
    STAGING_DIGEST_VERSION,
    STAGING_SCHEMA_VERSION,
    build_staging_delta,
    compute_chain_hash,
    compute_extras_digest,
    compute_staging_digest,
    encode_token_ids,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.protocols import (
    CaptureAdapter,
    StagingSink,
    StagingSource,
    WeightVersionProvider,
    install_capture,
)
from nemo_gym.token_id_capture.staging.rebuild import (
    ExtrasCommitment,
    LinearizedRow,
    RebuildError,
    ReceiptVerificationError,
    WeightVersionSpan,
    linearize,
    verify_and_linearize,
)
from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    CaptureAdmission,
    CaptureDisposition,
    CaptureLedgerCommit,
    CaptureMode,
    CommitCoords,
    RolloutReceipt,
    StagedCallBaseSnapshot,
    StagedCallRecord,
    StagedCallSnapshot,
    StageResult,
    staging_key,
)
from nemo_gym.token_id_capture.staging.routes import (
    MISSING_ROUTE_SENTINEL,
    RoutedExpertsFragment,
    RouteSpanMode,
    classify_route_span,
    decode_routed_experts,
    encode_routed_experts,
    routed_experts_token_count,
)
from nemo_gym.token_id_capture.staging.terminal import TerminalSelection, select_terminal_call

# Terminal selection is shared with the existing capture path.
# Rows with a ``staging_key`` use their recorded response and content fingerprints.
# A supplied ``declared_response_id`` must match exactly one row.
# Terminal selection does not read staged tokens.
# ``verify_and_linearize`` validates the selected token chain afterward.
from nemo_gym.token_id_capture.terminal import (
    TerminalAttribution,
    resolve_terminal,
)


__all__ = [
    "EMPTY_EXTRAS_DIGEST",
    "EXTRAS_DIGEST_VERSION",
    "MISSING_ROUTE_SENTINEL",
    "STAGING_DIGEST_VERSION",
    "STAGING_SCHEMA_VERSION",
    "CallRecord",
    "CaptureLedgerCommit",
    "ActiveCall",
    "CaptureAdapter",
    "CaptureError",
    "CaptureHost",
    "CaptureAdmission",
    "CaptureDisposition",
    "CaptureMode",
    "CommitCoords",
    "ExtrasCommitment",
    "LinearizedRow",
    "ReceiptVerificationError",
    "RebuildError",
    "RolloutReceipt",
    "RolloutTokenCapture",
    "RoutedExpertsFragment",
    "RouteSpanMode",
    "StagedCallBaseSnapshot",
    "StagedCallRecord",
    "StagedCallSnapshot",
    "StageResult",
    "StagingSink",
    "StagingSource",
    "StreamingUnsupportedError",
    "TerminalAttribution",
    "TerminalSelection",
    "WeightVersionProvider",
    "WeightVersionSpan",
    "assert_golden_vectors",
    "build_staging_delta",
    "classify_route_span",
    "compute_chain_hash",
    "compute_extras_digest",
    "compute_staging_digest",
    "decode_routed_experts",
    "encode_routed_experts",
    "encode_token_ids",
    "hash_token_ids",
    "install_capture",
    "linearize",
    "load_golden_vectors",
    "routed_experts_token_count",
    "resolve_terminal",
    "select_terminal_call",
    "staging_key",
    "verify_and_linearize",
]
