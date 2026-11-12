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

This package is token staging's public API.
Code outside ``nemo_gym/`` imports staging names from ``nemo_gym.models.token_capture.staging``.
"""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.models.token_capture.staging.capture import (
        ActiveCall,
        CaptureError,
        CaptureHost,
        RolloutTokenCapture,
        StreamingUnsupportedError,
    )
    from nemo_gym.models.token_capture.staging.conformance import assert_golden_vectors, load_golden_vectors
    from nemo_gym.models.token_capture.staging.digest import (
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
    from nemo_gym.models.token_capture.staging.protocols import (
        CaptureAdapter,
        StagingSink,
        StagingSource,
        WeightVersionProvider,
        install_capture,
    )
    from nemo_gym.models.token_capture.staging.rebuild import (
        ExtrasCommitment,
        LinearizedRow,
        RebuildError,
        ReceiptVerificationError,
        WeightVersionSpan,
        linearize,
        verify_and_linearize,
    )
    from nemo_gym.models.token_capture.staging.records import (
        INVALID_COMMIT_COORDS_REASON,
        WORKER_MISSING_COMMIT_COORDS_REASON,
        CallRecord,
        CaptureAdmission,
        CaptureDisposition,
        CaptureLedgerCommit,
        CaptureMode,
        CommitCoords,
        GenerationCutContinuation,
        RolloutManifest,
        RolloutReceipt,
        StagedCallBaseSnapshot,
        StagedCallRecord,
        StagedCallSnapshot,
        StageResult,
        staging_key,
    )
    from nemo_gym.models.token_capture.staging.routes import (
        MISSING_ROUTE_SENTINEL,
        RoutedExpertsFragment,
        RouteSpanMode,
        classify_route_span,
        decode_routed_experts,
        encode_routed_experts,
        routed_experts_token_count,
    )
    from nemo_gym.models.token_capture.staging.terminal import TerminalSelection, select_terminal_call


# What code outside the library, such as servers, imports from this package.
# Each name's submodule loads on first use.
# Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "ActiveCall": "capture",
    "CallRecord": "records",
    "CaptureAdapter": "protocols",
    "CaptureAdmission": "records",
    "CaptureDisposition": "records",
    "CaptureError": "capture",
    "CaptureHost": "capture",
    "CaptureLedgerCommit": "records",
    "CaptureMode": "records",
    "CommitCoords": "records",
    "EMPTY_EXTRAS_DIGEST": "digest",
    "EXTRAS_DIGEST_VERSION": "digest",
    "ExtrasCommitment": "rebuild",
    "GenerationCutContinuation": "records",
    "INVALID_COMMIT_COORDS_REASON": "records",
    "LinearizedRow": "rebuild",
    "MISSING_ROUTE_SENTINEL": "routes",
    "RebuildError": "rebuild",
    "ReceiptVerificationError": "rebuild",
    "RolloutManifest": "records",
    "RolloutReceipt": "records",
    "RolloutTokenCapture": "capture",
    "RouteSpanMode": "routes",
    "RoutedExpertsFragment": "routes",
    "STAGING_DIGEST_VERSION": "digest",
    "STAGING_SCHEMA_VERSION": "digest",
    "StageResult": "records",
    "StagedCallBaseSnapshot": "records",
    "StagedCallRecord": "records",
    "StagedCallSnapshot": "records",
    "StagingSink": "protocols",
    "StagingSource": "protocols",
    "StreamingUnsupportedError": "capture",
    "TerminalSelection": "terminal",
    "WORKER_MISSING_COMMIT_COORDS_REASON": "records",
    "WeightVersionProvider": "protocols",
    "WeightVersionSpan": "rebuild",
    "assert_golden_vectors": "conformance",
    "build_staging_delta": "digest",
    "classify_route_span": "routes",
    "compute_chain_hash": "digest",
    "compute_extras_digest": "digest",
    "compute_staging_digest": "digest",
    "decode_routed_experts": "routes",
    "encode_routed_experts": "routes",
    "encode_token_ids": "digest",
    "hash_token_ids": "digest",
    "install_capture": "protocols",
    "linearize": "rebuild",
    "load_golden_vectors": "conformance",
    "routed_experts_token_count": "routes",
    "select_terminal_call": "terminal",
    "staging_key": "records",
    "verify_and_linearize": "rebuild",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
