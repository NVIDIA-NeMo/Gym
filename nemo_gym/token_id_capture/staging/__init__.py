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

"""The gate-authoritative capture core (token-in / token-out staging).

This subpackage is the dependency-free integration SDK an RL framework and an
inference backend build against: wire records (``records``), the framework
protocols (``protocols``), the staging digest (``digest``), the pure
per-rollout lineage state machine (``lineage``), the finalizer's rebuild /
linearize semantics (``rebuild``), and the installable conformance kit
(``conformance``).

Purity rule (§ 3.0 of the gate-authoritative design): every module in this
subpackage imports with **no serving dependencies** -- no fastapi, no ray, no
torch, no TransferQueue, no aiohttp -- so it is importable inside any
framework's worker process. Enforced by a subprocess-import test that walks
this subpackage. Engine adapters (``token_id_capture/adapters/``) and gate
hosting are deliberately outside it.

It is separated from the flat ``token_id_capture`` package (#2124's capture
core: ``TokenEntry`` store/sink/reader/routes) so the boundary between the
upstream base and the gate-authoritative layer is structural; the
``TokenSink``/``TokenSource`` protocols here are distinct from #2124's
same-named exports and must be imported from this subpackage.
"""

from nemo_gym.token_id_capture.staging.protocols import (
    CaptureAdapter,
    TokenSink,
    TokenSource,
    WeightVersionProvider,
    install_capture,
)
from nemo_gym.token_id_capture.staging.records import (
    SCHEMA_VERSION,
    CallRecord,
    CaptureDisposition,
    CaptureMode,
    CommitCoords,
    RolloutReceipt,
    StagedCallRecord,
    StagedCallSnapshot,
    StageResult,
    staging_key,
)


__all__ = [
    "SCHEMA_VERSION",
    "CallRecord",
    "CaptureAdapter",
    "CaptureDisposition",
    "CaptureMode",
    "CommitCoords",
    "RolloutReceipt",
    "StagedCallRecord",
    "StagedCallSnapshot",
    "StageResult",
    "TokenSink",
    "TokenSource",
    "WeightVersionProvider",
    "install_capture",
    "staging_key",
]
