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

"""The gate-authoritative staging core (token-in / token-out custody).

This subpackage is the dependency-free integration SDK an RL framework and an
inference backend build against: wire records (``records``), the framework
protocols (``protocols``), and the staging digest (``digest``); the pure
per-rollout lineage state machine (``lineage``), the finalizer's terminal-aware
linearize (``rebuild``), and the installable conformance kit (``conformance``)
join it in follow-up commits.

Purity rule (§ 3.0 of the gate-authoritative design): every module in this
subpackage imports with **no serving dependencies** -- no fastapi, no ray, no
torch, no TransferQueue, no aiohttp -- so it is importable inside any
framework's worker process. Enforced by a subprocess-import test that walks
this subpackage. Engine adapters (``token_id_capture/adapters/``) and gate
hosting are deliberately outside it.

It layers on the base capture core (``nemo_gym.token_id_capture``): the base
owns capture middleware, lineage resolution (``LineageIndex``), prefix supply,
and the trajectory builder; ``staging`` adds the training-grade custody wire --
worker-to-storage delta staging and the gate's receipt accounting. Its
protocols are named ``StagingSink``/``StagingSource`` to stay distinct from the
base's entry-based ``TokenSink``/``TokenSource``.
"""

from nemo_gym.token_id_capture.staging.protocols import (
    CaptureAdapter,
    StagingSink,
    StagingSource,
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
    "StagingSink",
    "StagingSource",
    "WeightVersionProvider",
    "install_capture",
    "staging_key",
]
