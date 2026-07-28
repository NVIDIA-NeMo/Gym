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

"""The protocols an RL framework implements to host gate-authoritative capture.

Gym owns the invariant -- lineage custody, the wire schema (``records.py``),
the digest, and rebuild semantics. A framework integrates by providing four
small implementations against this module:

* ``TokenSink`` -- WHERE staged deltas go (NeMo-RL: a TransferQueue partition).
* ``TokenSource`` -- the finalizer's read-back of staged rows by key.
* ``WeightVersionProvider`` -- the trainer-owned weight version for per-call
  tagging.
* one ``install_capture(...)`` call at worker startup.

Staging keys are opaque to Gym: TQ keys, file paths, and redis keys are all
valid. Conformance is tested, not trusted -- every framework implementation
runs the golden fixtures in ``conformance/``.

Namespacing note: this module's ``TokenSink``/``TokenSource`` are the staging
protocols of the gate-authoritative design. The #2124 capture core exports a
file-store ``TokenSink`` dataclass and an async ``TokenSource`` protocol under
the same names from ``nemo_gym.token_id_capture``; these live only under the
``staging`` subpackage, so the collision is resolved by namespace: import them
as ``from nemo_gym.token_id_capture.staging import TokenSink``.

This module is part of the dependency-free capture core: stdlib + pydantic
(via ``records``) only.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from nemo_gym.token_id_capture.staging.records import StagedCallRecord, StagedCallSnapshot, StageResult


@runtime_checkable
class TokenSink(Protocol):
    """Durable staging for per-call token deltas (the only heavy hop).

    ``stage`` must make the record's bytes durable before returning
    ``StageResult(ok=True)`` -- the capture layer acks the model call (and so
    releases the child-enabling marker) only after a successful stage.
    Implementations must be safe to call concurrently from worker threads.
    """

    def stage(self, record: StagedCallRecord) -> StageResult: ...


@runtime_checkable
class TokenSource(Protocol):
    """The finalizer's read-back of staged rows.

    ``fetch`` returns one snapshot per staging key, in the order requested.
    A missing or unreadable row must raise ``KeyError`` (the finalizer maps
    that to a placeholder), never silently skip.
    """

    def fetch(self, staging_keys: list[str]) -> list[StagedCallSnapshot]: ...


@runtime_checkable
class WeightVersionProvider(Protocol):
    """Trainer state: the weight version to tag the next model call with."""

    def __call__(self) -> int: ...


class CaptureAdapter(Protocol):
    """Engine-specific glue an inference backend contributes (one module per
    engine; ``adapters/vllm.py`` first). The engine-blind capture core drives
    these hooks; everything else -- record build, digest, fail-closed
    stage-then-respond ordering -- is shared.

    S1 freezes the seam; the vLLM implementation lands in S2.
    """

    def enter_prefix(self, request_payload: dict[str, Any], prefix_ids: list[int]) -> dict[str, Any]:
        """Attach exact prefix ids to an engine request (token-in mode)."""
        ...

    def extract_generation(self, response_payload: dict[str, Any]) -> tuple[list[int], list[float]]:
        """Return (generated token ids, per-token logprobs) natively -- no
        string parsing, no second /tokenize round trip."""
        ...

    def extract_prompt_ids(self, response_payload: dict[str, Any]) -> list[int]:
        """Return the exact prompt ids the engine ran on."""
        ...


def install_capture(
    serving_layer: Any,
    *,
    sink: TokenSink,
    weight_version_fn: WeightVersionProvider,
    adapter: Optional[CaptureAdapter] = None,
) -> None:
    """Wire gate-authoritative capture into a worker's serving layer.

    The single call a framework makes at worker startup. The working
    implementation lands with the engine-blind capture core in S2
    (``capture.py``); the signature is frozen at the S1 gate so framework
    wiring can be written against it.
    """
    raise NotImplementedError("install_capture is wired by the S2 capture core (nemo_gym.token_id_capture.capture)")
