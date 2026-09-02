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

"""Framework interfaces for the dependency-free staging contract.

These synchronous interfaces are intentional: the first consumer is an
inference worker whose capture hook is synchronous. Implementations must not
hide an asynchronous network client behind these methods; a serving host must
move a blocking implementation off its event loop explicitly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from nemo_gym.token_id_capture.staging.records import StagedCallBaseSnapshot, StagedCallRecord, StageResult


@runtime_checkable
class StagingSink(Protocol):
    """Make one normalized staged call durable before returning success."""

    def stage(self, record: StagedCallRecord) -> StageResult: ...


@runtime_checkable
class StagingSource(Protocol):
    """Fetch exactly one base snapshot per requested opaque staging key, in order.

    Sources return metadata-only base rows; extras payloads stay behind the
    staging key for consumers to fetch and digest-verify at point of use.
    """

    def fetch(self, staging_keys: list[str]) -> list[StagedCallBaseSnapshot]: ...


@runtime_checkable
class WeightVersionProvider(Protocol):
    """Return the trainer-owned weight version stamped at call admission."""

    def __call__(self) -> int: ...


class CaptureAdapter(Protocol):
    """Extract engine-native capture material without owning its lifecycle."""

    def enter_prefix(self, request_payload: dict[str, Any], prefix_ids: list[int]) -> dict[str, Any]:
        """Attach an exact parent prefix to an engine request."""
        ...

    def extract_prompt_ids(self, response_payload: dict[str, Any]) -> list[int]:
        """Return the exact prompt token IDs used for generation."""
        ...

    def extract_generation(self, response_payload: dict[str, Any]) -> tuple[list[int], list[float]]:
        """Return exact generated token IDs and selected-token log probabilities."""
        ...

    def extract_extras(self, response_payload: dict[str, Any]) -> dict[str, Any] | None:
        """Return optional versioned engine-native per-token material."""
        ...


def install_capture(
    serving_layer: Any,
    *,
    sink: StagingSink,
    weight_version_fn: WeightVersionProvider,
    adapter: CaptureAdapter | None = None,
) -> Any:
    """Install worker-owned staging without importing serving dependencies."""
    # Deferred to avoid the protocol/capture implementation import cycle.
    from nemo_gym.token_id_capture.staging.capture import install_capture as _install

    return _install(
        serving_layer,
        sink=sink,
        weight_version_fn=weight_version_fn,
        adapter=adapter,
    )
