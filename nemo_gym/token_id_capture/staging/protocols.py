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

from typing import Protocol, runtime_checkable

from nemo_gym.token_id_capture.staging.records import StagedCallRecord, StagedCallSnapshot, StageResult


@runtime_checkable
class StagingSink(Protocol):
    """Make one normalized staged call durable before returning success."""

    def stage(self, record: StagedCallRecord) -> StageResult: ...


@runtime_checkable
class StagingSource(Protocol):
    """Fetch exactly one snapshot per requested opaque staging key, in order."""

    def fetch(self, staging_keys: list[str]) -> list[StagedCallSnapshot]: ...


@runtime_checkable
class WeightVersionProvider(Protocol):
    """Return the trainer-owned weight version stamped at call admission."""

    def __call__(self) -> int: ...
