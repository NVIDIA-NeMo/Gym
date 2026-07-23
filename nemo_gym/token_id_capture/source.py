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

"""The seam a trajectory builder reads training tokens through.

``TokenSource`` is the only interface a trajectory builder needs: given a
rollout id, return its ``TokenEntry`` records. This decouples "where the tokens
came from" from "how the trajectory is assembled". ``CaptureTokenSource`` is the
implementation backed by Gym's capture store (through a local or HTTP reader).
Alternative sources -- e.g. records staged by a training framework's own
transport -- can implement the same protocol without changing the builder.
"""

from __future__ import annotations

from typing import Protocol

from nemo_gym.token_id_capture.reader import TokenReader
from nemo_gym.token_id_capture.records import TokenEntry


class TokenSource(Protocol):
    async def tokens_for(self, rollout_id: str) -> list[TokenEntry]: ...


class CaptureTokenSource:
    """A ``TokenSource`` backed by Gym's capture store via a ``TokenReader``."""

    def __init__(self, reader: TokenReader) -> None:
        self._reader = reader

    async def tokens_for(self, rollout_id: str) -> list[TokenEntry]:
        return await self._reader.read(rollout_id)
