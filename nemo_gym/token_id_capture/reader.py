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

"""Readers that return a rollout's ``TokenEntry`` records.

Two readers implement the same async ``read`` method:

* ``LocalTokenReader`` reads the store's files directly. Use it when the reader
  runs in the same process (or box) as the model server that wrote them.
* ``HttpTokenReader`` reads over the model server's ``/ng-capture/tokens``
  route. Use it when the trainer is not co-located with the store -- the common
  case once serving and training run on different nodes. It goes through Gym's
  shared aiohttp client, so reading at high rollout concurrency never stalls the
  event loop.
"""

from __future__ import annotations

from typing import Protocol

from nemo_gym.server_utils import raise_for_status, request
from nemo_gym.token_id_capture.records import TokenEntry
from nemo_gym.token_id_capture.store import TokenCaptureStore


class TokenReader(Protocol):
    async def read(self, rollout_id: str) -> list[TokenEntry]: ...


class LocalTokenReader:
    def __init__(self, store: TokenCaptureStore) -> None:
        self._store = store

    async def read(self, rollout_id: str) -> list[TokenEntry]:
        return self._store.read_entries(rollout_id)


class HttpTokenReader:
    def __init__(self, base_url: str, api_key: str = "dummy_key") -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def read(self, rollout_id: str) -> list[TokenEntry]:
        # Timeouts and retries are governed by Gym's shared aiohttp client.
        url = f"{self._base_url}/ng-capture/tokens/{rollout_id}"
        response = await request("GET", url, headers=self._headers)
        await raise_for_status(response)
        text = await response.text()
        return [TokenEntry.model_validate_json(line) for line in text.splitlines() if line.strip()]
