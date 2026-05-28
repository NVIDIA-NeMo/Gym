# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import logging

from nemo_gym.adapters.types import AdapterRequest, RequestInterceptor


logger = logging.getLogger(__name__)

_VALID_STRATEGIES = {"replace", "append", "prepend"}


class Interceptor(RequestInterceptor):
    def __init__(
        self,
        *,
        system_message: str,
        strategy: str = "prepend",
    ) -> None:
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy {strategy!r}, must be one of {_VALID_STRATEGIES}")
        self._message = system_message
        self._strategy = strategy

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        messages: list = req.body.setdefault("messages", [])
        sys_msg = {"role": "system", "content": self._message}

        if self._strategy == "replace":
            non_system = [m for m in messages if m.get("role") != "system"]
            req.body["messages"] = [sys_msg] + non_system

        elif self._strategy == "append":
            messages.append(sys_msg)

        else:  # prepend
            messages.insert(0, sys_msg)

        return req
