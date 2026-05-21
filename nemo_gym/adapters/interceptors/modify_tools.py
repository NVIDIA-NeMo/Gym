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

import copy
import logging
from typing import Any

from nemo_gym.adapters.types import AdapterRequest, RequestInterceptor


logger = logging.getLogger(__name__)


def _apply_modifications(
    tools: list[dict[str, Any]],
    strip: frozenset[str],
    add: dict[str, dict[str, Any]],
) -> int:
    count = 0
    for tool in tools:
        fn = tool.get("function") or tool
        params = fn.get("parameters") or {}
        props = params.get("properties")
        if not isinstance(props, dict):
            continue

        req: list[str] | None = params.get("required")

        for field in strip:
            if field in props:
                del props[field]
                count += 1
            if isinstance(req, list) and field in req:
                req.remove(field)

        for field, schema in add.items():
            if field not in props:
                props[field] = copy.deepcopy(schema)
                count += 1
    return count


class Interceptor(RequestInterceptor):
    def __init__(
        self,
        *,
        strip_properties: list[str] | None = None,
        add_properties: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._strip = frozenset(strip_properties or [])
        self._add: dict[str, dict[str, Any]] = add_properties or {}
        self._logged_once = False

        if not self._strip and not self._add:
            logger.warning("modify_tools: no modifications configured — interceptor is a no-op")

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        tools = req.body.get("tools")
        if tools:
            n = _apply_modifications(tools, self._strip, self._add)
            if n and not self._logged_once:
                logger.info(
                    "modify_tools: applied %d change(s) (strip=%s, add=%s)",
                    n,
                    sorted(self._strip),
                    sorted(self._add),
                )
                self._logged_once = True
        return req
