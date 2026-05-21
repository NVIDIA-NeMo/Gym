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
import re

from nemo_gym.adapters.types import AdapterResponse, ResponseInterceptor


logger = logging.getLogger(__name__)

_THINK = re.compile(r"^\s*<think>(.*?)</think>", re.DOTALL)


class Interceptor(ResponseInterceptor):
    best_effort = True

    @staticmethod
    def _normalize_message(msg: dict) -> bool:
        if "reasoning_content" in msg:
            return False
        if "reasoning" in msg:
            msg["reasoning_content"] = msg.pop("reasoning")
            return True
        content = msg.get("content")
        if not isinstance(content, str):
            return False
        m = _THINK.match(content)
        if m:
            msg["reasoning_content"] = m.group(1).strip()
            msg["content"] = content[m.end() :].lstrip()
            return True
        return False

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        body = resp.body
        if not isinstance(body, dict):
            return resp
        changed = 0
        for ch in body.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message")
            if isinstance(msg, dict) and self._normalize_message(msg):
                changed += 1
        if changed:
            logger.debug("reasoning normalized choices=%d", changed)
        return resp
