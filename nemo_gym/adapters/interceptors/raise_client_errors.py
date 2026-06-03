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
from typing import Any

from nemo_gym.adapters.types import AdapterResponse, ResponseInterceptor


logger = logging.getLogger(__name__)

_RETRIABLE = {408, 429}


class Interceptor(ResponseInterceptor):
    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        if 400 <= resp.status_code < 500 and resp.status_code not in _RETRIABLE:
            body: Any = resp.body
            detail = body if isinstance(body, (str, bytes)) else body.get("error", body)
            logger.error(
                "client error %d: %s",
                resp.status_code,
                detail,
            )
            raise RuntimeError(f"Upstream returned {resp.status_code}: {detail}")
        return resp
