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
"""Opt-in routing transport: finite requests, no hidden generation retries."""

from aiohttp import ClientTimeout
from pydantic import Field

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.server_utils import request


class RoutedVLLMClient(NeMoGymAsyncOpenAI):
    inference_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)

    async def _request_with_retry(self, **request_kwargs):
        # One attempt at this layer, including 429 and connection errors. The
        # returned response/body shares this aiohttp total deadline.
        request_kwargs["_max_connection_retries"] = 0
        request_kwargs["_retry"] = False
        request_kwargs["timeout"] = ClientTimeout(total=self.inference_timeout_seconds)
        return await request(**request_kwargs)
