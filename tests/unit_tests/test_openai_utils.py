# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI, NeMoGymResponseCreateParamsNonStreaming


class TestOpenAIUtils:
    async def test_NeMoGymAsyncOpenAI(self) -> None:
        NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")

    def test_semaphore_disabled_by_default(self) -> None:
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")
        assert client.max_concurrent_requests is None
        assert client._get_semaphore() is None

    @pytest.mark.asyncio
    async def test_semaphore_caps_concurrency(self) -> None:
        client = NeMoGymAsyncOpenAI(
            api_key="abc",
            base_url="https://api.openai.com/v1",
            max_concurrent_requests=2,
        )
        sem = client._get_semaphore()
        assert sem is not None
        # Same instance returned across calls (lazy init only fires once).
        assert client._get_semaphore() is sem

        in_flight = 0
        peak = 0

        async def worker() -> None:
            nonlocal in_flight, peak
            async with sem:
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1

        await asyncio.gather(*(worker() for _ in range(8)))
        assert peak == 2


class TestNeMoGymResponseCreateParamsNonStreaming:
    def test_seed_rejected_at_top_level(self) -> None:
        """seed is not part of the OpenAI Responses schema; it must be passed via metadata.extra_body."""
        with pytest.raises(ValidationError):
            NeMoGymResponseCreateParamsNonStreaming(input="hello", seed=42)

    def test_seed_via_metadata_extra_body(self) -> None:
        """seed passed through metadata.extra_body round-trips through the strict schema."""
        params = NeMoGymResponseCreateParamsNonStreaming(input="hello", metadata={"extra_body": '{"seed": 42}'})
        assert params.metadata["extra_body"] == '{"seed": 42}'

    def test_unknown_field_still_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            NeMoGymResponseCreateParamsNonStreaming(input="hello", not_a_real_field=1)
