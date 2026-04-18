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
import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI, NeMoGymResponseCreateParamsNonStreaming


class TestOpenAIUtils:
    async def test_NeMoGymAsyncOpenAI(self) -> None:
        NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")


class TestNeMoGymResponseCreateParamsNonStreaming:
    def test_accepts_seed(self) -> None:
        """seed is a vLLM vendor extension on the Responses endpoint; the strict schema must accept it."""
        params = NeMoGymResponseCreateParamsNonStreaming(input="hello", seed=42)
        assert params.seed == 42

    def test_seed_round_trip_dump_and_validate(self) -> None:
        """Constructing with seed, dumping, and re-validating preserves the field."""
        params = NeMoGymResponseCreateParamsNonStreaming(input="hello", seed=7)
        dumped = params.model_dump(exclude_none=True)
        assert dumped["seed"] == 7

        reloaded = NeMoGymResponseCreateParamsNonStreaming.model_validate(dumped)
        assert reloaded.seed == 7
        assert reloaded.input == "hello"

    def test_seed_defaults_to_none(self) -> None:
        params = NeMoGymResponseCreateParamsNonStreaming(input="hello")
        assert params.seed is None

    def test_unknown_field_still_forbidden(self) -> None:
        """Adding seed must not weaken extra='forbid' for unrelated unknown fields."""
        with pytest.raises(ValidationError):
            NeMoGymResponseCreateParamsNonStreaming(input="hello", not_a_real_field=1)
