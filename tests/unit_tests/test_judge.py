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
from pydantic import BaseModel, ConfigDict

from nemo_gym.judge import (
    JUDGE_FAILURE_CLASS,
    NG_FAILURE_CLASS_KEY,
    NG_JUDGE_ERROR_KEY,
    NG_JUDGE_FAILED_KEY,
    judge_failure,
    run_judge,
)


class _Resp(BaseModel):
    model_config = ConfigDict(extra="allow")

    reward: float = 1.0
    response: dict = {}


class TestRunJudge:
    @pytest.mark.asyncio
    async def test_success_returns_result_no_error(self) -> None:
        async def ok():
            return "verdict"

        result, error = await run_judge(ok())
        assert result == "verdict"
        assert error is None

    @pytest.mark.asyncio
    async def test_exception_recorded_verbatim(self) -> None:
        async def boom():
            raise RuntimeError("judge timeout")

        result, error = await run_judge(boom())
        assert result is None
        assert error == "RuntimeError: judge timeout"


class TestJudgeFailure:
    def test_stamps_routing_keys_and_zero_reward(self) -> None:
        out = judge_failure(_Resp(reward=1.0, response={"final": "answer"}), "RuntimeError: boom")
        data = out.model_dump()
        assert data["reward"] == 0.0
        assert data[NG_FAILURE_CLASS_KEY] == JUDGE_FAILURE_CLASS
        assert data[NG_JUDGE_FAILED_KEY] is True
        assert data[NG_JUDGE_ERROR_KEY] == "RuntimeError: boom"
        # The model's final output is carried for a later judge-only replay.
        assert data["response"] == {"final": "answer"}
        # Transient: never terminal, so resume re-dispatches it.
        assert "_ng_failure_terminal" not in data

    def test_empty_error_defaults_to_message(self) -> None:
        out = judge_failure(_Resp(), None)
        assert out.model_dump()[NG_JUDGE_ERROR_KEY] == "empty judge response"
