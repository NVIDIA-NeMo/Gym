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
from pytest import approx

from nemo_gym.judge import judge_failure_metrics, run_judge


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


class TestJudgeFailureMetrics:
    def test_counts_failures_and_excludes_from_reward(self) -> None:
        tasks = [
            [{"reward": 1.0}, {"reward": 0.0, "judge_failed": True}],
            [{"reward": 1.0}],
        ]
        m = judge_failure_metrics(tasks)
        assert m["judge_failures"] == 1
        # Judge-ok-only mean excludes the failed sample: (1.0 + 1.0) / 2.
        assert m["reward[judge_ok_only]"] == approx(1.0)

    def test_all_failed_returns_none(self) -> None:
        tasks = [[{"reward": 0.0, "judge_failed": True}]]
        m = judge_failure_metrics(tasks)
        assert m["judge_failures"] == 1
        assert m["reward[judge_ok_only]"] is None

    def test_empty_tasks_returns_empty(self) -> None:
        assert judge_failure_metrics([]) == {}
