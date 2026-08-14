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
"""Judge retry budget: the 2026-07-15 inference-api 529 outage exhausted the old
10-attempt / 30s-cap schedule (~2.9 min total) and falsely zeroed 25 samples in a
single full-400 run. The budget is now 15 attempts capped at 120s (~18 min total),
long enough to ride out a multi-minute overload window."""

from resources_servers.browsecomp_advanced_harness.app import (
    JUDGE_BACKOFF_CAP_S,
    TavilySearchResourcesServer,
    _judge_backoff_s,
)


class TestJudgeRetryBudget:
    def test_max_attempts_is_15(self) -> None:
        assert TavilySearchResourcesServer.JUDGE_MAX_ATTEMPTS == 15

    def test_backoff_cap_is_120s(self) -> None:
        assert JUDGE_BACKOFF_CAP_S == 120

    def test_backoff_schedule_exponential_then_capped(self) -> None:
        schedule = [_judge_backoff_s(a) for a in range(15)]
        assert schedule[:8] == [1, 2, 4, 8, 16, 32, 64, 120]
        assert all(s == 120 for s in schedule[7:])

    def test_total_window_rides_out_multi_minute_outage(self) -> None:
        total = sum(_judge_backoff_s(a) for a in range(15))
        assert total >= 15 * 60  # >= 15 minutes of cumulative retry window
