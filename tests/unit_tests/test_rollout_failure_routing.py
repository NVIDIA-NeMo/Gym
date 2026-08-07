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
from unittest.mock import MagicMock

from aiohttp import ClientResponseError

from nemo_gym.rollout_collection import (
    AGENT_REQUEST_FAILED_FAILURE_CLASS,
    NG_FAILURE_CLASS_KEY,
    NG_TERMINAL_KEY,
    _agent_request_failure_result,
    format_rollout_failure_summary,
    summarize_rollout_failures,
)


def _client_response_error(content: bytes = b"upstream exploded") -> ClientResponseError:
    error = ClientResponseError(MagicMock(), (), status=500, message="Internal Server Error")
    error.response_content = content
    return error


class TestAgentRequestFailureResult:
    def test_carries_the_routing_key_and_a_zero_reward(self) -> None:
        result = _agent_request_failure_result(MagicMock(status=500), _client_response_error())

        assert AGENT_REQUEST_FAILED_FAILURE_CLASS == result[NG_FAILURE_CLASS_KEY]
        assert 0.0 == result["reward"]
        assert "500" in result["error"]
        assert "upstream exploded" in result["error"]

    def test_is_not_terminal_so_resume_re_dispatches_it(self) -> None:
        """An agent unreachable for one attempt is usually reachable for the next."""
        result = _agent_request_failure_result(MagicMock(status=500), _client_response_error())

        assert NG_TERMINAL_KEY not in result

    def test_truncates_a_large_upstream_body(self) -> None:
        result = _agent_request_failure_result(MagicMock(status=500), _client_response_error(b"x" * 10_000))

        assert len(result["error"]) < 2_200

    def test_tolerates_a_missing_or_undecodable_body(self) -> None:
        error = ClientResponseError(MagicMock(), (), status=502, message="Bad Gateway")
        assert "502" in _agent_request_failure_result(MagicMock(status=502), error)["error"]

        assert (
            "502" in _agent_request_failure_result(MagicMock(status=502), _client_response_error(b"\xff\xfe"))["error"]
        )


class TestFailureSummary:
    def test_counts_by_class_and_ignores_successes(self) -> None:
        results = [
            {"reward": 1.0},
            {"reward": 0.0, NG_FAILURE_CLASS_KEY: AGENT_REQUEST_FAILED_FAILURE_CLASS},
            {"reward": 0.0, NG_FAILURE_CLASS_KEY: AGENT_REQUEST_FAILED_FAILURE_CLASS},
            {"reward": 0.0, NG_FAILURE_CLASS_KEY: "judge_failed"},
        ]

        assert {AGENT_REQUEST_FAILED_FAILURE_CLASS: 2, "judge_failed": 1} == summarize_rollout_failures(results)

    def test_no_failures_summarises_to_nothing(self) -> None:
        assert {} == summarize_rollout_failures([{"reward": 1.0}, {"reward": 0.0}])

    def test_summary_names_the_counts_the_surviving_total_and_the_sidecar(self) -> None:
        summary = format_rollout_failure_summary(
            {AGENT_REQUEST_FAILED_FAILURE_CLASS: 3}, num_results=10, failures_fpath="out_failures.jsonl"
        )

        assert "3 / 10" in summary
        assert AGENT_REQUEST_FAILED_FAILURE_CLASS in summary
        assert "out_failures.jsonl" in summary
        # The scores that get reported cover only the 7 that produced a result.
        assert "cover the 7" in summary

    def test_summary_counts_every_class_without_judging_any(self) -> None:
        """Reporting is not policy: no failure class is singled out or gated on here."""
        counts = summarize_rollout_failures(
            [
                {NG_FAILURE_CLASS_KEY: AGENT_REQUEST_FAILED_FAILURE_CLASS},
                {NG_FAILURE_CLASS_KEY: "judge_failed"},
                {NG_FAILURE_CLASS_KEY: "verify_failed"},
            ]
        )
        summary = format_rollout_failure_summary(counts, num_results=3, failures_fpath="out_failures.jsonl")

        for failure_class in (AGENT_REQUEST_FAILED_FAILURE_CLASS, "judge_failed", "verify_failed"):
            assert failure_class in summary
