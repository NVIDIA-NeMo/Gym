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
"""The infra-retry cap note belongs to infra-cap decisions; an exhausted attempt cap says what raises it."""

from nemo_gym.orchestration.recovery import CollectedEvidence, RecoveryAction, decide_from_collected
from tests.unit_tests.orchestration.test_eval_recovery_gate import infra_failure, inventory


def _collected(rows: int, failures: list[dict], **fields) -> CollectedEvidence:
    return CollectedEvidence(materialized=inventory(rows), results=[], failures=failures, **fields)


def _decide(collected: CollectedEvidence, **options):
    return decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False, **options)


def test_an_exhausted_stop_names_what_extend_retries_does_not_raise_and_carries_no_cap_note() -> None:
    exhausted = _collected(1, [infra_failure(0, attempt=a) for a in range(3)])

    decision = _decide(exhausted, extend_retries=1)

    assert decision.action == RecoveryAction.STOP and decision.identities["exhausted"] == [(0, 0)]
    assert (
        "--extend-retries raises infra retries only; it does not raise the per-identity attempt cap "
        "(max_rollout_attempts=3)" in decision.reason
    )
    assert "infra-retry cap" not in decision.reason


def test_an_infra_cap_decision_keeps_the_cap_note() -> None:
    retryable = _collected(2, [infra_failure(1, attempt=0)])

    resubmit = _decide(retryable, extend_retries=1)
    assert resubmit.action == RecoveryAction.RESUBMIT
    assert "infra-retry cap 2: 1 configured + 0 recorded extension + 1 requested now" in resubmit.reason

    spent = _decide(
        _collected(2, [infra_failure(1, attempt=0)], infra_retries_used=2, infra_retries_state="ok"), extend_retries=1
    )
    assert spent.action == RecoveryAction.STOP and "the stored infra-retry budget is used (2 of 2)" in spent.reason
    assert "infra-retry cap 2" in spent.reason
