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
"""The resume script's guard after a possibly delivered request: what it continues or names from the saved evidence."""

import hashlib

from nemo_gym.orchestration.api import ResumeConfig
from tests.unit_tests.eval.recovery.test_possibly_delivered_recovery import _lost_connection_run
from tests.unit_tests.orchestration.test_resume_script import _run_prologue


def _digests(job):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (job / "artifacts").glob("*") if p.is_file()}


def test_the_guard_continues_a_failed_allocation_with_lost_connections_as_an_infra_retry(tmp_path):
    _lost_connection_run(tmp_path)
    before = _digests(tmp_path)
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), "FAILED", exit_code="75:0", prev_job_id="100")
    assert "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert "Saved recovery evidence: 2 identity(ies) need the policy again" in result.stdout
    assert (tmp_path / ".gym_infra_retries").read_text().strip() == "1"
    assert _digests(tmp_path) == before


def test_the_guard_names_the_saved_evidence_when_nothing_is_retryable(tmp_path):
    _lost_connection_run(tmp_path, lost=0, lost_with_partial=1)
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), "FAILED", exit_code="75:0", prev_job_id="100")
    assert "REACHED_WORK" not in result.stdout
    marker = (tmp_path / ".gym_chain_incomplete").read_text()
    assert "without an infrastructure cause" not in marker
    assert marker.startswith("Saved recovery evidence has no eligible retries; 1 failure(s)")
    assert "retryable 0 (resumed), unknown 1" in marker
    assert marker.rstrip().endswith("(previous job 100 ended FAILED)")
