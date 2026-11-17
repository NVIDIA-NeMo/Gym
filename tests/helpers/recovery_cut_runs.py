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
"""Evidence builders the recovery and orchestration tests share."""

import hashlib
import json
import subprocess
import sys

import pytest

from nemo_gym.eval.recovery.evidence import COLLECT_SCRIPT, CollectedEvidence, select_managed_evidence
from nemo_gym.eval.recovery.planner import decide_from_collected
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore


def _cut_run(
    job, *, scored=2, in_flight=2, not_started=2, masked=0, unowned=0, retryable=0, allocation="100", chain=True
):
    """A run cut while ``in_flight`` reservations had no row; ``masked`` results, ``unowned`` and ``retryable`` failures."""
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    total = scored + in_flight + not_started + masked + unowned + retryable
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(total)]
    source = job / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare():
        inputs = json.loads(json.dumps(rows))
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": 3})

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SLURM_JOB_ID", allocation)
        with RolloutStore.start_or_resume(artifacts / "rollouts.jsonl", prepare, resume=False) as store:
            index = 0
            for _ in range(scored):
                row = dict(rows[index])
                store.record_dispatch(row)
                store.record_outcome(row | {"reward": 1.0})
                index += 1
            for _ in range(masked):
                row = dict(rows[index])
                store.record_dispatch(row)
                store.record_outcome(row | {"reward": 0.0, "mask_sample": True, "evaluation_completed": False})
                index += 1
            for _ in range(unowned):
                row = dict(rows[index])
                store.record_dispatch(row)
                store.record_outcome(row | {"_ng_failure_class": "unowned_class", "failure_kind": "not_registered"})
                index += 1
            for _ in range(retryable):
                row = dict(rows[index])
                store.record_dispatch(row)
                store.record_outcome(
                    row
                    | {
                        "failure_kind": "provider_unavailable",
                        "_ng_failure_class": "evaluation",
                        "_ng_failure_terminal": False,
                        "_ng_failure_max_attempts": 3,
                    }
                )
                index += 1
            for _ in range(in_flight):
                # Reserved and dispatched; the allocation ended before any outcome was saved.
                store.record_dispatch(dict(rows[index]))
                index += 1
    if chain:
        (job / ".gym_job_chain").write_text(f"{allocation}\n")
        (job / ".gym_accumulated_walltime").write_text("14400\n")
    return artifacts / "rollouts.jsonl"


def _decide(job, *, allocation_cut, live_job=False):
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True
    )
    collected = select_managed_evidence(CollectedEvidence.model_validate_json(raw.stdout), job)
    assert collected.managed_reader_error is None
    return collected, decide_from_collected(
        collected, max_infra_retries=2, max_walltime_s=None, live_job=live_job, allocation_cut=allocation_cut
    )


def _digests(job):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (job / "artifacts").glob("*") if p.is_file()}
