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
from pathlib import Path

from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.eval.recovery.evidence import COLLECT_SCRIPT, CollectedEvidence
from nemo_gym.eval.recovery.policy import RecoveryOutcomePolicyConfig


# Use the response serialized by the actual typed producer/verifier contract, not an output-only placeholder.
SAVED_ANSWER = NeMoGymResponse(
    id="response-saved",
    object="response",
    created_at=0,
    model="synthetic",
    output=[
        {
            "id": "message-saved",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "B", "annotations": []}],
        }
    ],
    parallel_tool_calls=False,
    tool_choice="auto",
    tools=[],
).model_dump(mode="json")


def ident(task: int, **extra) -> dict:
    return {"_ng_task_index": task, "_ng_rollout_index": 0, **extra}


def write_job_dir(
    root: Path,
    *,
    n: int,
    results: list[dict],
    failures: list[dict],
    offset: int | None = None,
    retries: int | None = None,
    walltime: int | None = None,
    chain: list[str] = (),
) -> Path:
    art = root / "artifacts"
    art.mkdir(parents=True)
    (art / "rollouts_materialized_inputs.jsonl").write_text(
        "".join(json.dumps(ident(t, responses_create_params={"input": f"prompt {t}"})) + "\n" for t in range(n))
    )
    (art / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (art / "rollouts_failures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in failures))
    if offset is not None:
        (art / "rollouts_resubmit_offset").write_text(str(offset))
    if retries is not None:
        (root / ".gym_infra_retries").write_text(str(retries))
    if walltime is not None:
        (root / ".gym_accumulated_walltime").write_text(str(walltime))
    if chain:
        (root / ".gym_job_chain").write_text("\n".join(chain))
    return root


def collect(job_dir: Path) -> CollectedEvidence:
    out = subprocess.run(
        [sys.executable, "-c", COLLECT_SCRIPT, str(job_dir)], capture_output=True, text=True, check=True
    )
    return CollectedEvidence.model_validate_json(out.stdout)


INFRA = dict(
    failure_kind="provider_unavailable",
    _ng_failure_terminal=False,
    _ng_failure_max_attempts=3,
    _ng_failure_events={
        "e": {
            "boundary": "model",
            "operation": "chat_completions",
            "operation_id": "op",
            "attempt_id": "a1",
            "cause_id": None,
        }
    },
)


def produce_quality(job: Path) -> None:
    """Use current raw saved rows; a grade-only fixture has unobserved response/trial health."""
    from nemo_gym.rollouts.rollout_health import run_health_checks

    run_health_checks(job / "artifacts" / "rollouts.jsonl", workers=1)


def collect_with_score(job_dir: Path, key: str) -> CollectedEvidence:
    out = subprocess.run(
        [sys.executable, "-c", COLLECT_SCRIPT, str(job_dir), key], capture_output=True, text=True, check=True
    )
    return CollectedEvidence.model_validate_json(out.stdout)


def final_model_zero(task: int, rollout: int) -> dict:
    """A terminal counts-as-zero sidecar row whose root event is the agent's policy request, caused by the model."""
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": rollout,
        "_ng_failure_class": "agent_run_error",
        "failure_kind": "model_transport_error",
        "_ng_failure_terminal": True,
        "_ng_failure_counts_as_zero": True,
        "_ng_failure_id": f"f-{task}-{rollout}",
        "_ng_failure_events": {
            f"f-{task}-{rollout}": {
                "boundary": "agent",
                "operation": "agent.policy_request",
                "failure_kind": "model_transport_error",
                "operation_id": "op-1",
                "attempt_id": "att-1",
                "cause_id": "cause",
            },
            "cause": {
                "boundary": "model",
                "operation": "model_request",
                "failure_kind": "model_request_error",
                "operation_id": "op-2",
                "attempt_id": "att-3",
                "cause_id": None,
            },
        },
    }


def _judge_failure(task: int, **extra) -> dict:
    return ident(task, _ng_failure_class="judge_failed", failure_kind="judge_failed", **extra)


class _StartsWith:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and other.startswith(self.prefix)

    def __repr__(self) -> str:
        return f"startswith({self.prefix!r})"


# A report without run.stats.responses leaves the response gate unknown, and the trial count is a lower bound.
HEALTH_UNKNOWN = _StartsWith("response/trial health unknown: ['")


def complete_job(tmp_path: Path) -> Path:
    job = write_job_dir(tmp_path / "job", n=4, results=[ident(t, reward=1.0) for t in range(4)], failures=[])
    (job / "artifacts" / "rollouts_aggregate_metrics.json").write_text(json.dumps({"agent": {"mean/reward": 0.75}}))
    write_postrun_receipt(job)
    return job


RECEIPT_FILES = {
    "rollouts_jsonl_fpath": "rollouts.jsonl",
    "resolved_config_yaml_fpath": "rollouts_resolved_config.yaml",
    "aggregate_metrics_json_fpath": "rollouts_aggregate_metrics.json",
    "failures_jsonl_fpath": "rollouts_failures.jsonl",
    "materialized_inputs_jsonl_fpath": "rollouts_materialized_inputs.jsonl",
}


def artifact_digests(job: Path) -> dict:
    files = {key: job / "artifacts" / name for key, name in RECEIPT_FILES.items()}
    return {key: hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None for key, f in files.items()}


def write_postrun_receipt(job: Path, run_id: str = "postrun-run") -> None:
    """The version-1 receipt a content-bound post-run export writes: its backend run id and the artifact digests."""
    receipt = {
        "version": 1,
        "run_id": run_id,
        "artifacts": artifact_digests(job),
        "uploaded_rollouts": True,
        "rollout_projection_version": 1,
    }
    (job / "artifacts" / "rollouts_mlflow_export_done").write_text(json.dumps(receipt))


STOP_PROVIDER = RecoveryOutcomePolicyConfig(by_failure_kind={"provider_unavailable": "fail"})


_SKIP_REASON = "CritPt smoke mode (fire_after): the batch was not sent to the grading API"


def _skipped_job(tmp_path, kind: str = "judge_skipped", n: int = 5):
    rows = [
        ident(t, mask_sample=True, reward=0.0, failure_kind=kind, failure_reason=_SKIP_REASON, response=SAVED_ANSWER)
        for t in range(n)
    ]
    return write_job_dir(tmp_path / "job", n=n, results=rows, failures=[])
