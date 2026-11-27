# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Saved runs written through a real Store, and the collector run as the run host runs it, for recovery tests."""

import json
import subprocess

from nemo_gym.eval.recovery.evidence import COLLECT_SCRIPT, CollectedEvidence
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore


def save_store_run(job, case, n=1):
    """Save a run of ``n`` rows into a real Store under ``job``; the first row is in ``case`` and the rest are healthy."""
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(n)]
    source = job / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare():
        inputs = json.loads(json.dumps(rows))
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": 3})

    output = artifacts / "rollouts.jsonl"
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for index in range(n):
            row = dict(rows[index])
            selected_case = case if index == 0 else "healthy"
            if selected_case == "missing":
                continue
            if selected_case in ("infra", "exhausted", "reset"):
                attempts = 3 if selected_case == "exhausted" else 2
                for attempt in range(attempts):
                    row = dict(rows[index])
                    store.allocate_attempt(row)
                    failure = json.loads(provider_failures_jsonl(index, attempt + 1, 3).splitlines()[-1])
                    store.record_outcome(row | failure)
                continue
            store.record_dispatch(row)
            if selected_case in ("healthy", "wrong", "prior_reservation", "duplicate_history"):
                store.record_outcome(row | {"reward": 0.0 if selected_case == "wrong" else 1.0})
                if selected_case in ("prior_reservation", "duplicate_history"):
                    store.allocate_attempt(row)
                    if selected_case == "duplicate_history":
                        store.record_outcome(row | {"reward": 0.0})
            elif selected_case in ("judge", "judge_then_reservation"):
                store.record_outcome(
                    row
                    | {
                        "_ng_failure_class": "judge_failed",
                        "failure_kind": "judge_failed",
                        "response": {
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [{"type": "output_text", "text": "synthetic saved answer"}],
                                }
                            ],
                        },
                    }
                )
                if selected_case == "judge_then_reservation":
                    store.allocate_attempt(row)
            elif selected_case == "final_zero":
                store.record_outcome(
                    row
                    | {
                        "_ng_failure_class": "agent_run_error",
                        "failure_kind": "model_context_limit",
                        "_ng_failure_terminal": True,
                        "_ng_failure_counts_as_zero": True,
                    }
                )
            elif selected_case == "unknown_kill":
                store.record_outcome(row | {"_ng_failure_class": "kill_shaped", "_ng_omitted_rollout": True})
            else:
                raise AssertionError(selected_case)
    if case == "reset":
        with RolloutStore.start_or_resume(output, prepare, resume=True, resubmit=True):
            pass
    return output


def collect_isolated(job):
    """Collect ``job``'s evidence with the stdlib collector run by the system Python in isolated mode, as the run host does."""
    result = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True
    )
    return CollectedEvidence.model_validate_json(result.stdout)


def provider_failures_jsonl(task: int, attempts: int, cap: int) -> str:
    """``attempts`` failed-attempt rows of a task, as saved-failure JSONL: provider unavailable, retries capped at ``cap``."""
    rows = [
        {
            "_ng_task_index": task,
            "_ng_rollout_index": 0,
            "_ng_attempt_index": attempt,
            "failure_kind": "provider_unavailable",
            "_ng_failure_class": "provider_unavailable",
            "_ng_failure_terminal": False,
            "_ng_failure_max_attempts": cap,
            "_ng_failure_id": f"f{task}-{attempt}",
            "_ng_failure_events": {
                f"f{task}-{attempt}": {
                    "event_id": f"f{task}-{attempt}",
                    "boundary": "model",
                    "failure_kind": "provider_unavailable",
                    "operation": "model_request",
                    "operation_id": f"op{task}",
                    "attempt_id": f"a{task}-{attempt}",
                    "cause_id": None,
                }
            },
        }
        for attempt in range(attempts)
    ]
    return "".join(json.dumps(row) + "\n" for row in rows)
