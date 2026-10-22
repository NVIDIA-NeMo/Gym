# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Carry a resources server's ``/verify`` reply the way an evaluation run does, for tests of its failure rows.

:func:`verify_reply_through_collection` validates the reply against the agent's verify response contract (a
``BaseVerifyResponse`` that keeps extra fields, as the simple agent's does), turns it into the collector's record and
failures-sidecar row, and returns what evaluation makes of it: the rows counted in the score and the run's
completeness fields. A row the agent rejects raises here, as it would fail the agent's ``/run``.
"""

from pathlib import Path
from typing import Any

import orjson
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseVerifyResponse


class _AgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def verify_reply_through_collection(
    verify_reply: dict[str, Any], output_dir: Path, *, expected: int = 10
) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[int, int], dict[str, Any]]:
    """``(record, counted rows, (unresolved, owed), completeness fields)`` for one rollout out of ``expected``."""
    from nemo_gym.rollout_collection import (
        NG_FAILURE_CAUSE_KEY,
        NG_FAILURE_CLASS_KEY,
        _episode_record,
        _failure_rows_counted_as_zero,
        _failures_path_for,
        _infra_unresolved_counts,
        _infra_unresolved_metrics,
        _latest_failure_rows,
        failure_cause_of_row,
    )

    agent_reply = _AgentVerifyResponse.model_validate(verify_reply).model_dump(mode="json")
    record = _episode_record({"task_id": "t0", "result": agent_reply})
    record |= {"_ng_task_index": 0, "_ng_rollout_index": 0}
    failures = _failures_path_for(output_dir / "rollouts.jsonl")
    if record.get(NG_FAILURE_CLASS_KEY) is not None:
        # A failure row goes to the sidecar, with the cause the rule reads, as the collector stamps it.
        record[NG_FAILURE_CAUSE_KEY] = failure_cause_of_row(record)
        failures.write_bytes(orjson.dumps(record) + b"\n")
    else:
        # A result stays in the rollouts file; it scores by its own reward and mask.
        failures.write_bytes(b"")
    counted = _failure_rows_counted_as_zero([failures], None, set())
    latest = _latest_failure_rows([failures])
    counts = _infra_unresolved_counts(latest, dict.fromkeys(latest, 1), set(), expected=expected, missing=0)
    return record, counted, counts, _infra_unresolved_metrics(*counts)
