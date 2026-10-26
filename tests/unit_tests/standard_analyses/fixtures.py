# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact numerical fixtures and persisted Gym trajectory contracts."""

from nemo_gym.rollout_observability import TrajectoryModelCall, TrajectoryRecord
from nemo_gym.standard_analyses.models import Call, Rollout


def demo_records():
    records = []
    for task_index in range(3):
        for repeat in range(4):
            lengths = (120, 400, 1600, 250, 12000 if repeat == 3 else 800, 300)
            calls = tuple(
                Call(
                    id=f"c{index}",
                    index=index,
                    output_tokens=length * (task_index + 1),
                    reasoning_tokens=int(length * 0.7) if repeat != 2 or index != 5 else None,
                    input_tokens=1000 + index * 800,
                    duration_s=length / 70 + repeat * 2,
                )
                for index, length in enumerate(lengths)
            )
            records.append(
                Rollout(
                    run_id="synthetic-demo",
                    task_id=str(task_index),
                    repeat=repeat,
                    agent="demo-agent",
                    model="synthetic-policy",
                    benchmark="synthetic-example",
                    config_id="demo-v1",
                    success=repeat != 3,
                    health="healthy",
                    capture_complete=True,
                    calls=calls,
                )
            )
    return records


def gym_rows():
    rows = []
    for r in demo_records():
        trajectory = TrajectoryRecord(
            task_id=r.task_id,
            rollout_id=f"{r.task_id}/{r.repeat}",
            model_calls=[
                TrajectoryModelCall(
                    model_call_id=c.id,
                    duration_ms=c.duration_s * 1000,
                    token_stats={
                        "completion_tokens": c.output_tokens,
                        "reasoning_tokens": c.reasoning_tokens,
                        "prompt_tokens": c.input_tokens,
                    },
                    response_metadata={"model": r.model},
                )
                for c in r.calls
            ],
        )
        rows.append(
            {
                "_ng_task_index": int(r.task_id),
                "_ng_rollout_index": r.repeat,
                "agent_ref": {"name": r.agent},
                "reward": int(r.success),
                "ng_model_call_capture": {"calls": [{"model_call_id": c.id} for c in r.calls]},
                "ng_trajectory": trajectory.model_dump(mode="json"),
            }
        )
    return rows
