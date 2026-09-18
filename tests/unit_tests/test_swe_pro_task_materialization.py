# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from benchmarks.swebench.pro.materialize_single_agent_tasks import materialize_row


def test_materialize_swe_pro_row_separates_identity_input_and_task_data() -> None:
    row = {
        "agent_ref": {"name": "legacy-agent"},
        "instance_id": "instance-1",
        "repo": "owner/repo",
        "responses_create_params": {"input": "fix it"},
        "task_source": "legacy-source",
    }

    materialized = materialize_row(row, taskset="swebench_pro_smoke", revision="demo-v1")

    assert materialized == {
        "task_id": {
            "taskset": "swebench_pro_smoke",
            "task_id": "instance-1",
            "revision": "demo-v1",
        },
        "task_input": {
            "responses_create_params": {"input": "fix it"},
            "task_data": {
                "instance_id": "instance-1",
                "repo": "owner/repo",
            },
        },
    }
