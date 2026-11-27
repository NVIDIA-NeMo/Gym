# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json


prepare = importlib.import_module("benchmarks.litmus-bench.prepare")


def test_prepared_rows_carry_the_source_uuid_as_task_id() -> None:
    source = [
        {
            "dataset": "litmus_bench",
            "question_family": "tier5_admet_direct",
            "question_license": "OpenMDW-1.1",
            "responses_create_params": {"input": [{"role": "user", "content": "Answer: number"}]},
            "expected_answer": 1.5,
            "answer_type": "float",
            "output_regex": r"Answer:\s*([-+]?\d*\.?\d+)",
            "match": {"rule": "abs_window", "abs_tol": 0.3},
            "property": "LOG HLM_CLint (mL/min/kg)",
            "method": "direct",
            "metadata": {"uuid": uuid, "sources": [{"dataset": "biogen", "row_id": "Mol1"}]},
        }
        for uuid in ("b2", "a1")
    ]

    rows = [json.loads(line) for line in prepare._render_rows(source, expected_rows=2).splitlines()]

    assert [row["task_id"] for row in rows] == ["b2", "a1"]
