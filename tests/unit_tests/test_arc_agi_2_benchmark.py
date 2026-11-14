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
import json

import pytest

from benchmarks.arc_agi_2 import prepare, score
from resources_servers.arc_code.grid import parse_answer


TASKS = {
    "aaaa0000": {
        "train": [{"input": [[1, 0]], "output": [[0, 1]]}],
        "test": [{"input": [[2, 0]], "output": [[0, 2]]}, {"input": [[3, 0]], "output": [[0, 3]]}],
    },
    "bbbb1111": {"train": [{"input": [[5]], "output": [[5, 5]]}], "test": [{"input": [[4]], "output": [[4, 4]]}]},
}


def test_rows_from_challenges_and_solutions(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare, "TOOLS_FPATH", tmp_path / "tools.jsonl")
    monkeypatch.setattr(prepare, "NO_TOOLS_FPATH", tmp_path / "no_tools.jsonl")
    challenges = {
        k: {"train": v["train"], "test": [{"input": t["input"]} for t in v["test"]]} for k, v in TASKS.items()
    }
    solutions = {k: [t["output"] for t in v["test"]] for k, v in TASKS.items()}
    (tmp_path / "c.json").write_text(json.dumps(challenges))
    (tmp_path / "s.json").write_text(json.dumps(solutions))
    prepare.from_challenges(tmp_path / "c.json", tmp_path / "s.json")

    tools = [json.loads(line) for line in (tmp_path / "tools.jsonl").open()]
    no_tools = [json.loads(line) for line in (tmp_path / "no_tools.jsonl").open()]
    assert [r["task_id"] for r in tools] == ["aaaa0000_0", "aaaa0000_1", "bbbb1111_0"]
    assert tools[1]["expected_output"] == [[0, 3]] and tools[1]["test_input"] == [[3, 0]]
    assert "responses_create_params" in tools[0] and "responses_create_params" not in no_tools[0]
    assert no_tools[0]["question"].endswith("Test input:\n2 0")
    # The grid format the prompt asks for is the one the resources server reads.
    assert parse_answer("<answer>\n" + prepare.grid_to_text(tools[0]["expected_output"]) + "\n</answer>") == [[0, 2]]


def test_score_counts_tasks_by_test_input_with_two_attempts():
    rows = [{"puzzle_id": "a"}, {"puzzle_id": "a"}, {"puzzle_id": "b"}]
    rollouts = [
        # a_0: 1 of 4 correct; a_1: 0 of 4; b_0: 4 of 4
        *({"_ng_task_index": 0, "reward": float(i == 0), "generated_tokens": 10} for i in range(4)),
        *({"_ng_task_index": 1, "reward": 0.0, "generated_tokens": 10} for _ in range(4)),
        *({"_ng_task_index": 2, "reward": 1.0, "response": {"usage": {"output_tokens": 40}}} for _ in range(4)),
    ]
    result = score.score(rows, rollouts)
    # task a: pass@1 = (0.25 + 0) / 2, pass@2 = (0.5 + 0) / 2; task b: 1 and 1
    assert result["pass@1"] == pytest.approx(100 * (0.125 + 1) / 2)
    assert result["pass@2"] == pytest.approx(100 * (0.25 + 1) / 2)
    assert result["mean_output_tokens"] == pytest.approx((8 * 10 + 4 * 40) / 12)
    assert result["test_inputs_without_rollouts"] == 0 and result["samples_per_input"] == [4]
