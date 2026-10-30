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
from pathlib import Path

from benchmarks.swebench.pro_v2.prepare import DATASET_REVISION, UPSTREAM_COMMIT, enrich_row, prepare


def make_upstream(root: Path, instance_id: str) -> None:
    task_dir = root / "v2" / "tasks" / instance_id
    assets = {
        "instruction.md": "Implement the PR.\n",
        "tests/test.sh": "#!/bin/bash\n",
        "tests/run_script.sh": "#!/bin/bash\n",
        "tests/parser.py": "print('parser')\n",
        "tests/test_patch.patch": "upstream test patch\n",
        "tests/config.json": json.dumps(
            {
                "fail_to_pass": '["new_test"]',
                "pass_to_pass": "[]",
                "selected_test_files_to_run": '["new_test"]',
            }
        ),
    }
    for relative_path, contents in assets.items():
        path = task_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")


def dataset_row(instance_id: str = "instance_example") -> dict:
    return {
        "repo": "example/repo",
        "instance_id": instance_id,
        "base_commit": "abc123",
        "patch": "patch",
        "test_patch": "hf test patch",
        "problem_statement": "Fix it",
        "fail_to_pass": '["stale_test"]',
        "pass_to_pass": '["stale_test"]',
        "selected_test_files_to_run": '["stale_test"]',
        "dockerhub_tag": "example-tag",
        "docker_image": "ghcr.io/scaleapi/swe-bench_pro-v2:instance_example",
    }


def test_enrich_row_embeds_the_upstream_verifier_files_and_instruction(tmp_path) -> None:
    make_upstream(tmp_path, "instance_example")

    row = enrich_row(dataset_row(), tmp_path)

    assert row["test_sh"] == "#!/bin/bash\n"
    assert row["run_script"] == "#!/bin/bash\n"
    assert row["parser_script"] == "print('parser')\n"
    assert row["test_patch"] == "upstream test patch\n"
    assert row["responses_create_params"]["input"] == [{"role": "user", "content": "Implement the PR.\n"}]
    assert (row["subset"], row["evaluator_commit"], row["dataset_revision"]) == (
        "pro_v2",
        UPSTREAM_COMMIT,
        DATASET_REVISION,
    )


def test_enrich_row_takes_the_test_lists_from_the_graded_config_json(tmp_path) -> None:
    make_upstream(tmp_path, "instance_example")

    row = enrich_row(dataset_row(), tmp_path)

    assert row["fail_to_pass"] == '["new_test"]'
    assert row["pass_to_pass"] == "[]"
    assert row["selected_test_files_to_run"] == '["new_test"]'


def test_prepare_writes_one_row_per_task(tmp_path) -> None:
    make_upstream(tmp_path, "instance_a")
    make_upstream(tmp_path, "instance_b")
    output = tmp_path / "out" / "benchmark.jsonl"

    prepare([dataset_row("instance_a"), dataset_row("instance_b")], tmp_path, output)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["instance_a", "instance_b"]
