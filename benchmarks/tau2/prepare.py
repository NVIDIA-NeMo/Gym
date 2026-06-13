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
"""Prepare Tau2 benchmark data."""

import json
from pathlib import Path
from subprocess import run


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "tau2_benchmark.jsonl"
AGI_EVAL_AIRLINE_TASKS_FPATH = (
    BENCHMARK_DIR.parent.parent
    / "responses_api_agents"
    / "tau2"
    / "prompt_overrides"
    / "agi_eval"
    / "airline_tasks.json"
)


def _load_agi_eval_airline_tasks() -> dict[str, dict]:
    if not AGI_EVAL_AIRLINE_TASKS_FPATH.exists():
        return {}

    tasks = json.loads(AGI_EVAL_AIRLINE_TASKS_FPATH.read_text())
    tasks_by_id = {task["id"]: task for task in tasks}
    serialized = json.dumps(tasks_by_id, ensure_ascii=False)
    markers = ["$647", "By NO MEANS you will upgrade your cabin"]
    missing = [marker for marker in markers if marker not in serialized]
    if missing:
        raise RuntimeError(
            f"AGI-Eval airline task override missing markers: {missing}"
        )
    return tasks_by_id


def prepare() -> Path:
    """Download and prepare Tau2 data. Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cwd = Path(__file__).parent
    data_dirpath = cwd / "nemo_gym_data"
    if not data_dirpath.exists():
        run(
            """git clone https://github.com/bxyu-nvidia/tau2-bench \
&& cd tau2-bench \
&& git checkout bxyu/nemo_gym_data \
&& bash dump_nemo_gym_data.sh \
&& cp -r nemo_gym_data ../nemo_gym_data \
&& cd .. \
&& rm -rf tau2-bench""",
            shell=True,
            cwd=cwd,
            check=True,
            executable="/bin/bash",
        )

    agi_eval_airline_tasks = _load_agi_eval_airline_tasks()

    samples = []
    for path in data_dirpath.glob("*/*.json"):
        data = json.loads(path.read_text())
        data["config"]["save_to"] = ""

        if data["config"]["domain"] == "airline" and agi_eval_airline_tasks:
            task_id = str(data["task"]["id"])
            if task_id not in agi_eval_airline_tasks:
                raise RuntimeError(f"Missing AGI-Eval airline task override for {task_id}")
            data["task"] = agi_eval_airline_tasks[task_id]

        # The default is `all_with_nl_assertions` which may actually be a mistake when running from CLI
        # We always see "nl": null or "nl": "No nl_assertions to evaluate" for results
        data["evaluation_type"] = "all"
        evaluation_criteria = data["task"]["evaluation_criteria"]
        reward_basis = evaluation_criteria.setdefault("reward_basis", ["DB", "COMMUNICATE"])
        if "NL_ASSERTION" in reward_basis:
            reward_basis.remove("NL_ASSERTION")

        # The actual prompts are constructed on the fly by Tau2-Bench
        # data["responses_create_params"]

        # Clean temperature sampling parameters
        data["config"]["llm_args_user"].pop("temperature")

        samples.append(data)

    count = 0
    with open(OUTPUT_FPATH, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
