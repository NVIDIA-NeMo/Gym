# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ADDITIONAL_NOTES below adapts agents/aide/additional_notes.txt from
# https://github.com/science-machine/biomlbench at d5b4488d642bfea40e73fcc6eb097b8f57118e4c
# (its two step-budget lines removed, its placeholders filled). That text is under this license:
#
#   MIT License
#
#   Copyright (c) 2025 BioML-bench team
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
"""Write BioML-bench v0.1-alpha tasks as Harbor task folders and the Gym JSONL index.

Each task runs in the upstream task image. The task data is prepared inside the sandbox at
start (``environment.py``), so this script writes only prompts, task settings and scripts.
"""

import json
import shutil
import urllib.request
from pathlib import Path

import yaml


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
TASKS_DIR = DATA_DIR / "tasks"
OUTPUT_FPATH = DATA_DIR / "biomlbench.jsonl"
# MD5s of each task's prepared files from DOCKER_IMAGE, checked in the sandbox before the agent starts.
CHECKSUMS_FPATH = BENCHMARK_DIR / "checksums.json"
AGENT_NAME = "biomlbench_harbor_agent"

SOURCE_REPO = "science-machine/biomlbench"
# The commit the image was built from: its task descriptions and configs match this tree, not later edits.
SOURCE_REVISION = "d5b4488d642bfea40e73fcc6eb097b8f57118e4c"
# The image BioML-bench v0.1a publishes for every agent (`scripts/pull_prebuilt_images.sh`).
DOCKER_IMAGE = "millerh1/biomlbench-env@sha256:4c978736382412148af2da9d4093f1bff0a9076147708f2e4b954f0a213f8aae"

# The v0.1a task list (`experiments/biomlbench_v0.1a.txt`, added with the MIT license in 673f3d8d) without
# its four Kaggle tasks: their data needs a Kaggle account that accepted each competition's rules, and a GPU.
TASK_IDS = (
    "proteingym-dms/SPIKE_SARS2_Starr_2020_binding",
    "proteingym-dms/SBI_STAAM_Tsuboyama_2023_2JVG",
    "proteingym-dms/PSAE_PICP2_Tsuboyama_2023_1PSE",
    "proteingym-dms/CBX4_HUMAN_Tsuboyama_2023_2K28",
    "proteingym-dms/Q8EG35_SHEON_Campbell_2022_indels",
    "proteingym-dms/CSN4_MOUSE_Tsuboyama_2023_1UFM_indels",
    "manual/open-problems-predict-modality",
    "manual/open-problems-single-cell-perturbations",
    "manual/open-problems-cell-cell-communication-ligand-target",
    "manual/open-problems-spatially-variable-genes",
    "manual/open-problems-label-projection",
    "polarishub/polaris-pkis2-egfr-wt-c-1",
    "polarishub/polaris-adme-fang-hclint-1",
    "polarishub/polaris-adme-fang-hppb-1",
    "polarishub/polaris-adme-fang-solu-1",
    "polarishub/tdcommons-cyp2d6-substrate-carbonmangels",
    "polarishub/tdcommons-lipophilicity-astrazeneca",
    "polarishub/tdcommons-caco2-wang",
    "polarishub/tdcommons-herg",
    "polarishub/tdcommons-bbb-martins",
)
# The paper's budget for non-imaging tasks: 16 vCPUs, 64 GB RAM, no GPU, 8 hours.
CPUS = 16
MEMORY_MB = 65536
STORAGE_MB = 102400
DEFAULT_TIME_LIMIT_SEC = 28800
# Sandbox start, data preparation and grading.
VERIFIER_TIMEOUT_SEC = 3600
# Sandbox start includes the 12 GB image pull and the in-sandbox data preparation.
BUILD_TIMEOUT_SEC = 3600

# The agent notes BioML-bench appends for AIDE (`agents/aide/additional_notes.txt`, text unchanged)
# without its two step-budget lines, which describe AIDE's own loop.
ADDITIONAL_NOTES = """ADDITIONAL NOTES
------

- **Compute**: You have access to a CPU with the appropriate drivers installed.
- **Total Runtime**: You have a maximum of {time_limit} hours to attempt this task. You may iteratively develop your solution within this time frame, but bear in mind that program runtime also counts towards this limit.
- DO NOT QUIT EARLY. KEEP GOING UNTILL YOU GET DECENT PERFORMANCE ON THE TASK.
- **Use as much of the available time as possible to refine and optimize your submission**: Your goal is to achieve the best possible score, not just to submit your first working solution. Consider strategies like validating your answer using the "Validate Submission" tool at the http://localhost:5000/validate endpoint, manually spot-checking predictions, building custom validation sets and grading functions, and comparing different algorithms. You may keep working until the time limit expires -- as long as your best submission is stored at the designated location at the end of your run, it will be picked up for grading.
"""


def _fetch(path: str) -> str:
    url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_REVISION}/{path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode()


def task_slug(task_id: str) -> str:
    """Harbor task folder name: the BioML-bench id with its folder separator replaced."""
    return task_id.replace("/", "__")


def _format_time(seconds: int) -> str:
    # Same rendering as AIDE's start.sh.
    return f"{seconds // 3600}hrs {seconds % 3600 // 60}mins {seconds % 60}secs"


def instruction(instructions: str, description: str, time_limit_sec: int) -> str:
    """Assemble the prompt the way BioML-bench's agent start scripts do."""
    notes = ADDITIONAL_NOTES.format(time_limit=_format_time(time_limit_sec))
    return f"{instructions}\n{notes}\nCOMPETITION INSTRUCTIONS\n------\n\n{description}"


def task_toml(task_id: str, time_limit_sec: int) -> str:
    # The verifier runs in a fresh sandbox that receives only the submission (see environment.py).
    return f"""version = "1.0"
artifacts = ["/home/submission"]

[metadata]
task_name = {json.dumps(task_id)}
source = "{SOURCE_REPO}@{SOURCE_REVISION}"

[agent]
timeout_sec = {time_limit_sec}
user = "nonroot"

[verifier]
timeout_sec = {VERIFIER_TIMEOUT_SEC}
user = "root"
environment_mode = "separate"

[environment]
docker_image = {json.dumps(DOCKER_IMAGE)}
workdir = "/home"
build_timeout_sec = {BUILD_TIMEOUT_SEC}
cpus = {CPUS}
memory_mb = {MEMORY_MB}
storage_mb = {STORAGE_MB}
gpus = 0
allow_internet = true
"""


def write_task(
    destination: Path, task_id: str, instructions: str, description: str, config: dict, checksums: dict
) -> None:
    time_limit_sec = int(config.get("time_limit") or DEFAULT_TIME_LIMIT_SEC)
    runtime = BENCHMARK_DIR / "biomlbench_task.py"
    task_json = json.dumps({"task_id": task_id, "checksums": checksums}, indent=2) + "\n"

    destination.mkdir(parents=True)
    (destination / "instruction.md").write_text(instruction(instructions, description, time_limit_sec))
    (destination / "task.toml").write_text(task_toml(task_id, time_limit_sec))
    for folder in ("environment", "tests"):
        (destination / folder).mkdir()
        shutil.copy2(runtime, destination / folder / runtime.name)
        (destination / folder / "task.json").write_text(task_json)
    # The agent's sandbox never held the answers: prepare them here from the same pins, then grade.
    (destination / "tests" / "test.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/verifier\nexport PATH=/opt/conda/envs/biomlb/bin:$PATH\n"
        f"python /tests/biomlbench_task.py prepare --task-id {task_id} --data-dir /private/data "
        "--public-dir /tmp/biomlbench-public --task-json /tests/task.json\n"
        f"exec python /tests/biomlbench_task.py grade --task-id {task_id} --data-dir /private/data "
        "--submission-dir /home/submission --output /logs/verifier/reward.json\n"
    )
    # Reference "solution": BioML-bench's dummy agent, the sample submission. Checks the lifecycle, scores no medal.
    (destination / "solution").mkdir()
    (destination / "solution" / "solve.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        'sample="$(ls /home/data/sample_submission.* | head -n 1)"\n'
        'cp "$sample" "/home/submission/submission.${sample##*.}"\n'
    )
    for script in (destination / "tests" / "test.sh", destination / "solution" / "solve.sh"):
        script.chmod(0o755)


def prepare() -> Path:
    """Write the 20 CPU tasks of BioML-bench v0.1a and their Gym index."""
    instructions = _fetch("environment/instructions.txt")
    checksums = json.loads(CHECKSUMS_FPATH.read_text())
    if missing := set(TASK_IDS) - checksums.keys():
        raise ValueError(f"{CHECKSUMS_FPATH.name} has no checksums for {sorted(missing)}")
    staging = DATA_DIR / "tasks.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for task_id in TASK_IDS:
        task_dir = f"biomlbench/tasks/{task_id}"
        config = yaml.safe_load(_fetch(f"{task_dir}/config.yaml"))
        if config["id"] != task_id:
            raise ValueError(f"{task_dir}/config.yaml names {config['id']!r}")
        description = _fetch(f"{task_dir}/description.md")
        write_task(staging / task_slug(task_id), task_id, instructions, description, config, checksums[task_id])
    if TASKS_DIR.exists():
        shutil.rmtree(TASKS_DIR)
    staging.rename(TASKS_DIR)

    with OUTPUT_FPATH.open("w") as output:
        for task_id in TASK_IDS:
            row = {
                "task_name": task_slug(task_id),
                "instance_id": f"biomlbench::{task_id}",
                "responses_create_params": {"input": []},
                "agent_ref": {"name": AGENT_NAME, "type": "responses_api_agents"},
            }
            output.write(json.dumps(row) + "\n")
    print(f"Prepared {len(TASK_IDS)} BioML-bench tasks at {TASKS_DIR}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
