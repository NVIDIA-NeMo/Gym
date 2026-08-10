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
"""Prepare the EnterpriseOps-Gym benchmark (oracle mode, public split).

Downloads the ServiceNow-AI/EnterpriseOps-Gym HuggingFace dataset (config = tool-set mode,
split = domain) and converts every domain — including hybrid — into one combined NeMo Gym
JSONL, baking in tool schemas from the per-domain snapshots shipped with the
enterpriseops_gym resources server (see snapshot_tools.py there).

Requires:
- Egress to huggingface.co. If the Hub is unreachable and NEMO_GYM_EOG_LOCAL_TASKS is set
  to an EnterpriseOps-Gym checkout's task folder root (containing <domain>/ task JSON
  dirs, e.g. data/revised), those local tasks are converted instead as a fallback.
- The MCP gym Docker containers only at RUN time, not at prepare time.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

from resources_servers.enterpriseops_gym.convert_tasks import (
    convert_task,
    load_snapshots,
    load_tasks_from_dir,
    load_tasks_from_hf,
)


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "enterpriseops_oracle_benchmark.jsonl"

# Set NEMO_GYM_EOG_HF_DATASET to a local snapshot of the dataset repo on machines without
# Hub egress (datasets.load_dataset accepts a local directory path in place of a repo id).
HF_REPO_ID = os.getenv("NEMO_GYM_EOG_HF_DATASET", "ServiceNow-AI/EnterpriseOps-Gym")
MODE = "oracle"
DOMAINS = ["calendar", "csm", "drive", "email", "hr", "itsm", "teams", "hybrid"]

SNAPSHOTS_DIR = BENCHMARK_DIR.parent.parent / "resources_servers" / "enterpriseops_gym" / "data" / "tools"
# All 7 gym snapshots; hybrid tasks reference multiple gyms, so hybrid uses the full union.
DOMAIN_SNAPSHOTS: Dict[str, List[str]] = {
    "calendar": ["calendar.json"],
    "csm": ["csm.json"],
    "drive": ["drive.json"],
    "email": ["email.json"],
    "hr": ["hr.json"],
    "itsm": ["itsm.json"],
    "teams": ["teams.json"],
    "hybrid": ["calendar.json", "csm.json", "drive.json", "email.json", "hr.json", "itsm.json", "teams.json"],
}

LOCAL_TASKS_ENV_VAR = "NEMO_GYM_EOG_LOCAL_TASKS"


def _convert_domain(domain: str, tasks, out_file) -> int:
    snapshot_paths = [SNAPSHOTS_DIR / name for name in DOMAIN_SNAPSHOTS[domain]]
    gym_tools = load_snapshots(snapshot_paths)
    num_written = 0
    for task_id, task in tasks:
        row = convert_task(task, task_id, domain, MODE, gym_tools)
        out_file.write(json.dumps(row) + "\n")
        num_written += 1
    return num_written


def prepare() -> Path:
    """Download and convert the EnterpriseOps-Gym public split. Returns the output path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    local_tasks_root = os.getenv(LOCAL_TASKS_ENV_VAR)
    total = 0
    with open(OUTPUT_FPATH, "w") as out_file:
        try:
            for domain in DOMAINS:
                tasks = load_tasks_from_hf(HF_REPO_ID, MODE, domain)
                count = _convert_domain(domain, tasks, out_file)
                print(f"{domain}: {count} tasks (HuggingFace)")
                total += count
        except Exception as e:
            if not local_tasks_root:
                raise RuntimeError(
                    f"Could not download {HF_REPO_ID} from HuggingFace ({type(e).__name__}: {e}). "
                    f"If this machine has no Hub egress, fetch the dataset elsewhere or set "
                    f"{LOCAL_TASKS_ENV_VAR}=<EOG checkout task root> (e.g. .../data/revised) to "
                    f"convert local task JSONs instead."
                ) from e
            print(f"HuggingFace unreachable ({type(e).__name__}); falling back to local tasks at {local_tasks_root}")
            out_file.seek(0)
            out_file.truncate()
            total = 0
            for domain_dir in sorted(Path(local_tasks_root).iterdir()):
                if not domain_dir.is_dir() or domain_dir.name not in DOMAIN_SNAPSHOTS:
                    continue
                tasks = load_tasks_from_dir(domain_dir)
                count = _convert_domain(domain_dir.name, tasks, out_file)
                print(f"{domain_dir.name}: {count} tasks (local fallback)")
                total += count

    print(f"Wrote {total} tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
