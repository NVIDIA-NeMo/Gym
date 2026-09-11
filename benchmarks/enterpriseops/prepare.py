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
"""Prepare an EnterpriseOps-Gym benchmark split.

``mode`` selects the upstream tool-set config on HuggingFace: ``oracle`` gives each task
exactly the tools it needs, and ``plus_5_tools`` / ``plus_10_tools`` / ``plus_15_tools`` add
that many distractor tools. Only the tool set differs; verifiers and scoring are identical,
so nothing in the resources server changes. Note the plus_N splits carry slightly FEWER
tasks than oracle (637 vs 649 upstream), so per-mode row counts are not comparable.

Downloads the ServiceNow-AI/EnterpriseOps-Gym HuggingFace dataset (config = tool-set mode,
split = domain) and converts every domain — including hybrid — into one combined NeMo Gym
JSONL, baking in tool schemas from the per-domain snapshots hosted alongside the
enterpriseops_gym resources server (fetched by its prepare.py; see snapshot_tools.py
there for how they are captured).

Requires:
- Egress to huggingface.co, for both the task dataset and the tool snapshots. Set
  NEMO_GYM_EOG_TOOLS_DIR to a directory holding the seven snapshots to skip the
  snapshot download. If the Hub is unreachable and NEMO_GYM_EOG_LOCAL_TASKS is set
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
from resources_servers.enterpriseops_gym.prepare import SNAPSHOT_FILENAMES, ensure_tool_snapshots


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_MODE = "oracle"
VALID_MODES = ("oracle", "plus_5_tools", "plus_10_tools", "plus_15_tools")


def output_fpath(mode: str = DEFAULT_MODE) -> Path:
    """Where a given mode's converted dataset lands. Must match the `jsonl_fpath` of the
    benchmark config that selects it, or `gym eval prepare` raises a ConfigError."""
    return DATA_DIR / f"enterpriseops_{mode}_benchmark.jsonl"


# Set NEMO_GYM_EOG_HF_DATASET to a local snapshot of the dataset repo on machines without
# Hub egress (datasets.load_dataset accepts a local directory path in place of a repo id).
HF_REPO_ID = os.getenv("NEMO_GYM_EOG_HF_DATASET", "ServiceNow-AI/EnterpriseOps-Gym")
DOMAINS = ["calendar", "csm", "drive", "email", "hr", "itsm", "teams", "hybrid"]

# One snapshot per single-domain gym; hybrid tasks reference multiple gyms, so hybrid
# uses the full union. Derived from SNAPSHOT_FILENAMES so the two cannot drift.
DOMAIN_SNAPSHOTS: Dict[str, List[str]] = {Path(name).stem: [name] for name in SNAPSHOT_FILENAMES}
DOMAIN_SNAPSHOTS["hybrid"] = list(SNAPSHOT_FILENAMES)

LOCAL_TASKS_ENV_VAR = "NEMO_GYM_EOG_LOCAL_TASKS"


def _convert_domain(domain: str, tasks, out_file, snapshots_dir: Path, mode: str) -> int:
    snapshot_paths = [snapshots_dir / name for name in DOMAIN_SNAPSHOTS[domain]]
    gym_tools = load_snapshots(snapshot_paths)
    num_written = 0
    for task_id, task in tasks:
        row = convert_task(task, task_id, domain, mode, gym_tools)
        out_file.write(json.dumps(row) + "\n")
        num_written += 1
    return num_written


def prepare(mode: str = DEFAULT_MODE) -> Path:
    """Download and convert one EnterpriseOps-Gym tool-set split. Returns the output path."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Valid modes: {', '.join(VALID_MODES)}")
    out_fpath = output_fpath(mode)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch the tool schemas BEFORE opening the output file: a download failure here
    # must not truncate an existing benchmark JSONL.
    snapshots_dir = ensure_tool_snapshots()

    local_tasks_root = os.getenv(LOCAL_TASKS_ENV_VAR)
    total = 0
    with open(out_fpath, "w") as out_file:
        try:
            for domain in DOMAINS:
                tasks = load_tasks_from_hf(HF_REPO_ID, mode, domain)
                count = _convert_domain(domain, tasks, out_file, snapshots_dir, mode)
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
                count = _convert_domain(domain_dir.name, tasks, out_file, snapshots_dir, mode)
                print(f"{domain_dir.name}: {count} tasks (local fallback)")
                total += count

    print(f"Wrote {total} tasks to {out_fpath} (mode={mode})")
    return out_fpath


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=VALID_MODES)
    prepare(parser.parse_args().mode)
