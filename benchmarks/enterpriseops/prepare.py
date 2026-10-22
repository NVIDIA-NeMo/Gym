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

Downloads a pinned revision of the ServiceNow-AI/EnterpriseOps-Gym HuggingFace dataset
(config = tool-set mode, split = domain) and converts every domain — including hybrid —
into one combined NeMo Gym JSONL, baking in tool schemas from the per-domain snapshots
shipped with the enterpriseops_gym resources server (see snapshot_tools.py there).

Requires:
- Egress to huggingface.co. If the pinned Hub revision cannot be loaded and
  NEMO_GYM_EOG_LOCAL_TASKS is set to a complete EnterpriseOps-Gym task folder root
  (containing all eight <domain>/ task JSON dirs), those local tasks are converted instead.
- The MCP gym Docker containers only at RUN time, not at prepare time.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
HF_REVISION = os.getenv(
    "NEMO_GYM_EOG_HF_REVISION",
    "c8e538eae8a6205294f0a86675fefdc1fac408f6",
)
MODE = "oracle"
DOMAINS = ["calendar", "csm", "drive", "email", "hr", "itsm", "teams", "hybrid"]
EXPECTED_PUBLIC_ORACLE_COUNTS = {
    "calendar": 61,
    "csm": 103,
    "drive": 64,
    "email": 67,
    "hr": 102,
    "itsm": 103,
    "teams": 61,
    "hybrid": 88,
}

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
Tasks = List[Tuple[str, Dict[str, Any]]]


def _convert_domain(domain: str, tasks, out_file) -> int:
    snapshot_paths = [SNAPSHOTS_DIR / name for name in DOMAIN_SNAPSHOTS[domain]]
    gym_tools = load_snapshots(snapshot_paths)
    num_written = 0
    for task_id, task in tasks:
        row = convert_task(task, task_id, domain, MODE, gym_tools)
        metadata = row["verifier_metadata"]
        verifiers = metadata["verifiers"]
        if not verifiers:
            raise ValueError(f"EnterpriseOps task '{task_id}' has no verifiers")

        configured_gyms = {server_config["mcp_server_name"] for server_config in metadata["gym_servers_config"]}
        unknown_verifier_gyms = sorted(
            {
                verifier["gym_name"]
                for verifier in verifiers
                if verifier.get("gym_name") and verifier["gym_name"] not in configured_gyms
            }
        )
        if unknown_verifier_gyms:
            raise ValueError(
                f"EnterpriseOps task '{task_id}' has verifiers for unconfigured gyms: {unknown_verifier_gyms}"
            )

        expected_tools = {
            metadata["selected_tool_aliases"].get(tool_name, tool_name)
            for tool_name in metadata["selected_tools"]
            if tool_name not in metadata["unavailable_selected_tools"]
        } - set(metadata["restricted_tools"])
        missing_tools = sorted(expected_tools - set(metadata["tool_to_gym"]))
        if missing_tools:
            raise ValueError(f"EnterpriseOps task '{task_id}' is missing selected tool schemas: {missing_tools}")
        if not row["responses_create_params"]["tools"]:
            raise ValueError(f"EnterpriseOps task '{task_id}' has no usable tools")

        out_file.write(json.dumps(row) + "\n")
        num_written += 1
    return num_written


def _load_hf_tasks() -> Dict[str, Tasks]:
    return {domain: load_tasks_from_hf(HF_REPO_ID, MODE, domain, revision=HF_REVISION) for domain in DOMAINS}


def _load_local_tasks(local_tasks_root: str) -> Dict[str, Tasks]:
    root = Path(local_tasks_root)
    if not root.is_dir():
        raise ValueError(f"{LOCAL_TASKS_ENV_VAR} is not a directory: {root}")

    missing_domains = [domain for domain in DOMAINS if not (root / domain).is_dir()]
    if missing_domains:
        raise ValueError(
            f"Local EnterpriseOps task root is incomplete; missing domain directories: "
            f"{', '.join(missing_domains)}. Refusing to write a partial canonical benchmark."
        )
    return {domain: load_tasks_from_dir(root / domain) for domain in DOMAINS}


def _validate_tasks(tasks_by_domain: Dict[str, Tasks], *, enforce_public_counts: bool) -> None:
    seen_task_ids: Dict[str, str] = {}
    for domain in DOMAINS:
        tasks = tasks_by_domain[domain]
        if not tasks:
            raise ValueError(f"EnterpriseOps domain '{domain}' contains no tasks")
        for task_id, _ in tasks:
            if task_id in seen_task_ids:
                raise ValueError(
                    f"EnterpriseOps task ID '{task_id}' is duplicated in domains "
                    f"'{seen_task_ids[task_id]}' and '{domain}'"
                )
            seen_task_ids[task_id] = domain

        if enforce_public_counts:
            expected = EXPECTED_PUBLIC_ORACLE_COUNTS[domain]
            if len(tasks) != expected:
                raise ValueError(
                    f"Pinned EnterpriseOps public oracle split '{domain}' expected {expected} tasks, "
                    f"found {len(tasks)}"
                )


def prepare() -> Path:
    """Download and convert the EnterpriseOps-Gym public split. Returns the output path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    local_tasks_root = os.getenv(LOCAL_TASKS_ENV_VAR)
    try:
        tasks_by_domain = _load_hf_tasks()
        source = f"HuggingFace revision {HF_REVISION}"
        enforce_public_counts = True
    except Exception as e:
        if not local_tasks_root:
            raise RuntimeError(
                f"Could not load {HF_REPO_ID} revision {HF_REVISION} from HuggingFace "
                f"({type(e).__name__}: {e}). If this machine has no Hub egress, fetch the "
                f"dataset elsewhere or set {LOCAL_TASKS_ENV_VAR}=<complete EOG task root>."
            ) from e
        print(
            f"Pinned HuggingFace dataset could not be loaded ({type(e).__name__}); "
            f"falling back to local tasks at {local_tasks_root}"
        )
        tasks_by_domain = _load_local_tasks(local_tasks_root)
        source = f"local fallback {local_tasks_root}"
        enforce_public_counts = False

    _validate_tasks(tasks_by_domain, enforce_public_counts=enforce_public_counts)

    temporary_output = OUTPUT_FPATH.with_suffix(f"{OUTPUT_FPATH.suffix}.tmp")
    total = 0
    try:
        with open(temporary_output, "w") as out_file:
            for domain in DOMAINS:
                count = _convert_domain(domain, tasks_by_domain[domain], out_file)
                print(f"{domain}: {count} tasks ({source})")
                total += count
        temporary_output.replace(OUTPUT_FPATH)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    print(f"Wrote {total} tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
