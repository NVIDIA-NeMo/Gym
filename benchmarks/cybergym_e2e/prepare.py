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
"""Prepare the CyberGym-E2E benchmark rows and fetch the selected tasks' assets.

Each row is a task *identity* -- `task_id` (`project/arvo_id`), `project`, and `mode` -- plus an empty
`responses_create_params.input`: this is an external-agent-loop benchmark, so the agent server builds
the real prompt and drives the episode itself; the row only names the task.

prepare() also downloads ONLY the selected tasks into a gitignored managed dir (data/assets/), so no
manual checkout is needed:
  - the public GitHub repo's code + per-task configs/scripts/patch (one pinned tarball), and
  - the gated HuggingFace `src.tgz`/`poc.bin`/`crash.log` (needs $HF_TOKEN and accepted access).
The harness reads this managed dir; those assets are never committed (license + answer-leak).

Task selection rewrites the single configured dataset file (Gym reads that path). An unknown id is a
hard error, never silently dropped (Zen: "Errors should never pass silently"):

    gym eval prepare --benchmark cybergym_e2e
    gym eval prepare --benchmark cybergym_e2e \\
        '++prepare_script_args.task_names=[jq/arvo_64574,yara/arvo_11945]'
"""

import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
TASKS_FILE = BENCHMARK_DIR / "tasks.txt"
# The single configured dataset path; selection rewrites THIS file (never a separate one), so a subset
# cannot silently masquerade as the full set -- the row count in the file is the set that will run.
OUTPUT_PATH = DATA_DIR / "cybergym_e2e_benchmark.jsonl"
# Managed, gitignored asset tree the harness reads (code + configs from GitHub, gated data from HF).
ASSETS_DIR = DATA_DIR / "assets"

# Upstream sources. Pin exact revisions before publishing (these mirror manifest.yaml's provenance).
REPO = "sunblaze-ucb/cybergym-e2e"
REPO_REF = os.environ.get("CYBERGYM_REPO_REF", "d0ae0f02e76ebc084389bf21e755f4b41ac9765a")
HF_DATASET = "sunblaze-ucb/cybergym-e2e"
HF_REVISION = os.environ.get("CYBERGYM_HF_REVISION", "3aee406e4e915e32527fea16de7f002630fa8c76")

# Only e2e is implemented end-to-end (harness run.py + the agent adapter). `patch-only` is defined by
# upstream CyberGym but not wired here yet, so it is intentionally not an accepted mode.
VALID_MODES = ("e2e",)

# The exact asset files the harness reads for a task (see harness/validate.py + template.py). A
# partial/stale download missing any of these would only fail deep inside a run, so prepare verifies
# them up front -- never publishing a dataset row for a task whose assets aren't actually present.
REPO_SHARED = ("scripts/validate.py", "scripts/install_validate_deps.sh")
REPO_PER_TASK = ("config.toml", "prepare.sh", "compile.sh", "run_poc.sh", "test.sh", "patch.diff")
HF_PER_TASK = ("src.tgz", "poc.bin")


def _missing_assets(selected: list[str], hf: bool = True) -> list[str]:
    """Asset files that should exist for `selected` but don't (empty == all present). `hf=False` checks
    only the repo side -- used to decide whether the cached repo tarball is complete."""
    missing = [s for s in REPO_SHARED if not (ASSETS_DIR / s).exists()]
    for t in selected:
        proj, task = t.split("/", 1)
        pdir = ASSETS_DIR / "projects" / proj
        for rel in ["project.toml", *(f"{task}/{f}" for f in REPO_PER_TASK)]:
            if not (pdir / rel).exists():
                missing.append(f"projects/{proj}/{rel}")
        if hf:
            for f in HF_PER_TASK:
                if not (ASSETS_DIR / "data" / "projects" / proj / task / f).exists():
                    missing.append(f"data/projects/{proj}/{task}/{f}")
    return missing


def _ensure_repo(selected: list[str]) -> None:
    """Download the public repo's code + per-task configs/scripts/patch (scripts/, projects/).

    A `.repo_ref` marker records the pinned revision; we re-download when it is missing, stale, or the
    cached tree is missing any selected task's files (a bare existence check can accept a partial or
    old-revision tree)."""
    marker = ASSETS_DIR / ".repo_ref"
    if (marker.exists() and marker.read_text().strip() == REPO_REF
            and not _missing_assets(selected, hf=False)):
        return
    url = f"https://github.com/{REPO}/archive/{REPO_REF}.tar.gz"
    print(f"[prepare] fetching repo {REPO}@{REPO_REF} ...")
    with tempfile.TemporaryDirectory() as td:
        tgz = Path(td) / "repo.tgz"
        urllib.request.urlretrieve(url, tgz)  # noqa: S310 (pinned public GitHub archive)
        with tarfile.open(tgz) as tf:
            tf.extractall(td, filter="data")  # filter guards against path traversal
        root = next(p for p in Path(td).iterdir() if p.is_dir())
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        for sub in ("scripts", "projects"):
            dst = ASSETS_DIR / sub
            if dst.exists():
                shutil.rmtree(dst)  # replace any stale/partial copy
            shutil.copytree(root / sub, dst)
    marker.write_text(REPO_REF)  # written only after a full, successful extract


def _ensure_hf_data(selected: list[str]) -> None:
    """Download the gated src.tgz/poc.bin/crash.log for the selected tasks into ASSETS_DIR/data/.

    snapshot_download pins the revision and verifies per-file, re-fetching anything missing or changed,
    so it is always run (a bare src.tgz existence check would accept a partial/stale download)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("huggingface_hub is required to fetch the gated dataset") from e
    patterns = [f"projects/{t.split('/', 1)[0]}/{t.split('/', 1)[1]}/*" for t in selected]
    print(f"[prepare] fetching gated HF data for {len(selected)} task(s) (needs $HF_TOKEN + accepted access)...")
    snapshot_download(repo_id=HF_DATASET, repo_type="dataset", revision=HF_REVISION,
                      allow_patterns=patterns, local_dir=str(ASSETS_DIR / "data"),
                      token=os.environ.get("HF_TOKEN"))


def _read_task_list(path: Path) -> list[str]:
    """Read `project/arvo_id` ids from a file, skipping blanks and '#' comments."""
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def _row(task_id: str, mode: str) -> dict:
    project, sep, _ = task_id.partition("/")
    if not sep:
        raise ValueError(f"task id is not in 'project/arvo_id' form: {task_id!r}")
    return {
        "task_id": task_id,
        "task_name": task_id,
        "project": project,
        "mode": mode,
        # The external agent loop constructs the prompt; the row carries no model-visible input.
        "responses_create_params": {"input": []},
    }


def prepare(task_names=None, mode: str = "e2e") -> Path:
    """Write benchmark rows to the configured path and return it. `task_names` (list) selects a subset
    of tasks.txt; with none, all available tasks are written."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    available = _read_task_list(TASKS_FILE)
    available_set = set(available)

    if task_names is None:
        selected = available
    else:
        unknown = [t for t in task_names if t not in available_set]
        if unknown:
            raise ValueError(
                f"unknown task id(s) {unknown}; not in {TASKS_FILE.name}. "
                f"Available: {sorted(available_set)}")
        seen, selected = set(), []  # preserve caller order, drop duplicates
        for t in task_names:
            if t not in seen:
                seen.add(t)
                selected.append(t)

    # Fetch the selected tasks' assets FIRST, then verify every required file is actually present, so
    # the dataset file never ends up pointing at assets that are missing or only partially downloaded.
    _ensure_repo(selected)
    _ensure_hf_data(selected)
    missing = _missing_assets(selected)
    if missing:
        raise SystemExit(
            "[prepare] required assets missing after fetch (unknown task id, or an incomplete/"
            "access-denied download); not writing the dataset:\n  " + "\n  ".join(missing))

    # Write atomically: build the full file in a temp sibling, then rename over the target, so an
    # interrupted write can't leave a truncated dataset that later reads as the real run set.
    tmp = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    with open(tmp, "w") as f:
        for task_id in selected:
            f.write(json.dumps(_row(task_id, mode)) + "\n")
    os.replace(tmp, OUTPUT_PATH)
    print(f"Wrote {len(selected)} task identit{'y' if len(selected) == 1 else 'ies'} to {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    prepare()
