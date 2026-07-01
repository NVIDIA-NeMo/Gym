# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare the shared 113-task DeepSWE benchmark ID manifest."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from responses_api_agents.deep_swe.benchmark import ensure_checkout, task_tree_digest
from responses_api_agents.deep_swe.secure_paths import default_private_path, ensure_private_directory


BENCHMARK_GIT_URL = "https://github.com/datacurve-ai/deep-swe.git"
AA_V1_COMMIT = "c33fa70e68d11d85f9e58abcd5d78643705e916e"  # pragma: allowlist secret
CURRENT_V1_1_COMMIT = "8cae5984d5dd0ee37445beff0e928dc10c331116"  # pragma: allowlist secret
EXPECTED_TASK_COUNT = 113
PROFILE_COMMITS = {
    "deep_swe_aa_v1": AA_V1_COMMIT,
    "deep_swe_v1_1": CURRENT_V1_1_COMMIT,
}
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_PATH = DATA_DIR / "deep_swe_benchmark_validation.jsonl"
CACHE_DIR = default_private_path("checkouts")
INPUT_MESSAGE = "Run the pinned DeepSWE task identified by verifier_metadata."


def _task_ids(checkout: Path) -> tuple[str, ...]:
    tasks_dir = checkout / "tasks"
    task_ids = tuple(sorted(path.name for path in tasks_dir.iterdir() if (path / "task.toml").is_file()))
    if len(task_ids) != EXPECTED_TASK_COUNT:
        raise ValueError(f"Expected {EXPECTED_TASK_COUNT} DeepSWE tasks, found {len(task_ids)} in {checkout}")
    return task_ids


async def _resolve_profiles() -> tuple[tuple[str, ...], dict[str, dict[str, str]]]:
    task_ids: tuple[str, ...] | None = None
    provenance: dict[str, dict[str, str]] = {}
    for profile, commit in PROFILE_COMMITS.items():
        checkout = await ensure_checkout(
            git_url=BENCHMARK_GIT_URL,
            commit=commit,
            cache_dir=ensure_private_directory(CACHE_DIR),
            benchmark_path=None,
            expected_task_count=EXPECTED_TASK_COUNT,
        )
        profile_task_ids = _task_ids(checkout)
        if task_ids is None:
            task_ids = profile_task_ids
        elif profile_task_ids != task_ids:
            raise ValueError(f"DeepSWE task IDs differ between pinned profiles; first mismatch is in {profile}")
        provenance[profile] = {
            "benchmark_git_commit": commit,
            "task_tree_sha256": task_tree_digest(checkout),
        }
    if task_ids is None:
        raise RuntimeError("No DeepSWE benchmark profiles are configured")
    return task_ids, provenance


def _record(task_id: str, provenance: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": INPUT_MESSAGE,
                }
            ]
        },
        "verifier_metadata": {
            "task_id": task_id,
            "benchmark_profiles": provenance,
        },
    }


def _write_jsonl(
    output_path: Path,
    task_ids: tuple[str, ...],
    provenance: dict[str, dict[str, str]],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for task_id in task_ids:
                stream.write(json.dumps(_record(task_id, provenance), ensure_ascii=False, sort_keys=True))
                stream.write("\n")
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def prepare() -> Path:
    """Write the deterministic ID-only manifest shared by the v1 and v1.1 profiles."""
    task_ids, provenance = asyncio.run(_resolve_profiles())
    return _write_jsonl(OUTPUT_PATH, task_ids, provenance)


if __name__ == "__main__":
    prepare()


__all__ = ["prepare"]
