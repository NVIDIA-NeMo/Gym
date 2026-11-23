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
"""Merge upstream VisionArena prompts with answers-only baselines from GitLab."""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from benchmarks.vision_arena_v1.build_validation import build_validation


DATASET_ID = "lmarena-ai/vision-arena-bench-v0.1"
DATASET_REVISION = "3abe18efa30d674a8903386b463ce278e7f41f2d"  # pragma: allowlist secret
BASELINE_ARTIFACT = "vision_arena_v1_baseline_answers.jsonl"


def prepare(*, baseline_responses: str | Path | None = None, output_path: str | Path | None = None) -> Path:
    output = Path(output_path) if output_path else Path(__file__).parent / "data/vision_arena_v1_validation.jsonl"
    if output.exists():
        if baseline_responses or output_path:
            raise FileExistsError(f"Output already exists: {output}; choose a new output_path to test preparation")
        return output
    if baseline_responses:
        responses = Path(baseline_responses).expanduser()
    else:
        from nemo_gym.config import DownloadJsonlDatasetGitlabConfig
        from nemo_gym.data import download_jsonl_dataset

        responses = Path(__file__).parent / "data" / BASELINE_ARTIFACT
        responses.parent.mkdir(parents=True, exist_ok=True)
        download_jsonl_dataset(
            DownloadJsonlDatasetGitlabConfig(
                dataset_name="vision_arena_v1_baseline_answers",
                version="0.0.1",
                artifact_fpath=BASELINE_ARTIFACT,
                output_fpath=str(responses),
            )
        )
    # Pin the source and require the exact fixed 486-answer evaluation subset.
    rows = [json.loads(line) for line in responses.read_text().splitlines() if line.strip()]
    if len(rows) != 486 or len({row["question_id"] for row in rows}) != 486:
        raise ValueError("Expected 486 distinct baseline answers")
    if any(not isinstance(row.get("baseline_answer"), str) or not row["baseline_answer"].strip() for row in rows):
        raise ValueError("Every baseline row must contain a nonempty baseline_answer")
    if any(row.get("source") != DATASET_ID or row.get("source_revision") != DATASET_REVISION for row in rows):
        raise ValueError("Baseline provenance does not match the pinned VisionArena dataset")
    questions = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        filename="data/train-00000-of-00001.parquet",
    )
    build_validation(questions=Path(questions), responses=responses, output=output, allow_partial=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-responses", help="Test a local answers-only artifact instead of GitLab")
    parser.add_argument("--output", help="Write to a separate path without replacing existing prompts")
    args = parser.parse_args()
    print(prepare(baseline_responses=args.baseline_responses, output_path=args.output))
