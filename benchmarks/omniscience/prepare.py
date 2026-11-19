# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prepare AA-Omniscience v1.1 evaluation data for NeMo Gym.

Downloads the AA-Omniscience v1.1 public set from Hugging Face and converts it to Gym JSONL for the omniscience
resources server. AA-Omniscience v1.1 has 3,000 questions, and its public set is 300 of them. v1.1 removed the 600
questions of the v1 public set, so v1.1 scores are not comparable with v1 scores.

Output is raw data, with no prompts baked in. Prompts are applied at rollout time via prompt_config
(e.g., +prompt_config=benchmarks/omniscience/prompts/generation.yaml).

Source: https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public-v1.1
"""

import json
from pathlib import Path

from nemo_gym.config import get_hf_token


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "omniscience_benchmark.jsonl"

BENCHMARK_VERSION = "1.1"
HF_REPO_ID = "ArtificialAnalysis/AA-Omniscience-Public-v1.1"
HF_SPLIT = "test"
# Pin the revision so an upstream edit does not silently change scores. This revision corrects question 122 of the
# initial v1.1 upload.
HF_REVISION = "1f9cbcec472000b28f9782205f1ef9498b45bd21"  # pragma: allowlist secret
EXPECTED_NUM_ROWS = 300


def is_prepared_data_current(fpath: Path) -> bool:
    """Whether a cached prepared file holds this dataset revision, so that a cached v1 file is prepared again."""
    num_rows = 0
    try:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if json.loads(line).get("aa_omniscience_dataset_revision") != HF_REVISION:
                    return False
                num_rows += 1
    except (OSError, ValueError):
        return False
    return num_rows == EXPECTED_NUM_ROWS


def prepare() -> Path:
    """Download the AA-Omniscience v1.1 public set and convert it to Gym JSONL format."""
    from datasets import load_dataset

    print(f"Downloading {HF_REPO_ID} (revision {HF_REVISION[:7]}) from Hugging Face...")
    ds = load_dataset(HF_REPO_ID, split=HF_SPLIT, revision=HF_REVISION, token=get_hf_token())
    if len(ds) != EXPECTED_NUM_ROWS:
        raise ValueError(f"{HF_REPO_ID} at {HF_REVISION[:7]} has {len(ds)} rows, expected {EXPECTED_NUM_ROWS}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in ds:
        row = {
            "id": entry["question_id"],
            "domain": entry["domain"],
            "subtopic": entry["subtopic"],
            "question": entry["question"],
            "expected_answer": entry["answer"],
            "aa_omniscience_version": BENCHMARK_VERSION,
            "aa_omniscience_dataset_revision": HF_REVISION,
        }
        rows.append(json.dumps(row, ensure_ascii=False) + "\n")

    with open(OUTPUT_FPATH, "w", encoding="utf-8") as f:
        f.writelines(rows)
    # Collation's metrics sidecars describe the rows this file replaced (v1 in a workspace that collated it), and a
    # stale one fails the next collation as a conflict. The next collation writes them again.
    metrics_fpath = OUTPUT_FPATH.with_name(f"{OUTPUT_FPATH.stem}_metrics.json")
    for sidecar_fpath in (metrics_fpath, metrics_fpath.with_name(f"{metrics_fpath.stem}_conflict.json")):
        sidecar_fpath.unlink(missing_ok=True)

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
