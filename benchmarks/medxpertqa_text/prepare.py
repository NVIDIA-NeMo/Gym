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

"""Prepare MedXpertQA Text (test split) for NeMo Gym.

Downloads Text/test.jsonl from TsinghuaC3I/MedXpertQA at a pinned revision, checks its
SHA-256 and size, and writes the rows unchanged: id, question (with its answer choices),
options, label, medical_task, body_system, question_type. The dev split is not used. The
prompt comes from prompt_config at rollout time.
"""

import hashlib
import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "medxpertqa_text_benchmark.jsonl"
HF_REPO_ID = "TsinghuaC3I/MedXpertQA"
HF_REVISION = "7e7c465a68eb2b866926bfa59c8c9d17a8daba65"  # pragma: allowlist secret
HF_FILENAME = "Text/test.jsonl"
EXPECTED_SHA256 = "07ea977f7e326909b51c883b65f6ad7f399fc6db07408ef3630f4e3048ec71b2"  # pragma: allowlist secret
EXPECTED_NUM_ROWS = 2450
FIELDS = ("id", "question", "options", "label", "medical_task", "body_system", "question_type")


def convert(raw: bytes) -> list[dict]:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"{HF_FILENAME} has SHA-256 {digest}, expected {EXPECTED_SHA256}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != EXPECTED_NUM_ROWS:
        raise ValueError(f"{HF_FILENAME} has {len(rows)} rows, expected {EXPECTED_NUM_ROWS}")
    for row in rows:
        if set(row) != set(FIELDS) or row["label"] not in row["options"]:
            raise ValueError(f"Unexpected row {row.get('id')}: fields {sorted(row)}")
    return [{field: row[field] for field in FIELDS} for row in rows]


def prepare() -> Path:
    from huggingface_hub import hf_hub_download

    from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict

    source = hf_hub_download(
        HF_REPO_ID,
        HF_FILENAME,
        repo_type="dataset",
        revision=HF_REVISION,
        token=get_global_config_dict().get(HF_TOKEN_KEY_NAME),
    )
    rows = convert(Path(source).read_bytes())
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FPATH.open("w", encoding="utf-8") as stream:
        stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
