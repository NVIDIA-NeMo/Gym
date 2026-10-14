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
"""Prepare AweAI-Team/DeNovoSWE training data for NeMo Gym, from the Hub.

This is a training dataset, not an eval benchmark -- there is no fixed/frozen reference split,
and the output is meant to feed a training run, not to be scored against as a leaderboard.

Writes only the rows already known to be usable: `data/supported_instance_ids.txt` is the result
of a full-set 3x golden-patch sweep -- a row not on that list either never resolves, never
produced a verdict, or resolves inconsistently across passes (see `aggregate_golden_patch.py` and
the README), so it is not worth carrying into the training jsonl at all rather than shipping a
row no agent run could usefully score.

Filters mirror `responses_api_agents/swe_agents/denovoswe_dataprocessor.py`'s
`DeNovoSWEDataProcessor._select` defaults (the already-validated reference converter for this
dataset): require a non-empty `passed_ptp` (nothing to grade against otherwise) and skip
`submodule_uninitialized` rows (the image's checkout is known-incomplete for those).

    DENOVO_SWE_LIMIT=200 python -m resources_servers.denovo_swe.prepare_denovo_swe
"""

import json
import os
import random
from pathlib import Path


SHUFFLE_SEED = 0

DATASET_NAME = "AweAI-Team/DeNovoSWE"
DATASET_FILENAME = "denovoswe_public.jsonl"
SUPPORTED_IDS_FPATH = Path(__file__).parent / "data" / "supported_instance_ids.txt"
OUTPUT_FPATH = Path(__file__).parent / "data" / "denovo_swe_training.jsonl"

AGENT_REF = {"type": "responses_api_agents", "name": "denovo_swe_opencode_sandboxed_agent"}


def _load_supported_ids(fpath: Path) -> set[str]:
    return {line.strip() for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip()}


def _iter_raw_rows():
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=DATASET_NAME, filename=DATASET_FILENAME, repo_type="dataset"))
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_row(raw: dict) -> dict:
    instance_id = (raw.get("instance_id") or "").lower()
    coverage = ((raw.get("repo_line_coverage") or {}).get("coverage_percent")) or 0.0
    row = {
        "instance_id": instance_id,
        "repo": raw.get("repo") or f"{raw.get('user', '')}/{instance_id}",
        "github_url": raw.get("github_url", ""),
        "language": "python",
        "workdir": raw.get("workdir", ""),
        # Docker Hub tags are lowercased on push; the row's own image/image_url already reflects
        # that, so use it directly rather than re-deriving from instance_id.
        "image_ref": f"docker.io/{raw.get('image_url') or raw.get('image') or f'aweaiteam/denovoswe:{instance_id}'}",
        "base_commit": raw.get("parent_commit", ""),
        "patch": "",
        "test_patch": raw.get("test_patch", "") or "",
        "document": raw.get("document", "") or "",
        "problem_statement": raw.get("document", "") or "",
        "pypi_name": raw.get("pypi_name", "") or "",
        "import_names": raw.get("import_names") or [],
        "passed_ptp": raw.get("passed_ptp") or [],
        "failed_ptp": raw.get("failed_ptp") or [],
        "test_binary_archive_b64": raw.get("test_binary_archive_b64") or "",
        "expected_coverage_percent": float(coverage),
    }
    row["responses_create_params"] = {"input": [{"role": "user", "content": row["document"]}]}
    row["agent_ref"] = AGENT_REF
    return row


def prepare(supported_ids: set[str], limit: int = 0) -> Path:
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for raw in _iter_raw_rows():
        if limit and len(rows) >= limit:
            break
        instance_id = (raw.get("instance_id") or "").lower()
        if instance_id not in supported_ids:
            continue
        rows.append(_build_row(raw))

    random.Random(SHUFFLE_SEED).shuffle(rows)

    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} DeNovoSWE problems to {OUTPUT_FPATH} (shuffled, seed={SHUFFLE_SEED})")
    missing = len(supported_ids) - len(rows)
    if missing:
        print(f"  {missing} supported id(s) from {SUPPORTED_IDS_FPATH.name} were not found in the Hub dataset")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare(
        _load_supported_ids(SUPPORTED_IDS_FPATH),
        int(os.environ.get("DENOVO_SWE_LIMIT") or 0),
    )
