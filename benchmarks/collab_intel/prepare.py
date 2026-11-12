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
"""Prepare the Collaborative Intelligence benchmark (Zou group, arXiv 2611.02633) for NeMo Gym.

Downloads the public release at a pinned commit (code MIT, data CC BY 4.0, 7 scenarios CC BY-NC-SA 4.0: evaluation use
only) into data/repo/ (the agent runs the upstream harness from it in its own venv, since it needs openai>=3) and writes one row per released scenario of the scored set (benchmark/final_set.json) and test seed.
The 80 held-out scenarios are not public, and the 3 encrypted GPQA-based scenarios are skipped.
"""

import io
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path


COMMIT = "3de5f7d0a652c0b393342b4fbae59b78034034ea"  # pragma: allowlist secret
URL = f"https://github.com/zou-group/collaborative-intelligence/archive/{COMMIT}.tar.gz"
DATA_DIR = Path(__file__).parent / "data"
REPO_DIR = DATA_DIR / "repo"
MARKER = REPO_DIR / ".commit"  # the commit REPO_DIR was extracted from
OUTPUT_FPATH = DATA_DIR / "collab_intel_benchmark.jsonl"


def prepare() -> Path:
    if not MARKER.exists() or MARKER.read_text().strip() != COMMIT:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
        raw = urllib.request.urlopen(URL, timeout=300).read()
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            members = [m for m in tar.getmembers() if "/" in m.name]
            for m in members:
                m.name = m.name.split("/", 1)[1]
            tar.extractall(REPO_DIR, members=members, filter="data")
        MARKER.write_text(COMMIT + "\n")
    final = json.loads((REPO_DIR / "benchmark" / "final_set.json").read_text())
    rows = []
    for dim, items in final["dimensions"].items():
        for it in items:
            if not (REPO_DIR / it["path"] / "episode.yaml").exists():
                continue  # held out or encrypted (GPQA)
            gate = it.get("gate_path") or it["path"]
            for seed in it.get("seeds", final["seeds"]):
                rows.append(
                    {
                        "responses_create_params": {
                            "input": [{"role": "user", "content": it["path"].rsplit("/", 1)[1]}]
                        },
                        "dimension": dim,
                        "scenario_path": it["path"],
                        "gate_path": gate,
                        "gate_seed": (it.get("gate_seeds") or [seed])[0],
                        "seed": seed,
                    }
                )
    OUTPUT_FPATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
