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
import hashlib
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from resources_servers.swarmbench.app import CAPABILITY_ONLY, LATEST_VERSION, TRACKS, TYPE_DIRS, generate, row_tools
from resources_servers.swarmbench.setup_vendor import ensure_family_b_repos


BENCHMARK_DIR = Path(__file__).parent
# v2.1: three tiers per track, each an (n, wall-clock budget) setting, with the canary. instances.json is v1.
MANIFEST_FPATH = BENCHMARK_DIR / "manifest_v2.json"
# The promoted types of the 100-type pilot, at each type's own tiers.
TYPES_MANIFEST_FPATH = BENCHMARK_DIR / "manifest_types.json"
TYPE_SEEDS = {"test": range(31000, 31003), "dev": range(39000, 39001)}  # disjoint from build and calibration seeds
GRACE_S = {"easy": 600, "medium": 600, "hard": 900}
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "swarmbench_benchmark.jsonl"
# The line SwarmBench's budgeted runner (recal/run.py) appends to the prompt.
BUDGET_LINE = (
    "Time budget: you have {b} seconds of wall-clock time for this whole task. Every tool result ends with the time "
    "used so far. When the budget is used up, the tools stop working and you get {g} more seconds to give the final "
    "answer, then the run is stopped. Only what your final answer (or, where the task grades the files you edit, the "
    "files) contains at that point counts, and partial results earn partial credit, so make sure you answer in time. "
    "Work on your own until you have the final answer: nobody will reply to progress updates or questions."
)


def instance_hash(prompt: str, out_dir: str) -> str:
    """SwarmBench's frozen-instance hash: the prompt, then every generated file by relative path."""
    h = hashlib.sha256(prompt.encode())
    for root, dirs, files in os.walk(out_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            p = os.path.join(root, f)
            h.update(b"\0" + os.path.relpath(p, out_dir).encode() + b"\0")
            with open(p, "rb") as fh:
                h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def build_row(instance: dict, canary: bool = False) -> dict:
    # Worlds are large (up to ~60 MB), so rows keep only the prompt; the server regenerates the world per rollout.
    meta = dict(instance, canary=canary)
    with tempfile.TemporaryDirectory(prefix="swarmbench-prepare-") as out_dir:
        task = generate(meta["track"], meta["n"], meta["seed"], out_dir, meta.get("track_version", 1), canary)
        if "sha256" in meta and instance_hash(task["prompt"], out_dir) != meta["sha256"]:
            raise ValueError(f"{meta.get('id')}: generated world does not match its frozen sha256")
    prompt = task["prompt"]
    if meta.get("budget_s"):
        prompt += "\n\n" + BUDGET_LINE.format(b=int(meta["budget_s"]), g=int(meta["grace_s"]))
    # The tools ride along for agents that read them from the row (simple_agent, swarm_agent); MCP agents ignore them.
    tools = row_tools(meta["track"], meta.get("track_version", 1))
    return {"prompt": prompt, "responses_create_params": {"tools": tools}, "verifier_metadata": meta}


def prepare(manifest: str | Path = MANIFEST_FPATH, split: str = "test") -> Path:
    """Rows of one split (test, or dev: one instance per track and tier). Every world is checked against its hash."""
    data = json.loads(Path(manifest).read_text())
    instances = [i for i in data["instances"] if i["split"] == split]
    ensure_family_b_repos()
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(partial(build_row, canary="canary" in data), instances))
    out = OUTPUT_FPATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} {split} instances of SwarmBench {data['version']} to {out} (all hashes match)")
    return out


def example_rows(tracks=("corpus", "search", "dag", "schedule", "rootcause")) -> list[dict]:
    """data/example.jsonl of the resources server: n=16 worlds at the latest track versions, 300 s budget."""
    rows = []
    for track in tracks:
        meta = {"track": track, "n": 16, "seed": 0, "track_version": LATEST_VERSION.get(track, 1), "tier": "example"}
        row = build_row(meta | {"budget_s": 300, "grace_s": 300}, canary=True)
        row["responses_create_params"]["input"] = [{"role": "user", "content": row.pop("prompt")}]
        rows.append(row)
    return rows


def build_types_manifest() -> Path:
    """Freeze the types manifest: every (type, tier) at TYPE_SEEDS, with the same hash as the v2 manifest."""
    instances = []
    for name in sorted(TYPE_DIRS):
        for tier, (n, budget_s, *_) in TRACKS[name].TIERS.items():
            for split, seeds in TYPE_SEEDS.items():
                for k, seed in enumerate(seeds):
                    instances.append(
                        {
                            "id": f"{name}-{tier}-{'dev-' if split == 'dev' else ''}{k:03d}",
                            "track": name,
                            "tier": tier,
                            "split": split,
                            "n": n,
                            "seed": seed,
                            "budget_s": budget_s,
                            "grace_s": GRACE_S[tier],
                            "track_version": 1,
                        }
                    )
    with ProcessPoolExecutor() as pool:
        for instance, digest in zip(instances, pool.map(_hash_instance, instances)):
            instance["sha256"] = digest
    meta = {
        "name": "SwarmBench types",
        "version": "types-1",
        "capability_only": sorted(CAPABILITY_ONLY),
        "hash": "as manifest_v2.json (prompt plus every generated file); the canary is in every type's prompt",
    }
    TYPES_MANIFEST_FPATH.write_text(json.dumps(meta | {"instances": instances}, indent=1) + "\n")
    return TYPES_MANIFEST_FPATH


def _hash_instance(instance: dict) -> str:
    with tempfile.TemporaryDirectory(prefix="swarmbench-hash-") as out_dir:
        task = generate(instance["track"], instance["n"], instance["seed"], out_dir)
        return instance_hash(task["prompt"], out_dir)


if __name__ == "__main__":
    prepare()
