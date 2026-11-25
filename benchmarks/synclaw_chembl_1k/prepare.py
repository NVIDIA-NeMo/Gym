# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the SynClaw ChEMBL-1000 test set.

Clones SynClaw at the revision pinned in resources_servers/synclaw/pyproject.toml, downloads the public
SynFormer ChEMBL target set (1,000 molecules), strips isotope labels from the targets and converts
each target into a SynClaw task row. Requires ``git`` and ``uv``.
"""

import json
import tempfile
from pathlib import Path

from resources_servers.synclaw.prepare_db import clone_pinned_synclaw, synclaw, uv_run


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "synclaw_chembl_1k_benchmark.jsonl"
AGENT_NAME = "synclaw_chembl_1k_agent"
SOURCE_NAME = "chembl_filtered_1k.txt"

# Exact-match scoring keeps isotopes but no stock building block carries them, so a labelled
# target such as an [18F] radiotracer could never be reconstructed exactly. Runs SynClaw's own
# strip_isotopes helper; prints {stripped SMILES: source SMILES} for the targets it changed.
STRIP_ISOTOPES = """
import json, sys
from synclaw.core.chem import strip_isotopes
from synclaw.tasks.datasets import read_smiles_file

source, destination = sys.argv[1:]
targets = read_smiles_file(source)
stripped = [strip_isotopes(smiles) for smiles in targets]
with open(destination, "w") as handle:
    handle.writelines(smiles + "\\n" for smiles in stripped)
print(json.dumps({s: t for s, t in zip(stripped, targets) if s != t}))
"""


def prepare() -> Path:
    """Build the ChEMBL-1000 task file and return its path."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        checkout = work / "synclaw"
        clone_pinned_synclaw(checkout)
        synclaw(checkout, "download-public-data", "--name", "synformer-chembl-1k", "--dest-root", str(work))
        targets = work / "targets.txt"
        originals = uv_run(
            checkout,
            "python",
            "-c",
            STRIP_ISOTOPES,
            str(work / "data" / "synformer_test_sets" / SOURCE_NAME),
            str(targets),
        )
        tasks = work / "tasks.jsonl"
        stats = synclaw(checkout, "make-taskset", str(targets), "--dataset", "chembl", "--output", str(tasks))
        rows = [json.loads(line) for line in tasks.read_text().splitlines()]

    for row in rows:
        row["source_file"] = SOURCE_NAME
        if row["target_smiles"] in originals:
            row["original_target_smiles"] = originals[row["target_smiles"]]
        row["responses_create_params"] = {"input": []}
        row["agent_ref"] = {"type": "responses_api_agents", "name": AGENT_NAME}

    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FPATH.open("w") as handle:
        handle.writelines(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    print(
        f"Wrote {len(rows)} tasks to {OUTPUT_FPATH} "
        f"({stats['invalid']} invalid, isotopes stripped from {len(originals)})"
    )
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
