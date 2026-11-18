# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage a private release candidate locally; never upload or replace a published release."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.partner_if import prepare
from benchmarks.partner_if.replay_assets import read_bundle


HARNESS = {
    "version": "gymnasium-v1",
    "max_parent_turns": 6,
    "max_policy_calls": 8,
    "max_delegate_turns": 6,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 8192,
    "wire_format": "gym-responses-to-chat",
}


def stage(rows_path: Path, assets_path: Path, output: Path, *, version: str = "2.1.8") -> Path:
    """Preserve every other row verbatim and mark pif_029's changed harness for a new release."""
    if version == prepare.VERSION or not version.strip():
        raise ValueError("A changed harness requires a new registry version")
    source = rows_path.read_bytes()
    assets = assets_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != prepare.SHA256:
        raise ValueError("Candidate input differs from the pinned source release")
    if hashlib.sha256(assets).hexdigest() != prepare.REPLAY_ASSET_SHA256:
        raise ValueError("Candidate assets differ from the pinned source release")
    members = read_bundle(assets_path)
    lines = source.decode().split("\n")
    migrated = []
    matches, row_count = 0, 0
    for line in lines:
        if not line.strip():
            migrated.append(line)
            continue
        row_count += 1
        row = json.loads(line)
        if row["verifier_metadata"]["partner_if"]["task_id"] == "pif_029":
            if row["responses_create_params"] != json.loads(members["actor-params.json"]):
                raise ValueError("Candidate actor differs from frozen assets")
            row["recorded_context_harness"] = dict(HARNESS)
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            matches += 1
        migrated.append(line)
    if matches != 1 or row_count != prepare.ROW_COUNT:
        raise ValueError("Candidate requires the complete release with exactly one pif_029")
    content = "\n".join(migrated).encode()
    manifest = {
        "status": "local_candidate_not_published",
        "dataset_name": prepare.DATASET_NAME,
        "proposed_version": version,
        "source_version": prepare.VERSION,
        "source_sha256": prepare.SHA256,
        "row_count": row_count,
        "artifacts": {
            prepare.ARTIFACT: hashlib.sha256(content).hexdigest(),
            prepare.REPLAY_ASSET_ARTIFACT: hashlib.sha256(assets).hexdigest(),
        },
        "recorded_context_harness": HARNESS,
    }
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / prepare.ARTIFACT).write_bytes(content)
    (output / prepare.REPLAY_ASSET_ARTIFACT).write_bytes(assets)
    manifest_path = output / "candidate.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=prepare.OUTPUT_FPATH)
    parser.add_argument("--assets", type=Path, default=prepare.REPLAY_ASSET_FPATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="2.1.8")
    args = parser.parse_args()
    print(stage(args.input_jsonl, args.assets, args.output_dir, version=args.version))
