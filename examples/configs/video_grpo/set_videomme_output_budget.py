# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add the explicit8192 RL output cap to a completed mixture, without media/inventory reads."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


VIDEO_AGENT = "videomme_benchmark_simple_agent"
AGENTIC = {"agentic_vbench_agent", "claweval_agent"}


def finalize(base_root, expected_base_manifest_sha256, output_root):
    raw_manifest = (base_root / "manifest.json").read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != expected_base_manifest_sha256:
        raise ValueError("Base manifest differs from its actual completed authority")
    manifest = json.loads(raw_manifest)
    output_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    counts = Counter()
    old_combined, old_video, old_agentic = (hashlib.sha256() for _ in range(3))
    new_combined, new_video = (hashlib.sha256() for _ in range(2))
    with (
        (base_root / "three_source_train.jsonl").open("rb") as source,
        (output_root / "three_source_train.jsonl").open("xb") as combined,
        (output_root / "videomme_train.jsonl").open("xb") as video,
    ):
        os.chmod(output_root / "three_source_train.jsonl", 0o600)
        os.chmod(output_root / "videomme_train.jsonl", 0o600)
        for line in source:
            old_combined.update(line)
            row = json.loads(line)
            name = row.get("task_source")
            counts[name] += 1
            if name == VIDEO_AGENT:
                old_video.update(line)
                params = row["responses_create_params"]
                if params.get("max_output_tokens") not in (None, 8192):
                    raise ValueError("Unexpected preexisting VideoMME output budget")
                params["max_output_tokens"] = 8192
                line = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                video.write(line)
                new_video.update(line)
            elif name in AGENTIC:
                old_agentic.update(line)
            else:
                raise ValueError("Unexpected task source in completed mixture")
            combined.write(line)
            new_combined.update(line)
        if counts != manifest["task_source_counts"]:
            raise ValueError("Completed mixture row counts changed")
        if old_combined.hexdigest() != manifest["outputs"]["three_source_train.jsonl"]["sha256"]:
            raise ValueError("Completed base mixture hash changed")
        if old_video.hexdigest() != manifest["outputs"]["videomme_train.jsonl"]["sha256"]:
            raise ValueError("Completed base VideoMME row order/hash changed")
        if old_agentic.hexdigest() != manifest["sources"]["AVB_Claw"]["sha256"]:
            raise ValueError("Original AVB/Claw bytes or order changed")
        for stream in (combined, video):
            stream.flush()
            os.fsync(stream.fileno())
    manifest["Video_MME_max_output_tokens"] = 8192
    manifest["output_budget_scope"] = (
        "Initial RL validations; deliberately differs from supplied VideoMME evaluator32000"
    )
    manifest["base_preparation"] = {"directory": str(base_root), "manifest_sha256": expected_base_manifest_sha256}
    manifest["outputs"] = {
        name: {"sha256": digest.hexdigest(), "bytes": (output_root / name).stat().st_size}
        for name, digest in [("three_source_train.jsonl", new_combined), ("videomme_train.jsonl", new_video)]
    }
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with (output_root / "manifest.json").open("xb") as stream:
        os.chmod(output_root / "manifest.json", 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    result = finalize(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "rows": result["rows"],
                "task_source_counts": result["task_source_counts"],
                "Video_MME_max_output_tokens": result["Video_MME_max_output_tokens"],
                "outputs": result["outputs"],
            }
        )
    )


if __name__ == "__main__":
    main()
