# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bundle Agent-X tasks and images without changing prompts or labels."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.opencode_multimodal_benchmarks.agentx_prepare import validate_rows
from benchmarks.opencode_multimodal_benchmarks.data import sha256_file


def bundle(source: Path, destination: Path, *, sha256: str) -> dict:
    """Copy images by content hash and record the original and portable JSONL hashes."""
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Agent-X source checksum mismatch")
    rows = [json.loads(line) for line in data.splitlines()]
    validate_rows(rows)
    destination.mkdir(parents=True, exist_ok=False)
    media = destination / "media"
    media.mkdir()
    count = 0
    for row in rows:
        for message in row["responses_create_params"]["input"]:
            for content in message.get("content", []):
                if isinstance(content, dict) and content.get("type") == "input_image":
                    image = Path(content["image_url"])
                    name = sha256_file(image) + image.suffix.lower()
                    target = media / name
                    if not target.exists():
                        shutil.copyfile(image, target)
                        count += 1
                    content["image_url"] = name
    output = destination / "agentx.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {"source_sha256": sha256, "dataset_sha256": sha256_file(output), "samples": len(rows), "images": count}
    (destination / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(bundle(args.source, args.destination, sha256=args.sha256), indent=2))
