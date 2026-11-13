# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the public Parquet release without changing original task/image bytes."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def prepare(output: str | Path, revision: str | None = None) -> dict:
    import pyarrow.parquet as pq

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    repo = "Warrieryes/AgentVista"
    with urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/{repo}/revision/{revision or 'main'}", timeout=60
    ) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    files = sorted(item["rfilename"] for item in metadata["siblings"] if item["rfilename"].endswith(".parquet"))
    if not files:
        raise ValueError("No Parquet source files in the pinned release")
    rows, sources = [], {}
    for name in files:
        path = output / "source" / Path(name).name
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(".partial")
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}", temporary)
            temporary.replace(path)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for original in pq.read_table(path).to_pylist():
            index = len(rows)
            record = {key: value for key, value in original.items() if key != "images"}
            record["question_id"] = f"agentvista-{index:03d}"
            record["solution"] = original["answer"]
            record["category"] = original["domain"]
            images = []
            for image_index, image in enumerate(original["images"]):
                raw = image["bytes"]
                if not raw:
                    raise ValueError("Expected embedded original image bytes")
                relative = f"images/{index:03d}_{image_index:02d}.image"
                target = output / relative
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(raw)
                images.append(relative)
            record["images"] = images
            rows.append(record)
    destination = output / "val.json"
    destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    receipt = {
        "repo": repo,
        "revision": revision,
        "tasks": len(rows),
        "images": sum(len(row["images"]) for row in rows),
        "source_sha256": sources,
        "dataset_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "license": "CC-BY-4.0",
        "source_split": "train",
        "note": "Upstream split name is not authorization to train on held-out benchmark tasks.",
    }
    (output / "preparation_receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.revision), indent=2))
