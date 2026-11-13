# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Record fingerprints of Gym's bundled Workplace outputs before the agent runs."""

import argparse
import hashlib
import json
from pathlib import Path


def write_sample_hashes(gym_root, destination):
    # Read the installed Workplace samples and save their output fingerprints,
    # so verification can recognize copied samples even if the originals change.
    gym_root, destination = Path(gym_root), Path(destination)
    sources = [
        gym_root / layout / "workplace_assistant/data/example_rollouts.jsonl"
        for layout in ("resources_servers", "environments")
    ]
    sources = [path for path in sources if path.is_file()]
    if not sources:
        raise FileNotFoundError("Gym has no bundled Workplace sample rollouts")
    hashes = set()
    for path in sources:
        row_count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                output = row["response"]["output"]
                if not isinstance(output, list):
                    raise ValueError(f"Bundled sample output must be a list: {path}")
                # Match the verifier's existing hash of response.output. Response
                # IDs, timestamps and rewards do not distinguish a copied output.
                hashes.add(hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest())
                row_count += 1
        if not row_count:
            raise ValueError(f"Bundled sample file is empty: {path}")
    # Some Gym versions include both Workplace paths with identical samples.
    # Save each fingerprint once, in a stable order.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"bundled_output_hashes": sorted(hashes)}, indent=2) + "\n", encoding="utf-8")


def main():
    # Read the installed Gym and destination paths supplied by the Docker build.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gym_root", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    write_sample_hashes(args.gym_root, args.destination)


if __name__ == "__main__":
    main()
