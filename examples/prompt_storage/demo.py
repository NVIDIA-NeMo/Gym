# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encode saved rollouts, expand them, and assert that every JSON value survived.

From the repository root:
    PYTHONPATH=. python examples/prompt_storage/demo.py SOURCE.jsonl NEW_OUTPUT_DIR
"""

import argparse
import gzip
import json
from pathlib import Path

from nemo_gym.atif_json import strict_json_loads
from nemo_gym.rollout_prompt_storage import decode_rollout, encode_rollout


def dump(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.read_bytes()
    documents = []
    restored = []
    originals = []
    operations = []
    for line in source.splitlines():
        if not line.strip():
            continue
        row = strict_json_loads(line)
        original = dump(row)
        document = encode_rollout(row)
        # Test the actual JSON serialization boundary, not only Python objects.
        decoded = decode_rollout(strict_json_loads(dump(document)))
        if dump(decoded) != original or dump(row) != original:
            raise AssertionError("round trip changed the row or encoding mutated its input")
        documents.append(dump(document))
        restored.append(dump(decoded))
        originals.append(original)
        calls = document["rollout"].get("ng_trajectory", {}).get("model_calls", [])
        for index, call in enumerate(calls):
            request = call.get("request") or {}
            for field in ("messages", "input"):
                if field in request:
                    encoded = request[field]
                    operations.append(
                        {
                            "row": len(originals) - 1,
                            "call": index,
                            "field": field,
                            "operation": "append" if "append_to" in encoded else "full",
                            **({"base_call": encoded["append_to"]} if "append_to" in encoded else {}),
                        }
                    )
    original_bytes = b"\n".join(originals) + b"\n"
    encoded_bytes = b"\n".join(documents) + b"\n"
    decoded_bytes = b"\n".join(restored) + b"\n"
    summary = {
        "rows": len(originals),
        "round_trip_exact": True,
        "source_bytes": len(source),
        "original_minified_bytes": len(original_bytes),
        "encoded_bytes": len(encoded_bytes),
        "original_gzip_bytes": len(gzip.compress(original_bytes, mtime=0)),
        "encoded_gzip_bytes": len(gzip.compress(encoded_bytes, mtime=0)),
        "operations": operations,
    }
    # Never overwrite either the input or an earlier experiment.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "encoded.jsonl").write_bytes(encoded_bytes)
    (args.output / "decoded.jsonl").write_bytes(decoded_bytes)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
