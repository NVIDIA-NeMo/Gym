# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#!/usr/bin/env python3
"""Convert a ds2/ds3-style verified jsonl to Gym-ready format for format_verification.

Transforms:
  - messages (strip assistant) -> responses_create_params.input
  - verifier dict passed through as-is

Works for both ds2 (freeform formatting, verifier.type=regex) and
ds3 (citation format, verifier.type=string_match).

Usage:
  python misc/prepare_data.py --input /path/to/ds2_verified.jsonl --output data/ds2_train.jsonl
  python misc/prepare_data.py --input /path/to/ds3_verified.jsonl --output data/ds3_train.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def convert_record(record: dict) -> dict:
    input_msgs = [m for m in record.get("messages", []) if m.get("role") != "assistant"]

    return {
        "responses_create_params": {"input": input_msgs},
        "verifier": record["verifier"],
    }


def main():
    parser = argparse.ArgumentParser(description="Convert verified ds2/ds3 jsonl to Gym-ready format")
    parser.add_argument("--input", "-i", required=True, help="Path to ds2_verified.jsonl or ds3_verified.jsonl")
    parser.add_argument("--output", "-o", required=True, help="Path to write Gym-ready jsonl")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        print(f"ERROR: {src} not found")
        sys.exit(1)

    dst.parent.mkdir(parents=True, exist_ok=True)

    total, written, skipped = 0, 0, 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            try:
                gym_record = convert_record(record)
                fout.write(json.dumps(gym_record, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                rid = record.get("metadata", {}).get("record_id", f"line_{total}")
                print(f"  SKIP  {rid}: {e}")
                skipped += 1

    print(f"{written}/{total} written to {dst}  ({skipped} skipped)")


if __name__ == "__main__":
    main()
