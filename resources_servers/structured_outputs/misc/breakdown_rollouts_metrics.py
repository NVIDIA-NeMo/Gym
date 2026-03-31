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
import argparse
import json
from collections import defaultdict
from statistics import mean


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pct(num, den):
    return f"{100 * num / den:.1f}%" if den else "N/A"


def fmt_count(num, den):
    return f"{num}/{den} ({pct(num, den)})"


def print_section(label, rows):
    n = len(rows)
    if n == 0:
        return
    rewards = [r.get("reward", 0.0) for r in rows]
    n_pass = sum(1 for r in rewards if r == 1.0)

    print(f"  {label}")
    print(f"    n = {n}")
    print(f"    pass: {fmt_count(n_pass, n)}")
    print(f"    mean reward: {mean(rewards):.4f}")
    print()


def main(args):
    rows = list(iter_jsonl(args.in_path))
    if not rows:
        print("No rows found.")
        return

    by_schema_type = defaultdict(list)
    for r in rows:
        st = r.get("schema_type", "unknown")
        by_schema_type[st].append(r)

    w = max(60, len(args.in_path) + 4)
    print("=" * w)
    print(f"  {args.in_path}")
    print("=" * w)
    print()

    print_section("OVERALL", rows)

    print("-" * w)
    print()

    for st in sorted(by_schema_type):
        print_section(f"schema_type={st}", by_schema_type[st])

    if args.by_fields:
        print("-" * w)
        print("  Breakdown by schema_fields_count")
        print("-" * w)
        print()

        by_fields = defaultdict(list)
        for r in rows:
            fc = r.get("schema_fields_count", "unknown")
            by_fields[fc].append(r)

        for fc in sorted(by_fields, key=lambda x: (isinstance(x, str), x)):
            print_section(f"fields={fc}", by_fields[fc])

    print("=" * w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--in-path", required=True)
    parser.add_argument("--by-fields", action="store_true", help="Also break down by schema_fields_count")
    args = parser.parse_args()
    main(args)
