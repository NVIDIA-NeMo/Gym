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
"""Convert SWE-bench rows into sandbox_agent rows graded by anyswe. TODO: r2e etc"""

import argparse
import json
from pathlib import Path


def _as_list(v) -> list[str]:
    if isinstance(v, str):
        try:
            return list(json.loads(v))
        except (json.JSONDecodeError, TypeError):
            return [v] if v else []
    return list(v or [])


def _instance_image(container_formatter, instance_id: str) -> str:
    fmt = container_formatter[0] if isinstance(container_formatter, list) else container_formatter
    fmt = fmt or "swebench/sweb.eval.x86_64.{instance_id}"
    if fmt.startswith("docker://"):
        fmt = fmt[len("docker://") :]
    tag = instance_id.replace("__", "_1776_").lower()
    image = fmt.format(instance_id=tag)
    if ":" not in image.rsplit("/", 1)[-1]:
        image += ":latest"
    return image


def _benchmark(info: dict, override: str) -> str:
    if override:
        return override
    name = str(info.get("dataset_name") or "")
    if "R2E-Gym" in name:
        return "r2e-gym"
    if "Multilingual" in name:
        return "swe-bench-multilingual"
    return "swe-bench"


def convert(row: dict, benchmark: str = "") -> dict:
    info = row.get("problem_info") or row.get("verifier_metadata")
    if not isinstance(info, dict):
        meta = (row.get("responses_create_params") or {}).get("metadata")
        info = meta if isinstance(meta, dict) else row
    inst = (
        json.loads(info["instance_dict"])
        if isinstance(info.get("instance_dict"), str)
        else dict(info["instance_dict"])
    )
    instance_id = info["instance_id"]
    fail_to_pass = _as_list(inst.get("FAIL_TO_PASS") or inst.get("fail_to_pass"))
    pass_to_pass = _as_list(inst.get("PASS_TO_PASS") or inst.get("pass_to_pass"))
    if not fail_to_pass:
        raise ValueError(f"no test directives for {instance_id}")
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": inst.get("problem_statement") or info.get("problem_statement", "")}],
            "metadata": {
                "docker_image": _instance_image(info.get("container_formatter"), instance_id),
                "patch_workdir": "/testbed",
            },
        },
        "verifier_metadata": {
            "instance_id": instance_id,
            "benchmark": _benchmark(info, benchmark),
            "test_patch": inst.get("test_patch", ""),
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "instance_dict": inst,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", default="environments/swe/data/example.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--benchmark", default="", choices=["", "swe-bench", "swe-bench-multilingual", "r2e-gym"])
    args = parser.parse_args()

    rows = []
    for line in Path(args.input_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(convert(json.loads(line), args.benchmark))
        except Exception as e:
            print(f"skip: {e}")
        if args.limit and len(rows) >= args.limit:
            break

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
