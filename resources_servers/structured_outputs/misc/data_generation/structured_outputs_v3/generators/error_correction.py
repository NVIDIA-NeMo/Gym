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
"""Category 6: Error correction -- fix corrupted structured output."""

import json
import random
from typing import Any, Dict, List

from templates import (
    ALL_FORMATS,
    CORRECTION_TEMPLATES,
    FORMAT_NAMES,
    make_gym_record,
    represent_schema,
)


def _corrupt_output(output_str: str, schema_dict: Dict, rng: random.Random) -> str:
    """Apply a random corruption to the output string."""
    corruption = rng.choice(["drop_field", "wrong_type", "extra_field", "syntax"])

    if corruption == "drop_field":
        try:
            obj = json.loads(output_str)
            if isinstance(obj, dict) and len(obj) > 1:
                key = rng.choice(list(obj.keys()))
                del obj[key]
                return json.dumps(obj, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass

    elif corruption == "wrong_type":
        try:
            obj = json.loads(output_str)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, int):
                        obj[k] = str(v)
                        return json.dumps(obj, ensure_ascii=False)
                    if isinstance(v, str) and v.isdigit():
                        obj[k] = int(v)
                        return json.dumps(obj, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass

    elif corruption == "extra_field":
        try:
            obj = json.loads(output_str)
            if isinstance(obj, dict):
                obj["_unexpected_field"] = "should_not_be_here"
                return json.dumps(obj, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass

    elif corruption == "syntax":
        if output_str.startswith("{"):
            pos = rng.randint(len(output_str) // 4, 3 * len(output_str) // 4)
            return output_str[:pos] + ",,," + output_str[pos:]
        return output_str + "\n<unexpected>"

    return output_str + "\n/* corrupted */"


def generate_error_correction(
    records: List[Dict[str, Any]],
    rng: random.Random,
    samples_per_record: int = 3,
    target_formats: List[str] = ALL_FORMATS,
    max_samples: int = 1000,
) -> List[Dict[str, Any]]:
    results = []
    json_records = [r for r in records if r.get("target_output_format") == "json"]
    if not json_records:
        json_records = records

    for record in json_records:
        schema_dict = record["_json_schema"]
        rid = record.get("_record_id", "unknown")
        original_output = record.get("target_output", "")
        source_fmt = record.get("target_output_format", "json")

        if not original_output:
            continue

        for _ in range(samples_per_record):
            if len(results) >= max_samples:
                return results

            corrupted = _corrupt_output(original_output, schema_dict, rng)
            if corrupted == original_output:
                continue

            schema_str = represent_schema(schema_dict, "json")
            prompt = rng.choice(CORRECTION_TEMPLATES).format(
                fmt=FORMAT_NAMES.get(source_fmt, source_fmt),
                schema=schema_str,
                corrupted=corrupted,
            )
            input_msgs = [{"role": "user", "content": prompt}]

            results.append(
                make_gym_record(
                    input_msgs=input_msgs,
                    schema_dict=schema_dict,
                    schema_type=source_fmt,
                    problem_type="error_correction",
                    schema_repr="json",
                    source_record_id=rid,
                )
            )
    return results
