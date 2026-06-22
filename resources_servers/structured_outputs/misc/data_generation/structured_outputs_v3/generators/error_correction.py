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
from typing import Any, Dict, List, Optional

import tomli_w
import yaml
from templates import (
    ALL_FORMATS,
    CORRECTION_TEMPLATES,
    FORMAT_NAMES,
    make_gym_record,
    represent_schema,
)


SUPPORTED_FORMATS = ("json", "yaml", "toml")


def _parse_output(output_str: str, fmt: str) -> Optional[Any]:
    """Parse output string into a Python object. Returns None on failure."""
    try:
        if fmt == "json":
            return json.loads(output_str)
        elif fmt == "yaml":
            return yaml.safe_load(output_str)
        elif fmt == "toml":
            import tomllib

            return tomllib.loads(output_str)
    except Exception:
        return None


def _serialize_output(obj: Any, fmt: str) -> Optional[str]:
    """Serialize a Python object back to the target format. Returns None on failure."""
    try:
        if fmt == "json":
            return json.dumps(obj, ensure_ascii=False)
        elif fmt == "yaml":
            return yaml.dump(obj, default_flow_style=False, allow_unicode=True).rstrip("\n")
        elif fmt == "toml":
            return tomli_w.dumps(obj).rstrip("\n")
    except Exception:
        return None


def _corrupt_output(output_str: str, schema_dict: Dict, fmt: str, rng: random.Random) -> str:
    """Apply a random corruption to the output string."""
    corruption = rng.choice(["drop_field", "wrong_type", "extra_field", "syntax"])

    if corruption in ("drop_field", "wrong_type", "extra_field"):
        obj = _parse_output(output_str, fmt)
        if obj is not None and isinstance(obj, dict):
            if corruption == "drop_field" and len(obj) > 1:
                key = rng.choice(list(obj.keys()))
                del obj[key]
                result = _serialize_output(obj, fmt)
                if result:
                    return result

            elif corruption == "wrong_type":
                for k, v in obj.items():
                    if isinstance(v, int):
                        obj[k] = str(v)
                        result = _serialize_output(obj, fmt)
                        if result:
                            return result
                        break
                    if isinstance(v, str) and v.isdigit():
                        obj[k] = int(v)
                        result = _serialize_output(obj, fmt)
                        if result:
                            return result
                        break

            elif corruption == "extra_field":
                obj["_unexpected_field"] = "should_not_be_here"
                result = _serialize_output(obj, fmt)
                if result:
                    return result

    elif corruption == "syntax":
        if fmt == "json":
            if output_str.startswith("{"):
                pos = rng.randint(len(output_str) // 4, 3 * len(output_str) // 4)
                return output_str[:pos] + ",,," + output_str[pos:]
        elif fmt == "yaml":
            lines = output_str.split("\n")
            if len(lines) > 2:
                idx = rng.randint(1, len(lines) - 1)
                lines[idx] = "  " + lines[idx] + " :" + lines[idx]
                return "\n".join(lines)
        elif fmt == "toml":
            lines = output_str.split("\n")
            if len(lines) > 1:
                idx = rng.randint(0, len(lines) - 1)
                lines[idx] = lines[idx] + ' = = "broken"'
                return "\n".join(lines)
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
    eligible = [r for r in records if r.get("target_output_format") in SUPPORTED_FORMATS]
    if not eligible:
        eligible = records

    for record in eligible:
        schema_dict = record["_json_schema"]
        rid = record.get("_record_id", "unknown")
        original_output = record.get("target_output", "")
        source_fmt = record.get("target_output_format", "json")

        if not original_output:
            continue

        for _ in range(samples_per_record):
            if len(results) >= max_samples:
                return results

            corrupted = _corrupt_output(original_output, schema_dict, source_fmt, rng)
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
