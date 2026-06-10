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
"""Category 5: Schema-only generation (no document context)."""

import random
from typing import Any, Dict, List

from templates import (
    ALL_FORMATS,
    FORMAT_NAMES,
    SCHEMA_ONLY_TEMPLATES,
    make_gym_record,
    represent_schema,
)


SCHEMA_REPR_MODES = ["json", "yaml", "python"]


def generate_schema_only(
    records: List[Dict[str, Any]],
    rng: random.Random,
    samples_per_record: int = 3,
    target_formats: List[str] = ALL_FORMATS,
    max_samples: int = 1000,
) -> List[Dict[str, Any]]:
    results = []
    for record in records:
        schema_dict = record["_json_schema"]
        rid = record.get("_record_id", "unknown")

        for _ in range(samples_per_record):
            if len(results) >= max_samples:
                return results

            target_fmt = rng.choice(target_formats)
            repr_mode = rng.choice(SCHEMA_REPR_MODES)
            schema_str = represent_schema(schema_dict, repr_mode)

            prompt = rng.choice(SCHEMA_ONLY_TEMPLATES).format(
                fmt=FORMAT_NAMES.get(target_fmt, target_fmt),
                schema=schema_str,
            )
            input_msgs = [{"role": "user", "content": prompt}]

            results.append(
                make_gym_record(
                    input_msgs=input_msgs,
                    schema_dict=schema_dict,
                    schema_type=target_fmt,
                    problem_type="schema_only",
                    schema_repr=repr_mode,
                    source_record_id=rid,
                )
            )
    return results
