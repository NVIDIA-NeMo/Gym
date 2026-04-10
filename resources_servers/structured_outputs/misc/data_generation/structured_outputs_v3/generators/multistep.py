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
"""Categories 3+4: Related and unrelated multistep problems."""

import random
from typing import Any, Dict, List

from templates import (
    ALL_FORMATS,
    FORMAT_NAMES,
    MULTISTEP_FOLLOWUP_TEMPLATES,
    SCHEMA_INSTRUCTIONS,
    USER_QUERY_INSTRUCTIONS,
    make_gym_record,
    represent_schema,
)


def generate_multistep_related(
    records: List[Dict[str, Any]],
    rng: random.Random,
    samples_per_record: int = 3,
    target_formats: List[str] = ALL_FORMATS,
    max_samples: int = 1000,
) -> List[Dict[str, Any]]:
    """Turn 1: original Q+A. Turn 2: follow-up asking for format conversion."""
    results = []
    for record in records:
        schema_dict = record["_json_schema"]
        rid = record.get("_record_id", "unknown")
        messages = record.get("messages", [])
        source_fmt = record.get("target_output_format", "json")

        if len(messages) < 2:
            continue

        user_msg_orig = next((m["content"] for m in messages if m["role"] == "user"), None)
        asst_msg_orig = next((m["content"] for m in messages if m["role"] == "assistant"), None)
        if not user_msg_orig or not asst_msg_orig:
            continue

        other_formats = [f for f in target_formats if f != source_fmt]
        if not other_formats:
            continue

        for _ in range(samples_per_record):
            if len(results) >= max_samples:
                return results

            target_fmt = rng.choice(other_formats)
            schema_str = represent_schema(schema_dict, "json")

            followup = rng.choice(MULTISTEP_FOLLOWUP_TEMPLATES).format(
                target_format=FORMAT_NAMES.get(target_fmt, target_fmt),
                schema=schema_str,
            )

            input_msgs = [
                {"role": "user", "content": user_msg_orig},
                {"role": "assistant", "content": asst_msg_orig},
                {"role": "user", "content": followup},
            ]

            results.append(
                make_gym_record(
                    input_msgs=input_msgs,
                    schema_dict=schema_dict,
                    schema_type=target_fmt,
                    problem_type="multistep_related",
                    schema_repr="json",
                    source_record_id=rid,
                    num_turns=2,
                    source_format=source_fmt,
                )
            )
    return results


def generate_multistep_unrelated(
    records: List[Dict[str, Any]],
    rng: random.Random,
    samples_per_record: int = 3,
    target_formats: List[str] = ALL_FORMATS,
    max_samples: int = 1000,
) -> List[Dict[str, Any]]:
    """History from record A, new extraction problem from record B."""
    results = []
    if len(records) < 2:
        return results

    for _ in range(min(len(records) * samples_per_record, max_samples)):
        if len(results) >= max_samples:
            return results

        rec_a, rec_b = rng.sample(records, 2)

        msgs_a = rec_a.get("messages", [])
        user_a = next((m["content"] for m in msgs_a if m["role"] == "user"), None)
        asst_a = next((m["content"] for m in msgs_a if m["role"] == "assistant"), None)
        if not user_a or not asst_a:
            continue

        schema_b = rec_b["_json_schema"]
        doc_b = rec_b.get("document", "")
        rid_b = rec_b.get("_record_id", "unknown")
        if not doc_b:
            continue

        target_fmt = rng.choice(target_formats)
        schema_str = represent_schema(schema_b, "json")

        system_msg = rng.choice(SCHEMA_INSTRUCTIONS[target_fmt]).format(schema=schema_str)
        user_query = rng.choice(USER_QUERY_INSTRUCTIONS)
        new_user_msg = f"{user_query}\n\nDocument:\n{doc_b}"

        input_msgs = [
            {"role": "user", "content": user_a},
            {"role": "assistant", "content": asst_a},
            {"role": "user", "content": f"{system_msg}\n{new_user_msg}"},
        ]

        results.append(
            make_gym_record(
                input_msgs=input_msgs,
                schema_dict=schema_b,
                schema_type=target_fmt,
                problem_type="multistep_unrelated",
                schema_repr="json",
                source_record_id=rid_b,
                num_turns=2,
            )
        )
    return results
