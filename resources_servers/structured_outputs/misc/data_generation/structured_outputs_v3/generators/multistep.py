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
    max_history_turns: int = 4,
) -> List[Dict[str, Any]]:
    """1-N unrelated history pairs, then a new extraction problem.

    max_history_turns controls how many prior user+assistant pairs can appear
    before the final question. Randomly picks 1..max_history_turns pairs per
    sample. With default max_history_turns=4, the model sees up to 4 prior
    turns + 1 new question = 5 turns total.
    """
    results = []
    usable = [r for r in records if _has_messages(r)]
    if len(usable) < 2:
        return results

    for _ in range(min(len(records) * samples_per_record, max_samples)):
        if len(results) >= max_samples:
            return results

        n_history = rng.randint(1, max_history_turns)
        needed = n_history + 1
        if len(usable) < needed:
            needed = len(usable)
            n_history = needed - 1
            if n_history < 1:
                continue

        picked = rng.sample(usable, needed)
        history_recs = picked[:n_history]
        target_rec = picked[-1]

        input_msgs = []
        for rec in history_recs:
            msgs = rec.get("messages", [])
            user = next((m["content"] for m in msgs if m["role"] == "user"), None)
            asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if user and asst:
                input_msgs.append({"role": "user", "content": user})
                input_msgs.append({"role": "assistant", "content": asst})

        if not input_msgs:
            continue

        schema_b = target_rec["_json_schema"]
        doc_b = target_rec.get("document", "")
        rid_b = target_rec.get("_record_id", "unknown")
        if not doc_b:
            continue

        target_fmt = rng.choice(target_formats)
        schema_str = represent_schema(schema_b, "json")

        system_msg = rng.choice(SCHEMA_INSTRUCTIONS[target_fmt]).format(schema=schema_str)
        user_query = rng.choice(USER_QUERY_INSTRUCTIONS)
        new_user_msg = f"{user_query}\n\nDocument:\n{doc_b}"
        input_msgs.append({"role": "user", "content": f"{system_msg}\n{new_user_msg}"})

        actual_turns = len([m for m in input_msgs if m["role"] == "user"])

        results.append(
            make_gym_record(
                input_msgs=input_msgs,
                schema_dict=schema_b,
                schema_type=target_fmt,
                problem_type="multistep_unrelated",
                schema_repr="json",
                source_record_id=rid_b,
                num_turns=actual_turns,
            )
        )
    return results


def _has_messages(record):
    msgs = record.get("messages", [])
    return any(m.get("role") == "user" for m in msgs) and any(m.get("role") == "assistant" for m in msgs)
