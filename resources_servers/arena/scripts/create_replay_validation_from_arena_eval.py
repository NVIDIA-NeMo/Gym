#!/usr/bin/env python3
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
"""Create a replay validation file from human evaluation results.

Only use the style control OFF score for comparison. The required reference length is only a schema
placeholder: it is the median of the evaluated model and opponent answers from
that battle, not the three fixed reference-model responses used by final v3.
"""

import argparse
import json
import statistics
from pathlib import Path

import tiktoken


def message_text(message: dict) -> str:
    return "\n".join(part["text"] for part in message["content"] if part["type"] == "text").strip()


def prompt_and_answer(conversation: list[dict], evaluation_index: int) -> tuple[list[dict], str]:
    """Return the prompt before the evaluated turn and that turn's answer."""
    messages = []
    answer = ""
    for message in conversation:
        text = message_text(message)
        if message["evaluation_index"] == evaluation_index and message["role"] == "assistant":
            answer = text
            break
        if text:
            messages.append({"role": message["role"], "content": text})
    return messages, answer


def flatten(messages: list[dict]) -> str:
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"]
    return "\n\n".join(f"[{message['role'].capitalize()}]: {message['content']}" for message in messages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evaluated-model", help="Inferred when one model appears in every battle")
    args = parser.parse_args()

    lines = [line for line in args.eval.expanduser().read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    # The evaluated model is normally the only model present in every battle.
    common_models = set.intersection(*({row["model_a_name"], row["model_b_name"]} for row in rows))
    evaluated_model = args.evaluated_model or (common_models.pop() if len(common_models) == 1 else None)
    if not evaluated_model:
        parser.error("Could not infer the evaluated model; pass --evaluated-model")

    encoding = tiktoken.encoding_for_model("gpt-4o")
    records = []
    for row in rows:
        # Orient every battle as evaluated model versus opponent.
        policy_is_a = row["model_a_name"] == evaluated_model
        if not policy_is_a and row["model_b_name"] != evaluated_model:
            raise ValueError(f"{evaluated_model!r} is absent from battle {row['id']}")
        policy_side, baseline_side = ("a", "b") if policy_is_a else ("b", "a")
        policy_conversation = row[f"full_conversation_{policy_side}"]
        messages, original_answer = prompt_and_answer(policy_conversation, row["evaluation_index"])
        _, baseline_answer = prompt_and_answer(row[f"full_conversation_{baseline_side}"], row["evaluation_index"])
        if not messages or not baseline_answer:
            continue

        winner = row["winner"]
        if winner in {"tie", "both_bad"}:
            oriented_winner = winner
        else:
            # Store outcomes relative to the evaluated model, independent of A/B order.
            policy_won = (winner == "model_a") == policy_is_a
            oriented_winner = "other" if policy_won else "baseline"

        lengths = [
            len(encoding.encode(answer, disallowed_special=())) for answer in (original_answer, baseline_answer)
        ]
        records.append(
            {
                "category": "lmarena_v3",
                "responses_create_params": {"input": messages},
                "question_id": row["id"],
                "question": flatten(messages),
                "baseline_answer": baseline_answer,
                "baseline_model": row[f"model_{baseline_side}_name"],
                "other_answer": original_answer,
                "other_model": evaluated_model,
                "winner": oriented_winner,
                # median length of the saved evaluated-model and opponent answers.
                # This is not the reference-model median used for v3 scoring; use replay scores OFF only for comparison.
                "style_reference_token_count": round(statistics.median(lengths)),
                "is_lmarena_v2_prompt": False,
                "metadata": {
                    "conversation_id": policy_conversation[0]["id"],
                    "evaluation_index": row["evaluation_index"],
                    "user_language": row.get("user_language"),
                    "tags": row.get("tags"),
                },
            }
        )

    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with args.output.expanduser().open("w", encoding="utf-8") as file:
        file.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    print(f"evaluated_model: {evaluated_model}")
    print(f"written: {len(records)}")
    print("style_reference: placeholder median of the evaluated and opponent answers")
    print("score: use Proxy OFF; Proxy ON is not comparable to final v3")
    print(f"output: {args.output.expanduser()}")


if __name__ == "__main__":
    main()
