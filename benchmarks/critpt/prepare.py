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

"""Prepare CritPt evaluation data for NeMo Gym.

Downloads CritPt-Benchmark/CritPt from HuggingFace (public dataset, no token required)
and prepares both user messages in responses_create_params.input. The CritPt agent
defers the final formatting message until the solve response is available.
"""

import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import JsonValue

from nemo_gym.core import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "critpt_benchmark.jsonl"

# Note: data/example.jsonl is a hand-curated repo artifact (5 problems spanning multiple
# physics domains) committed under resources_servers/critpt/data/example.jsonl. It is
# intentionally NOT regenerated here — running prepare() will not touch it.


def prepare_row(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Prepare both CritPt turns without exposing verifier-only fields to the model.

    Existing solve messages and generation parameters are preserved. Re-preparing an
    input that already ends with the two user messages does not append another turn.
    """
    prepared = dict(row)
    code_template = prepared.get("code_template")
    if not isinstance(code_template, str):
        raise ValueError("CritPt preparation requires a string code_template")
    raw_params = prepared.get("responses_create_params")
    if not isinstance(raw_params, dict) or not raw_params.get("input"):
        prompt = load_prompt_config(str(BENCHMARK_DIR / "prompts/turn1.yaml"))
        prepared = apply_prompt_to_row(prepared, prompt)
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(prepared["responses_create_params"])
    messages = params.input
    if isinstance(messages, str):
        messages = [NeMoGymEasyInputMessage(role="user", content=messages)]
    else:
        messages = list(messages)
    if not messages or getattr(messages[-1], "role", None) != "user":
        raise ValueError("CritPt solve input must end with a user message")
    if len(messages) < 2 or getattr(messages[-2], "role", None) != "user":
        prompt = load_prompt_config(str(BENCHMARK_DIR / "prompts/turn2.yaml"))
        messages.append(NeMoGymEasyInputMessage(role="user", content=prompt.user.format(code_template=code_template)))
    params = params.model_copy(update={"input": messages})
    prepared["responses_create_params"] = params.model_dump(mode="json", exclude_unset=True)
    return prepared


def is_prepared_data_current(fpath: Path) -> bool:
    """Whether a cached prepared file has rows and both user messages prepared on every row."""
    has_rows = False
    try:
        with fpath.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                has_rows = True
                messages = ((json.loads(line).get("responses_create_params") or {}).get("input")) or []
                if not isinstance(messages, list) or len(messages) < 2:
                    return False
                if any(not isinstance(message, dict) or message.get("role") != "user" for message in messages[-2:]):
                    return False
    except (json.JSONDecodeError, AttributeError):
        return False
    return has_rows


def prepare() -> Path:
    from datasets import load_dataset

    print("Downloading CritPt-Benchmark/CritPt from HuggingFace...")
    ds = load_dataset("CritPt-Benchmark/CritPt", split="train")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for problem in ds:
        row = {
            "problem_id": problem["problem_id"],
            "problem": problem["problem_description"],
            "code_template": problem["code_template"],
            "uuid": problem["problem_id"],
        }
        rows.append(json.dumps(prepare_row(row)) + "\n")

    with open(OUTPUT_FPATH, "w") as f:
        f.writelines(rows)
    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")

    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
