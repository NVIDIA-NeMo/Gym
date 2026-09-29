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
"""Prepare GPQA-X benchmark data.

Ports NeMo Skills' `gpqa-x` benchmark to Gym. Each source split is a target
language, and each output row carries a per-row extraction regex compatible
with the mcqa resource server.
"""

import argparse
import importlib.util
import json
import uuid
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "gpqa-x_benchmark.jsonl"

HF_REPO_ID = "nvidia/Nemotron-Multilinugual-Eval-GPQA"


def _load_utils():
    utils_file = BENCHMARK_DIR / "gpqa_x_utils.py"
    spec = importlib.util.spec_from_file_location("gpqa_x_utils", utils_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_utils = _load_utils()
BOXED_INSTRUCTIONS = _utils.BOXED_INSTRUCTIONS
EN_INSTRUCTION = _utils.EN_INSTRUCTION
EXTRACT_REGEX = _utils.EXTRACT_REGEX
SUPPORTED_LANGUAGES = _utils.SUPPORTED_LANGUAGES


def format_entry(entry: dict, lang: str, prompt_language: str) -> dict:
    instruction = BOXED_INSTRUCTIONS[lang] if prompt_language == "target" else EN_INSTRUCTION
    question = f"{instruction}\n\n{entry['problem']}\n\n{entry['options']}"
    seed_str = json.dumps(
        {"problem": entry["problem"], "options": entry["options"], "language": lang},
        sort_keys=True,
        ensure_ascii=False,
    )
    row_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed_str))

    return {
        "question": question,
        "problem": entry["problem"],
        "expected_answer": str(entry["expected_answer"]),
        "options": [
            {"A": entry["A"]},
            {"B": entry["B"]},
            {"C": entry["C"]},
            {"D": entry["D"]},
        ],
        "template_metadata": {"output_regex": EXTRACT_REGEX},
        "A": entry["A"],
        "B": entry["B"],
        "C": entry["C"],
        "D": entry["D"],
        "subset_for_metrics": lang,
        "target_language": lang,
        "uuid": row_uuid,
    }


def prepare(languages: list[str] | None = None, prompt_language: str = "target") -> Path:
    """Download and prepare multilingual GPQA benchmark data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if languages is None:
        languages = list(SUPPORTED_LANGUAGES)

    count = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for lang in languages:
            print(f"Loading language: {lang} from {HF_REPO_ID}")
            ds = load_dataset(HF_REPO_ID, split=lang)
            for entry in ds:
                fout.write(json.dumps(format_entry(entry, lang, prompt_language), ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default=SUPPORTED_LANGUAGES, nargs="+")
    parser.add_argument(
        "--prompt_language",
        default="target",
        choices=["target", "en"],
        help="Use target-language or English instruction prefix in the baked question text.",
    )
    args = parser.parse_args()
    prepare(languages=args.languages, prompt_language=args.prompt_language)
