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
"""Prepare PolyMath benchmark data.

Mirrors NeMo Skills' ``nemo_skills/dataset/polymath/prepare.py``: for
each (language, difficulty) split of ``Qwen/PolyMath`` on Hugging Face,
format every problem with the upstream PolyMath ``QUESTION_TEMPLATE``
(question + per-language instruction + optional language_control
suffix) and emit one JSONL row.

Per-row fields populated:

  - ``question``: formatted user prompt (question + instruction).
  - ``expected_answer``: gold answer.
  - ``language``: ISO language code (e.g. ``en``, ``zh``).
  - ``subset_for_metrics``: same value as ``language`` — preserved for
    cross-pipeline parity tooling that keys off the Skills field name.
  - ``difficulty``: one of ``low``, ``medium``, ``high``, ``top``.
  - ``weight``: 1 / 2 / 4 / 8 for low / medium / high / top — read by
    the polymath resources server's weighted aggregation.
  - ``language_control_mode``: opt-in language-control mode (or
    ``None``).

Default = all 18 supported languages × all 4 difficulty splits =
~2,250 rows, language_control disabled. The instruction text and
language_control suffix come from upstream PolyMath's ``instruction.py``
(downloaded at prepare time so any upstream wording change flows
through).
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "polymath_benchmark.jsonl"

INSTRUCTIONS_URL = "https://raw.githubusercontent.com/QwenLM/PolyMath/main/instruction.py"

DIFFICULTY_LEVEL_DIC = {"low": 1, "medium": 2, "high": 4, "top": 8}

QUESTION_TEMPLATE = """
{question}

{instruction} {lang_control}
""".strip()


def _load_instructions(url: str) -> tuple[dict, dict, dict]:
    """Pull PolyMath's per-language instruction dicts from the upstream repo."""
    with urllib.request.urlopen(url) as response:
        ns: dict = {}
        exec(compile(response.read().decode("utf-8"), url, "exec"), ns)
        return ns["language_dic"], ns["query_dic"], ns["language_control"]


def format_entry(
    entry: dict,
    language: str,
    difficulty: str,
    query_dic: dict,
    language_control_dic: dict,
    language_control_mode: str | None = None,
) -> dict:
    problem = entry["question"]
    instruction = query_dic[language]
    lang_control = language_control_dic[language_control_mode][language] if language_control_mode else ""
    question = QUESTION_TEMPLATE.format(question=problem, instruction=instruction, lang_control=lang_control).strip()
    return {
        "question": question,
        "problem": problem,
        "expected_answer": entry["answer"],
        "language": language,
        # Preserved for cross-pipeline tooling that keys off Skills' field name.
        "subset_for_metrics": language,
        "difficulty": difficulty,
        "weight": DIFFICULTY_LEVEL_DIC[difficulty],
        "language_control_mode": language_control_mode,
    }


def prepare(
    languages: list[str] | None = None,
    language_control_mode: str | None = None,
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    language_dic, query_dic, language_control_dic = _load_instructions(INSTRUCTIONS_URL)
    supported = list(language_dic.keys())
    if languages is None:
        languages = supported
    invalid = set(languages) - set(supported)
    if invalid:
        raise ValueError(f"Unsupported languages: {invalid}. Supported: {supported}")
    if language_control_mode is not None and language_control_mode not in language_control_dic:
        raise ValueError(
            f"Unsupported language_control mode: {language_control_mode!r}. "
            f"Supported: {list(language_control_dic.keys())}"
        )

    count = 0
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as fout:
        for language in languages:
            for difficulty in DIFFICULTY_LEVEL_DIC:
                print(f"Loading Qwen/PolyMath {language}/{difficulty}...")
                ds = load_dataset("Qwen/PolyMath", name=language, split=difficulty)
                for entry in ds:
                    formatted = format_entry(
                        entry,
                        language=language,
                        difficulty=difficulty,
                        query_dic=query_dic,
                        language_control_dic=language_control_dic,
                        language_control_mode=language_control_mode,
                    )
                    fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                    count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PolyMath multilingual math benchmark.")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Subset of languages to download. Default: all 18 supported.",
    )
    parser.add_argument(
        "--language_control",
        default=None,
        help="Language control mode (forcing_raw / forcing_en / forcing_prefer). Disabled by default.",
    )
    args = parser.parse_args()
    prepare(languages=args.languages, language_control_mode=args.language_control)


if __name__ == "__main__":
    main()
