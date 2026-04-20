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
"""Prepare WMT24++ benchmark data.

Mirrors NeMo-Skills' `nemo_skills/dataset/wmt24pp/prepare.py` row-for-row:
one config per en-<tgt> language pair from `google/wmt24pp` (`train` split,
which is the only split upstream), interleaved by language in the order
they are listed.

Per-row fields match Skills:
  - text, translation, source_language, target_language,
    source_lang_name, target_lang_name

Those field names are referenced by both the prompt template
(benchmarks/wmt24pp/prompts/default.yaml) and the wmt_translation
resource server's verify() / compute_metrics().
"""

import json
from pathlib import Path

from datasets import load_dataset
from langcodes import Language


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "wmt24pp_benchmark.jsonl"

HF_REPO_ID = "google/wmt24pp"

# Same five targets + same order as Skills' default. Keeping the order
# stable is what makes the interleaved JSONL byte-comparable.
DEFAULT_TARGET_LANGUAGES = ["de_DE", "es_MX", "fr_FR", "it_IT", "ja_JP"]


def prepare(target_languages: list[str] | None = None) -> Path:
    """Download and interleave WMT24++ en-<tgt> pairs. Returns the output file path."""
    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets: dict = {}
    for lang in target_languages:
        print(f"Loading {HF_REPO_ID} config en-{lang}...")
        datasets[lang] = load_dataset(HF_REPO_ID, f"en-{lang}")["train"]

    count = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for tgt_lang in target_languages:
            for src, tgt in zip(
                datasets[tgt_lang]["source"],
                datasets[tgt_lang]["target"],
                strict=True,
            ):
                row = {
                    "text": src,
                    "translation": tgt,
                    "source_language": "en",
                    "target_language": tgt_lang,
                    "source_lang_name": "English",
                    "target_lang_name": Language(tgt_lang[:2]).display_name(),
                }
                fout.write(json.dumps(row) + "\n")
                count += 1

    print(f"Wrote {count} rows to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
