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
"""Prepare m-Arena Hard v2 (multilingual) benchmark data.

Loads ``CohereLabs/m-ArenaHard-v2.0`` from Hugging Face, iterates over
all 23 language configs (498 rows each → ~11,454 total), and writes one
row per ``(language, question_id)`` with the fields the ``arena_judge``
resources server consumes at the top level (``question``,
``baseline_answer``, ``category``, ``uid``):

- HF dataset: https://huggingface.co/datasets/CohereLabs/m-ArenaHard-v2.0
- HF columns: ``{question_id, category, subcategory, prompt, language}``
- Categories preserved per-row (``hard_prompt`` or ``creative_writing``)

Baselines are NOT shipped with the upstream HF dataset. To produce
``baseline_answer`` for end-to-end judging, supply ``--baseline-file``
pointing at a JSONL with rows ``{language, question_id, generation}``;
the script joins by ``(language, question_id)``. When omitted,
``baseline_answer`` is left as the empty string and the resulting file
is suitable only for prompt inspection / dry-run paths.

Per-language Arena-Elo aggregation is deferred — for now we carry
``language`` and ``subset_for_metrics`` (the language code, e.g. ``"en"``)
so downstream tooling can group when ready.
"""

import argparse
import json
from pathlib import Path

import datasets


HF_DATASET = "CohereLabs/m-ArenaHard-v2.0"

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "m_arena_hard_v2_benchmark.jsonl"


def _format_entry(row: dict, language: str) -> dict:
    """Map an HF row + language code to the arena_judge contract."""
    entry = {
        # arena_judge consumes ``uid``; HF ships ``question_id``.
        "uid": row["question_id"],
        # arena_judge + the prompt template expect ``question``; HF ships ``prompt``.
        "question": row["prompt"],
        # Default empty so the row stays valid even without a baseline file;
        # judging is a no-op in that case.
        "baseline_answer": "",
        # Preserve per-row category (``hard_prompt`` / ``creative_writing``);
        # arena_judge picks the matching judge prompt template.
        "category": row["category"],
        "language": language,
        # Per-language metrics are deferred; we carry the bucket key now so
        # downstream Arena-Elo aggregation can group on it without re-prep.
        "subset_for_metrics": language,
    }
    subcategory = row.get("subcategory")
    if subcategory:
        entry["subcategory"] = subcategory
    return entry


def prepare(languages: list[str] | None = None, baseline_file: str | None = None) -> Path:
    """Download and write ``m_arena_hard_v2_benchmark.jsonl``. Returns the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    supported_languages = datasets.get_dataset_config_names(HF_DATASET)
    if languages is None:
        languages = supported_languages
    else:
        invalid = set(languages) - set(supported_languages)
        if invalid:
            raise ValueError(f"Unsupported languages: {sorted(invalid)}. Supported: {supported_languages}")

    all_entries: list[dict] = []
    for language in languages:
        print(f"Loading {HF_DATASET} (language={language}) ...")
        ds = datasets.load_dataset(HF_DATASET, name=language, split="test")
        for row in ds:
            all_entries.append(_format_entry(row, language))

    # Optional baseline join: rows with ``{language, question_id, generation}``,
    # keyed by ``(language, question_id)``. Mirrors the Skills prepare flag.
    if baseline_file:
        baseline_lookup: dict[tuple[str, str], str] = {}
        with open(baseline_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line)
                baseline_lookup[(data["language"], data["question_id"])] = data["generation"]
        for entry in all_entries:
            key = (entry["language"], entry["uid"])
            if key not in baseline_lookup:
                raise ValueError(f"{key} not found in baseline file {baseline_file}")
            entry["baseline_answer"] = baseline_lookup[key]

    with open(OUTPUT_FPATH, "wt", encoding="utf-8") as fout:
        for entry in all_entries:
            json.dump(entry, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Wrote {len(all_entries)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare m-ArenaHard-v2.0 multilingual benchmark.")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Subset of HF language configs to include (default: all).",
    )
    parser.add_argument(
        "--baseline-file",
        default=None,
        help=(
            "Optional JSONL with rows {language, question_id, generation}. "
            "Joined by (language, question_id) to populate baseline_answer."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare(languages=args.languages, baseline_file=args.baseline_file)
