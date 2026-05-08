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
"""Prepare m-ArenaHard multilingual benchmark data.

Loads the
[``CohereLabs/m-ArenaHard``](https://huggingface.co/datasets/CohereLabs/m-ArenaHard)
HuggingFace dataset across every language config and writes one row per
``(language, question_id)`` with the fields the ``arena_judge``
resources server consumes at the top level (``question``, ``uid``,
``category``, ``baseline_answer``).

The HF dataset only ships ``{question_id, cluster, category, prompt}``
columns and does **not** include any baseline model answer. To run the
full Arena-Elo judging end-to-end, supply ``--baseline-file``: a JSONL
with rows ``{"language", "question_id", "generation"}`` joined in by
``(language, question_id)``. This mirrors NeMo Skills, which generates
the baseline with a separate ``ns generate`` run and feeds the
resulting JSONL back into ``prepare`` via ``--baseline-file``.

When ``--baseline-file`` is omitted, ``baseline_answer`` is set to the
empty string so the JSONL is still well-formed for tooling that only
needs the question set (e.g. data exploration or Skills-vs-Gym prepare
parity checks). ``ng_prepare_benchmark`` calls ``prepare()`` with no
arguments and so always takes this no-baseline path; use the script
entry-point (``python prepare.py --baseline-file ...``) for full
arena_judge runs.

The HF dataset's ``category`` column is preserved as
``original_category`` for traceability; ``category`` itself is
overwritten with the literal ``"hard_prompt"`` to route every prompt
through ``arena_judge``'s standard hard-prompt judge template (matches
Skills' ``m-arena-hard/prepare.py``). The per-row ``language`` is also
emitted under ``subset_for_metrics`` to set up future per-language
Arena-Elo aggregation (deferred — needs ``arena_judge`` to honour
``subset_key=language``).
"""

import argparse
import json
from pathlib import Path

import datasets


HF_DATASET = "CohereLabs/m-ArenaHard"

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "m_arena_hard_benchmark.jsonl"


def _format_entry(row: dict, language: str) -> dict:
    return {
        # arena_judge expects the question identifier under ``uid``.
        "uid": row["question_id"],
        "question": row["prompt"],
        "language": language,
        # arena_judge expects "hard_prompt" or "creative_writing"; m-ArenaHard
        # has no creative_writing split, so route everything through the
        # standard hard-prompt judge template (matches Skills).
        "category": "hard_prompt",
        "original_category": row["category"],
        "cluster": row["cluster"],
        # Carries the language code (e.g. "en", "de") for future per-language
        # Arena-Elo aggregation in arena_judge.
        "subset_for_metrics": language,
    }


def prepare(languages: list[str] | None = None, baseline_file: str | None = None) -> Path:
    """Download m-ArenaHard from HF, optionally join baselines, write JSONL.

    Called with no arguments by ``ng_prepare_benchmark``. Returns the
    path to the written JSONL.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    supported = datasets.get_dataset_config_names(HF_DATASET)
    if languages is None:
        languages = supported
    invalid = set(languages) - set(supported)
    if invalid:
        raise ValueError(f"Unsupported languages: {sorted(invalid)}. Supported: {supported}")

    all_entries: list[dict] = []
    for language in languages:
        print(f"Loading {HF_DATASET} ({language}) ...")
        ds = datasets.load_dataset(HF_DATASET, name=language, split="test")
        for row in ds:
            all_entries.append(_format_entry(row, language))

    if baseline_file:
        # Mirror Skills: external JSONL with rows {language, question_id, generation}.
        baseline_lookup: dict[tuple[str, str], str] = {}
        with open(baseline_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line)
                baseline_lookup[(data["language"], data["question_id"])] = data["generation"]
        for entry in all_entries:
            key = (entry["language"], entry["uid"])
            if key not in baseline_lookup:
                raise ValueError(f"({key[0]}, {key[1]}) not found in baseline file {baseline_file}")
            entry["baseline_answer"] = baseline_lookup[key]
    else:
        # No baseline supplied — emit empty string so the JSONL is still
        # well-formed. Full arena_judge runs require --baseline-file; see README.
        for entry in all_entries:
            entry["baseline_answer"] = ""

    with open(OUTPUT_FPATH, "wt", encoding="utf-8") as fout:
        for entry in all_entries:
            json.dump(entry, fout, ensure_ascii=False)
            fout.write("\n")
    print(f"Wrote {len(all_entries)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare m-ArenaHard multilingual benchmark.")
    parser.add_argument(
        "--languages",
        default=None,
        nargs="+",
        help="Language configs to include (default: all supported by the HF dataset).",
    )
    parser.add_argument(
        "--baseline-file",
        default=None,
        help=(
            "Path to JSONL with baseline answers (rows must include 'language', "
            "'question_id', and 'generation'). When omitted, baseline_answer is set to "
            "the empty string."
        ),
    )
    args = parser.parse_args()
    prepare(languages=args.languages, baseline_file=args.baseline_file)
