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
"""Prepare UGPhysics benchmark data for NeMo Gym.

Downloads the 13 physics subjects of the UGPhysics dataset
(``UGPhysics/ugphysics``) for the English split and writes Gym JSONL.
The structure mirrors NeMo Skills' ``nemo_skills/dataset/ugphysics/
prepare.py`` so the rows are byte-identical except for renamed fields:

  - Skills' ``problem``           -> Gym ``question``
  - Skills' ``answers``           -> Gym ``expected_answer``
  - Skills' ``subject``           -> kept as ``subject`` (Skills also
    duplicates this into ``subset_for_metrics``; we keep only ``subject``
    for the Tier-2 metrics breakdown).

The ``solution``, ``answer_type``, ``language``, ``is_multiple_answer``,
``index``, ``prompt_sentence``, and ``boxed_answer_example`` fields are
forwarded as-is — the prompt template needs ``prompt_sentence`` and
``boxed_answer_example`` for the format-string substitution, the judge
reads ``solution``, and per-row metadata helps downstream analysis.

English-only — matches Skills' ``EVAL_SPLIT = "en"``.  The Chinese
split exists upstream but is not wired into ``config.yaml``; if you
need it, call ``load_data("zh")`` and ``save_data`` directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from datasets import load_dataset
from tqdm import tqdm


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_OUTPUT = DATA_DIR / "ugphysics_benchmark.jsonl"


# Verbatim copy of Skills' prepare.py mappings — tweaking these would
# break parity.
OB_ANS_TYPE_ID2EN = {
    "IN": "a range interval",
    "TF": "either True or False",
    "EX": "an expression",
    "EQ": "an equation",
    "MC": "one option of a multiple choice question",
    "NV": "a numerical value without units",
    "TUP": "multiple numbers, separated by comma, such as (x, y, z)",
}

# 13 physics subjects in the UGPhysics dataset. Order kept identical to
# NeMo Skills' prepare.py for stable Tier-2 metric ordering.
SUBSETS = [
    "AtomicPhysics",
    "ClassicalElectromagnetism",
    "ClassicalMechanics",
    "Electrodynamics",
    "GeometricalOptics",
    "QuantumMechanics",
    "Relativity",
    "SemiconductorPhysics",
    "Solid-StatePhysics",
    "StatisticalMechanics",
    "TheoreticalMechanics",
    "Thermodynamics",
    "WaveOptics",
]


def get_prompt_sentence(answer_type: str, is_multiple_answer: bool) -> str:
    """Build the prompt sentence describing the expected answer format.

    Verbatim port of Skills'
    ``nemo_skills/dataset/ugphysics/prepare.py::get_prompt_sentence``.
    """
    types = [t.strip() for t in answer_type.split(",")]
    descriptions = [OB_ANS_TYPE_ID2EN.get(t, t) for t in types]
    if not is_multiple_answer:
        return f"The answer of the problem should be {descriptions[0]}."
    elif len(set(descriptions)) == 1:
        return f"The problem has multiple answers, each of them should be {descriptions[0]}."
    else:
        return f"The problem has multiple answers, with the answers in order being {', '.join(descriptions)}."


def get_boxed_answer_example(is_multiple_answer: bool) -> str:
    """Return the boxed-answer placeholder string for the prompt."""
    if is_multiple_answer:
        return r"\boxed{multiple answers connected with commas}"
    return r"\boxed{answer}(unit)"


def format_entry(entry: dict) -> dict:
    """Map a UGPhysics dataset row to a Gym JSONL row.

    Field renames vs Skills:
      - Skills also writes ``subset_for_metrics``; here we keep only
        ``subject`` (used by ``compute_subset_metrics`` as the Tier-2
        stratifier). The two carried the same value in Skills, so no
        information is lost.
      - ``problem`` -> ``question``; ``answers`` -> ``expected_answer``.
    """
    is_multiple_answer = entry["is_multiple_answer"]
    answer_type = entry["answer_type"]
    return {
        "index": entry["index"],
        "question": entry["problem"],
        "expected_answer": entry["answers"],
        "solution": entry["solution"],
        "answer_type": answer_type,
        "subject": entry["subject"],
        "language": entry["language"].lower(),
        "is_multiple_answer": is_multiple_answer,
        "prompt_sentence": get_prompt_sentence(answer_type, is_multiple_answer),
        "boxed_answer_example": get_boxed_answer_example(is_multiple_answer),
    }


def load_data(lang_split: str) -> List[dict]:
    """Load all 13 subjects for a single language split."""
    data: List[dict] = []
    for subset in tqdm(SUBSETS, desc=f"Loading {lang_split} subsets"):
        subset_data = load_dataset("UGPhysics/ugphysics", subset, split=lang_split)
        data.extend(subset_data)
    return data


def save_data(data: Iterable[dict], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_path.name}"):
            json.dump(format_entry(entry), fout, ensure_ascii=False)
            fout.write("\n")


def prepare() -> Path:
    """Download the English UGPhysics split and convert to Gym JSONL.

    English-only matches Skills' ``EVAL_SPLIT = "en"``.  The dataset
    also has a Chinese split — call ``load_data("zh")`` and
    ``save_data`` directly if needed.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_data(load_data("en"), DEFAULT_OUTPUT)
    return DEFAULT_OUTPUT


if __name__ == "__main__":
    prepare()
