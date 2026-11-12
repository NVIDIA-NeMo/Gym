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

"""Upstream's prompt construction for NMRArena, reproduced from ``dataset/llm_track.ipynb``.

The system template is vendored byte for byte under ``prompts/`` (see
``prompts/NOTICE``). The user prompt and the NMR-string normalisation are short
enough that reproducing the code is clearer than vendoring the notebook; both
follow the notebook's ``build_user_prompt`` and ``_normalize_nmr`` and are checked
against the request the notebook logged in ``results/LLM_results/llm_rep1_raw.jsonl``.
"""

import re
from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt.txt"

# Upstream ``NUM_CANDIDATES`` and ``MIN_SLOTS``: the model is asked for up to ten ranked
# structures and to fill at least ``min(MIN_SLOTS, n)`` slots.
NUM_CANDIDATES = 10
MIN_SLOTS = 10

# Upstream ``TEMPERATURE`` and ``MAX_TOKENS``: "identical settings for every model —
# temperature 1.0, 24K max tokens, provider-default reasoning" (README). No ``top_p``
# and no seed are set upstream.
TEMPERATURE = 1.0
MAX_OUTPUT_TOKENS = 24576


def load_system_template() -> str:
    return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")


def system_prompt(n: int = NUM_CANDIDATES) -> str:
    """Upstream ``system_prompt(n)``: the template formatted with ``n`` and ``min(MIN_SLOTS, n)``."""
    return load_system_template().format(n=n, min_slots=min(MIN_SLOTS, n))


def normalize_nmr(s: str) -> str:
    """Upstream ``_normalize_nmr``: drop the leading ``H_NMR``/``C_NMR`` label, fix solvent spellings, squeeze spaces."""
    s = (s or "").strip()
    s = re.sub(r"^\s*[\dHC]*_?NMR\s*", "", s)  # drop leading 1H/13C NMR label
    s = re.sub(r"_(d\d)", r"-\1", s)  # DMSO_d6 -> DMSO-d6
    s = re.sub(r"([A-Za-z])_(\d)", r"\1\2", s)  # CDCl_3 -> CDCl3
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_user_prompt(h_nmr: str, c_nmr: str, n: int = NUM_CANDIDATES) -> str:
    """Upstream ``build_user_prompt``; the four lines and their wording are the notebook's."""
    lines = ["Determine the structure of a single organic molecule from its NMR data."]
    lines.append(f"1H NMR: {normalize_nmr(h_nmr)}")
    lines.append(f"13C NMR: {normalize_nmr(c_nmr)}")
    lines.append(f"\nPropose up to {n} candidate structures, ranked best-first, in the required JSON format.")
    return "\n".join(lines)


def build_messages(h_nmr: str, c_nmr: str, n: int = NUM_CANDIDATES) -> list[dict[str, str]]:
    """Upstream ``build_messages``: one system turn and one user turn, zero-shot."""
    return [
        {"role": "system", "content": system_prompt(n)},
        {"role": "user", "content": build_user_prompt(h_nmr, c_nmr, n)},
    ]
