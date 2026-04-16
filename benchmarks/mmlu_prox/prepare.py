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

"""Prepare MMLU-ProX evaluation data for NeMo Gym.

Downloads MMLU-ProX from HuggingFace for each language and converts to Gym
JSONL format compatible with the mcqa resource server.

Each row embeds the full language-specific formatted question (description +
options) in the `question` field, and carries a per-row `template_metadata`
with an `output_regex` for language-specific answer extraction.
"""

import importlib.util
import json
import tempfile
import urllib.request
import uuid
from pathlib import Path

from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "mmlu_prox_benchmark.jsonl"
LANG_LIBS_URL = "https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/refs/heads/main/lm_eval/tasks/mmlu_prox/lang_libs.py"
DEFAULT_LANGUAGES = ["en", "de", "es", "fr", "it", "ja"]


def _download_and_parse_lang_libs() -> tuple:
    """Download lang_libs.py from EleutherAI GitHub (or use cached copy) and return LANG_LIBS, LANG_SUBJECTS."""
    cached_path = DATA_DIR / "lang_libs.py"

    if cached_path.exists():
        print(f"Using cached lang_libs.py from {cached_path}")
        lang_libs_path = str(cached_path)
    else:
        print(f"Downloading lang_libs.py from {LANG_LIBS_URL}...")
        try:
            with urllib.request.urlopen(LANG_LIBS_URL) as response:
                content = response.read().decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to download lang_libs.py: {e}")

        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            cached_path.write_text(content, encoding="utf-8")
            print(f"Cached lang_libs.py to {cached_path}")
            lang_libs_path = str(cached_path)
        except Exception as e:
            print(f"Warning: could not cache lang_libs.py ({e}), using temp file")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
                tmp.write(content)
                lang_libs_path = tmp.name

    spec = importlib.util.spec_from_file_location("lang_libs", lang_libs_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create module spec for lang_libs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "LANG_LIBS") or not hasattr(module, "LANG_SUBJECTS"):
        raise RuntimeError("LANG_LIBS or LANG_SUBJECTS not found in lang_libs.py")

    return module.LANG_LIBS, module.LANG_SUBJECTS


def _format_entry(entry: dict, language: str, lang_libs: dict, lang_subjects: dict) -> dict:
    """Convert a raw HuggingFace row into Gym JSONL format."""
    category = entry["category"].replace(" ", "_")

    # Collect options from option_0 .. option_9
    choices = [entry[f"option_{i}"] for i in range(10)]
    letters = [chr(ord("A") + i) for i in range(10)]

    # Build options list for the mcqa verifier
    options = [{letter: text} for letter, text in zip(letters, choices)]

    # Build the full formatted question using language-specific templates from lang_libs:
    # lang_libs[lang][0] = question prefix, [1] = answer prefix, [3] = description template,
    # [5] = answer extraction format string
    subject = lang_subjects[language][category]
    description = lang_libs[language][3].format(subject=subject, ans_suffix=lang_libs[language][5].format("X")) + "\n"
    options_text = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, choices))
    question = (
        f"{description}{lang_libs[language][0]}\n{entry['question']}\n{lang_libs[language][1]}\n{options_text}\n"
    )

    # Build language-specific answer extraction regex
    extract_regex = lang_libs[language][5].replace("({})", r"\(?([ABCDEFGHIJ])\)?")
    if language == "en":
        extract_regex = extract_regex.lstrip("the").strip()
        extract_regex = extract_regex.replace("\\(", "\\**\\(")
        extract_regex = extract_regex.replace("\\)?", "\\)?\\**")

    # Same question appears across multiple languages, so we include language in the UUID seed.
    seed_str = json.dumps({"question": entry["question"], "options": choices, "language": language}, sort_keys=True)
    row_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed_str))

    return {
        "question": question,
        "options": options,
        "expected_answer": entry["answer"],
        "template_metadata": {"output_regex": extract_regex},
        # language and category are not used but useful for analysis
        "language": language,
        "category": category,
        "uuid": row_uuid,
    }


def prepare(languages: list[str] = DEFAULT_LANGUAGES) -> Path:
    """Download MMLU-ProX test data for each language and convert to Gym JSONL format."""
    from datasets import load_dataset

    lang_libs, lang_subjects = _download_and_parse_lang_libs()
    print("Successfully loaded lang_libs data.")

    hf_token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for language in languages:
        print(f"Downloading MMLU-ProX [{language}] from HuggingFace...")
        ds = load_dataset("li-lab/MMLU-ProX", language, split="test", token=hf_token)
        for example in ds:
            row = _format_entry(example, language, lang_libs, lang_subjects)
            rows.append(json.dumps(row) + "\n")
        print(f"  {len(ds)} examples loaded for language '{language}'")

    with open(OUTPUT_FPATH, "w") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} total problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
