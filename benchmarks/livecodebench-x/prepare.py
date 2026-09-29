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
"""Prepare multilingual LiveCodeBench-X benchmark data for Gym.

Ports NeMo Skills' `livecodebench-x` benchmark
(`nemo_skills/dataset/livecodebench-x/prepare.py`) to Gym. Each output row
contains:

- The translated `question` (instruction prefix + translated problem statement),
  produced exactly as Skills produces it.
- `verifier_metadata.unit_tests.{inputs,outputs,fn_name}` — the canonical
  LCB unit tests, joined on `task_id` from `livecodebench/code_generation_lite`.
  Skills relies on the upstream `livecodebench` package to look these up at
  evaluation time; Gym's `code_gen` resource server requires them baked into
  the JSONL, so the join happens here in `prepare.py` rather than at run time.

The English-instruction prefix is also supported via `--prompt_language en`,
matching Skills' `--prompt_language` flag.
"""

import argparse
import base64
import importlib.util
import json
import pickle
import zlib
from pathlib import Path

from datasets import load_dataset


def _decode_test_cases(raw) -> list:
    """Decode test cases from `livecodebench/code_generation_lite`.

    Public test cases are plain JSON. Private test cases are base64+zlib+pickle
    encoded. This is the exact same decoding used by
    `benchmarks/livecodebench/prepare_utils.py::_decode_test_cases`; we inline
    it here to avoid cross-benchmark imports of a private symbol.
    """
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "livecodebench-x_benchmark.jsonl"

MULTILINGUAL_HF_REPO_ID = "nvidia/Nemotron-Multilinugual-Eval-LCB"
CANONICAL_HF_REPO_ID = "livecodebench/code_generation_lite"
CANONICAL_HF_REVISION = "refs/pr/7"


def _load_utils():
    utils_file = BENCHMARK_DIR / "livecodebench_x_utils.py"
    spec = importlib.util.spec_from_file_location("livecodebench_x_utils", utils_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_utils = _load_utils()
CODEGEN_INSTRUCTIONS = _utils.CODEGEN_INSTRUCTIONS
EN_INSTRUCTION = _utils.EN_INSTRUCTION
SUPPORTED_LANGUAGES = _utils.SUPPORTED_LANGUAGES
SUPPORTED_VERSIONS = _utils.SUPPORTED_VERSIONS


def _build_canonical_index(version: str) -> dict[str, dict]:
    """Return {task_id: {unit_tests, difficulty}} for the given LCB version.

    `unit_tests` carries `inputs`, `outputs`, and optional `fn_name`. Mirrors
    `benchmarks/livecodebench/prepare_utils.py::prepare_from_hf_raw`, but keyed
    by `task_id` (== upstream `question_id`) instead of producing rows directly.
    """
    print(f"Downloading LiveCodeBench {version} from HuggingFace ({CANONICAL_HF_REPO_ID})...")
    ds = load_dataset(
        CANONICAL_HF_REPO_ID,
        f"release_{version}",
        split="test",
        revision=CANONICAL_HF_REVISION,
    )

    index: dict[str, dict] = {}
    for example in ds:
        task_id = example.get("question_id", "")
        if not task_id:
            continue

        pub = _decode_test_cases(example.get("public_test_cases", ""))
        priv = _decode_test_cases(example.get("private_test_cases", ""))
        inputs = [tc["input"] for tc in pub] + [tc["input"] for tc in priv]
        outputs = [tc["output"] for tc in pub] + [tc["output"] for tc in priv]

        meta = example.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta) if meta else {}

        index[task_id] = {
            "unit_tests": {
                "inputs": inputs,
                "outputs": outputs,
                "fn_name": meta.get("func_name") or None,
            },
            "difficulty": example.get("difficulty", "unknown"),
        }
    print(f"  Indexed {len(index)} canonical {version} problems by task_id")
    return index


def format_entry(entry: dict, lang: str, prompt_language: str, canonical: dict) -> dict | None:
    """Build a Gym JSONL row for one (multilingual entry, language) pair.

    Returns None if the multilingual `task_id` cannot be found in the canonical
    LCB index — those rows are dropped (D4 row-count check catches drift).
    """
    task_id = entry["task_id"]
    if task_id not in canonical:
        return None

    instruction = CODEGEN_INSTRUCTIONS[lang] if prompt_language == "target" else EN_INSTRUCTION
    canonical_row = canonical[task_id]

    return {
        "question": f"{instruction}\n\n{entry['question']}",
        "problem": entry["question"],
        "task_id": task_id,
        "release_version": entry["release_version"],
        "subset_for_metrics": lang,
        "target_language": lang,
        "verifier_metadata": {
            "unit_tests": canonical_row["unit_tests"],
            "difficulty": canonical_row["difficulty"],
        },
    }


def prepare(
    languages: list[str] | None = None,
    versions: list[str] | None = None,
    prompt_language: str = "target",
) -> Path:
    """Download and prepare multilingual LiveCodeBench-X benchmark data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if languages is None:
        languages = list(SUPPORTED_LANGUAGES)
    if versions is None:
        versions = list(SUPPORTED_VERSIONS)

    count = 0
    skipped = 0
    # Build each version's canonical index lazily so we can free it before
    # moving on. Each index can hold hundreds of MB of decoded test cases.
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for version in versions:
            canonical = _build_canonical_index(version)
            for lang in languages:
                print(f"Loading multilingual ({version}/{lang}) from {MULTILINGUAL_HF_REPO_ID}")
                ds = load_dataset(MULTILINGUAL_HF_REPO_ID, version, split=lang)
                for entry in ds:
                    row = format_entry(entry, lang, prompt_language, canonical)
                    if row is None:
                        skipped += 1
                        continue
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
            del canonical

    print(f"Wrote {count} problems to {OUTPUT_FPATH} (skipped {skipped} with missing canonical match)")
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default=SUPPORTED_LANGUAGES, nargs="+")
    parser.add_argument("--versions", default=SUPPORTED_VERSIONS, nargs="+")
    parser.add_argument(
        "--prompt_language",
        default="target",
        choices=["target", "en"],
        help="Use target-language or English instruction prefix in the baked question text.",
    )
    args = parser.parse_args()
    prepare(languages=args.languages, versions=args.versions, prompt_language=args.prompt_language)
