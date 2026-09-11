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
"""Prepare HLE-Verified evaluation data for NeMo Gym.

Downloads HLE-Verified from HuggingFace and converts it to Gym JSONL format.
By default, only text questions from the Gold and Revision subsets are included.
Pass ``include_vision=True`` to include image questions and materialize each row's
``responses_create_params.input`` for use with ``config_vision.yaml``.
"""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_OUTPUT = DATA_DIR / "hle_verified_benchmark.jsonl"
DEFAULT_OUTPUT_VISION = DATA_DIR / "hle_verified_benchmark_vision.jsonl"

# Prompt applied at prepare time for vision rows and at rollout time for text rows.
PROMPT_CONFIG_FPATH = BENCHMARK_DIR / "prompts" / "default.yaml"

REPO_ID = "skylenage/HLE-Verified"

# Map dataset labels to the subset names accepted by --subset.
VERIFIED_CLASSES_MAP = {
    "Gold subset": "gold",
    "Revision subset": "revision",
    "Uncertain subset": "uncertain",
}
VERIFIED_CLASSES_REVERSE_MAP = {v: k for k, v in VERIFIED_CLASSES_MAP.items()}

# Default verified classes: Gold + Revision.
DEFAULT_SUBSET = "text"
SUBSETS = (DEFAULT_SUBSET, "all") + tuple(VERIFIED_CLASSES_MAP.values())

# Metadata stored in the dataset's JSON column.
_PACKED_FIELDS = ("author_name", "rationale", "answer_type", "canary", "image")


def _unpack(row: dict) -> dict:
    """Unpack metadata from the JSON column, preserving existing top-level values."""
    packed: dict = {}
    raw = row.get("json")
    if isinstance(raw, str) and raw:
        try:
            packed = json.loads(raw)
        except json.JSONDecodeError:
            packed = {}

    unpacked = dict(row)
    for field in _PACKED_FIELDS:
        if unpacked.get(field) is None:
            unpacked[field] = packed.get(field)
    return unpacked


def keep_row(row: dict, subset: str, include_vision: bool = False) -> bool:
    """Select rows by verified class, excluding images unless include_vision is set."""
    if row.get("image") and not include_vision:
        return False

    verified_class = row.get("Verified_Classes")
    if subset == "all":
        return True
    if subset == DEFAULT_SUBSET:
        return verified_class != VERIFIED_CLASSES_REVERSE_MAP["uncertain"]
    return verified_class == VERIFIED_CLASSES_REVERSE_MAP[subset]


@lru_cache(maxsize=1)
def _hle_build_input():
    """Load the shared HLE input builder from this checkout.

    Loading by path avoids importing another installation's benchmarks package.
    """
    path = BENCHMARK_DIR.parent / "hle" / "prepare.py"
    spec = importlib.util.spec_from_file_location("_hle_prepare_for_hle_verified", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmarks/hle's prepare.py from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._build_input


def format_entry(row: dict, prompt_config: Any = None) -> dict:
    """Convert an unpacked dataset row to Gym JSONL format.

    When prompt_config is provided, materialize the input messages and image blocks.
    Use these rows with prompt_config: null at rollout time.
    """
    entry = {
        "question": row["question"],
        "expected_answer": row["answer"],
        # Metadata for analysis; not used for grading.
        "answer_type": row.get("answer_type"),
        "uuid": row["id"],
        "category": row.get("category"),
        "raw_subject": row.get("raw_subject"),
        "verified_class": VERIFIED_CLASSES_MAP.get(row.get("Verified_Classes"), row.get("Verified_Classes")),
    }
    if prompt_config is not None:
        build_input = _hle_build_input()

        entry["has_image"] = bool(row.get("image"))
        entry["responses_create_params"] = {
            "input": build_input(prompt_config, row["question"], row.get("image") or "")
        }
    return entry


def prepare(
    subset: str = DEFAULT_SUBSET,
    output_fpath: Optional[Union[str, Path]] = None,
    include_vision: bool = False,
) -> Path:
    """Download HLE-Verified and convert to Gym JSONL format.

    Args:
        subset: Verified classes to keep. "text" selects Gold + Revision;
            "gold", "revision", and "uncertain" select one class; "all" keeps all.
        output_fpath: Output JSONL path. Defaults to the text or vision dataset
            path under benchmarks/hle_verified/data/.
        include_vision: Include image questions and materialize input messages.
            Use the resulting dataset with prompt_config: null.

    Returns:
        Path to the written JSONL file.
    """
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {SUBSETS}")

    from datasets import load_dataset

    from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict

    print(f"Downloading {REPO_ID} from HuggingFace...")
    hf_token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)
    ds = load_dataset(REPO_ID, split="train", token=hf_token)

    prompt_config = None
    if include_vision:
        from nemo_gym.prompt import load_prompt_config

        prompt_config = load_prompt_config(str(PROMPT_CONFIG_FPATH))

    if output_fpath is not None:
        output_fpath = Path(output_fpath)
    else:
        output_fpath = DEFAULT_OUTPUT_VISION if include_vision else DEFAULT_OUTPUT
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    n_skipped = 0
    n_image = 0
    for raw_row in ds:
        row: dict[str, Any] = _unpack(raw_row)
        if not keep_row(row, subset, include_vision=include_vision):
            n_skipped += 1
            continue
        n_image += bool(row.get("image"))
        lines.append(json.dumps(format_entry(row, prompt_config), ensure_ascii=False) + "\n")

    with open(output_fpath, "w", encoding="utf-8") as f:
        f.writelines(lines)

    modality = f", including {n_image} image questions" if include_vision else ""
    print(f"Wrote {len(lines)} problems to {output_fpath} (skipped {n_skipped} outside subset {subset!r}{modality})")
    return output_fpath


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare HLE-Verified benchmark data.")
    parser.add_argument(
        "--subset",
        default=DEFAULT_SUBSET,
        choices=SUBSETS,
        help="Which verified classes to keep (default: text = Gold + Revision).",
    )
    parser.add_argument(
        "--include-vision",
        action="store_true",
        help="Keep image questions and materialize inputs (writes hle_verified_benchmark_vision.jsonl).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT}, or the _vision variant).",
    )
    args = parser.parse_args()
    prepare(subset=args.subset, output_fpath=args.output, include_vision=args.include_vision)
