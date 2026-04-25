# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MMMU-Pro for NeMo Gym.

``format_entry`` matches NeMo-Skills ``nemo_skills/dataset/mmmu-pro/prepare.py``.
Adds Gym-only fields for ``labbench2_vlm_agent`` + ``mcqa``: each sample gets a
unique subdirectory under ``data/images/<safe_id>/`` with ``question.png`` (moved
from NeMo's flat ``images/<id>.png`` so the embedder loads one image per row).
"""

import ast
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional


# Allow `python prepare.py` from this directory (without an editable `pip install`).
_GYM_ROOT = Path(__file__).resolve().parents[2]
if str(_GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(_GYM_ROOT))


def _hf_token() -> Optional[str]:
    """HuggingFace token: env first, else NeMo Gym global config (if importable)."""
    t = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if t:
        return t
    try:
        from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict
    except Exception:
        return None
    return get_global_config_dict().get(HF_TOKEN_KEY_NAME)


def get_mcq_fields(question: str, choices: list) -> dict:
    """Same as NeMo-Skills ``nemo_skills/dataset/utils.get_mcq_fields``."""
    options_dict = {chr(ord("A") + i): option for i, option in enumerate(choices)}
    options_text = "\n".join(f"{letter}) {option}" for letter, option in options_dict.items())
    question = question.strip("\n")
    return {
        "problem": f"{question}\n\n{options_text}",
        "options": options_text,
        **options_dict,
    }


def format_entry(entry: dict, images_dir: Path) -> dict | None:
    """NeMo-Skills ``mmmu-pro/prepare.format_entry``."""
    if entry["image"] is None:
        return None

    image_filename = f"{entry['id']}.png"
    image_path = images_dir / image_filename
    entry["image"].save(image_path)

    options = ast.literal_eval(entry["options"])
    mcq_fields = get_mcq_fields("", options)
    subject = entry["subject"].replace(" ", "_")

    return {
        "problem": mcq_fields["problem"],
        "image_path": f"images/{image_filename}",
        "expected_answer": entry["answer"],
        "subset_for_metrics": subject,
        "id": entry["id"],
        "_options_list": options,
    }


def _safe_fs_id(raw_id) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(raw_id))


def _to_gym_row(ne: dict, images_dir: Path) -> dict:
    """NeMo row -> Gym JSONL."""
    safe = _safe_fs_id(ne["id"])
    flat = images_dir / f"{ne['id']}.png"
    per_dir = images_dir / safe
    per_dir.mkdir(parents=True, exist_ok=True)
    dest = per_dir / "question.png"
    if flat.exists():
        shutil.move(str(flat), str(dest))

    options_list = ne["_options_list"]
    letters = [chr(ord("A") + i) for i in range(len(options_list))]
    options = [{letters[i]: options_list[i]} for i in range(len(options_list))]
    letter = str(ne["expected_answer"]).strip().upper()
    user_text = (
        "Answer the following multiple choice question. The last line of your response "
        "should be in the following format: 'Answer: A/B/C/D/E/F/G/H/I/J' (e.g. 'Answer: A').\n\n"
        f"{ne['problem']}"
    )

    out = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                }
            ]
        },
        "verifier_metadata": {"media_dir": (Path("images") / safe).as_posix()},
        "options": options,
        "expected_answer": letter,
        "grading_mode": "lenient_answer_colon_md",
        "subset_for_metrics": ne["subset_for_metrics"],
        "id": ne["id"],
        "image_path": ne["image_path"],
        "problem": ne["problem"],
    }
    return out


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
OUTPUT_FPATH = DATA_DIR / "mmmu-pro_benchmark.jsonl"


def prepare() -> Path:
    from datasets import load_dataset
    from tqdm.auto import tqdm

    hf_token = _hf_token()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MMMU/MMMU_Pro vision test split...")
    dataset = load_dataset("MMMU/MMMU_Pro", "vision", split="test", token=hf_token)

    count = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for entry in tqdm(dataset, desc="MMMU-Pro"):
            ne = format_entry(entry, IMAGES_DIR)
            if ne is None:
                continue
            row = _to_gym_row(ne, IMAGES_DIR)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} entries to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
