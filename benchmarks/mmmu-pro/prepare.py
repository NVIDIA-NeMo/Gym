# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MMMU-Pro (vision) benchmark data for NeMo Gym.

Ports NeMo-Skills' ``mmmu-pro`` dataset preparation. Images are written under
``data/media/<id>/`` and JSONL rows embed lightweight ``verifier_metadata.media_dir``
for ``labbench2_vlm_agent`` to inject ``input_image`` blocks at rollout time.
Verification uses the ``mcqa`` resource server (letter match, ``Answer:`` / markdown).
"""

import ast
import json
import uuid
from pathlib import Path

from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
MEDIA_ROOT = DATA_DIR / "media"
OUTPUT_FPATH = DATA_DIR / "mmmu-pro_benchmark.jsonl"


def _mcq_problem_from_options(options: list[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(options))]
    lines = [f"{letters[i]}) {options[i]}" for i in range(len(options))]
    return "\n\n".join(lines)


def format_entry(entry: dict, images_root: Path) -> dict | None:
    if entry.get("image") is None:
        return None

    media_key = str(uuid.uuid4())
    media_dir = images_root / media_key
    media_dir.mkdir(parents=True, exist_ok=True)
    image_filename = "question.png"
    image_path = media_dir / image_filename
    entry["image"].save(image_path)

    options_list = ast.literal_eval(entry["options"])
    options_text = _mcq_problem_from_options(options_list)
    letters = [chr(ord("A") + i) for i in range(len(options_list))]
    options = [{letters[i]: options_list[i]} for i in range(len(options_list))]
    subject = str(entry["subject"]).replace(" ", "_")
    letter = str(entry["answer"]).strip().upper()
    stem = (entry.get("question") or "").strip()
    problem = f"{stem}\n\n{options_text}".strip()
    user_text = (
        "Answer the following multiple choice question. The last line of your response "
        "should be in the following format: 'Answer: A/B/C/D/E/F/G/H/I/J' (e.g. 'Answer: A').\n\n"
        f"{problem}"
    )

    return {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                }
            ]
        },
        "verifier_metadata": {"media_dir": (Path("media") / media_key).as_posix()},
        "options": options,
        "expected_answer": letter,
        "grading_mode": "lenient_answer_colon_md",
        "subset_for_metrics": subject,
        "id": entry.get("id"),
    }


def prepare() -> Path:
    from datasets import load_dataset
    from tqdm.auto import tqdm

    hf_token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading MMMU/MMMU_Pro vision test split...")
    dataset = load_dataset("MMMU/MMMU_Pro", "vision", split="test", token=hf_token)

    count = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for entry in tqdm(dataset, desc="MMMU-Pro"):
            row = format_entry(entry, MEDIA_ROOT)
            if row is None:
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} entries to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
