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

"""Prepare HLE evaluation data for NeMo Gym.

Downloads Humanity's Last Exam (HLE) from HuggingFace and converts to Gym JSONL
format compatible with the equivalence_llm_judge resource server.

By default only text-only questions are included (image questions are filtered
out). Pass ``include_vision=True`` to also include the multimodal (image)
subset. In vision mode every row is fully materialized: the prompt template is
baked into ``responses_create_params.input`` and image questions carry an
``input_image`` block, so the vision dataset is self-contained and needs no
``prompt_config`` at rollout time (see ``config.yaml``).
"""

from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "hle_benchmark.jsonl"
OUTPUT_VISION_FPATH = DATA_DIR / "hle_benchmark_vision.jsonl"

# Prompt template used to materialize inputs in vision mode. Kept in sync with the
# text-only path, which applies this same template at rollout time via prompt_config.
PROMPT_CONFIG_FPATH = BENCHMARK_DIR / "prompts" / "default.yaml"


def _to_data_uri(image: str) -> str:
    """Normalize an HLE image field into a data URI usable as an ``image_url``.

    The ``cais/hle`` image column stores a base64 data URI string (e.g.
    ``data:image/png;base64,...``) for image questions. If a raw base64 payload
    is encountered instead, wrap it with a PNG data-URI prefix.
    """
    if image.startswith("data:"):
        return image
    return f"data:image/png;base64,{image}"


def _build_input(prompt_config, question: str, image: str) -> list:
    """Build a materialized ``responses_create_params.input`` for one row.

    Applies the shared prompt template, then (for image questions) rewrites the
    user turn into a multimodal content list with an ``input_image`` block.
    """
    from nemo_gym.prompt import fill_prompt

    messages = fill_prompt(prompt_config, {"question": question})
    if not image:
        return messages

    # Rewrite the (string) user turn into a multimodal content list so the image
    # travels alongside the question text.
    for msg in messages:
        if msg["role"] == "user":
            msg["content"] = [
                {"type": "input_text", "text": msg["content"]},
                {"type": "input_image", "image_url": _to_data_uri(image), "detail": "high"},
            ]
            break
    return messages


def prepare(include_vision: bool = False) -> Path:
    """Download HLE test data and convert to Gym JSONL format.

    Args:
        include_vision: When ``False`` (default), only text-only questions are
            kept and rows carry raw fields (``question``/``expected_answer``/...)
            to be templated at rollout time via ``prompt_config``. When ``True``,
            the full dataset (text + image questions) is included and every row
            is fully materialized with ``responses_create_params.input`` (image
            questions carry an ``input_image`` block); use with ``prompt_config:
            null``.

    Returns:
        Path to the written JSONL file.
    """
    import json

    from datasets import load_dataset

    from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict

    print("Downloading HLE from HuggingFace...")
    hf_token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)
    ds = load_dataset("cais/hle", split="test", token=hf_token)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    prompt_config = None
    if include_vision:
        from nemo_gym.prompt import load_prompt_config

        prompt_config = load_prompt_config(str(PROMPT_CONFIG_FPATH))

    # Use arrow format to avoid decoding the Image column, which requires Pillow.
    # In arrow format the image column is a raw string: empty for text-only questions,
    # base64 data (a data URI) for image questions.
    rows = []
    n_image = 0
    for batch in ds.with_format("arrow").iter(batch_size=500):
        for i in range(batch.num_rows):
            image = batch.column("image")[i].as_py()
            if image:
                n_image += 1
                if not include_vision:
                    continue

            question = batch.column("question")[i].as_py()
            row = {
                "question": question,
                "expected_answer": batch.column("answer")[i].as_py(),
                "answer_type": batch.column("answer_type")[i].as_py(),  # not used for grading; useful for analysis
                "uuid": batch.column("id")[i].as_py(),
            }
            if include_vision:
                # Materialize the prompt so the row is self-contained (no prompt_config needed).
                row["has_image"] = bool(image)
                row["responses_create_params"] = {"input": _build_input(prompt_config, question, image)}
            rows.append(json.dumps(row) + "\n")

    output_fpath = OUTPUT_VISION_FPATH if include_vision else OUTPUT_FPATH
    with open(output_fpath, "w") as f:
        f.writelines(rows)

    if include_vision:
        print(f"Wrote {len(rows)} problems to {output_fpath} (including {n_image} image questions)")
    else:
        print(f"Wrote {len(rows)} problems to {output_fpath} (skipped {n_image} image questions)")
    return output_fpath


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare HLE benchmark data.")
    parser.add_argument(
        "--include-vision",
        action="store_true",
        help="Include the multimodal (image) subset and materialize inputs (writes hle_benchmark_vision.jsonl).",
    )
    args = parser.parse_args()
    prepare(include_vision=args.include_vision)
