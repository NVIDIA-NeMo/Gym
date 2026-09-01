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
"""
The prepare_* functions in this file are written to exactly match the input observed in the VLMEvalKit OpenAI API call.
"""

import base64
from collections import Counter
from pathlib import Path

import orjson
from app import VlmEvalKitResourcesServer
from pandas import DataFrame
from vlmeval.dataset.image_mcq import ImageMCQDataset
from vlmeval.dataset.image_vqa import OCRBench
from vlmeval.dataset.utils.multiple_choice import build_choices


_IMAGE_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def image_file_to_data_url(fpath: str) -> str:
    """Base64-encode a build_prompt-dumped image file into a data URL."""
    mime = _IMAGE_MIME_BY_SUFFIX.get(Path(fpath).suffix.lower(), "image/jpeg")
    with open(fpath, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode('ascii')}"


def segments_to_content_items(segments, detail: str = "high") -> list:
    """Convert mcore ``build_prompt`` segments into Responses API content items IN ORDER.

    Multi-image datasets (interleaved image/text segment lists) produce
    interleaved ``{'type': 'image'|'text', 'value': ...}`` segments; preserving that
    exact order is the whole point — do NOT flatten to one image + one text block.
    Image values are file paths dumped by ``build_prompt`` and are base64-encoded from
    disk. Empty-string text segments (some preps emit them when the question
    starts with an image tag) are dropped: they carry no content and some Responses
    API backends reject empty ``input_text`` items.
    """
    items = []
    for seg in segments:
        if seg["type"] == "image":
            items.append(
                {
                    "type": "input_image",
                    "image_url": image_file_to_data_url(seg["value"]),
                    "detail": detail,
                }
            )
        elif seg["type"] == "text":
            if seg["value"] != "":
                items.append({"type": "input_text", "text": seg["value"]})
        else:
            raise ValueError(f"Unsupported segment type: {seg['type']!r}")
    return items


def prepare_OCRBench():
    dataset_name = "OCRBench"

    data = OCRBench(dataset=dataset_name).load_data(dataset_name)

    print(f"Columns: {data.columns}")
    print(data.head())

    assert list(data.columns) == ["index", "image", "question", "answer", "category"]

    f = open(f"data/{dataset_name}_validation.jsonl", "wb")
    for _, vlmevalkit_row in data.iterrows():
        gym_row = {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{vlmevalkit_row['image']}",
                                "detail": "high",
                            },
                            {
                                "type": "input_text",
                                "text": vlmevalkit_row["question"],
                            },
                        ],
                    }
                ],
            },
            "answer": eval(vlmevalkit_row["answer"]),
            "category": vlmevalkit_row["category"],
            "benchmark_name": dataset_name,
        }
        f.write(orjson.dumps(gym_row) + b"\n")


def prepare_MMBench_DEV_EN_V11():
    dataset_name = "MMBench_DEV_EN_V11"

    dataset = ImageMCQDataset(dataset=dataset_name)
    data: DataFrame = dataset.load_data(dataset_name)

    print(f"""Columns: {data.columns}
Data:
{data}
Data head:
{data.head()}""")

    # From https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/dataset/utils/multiple_choice.py#L513
    get_group = lambda i: int(i % 1e6)
    group_counts = Counter(map(get_group, data["index"]))

    # We sort this dataset so that samples in a group are adjacent to each other rather than spread apart
    # At runtime, this data will be read in order and this results in much more efficient processing
    # This key is the same as get_group, just for a pd.Series
    data = data.sort_values("index", key=lambda i: i.astype(int) % 1e6)

    assert list(data.columns) == [
        "index",
        "question",
        "hint",
        "A",
        "B",
        "C",
        "D",
        "answer",
        "category",
        "image",
        "l2-category",
        "split",
    ]

    f = open(f"data/{dataset_name}_validation.jsonl", "wb")
    for _, vlmevalkit_row in data.iterrows():
        messages = dataset.build_prompt(vlmevalkit_row)

        group = get_group(vlmevalkit_row["index"])

        has_image = group == int(vlmevalkit_row["index"])
        if has_image:
            image = vlmevalkit_row["image"]
        if not has_image:  # Is not valid image, rather is an image reference
            image = data[data["index"] == int(vlmevalkit_row["image"])].iloc[0]["image"]

        gym_row = {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image}",
                                "detail": "high",
                            },
                            {
                                "type": "input_text",
                                "text": messages[-1]["value"],
                            },
                        ],
                    },
                ]
            },
            "answer": vlmevalkit_row["answer"],
            "category": vlmevalkit_row["category"],
            "benchmark_name": dataset_name,
            "group": group,
            "group_size": group_counts[group],
            # Choices is built here https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/dataset/utils/multiple_choice.py#L337
            "choices": build_choices(vlmevalkit_row),
        }
        f.write(orjson.dumps(gym_row) + b"\n")


def prepare_OCRBench_v2(output_fpath: str = "data/OCRBench_v2_validation.jsonl") -> Path:
    """Build the Gym JSONL for OCRBench v2 via the pinned VLMEvalKitMcore.

    The mcore class is the monolith ``OCRBench_v2`` (vlmeval/dataset/image_vqa.py:3513,
    dataset name 'OCRBench_v2'); prompts come from the inherited
    ``ImageBaseDataset.build_prompt`` (image + raw question). Carries EVERYTHING the
    per-type scoring dispatcher reads (``process_predictions``,
    vlmeval/dataset/utils/ocrbrnch_v2_eval.py:44): ``type`` (= the TSV ``category``
    column, stored as the Gym ``category``), ``question``, ``answers``, ``bbox``,
    ``content``, and ``eval`` when present. Field parsing mirrors
    ``OCRBench_v2.evaluate`` (image_vqa.py:3535-3557): answer/bbox/content are
    stringified Python literals with 'without bbox' / 'without content' /
    'without eval' sentinels.

    NOTE: the ~1.4 GB TSV gets LOCALIZED at load time (base64 images are written to
    ``<LMUData>/images/OCRBench_v2/`` and the DataFrame carries ``image_path``, NOT an
    ``image`` column), so the payload is built from the ``build_prompt`` segments via
    ``segments_to_content_items``, which base64-encodes the dumped files from disk.
    """
    import ast

    from vlmeval.dataset.image_vqa import OCRBench_v2

    dataset_name = "OCRBench_v2"

    dataset = OCRBench_v2(dataset=dataset_name)
    data: DataFrame = dataset.data

    print(f"Columns: {data.columns}")
    print(data.head())

    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_fpath, "wb") as f:
        for _, vlmevalkit_row in data.iterrows():
            # [image (localized file path), question text] — the images live on disk
            # (image_path column), so encode via segments_to_content_items.
            messages = dataset.build_prompt(vlmevalkit_row)
            content_items = segments_to_content_items(messages)

            # Literal-parsing mirrors OCRBench_v2.evaluate exactly (image_vqa.py:3535,
            # :3543-3544) so the verify request carries reference-identical inputs.
            answers = ast.literal_eval(vlmevalkit_row["answer"])
            bbox_raw = vlmevalkit_row["bbox"]
            bbox = ast.literal_eval(bbox_raw) if bbox_raw != "without bbox" else bbox_raw
            content_raw = vlmevalkit_row["content"]
            content = ast.literal_eval(content_raw) if content_raw != "without content" else content_raw

            gym_row = {
                "responses_create_params": {"input": [{"role": "user", "content": content_items}]},
                "answer": answers,
                # data_item['type'] in the reference dispatcher, e.g. 'cognition VQA en',
                # 'full-page OCR cn' — also the EN/ZH bucketing key of the official
                # aggregation (ocrbench_v2_aggregate_accuracy, ocrbrnch_v2_eval.py:360).
                "category": vlmevalkit_row["category"],
                "benchmark_name": dataset_name,
                "index": int(vlmevalkit_row["index"]),
                "question": vlmevalkit_row["question"],
                "bbox": bbox,
                "content": content,
            }
            # 'multiple choice' / 'case sensitive'; only set when present, mirroring
            # evaluate (:3556-3557).
            evals = vlmevalkit_row["eval"]
            if evals != "without eval":
                gym_row["eval"] = evals
            f.write(orjson.dumps(gym_row) + b"\n")

    return output_fpath


def _nan_to_none(row, key):
    value = row.get(key)
    return None if value is None or (isinstance(value, float) and value != value) else value


def prepare_OCR_Reasoning(output_fpath: str = "data/OCR_Reasoning_validation.jsonl") -> Path:
    """Build the Gym JSONL for OCR-Reasoning using the pinned VLMEvalKitMcore library.

    Prompt text comes from the mcore ``OCR_Reasoning.build_prompt`` (question + the
    language-dependent step-by-step instruction embedding the ``format`` column). Carries
    everything the reference scorer needs (vlmeval/dataset/utils/ocr_reasoning.py):
    ``reasoning`` (the reference chain rated by the judge, OcrR_auxeval:102), plus
    ``question_type``/``answer_type``/``choices``/``answer_option`` read by post_check
    (:68-94). ``category`` is the ``task`` column — OcrR_acc's per-task breakdown.
    Nano 3 Omni paper target (arXiv 2604.24954): 54.14 (reasoning-on).
    """
    from vlmeval.dataset.image_vqa import OCR_Reasoning

    dataset_name = "OCR_Reasoning"

    dataset = OCR_Reasoning(dataset=dataset_name)
    data: DataFrame = dataset.data

    print(f"Columns: {data.columns}")
    print(data.head())

    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_fpath, "wb") as f:
        for _, vlmevalkit_row in data.iterrows():
            messages = dataset.build_prompt(vlmevalkit_row)
            text = "\n".join(m["value"] for m in messages if m["type"] == "text")

            gym_row = {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{vlmevalkit_row['image']}",
                                    "detail": "high",
                                },
                                {
                                    "type": "input_text",
                                    "text": text,
                                },
                            ],
                        },
                    ]
                },
                "answer": vlmevalkit_row["answer"],
                "category": vlmevalkit_row["task"],
                "benchmark_name": dataset_name,
                "index": int(vlmevalkit_row["index"]),
                "question": vlmevalkit_row["question"],
                # Reference reasoning chain — judged for the reason_score (OcrR_auxeval).
                "reasoning": vlmevalkit_row["reasoning"],
                # post_check inputs; `choices` stays the raw TSV string (post_check evals it).
                "question_type": _nan_to_none(vlmevalkit_row, "question_type"),
                "answer_type": _nan_to_none(vlmevalkit_row, "answer_type"),
                "choices": _nan_to_none(vlmevalkit_row, "choices"),
                "answer_option": _nan_to_none(vlmevalkit_row, "answer_option"),
                # Prompt-construction metadata, kept for official-parity reconstruction.
                "language": _nan_to_none(vlmevalkit_row, "language"),
                "format": _nan_to_none(vlmevalkit_row, "format"),
            }
            f.write(orjson.dumps(gym_row) + b"\n")

    return output_fpath


if __name__ == "__main__":
    VlmEvalKitResourcesServer.setup_VLMEvalKit()

    prepare_OCRBench()
    prepare_MMBench_DEV_EN_V11()
    # prepare_OCRBench_v2() / prepare_OCR_Reasoning() are opt-in via
    # benchmarks/ocrbench_v2/prepare.py and benchmarks/ocr_reasoning/prepare.py.
