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
"""Turn GEOBench-VLM records into NemoGym rows.

The released evaluator reads ``Single/qa.json`` and ``Temporal/qa.json``.
Hugging Face publishes the single-image split as
``aialliance/GEOBench-VLM``, config ``single``, 3,211 questions. Each
question can carry several prompts. The paper's "10,000 instructions" count
is that expanded prompt set. This module keeps one row per prompt.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MCQ_INSTRUCTION = (
    "For the given the Multiple Choice Question Answer below, analyze the "
    "question and answer strictly from one of the options below. Strictly "
    "answer the choice only. No additional text. Provide only the letter "
    "(A., B., C., D. or E.) corresponding to the correct answer for the "
    "multiple-choice question given."
)

# 1x1 PNG used by the committed example rows. The benchmark images stay in
# the upstream dataset and are not redistributed here.
EXAMPLE_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def _data_url(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def image_block(source: str | bytes) -> dict[str, str]:
    if isinstance(source, bytes):
        url = _data_url(source)
    elif source.startswith("data:") or source.startswith("http://") or source.startswith("https://"):
        url = source
    else:
        path = Path(source)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        url = _data_url(path.read_bytes(), mime)
    return {"type": "input_image", "image_url": url, "detail": "high"}


def prompt_text(record: dict[str, Any], prompt: str) -> str:
    description = (record.get("cls_description") or "").strip()
    options = record.get("options") or ""
    if record.get("ground_truth_option") or options:
        choices = f"Options: {options}" if options else ""
        parts = [MCQ_INSTRUCTION, description, prompt, choices]
        return "\n".join(part for part in parts if part)
    return prompt


def content_blocks(record: dict[str, Any], prompt: str, images: Iterable[str | bytes]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    image_list = list(images)
    if len(image_list) >= 2:
        blocks.append(
            {
                "type": "input_text",
                "text": "The following images show the condition before and after the event.",
            }
        )
        blocks.append({"type": "input_text", "text": "This is the 'pre' image:"})
        blocks.append(image_block(image_list[0]))
        blocks.append({"type": "input_text", "text": "This is the 'post' image:"})
        blocks.append(image_block(image_list[-1]))
    elif image_list:
        blocks.append(image_block(image_list[0]))
    blocks.append({"type": "input_text", "text": prompt_text(record, prompt)})
    return blocks


def dataset_row(
    record: dict[str, Any],
    *,
    prompt_index: int = 0,
    split: str = "single",
    images: Iterable[str | bytes] | None = None,
) -> dict[str, Any]:
    prompts = list(record.get("prompts") or [record.get("question") or ""])
    prompt = prompts[prompt_index]
    question_id = record.get("question_id", record.get("Question_ID", ""))
    image_sources = list(images) if images is not None else _record_images(record)
    task = str(record.get("task") or record.get("Task") or "")
    return {
        "activity": "geobench_vlm",
        "split": split,
        "task": task,
        "question_id": str(question_id),
        "prompt_index": prompt_index,
        "ground_truth": str(record.get("ground_truth") or record.get("Tr_Answer") or ""),
        "ground_truth_option": str(record.get("ground_truth_option") or record.get("Ground_Truth_Option") or ""),
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": content_blocks(record, prompt, image_sources),
                }
            ],
            "temperature": 0,
            "max_output_tokens": 128,
        },
    }


def _record_images(record: dict[str, Any]) -> list[str | bytes]:
    image = record.get("image")
    path = record.get("image_path") or record.get("Image")
    if isinstance(path, list):
        return [str(item) for item in (path[:1] + path[-1:])]
    if isinstance(image, (bytes, str)) and image:
        return [image]
    if path:
        return [str(path)]
    return []


def iter_rows(records: Iterable[dict[str, Any]], *, split: str) -> Iterable[dict[str, Any]]:
    for record in records:
        prompts = list(record.get("prompts") or [""])
        for index in range(len(prompts)):
            yield dataset_row(record, prompt_index=index, split=split)


def load_qa_tree(root: Path) -> list[dict[str, Any]]:
    """Read the official on-disk layout: ``Single/qa.json`` and ``Temporal/qa.json``."""
    rows: list[dict[str, Any]] = []
    for split, filename in (("single", "Single/qa.json"), ("temporal", "Temporal/qa.json")):
        path = root / filename
        if not path.exists():
            continue
        records = json.loads(path.read_text())
        for record in records:
            image_path = record.get("image_path")
            if isinstance(image_path, list):
                record = {
                    **record,
                    "image_path": [str(root / item) for item in image_path],
                }
            elif image_path:
                record = {**record, "image_path": str(root / image_path)}
            rows.extend(iter_rows([record], split=split))
    if not rows:
        raise FileNotFoundError(f"No questions under {root}; expected Single/qa.json or Temporal/qa.json")
    return rows


HF_DATASET = "aialliance/GEOBench-VLM"
HF_REVISION = "b3485fe4971213ee13f91960638ddc5c4a098e70"
SINGLE_PARQUET_SHA256 = "81bb88e17456a7c32be157a49e07be38f51de24e80c14c0ccf8f3d78f0d15a98"

# The 1,557-question perception subset: eight single-image classification and
# counting tasks, with their question counts at HF_REVISION. "Sptatial" is the
# upstream task name.
PERCEPTION_COUNTS = {
    "Sptatial Relation Classification": 466,
    "Aircraft Type Classification": 100,
    "Ship Type Classification": 207,
    "General Aircraft Counting": 100,
    "Specific Aircraft Type Counting": 140,
    "General Vehicle Counting": 150,
    "Specific Vehicle Type Counting": 224,
    "Building Counting": 170,
}
PERCEPTION_TASKS = tuple(PERCEPTION_COUNTS)
PERCEPTION_INSTRUCTION = "Answer with only the option letter."


def _jpeg_data_url(raw: bytes) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.open(io.BytesIO(raw)).convert("RGB").save(buffer, format="JPEG", quality=92)
    return _data_url(buffer.getvalue(), "image/jpeg")


def perception_row(record: dict[str, Any], *, blind: bool = False) -> dict[str, Any]:
    """One row per question (first prompt), text before the image; ``blind`` drops the image."""
    question_id = record.get("question_id")
    if not record.get("prompts") or not record.get("options") or not record.get("ground_truth_option"):
        raise ValueError(f"Question {question_id} is missing its prompt, options, or gold option")
    text = f"{record['prompts'][0]}\n{record['options']}\n{PERCEPTION_INSTRUCTION}"
    content: list[dict[str, str]] = [{"type": "input_text", "text": text}]
    if not blind:
        image = (record.get("image") or {}).get("bytes")
        if not image:
            raise ValueError(f"Question {question_id} has no image bytes")
        content.append({"type": "input_image", "image_url": _jpeg_data_url(image), "detail": "high"})
    return {
        "activity": "geobench_vlm",
        "split": "single",
        "task": record["task"],
        "question_id": str(record["question_id"]),
        "prompt_index": 0,
        "ground_truth": record["ground_truth"],
        "ground_truth_option": record["ground_truth_option"],
        "responses_create_params": {
            "input": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_output_tokens": 8192,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_perception(parquet: Path, data_dir: Path, *, check_sha256: bool = True) -> list[Path]:
    """Write ``perception_1557.jsonl`` and its no-image ``perception_1557_blind.jsonl``
    from a local copy of the Hugging Face ``Single.parquet`` at ``HF_REVISION``."""
    import pyarrow.parquet as pq

    if check_sha256 and (actual := _sha256(parquet)) != SINGLE_PARQUET_SHA256:
        raise ValueError(
            f"{parquet} has SHA-256 {actual}, expected {SINGLE_PARQUET_SHA256} "
            f"({HF_DATASET} at revision {HF_REVISION}). Pass check_sha256=False to build anyway."
        )
    records = [r for r in pq.read_table(parquet).to_pylist() if r["task"] in PERCEPTION_COUNTS]
    counts = Counter(r["task"] for r in records)
    if counts != Counter(PERCEPTION_COUNTS):
        raise ValueError(f"Perception task counts {dict(counts)} differ from {PERCEPTION_COUNTS}")
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, blind in (("perception_1557.jsonl", False), ("perception_1557_blind.jsonl", True)):
        path = data_dir / name
        with path.open("w") as stream:
            for record in records:
                stream.write(json.dumps(perception_row(record, blind=blind), separators=(",", ":")) + "\n")
        paths.append(path)
    return paths


def example_records() -> list[dict[str, Any]]:
    image = f"data:image/png;base64,{EXAMPLE_PNG}"
    return [
        {
            "question_id": 1,
            "task": "Scene Classification",
            "cls_description": "Overhead optical image.",
            "prompts": ["What scene is shown?"],
            "options": "A. Forest  B. Harbor  C. Desert  D. Snow  E. Airport",
            "options_list": ["Forest", "Harbor", "Desert", "Snow", "Airport"],
            "ground_truth": "Harbor",
            "ground_truth_option": "B",
            "image": image,
            "split": "single",
        },
        {
            "question_id": 2,
            "task": "Object Counting",
            "cls_description": "Count the aircraft.",
            "prompts": ["How many aircraft are visible?"],
            "options": "A. 0  B. 1  C. 3  D. 10  E. 50",
            "options_list": ["0", "1", "3", "10", "50"],
            "ground_truth": "3",
            "ground_truth_option": "C",
            "image": image,
            "split": "single",
        },
        {
            "question_id": 3,
            "task": "Disaster Type Classification",
            "cls_description": "Before and after the event.",
            "prompts": ["What event changed this area?"],
            "options": "A. Fire  B. Flood  C. Earthquake  D. Drought  E. Volcano",
            "options_list": ["Fire", "Flood", "Earthquake", "Drought", "Volcano"],
            "ground_truth": "Flood",
            "ground_truth_option": "B",
            "image_path": [image, image],
            "split": "temporal",
        },
        {
            "question_id": 4,
            "task": "Land Use Classification",
            "cls_description": "Classify the dominant land use.",
            "prompts": ["What is the dominant land use?"],
            "options": "A. Residential  B. Industrial  C. Farmland  D. Water  E. Forest",
            "options_list": ["Residential", "Industrial", "Farmland", "Water", "Forest"],
            "ground_truth": "Farmland",
            "ground_truth_option": "C.",
            "image": image,
            "split": "single",
        },
        {
            "question_id": 5,
            "task": "Object Classification",
            "cls_description": "Identify the vessel class.",
            "prompts": ["Which vessel class is shown?", "Name the vessel class in this crop."],
            "options": "A. Cargo  B. Fishing  C. Destroyer  D. Tanker  E. Sailboat",
            "options_list": ["Cargo", "Fishing", "Destroyer", "Tanker", "Sailboat"],
            "ground_truth": "Destroyer",
            "ground_truth_option": "C",
            "image": image,
            "split": "single",
        },
    ]


def example_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in example_records():
        split = record["split"]
        rows.extend(iter_rows([record], split=split))
    return rows


def materialize(path: Path, rows: Iterable[dict[str, Any]] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen = list(rows) if rows is not None else example_rows()
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in chosen))
    return path


def main() -> None:
    destination = Path(__file__).parent / "data" / "example.jsonl"
    materialize(destination)
    print(f"wrote {sum(1 for _ in destination.open())} rows to {destination}")


if __name__ == "__main__":
    main()
