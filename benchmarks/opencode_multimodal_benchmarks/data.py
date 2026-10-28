# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted material: Apache-2.0, Copyright 2023 VLMEvalKit Authors.
# Source: https://github.com/open-compass/VLMEvalKit/blob/ebac2d9bda8fa28dc426a3b2ccc8f53c886dc0c5/vlmeval/dataset/babyvision.py
# Full license: resources_servers/opencode_multimodal_benchmarks/third-party-notices.md
"""Prepare visual tasks without importing a model or VLMEvalKit."""

import ast
import base64
import csv
import hashlib
import io
import json
import re
import string
import sys
from pathlib import Path

from PIL import Image


BENCHMARKS = ("VTCBench", "VStarBench", "HRBench8K", "BabyVision")
ANSWER_INSTRUCTION = (
    "End your response with a separate line in this exact format:\n"
    "Answer: <final answer>\n"
    "For multiple-choice questions, use only the option letter. Otherwise, "
    "use only the requested word, phrase, number, or expression."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def options_from_row(row: dict) -> dict[str, str]:
    return {key: row[key].strip() for key in string.ascii_uppercase if row.get(key, "").strip()}


def question(row: dict, benchmark: str, answer_format: bool) -> str:
    text = row["question"].strip()
    options = options_from_row(row)
    if benchmark == "BabyVision":
        # Original final-answer instruction (also in VLMEvalKit babyvision.py):
        # https://github.com/UniPat-AI/BabyVision/blob/7f92fd4b1dc1c68b7b936a9bc09c68b4a944a55a/babyvision_eval/evaluate_model.py
        text += "\nThink about the question and give your final answer in \\boxed{Answer} format."
    else:
        text = f"Question: {text}\n"
        if row.get("hint", "").strip():
            text = f"Hint: {row['hint'].strip()}\n" + text
        if options:
            text += "Options:\n" + "".join(f"{key}. {value}\n" for key, value in options.items())
            text += "Please select the correct answer from the options above."
            if benchmark == "VTCBench":
                text += " Answer with the option letter only."
        elif benchmark == "VTCBench":
            text += "Answer the question using a single word or phrase."
    if answer_format and benchmark == "VTCBench":
        text += "\n\n" + ANSWER_INSTRUCTION
    return text


def _values(value: str) -> list[str]:
    if value.startswith("["):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("Image lists must contain strings")
        return parsed
    return [value]


def _store_image(raw: bytes, root: Path) -> Path:
    with Image.open(io.BytesIO(raw)) as image:
        extension = {
            "JPEG": ".jpg",
            "MPO": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
        }.get(image.format)
        if extension is None:
            raise ValueError(f"Unsupported image format: {image.format}")
        image.verify()
    destination = root / (hashlib.sha256(raw).hexdigest() + extension)
    if not destination.exists():
        destination.write_bytes(raw)
    return destination


def _image(value: str, source_root: Path, images: Path) -> Path:
    if value.startswith("data:image/"):
        raw = base64.b64decode(value.split(",", 1)[1], validate=True)
    else:
        if len(value) >= 64 and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", value):
            raw = base64.b64decode(value, validate=True)
        else:
            path = Path(value)
            path = path if path.is_absolute() else source_root / path
            raw = path.read_bytes()
    return _store_image(raw, images)


def prepare(
    tsv: Path,
    benchmark: str,
    output: Path,
    *,
    image_root: Path | None = None,
    answer_format: bool = True,
) -> tuple[list[dict], dict]:
    """Decode images unchanged and retain labels outside agent workspaces."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    answer_format = answer_format and benchmark == "VTCBench"
    csv.field_size_limit(sys.maxsize)
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    source_root = image_root or tsv.parent
    existing = output / "dataset.json"
    source_hash = sha256_file(tsv)
    if existing.exists():
        provenance = json.loads(existing.read_text())
        if (
            provenance["dataset_sha256"] != source_hash
            or provenance["benchmark"] != benchmark
            or provenance["answer_format_instruction"] != answer_format
            or provenance["image_root"] != str(source_root.resolve())
        ):
            raise ValueError("Prepared dataset differs; use a new output directory")
        manifest = output / "tasks.jsonl"
        if provenance["tasks_sha256"] != sha256_file(manifest):
            raise ValueError(f"Prepared task manifest changed: {manifest}")
        tasks = [json.loads(line) for line in manifest.read_text().splitlines()]
        for image in {path for task in tasks for path in task["images"]}:
            path = Path(image)
            if sha256_file(path) != path.stem:
                raise ValueError(f"Prepared image content changed: {path}")
        return tasks, provenance
    tasks = []
    image_by_id: dict[str, list[str]] = {}
    references = {}
    with tsv.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = set(reader.fieldnames or [])
        if not {"index", "question", "answer"} <= fields:
            raise ValueError(f"Missing task columns in {tsv}: {fields}")
        if not {"image", "image_path"} & fields:
            raise ValueError(f"Missing image column in {tsv}")
        if benchmark == "HRBench8K" and "cycle_category" not in fields:
            raise ValueError("HRBench8K requires cycle_category")
        for row in reader:
            if benchmark == "HRBench8K" and not row["cycle_category"].strip():
                raise ValueError("HRBench8K requires nonempty cycle_category")
            identifier = row["index"].strip()
            if not identifier or identifier in image_by_id:
                raise ValueError(f"Empty or duplicate sample ID: {identifier!r}")
            if not row["question"].strip() or not row["answer"].strip():
                raise ValueError(f"Missing question/answer in sample {identifier}")
            image_by_id[identifier] = []
            paths = row.get("image_path", "").strip()
            raw_images = row.get("image", "").strip()
            local_paths = _values(paths) if paths else []
            use_paths = local_paths and all((source_root / path).is_file() for path in local_paths)
            values = local_paths if use_paths else _values(raw_images or paths)
            if len(values) == 1 and values[0].isdigit():
                references[identifier] = values[0]
            else:
                image_by_id[identifier] = [str(_image(value, source_root, images).resolve()) for value in values]
            tasks.append(
                {
                    "index": identifier,
                    "benchmark": benchmark,
                    "question": row["question"],
                    "prompt": question(row, benchmark, answer_format),
                    "answer": row["answer"],
                    "options": options_from_row(row),
                    "category": row.get("category", ""),
                    "l2_category": row.get("l2_category", row.get("l2-category", "")),
                    "ans_type": row.get("ans_type", ""),
                    "cycle_category": row.get("cycle_category", ""),
                    "split": row.get("split", "none"),
                }
            )
    for task in tasks:
        identifier = task["index"]
        seen = set()
        resolved = identifier
        while resolved in references:
            if resolved in seen:
                raise ValueError(f"Cyclic image reference for {identifier}")
            seen.add(resolved)
            resolved = references[resolved]
        if resolved not in image_by_id or not image_by_id[resolved]:
            raise ValueError(f"Unresolved image reference for {identifier}")
        task["images"] = image_by_id[resolved]
    if not tasks:
        raise ValueError(f"Empty dataset: {tsv}")
    manifest = output / "tasks.jsonl"
    temporary = manifest.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for task in tasks:
            stream.write(json.dumps(task, ensure_ascii=False) + "\n")
    temporary.replace(manifest)
    provenance = {
        "benchmark": benchmark,
        "dataset": str(tsv.resolve()),
        "dataset_sha256": source_hash,
        "tasks_sha256": sha256_file(manifest),
        "samples": len(tasks),
        "answer_format_instruction": answer_format,
        "image_preprocessing": "original bytes; no resize",
        "image_root": str(source_root.resolve()),
    }
    atomic_json(output / "dataset.json", provenance)
    return tasks, provenance
