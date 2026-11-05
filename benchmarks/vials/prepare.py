# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download pinned VIALS data and prepare direct multimodal Gym JSONL.

Conversion follows VIALS's MIT-licensed vials/download.py at commit
002a0d25031f02a760da5d14f2c29bbd26eb06e4. See UPSTREAM_LICENSE.
The direct prompt is materialized from prompt.yaml. Images are embedded as data
URLs in responses_create_params.input; ground truth remains outside model input.
"""

import base64
import html
import json
import math
import mimetypes
from collections.abc import Iterable
from pathlib import Path

from nemo_gym.prompt import PromptConfig, fill_prompt, load_prompt_config


BENCHMARK_DIR = Path(__file__).resolve().parent
GYM_ROOT = BENCHMARK_DIR.parents[1]
OUTPUT_PATH = BENCHMARK_DIR / "data" / "vials.jsonl"
HF_REPO = "handshake-ai-research/VIALS"
HF_REVISION = "9657bf84c6f6e470e1c31123afe3f9152c3655bd"
HF_CONFIG = "tasks"
HF_SPLIT = "test"
PROMPT_PATH = BENCHMARK_DIR / "prompt.yaml"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _numeric_bound(value: float | str | None) -> float | None:
    if value is None:
        return None
    bound = float(value)
    if math.isnan(bound):
        return None
    if not math.isfinite(bound):
        raise ValueError(f"Non-finite numeric answer bound: {value!r}")
    return bound


def _convert_row(row: dict, *, image_dir: Path) -> dict:
    task_id = row["id"]
    if not task_id or Path(task_id).name != task_id or task_id in {".", ".."}:
        raise ValueError(f"Invalid VIALS task ID: {task_id!r}")
    metadata = row["metadata"]
    images = row["images"]
    filenames = metadata["image_filenames"]
    if not images or len(images) != len(filenames):
        raise ValueError(f"Task {task_id}: image and filename counts must match and be nonzero")

    task_dir = image_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    seen_names = set()
    for index, (image, declared_name) in enumerate(zip(images, filenames, strict=True), start=1):
        fallback = f"{task_id}.png" if len(images) == 1 else f"{task_id}_{index}.png"
        filename = Path((declared_name or "").strip()).name
        if filename in {"", ".", ".."}:
            filename = fallback
        if Path(filename).suffix.lower() not in _IMAGE_EXTENSIONS:
            filename += ".png"
        if filename in seen_names:
            raise ValueError(f"Task {task_id}: duplicate image filename after normalization: {filename}")
        seen_names.add(filename)
        payload = image["bytes"]
        if not isinstance(payload, bytes):
            raise ValueError(f"Task {task_id}: image {index} has no embedded image bytes")
        path = task_dir / filename
        if not path.exists() or path.read_bytes() != payload:
            path.write_bytes(payload)
        # Default paths are portable within a Gym checkout; external outputs use absolute paths.
        image_paths.append(str(path.relative_to(GYM_ROOT) if path.is_relative_to(GYM_ROOT) else path))

    return {
        "task_id": task_id,
        "question": html.unescape(row["question"]),
        "expected_answer": html.unescape(row["answer"]),
        "image_paths": image_paths,
        "answer_lower_bound": _numeric_bound(metadata.get("answer_lower_bound")),
        "answer_upper_bound": _numeric_bound(metadata.get("answer_upper_bound")),
        "domain": row["domain"],
        "secondary_tags": list(metadata.get("secondary_tags") or []),
    }


def _write_rows(rows: Iterable[dict], output: Path, *, prompt_config: PromptConfig) -> None:
    """Embed all panels in order and write self-contained Responses API inputs."""
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            # Only the question can fill the model prompt; grading fields stay outside input.
            messages = fill_prompt(prompt_config, {"question": row["question"]})
            images = []
            for image_path in row["image_paths"]:
                path = Path(image_path)
                if not path.is_absolute():
                    path = GYM_ROOT / path
                media_type = mimetypes.guess_type(path.name)[0] or "image/png"
                encoded = base64.standard_b64encode(path.read_bytes()).decode("ascii")
                images.append(
                    {"type": "input_image", "image_url": f"data:{media_type};base64,{encoded}", "detail": "auto"}
                )
            for message in messages:
                if message["role"] == "user":
                    # Match upstream VIALS ordering: every panel, then the question and answer instructions.
                    message["content"] = [*images, {"type": "input_text", "text": message["content"]}]
            prepared_row = {**row, "responses_create_params": {"input": messages}}
            # Write directly so Arrow doesn't insert null keys into mixed text/image content blocks.
            stream.write(json.dumps(prepared_row, ensure_ascii=False, allow_nan=False) + "\n")


def prepare(*, output: Path = OUTPUT_PATH) -> Path:
    """Write the full Gym dataset and up to five representative example rows."""
    from datasets import Image, load_dataset

    output = Path(output).resolve()
    example_path = output.parent / "example.jsonl"
    if output == example_path:
        raise ValueError("Full dataset output must differ from example.jsonl")
    prompt_config = load_prompt_config(str(PROMPT_PATH))
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(HF_REPO, HF_CONFIG, split=HF_SPLIT, revision=HF_REVISION)
    ids = dataset["id"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("VIALS must contain at least one task, with unique task IDs")
    dataset = dataset.cast_column("images", [Image(decode=False)])
    prepared = dataset.map(
        lambda row: _convert_row(row, image_dir=output.parent / "images"),
        remove_columns=dataset.column_names,
        # Mapping writes files: Arrow cache reuse must not skip missing-image restoration.
        load_from_cache_file=False,
        desc="Preparing VIALS tasks and images",
    )
    _write_rows(prepared, output, prompt_config=prompt_config)

    # Prefer one single-image, one multi-panel, and one numeric-bound task.
    selected = []
    for matches in (
        lambda row: len(row["image_paths"]) == 1,
        lambda row: len(row["image_paths"]) > 1,
        lambda row: row["answer_lower_bound"] is not None or row["answer_upper_bound"] is not None,
    ):
        for index, row in enumerate(prepared):
            if index not in selected and matches(row):
                selected.append(index)
                break
    for index in range(len(prepared)):
        if len(selected) >= min(5, len(prepared)):
            break
        if index not in selected:
            selected.append(index)
    _write_rows(prepared.select(selected), example_path, prompt_config=prompt_config)
    print(f"Prepared {len(prepared)} VIALS tasks: {output}; {len(selected)} examples: {example_path}")
    return output


if __name__ == "__main__":
    prepare()
