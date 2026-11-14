# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the pinned VIALS dataset directly into Harbor task folders.

Image handling adapts the MIT-licensed upstream vials/download.py at
002a0d25031f02a760da5d14f2c29bbd26eb06e4. See ../UPSTREAM_LICENSE.
"""

import argparse
import html
import json
import math
import re
import shutil
from pathlib import Path
from string import Template

from nemo_gym.prompt import load_prompt_config


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = Path(__file__).parent / "task-template"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "data" / "harbor"
HF_REPO = "handshake-ai-research/VIALS"
HF_REVISION = "9657bf84c6f6e470e1c31123afe3f9152c3655bd"
HF_CONFIG = "tasks"
HF_SPLIT = "test"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _task_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Invalid VIALS task ID: {value!r}")
    return f"vials-{value}"


def _numeric_bound(value: float | str | None) -> float | None:
    if value is None:
        return None
    bound = float(value)
    if math.isnan(bound):
        return None
    if not math.isfinite(bound):
        raise ValueError(f"Non-finite numeric answer bound: {value!r}")
    return bound


def _validate_row(row: dict) -> None:
    task_id = _task_id(row["id"])
    for key in ("question", "answer"):
        if not isinstance(row.get(key), str) or not html.unescape(row[key]).strip():
            raise ValueError(f"{task_id} requires a nonempty {key}")
    images = row["images"]
    if not images or len(images) != len(row["metadata"]["image_filenames"]):
        raise ValueError(f"{task_id}: image and filename counts must match and be nonzero")
    if any(not isinstance(image.get("bytes"), bytes) or not image["bytes"] for image in images):
        raise ValueError(f"{task_id}: images must contain embedded bytes")
    lower = _numeric_bound(row["metadata"].get("answer_lower_bound"))
    upper = _numeric_bound(row["metadata"].get("answer_upper_bound"))
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"{task_id}: answer_lower_bound must not exceed answer_upper_bound")


def _write_task(row: dict, destination: Path, *, docker_image: str) -> None:
    shutil.copytree(TEMPLATE_DIR, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    input_dir = destination / "environment" / "input"
    verifier_images = destination / "tests" / "images"
    input_dir.mkdir(parents=True)
    verifier_images.mkdir()
    names = []
    for index, (image, filename) in enumerate(zip(row["images"], row["metadata"]["image_filenames"], strict=True), 1):
        suffix = Path(filename or "").suffix.lower()
        if suffix not in _IMAGE_EXTENSIONS:
            suffix = ".png"
        stem = "image" if len(row["images"]) == 1 else f"image_{index}"
        name = stem + suffix
        (input_dir / name).write_bytes(image["bytes"])
        # Trusted copies are independent of any agent edits to /task images.
        (verifier_images / name).write_bytes(image["bytes"])
        names.append(name)
    prompts = load_prompt_config(BENCHMARK_DIR / "prompt.yaml")
    public_input = {
        "system": prompts.system,
        "user": prompts.user.format(question=html.unescape(row["question"])),
        "images": names,
    }
    (input_dir / "vials_input.json").write_text(json.dumps(public_input) + "\n")
    (destination / "instruction.md").write_text(
        Template((TEMPLATE_DIR / "instruction.md").read_text()).substitute(
            image_paths=", ".join(f"/task/{name}" for name in names), question=html.unescape(row["question"])
        )
    )
    (destination / "task.toml").write_text(
        Template((TEMPLATE_DIR / "task.toml").read_text()).substitute(
            task_name=json.dumps(f"vials/{_task_id(row['id'])}")
        )
    )
    (destination / "environment" / "Dockerfile").write_text(
        f"FROM {docker_image}\nCOPY --chown=agent:agent input/ /task/\nWORKDIR /task\nUSER agent\n"
    )
    payload = {
        "question": html.unescape(row["question"]),
        "expected_answer": html.unescape(row["answer"]),
        "answer_lower_bound": _numeric_bound(row["metadata"].get("answer_lower_bound")),
        "answer_upper_bound": _numeric_bound(row["metadata"].get("answer_upper_bound")),
        "images": [f"images/{name}" for name in names],
        "domain": row["domain"],
        "secondary_tags": list(row["metadata"].get("secondary_tags") or []),
    }
    (destination / "tests" / "verifier_input.json").write_text(json.dumps(payload, allow_nan=False) + "\n")
    shutil.copy2(BENCHMARK_DIR / "UPSTREAM_LICENSE", destination / "tests" / "UPSTREAM_LICENSE")
    (destination / "tests" / "test.sh").chmod(0o755)


def prepare(
    *,
    docker_image: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Write Harbor tasks directly from Hugging Face and a bridge compatibility index.

    No intermediate evaluation dataset is written. ``gym.jsonl`` contains only
    task names and empty inputs required by Gym's current shared Harbor bridge.
    """
    from datasets import Image, load_dataset

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", docker_image):
        raise ValueError("docker_image must be a container image reference")
    output_dir = Path(output_dir).expanduser().resolve()
    dataset = load_dataset(HF_REPO, HF_CONFIG, split=HF_SPLIT, revision=HF_REVISION)
    ids = [_task_id(value) for value in dataset["id"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("VIALS requires a nonempty dataset with unique task IDs")
    dataset = dataset.cast_column("images", [Image(decode=False)])
    rows = list(dataset.select(range(min(len(dataset), limit or len(dataset)))))
    selected_ids = {_task_id(row["id"]) for row in rows}
    if any(path.name not in selected_ids for path in output_dir.glob("vials-*")):
        raise ValueError("Output contains tasks outside this selection; choose a new output_dir")
    # Validate every selected row before replacing any existing task.
    for row in rows:
        _validate_row(row)
        destination = output_dir / _task_id(row["id"])
        if destination.is_symlink():
            raise ValueError(f"Harbor task output must not be a symlink: {destination}")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Harbor task output already exists: {destination}; use overwrite=True")
    output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for row in rows:
        destination = output_dir / _task_id(row["id"])
        if destination.exists():
            shutil.rmtree(destination)
        _write_task(row, destination, docker_image=docker_image)
        index.append({"task_name": destination.name, "responses_create_params": {"input": []}})
    path = output_dir / "gym.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in index))
    chosen = []
    for predicate in (
        lambda row: len(row["images"]) == 1,
        lambda row: len(row["images"]) > 1,
        lambda row: any(
            _numeric_bound(row["metadata"].get(key)) is not None
            for key in ("answer_lower_bound", "answer_upper_bound")
        ),
    ):
        for row in rows:
            name = _task_id(row["id"])
            if name not in chosen and predicate(row):
                chosen.append(name)
                break
    chosen += [entry["task_name"] for entry in index if entry["task_name"] not in chosen][: max(0, 5 - len(chosen))]
    (output_dir / "example.jsonl").write_text(
        "".join(json.dumps({"task_name": name, "responses_create_params": {"input": []}}) + "\n" for name in chosen)
    )
    shutil.copy2(BENCHMARK_DIR / "attributions.csv", output_dir / "attributions.csv")
    print(f"Prepared {len(index)} VIALS Harbor tasks: {output_dir}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-image", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    prepare(**vars(parser.parse_args()))
