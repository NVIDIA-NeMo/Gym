# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export prepared VIALS questions as tool-assisted Harbor tasks."""

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from string import Template
from urllib.parse import urlsplit

from benchmarks.vials.prepare import BENCHMARK_DIR, GYM_ROOT, OUTPUT_PATH
from benchmarks.vials.prepare import prepare as prepare_direct


TEMPLATE_DIR = Path(__file__).parent / "task-template"
DEFAULT_OUTPUT_DIR = OUTPUT_PATH.parent / "harbor"


def _task_id(row: dict) -> str:
    task_id = row["task_id"]
    if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", task_id):
        raise ValueError(f"Invalid VIALS task ID: {task_id!r}")
    return f"vials-{task_id}"


def _image_path(value: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else GYM_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"VIALS image does not exist: {path}")
    return path


def _write_task(row: dict, destination: Path, *, docker_image: str, verifier_url: str, verifier_host: str) -> None:
    images = [_image_path(value) for value in row["image_paths"]]
    shutil.copytree(TEMPLATE_DIR, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    input_dir = destination / "environment" / "input"
    input_dir.mkdir(parents=True)
    names = []
    for index, image in enumerate(images, start=1):
        stem = "image" if len(images) == 1 else f"image_{index}"
        name = stem + image.suffix.lower()
        shutil.copy2(image, input_dir / name)
        names.append(f"/task/{name}")
    instruction = Template((TEMPLATE_DIR / "instruction.md").read_text()).substitute(
        image_paths=", ".join(names), question=row["question"]
    )
    (destination / "instruction.md").write_text(instruction)
    config = Template((TEMPLATE_DIR / "task.toml").read_text()).substitute(
        task_name=json.dumps(f"vials/{_task_id(row)}"),
        verifier_url=json.dumps(verifier_url),
        verifier_host=json.dumps(verifier_host),
    )
    (destination / "task.toml").write_text(config)
    # Harbor builds only this small input layer on top of the shared runtime.
    (destination / "environment" / "Dockerfile").write_text(
        f"FROM {docker_image}\nCOPY --chown=agent:agent input/ /task/\nWORKDIR /task\nUSER agent\n"
    )
    # Verifier assets are staged by Harbor after the agent finishes, never COPY'd into the image.
    payload = {
        key: row[key]
        for key in (
            "question",
            "expected_answer",
            "responses_create_params",
            "answer_lower_bound",
            "answer_upper_bound",
        )
        if key in row
    }
    (destination / "tests" / "verifier_input.json").write_text(json.dumps(payload, allow_nan=False) + "\n")
    shutil.copy2(BENCHMARK_DIR / "UPSTREAM_LICENSE", destination / "tests" / "UPSTREAM_LICENSE")
    (destination / "tests" / "test.sh").chmod(0o755)


def prepare(
    *,
    docker_image: str,
    verifier_url: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    source: str | Path = OUTPUT_PATH,
    limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Download if necessary, export Harbor tasks, and return their Gym JSONL index.

    ``docker_image`` is the shared scientific-Python base image, built from the
    adjacent Dockerfile. ``verifier_url`` is the reachable Gym VIALS resources
    server's /verify endpoint; it is not a provider endpoint or an API key.
    A limit restricts export, not the source download. Existing tasks are never
    replaced unless overwrite=True. The ordinary direct dataset is preserved.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", docker_image):
        raise ValueError("docker_image must be a container image reference")
    endpoint = urlsplit(verifier_url)
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.hostname
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.query
        or endpoint.fragment
        or endpoint.path != "/verify"
    ):
        raise ValueError("verifier_url must be an HTTP(S) /verify endpoint without credentials or query parameters")
    source = Path(source).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not source.exists():
        prepare_direct(output=source)
    rows = []
    seen = set()
    with source.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = _task_id(row)
            if task_id in seen:
                raise ValueError(f"Duplicate VIALS task ID: {task_id}")
            seen.add(task_id)
            for key in ("question", "expected_answer"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{task_id} requires a nonempty {key}")
            if not row.get("image_paths") or not isinstance(row.get("responses_create_params"), dict):
                raise ValueError(f"{task_id} requires image_paths and prepared model inputs")
            for key in ("answer_lower_bound", "answer_upper_bound"):
                if row.get(key) is not None and not math.isfinite(float(row[key])):
                    raise ValueError(f"{task_id} has a non-finite {key}")
            if limit is None or len(rows) < limit:
                rows.append(row)
    if not rows:
        raise ValueError("VIALS source contains no tasks")
    selected_ids = {_task_id(row) for row in rows}
    stale = [path.name for path in output_dir.glob("vials-*") if path.name not in selected_ids]
    if stale:
        raise ValueError("Output contains tasks outside this selection; choose a new output_dir")
    # Preflight the entire selected set before replacing any existing output.
    for row in rows:
        destination = output_dir / _task_id(row)
        if destination.is_symlink():
            raise ValueError(f"Harbor task output must not be a symlink: {destination}")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Harbor task output already exists: {destination}; use overwrite=True")
        for image in row["image_paths"]:
            _image_path(image)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for row in rows:
        destination = output_dir / _task_id(row)
        if destination.exists():
            shutil.rmtree(destination)
        _write_task(
            row, destination, docker_image=docker_image, verifier_url=verifier_url, verifier_host=endpoint.hostname
        )
        index.append({"task_name": destination.name, "responses_create_params": {"input": []}})
    path = output_dir / "gym.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in index))
    shutil.copy2(BENCHMARK_DIR / "attributions.csv", output_dir / "attributions.csv")
    print(f"Exported {len(index)} tool-assisted VIALS tasks to {output_dir}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-image", required=True)
    parser.add_argument("--verifier-url", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    prepare(**vars(parser.parse_args()))
