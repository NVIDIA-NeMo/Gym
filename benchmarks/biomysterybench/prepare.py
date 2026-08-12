# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare gated BioMysteryBench data for native AnyTerminal execution.

The benchmark is evaluation-only.  Preparation downloads the pinned gated
Hugging Face artifacts, safely extracts each selected archive, and writes a
Gym benchmark JSONL.  Hidden answer rubrics remain top-level verification data;
they are never placed in ``responses_create_params`` or mounted into the policy
agent sandbox.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


HF_REPO_ID = "Anthropic/BioMysteryBench-full"
DEFAULT_DOCKER_IMAGE = "biomysterybench-runtime:v12"


@dataclass(frozen=True)
class DatasetRelease:
    revision: str
    expected_task_count: int
    expected_split_counts: dict[str, int]
    output_filename: str


RELEASES = {
    # Exact pre-audit release used for Anthropic's published 99-task result.
    "official-99": DatasetRelease(
        revision="a066d4135d087934f1c5399f45ca7f2cd4bd0675",
        expected_task_count=99,
        expected_split_counts={"yes": 76, "no": 23},
        output_filename="biomysterybench_official_99.jsonl",
    ),
    # Current corrected release, after the July 2026 answer-key audit.
    "v11": DatasetRelease(
        revision="b5a889c4757214ec9a6ade876b734f920a7799db",
        expected_task_count=90,
        expected_split_counts={"yes": 73, "no": 17},
        output_filename="biomysterybench_v11.jsonl",
    ),
}
DEFAULT_RELEASE = "v11"
# Backward-compatible name used by downstream imports and release documentation.
HF_REVISION = RELEASES[DEFAULT_RELEASE].revision

BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_CACHE_DIR = DATA_DIR / "cache"
DEFAULT_OUTPUT = DATA_DIR / RELEASES[DEFAULT_RELEASE].output_filename
DOWNLOAD_ATTEMPTS = 8


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from nemo_gym.global_config import get_hf_token

            token = get_hf_token()
        except Exception:
            token = None
    if not token:
        raise RuntimeError(
            "BioMysteryBench is gated. Accept its Hugging Face access terms and set HF_TOKEN before preparation."
        )
    return token


def _download(filename: str, token: str, revision: str) -> Path:
    try:
        from huggingface_hub import close_session, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to prepare BioMysteryBench") from exc

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            return Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    filename=filename,
                    revision=revision,
                    token=token,
                )
            )
        except Exception as exc:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            if status_code in {400, 401, 403, 404} or attempt == DOWNLOAD_ATTEMPTS:
                raise

            # huggingface_hub 1.25 can leave its shared HTTPX client closed
            # after a failed internal retry. Reset it before our outer retry so
            # a long gated-data preparation can resume after transient TLS or
            # CDN failures instead of discarding all prior task progress.
            close_session()
            delay = min(60, 2 ** (attempt - 1))
            print(
                f"Transient download failure for {filename} "
                f"({type(exc).__name__}); retrying in {delay}s "
                f"[{attempt}/{DOWNLOAD_ATTEMPTS}]",
                flush=True,
            )
            time.sleep(delay)

    raise AssertionError("unreachable")


def _load_problem_rows(problems_csv: Path, release: DatasetRelease) -> list[dict[str, str]]:
    with problems_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"id", "question", "answer_rubric", "allowed_domains", "human_solvable"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"unexpected problems.csv schema: {list(rows[0]) if rows else 'empty'}")
    split_counts = {value: sum(row["human_solvable"] == value for row in rows) for value in ("yes", "no")}
    if len(rows) != release.expected_task_count or split_counts != release.expected_split_counts:
        raise ValueError(
            "pinned BioMysteryBench integrity check failed: "
            f"revision={release.revision}, rows={len(rows)}, splits={split_counts}"
        )
    return rows


def _select_rows(
    rows: list[dict[str, str]], task_ids: Iterable[str] | None = None, limit: int | None = None
) -> list[dict[str, str]]:
    if task_ids:
        requested = list(dict.fromkeys(task_ids))
        by_id = {row["id"]: row for row in rows}
        missing = [task_id for task_id in requested if task_id not in by_id]
        if missing:
            raise ValueError(f"unknown BioMysteryBench task id(s): {', '.join(missing)}")
        selected = [by_id[task_id] for task_id in requested]
    else:
        selected = list(rows)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        selected = selected[:limit]
    return selected


def _safe_member_path(info: zipfile.ZipInfo) -> PurePosixPath:
    member = PurePosixPath(info.filename)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"unsafe path in archive: {info.filename!r}")
    unix_mode = info.external_attr >> 16
    if stat.S_ISLNK(unix_mode):
        raise ValueError(f"symbolic links are not allowed in task archives: {info.filename!r}")
    return member


def _archive_sha256(archive: Path) -> str:
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> dict[str, int | str]:
    """Idempotently extract a task archive with traversal and disk-space checks."""

    archive_hash = _archive_sha256(archive)
    marker = destination / ".biomysterybench-extracted.json"
    if marker.is_file():
        try:
            cached = json.loads(marker.read_text())
            if cached.get("archive_sha256") == archive_hash:
                return cached
        except (OSError, ValueError):
            pass

    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for info in members:
            _safe_member_path(info)
        uncompressed_bytes = sum(info.file_size for info in members)

        destination.parent.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(destination.parent).free
        if free_bytes < uncompressed_bytes + 1024**3:
            raise OSError(
                f"insufficient free space to extract {archive.name}: need at least "
                f"{uncompressed_bytes + 1024**3} bytes, have {free_bytes}"
            )

        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
            temporary_path = Path(temporary)
            for info in members:
                member = _safe_member_path(info)
                target = temporary_path.joinpath(*member.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)

            metadata: dict[str, int | str] = {
                "archive_sha256": archive_hash,
                "compressed_bytes": archive.stat().st_size,
                "uncompressed_bytes": uncompressed_bytes,
                "file_count": sum(not info.is_dir() for info in members),
            }
            (temporary_path / marker.name).write_text(json.dumps(metadata, sort_keys=True) + "\n")
            if destination.exists():
                shutil.rmtree(destination)
            Path(temporary).replace(destination)
            return metadata


def _allowed_domains(raw: str) -> list[str]:
    return [domain.strip() for domain in raw.split(",") if domain.strip()]


def _resource_defaults(uncompressed_bytes: int) -> dict[str, str]:
    gib = max(1, (uncompressed_bytes + 1024**3 - 1) // 1024**3)
    memory_mb = min(131072, max(16384, int(gib * 3072)))
    return {
        "cpus": "4",
        "memory_mb": str(memory_mb),
        "storage_mb": str(max(32768, int(gib * 3072))),
        "gpus": "0",
    }


def _gym_row(
    row: dict[str, str],
    data_dir: Path,
    extraction: dict[str, int | str],
    image: str,
    revision: str,
) -> dict:
    allowed_domains = _allowed_domains(row["allowed_domains"])
    resources = _resource_defaults(int(extraction["uncompressed_bytes"]))
    return {
        "id": row["id"],
        "question": row["question"],
        "expected_answer": row["answer_rubric"],
        "human_solvable": row["human_solvable"],
        "allowed_domains": allowed_domains,
        "dataset_revision": revision,
        "responses_create_params": {
            "input": [{"role": "user", "content": row["question"]}],
            "metadata": {
                "instance_id": f"biomysterybench::{row['id']}",
                "task_name": row["id"],
                "docker_image": image,
                "data_dir": str(data_dir.resolve()),
                "workdir": "/workspace",
                "agent_timeout_sec": "14400",
                "verifier_timeout_sec": "600",
                # Responses-API metadata is string-valued. Keep the verifier's
                # top-level copy typed, and serialize the sandbox copy so it
                # survives Gym request validation without losing structure.
                "allowed_domains": json.dumps(allowed_domains, separators=(",", ":")),
                "dataset_revision": revision,
                "compressed_bytes": str(extraction["compressed_bytes"]),
                "uncompressed_bytes": str(extraction["uncompressed_bytes"]),
                **resources,
            },
        },
    }


def prepare(
    *,
    output: Path | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    task_ids: Iterable[str] | None = None,
    limit: int | None = None,
    image: str = DEFAULT_DOCKER_IMAGE,
    release_name: str = DEFAULT_RELEASE,
) -> Path:
    release = RELEASES[release_name]
    output = output or DATA_DIR / release.output_filename
    token = _hf_token()
    rows = _load_problem_rows(_download("problems.csv", token, release.revision), release)
    selected = _select_rows(rows, task_ids, limit)
    extracted_root = cache_dir / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(selected, start=1):
            task_id = row["id"]
            print(f"[{index}/{len(selected)}] preparing {task_id}", flush=True)
            archive = _download(f"data/{task_id}.zip", token, release.revision)
            data_dir = extracted_root / task_id
            extraction = safe_extract(archive, data_dir)
            stream.write(
                json.dumps(
                    _gym_row(row, data_dir, extraction, image, release.revision),
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary_output.replace(output)
    print(
        f"Wrote {len(selected)} BioMysteryBench {release_name} tasks at {release.revision} to {output}",
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--task", dest="task_ids", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--release", choices=sorted(RELEASES), default=DEFAULT_RELEASE)
    parser.add_argument("--image", default=os.environ.get("BIOMYSTERYBENCH_IMAGE", DEFAULT_DOCKER_IMAGE))
    args = parser.parse_args()
    prepare(
        output=args.output,
        cache_dir=args.cache_dir,
        task_ids=args.task_ids,
        limit=args.limit,
        image=args.image,
        release_name=args.release,
    )


if __name__ == "__main__":
    main()
