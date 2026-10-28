# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned Agent-X GoalAcc subset from public source or portable Gym rows."""

import hashlib
import json
from pathlib import Path

from PIL import Image

from benchmarks.opencode_multimodal_benchmarks.agentx_source import convert_source, reference_identity
from benchmarks.opencode_multimodal_benchmarks.canonical import fingerprint, manifest
from benchmarks.opencode_multimodal_benchmarks.data import atomic_json, sha256_file


def validate_rows(rows: list[dict], *, media_root: Path | None = None) -> None:
    """Validate the entire canonical split, labels and media before selecting a smoke subset."""
    canonical = manifest("AgentX")
    try:
        ids = [row["verifier_metadata"]["index"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(canonical["tasks"]):
            raise ValueError("Agent-X IDs differ from the pinned 605-question GoalAcc subset")
        for row in rows:
            task = row["verifier_metadata"]
            entry = canonical["tasks"][task["index"]]
            if task["benchmark"] != "AgentX_GoalAcc" or task["split"] != "goal_acc_non_generative":
                raise ValueError("Incorrect Agent-X benchmark/split")
            if fingerprint(reference_identity(task)) != entry["reference_sha256"]:
                raise ValueError(
                    f"Agent-X public reference/schema mismatch at ID {task['index']}; rebuild from source"
                )
            declared_media = task["source_media_sha256"]
            declared_media = {Path(name).name: digest for name, digest in declared_media.items()}
            if declared_media != dict(entry["media"]):
                raise ValueError(f"Agent-X source media differs at ID {task['index']}")
            messages = row["responses_create_params"]["input"]
            if not messages or any(message["role"] != "user" for message in messages):
                raise ValueError("Agent-X task input must contain only user messages")
            content = [part for message in messages for part in message["content"]]
            if task["question"] not in "\n".join(p["text"] for p in content if p["type"] == "input_text"):
                raise ValueError(f"Agent-X question missing from policy input: {task['index']}")
            attachments = [p for p in content if p["type"] == "input_image"]
            if not attachments or any(p["type"] not in ("input_text", "input_image") for p in content):
                raise ValueError("Agent-X requires image/frame attachments")
            hashes = []
            for attachment in attachments:
                relative = Path(attachment["image_url"])
                path = relative
                if media_root is not None:
                    path = (media_root / relative).resolve(strict=True)
                    if relative.is_absolute() or not path.is_relative_to(media_root):
                        raise ValueError("Agent-X bundle image must be relative to media_root")
                digest = sha256_file(path)
                if digest != path.stem:
                    raise ValueError(f"Agent-X image checksum mismatch: {relative}")
                with Image.open(path) as image:
                    image.verify()
                hashes.append(digest)
                attachment["image_url"] = str(path.resolve())
            if task["modality"] == "Image":
                if hashes != [digest for _, digest in entry["media"]]:
                    raise ValueError(f"Agent-X attached image differs from public source: {task['index']}")
            else:
                sampling = task["video_sampling"]
                indices = sampling["indices"]
                if not 1 <= len(hashes) <= 128 or len(indices) != len(hashes):
                    raise ValueError("Agent-X video frame count differs from sampling metadata")
                if sampling["fps"] <= 0 or any(
                    not isinstance(i, int) or not 0 <= i < sampling["frame_count"] for i in indices
                ):
                    raise ValueError("Invalid Agent-X video sampling metadata")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid Agent-X task schema: {error}") from error


def prepare(
    dataset: str | None = None,
    sha256: str | None = None,
    limit=None,
    media_root: str | None = None,
    source_root: str | None = None,
    prepared_dir: str | None = None,
) -> Path:
    """Use source_root for the official release, or dataset/sha256 for a validated portable bundle."""
    if isinstance(limit, str):
        limit = None if limit.lower() in ("none", "null") else int(limit)
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer or null")
    output = Path("benchmarks/opencode_multimodal_benchmarks/data/agentx.jsonl")
    if source_root is not None:
        if dataset is not None or sha256 is not None or media_root is not None:
            raise ValueError("Choose either public source_root or a prepared dataset bundle")
        materialized = Path(prepared_dir) if prepared_dir else output.with_suffix("")
        rows = convert_source(Path(source_root).resolve(strict=True), materialized)
    else:
        if dataset is None or sha256 is None:
            raise ValueError("Supply source_root, or dataset with its sha256")
        data = Path(dataset).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError("Agent-X prepared dataset checksum mismatch")
        rows = [json.loads(line) for line in data.splitlines()]
    validate_rows(rows, media_root=Path(media_root).resolve(strict=True) if media_root else None)
    rows = rows[:limit]
    for row in rows:
        row["verifier_metadata"]["expected_samples"] = len(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(output)
    canonical = manifest("AgentX")
    atomic_json(
        output.with_suffix(".provenance.json"),
        {
            "source_repo": canonical["source_repo"],
            "source_revision": canonical["source_revision"],
            "canonical_samples": canonical["samples"],
            "prepared_samples": len(rows),
            "tasks_sha256": sha256_file(output),
            "additional_generative_ids": canonical["additional_generative_ids"],
            "video_sampling": "uniform ~1 fps; at most 128 JPEG frames",
        },
    )
    return output
