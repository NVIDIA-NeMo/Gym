# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned non-generative Agent-X subset for independent reference grading."""

import hashlib
import json
from pathlib import Path

from benchmarks.opencode_multimodal_benchmarks.agentx_media import image_url, read_image
from benchmarks.opencode_multimodal_benchmarks.agentx_source import (
    convert_source,
    policy_prompt,
    reference_identity,
    validate_video_frames,
)
from benchmarks.opencode_multimodal_benchmarks.canonical import fingerprint, manifest
from benchmarks.opencode_multimodal_benchmarks.data import atomic_json, sha256_file


def validate_rows(rows: list[dict], *, media_root: Path | None = None) -> None:
    """Validate the entire canonical split, labels and media before selecting a smoke subset."""
    canonical = manifest("AgentX")
    try:
        ids = [row["verifier_metadata"]["index"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(canonical["tasks"]):
            raise ValueError("Agent-X IDs differ from the pinned 605-question non-generative subset")
        for row in rows:
            task = row["verifier_metadata"]
            identifier = str(task["index"])
            if "task_id" in row and row["task_id"] != identifier:
                raise ValueError("Agent-X task_id differs from the canonical source index")
            row["task_id"] = identifier
            entry = canonical["tasks"][task["index"]]
            if task["benchmark"] != "AgentX" or task["split"] != "non_generative":
                raise ValueError("Incorrect Agent-X benchmark/split")
            if fingerprint(reference_identity(task)) != entry["reference_sha256"]:
                raise ValueError(
                    f"Agent-X public reference/schema mismatch at ID {task['index']}; rebuild from source"
                )
            declared_media = task["source_media_sha256"]
            declared_media = {Path(name).name: digest for name, digest in declared_media.items()}
            if declared_media != dict(entry["media"]):
                raise ValueError(f"Agent-X source media differs at ID {task['index']}")
            params = row["responses_create_params"]
            if params.get("tools"):
                raise ValueError("Agent-X canonical policy input cannot include task-supplied tools")
            if any(
                params.get(key) is not None
                for key in ("instructions", "prompt", "previous_response_id", "conversation", "text")
            ):
                raise ValueError("Agent-X canonical policy input cannot include additional prompt context")
            messages = params["input"]
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise ValueError("Agent-X canonical policy input requires one user message")
            content = messages[0]["content"]
            attachments = [p for p in content if p["type"] == "input_image"]
            if not attachments or any(p["type"] not in ("input_text", "input_image") for p in content):
                raise ValueError("Agent-X requires image/frame attachments")
            hashes = []
            for attachment in attachments:
                data, mime = read_image(attachment["image_url"], media_root=media_root)
                hashes.append(hashlib.sha256(data).hexdigest())
                attachment["image_url"] = image_url(data, mime)
            if task["modality"] == "Image":
                if hashes != [digest for _, digest in entry["media"]]:
                    raise ValueError(f"Agent-X attached image differs from public source: {task['index']}")
            else:
                validate_video_frames(task, hashes, entry)
            if (
                content[0]["type"] != "input_text"
                or content[0]["text"] != policy_prompt(task)
                or any(part["type"] != "input_image" for part in content[1:])
            ):
                raise ValueError(f"Agent-X canonical policy prompt mismatch at ID {task['index']}")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid Agent-X task schema: {error}") from error


def prepare(
    dataset: str | None = None,
    sha256: str | None = None,
    limit: int | str | None = None,
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
            "scorer": "agentx-reference-judge-v4",
            "official_goalacc": False,
            "canonical_samples": canonical["samples"],
            "prepared_samples": len(rows),
            "tasks_sha256": sha256_file(output),
            "additional_generative_ids": canonical["additional_generative_ids"],
            "video_sampling": "uniform ~1 fps; at most 128 JPEG frames",
        },
    )
    return output
