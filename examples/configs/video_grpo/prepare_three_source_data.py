# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interleave materialized Harbor or ClawEval rows with Video-MME annotations without decoding media.

Agentic-VBench rows come from stock Harbor materialization. This helper preserves their complete serialized
rows; it does not translate the former flat AVB representation or bind task folders and judges.
"""

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import re
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path


VIDEO_AGENT = "videomme_benchmark_simple_agent"
CLAW_AGENT = "claweval_agent"


def videomme_prompt_text(question: str, candidates: str) -> str:
    """Preserve the reference Video-MME question, option order, and answer instruction."""
    joined = question + "\n" + "\n".join(ast.literal_eval(candidates))
    joined += "\nRespond with only the letter (A, B, C, or D) of the correct option."
    return f"{joined}\nAnswer: "


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1024)
def cached_frames(video, frame_root):
    """Reuse one complete native 2FPS cache group in original ordinal order."""
    pattern = re.compile(r"^frame-(\d+)-of-(\d+)-2(?:\.0)?fps\.jpg$")
    groups = {}
    with os.scandir(frame_root / video) as entries:
        for index, entry in enumerate(entries):
            if index >= 4096:
                raise ValueError("Native frame cache directory exceeds4096 entries")
            match = pattern.fullmatch(entry.name)
            if match:
                ordinal, count = map(int, match.groups())
                if ordinal in groups.setdefault(count, {}):
                    raise ValueError("Ambiguous native2FPS cache filename aliases")
                groups[count][ordinal] = Path(entry.path)
    complete = [
        paths for count, paths in groups.items() if 1 <= count <= 256 and set(paths) == set(range(1, count + 1))
    ]
    if len(complete) != 1:
        raise ValueError("Expected exactly one complete native2FPS cache group of at most256 frames")
    paths = tuple(complete[0][index] for index in sorted(complete[0]))
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise ValueError("Native frame cache contains missing or empty frames")
    return paths


def videomme_row(annotation: Mapping[str, str], frame_root: Path) -> dict[str, object]:
    """Keep native question/options/verifier fields and expose cached frames as ordinary images."""
    video = annotation["video"]
    if not video or Path(video).name != video or "\\" in video:
        raise ValueError("Video-MME video identifier must be a single filename stem")
    if annotation["video_path"] != f"./video/{video}.mp4":
        raise ValueError("Video-MME annotation video_path disagrees with its video identifier")
    paths = cached_frames(video, frame_root)
    # Ordinary images use Nano820's real Nemotron4/Pixtral image path in policy
    # and GEN. No native-video conversion flags or duplicate metadata injection.
    return {
        "task_source": VIDEO_AGENT,
        "responses_create_params": {
            # Explicit initial RL cap, shared with the coding tasks. The supplied
            # evaluator's32000-token budget is not claimed as reproduced here.
            "max_output_tokens": 8192,
            "input": [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "input_image", "image_url": path.as_uri(), "detail": "high"} for path in paths],
                        {
                            "type": "input_text",
                            "text": videomme_prompt_text(annotation["question"], annotation["candidates"]),
                        },
                    ],
                }
            ],
        },
        "answer": annotation["answer"],
        "category": annotation["duration"],
        "benchmark_name": "Video_MME",
        "index": int(annotation["index"]),
        "question": annotation["question"],
        "duration": annotation["duration"],
        "domain": annotation["domain"] or None,
        "sub_category": annotation["sub_category"] or None,
        "task_type": annotation["task_type"] or None,
        "video": video,
        "candidates": annotation["candidates"],
    }


def interleave(agentic_lines, video_lines):
    """Stable proportional merge, preserving both source orders without replication."""
    if not agentic_lines or not video_lines:
        raise ValueError("Both source streams must be nonempty")
    emitted = 0
    for index, line in enumerate(agentic_lines):
        yield line
        target = (index + 1) * len(video_lines) // len(agentic_lines)
        while emitted < target:
            yield video_lines[emitted]
            emitted += 1
    assert emitted == len(video_lines)


def prepare(
    *,
    agentic_jsonl: Path,
    expected_agentic_sha256: str,
    videomme_tsv: Path,
    expected_videomme_sha256: str,
    frame_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Create a fresh data directory; input hashes bind the exact reviewed source files."""
    agentic_raw = agentic_jsonl.read_bytes()
    video_raw = videomme_tsv.read_bytes()
    if sha256(agentic_raw) != expected_agentic_sha256 or sha256(video_raw) != expected_videomme_sha256:
        raise ValueError("Input dataset hash differs from the supplied authority")
    agentic_lines = agentic_raw.splitlines(keepends=True)
    if any(not line.endswith(b"\n") or not line.strip() for line in agentic_lines):
        raise ValueError("Agentic source must contain newline-terminated nonempty rows")
    agentic_rows = [json.loads(line) for line in agentic_lines]
    for row in agentic_rows:
        if row.get("task_source") == CLAW_AGENT:
            continue
        task_id = row.get("task_id")
        task_input = row.get("task_input")
        if not (
            isinstance(task_id, dict)
            and isinstance(task_id.get("taskset"), str)
            and isinstance(task_id.get("task_id"), str)
            and isinstance(task_input, dict)
        ):
            raise ValueError("Use stock materialized Harbor rows, or ClawEval rows")
        params = task_input.get("responses_create_params")
        turns = task_input.get("turns")
        if not (
            (isinstance(params, dict) and not turns)
            or (
                params is None
                and isinstance(turns, list)
                and turns
                and all(
                    isinstance(turn, dict) and isinstance(turn.get("responses_create_params"), dict) for turn in turns
                )
            )
        ):
            raise ValueError("Stock Harbor rows require request parameters or turns, not both")
    counts = Counter(row.get("task_source") or row["task_id"]["taskset"] for row in agentic_rows)
    frame_root = frame_root.resolve(strict=True)
    if not frame_root.is_dir():
        raise NotADirectoryError(frame_root)
    annotations = list(csv.DictReader(io.StringIO(video_raw.decode("utf-8")), delimiter="\t"))
    if not annotations:
        raise ValueError("Video-MME annotations must be nonempty")
    if len({int(row["index"]) for row in annotations}) != len(annotations):
        raise ValueError("Video-MME annotation indexes must be unique")
    video_lines = [
        (json.dumps(videomme_row(row, frame_root), ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for row in annotations
    ]
    video_bytes = b"".join(video_lines)
    combined_bytes = b"".join(interleave(agentic_lines, video_lines))
    counts[VIDEO_AGENT] = len(video_lines)
    manifest = {
        "status": "PREPARED_AGENTIC_VIDEO_DATA_NO_MODEL_OR_MEDIA_DECODE",
        "sources": {
            "agentic_rows": {
                "path": str(agentic_jsonl.resolve()),
                "sha256": sha256(agentic_raw),
                "rows": len(agentic_lines),
            },
            "Video_MME": {"path": str(videomme_tsv.resolve()), "sha256": sha256(video_raw), "rows": len(video_lines)},
        },
        "frame_root": str(frame_root),
        "task_source_counts": dict(counts),
        "rows": sum(counts.values()),
        "outputs": {
            name: {"sha256": sha256(data), "bytes": len(data)}
            for name, data in [("videomme_train.jsonl", video_bytes), ("three_source_train.jsonl", combined_bytes)]
        },
        "ordering": (
            "After each input agentic row i, emit Video-MME rows through "
            f"floor((i+1)*VideoMME_rows/{len(agentic_lines)}); both source orders preserved."
        ),
        "original_agentic_lines_byte_exact": True,
        "no_source_repetition_or_tracking": True,
        "media_transport": "Ordered ordinary input_image file:// references; no metadata.video_path or _is_video_frame flags",
        "media_sampling": "Existing native2FPS cache groups, capped at256 and uniformly sampled over the source video by the pinned evaluator; no new decoding",
        "timestamp_prompt_limit": "Original question/options and answer instruction retained; no timestamps invented from cache ordinals",
        "Video_MME_max_output_tokens": 8192,
        "output_budget_scope": "Initial RL validations; deliberately differs from supplied VideoMME evaluator32000",
        "verifier": "Original N4 Video_MME reference exact matcher; input agentic rows unchanged",
    }
    output_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    for name, data in [
        ("videomme_train.jsonl", video_bytes),
        ("three_source_train.jsonl", combined_bytes),
        ("manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()),
    ]:
        with (output_root / name).open("xb") as stream:
            os.chmod(output_root / name, 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if (output_root / name).read_bytes() != data:
            raise OSError("Prepared dataset readback differs")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agentic-jsonl", type=Path, required=True)
    parser.add_argument("--expected-agentic-sha256", required=True)
    parser.add_argument("--videomme-tsv", type=Path, required=True)
    parser.add_argument("--expected-videomme-sha256", required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(**vars(args))
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["rows"],
                "task_source_counts": result["task_source_counts"],
                "outputs": result["outputs"],
            }
        )
    )


if __name__ == "__main__":
    main()
