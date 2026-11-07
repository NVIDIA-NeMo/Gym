# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the pinned public Agent-X release to final-answer Gym tasks.

GoalAcc subset/exclusion selection follows the original metrics notebook:
https://github.com/mbzuai-oryx/Agent-X/blob/1dcd5a66267f0582898c692a5eef9d4a52d0dc2f/analysis/metrics.ipynb
"""

import json
import math
from pathlib import Path

from benchmarks.opencode_multimodal_benchmarks.canonical import fingerprint, manifest
from benchmarks.opencode_multimodal_benchmarks.data import ANSWER_INSTRUCTION, _image, sha256_file
from benchmarks.opencode_multimodal_benchmarks.prepare import task_row


def reference_metadata(identifier: str, record: dict, reference: dict) -> dict:
    user = next(item for item in record["dialogs"] if item["role"] == "user")
    content = user["content"]
    question = (
        content
        if isinstance(content, str)
        else "\n".join(item["text"] for item in content if item.get("type") == "text" and item.get("text"))
    )
    question = question.strip() or reference["query"].strip()
    raw_type = reference["query_type"].strip()
    query_type = raw_type.lower().replace("objetive", "objective")
    answer = reference.get("final_answer")
    if answer is None:
        # The pinned release nests task 522's answer inside tool_metadata.
        answer = reference.get("tool_metadata", {}).get("final_answer")
    if answer is None:
        raise ValueError(f"Missing Agent-X reference at ID {identifier}")
    if not isinstance(answer, dict):
        answer = {"value": str(answer), "justification": ""}
    gt = {key: str(answer.get(key, "")).strip() for key in ("value", "justification")}
    modality = "Video" if any(item["type"] == "video" for item in record["files"]) else "Image"
    if not question or query_type not in ("objective", "subjective") or not gt["value"]:
        raise ValueError(f"Invalid Agent-X reference at ID {identifier}")
    return {
        "index": identifier,
        "source_index": identifier,
        "benchmark": "AgentX_GoalAcc",
        "question": question,
        "answer": gt["value"],
        "GT_json": json.dumps(gt, ensure_ascii=False),
        "query_type": query_type,
        "query_type_raw": raw_type,
        "category": query_type,
        "l2_category": "",
        "ans_type": query_type,
        "options": {},
        "split": "goal_acc_non_generative",
        "modality": modality,
    }


def reference_identity(task: dict) -> dict:
    return {key: task[key] for key in ("index", "question", "answer", "query_type", "query_type_raw", "modality")} | {
        "ground_truth": json.loads(task["GT_json"])
    }


def sample_video(path: Path, directory: Path) -> tuple[list[Path], str, dict]:
    """Uniform ~1 fps, up to 128 JPEG frames; this is an explicit harness policy."""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f"Invalid video: {path}")
        wanted = min(128, max(1, int(count / fps)))
        indices = np.linspace(0, count - 1, wanted).round().astype(int)
        directory.mkdir(parents=True, exist_ok=True)
        frames = []
        for position, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                # Some containers cannot seek to the last frame; decode that same frame sequentially.
                sequential = cv2.VideoCapture(str(path))
                try:
                    for _ in range(int(index) + 1):
                        ok, frame = sequential.read()
                        if not ok:
                            break
                finally:
                    sequential.release()
            target = directory / f"frame-{position + 1}-of-{wanted}.jpg"
            if not ok or not cv2.imwrite(str(target), frame):
                raise ValueError(f"Cannot decode video frame {index}: {path}")
            frames.append(target)
        prefix = "This is a video:\n" + "".join(
            f"Frame {i + 1} sampled at {index / fps:.2f} seconds: <image>\n" for i, index in enumerate(indices)
        )
        return frames, prefix, {"fps": fps, "frame_count": count, "indices": [int(i) for i in indices]}
    finally:
        capture.release()


def convert_source(root: Path, output: Path) -> list[dict]:
    """Read official annotations and verify every source asset before converting tasks."""
    canonical = manifest("AgentX")
    for name, digest in canonical["source_sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Agent-X public annotation checksum mismatch: {name}")
    dataset = json.loads((root / "dataset.json").read_text())
    references = json.loads((root / "data.json").read_text())
    media_root = root / "files" if (root / "files").is_dir() else root / "image"
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    # Validate the whole source before writing any tasks, including when a later limit is requested.
    for entry in canonical["tasks"].values():
        for name, digest in entry["media"]:
            path = media_root / name
            if sha256_file(path) != digest:
                raise ValueError(f"Agent-X public media checksum mismatch: {name}")
    rows = []
    identifiers = sorted(canonical["tasks"], key=lambda i: (canonical["tasks"][i]["modality"] == "Video", int(i)))
    for identifier in identifiers:
        entry = canonical["tasks"][identifier]
        task = reference_metadata(identifier, dataset[identifier], references[identifier][0])
        if fingerprint(reference_identity(task)) != entry["reference_sha256"]:
            raise ValueError(f"Agent-X public reference mismatch: {identifier}")
        media = [media_root / name for name, _ in entry["media"]]
        task["source_media_sha256"] = dict(entry["media"])
        prefix = ""
        if task["modality"] == "Video":
            media, prefix, task["video_sampling"] = sample_video(media[0], output / "frames" / identifier)
        task["images"] = [str(_image(str(path.resolve()), root, images).resolve()) for path in media]
        task["prompt"] = (
            prefix + ("\n" if prefix else "") + "Question: " + task["question"] + "\n\n" + ANSWER_INSTRUCTION
        )
        task["expected_samples"] = len(canonical["tasks"])
        rows.append(task_row(task))
    return rows
