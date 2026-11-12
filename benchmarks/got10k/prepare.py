# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert an extracted GOT-10k split into Gym benchmark JSONL.

GOT-10k is distributed under registration at http://got-10k.aitestunion.com and
cannot be downloaded automatically. Point GOT10K_DATASET_ROOT at the extracted
dataset (the directory that contains `val/`, or the `val/` directory itself).

Each sequence becomes one task: the first frame with the target's box, followed
by frames sampled at 1 frame per second, all embedded as data URLs at their
original resolution. Ground-truth boxes and visibility labels stay in
`verifier_metadata` and never enter the prompt.
"""

import base64
import json
import math
import os
import re
import struct
from pathlib import Path
from typing import Any


SPLIT = "val"
SAMPLE_FPS = 1.0
GRID = 1000

BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "got10k_benchmark.jsonl"

IMAGE_MIME_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
# JPEG start-of-frame markers, which carry the image dimensions.
JPEG_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}

REFERENCE_TEXT = "Image 0 is the reference image. The target object is at xyxy bbox {box} on a 0-1000 grid."
INSTRUCTION_TEXT = (
    "Locate the same object in every requested subsequent image. "
    "Return boxes for frames {frames}. "
    "Return only valid JSON inside <answer>...</answer> in this exact shape: "
    '<answer>{{"boxes":[{entries}]}}</answer>. '
    "Coordinates must be integer values on a 0-1000 grid using xyxy order "
    "(each axis spans 0-1000, independent of pixel size)."
)


def split_root() -> Path:
    root = os.environ.get("GOT10K_DATASET_ROOT")
    if not root:
        raise RuntimeError(
            "GOT10K_DATASET_ROOT is not set. GOT-10k requires registration and is not downloaded automatically; "
            f"set it to the extracted dataset directory (the one containing '{SPLIT}/')."
        )
    path = Path(root)
    return path / SPLIT if (path / SPLIT).is_dir() else path


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) read from a PNG or JPEG header."""
    data = path.read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    if data[:2] == b"\xff\xd8":
        pos = 2
        while pos + 9 < len(data):
            if data[pos] != 0xFF:
                pos += 1
                continue
            marker = data[pos + 1]
            # Fill bytes, standalone markers and restart markers carry no length.
            if marker == 0xFF:
                pos += 1
                continue
            if marker in (0x00, 0x01) or 0xD0 <= marker <= 0xD9:
                pos += 2
                continue
            if marker in JPEG_SOF_MARKERS:
                height, width = struct.unpack(">HH", data[pos + 5 : pos + 9])
                return width, height
            pos += 2 + struct.unpack(">H", data[pos + 2 : pos + 4])[0]
    raise ValueError(f"Could not read image dimensions from {path}")


def image_data_url(path: Path) -> str:
    return f"data:{IMAGE_MIME_TYPES[path.suffix.lower()]};base64," + base64.b64encode(path.read_bytes()).decode()


def read_boxes(path: Path) -> list[list[float]]:
    return [[float(x) for x in line.split(",")] for line in path.read_text().splitlines() if line.strip()]


def read_anno_fps(path: Path) -> float | None:
    """Annotation rate from meta_info.ini, e.g. `anno_fps: 10Hz` -> 10.0."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "anno_fps":
            match = re.search(r"[-+]?(?:\d*\.\d+|\d+\.?)", value)
            return float(match.group(0)) if match else None
    return None


def select_target_indices(num_frames: int, anno_fps: float) -> list[int]:
    """0-based indices of the frames to localize: one per second of video after the first frame."""
    if SAMPLE_FPS >= anno_fps:
        return list(range(1, num_frames))
    step = 1.0 / SAMPLE_FPS
    duration = (num_frames - 1) / anno_fps
    count = max(0, math.ceil((duration + 1e-9 - step) / step))
    indices: list[int] = []
    for i in range(count):
        index = min(max(int(round((step + i * step) * anno_fps)), 1), num_frames - 1)
        if index not in indices:
            indices.append(index)
    return indices


def grid_box_text(box: list[float], width: int, height: int) -> str:
    """Pixel xywh -> '[x1,y1,x2,y2]' on the 0-1000 grid, clamped to the image."""
    x, y, w, h = box
    normalized = [x / width, y / height, (x + w) / width, (y + h) / height]
    values = [min(GRID, max(0, round(min(max(v, 0.0), 1.0) * GRID))) for v in normalized]
    return "[" + ",".join(str(v) for v in values) + "]"


def load_sequence(seq_dir: Path) -> dict[str, Any]:
    """Read one sequence, refusing anything incomplete or misaligned."""
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"GOT-10k sequence directory not found: {seq_dir}")
    frames = sorted(path for path in seq_dir.iterdir() if path.suffix.lower() in IMAGE_MIME_TYPES)
    if len(frames) < 2:
        raise ValueError(f"GOT-10k sequence {seq_dir.name} has fewer than two frames")
    boxes = read_boxes(seq_dir / "groundtruth.txt")
    if len(boxes) != len(frames) or any(len(box) != 4 for box in boxes):
        raise ValueError(
            f"GOT-10k sequence {seq_dir.name}: {len(frames)} frames but {len(boxes)} usable annotations; "
            "refusing a partial or misaligned evaluation."
        )
    cover = [float(line) for line in (seq_dir / "cover.label").read_text().split()]
    if len(cover) != len(frames) or not all(math.isfinite(value) for value in cover):
        raise ValueError(
            f"GOT-10k sequence {seq_dir.name}: cover.label must hold one finite visibility label per frame."
        )
    anno_fps = read_anno_fps(seq_dir / "meta_info.ini")
    if anno_fps is None or anno_fps <= 0:
        raise ValueError(f"GOT-10k sequence {seq_dir.name} has no valid anno_fps; cannot sample at 1 fps.")
    return {"name": seq_dir.name, "frames": frames, "boxes": boxes, "cover": cover, "anno_fps": anno_fps}


def convert(sequence: dict[str, Any]) -> dict[str, Any] | None:
    """One sequence -> one benchmark row, or None when it is too short to yield a target frame."""
    frames = sequence["frames"]
    target_indices = select_target_indices(len(frames), sequence["anno_fps"])
    if not target_indices:
        return None
    width, height = image_size(frames[0])
    frame_ids = list(range(1, len(target_indices) + 1))

    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": REFERENCE_TEXT.format(box=grid_box_text(sequence["boxes"][0], width, height))},
        {"type": "input_image", "detail": "auto", "image_url": image_data_url(frames[0])},
    ]
    for frame_id, index in zip(frame_ids, target_indices):
        content.append({"type": "input_text", "text": f"Image {frame_id}:"})
        content.append({"type": "input_image", "detail": "auto", "image_url": image_data_url(frames[index])})
    instruction = INSTRUCTION_TEXT.format(
        frames="[" + ",".join(str(frame_id) for frame_id in frame_ids) + "]",
        entries=",".join(f'{{"frame":{frame_id},"bbox":[x1,y1,x2,y2]}}' for frame_id in frame_ids),
    )
    content.append({"type": "input_text", "text": instruction})

    return {
        "responses_create_params": {"input": [{"role": "user", "content": content}]},
        "verifier_metadata": {
            "sequence": sequence["name"],
            "split": SPLIT,
            "anno_fps": sequence["anno_fps"],
            "image_width": width,
            "image_height": height,
            "targets": [
                {
                    "frame": frame_id,
                    "source_frame": index + 1,
                    "bbox": sequence["boxes"][index],
                    "visible": sequence["cover"][index],
                }
                for frame_id, index in zip(frame_ids, target_indices)
            ],
        },
    }


def sequence_names(root: Path) -> list[str]:
    list_file = root / "list.txt"
    if not list_file.is_file():
        raise FileNotFoundError(f"GOT-10k split list missing: {list_file}. Cannot verify a complete extraction.")
    return [line.strip() for line in list_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare() -> Path:
    root = split_root()
    names = sequence_names(root)
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as out:
        for name in names:
            row = convert(load_sequence(root / name))
            if row is None:
                print(f"Skipping {name}: too short to yield a target frame at {SAMPLE_FPS} fps")
                continue
            out.write(json.dumps(row) + "\n")
            written += 1
    print(f"Wrote {written} of {len(names)} sequences to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
