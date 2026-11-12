# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic Molmo2-VideoTrackEval-layout clips: bright rectangles drifting over a dark background.

Used by the prepare tests and to generate the committed example rows, which
therefore contain no imagery from the source video datasets:

    python -m benchmarks.molmo2_videotrack.tests.synthetic
"""

import json
import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any


WIDTH, HEIGHT = 96, 64
EXAMPLE_FPATH = Path(__file__).parents[3] / "resources_servers" / "molmo2_videotrack" / "data" / "example.jsonl"
COLORS = [(230, 90, 40), (60, 200, 120), (70, 120, 240)]

Box = tuple[int, int, int, int]


def png_bytes(width: int, height: int, boxes: list[tuple[Box, tuple[int, int, int]]]) -> bytes:
    """An RGB PNG with each (x, y, w, h) box filled in its colour."""

    def pixel(x: int, y: int) -> bytes:
        for (x0, y0, w, h), color in boxes:
            if x0 <= x < x0 + w and y0 <= y < y0 + h:
                return bytes(color)
        return bytes((24, 28, 36))

    raw = b"".join(b"\x00" + b"".join(pixel(x, y) for x in range(width)) for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def rle_string(runs: list[int]) -> str:
    """Compress COCO RLE run lengths into the string form used by the dataset."""
    chars = []
    for index, run in enumerate(runs):
        value = run - runs[index - 2] if index > 2 else run
        more = True
        while more:
            char = value & 0x1F
            value >>= 5
            more = value != -1 if char & 0x10 else value != 0
            chars.append(chr((char | 0x20 if more else char) + 48))
    return "".join(chars)


def box_mask(box: Box, width: int = WIDTH, height: int = HEIGHT) -> dict[str, Any]:
    """COCO RLE mask ({"size": [height, width], "counts": str}) of a filled box."""
    x0, y0, w, h = box
    runs = [0]
    for index in range(width * height):
        column, row = divmod(index, height)
        inside = x0 <= column < x0 + w and y0 <= row < y0 + h
        if inside == (len(runs) % 2 == 0):
            runs[-1] += 1
        else:
            runs.append(1)
    return {"size": [height, width], "counts": rle_string(runs)}


def write_clip(
    videos_dir: Path,
    category: str,
    clip: str,
    expression: str,
    objects: list[dict[str, Any]],
    n_frames: int = 19,
    fps: float = 6.0,
) -> dict[str, Any]:
    """Write a clip's frames and return its annotation row.

    Each object is {"start": (x, y), "velocity": (dx, dy), "size": (w, h), "hidden": (frame indices)}.
    Frames are PNG data in files named like the real dataset's.
    """
    clip_dir = videos_dir / category / clip
    clip_dir.mkdir(parents=True)
    points: list[dict[str, Any]] = []
    masks: list[dict[str, Any]] = []
    for object_id, _ in enumerate(objects):
        points.append({"object_id": str(object_id), "points": []})
        masks.append({"object_id": str(object_id), "masks": []})
    for index in range(n_frames):
        drawn = []
        for position, obj in enumerate(objects):
            (x, y), (dx, dy), (w, h) = obj["start"], obj["velocity"], obj["size"]
            box = (int(x + dx * index), int(y + dy * index), w, h)
            if index in obj.get("hidden", ()):
                points[position]["points"].append(None)
                masks[position]["masks"].append(None)
                continue
            drawn.append((box, COLORS[position % len(COLORS)]))
            points[position]["points"].append([box[0] + w / 2, box[1] + h / 2])
            masks[position]["masks"].append(box_mask(box))
        (clip_dir / f"{index:05d}.jpg").write_bytes(png_bytes(WIDTH, HEIGHT, drawn))
    return {
        "id": f"{clip}_0",
        "video": clip,
        "clip": clip,
        "exp": expression,
        "points": points,
        "masks": masks,
        "w": WIDTH,
        "h": HEIGHT,
        "n_frames": n_frames,
        "fps": fps,
    }


LEFT = {"start": (6, 8), "velocity": (1.5, 0.5), "size": (16, 12)}
RIGHT = {"start": (70, 40), "velocity": (-1.5, -0.6), "size": (14, 14)}
EXAMPLE_CLIPS = [
    {"clip": "synthetic_000001", "expression": "the orange rectangle", "objects": [LEFT]},
    {"clip": "synthetic_000002", "expression": "every rectangle", "objects": [LEFT, RIGHT]},
    {
        "clip": "synthetic_000003",
        "expression": "the rectangle that leaves the scene",
        "objects": [{**LEFT, "hidden": tuple(range(10, 25))}],
        "n_frames": 25,
    },
    {
        "clip": "synthetic_000004",
        "expression": "the rectangles moving left",
        "objects": [RIGHT, {**RIGHT, "start": (76, 6), "velocity": (-2.0, 0.4), "hidden": (6,)}],
        "n_frames": 13,
    },
    {
        "clip": "synthetic_000005",
        "expression": "the slow rectangle",
        "objects": [{**LEFT, "velocity": (0.5, 0.2)}],
        "n_frames": 61,
        "fps": 30.0,
    },
]


def main() -> None:
    from benchmarks.molmo2_videotrack import prepare

    os.environ["MOLMO2_VIDEOTRACK_EMBED_FRAMES"] = "1"
    with tempfile.TemporaryDirectory() as tmp:
        videos_dir = Path(tmp) / "videos"
        rows = [prepare.convert(write_clip(videos_dir, "misc", **spec), "misc", videos_dir) for spec in EXAMPLE_CLIPS]
    EXAMPLE_FPATH.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLE_FPATH.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} example rows to {EXAMPLE_FPATH}")


if __name__ == "__main__":
    main()
