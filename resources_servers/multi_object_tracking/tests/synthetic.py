# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic multi-object tracking rows: bright rectangles drifting over a dark background.

Generates the committed example rows, which therefore contain no dataset imagery:

    python -m resources_servers.multi_object_tracking.tests.synthetic
"""

import base64
import json
import struct
import zlib
from pathlib import Path


WIDTH, HEIGHT = 96, 64
GRID = 1000
EXAMPLE_FPATH = Path(__file__).parents[1] / "data" / "example.jsonl"
COLORS = [(230, 90, 40), (60, 200, 120), (70, 120, 240)]

Box = tuple[int, int, int, int]


def png_bytes(width: int, height: int, boxes: list[tuple[Box, tuple[int, int, int]]]) -> bytes:
    """An RGB PNG with each (x, y, w, h) box filled in its color; later boxes are drawn on top."""
    background = bytes((24, 28, 36))

    def pixel(x: int, y: int) -> bytes:
        for (x0, y0, w, h), color in reversed(boxes):
            if x0 <= x < x0 + w and y0 <= y < y0 + h:
                return bytes(color)
        return background

    raw = b"".join(b"\x00" + b"".join(pixel(x, y) for x in range(width)) for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def grid_xyxy(box: Box) -> list[int]:
    x, y, w, h = box
    return [
        round(x / WIDTH * GRID),
        round(y / HEIGHT * GRID),
        round((x + w) / WIDTH * GRID),
        round((y + h) / HEIGHT * GRID),
    ]


def _image(boxes: list[tuple[Box, tuple[int, int, int]]]) -> dict:
    data = base64.b64encode(png_bytes(WIDTH, HEIGHT, boxes)).decode()
    return {"type": "input_image", "detail": "auto", "image_url": f"data:image/png;base64,{data}"}


def build_row(name: str, objects: list[dict], num_frames: int) -> dict:
    """One row. Each object is {"start": (x, y), "velocity": (dx, dy), "size": (w, h), "hidden": frames}."""

    def box(obj: dict, frame: int) -> Box:
        return (
            int(obj["start"][0] + obj["velocity"][0] * frame),
            int(obj["start"][1] + obj["velocity"][1] * frame),
            obj["size"][0],
            obj["size"][1],
        )

    def frame_boxes(frame: int) -> list[tuple[Box, tuple[int, int, int]]]:
        return [
            (box(obj, frame), COLORS[index % len(COLORS)])
            for index, obj in enumerate(objects)
            if frame not in obj.get("hidden", ())
        ]

    shape = '{"tracks":[{"box":1,"frame":1,"bbox":[x1,y1,x2,y2],"visible":true}]}'
    listing = "\n".join(f"- Box {index}: {grid_xyxy(box(obj, 0))}" for index, obj in enumerate(objects, start=1))
    content: list[dict] = [
        {
            "type": "input_text",
            "text": (
                f"You are given {num_frames + 1} frames sampled in order from one video. The first frame is the "
                "reference (frame 0). Every coordinate is an integer on a 0-1000 grid with the origin at the "
                "top-left corner.\nFollow these boxed objects through every later frame:\n" + listing
            ),
        },
        _image(frame_boxes(0)),
    ]
    for frame in range(1, num_frames + 1):
        content += [{"type": "input_text", "text": f"Frame {frame}:"}, _image(frame_boxes(frame))]
    content.append(
        {
            "type": "input_text",
            "text": (
                "For every listed box and every frame after the reference, report the object's xyxy bounding box "
                "on the 0-1000 grid and whether it is visible. Return only valid JSON inside <answer>...</answer> "
                f"in this exact shape: <answer>{shape}</answer>. "
                "Include exactly one entry for each (box, frame) pair."
            ),
        }
    )
    return {
        "responses_create_params": {"input": [{"role": "user", "content": content}]},
        "verifier_metadata": {
            "sequence": name,
            "split": "example",
            "image_width": WIDTH,
            "image_height": HEIGHT,
            "objects": [
                {
                    "id": index,
                    "targets": [
                        {
                            "frame": frame,
                            "bbox": [float(value) for value in box(obj, frame)],
                            "visible": 0.0 if frame in obj.get("hidden", ()) else 1.0,
                        }
                        for frame in range(1, num_frames + 1)
                    ],
                }
                for index, obj in enumerate(objects, start=1)
            ],
        },
    }


def ground_truth_answer(row: dict) -> str:
    """The answer that scores 1.0 on a row built by `build_row`."""
    tracks = []
    for obj in row["verifier_metadata"]["objects"]:
        for target in obj["targets"]:
            x, y, w, h = (int(value) for value in target["bbox"])
            visible = target["visible"] > 0
            tracks.append(
                {"box": obj["id"], "frame": target["frame"], "bbox": grid_xyxy((x, y, w, h)), "visible": visible}
            )
    return "<answer>" + json.dumps({"tracks": tracks}, separators=(",", ":")) + "</answer>"


A = {"start": (8, 10), "velocity": (6, 2), "size": (20, 14)}
B = {"start": (60, 40), "velocity": (-5, -3), "size": (16, 12)}
C = {"start": (40, 4), "velocity": (0, 8), "size": (12, 18)}
EXAMPLE_ROWS = [
    {"name": "synthetic_000001", "objects": [A, B], "num_frames": 3},
    {"name": "synthetic_000002", "objects": [A], "num_frames": 4},
    {"name": "synthetic_000003", "objects": [A, B, C], "num_frames": 3},
    {"name": "synthetic_000004", "objects": [B, {**C, "hidden": (2,)}], "num_frames": 4},
    {"name": "synthetic_000005", "objects": [C, {**A, "hidden": (1, 2)}], "num_frames": 2},
]


def example_rows() -> list[dict]:
    return [build_row(**spec) for spec in EXAMPLE_ROWS]


def main() -> None:
    EXAMPLE_FPATH.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLE_FPATH.write_text("".join(json.dumps(row) + "\n" for row in example_rows()))
    print(f"wrote {EXAMPLE_FPATH}")


if __name__ == "__main__":
    main()
