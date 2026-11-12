# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic V-STaR-layout items: a bright rectangle drifting over a dark background.

Used by the prepare tests and to generate the committed example rows, which
therefore contain no V-STaR footage:

    python -m benchmarks.vstar_spatiotemporal.tests.synthetic
"""

import json
import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any


WIDTH, HEIGHT = 96, 64
SIZE = (20, 14)
EXAMPLE_FPATH = Path(__file__).parents[3] / "resources_servers" / "vstar_spatiotemporal" / "data" / "example.jsonl"


def png_bytes(width: int, height: int, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> bytes:
    """An RGB PNG with `box` (x, y, w, h) filled in `color`."""
    x0, y0, w, h = box
    background, foreground = bytes((24, 28, 36)), bytes(color)
    raw = b"".join(
        b"\x00" + b"".join(foreground if x0 <= x < x0 + w and y0 <= y < y0 + h else background for x in range(width))
        for y in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def box_at(second: float, start: tuple[int, int] = (8, 10), velocity: tuple[float, float] = (12.0, 6.0)) -> list[int]:
    """[xmin, ymin, xmax, ymax] of the rectangle at `second`."""
    x, y = int(start[0] + velocity[0] * second), int(start[1] + velocity[1] * second)
    return [x, y, x + SIZE[0], y + SIZE[1]]


def make_item(
    vid: str,
    seconds: int = 5,
    fps: float = 10.0,
    span: tuple[float, float] = (1.2, 3.6),
    color: str = "orange",
    domain: str = "Synthetic",
) -> dict[str, Any]:
    """One annotation in the V_STaR_test.json layout, with a box at every whole second inside `span`."""
    whole_seconds = [t for t in range(seconds + 1) if span[0] <= t <= span[1]]
    return {
        "vid": vid,
        "domain": domain,
        "fps": fps,
        "width": WIDTH,
        "height": HEIGHT,
        "frame_count": int(seconds * fps),
        "question": f"what moves across the {color} frame?",
        "answer": f"a {color} rectangle",
        "temporal_question": f"When does the {color} rectangle cross the middle?",
        "timestamps": list(span),
        "spatial_question": f"Where is the {color} rectangle while it crosses the middle?",
        "spatial_question_2": f"Where is the {color} rectangle?",
        "bboxes": [dict(zip(("xmin", "ymin", "xmax", "ymax"), box_at(t)), timestamp=t) for t in whole_seconds],
    }


def frame_pngs(times: list[float], color: tuple[int, int, int] = (230, 90, 40)) -> list[bytes]:
    """One PNG per time, showing the rectangle where `box_at` puts it."""
    frames = []
    for t in times:
        xmin, ymin, xmax, ymax = box_at(t)
        frames.append(png_bytes(WIDTH, HEIGHT, (xmin, ymin, xmax - xmin, ymax - ymin), color))
    return frames


def write_frames(directory: Path, times: list[float]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, data in enumerate(frame_pngs(times), 1):
        path = directory / f"{number:02d}.png"
        path.write_bytes(data)
        paths.append(path)
    return paths


EXAMPLE_ITEM = make_item("synthetic_0001")
EXAMPLE_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]


def main() -> None:
    from benchmarks.vstar_spatiotemporal import prepare

    os.environ["VSTAR_EMBED_FRAMES"] = "1"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames = write_frames(root / EXAMPLE_ITEM["vid"], EXAMPLE_TIMES)
        rows = [
            prepare.convert(EXAMPLE_ITEM, 0, sub_task, frames, EXAMPLE_TIMES, root) for sub_task in prepare.SUB_TASKS
        ]
    EXAMPLE_FPATH.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLE_FPATH.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} example rows to {EXAMPLE_FPATH}")


if __name__ == "__main__":
    main()
