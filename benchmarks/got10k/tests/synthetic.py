# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic GOT-10k-layout sequences: a bright rectangle drifting over a dark background.

Used by the prepare tests and to generate the committed example rows, which
therefore contain no GOT-10k imagery:

    python -m benchmarks.got10k.tests.synthetic
"""

import json
import os
import struct
import tempfile
import zlib
from pathlib import Path


WIDTH, HEIGHT = 96, 64
EXAMPLE_FPATH = Path(__file__).parents[3] / "resources_servers" / "got10k" / "data" / "example.jsonl"


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


def write_sequence(
    root: Path,
    name: str,
    num_frames: int = 31,
    anno_fps: float = 10.0,
    start: tuple[int, int] = (8, 10),
    velocity: tuple[float, float] = (1.5, 0.5),
    size: tuple[int, int] = (20, 14),
    hidden_frames: tuple[int, ...] = (),
    color: tuple[int, int, int] = (230, 90, 40),
) -> Path:
    """Write one sequence directory (frames, groundtruth.txt, cover.label, meta_info.ini)."""
    seq_dir = root / name
    seq_dir.mkdir(parents=True)
    boxes = []
    for index in range(num_frames):
        box = (int(start[0] + velocity[0] * index), int(start[1] + velocity[1] * index), size[0], size[1])
        boxes.append(box)
        (seq_dir / f"{index + 1:08d}.png").write_bytes(png_bytes(WIDTH, HEIGHT, box, color))
    (seq_dir / "groundtruth.txt").write_text("".join(f"{x:.4f},{y:.4f},{w:.4f},{h:.4f}\n" for x, y, w, h in boxes))
    (seq_dir / "cover.label").write_text("".join(f"{0 if i in hidden_frames else 8}\n" for i in range(num_frames)))
    (seq_dir / "meta_info.ini").write_text(f"[METAINFO]\nurl: synthetic\nanno_fps: {anno_fps:g}Hz\n")
    return seq_dir


def write_split(root: Path, names: list[str], **kwargs) -> Path:
    split = root / "val"
    for name in names:
        write_sequence(split, name, **kwargs)
    (split / "list.txt").write_text("\n".join(names) + "\n")
    return split


EXAMPLE_SEQUENCES = [
    {"name": "synthetic_000001"},
    {"name": "synthetic_000002", "start": (60, 40), "velocity": (-1.5, -0.8), "color": (60, 200, 120)},
    {"name": "synthetic_000003", "num_frames": 41, "velocity": (1.0, 0.0), "size": (12, 30), "hidden_frames": (20,)},
    {"name": "synthetic_000004", "num_frames": 21, "start": (30, 5), "velocity": (0.0, 2.0), "color": (70, 120, 240)},
    {"name": "synthetic_000005", "num_frames": 51, "anno_fps": 25.0, "velocity": (0.9, 0.4), "size": (30, 20)},
]


def main() -> None:
    from benchmarks.got10k import prepare

    with tempfile.TemporaryDirectory() as tmp:
        split = Path(tmp) / "val"
        for spec in EXAMPLE_SEQUENCES:
            write_sequence(split, **spec)
        (split / "list.txt").write_text("\n".join(spec["name"] for spec in EXAMPLE_SEQUENCES) + "\n")
        os.environ["GOT10K_DATASET_ROOT"] = tmp
        rows = [prepare.convert(prepare.load_sequence(split / name)) for name in prepare.sequence_names(split)]
    EXAMPLE_FPATH.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLE_FPATH.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} example rows to {EXAMPLE_FPATH}")


if __name__ == "__main__":
    main()
