# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a prepared Molmo2-VideoTrackEval tree into Gym benchmark JSONL.

The dataset (allenai/Molmo2-VideoTrackEval) ships annotations only. The video
frames come from five source datasets with their own terms and must be
extracted once with the dataset's `scripts/preprocess_videos.py`, giving

    <root>/data/<category>/*.parquet
    <root>/videos/<category>/<clip>/<local frame index, 5 digits>.jpg

Point MOLMO2_VIDEOTRACK_ROOT at <root>.

Each annotation row (one referring expression over one clip) becomes one task
whose prompt holds the clip's frames sampled 1 second apart. Ground-truth
points and masks stay in `verifier_metadata` and never enter the prompt.

By default frames are referenced as file:// URLs, which keeps the JSONL small
(about 2.3 GB of frames are shared between expressions) but requires a model
server that reads local files, such as vLLM with --allowed-local-media-path.
Set MOLMO2_VIDEOTRACK_FRAME_URL_ROOT when the server sees <root>/videos at a
different path. Set MOLMO2_VIDEOTRACK_EMBED_FRAMES=1 to embed frames as data
URLs instead (about 7 GB of JSONL).
"""

import base64
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote


HF_REPO = "allenai/Molmo2-VideoTrackEval"

# Category -> annotation file, in the order tasks are written.
CATEGORIES = {
    "misc": "data/misc/sav_point_tracks_with_masks.parquet",
    "animal": "data/animal/APTv2_point_tracks_with_masks.parquet",
    "dance": "data/dance/dancetrack_point_tracks_with_masks.parquet",
    "person": "data/person/personpath22_point_tracks_with_masks.parquet",
    "sports": "data/sports/sportsmot_point_tracks_with_masks.parquet",
}
COLUMNS = ["id", "video", "clip", "exp", "points", "masks", "w", "h", "n_frames", "fps"]

BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "molmo2_videotrack_benchmark.jsonl"

INTRO_TEXT = (
    "The following {n} frames are taken 1 second apart, in temporal order, from a single video.\n"
    'Track every object matching this description: "{expression}".\n'
    "There may be one or several matching objects."
)
INSTRUCTION_TEXT = (
    "For each frame, report the [x,y] location of every matching object that is visible in that frame, "
    "keeping a consistent integer id per object across frames. If an object is not visible in a frame, "
    "omit its entry for that frame. Return only valid JSON inside <answer>...</answer> in this exact shape: "
    '<answer>{"tracks":[{"id":1,"frame":1,"xy":[x,y]}]}</answer>. '
    "Coordinates must be integer values on a 0-1000 grid using [x,y] order "
    "(each axis spans 0-1000, independent of pixel size)."
)


def dataset_root() -> Path:
    root = os.environ.get("MOLMO2_VIDEOTRACK_ROOT")
    if not root:
        raise RuntimeError(
            f"MOLMO2_VIDEOTRACK_ROOT is not set. {HF_REPO} ships annotations only; set it to a directory holding "
            "data/<category>/*.parquet and the frames extracted by the dataset's scripts/preprocess_videos.py."
        )
    return Path(root)


def frame_url(frame: Path, videos_dir: Path) -> str:
    """The image_url for one frame: a data URL, or a file:// URL as the model server sees it."""
    if os.environ.get("MOLMO2_VIDEOTRACK_EMBED_FRAMES", "0") == "1":
        data = frame.read_bytes()
        mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(data).decode()
    url_root = os.environ.get("MOLMO2_VIDEOTRACK_FRAME_URL_ROOT") or str(videos_dir.absolute())
    return "file://" + quote(f"{url_root.rstrip('/')}/{frame.relative_to(videos_dir).as_posix()}")


def sampled_frame_indices(n_frames: int, fps: float) -> list[int]:
    """Local frame indices 1 second apart, starting at the clip's first frame."""
    return list(range(0, n_frames, max(int(round(fps)), 1)))


def visible_objects(row: dict[str, Any], index: int) -> list[dict[str, Any]]:
    """Objects with both a point and a mask at this frame, i.e. the ones visible in it."""
    objects = []
    for track, masks in zip(row["points"], row["masks"]):
        point = track["points"][index] if index < len(track["points"]) else None
        mask = masks["masks"][index] if index < len(masks["masks"]) else None
        if point is None or len(point) < 2 or mask is None or mask.get("counts") is None:
            continue
        x, y = float(point[0]), float(point[1])
        if math.isnan(x) or math.isnan(y):
            continue
        objects.append({"xy": [x, y], "mask": {"size": [int(v) for v in mask["size"]], "counts": mask["counts"]}})
    return objects


def convert(row: dict[str, Any], category: str, videos_dir: Path) -> dict[str, Any]:
    """One annotation row -> one benchmark row. Raises if a sampled frame is missing."""
    clip_dir = videos_dir / category / str(row["clip"])
    indices = sampled_frame_indices(int(row["n_frames"]), float(row["fps"]))
    frames = [clip_dir / f"{index:05d}.jpg" for index in indices]
    missing = [frame for frame in frames if not frame.is_file()]
    if missing or not frames:
        raise FileNotFoundError(
            f"Molmo2-VideoTrackEval: clip {category}/{row['clip']} is missing {len(missing)} of {len(frames)} "
            f"required 1 fps frames (first: {missing[0] if missing else clip_dir}). Refusing a partial evaluation."
        )

    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": INTRO_TEXT.format(n=len(frames), expression=row["exp"])}
    ]
    for number, frame in enumerate(frames, start=1):
        content.append({"type": "input_text", "text": f"Frame {number}:"})
        content.append({"type": "input_image", "detail": "auto", "image_url": frame_url(frame, videos_dir)})
    content.append({"type": "input_text", "text": INSTRUCTION_TEXT})

    return {
        "responses_create_params": {"input": [{"role": "user", "content": content}]},
        "verifier_metadata": {
            "id": row["id"],
            "category": category,
            "video": row["video"],
            "clip": row["clip"],
            "expression": row["exp"],
            "fps": float(row["fps"]),
            "image_width": int(row["w"]),
            "image_height": int(row["h"]),
            "frames": [
                {"frame": number, "source_frame": index, "objects": visible_objects(row, index)}
                for number, index in enumerate(indices, start=1)
            ],
        },
    }


def iter_annotation_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Annotation rows in file order, a few at a time: every row carries masks for all its frames."""
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=COLUMNS):
        yield from batch.to_pylist()


def prepare() -> Path:
    root = dataset_root()
    videos_dir = root / "videos"
    for category, relative in CATEGORIES.items():
        if not (root / relative).is_file():
            raise FileNotFoundError(f"Molmo2-VideoTrackEval: annotations missing for {category}: {root / relative}")

    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as out:
        for category, relative in CATEGORIES.items():
            counts[category] = 0
            for row in iter_annotation_rows(root / relative):
                out.write(json.dumps(convert(row, category, videos_dir)) + "\n")
                counts[category] += 1
    print(f"Wrote {sum(counts.values())} tasks to {OUTPUT_FPATH}: {counts}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
