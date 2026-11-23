# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MM-Lifelong v1.1 single-turn rows: day_test, week_test and month_val, the 1,023 questions graded against
frozen rubrics (month_dev has none).

The annotations, the grader prompt, the manifest and the checksums are the Hugging Face dataset MM-Lifelong/MM-Lifelong
at a pinned revision (MIT License). They are read from ``data_dir`` when present there, else downloaded into it, and
checked against the release's SHA256SUMS, whose own SHA-256 is pinned below. The videos are third-party recordings and
are not redistributed. The model sees a split through the release's frame cache (``frames/<domain>/shard_*.lmdb``:
JPEG frames at 2 fps on the split's timeline, frame N at N / 2 seconds), which the user downloads into ``data_dir``;
every shard read is checked against SHA256SUMS first.

Every question of a split asks about the split's whole recording, so a split has one model input: ``num_frames``
frames at the centres of equal bins over the timeline, written once as a short MP4 at 1 fps
(``<frames_dir>/<split>.mp4``, and ``<split>.json`` with the cache frame and timeline second of each frame). A row
sends that MP4 as a ``file://`` URL under ``frames_url_dir`` (where the model server reads ``frames_dir``) and the
question as it is, without a system prompt; the reference answer and the rubric go to the judge only
(``grading: reference_levels``, with the split as the row's subset). Rows take the splits in turn, so any prefix of
the file covers all three.

Reading the cache and writing the MP4s needs lmdb, PyAV and Pillow (the dataset's ``prepare_dependencies``).
"""

import bisect
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

from resources_servers.video_qa_rubric.task import prepare_single_turn


LOG = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "mm_lifelong_v1_1_benchmark.jsonl"
REPO_ID = "MM-Lifelong/MM-Lifelong"
DATASET_REVISION = "efa197479b25fbcf4f7881f9218d703c57dfc9bb"  # pragma: allowlist secret
CHECKSUMS = "SHA256SUMS"
CHECKSUMS_SHA256 = "b46243b886fa7bef00a1cd8c7ffd376af825280f01a2ca6b99f5b6cd93b75bad"  # pragma: allowlist secret
MANIFEST = "manifest.json"
PROMPTS = "v1.1/grading/prompts.json"
# The evaluation splits, in the order the rows take them, with their domain in the manifest and frames/ and their size.
SPLITS = {"month_val": ("month", 623), "week_test": ("week", 200), "day_test": ("day", 200)}
TASK_PREFIX = "mm_lifelong_v1_1"
DEFAULT_NUM_FRAMES = 256


def release_file(data_dir: Path, relpath: str) -> Path:
    """The release's file RELPATH under DATA_DIR, downloaded there from the pinned revision when missing."""
    path = data_dir / relpath
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        hf_hub_download(REPO_ID, relpath, repo_type="dataset", revision=DATASET_REVISION, local_dir=data_dir)
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(path: Path, expected: str) -> Path:
    found = file_sha256(path)
    if found != expected:
        raise ValueError(f"{path} has SHA-256 {found}, expected {expected} ({REPO_ID}@{DATASET_REVISION})")
    return path


def read_checksums(data_dir: Path) -> dict[str, str]:
    """SHA256SUMS of the pinned revision (checked against its pinned SHA-256), by relative path."""
    text = checked(release_file(data_dir, CHECKSUMS), CHECKSUMS_SHA256).read_text(encoding="utf-8")
    sums = {}
    for line in text.splitlines():
        if line.strip():
            digest, relpath = line.split(maxsplit=1)
            sums[relpath.strip()] = digest
    return sums


def release_json_lines(data_dir: Path, checksums: dict[str, str], relpath: str) -> list[dict[str, Any]]:
    path = checked(release_file(data_dir, relpath), checksums[relpath])
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_release(data_dir: Path) -> tuple[dict[str, str], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """The release's checksums, manifest and the questions of each evaluation split, all checked."""
    checksums = read_checksums(data_dir)
    manifest = json.loads(checked(release_file(data_dir, MANIFEST), checksums[MANIFEST]).read_text(encoding="utf-8"))
    questions = {}
    for split, (domain, count) in SPLITS.items():
        items = release_json_lines(data_dir, checksums, f"v1.1/annotations/{split}.jsonl")
        if len(items) != count or any(item.get("split") != split or not item.get("rubric") for item in items):
            raise ValueError(f"{split}: expected {count} questions of the split, each with its rubric")
        timeline = manifest["subsets"][domain]["duration_seconds"]
        for item in items:
            for start, end in item.get("total_intervals") or []:
                # Four v1.1 intervals are written end first.
                if not 0 <= min(start, end) <= max(start, end) <= timeline:
                    raise ValueError(f"{item['id']}: evidence [{start}, {end}] is outside the {timeline} s timeline")
        questions[split] = items
    return checksums, manifest, questions


def sample_frame_ids(total_frames: int, num_frames: int) -> list[int]:
    """The cache frames at the centres of NUM_FRAMES equal bins over a timeline of TOTAL_FRAMES frames."""
    if not 1 <= num_frames <= total_frames:
        raise ValueError(f"num_frames must be from 1 to {total_frames}")
    return [int((index + 0.5) * total_frames / num_frames) for index in range(num_frames)]


class FrameCache:
    """One domain of the release's LMDB frame cache, read-only, by global frame ID (an eight-digit ASCII key)."""

    def __init__(self, data_dir: Path, subset: dict[str, Any]) -> None:
        self.data_dir = data_dir
        self.subset = subset
        self.shards = sorted(subset["shards"], key=lambda shard: shard["first_frame_id"])
        self.firsts = [shard["first_frame_id"] for shard in self.shards]
        self.environments: dict[int, Any] = {}

    def shard_paths(self, frame_ids: list[int]) -> list[str]:
        """The shard files (relative paths) that hold FRAME_IDS."""
        return sorted({self.shards[self.shard_index(frame_id)]["path"] for frame_id in frame_ids})

    def shard_index(self, frame_id: int) -> int:
        if not 0 <= frame_id < self.subset["total_frames"]:
            raise IndexError(f"Frame {frame_id} is outside the {self.subset['domain']} cache")
        return bisect.bisect_right(self.firsts, frame_id) - 1

    def jpeg(self, frame_id: int) -> bytes:
        import lmdb

        index = self.shard_index(frame_id)
        if index not in self.environments:
            directory = self.data_dir / Path(self.shards[index]["path"]).parent
            self.environments[index] = lmdb.open(str(directory), readonly=True, lock=False, readahead=False)
        with self.environments[index].begin() as transaction:
            value = transaction.get(f"{frame_id:08d}".encode())
        if value is None:
            raise KeyError(f"Frame {frame_id} is missing from {self.shards[index]['path']}")
        return value

    def close(self) -> None:
        for environment in self.environments.values():
            environment.close()
        self.environments.clear()


def check_shards(data_dir: Path, checksums: dict[str, str], relpaths: list[str]) -> None:
    """Refuse a shard that SHA256SUMS does not list or whose bytes differ; the shards are hashed in parallel."""
    missing = [relpath for relpath in relpaths if relpath not in checksums]
    if missing:
        raise ValueError(f"{CHECKSUMS} does not list {missing[:3]}")
    with ThreadPoolExecutor(max_workers=min(8, len(relpaths))) as pool:
        for relpath, found in zip(relpaths, pool.map(lambda relpath: file_sha256(data_dir / relpath), relpaths)):
            if found != checksums[relpath]:
                raise ValueError(f"{data_dir / relpath} differs from {CHECKSUMS} ({REPO_ID}@{DATASET_REVISION})")


def write_frames_video(cache: FrameCache, frame_ids: list[int], path: Path) -> list[dict[str, float | int]]:
    """Write FRAME_IDS as an H.264 MP4 at 1 fps; return each frame's cache frame and timeline second."""
    import av
    from PIL import Image

    # The manifest's frame size, made even for yuv420p; a frame of another size is letterboxed into it.
    width, height = cache.subset["width"] // 2 * 2, cache.subset["height"] // 2 * 2
    partial = path.with_name(path.name + ".partial")
    with av.open(str(partial), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=1, options={"crf": "18"})
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for frame_id in frame_ids:
            image = Image.open(BytesIO(cache.jpeg(frame_id))).convert("RGB")
            if image.size != (width, height):
                scale = min(width / image.width, height / image.height)
                resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
                image = Image.new("RGB", (width, height))
                image.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    os.replace(partial, path)
    fps = cache.subset["fps_cached"]
    return [{"frame_id": frame_id, "timeline_seconds": frame_id / fps} for frame_id in frame_ids]


def interleave(rows_by_split: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """The splits' rows in turn, so that any prefix covers every split."""
    rows = []
    for index in range(max(len(rows) for rows in rows_by_split)):
        rows.extend(rows[index] for rows in rows_by_split if index < len(rows))
    return rows


def prepare(
    data_dir: str | None = None,
    frames_dir: str | None = None,
    frames_url_dir: str | None = None,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> Path:
    """Write the frame MP4s and the rows to OUTPUT_FPATH; return OUTPUT_FPATH.

    ``data_dir`` holds the release's frame cache (``frames/``) and receives the annotation files; ``frames_dir``
    receives the MP4s (default ``<data_dir>/frames_<num_frames>``); ``frames_url_dir`` is where the model server reads
    them (default ``frames_dir``), so the rows' ``file://`` URLs point there.
    """
    if data_dir is None:
        raise ValueError("Set mm_lifelong_v1_1_data_dir to the local copy of the release's frame cache (frames/)")
    root = Path(data_dir).absolute()
    out = Path(frames_dir).absolute() if frames_dir else root / f"frames_{num_frames}"
    url_dir = frames_url_dir or str(out)
    if not url_dir.startswith("/"):
        raise ValueError("frames_url_dir must be an absolute path")
    checksums, manifest, questions = load_release(root)
    out.mkdir(parents=True, exist_ok=True)
    rows_by_split = []
    for split, items in questions.items():
        subset = manifest["subsets"][SPLITS[split][0]]
        frame_ids = sample_frame_ids(subset["total_frames"], num_frames)
        cache = FrameCache(root, subset)
        check_shards(root, checksums, cache.shard_paths(frame_ids))
        try:
            frames = write_frames_video(cache, frame_ids, out / f"{split}.mp4")
        finally:
            cache.close()
        record = {"dataset_revision": DATASET_REVISION, "domain": subset["domain"], "frames": frames}
        (out / f"{split}.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        LOG.info("%s: %d questions, %d frames over %.1f s", split, len(items), num_frames, subset["duration_seconds"])
        video_url = f"file://{url_dir.rstrip('/')}/{split}.mp4"
        rows_by_split.append(
            [
                prepare_single_turn(
                    f"{TASK_PREFIX}/{item['id']}",
                    item["question"],
                    item["rubric"],
                    video_url=video_url,
                    prompt="{question}",
                    grading="reference_levels",
                    reference_answer=item["answer"],
                    subset=split,
                )
                for item in items
            ]
        )
    rows = interleave(rows_by_split)
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FPATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return OUTPUT_FPATH
