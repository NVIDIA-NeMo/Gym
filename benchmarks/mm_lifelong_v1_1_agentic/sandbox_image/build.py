# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the sandbox image of one MM-Lifelong v1.1 domain (day, week or month) from your own copy of its videos.

The dataset's videos are third-party recordings that may not be redistributed, so no image is published: each user
builds the three images from the videos they downloaded from the Hugging Face dataset MM-Lifelong/MM-Lifelong, e.g.

    hf download MM-Lifelong/MM-Lifelong --repo-type dataset \\
        --revision efa197479b25fbcf4f7881f9218d703c57dfc9bb --include 'videos/day/*' --local-dir DATA_DIR

The script

1. lists the domain's files at the pinned revision with the SHA-256 and size the Hub keeps for each (no download) and
   checks every local file against them;
2. measures each file's playable duration as the dataset's agentic runner does (ffprobe ``format=duration``) and
   writes the two catalogs into ``OUTPUT_DIR/<domain>/``: ``catalog.json`` (path, bytes and SHA-256 of every file,
   which the image holds and the resources server checks) and ``video_catalog.json`` (what the model reads: the
   runner's records of path, duration and the recording's place in time from the dataset's timeline,
   src/codex_video_timeline.json at revision b8a87afff4e24e76f6333937fb7b1c77dd5036fc, MIT License);
3. writes a Dockerfile that adds the videos to the video tools' image (BASE_IMAGE, built from this repository's
   resources_servers/video_mcqa/sandbox_image) under /testbed/videos/<domain>/, one layer per top-level file or day
   folder and read-only, and the catalog as /testbed/videos/catalog.json; and builds it with
   ``docker buildx build`` unless ``--no-build`` is given.

The agentic benchmark's prepare.py reads ``OUTPUT_DIR`` (``mm_lifelong_v1_1_agentic_media_dir``) to pin each row to
its image's catalog. Usage:

    python benchmarks/mm_lifelong_v1_1_agentic/sandbox_image/build.py --domain day --data-dir DATA_DIR \\
        --base-image BASE_IMAGE --tag IMAGE_TAG --output-dir OUTPUT_DIR
"""

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ID = "MM-Lifelong/MM-Lifelong"
DATASET_REVISION = "efa197479b25fbcf4f7881f9218d703c57dfc9bb"  # pragma: allowlist secret
TIMELINE = "src/codex_video_timeline.json"
TIMELINE_REVISION = "b8a87afff4e24e76f6333937fb7b1c77dd5036fc"  # pragma: allowlist secret
TIMELINE_SHA256 = "c6fdf4191a6bfe52d6f620223bba9795b0d547e1183f7f399cce863626438b5f"  # pragma: allowlist secret
DOMAINS = ("day", "week", "month")
SANDBOX_VIDEOS = PurePosixPath("/testbed/videos")


def natural_key(path: str) -> list[int | str]:
    """The runner's file order: numbers by value, other text case-blind."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path)]


def release_files(domain: str) -> dict[str, tuple[int, str]]:
    """The domain's files at the pinned revision, relative to videos/<domain>/, with the size and SHA-256 the Hub
    keeps for each."""
    from huggingface_hub import HfApi

    files = {}
    tree = HfApi().list_repo_tree(
        REPO_ID, path_in_repo=f"videos/{domain}", repo_type="dataset", revision=DATASET_REVISION, recursive=True
    )
    for entry in tree:
        lfs = getattr(entry, "lfs", None)
        if lfs is not None:
            files[entry.path.removeprefix(f"videos/{domain}/")] = (lfs.size, lfs.sha256)
    if not files:
        raise ValueError(f"{REPO_ID}@{DATASET_REVISION} lists no videos/{domain}/ files")
    return files


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def check_videos(videos: Path, expected: dict[str, tuple[int, str]], workers: int) -> None:
    """Every expected file is present with its size and SHA-256 (sizes first: they cost no read)."""
    for relpath, (size, _) in expected.items():
        path = videos / relpath
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"{path} is missing or not {size} bytes, as {REPO_ID}@{DATASET_REVISION} has it")
    relpaths = sorted(expected)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for relpath, found in zip(relpaths, pool.map(lambda relpath: file_sha256(videos / relpath), relpaths)):
            if found != expected[relpath][1]:
                raise ValueError(f"{videos / relpath} differs from {REPO_ID}@{DATASET_REVISION}")


def duration(path: Path, ffprobe: str) -> float:
    """The playable length in seconds, as the dataset's agentic runner reads it."""
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    seconds = float(json.loads(result.stdout)["format"]["duration"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{path} has no positive duration")
    return seconds


def model_records(domain: str, durations: dict[str, float], timeline: dict[str, Any]) -> list[dict[str, Any]]:
    """The runner's catalog records: duration, the recording's place in time, then the file's path in the sandbox."""
    records = []
    for relpath in sorted(durations, key=natural_key):
        record: dict[str, Any] = {"duration_seconds": durations[relpath]}
        if domain == "month":
            (place,) = [entry for entry in timeline["month"] if entry["video_id"] == PurePosixPath(relpath).stem]
            record.update(place)
        elif domain == "week":
            match = re.search(r"day[_ -]?(\d+)", relpath, re.IGNORECASE)
            if match is None:
                raise ValueError(f"The week file {relpath} has no day folder")
            day = int(match.group(1))
            (session,) = [entry for entry in timeline["week"] if entry["day"] == day]
            record.update(
                day=day, day_physical_start=session["physical_start"], day_physical_end=session["physical_end"]
            )
        else:
            record.update(
                session_physical_start=timeline["day"]["physical_start"],
                session_physical_end=timeline["day"]["physical_end"],
            )
        record["path"] = str(SANDBOX_VIDEOS / domain / relpath)
        records.append(record)
    return records


def dockerfile(domain: str, relpaths: list[str]) -> str:
    """The image: the base, then one read-only layer per top-level file or folder of the domain, then the catalog."""
    lines = ["# syntax=docker/dockerfile:1.7", "ARG BASE_IMAGE", "FROM ${BASE_IMAGE}"]
    # COPY --chmod would give a folder it creates the files' mode, and a folder needs its execute bits.
    lines.append(f"RUN mkdir -p -m 0555 {SANDBOX_VIDEOS / domain}")
    for top in sorted({relpath.split("/", 1)[0] for relpath in relpaths}, key=natural_key):
        mode = "0444" if top in relpaths else "0555"
        target = SANDBOX_VIDEOS / domain / top
        lines.append(f"COPY --from=videos --chmod={mode} {shlex.quote(top)} {shlex.quote(str(target))}")
    lines.append(f"COPY --chmod=0444 catalog.json {SANDBOX_VIDEOS / 'catalog.json'}")
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> Path:
    """Check the videos, write the catalogs and the Dockerfile, and build the image; return the output folder."""
    from huggingface_hub import hf_hub_download

    videos = args.data_dir / "videos" / args.domain
    expected = release_files(args.domain)
    check_videos(videos, expected, args.workers)
    timeline_path = Path(hf_hub_download(REPO_ID, TIMELINE, repo_type="dataset", revision=TIMELINE_REVISION))
    if file_sha256(timeline_path) != TIMELINE_SHA256:
        raise ValueError(f"{timeline_path} differs from {REPO_ID}@{TIMELINE_REVISION}")
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    relpaths = sorted(expected, key=natural_key)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        durations = dict(zip(relpaths, pool.map(lambda relpath: duration(videos / relpath, args.ffprobe), relpaths)))
    output = args.output_dir / args.domain
    output.mkdir(parents=True, exist_ok=True)
    catalog = {
        "videos": [
            {
                "path": str(SANDBOX_VIDEOS / args.domain / relpath),
                "bytes": expected[relpath][0],
                "sha256": expected[relpath][1],
            }
            for relpath in relpaths
        ]
    }
    (output / "catalog.json").write_text(json.dumps(catalog, indent=1) + "\n", encoding="utf-8")
    records = model_records(args.domain, durations, timeline)
    (output / "video_catalog.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (output / "Dockerfile").write_text(dockerfile(args.domain, relpaths), encoding="utf-8")
    print(f"{output / 'catalog.json'}: sha256 {file_sha256(output / 'catalog.json')}")
    if not args.no_build:
        subprocess.run(
            [
                "docker",
                "buildx",
                "build",
                "--build-arg",
                f"BASE_IMAGE={args.base_image}",
                "--build-context",
                f"videos={videos}",
                "--tag",
                args.tag,
                "--load",
                str(output),
            ],
            check=True,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="holds the dataset's videos/<domain>/")
    parser.add_argument("--output-dir", type=Path, required=True, help="receives <domain>/ with the catalogs")
    parser.add_argument("--base-image", help="an image built from resources_servers/video_mcqa/sandbox_image")
    parser.add_argument("--tag", help="the image to build")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-build", action="store_true", help="write the catalogs and the Dockerfile only")
    args = parser.parse_args()
    if not args.no_build and not (args.base_image and args.tag):
        parser.error("--base-image and --tag are needed to build the image")
    build(args)


if __name__ == "__main__":
    main()
