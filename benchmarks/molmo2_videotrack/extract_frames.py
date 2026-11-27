# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the frame folders that `prepare.py` reads, from the five source video datasets.

allenai/Molmo2-VideoTrackEval ships annotations only. Each annotation row covers
frames [start_frame, end_frame] of one source video, and its `points` and `masks`
arrays hold one entry per frame of that range. This script writes, for every clip,

    <root>/videos/<category>/<clip>/00000.jpg .. <n_frames - 1, 5 digits>.jpg

so that file `i` is the frame the annotation arrays describe at index `i`.

Download each source dataset under its own terms, then pass the folders you have (from the repository root):

    python -m benchmarks.molmo2_videotrack.extract_frames --root /path/to/Molmo2-VideoTrackEval \\
        --aptv2 /path/to/APTv2/videos --dancetrack /path/to/dancetrack/val \\
        --sportsmot /path/to/sportsmot/val --sav /path/to/sav_test/JPEGImages_24fps \\
        --personpath22 /path/to/personpath22/videos

How clip frame `i` (source frame `j = start_frame + i`) is found in each source:

    animal  --aptv2         <video>/<j, 4 digits>.jpg
    dance   --dancetrack    <video>/img1/<j + 1, 8 digits>.jpg
    sports  --sportsmot     <video>/img1/<j + 1, 6 digits>.jpg
    misc    --sav           <video>/<4 * j, 5 digits>.jpg  (annotations are at 6 fps, SA-V frames at 24 fps)
    person  --personpath22  frame j, counted from 0, of <video>.mp4 as ffmpeg decodes it

PersonPath22 needs `ffmpeg` on PATH and the videos as its own download script
writes them, including those it fetches and re-encodes from MEVA, VIRAT and
PathTrack. A category whose folder is not given is skipped. A clip folder that
already holds its frames is kept, so an interrupted run can be repeated.
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Callable, Iterator, NamedTuple, Optional

from benchmarks.molmo2_videotrack.prepare import CATEGORIES


class Clip(NamedTuple):
    """The frame range of one source video that a clip's annotations cover."""

    name: str
    video: str
    start_frame: int
    end_frame: int


# Category -> where source frame `j` of a video is, for the sources that ship frames as image files.
FRAME_FILE_SOURCES: dict[str, Callable[[Path, str, int], Path]] = {
    "animal": lambda source, video, j: source / video / f"{j:04d}.jpg",
    "dance": lambda source, video, j: source / video / "img1" / f"{j + 1:08d}.jpg",
    "sports": lambda source, video, j: source / video / "img1" / f"{j + 1:06d}.jpg",
    "misc": lambda source, video, j: source / video / f"{4 * j:05d}.jpg",
}
SOURCE_OPTIONS = {
    "animal": "aptv2",
    "dance": "dancetrack",
    "sports": "sportsmot",
    "misc": "sav",
    "person": "personpath22",
}


def clips(root: Path, category: str) -> list[Clip]:
    """The distinct clips of one category, in annotation order."""
    import pyarrow.parquet as pq

    path = root / CATEGORIES[category]
    if not path.is_file():
        raise FileNotFoundError(f"Molmo2-VideoTrackEval: {path} is missing.")
    columns = ["clip", "video", "start_frame", "end_frame", "n_frames"]
    found: dict[str, Clip] = {}
    for row in pq.read_table(path, columns=columns).to_pylist():
        clip = Clip(str(row["clip"]), str(row["video"]), int(row["start_frame"]), int(row["end_frame"]))
        if clip.end_frame - clip.start_frame + 1 != int(row["n_frames"]):
            raise ValueError(
                f"Molmo2-VideoTrackEval: clip {category}/{clip.name} declares {row['n_frames']} frames "
                f"but covers frames {clip.start_frame} to {clip.end_frame}."
            )
        if found.setdefault(clip.name, clip) != clip:
            raise ValueError(f"Molmo2-VideoTrackEval: clip {category}/{clip.name} has two different frame ranges.")
    return list(found.values())


def clip_is_complete(clip_dir: Path, clip: Clip) -> bool:
    """Does the folder hold exactly the clip's frames?"""
    expected = {f"{index:05d}.jpg" for index in range(clip.end_frame - clip.start_frame + 1)}
    return clip_dir.is_dir() and {entry.name for entry in clip_dir.iterdir()} == expected


def write_clip(clip_dir: Path, frames: list[Path]) -> None:
    """Copy the frames to `clip_dir` as 00000.jpg, 00001.jpg, ...; the folder appears only when complete."""
    missing = [frame for frame in frames if not frame.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Molmo2-VideoTrackEval: {len(missing)} of the {len(frames)} source frames of clip "
            f"{clip_dir.parent.name}/{clip_dir.name} are missing (first: {missing[0]})."
        )
    clip_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(dir=clip_dir.parent, prefix=".partial-"))
    try:
        for index, frame in enumerate(frames):
            shutil.copyfile(frame, staging / f"{index:05d}.jpg")
        if clip_dir.exists():
            shutil.rmtree(clip_dir)
        staging.rename(clip_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def decoded_frames(video: Path, staging: Path) -> list[Path]:
    """Every frame of a video file, decoded by ffmpeg into `staging` in order."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Molmo2-VideoTrackEval: ffmpeg is needed on PATH to decode the PersonPath22 videos.")
    if not video.is_file():
        raise FileNotFoundError(f"Molmo2-VideoTrackEval: source video {video} is missing.")
    command = [ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(video), "-start_number", "0", "-q:v", "2"]
    subprocess.run([*command, str(staging / "%06d.jpg")], check=True)
    return sorted(staging.iterdir())


def extract_category(root: Path, category: str, source: Path) -> Iterator[str]:
    """Write the missing clip folders of one category, yielding each clip name as it is written."""
    category_dir = root / "videos" / category
    pending = [clip for clip in clips(root, category) if not clip_is_complete(category_dir / clip.name, clip)]

    if category in FRAME_FILE_SOURCES:
        locate = FRAME_FILE_SOURCES[category]
        for clip in pending:
            frames = [locate(source, clip.video, j) for j in range(clip.start_frame, clip.end_frame + 1)]
            write_clip(category_dir / clip.name, frames)
            yield clip.name
        return

    # Several clips share one video file, so each video is decoded once.
    by_video: dict[str, list[Clip]] = {}
    for clip in pending:
        by_video.setdefault(clip.video, []).append(clip)
    for video, video_clips in by_video.items():
        with TemporaryDirectory(dir=root, prefix=".decoded-") as staging:
            frames = decoded_frames(source / f"{video}.mp4", Path(staging))
            for clip in video_clips:
                if clip.end_frame >= len(frames):
                    raise ValueError(
                        f"Molmo2-VideoTrackEval: clip {category}/{clip.name} needs frame {clip.end_frame} of "
                        f"{video}.mp4, which decodes to only {len(frames)} frames."
                    )
                write_clip(category_dir / clip.name, frames[clip.start_frame : clip.end_frame + 1])
                yield clip.name


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="dataset root holding data/; videos/ is written here")
    parser.add_argument("--aptv2", type=Path, help="APTv2 folder holding <video>/<frame, 4 digits>.jpg")
    parser.add_argument("--dancetrack", type=Path, help="DanceTrack split folder holding <video>/img1/")
    parser.add_argument("--sportsmot", type=Path, help="SportsMOT split folder holding <video>/img1/")
    parser.add_argument("--sav", type=Path, help="SA-V JPEGImages_24fps folder holding <video>/<frame, 5 digits>.jpg")
    parser.add_argument("--personpath22", type=Path, help="PersonPath22 folder holding <video>.mp4")
    args = parser.parse_args(argv)

    sources = {category: getattr(args, option) for category, option in SOURCE_OPTIONS.items()}
    if not any(sources.values()):
        parser.error("give at least one source folder")
    for category, source in sources.items():
        if source is None:
            print(f"{category}: skipped, --{SOURCE_OPTIONS[category]} not given")
            continue
        written = sum(1 for _ in extract_category(args.root, category, source))
        print(f"{category}: wrote {written} clips, {len(clips(args.root, category))} in total")


if __name__ == "__main__":
    main()
