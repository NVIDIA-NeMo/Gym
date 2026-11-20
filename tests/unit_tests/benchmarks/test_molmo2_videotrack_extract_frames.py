# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.molmo2_videotrack import extract_frames, prepare


# Frame files that ship as images, per category: (first index in the source, step between annotated frames, name).
SOURCE_LAYOUTS = {
    "animal": (0, 1, lambda video, number: f"{video}/{number:04d}.jpg"),
    "dance": (1, 1, lambda video, number: f"{video}/img1/{number:08d}.jpg"),
    "sports": (1, 1, lambda video, number: f"{video}/img1/{number:06d}.jpg"),
    "misc": (0, 4, lambda video, number: f"{video}/{number:05d}.jpg"),
}


def write_annotations(root: Path, category: str, rows: list[dict]) -> None:
    path = root / prepare.CATEGORIES[category]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def annotation(clip: str, video: str, start: int, end: int) -> dict:
    return {"clip": clip, "video": video, "start_frame": start, "end_frame": end, "n_frames": end - start + 1}


def write_source_frames(source: Path, category: str, video: str, count: int) -> None:
    """`count` annotated-rate frames of a video; each file's content is its own number in the source."""
    first, step, name = SOURCE_LAYOUTS[category]
    for number in range(first, first + count * step):
        path = source / name(video, number)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(number))


def clip_contents(root: Path, category: str, clip: str) -> dict[str, str]:
    return {path.name: path.read_text() for path in sorted((root / "videos" / category / clip).iterdir())}


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        # Clip frame i is source frame start_frame + i = 3 + i.
        ("animal", ["3", "4", "5"]),  # files are numbered from 0
        ("dance", ["4", "5", "6"]),  # files are numbered from 1
        ("sports", ["4", "5", "6"]),  # files are numbered from 1
        ("misc", ["12", "16", "20"]),  # 6 fps annotations over 24 fps frames
    ],
)
def test_clip_frames_are_the_annotated_range_of_the_source_video(
    tmp_path: Path, category: str, expected: list[str]
) -> None:
    root, source = tmp_path / "root", tmp_path / "source"
    # Two expressions over one clip, and a second clip of the same video.
    rows = [
        annotation("first", "video", 3, 5),
        annotation("first", "video", 3, 5),
        annotation("second", "video", 0, 1),
    ]
    write_annotations(root, category, rows)
    write_source_frames(source, category, "video", count=8)

    assert list(extract_frames.extract_category(root, category, source)) == ["first", "second"]

    assert clip_contents(root, category, "first") == dict(zip(["00000.jpg", "00001.jpg", "00002.jpg"], expected))
    assert sorted(clip_contents(root, category, "second")) == ["00000.jpg", "00001.jpg"]
    # The frames are copies: a model server that only reads below the frames folder can open them.
    assert not any(path.is_symlink() for path in (root / "videos").rglob("*.jpg"))


def test_complete_clips_are_kept_and_incomplete_ones_rebuilt(tmp_path: Path) -> None:
    root, source = tmp_path / "root", tmp_path / "source"
    write_annotations(root, "animal", [annotation("kept", "video", 0, 1), annotation("short", "video", 2, 4)])
    write_source_frames(source, "animal", "video", count=5)
    assert len(list(extract_frames.extract_category(root, "animal", source))) == 2

    kept = root / "videos" / "animal" / "kept" / "00000.jpg"
    kept.write_text("edited after extraction")
    (root / "videos" / "animal" / "short" / "00002.jpg").unlink()

    assert list(extract_frames.extract_category(root, "animal", source)) == ["short"]
    assert kept.read_text() == "edited after extraction"
    assert clip_contents(root, "animal", "short") == {"00000.jpg": "2", "00001.jpg": "3", "00002.jpg": "4"}


def test_a_clip_with_a_missing_source_frame_gets_no_folder(tmp_path: Path) -> None:
    root, source = tmp_path / "root", tmp_path / "source"
    write_annotations(root, "sports", [annotation("clip", "video", 0, 4)])
    write_source_frames(source, "sports", "video", count=4)  # the clip needs five

    with pytest.raises(FileNotFoundError, match=r"1 of the 5 source frames of clip sports/clip are missing"):
        list(extract_frames.extract_category(root, "sports", source))

    assert not (root / "videos").exists()


def test_annotations_whose_frame_count_disagrees_with_their_range_are_refused(tmp_path: Path) -> None:
    row = {**annotation("clip", "video", 2, 6), "n_frames": 4}
    write_annotations(tmp_path, "dance", [row])

    with pytest.raises(ValueError, match=r"dance/clip declares 4 frames but covers frames 2 to 6"):
        extract_frames.clips(tmp_path, "dance")


def test_only_the_categories_whose_source_is_given_are_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, source = tmp_path / "root", tmp_path / "source"
    write_annotations(root, "misc", [annotation("clip", "video", 1, 2)])
    write_source_frames(source, "misc", "video", count=3)

    extract_frames.main(["--root", str(root), "--sav", str(source)])

    assert clip_contents(root, "misc", "clip") == {"00000.jpg": "4", "00001.jpg": "8"}
    assert [path.name for path in (root / "videos").iterdir()] == ["misc"]
    output = capsys.readouterr().out
    assert "misc: wrote 1 clips, 1 in total" in output
    assert "person: skipped, --personpath22 not given" in output


def test_extracted_folders_are_accepted_by_prepare(tmp_path: Path) -> None:
    """The folder names and frame counts are the ones prepare.py looks for."""
    root, source = tmp_path / "root", tmp_path / "source"
    write_annotations(root, "misc", [annotation("clip", "video", 1, 13)])
    write_source_frames(source, "misc", "video", count=14)
    list(extract_frames.extract_category(root, "misc", source))

    row = {"id": "clip_0", "video": "video", "clip": "clip", "exp": "the box", "points": [], "masks": []}
    converted = prepare.convert({**row, "w": 96, "h": 64, "n_frames": 13, "fps": 6.0}, "misc", root / "videos")

    frames = converted["verifier_metadata"]["frames"]
    assert [frame["source_frame"] for frame in frames] == [0, 6, 12]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_video_file_clips_are_cut_by_decoded_frame_index(tmp_path: Path) -> None:
    root, source = tmp_path / "root", tmp_path / "source"
    source.mkdir()
    encode = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=2"]
    subprocess.run([*encode, str(source / "video.mp4")], check=True)
    write_annotations(root, "person", [annotation("early", "video", 0, 2), annotation("late", "video", 6, 9)])

    assert list(extract_frames.extract_category(root, "person", source)) == ["early", "late"]

    reference = tmp_path / "reference"
    reference.mkdir()
    decoded = extract_frames.decoded_frames(source / "video.mp4", reference)
    assert len(decoded) == 10
    late = sorted((root / "videos" / "person" / "late").iterdir())
    assert [path.name for path in late] == ["00000.jpg", "00001.jpg", "00002.jpg", "00003.jpg"]
    assert [path.read_bytes() for path in late] == [path.read_bytes() for path in decoded[6:10]]
    assert late[0].read_bytes() != late[1].read_bytes()
    # The decoding folder is removed.
    assert sorted(path.name for path in root.iterdir()) == ["data", "videos"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_a_clip_past_the_end_of_its_video_is_refused(tmp_path: Path) -> None:
    root, source = tmp_path / "root", tmp_path / "source"
    source.mkdir()
    encode = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=1"]
    subprocess.run([*encode, str(source / "video.mp4")], check=True)
    write_annotations(root, "person", [annotation("clip", "video", 3, 5)])

    with pytest.raises(ValueError, match=r"needs frame 5 of video.mp4, which decodes to only 5 frames"):
        list(extract_frames.extract_category(root, "person", source))
    assert not (root / "videos" / "person" / "clip").exists()
