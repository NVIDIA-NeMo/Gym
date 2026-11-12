# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import struct
from pathlib import Path

import pytest

from benchmarks.got10k import prepare
from benchmarks.got10k.tests.synthetic import HEIGHT, WIDTH, png_bytes, write_sequence, write_split
from nemo_gym.environment import load_manifest, resolve_manifest_config_path
from nemo_gym.environment.validation import _mirror_differences, _resolve_manifest_composition


BENCHMARK_DIR = Path(prepare.__file__).parent


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GOT10K_DATASET_ROOT", str(tmp_path))
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "out" / "got10k_benchmark.jsonl")
    return tmp_path


def test_manifest_mirrors_the_resolved_config() -> None:
    path = BENCHMARK_DIR / "manifest.yaml"
    manifest = load_manifest(path)
    composition = _resolve_manifest_composition(
        resolve_manifest_config_path(path, manifest), dataset_owner=manifest.dataset_owner
    )

    assert _mirror_differences(manifest, composition) == {}


@pytest.mark.parametrize(
    ("num_frames", "anno_fps", "expected"),
    [
        (60, 10.0, [10, 20, 30, 40, 50]),  # 5.9 s of video
        (61, 10.0, [10, 20, 30, 40, 50, 60]),  # exactly 6.0 s: the last frame is included
        (10, 10.0, []),  # shorter than one second
        (51, 25.0, [25, 50]),
        (16, 2.5, [2, 5, 8, 10, 12, 15]),  # non-integer rate: round-half-to-even on 2.5 and 12.5
        (4, 1.0, [1, 2, 3]),  # annotated at or below the sampling rate: every frame
        (4, 0.5, [1, 2, 3]),
    ],
)
def test_target_frames_are_one_per_second_after_the_first(
    num_frames: int, anno_fps: float, expected: list[int]
) -> None:
    assert prepare.select_target_indices(num_frames, anno_fps) == expected


def test_reference_box_is_clamped_to_the_image_on_the_grid() -> None:
    assert prepare.grid_box_text([178.0, 220.0, 456.0, 315.0], 1280, 720) == "[139,306,495,743]"
    assert prepare.grid_box_text([-20.0, 10.0, 200.0, 100.0], 100, 100) == "[0,100,1000,1000]"


def test_image_size_reads_png_and_jpeg_headers(tmp_path: Path) -> None:
    png = tmp_path / "a.png"
    png.write_bytes(png_bytes(33, 21, (0, 0, 1, 1), (255, 0, 0)))
    # SOI, a fill byte, an APP0 segment to skip, a restart marker, then a progressive start-of-frame.
    jpeg = tmp_path / "a.jpg"
    jpeg.write_bytes(
        b"\xff\xd8\xff"
        + b"\xff\xe0"
        + struct.pack(">H", 6)
        + b"JFIF"
        + b"\x00\xff\xd0"
        + b"\xff\xc2"
        + struct.pack(">HBHH", 17, 8, 720, 1280)
        + bytes(12)
    )
    broken = tmp_path / "b.jpg"
    broken.write_bytes(b"\xff\xd8" + bytes(32))

    assert prepare.image_size(png) == (33, 21)
    assert prepare.image_size(jpeg) == (1280, 720)
    with pytest.raises(ValueError, match="image dimensions"):
        prepare.image_size(broken)
    not_an_image = tmp_path / "c.png"
    not_an_image.write_bytes(b"GIF89a")
    with pytest.raises(ValueError, match="image dimensions"):
        prepare.image_size(not_an_image)


def test_prepare_writes_one_row_per_sequence_with_ground_truth_out_of_the_prompt(dataset: Path) -> None:
    split = write_split(dataset, ["seq_b", "seq_a"], num_frames=31, hidden_frames=(20,))
    write_sequence(split, "too_short", num_frames=5)
    (split / "list.txt").write_text("seq_b\n\nseq_a\ntoo_short\n")

    output = prepare.prepare()
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    # list.txt order is kept; the sequence under one second long yields no target and is skipped.
    assert [row["verifier_metadata"]["sequence"] for row in rows] == ["seq_b", "seq_a"]
    row = rows[0]
    content = row["responses_create_params"]["input"][0]["content"]
    assert [part["type"] for part in content] == ["input_text", "input_image"] * 4 + ["input_text"]
    assert content[0]["text"] == (
        "Image 0 is the reference image. The target object is at xyxy bbox [83,156,292,375] on a 0-1000 grid."
    )
    assert [content[i]["text"] for i in (2, 4, 6)] == ["Image 1:", "Image 2:", "Image 3:"]
    assert content[-1]["text"] == (
        "Locate the same object in every requested subsequent image. Return boxes for frames [1,2,3]. "
        "Return only valid JSON inside <answer>...</answer> in this exact shape: "
        '<answer>{"boxes":[{"frame":1,"bbox":[x1,y1,x2,y2]},{"frame":2,"bbox":[x1,y1,x2,y2]},'
        '{"frame":3,"bbox":[x1,y1,x2,y2]}]}</answer>. '
        "Coordinates must be integer values on a 0-1000 grid using xyxy order "
        "(each axis spans 0-1000, independent of pixel size)."
    )
    # Frames are embedded unmodified: reference frame first, then frames 11, 21 and 31.
    for part, frame in zip(content[1::2], (1, 11, 21, 31)):
        header, _, payload = part["image_url"].partition(",")
        assert header == "data:image/png;base64"
        assert base64.b64decode(payload) == (split / "seq_b" / f"{frame:08d}.png").read_bytes()

    metadata = row["verifier_metadata"]
    assert (metadata["image_width"], metadata["image_height"], metadata["anno_fps"]) == (WIDTH, HEIGHT, 10.0)
    assert metadata["targets"] == [
        {"frame": 1, "source_frame": 11, "bbox": [23.0, 15.0, 20.0, 14.0], "visible": 8.0},
        {"frame": 2, "source_frame": 21, "bbox": [38.0, 20.0, 20.0, 14.0], "visible": 0.0},
        {"frame": 3, "source_frame": 31, "bbox": [53.0, 25.0, 20.0, 14.0], "visible": 8.0},
    ]
    # Only the reference box is model-visible; no target box leaks into the prompt text.
    prompt_text = " ".join(part["text"] for part in content if part["type"] == "input_text")
    assert "[240,234,448,453]" not in prompt_text


def test_dataset_root_may_be_the_split_directory_itself(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split = write_split(dataset, ["seq_a"])
    monkeypatch.setenv("GOT10K_DATASET_ROOT", str(split))

    assert prepare.split_root() == split
    assert len(prepare.prepare().read_text().splitlines()) == 1


def test_a_missing_dataset_root_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOT10K_DATASET_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="GOT10K_DATASET_ROOT is not set"):
        prepare.prepare()


def test_incomplete_extractions_are_refused(dataset: Path) -> None:
    split = write_split(dataset, ["ok"])

    def broken(name: str) -> Path:
        return write_sequence(split, name)

    (split / "list.txt").unlink()
    with pytest.raises(FileNotFoundError, match="split list missing"):
        prepare.prepare()

    with pytest.raises(FileNotFoundError, match="sequence directory not found"):
        prepare.load_sequence(split / "absent")

    seq = write_sequence(split, "one_frame", num_frames=1)
    with pytest.raises(ValueError, match="fewer than two frames"):
        prepare.load_sequence(seq)

    seq = broken("missing_annotation")
    (seq / "groundtruth.txt").write_text((seq / "groundtruth.txt").read_text().split("\n", 1)[1])
    with pytest.raises(ValueError, match="31 frames but 30 usable annotations"):
        prepare.load_sequence(seq)

    seq = broken("short_cover")
    (seq / "cover.label").write_text("8\n8\n")
    with pytest.raises(ValueError, match="one finite visibility label per frame"):
        prepare.load_sequence(seq)

    seq = broken("no_rate")
    (seq / "meta_info.ini").write_text("[METAINFO]\nurl: synthetic\nanno_fps: unknown\n")
    with pytest.raises(ValueError, match="no valid anno_fps"):
        prepare.load_sequence(seq)

    seq = broken("no_meta")
    (seq / "meta_info.ini").unlink()
    with pytest.raises(ValueError, match="no valid anno_fps"):
        prepare.load_sequence(seq)

    seq = broken("no_rate_key")
    (seq / "meta_info.ini").write_text("[METAINFO]\nurl: synthetic\n")
    with pytest.raises(ValueError, match="no valid anno_fps"):
        prepare.load_sequence(seq)


def test_committed_example_rows_are_the_synthetic_sequences() -> None:
    from benchmarks.got10k.tests import synthetic

    rows = [json.loads(line) for line in synthetic.EXAMPLE_FPATH.read_text().splitlines()]

    assert [row["verifier_metadata"]["sequence"] for row in rows] == [
        spec["name"] for spec in synthetic.EXAMPLE_SEQUENCES
    ]
