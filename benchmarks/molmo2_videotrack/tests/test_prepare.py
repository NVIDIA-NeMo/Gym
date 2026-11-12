# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.molmo2_videotrack import prepare
from benchmarks.molmo2_videotrack.tests import synthetic
from nemo_gym.environment import load_manifest, resolve_manifest_config_path
from nemo_gym.environment.validation import _mirror_differences, _resolve_manifest_composition


BENCHMARK_DIR = Path(prepare.__file__).parent
MOVER = {"start": (6, 8), "velocity": (1.5, 0.5), "size": (16, 12)}


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A dataset root with one clip per category; `misc` has two expressions over two clips."""
    monkeypatch.setenv("MOLMO2_VIDEOTRACK_ROOT", str(tmp_path))
    monkeypatch.delenv("MOLMO2_VIDEOTRACK_EMBED_FRAMES", raising=False)
    monkeypatch.delenv("MOLMO2_VIDEOTRACK_FRAME_URL_ROOT", raising=False)
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "out" / "molmo2_videotrack_benchmark.jsonl")
    videos_dir = tmp_path / "videos"
    for category, relative in prepare.CATEGORIES.items():
        rows = [synthetic.write_clip(videos_dir, category, f"{category}_clip", f"the {category} box", [MOVER])]
        if category == "misc":
            hidden = {**MOVER, "start": (60, 30), "velocity": (-1.0, 0.0), "hidden": (6, 12)}
            rows.append(
                synthetic.write_clip(videos_dir, category, "two objects", 'both "boxes"', [MOVER, hidden], n_frames=14)
            )
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
    return tmp_path


def test_manifest_mirrors_the_resolved_config() -> None:
    path = BENCHMARK_DIR / "manifest.yaml"
    manifest = load_manifest(path)
    composition = _resolve_manifest_composition(
        resolve_manifest_config_path(path, manifest), dataset_owner=manifest.dataset_owner
    )

    assert _mirror_differences(manifest, composition) == {}


@pytest.mark.parametrize(
    ("n_frames", "fps", "expected"),
    [
        (19, 6.0, [0, 6, 12, 18]),
        (18, 6.0, [0, 6, 12]),
        (61, 29.97, [0, 30, 60]),  # the rate is rounded to whole frames
        (5, 25.0, [0]),
        (3, 0.2, [0, 1, 2]),  # a rate below one frame per second keeps every frame
    ],
)
def test_frames_are_sampled_one_second_apart_from_the_first(n_frames: int, fps: float, expected: list[int]) -> None:
    assert prepare.sampled_frame_indices(n_frames, fps) == expected


def test_prepare_writes_every_expression_in_category_order(dataset: Path) -> None:
    rows = [json.loads(line) for line in prepare.prepare().read_text().splitlines()]

    assert [(row["verifier_metadata"]["category"], row["verifier_metadata"]["clip"]) for row in rows] == [
        ("misc", "misc_clip"),
        ("misc", "two objects"),
        ("animal", "animal_clip"),
        ("dance", "dance_clip"),
        ("person", "person_clip"),
        ("sports", "sports_clip"),
    ]


def test_prompt_shows_frames_by_file_url_and_keeps_ground_truth_out(dataset: Path) -> None:
    row = json.loads(prepare.prepare().read_text().splitlines()[1])
    content = row["responses_create_params"]["input"][0]["content"]

    # 14 frames at 6 fps: local frames 0, 6 and 12.
    assert [part["type"] for part in content] == ["input_text"] + ["input_text", "input_image"] * 3 + ["input_text"]
    assert content[0]["text"] == (
        "The following 3 frames are taken 1 second apart, in temporal order, from a single video.\n"
        'Track every object matching this description: "both "boxes"".\n'
        "There may be one or several matching objects."
    )
    assert [content[i]["text"] for i in (1, 3, 5)] == ["Frame 1:", "Frame 2:", "Frame 3:"]
    # The clip name's space is percent-encoded.
    assert [content[i]["image_url"] for i in (2, 4, 6)] == [
        f"file://{dataset}/videos/misc/two%20objects/{index:05d}.jpg" for index in (0, 6, 12)
    ]
    assert content[-1]["text"] == prepare.INSTRUCTION_TEXT
    assert '<answer>{"tracks":[{"id":1,"frame":1,"xy":[x,y]}]}</answer>' in prepare.INSTRUCTION_TEXT

    metadata = row["verifier_metadata"]
    assert {key: value for key, value in metadata.items() if key != "frames"} == {
        "id": "two objects_0",
        "category": "misc",
        "video": "two objects",
        "clip": "two objects",
        "expression": 'both "boxes"',
        "fps": 6.0,
        "image_width": synthetic.WIDTH,
        "image_height": synthetic.HEIGHT,
    }
    # The second object is hidden in local frames 6 and 12, so only the first is listed there.
    assert [(frame["frame"], frame["source_frame"], len(frame["objects"])) for frame in metadata["frames"]] == [
        (1, 0, 2),
        (2, 6, 1),
        (3, 12, 1),
    ]
    first, second = metadata["frames"][0]["objects"]
    assert first == {"xy": [14.0, 14.0], "mask": synthetic.box_mask((6, 8, 16, 12))}
    assert second["xy"] == [68.0, 36.0]
    prompt_text = " ".join(part["text"] for part in content if part["type"] == "input_text")
    assert "14.0" not in prompt_text and first["mask"]["counts"] not in prompt_text


def test_frame_urls_can_point_at_the_path_the_model_server_sees(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOLMO2_VIDEOTRACK_FRAME_URL_ROOT", "/mnt/frames/")

    row = json.loads(prepare.prepare().read_text().splitlines()[0])

    assert row["responses_create_params"]["input"][0]["content"][2]["image_url"] == (
        "file:///mnt/frames/misc/misc_clip/00000.jpg"
    )


def test_frames_can_be_embedded_instead(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOLMO2_VIDEOTRACK_EMBED_FRAMES", "1")
    jpeg = dataset / "videos" / "misc" / "misc_clip" / "00006.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")

    content = json.loads(prepare.prepare().read_text().splitlines()[0])["responses_create_params"]["input"][0][
        "content"
    ]

    # The synthetic frames are PNG data; anything else is declared as JPEG.
    header, _, payload = content[2]["image_url"].partition(",")
    assert header == "data:image/png;base64"
    assert base64.b64decode(payload) == (dataset / "videos" / "misc" / "misc_clip" / "00000.jpg").read_bytes()
    assert content[4]["image_url"] == "data:image/jpeg;base64," + base64.b64encode(jpeg.read_bytes()).decode()


def test_objects_without_a_point_or_a_mask_are_not_visible() -> None:
    mask = synthetic.box_mask((1, 1, 2, 2))

    def row(points: list, masks: list) -> dict:
        return {"points": [{"points": points}], "masks": [{"masks": masks}]}

    assert len(prepare.visible_objects(row([[3.0, 4.0]], [mask]), 0)) == 1
    assert prepare.visible_objects(row([None], [mask]), 0) == []
    assert prepare.visible_objects(row([[3.0, 4.0]], [None]), 0) == []
    assert prepare.visible_objects(row([[3.0]], [mask]), 0) == []
    assert prepare.visible_objects(row([[float("nan"), 4.0]], [mask]), 0) == []
    assert prepare.visible_objects(row([[3.0, 4.0]], [{"size": [4, 4], "counts": None}]), 0) == []
    # An index past the end of a track counts as not visible.
    assert prepare.visible_objects(row([[3.0, 4.0]], [mask]), 1) == []


def test_a_missing_dataset_root_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MOLMO2_VIDEOTRACK_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="MOLMO2_VIDEOTRACK_ROOT is not set"):
        prepare.prepare()


def test_incomplete_datasets_are_refused(dataset: Path) -> None:
    (dataset / "videos" / "sports" / "sports_clip" / "00012.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="sports/sports_clip is missing 1 of 4 required 1 fps frames"):
        prepare.prepare()

    (dataset / prepare.CATEGORIES["dance"]).unlink()
    with pytest.raises(FileNotFoundError, match="annotations missing for dance"):
        prepare.prepare()


def test_committed_example_rows_are_the_synthetic_clips() -> None:
    rows = [json.loads(line) for line in synthetic.EXAMPLE_FPATH.read_text().splitlines()]

    assert [row["verifier_metadata"]["clip"] for row in rows] == [spec["clip"] for spec in synthetic.EXAMPLE_CLIPS]
    assert all(
        part["image_url"].startswith("data:image/png;base64,")
        for row in rows
        for part in row["responses_create_params"]["input"][0]["content"]
        if part["type"] == "input_image"
    )
