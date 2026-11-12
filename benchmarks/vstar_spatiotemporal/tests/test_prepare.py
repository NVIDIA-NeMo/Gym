# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from pathlib import Path

import pytest

from benchmarks.vstar_spatiotemporal import prepare
from benchmarks.vstar_spatiotemporal.tests import synthetic
from nemo_gym.environment import load_manifest, resolve_manifest_config_path
from nemo_gym.environment.validation import _mirror_differences, _resolve_manifest_composition


BENCHMARK_DIR = Path(prepare.__file__).parent
TIMES = [0.0, 1.0, 2.0, 3.0]


@pytest.fixture
def decoded(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replaces the video decoder with a stand-in; lists the videos it was asked to decode."""
    names: list[str] = []

    def extract_frames(video: Path) -> tuple[list[bytes], list[float]]:
        names.append(video.name)
        return synthetic.frame_pngs(TIMES), TIMES

    monkeypatch.setattr(prepare, "extract_frames", extract_frames)
    return names


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decoded: list[str]) -> Path:
    """A dataset root with two items on one video and one on another."""
    monkeypatch.setenv("VSTAR_ROOT", str(tmp_path))
    for name in ("VSTAR_FRAME_DIR", "VSTAR_FRAME_URL_ROOT", "VSTAR_EMBED_FRAMES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "out" / "vstar_spatiotemporal_benchmark.jsonl")
    monkeypatch.setattr(prepare, "DEFAULT_FRAME_DIR", tmp_path / "out" / "frames")
    items = [
        synthetic.make_item("clip one"),
        synthetic.make_item("clip one", span=(0.0, 2.0), color="green"),
        synthetic.make_item("clip&two", domain="Sports"),
    ]
    (tmp_path / prepare.ANNOTATION_FILE).write_text(json.dumps(items))
    (tmp_path / "videos" / "nested").mkdir(parents=True)
    (tmp_path / "videos" / "nested" / "clip one.mp4").write_bytes(b"")
    (tmp_path / "videos" / "clip&two.mp4").write_bytes(b"")
    return tmp_path


def _rows() -> list[dict]:
    return [json.loads(line) for line in prepare.prepare().read_text().splitlines()]


def test_manifest_mirrors_the_resolved_config() -> None:
    path = BENCHMARK_DIR / "manifest.yaml"
    manifest = load_manifest(path)
    composition = _resolve_manifest_composition(
        resolve_manifest_config_path(path, manifest), dataset_owner=manifest.dataset_owner
    )

    assert _mirror_differences(manifest, composition) == {}


@pytest.mark.parametrize(
    ("total_frames", "fps", "expected"),
    [
        (150, 30.0, [0, 30, 60, 90, 120]),  # 5 seconds: one frame per second
        (91, 29.97, [0, 29, 58, 87]),  # the step is the frame rate rounded down
        (480, 30.0, list(range(0, 480, 30))),  # exactly 16 seconds
        (510, 30.0, [0, 33, 67, 101, 135, 169, 203, 237, 271, 305, 339, 373, 407, 441, 475, 509]),  # 17 s: spread out
        (20, 30.0, [0]),
    ],
)
def test_frames_are_one_second_apart_or_sixteen_spread_over_a_longer_video(
    total_frames: int, fps: float, expected: list[int]
) -> None:
    assert prepare.sampled_frame_indices(total_frames, fps) == expected


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((640, 360), (644, 364)),
        ((1920, 1080), (1932, 1092)),
        ((28, 28), (56, 56)),  # below the minimum area: scaled up
        ((8000, 4000), (5068, 2520)),  # above the maximum area: scaled down
    ],
)
def test_frames_are_resized_onto_the_28_pixel_grid(size: tuple[int, int], expected: tuple[int, int]) -> None:
    assert prepare.resized_size(*size) == expected


def test_every_item_becomes_five_tasks_in_annotation_order(dataset: Path, decoded: list[str]) -> None:
    rows = _rows()

    assert [(row["verifier_metadata"]["item_index"], row["verifier_metadata"]["sub_task"]) for row in rows] == [
        (index, sub_task) for index in range(3) for sub_task in prepare.SUB_TASKS
    ]
    assert [row["verifier_metadata"]["vid"] for row in rows[::5]] == ["clip one", "clip one", "clip&two"]
    # A video shared by two items is decoded once.
    assert decoded == ["clip one.mp4", "clip&two.mp4"]


def test_frames_are_interleaved_with_their_captions_and_ground_truth_stays_out(dataset: Path) -> None:
    row = _rows()[0]
    system, user = row["responses_create_params"]["input"]

    assert system == {"role": "system", "content": prepare.SYSTEM_PROMPT}
    content = user["content"]
    assert [part["type"] for part in content] == ["input_text", "input_image"] * 4 + ["input_text"]
    assert [content[i]["text"] for i in (0, 2, 4, 6)] == [
        "Frame 1 at 0.0s: ",
        "\nFrame 2 at 1.0s: ",
        "\nFrame 3 at 2.0s: ",
        "\nFrame 4 at 3.0s: ",
    ]
    # The space in the video name is percent-encoded.
    assert [content[i]["image_url"] for i in (1, 3, 5, 7)] == [
        f"file://{dataset}/out/frames/clip%20one/{number:02d}.jpg" for number in (1, 2, 3, 4)
    ]
    assert content[-1]["text"] == (
        "\nAnswer the question about the video: what moves across the orange frame?\n"
        "(If the answer is a person, you don't need to identify the person.)\nAnswer in one or two sentences."
    )

    metadata = row["verifier_metadata"]
    assert metadata == {
        "item_index": 0,
        "sub_task": "vqa",
        "vid": "clip one",
        "domain": "Synthetic",
        "fps": 10.0,
        "frame_count": 50,
        "width": synthetic.WIDTH,
        "height": synthetic.HEIGHT,
        "question": "what moves across the orange frame?",
        "answer": "a orange rectangle",
        "timestamps": [1.2, 3.6],
        "bboxes": [
            {"xmin": 32, "ymin": 22, "xmax": 52, "ymax": 36, "timestamp": 2},
            {"xmin": 44, "ymin": 28, "xmax": 64, "ymax": 42, "timestamp": 3},
        ],
        "frame_times": TIMES,
    }
    prompt_text = " ".join(part["text"] for part in content if part["type"] == "input_text")
    assert "a orange rectangle" not in prompt_text and "1.2" not in prompt_text


def test_each_sub_task_asks_its_own_question(dataset: Path) -> None:
    rows = {row["verifier_metadata"]["sub_task"]: row for row in _rows()[:5]}

    def texts(sub_task: str) -> tuple[str, str]:
        content = rows[sub_task]["responses_create_params"]["input"][1]["content"]
        return content[0]["text"], content[-1]["text"]

    # The frame captions come after the video length where the question states it.
    first, last = texts("temporal")
    assert first == "This video is 5.0 seconds long. Frame 1 at 0.0s: "
    assert last == (
        "\n\nAnswer the question about the video: When does the orange rectangle cross the middle?\n"
        'Answer in the form "From <t>start</t>s to <t>end</t>s" using the times of the sampled frames.'
    )
    # Chain 1 "where" is told the span; chain 2 only how many seconds it lasts.
    first, last = texts("spatial")
    assert first == "Frame 1 at 0.0s: "
    assert "The event occurs from 1.2s to 3.6s;" in last and last.endswith(prepare.CLAUSES["where_box"])
    _, last = texts("spatial_2")
    assert "(about 2 second(s))" in last and "1.2" not in last and last.endswith(prepare.CLAUSES["where_nowindow"])
    # Chain 2 "when" is given the boxes on the 0-1000 grid, without their times.
    _, last = texts("temporal_2")
    assert (
        "There are 2 boxes of the key object related to the question (0-1000 grid, time unknown): "
        "<box>[333,344,542,562]</box>, <box>[458,438,667,656]</box>.\n"
    ) in last
    assert "1.2" not in last and "3.6" not in last


def test_whole_second_spans_are_written_without_decimals() -> None:
    item = synthetic.make_item("clip", span=(2.0, 4.04))

    assert "The event occurs from 2s to 4s;" in prepare.question_text(item, "spatial")
    with pytest.raises(ValueError, match="unknown sub-task"):
        prepare.question_text(item, "counting")


def test_images_go_first_when_the_markers_do_not_match_them() -> None:
    # The question itself contains an image marker, so the markers no longer line up with the frames.
    prompt = "Frame 1 at 0.0s: <image>\nWhat does <image> mean here?"

    content = prepare.user_content(prompt, ["file:///a.jpg"])

    assert content == [
        {"type": "input_image", "detail": "auto", "image_url": "file:///a.jpg"},
        {"type": "input_text", "text": "Frame 1 at 0.0s: \nWhat does  mean here?"},
    ]


def test_extracted_frames_are_reused(dataset: Path, decoded: list[str]) -> None:
    prepare.prepare()
    prepare.prepare()
    assert decoded == ["clip one.mp4", "clip&two.mp4"]

    # A missing frame file means the video is decoded again.
    (dataset / "out" / "frames" / "clip&two" / "02.jpg").unlink()
    prepare.prepare()
    assert decoded == ["clip one.mp4", "clip&two.mp4", "clip&two.mp4"]


def test_frame_urls_can_point_at_the_path_the_model_server_sees(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VSTAR_FRAME_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("VSTAR_FRAME_URL_ROOT", "/mnt/frames/")

    row = _rows()[-1]

    assert (tmp_path / "elsewhere" / "clip&two" / "01.jpg").is_file()
    assert row["responses_create_params"]["input"][1]["content"][1]["image_url"] == (
        "file:///mnt/frames/clip%26two/01.jpg"
    )


def test_frames_can_be_embedded_instead(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VSTAR_EMBED_FRAMES", "1")

    content = _rows()[0]["responses_create_params"]["input"][1]["content"]

    # The stand-in frames are PNG data; real frames are JPEG.
    header, _, payload = content[1]["image_url"].partition(",")
    assert header == "data:image/png;base64"
    assert base64.b64decode(payload) == synthetic.frame_pngs(TIMES)[0]


def test_a_missing_dataset_is_explained(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (dataset / "videos" / "clip&two.mp4").unlink()
    with pytest.raises(FileNotFoundError, match=r"1 annotated videos are not under .* \(first: clip&two.mp4\)"):
        prepare.prepare()

    (dataset / prepare.ANNOTATION_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="annotations missing"):
        prepare.prepare()

    monkeypatch.delenv("VSTAR_ROOT")
    with pytest.raises(RuntimeError, match="VSTAR_ROOT is not set"):
        prepare.prepare()


def test_videos_are_decoded_one_frame_per_second(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    import io

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for index in range(35):
        # The brightness of each frame encodes its index.
        writer.write(np.full((48, 64, 3), index * 7, dtype=np.uint8))
    writer.release()

    frames, times = prepare.extract_frames(path)

    assert times == [0.0, 1.0, 2.0, 3.0]
    images = [image_module.open(io.BytesIO(frame)) for frame in frames]
    # 64x48 is resized onto the 28-pixel grid.
    assert {image.size for image in images} == {(56, 56)}
    assert [round(np.asarray(image).mean() / 7 / 10) for image in images] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="Could not open video"):
        prepare.extract_frames(tmp_path / "absent.mp4")


def test_a_missing_opencv_is_explained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    # A None entry makes `import cv2` raise ImportError whether or not OpenCV is installed.
    monkeypatch.setitem(sys.modules, "cv2", None)

    with pytest.raises(RuntimeError, match="Install opencv-python-headless"):
        prepare.extract_frames(tmp_path / "clip.mp4")


def test_committed_example_rows_are_the_synthetic_item() -> None:
    rows = [json.loads(line) for line in synthetic.EXAMPLE_FPATH.read_text().splitlines()]

    assert [row["verifier_metadata"]["sub_task"] for row in rows] == list(prepare.SUB_TASKS)
    assert {row["verifier_metadata"]["vid"] for row in rows} == {synthetic.EXAMPLE_ITEM["vid"]}
    assert all(
        part["image_url"].startswith("data:image/png;base64,")
        for row in rows
        for part in row["responses_create_params"]["input"][1]["content"]
        if part["type"] == "input_image"
    )
