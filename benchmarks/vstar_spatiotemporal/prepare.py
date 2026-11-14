# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert V-STaR (video spatio-temporal reasoning) into Gym benchmark JSONL.

V-STaR (V-STaR-Bench/V-STaR on Hugging Face) is not downloaded automatically.
Point VSTAR_ROOT at a directory holding

    <root>/V_STaR_test.json
    <root>/videos/**/<vid>.mp4

Every annotated item becomes five independent tasks, one per sub-task:

    vqa         "what": free-text answer, rated by an LLM judge
    temporal    "when" (chain 1): time span of the event
    spatial     "where" (chain 1): a box per whole second inside the given span
    spatial_2   "where" (chain 2): boxes with no span given
    temporal_2  "when" (chain 2): time span, given the boxes without times

The video is not sent as a video. Up to 16 frames are decoded with OpenCV, one
per second (evenly spaced over the whole video when it is longer than 16
seconds), resized onto a 28-pixel grid, and sent as images, each preceded by
the caption `Frame i at <t>s: `. Boxes are requested as integers on a 0-1000
grid. Ground truth stays in `verifier_metadata` and never enters the prompt,
except where the sub-task itself hands part of it to the model (the span for
`spatial`, the boxes for `temporal_2`).

Frames are written once as JPEG files to VSTAR_FRAME_DIR (default
`benchmarks/vstar_spatiotemporal/data/frames`) and referenced as file:// URLs,
which requires a model server that reads local files, such as vLLM with
--allowed-local-media-path. Set VSTAR_FRAME_URL_ROOT when the server sees that
directory at a different path. Set VSTAR_EMBED_FRAMES=1 to embed the frames as
data URLs instead (each video's frames are then repeated in its tasks).
"""

import base64
import io
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote


HF_REPO = "V-STaR-Bench/V-STaR"
ANNOTATION_FILE = "V_STaR_test.json"

SUB_TASKS = ("vqa", "temporal", "spatial", "spatial_2", "temporal_2")
SAMPLE_FPS = 1.0
MAX_FRAMES = 16
# Frames are resized so both sides are multiples of this many pixels.
RESIZE_FACTOR = 28
RESIZE_MIN_PIXELS = 4 * RESIZE_FACTOR * RESIZE_FACTOR
RESIZE_MAX_PIXELS = 16384 * RESIZE_FACTOR * RESIZE_FACTOR
JPEG_QUALITY = 95

BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "vstar_spatiotemporal_benchmark.jsonl"
DEFAULT_FRAME_DIR = BENCHMARK_DIR / "data" / "frames"

# Where the frame captions go inside a question; replaced by the captions, or the captions are prepended.
VIDEO_PAD = "<|vision_start|><|video_pad|><|vision_end|>"
IMAGE_TOKEN = "<image>"

SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user provides a video (or image) and asks a question, and the "
    "Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the "
    "user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> "
    "tags, respectively. All reasoning must be grounded in visual evidence. Name each relevant object once as "
    '<obj>name</obj> and give it an integer id; afterwards refer to it as "id k". Give every time as <t>seconds</t>s. '
    "Give locations as <point>[x,y]</point> or <box>[x1,y1,x2,y2]</box> with integer coordinates on a 0-1000 grid. "
    "The answer must follow exactly the format requested in the question."
)

# The format clause that ends each question.
CLAUSES = {
    "what": "Answer in one or two sentences.",
    "when": 'Answer in the form "From <t>start</t>s to <t>end</t>s" using the times of the sampled frames.',
    "where_box": 'Return only valid JSON in this exact shape: {"tracks":[{"id":1,"t":time_in_seconds,"box":[x1,y1,x2,y2]}]} '
    "with one entry for every whole second inside the window (t = 0, 1, 2, ...), inclusive of both ends; "
    "coordinates are integers on a 0-1000 grid.",
    "where_nowindow": 'Return only valid JSON in this exact shape: {"from":start_seconds,"to":end_seconds,"tracks":'
    '[{"id":1,"t":time_in_seconds,"box":[x1,y1,x2,y2]}]} with one entry for every whole second inside '
    "the window (t = start, start+1, ..., end); coordinates are integers on a 0-1000 grid.",
}


def dataset_root() -> Path:
    root = os.environ.get("VSTAR_ROOT")
    if not root:
        raise RuntimeError(
            f"VSTAR_ROOT is not set. {HF_REPO} is not downloaded automatically; set it to a directory holding "
            f"{ANNOTATION_FILE} and videos/**/<vid>.mp4."
        )
    return Path(root)


def frame_dir() -> Path:
    return Path(os.environ.get("VSTAR_FRAME_DIR") or DEFAULT_FRAME_DIR)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def sampled_frame_indices(total_frames: int, video_fps: float) -> list[int]:
    """Indices of the frames shown to the model: one per second, or MAX_FRAMES evenly spaced for a longer video."""
    import numpy as np

    if int(total_frames / video_fps * SAMPLE_FPS) > MAX_FRAMES:
        return [int(index) for index in np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int)]
    return list(range(0, total_frames, int(video_fps / SAMPLE_FPS)))[:MAX_FRAMES]


def resized_size(width: int, height: int) -> tuple[int, int]:
    """(width, height) on the RESIZE_FACTOR grid, within the pixel limits, keeping the aspect ratio as far as possible."""
    factor = RESIZE_FACTOR
    new_height = max(factor, round(height / factor) * factor)
    new_width = max(factor, round(width / factor) * factor)
    if new_height * new_width > RESIZE_MAX_PIXELS:
        beta = math.sqrt((height * width) / RESIZE_MAX_PIXELS)
        new_height = math.floor(height / beta / factor) * factor
        new_width = math.floor(width / beta / factor) * factor
    elif new_height * new_width < RESIZE_MIN_PIXELS:
        beta = math.sqrt(RESIZE_MIN_PIXELS / (height * width))
        new_height = math.ceil(height * beta / factor) * factor
        new_width = math.ceil(width * beta / factor) * factor
    return new_width, new_height


def extract_frames(video: Path) -> tuple[list[bytes], list[float]]:
    """(JPEG bytes, time in seconds) of each sampled frame. A frame that fails to decode is left out."""
    try:
        import cv2
    except ImportError as error:
        # Gym's dependency overrides exclude OpenCV, so `prepare_dependencies` cannot install it.
        raise RuntimeError(
            "V-STaR preparation decodes the videos with OpenCV, which Gym does not install. Install "
            "opencv-python-headless into this environment yourself (from outside the Gym checkout, whose uv "
            "overrides drop it), then run the preparation again."
        ) from error
    from PIL import Image

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[bytes] = []
    times: list[float] = []
    for index in sampled_frame_indices(total_frames, video_fps):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize(resized_size(*image.size))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        frames.append(buffer.getvalue())
        times.append(float(index / video_fps))
    capture.release()
    if not frames:
        raise ValueError(f"No frames extracted from video: {video}")
    return frames, times


def video_frames(vid: str, video: Path, frames_root: Path) -> tuple[list[Path], list[float]]:
    """Paths and times of one video's frames, extracting them on first use."""
    directory = frames_root / vid
    meta_file = directory / "frames.json"
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        paths = [directory / name for name in meta["files"]]
        if all(path.is_file() for path in paths):
            return paths, meta["times"]
    frames, times = extract_frames(video)
    directory.mkdir(parents=True, exist_ok=True)
    names = [f"{number:02d}.jpg" for number in range(1, len(frames) + 1)]
    for name, data in zip(names, frames):
        (directory / name).write_bytes(data)
    # Written last, so an interrupted extraction is redone.
    meta_file.write_text(json.dumps({"files": names, "times": times}), encoding="utf-8")
    return [directory / name for name in names], times


def frame_url(frame: Path, frames_root: Path) -> str:
    """The image_url for one frame: a data URL, or a file:// URL as the model server sees it."""
    if os.environ.get("VSTAR_EMBED_FRAMES", "0") == "1":
        data = frame.read_bytes()
        mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(data).decode()
    url_root = os.environ.get("VSTAR_FRAME_URL_ROOT") or str(frames_root.absolute())
    return "file://" + quote(f"{url_root.rstrip('/')}/{frame.relative_to(frames_root).as_posix()}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def frame_captions(times: list[float]) -> str:
    return "".join(f"Frame {number} at {round(float(t), 1)}s: {IMAGE_TOKEN}\n" for number, t in enumerate(times, 1))


def format_seconds(value: float) -> str:
    value = round(float(value), 1)
    return str(int(value)) if value == int(value) else f"{value:.1f}"


def ground_truth_boxes(item: dict[str, Any]) -> list[list[float]]:
    return [[box["xmin"], box["ymin"], box["xmax"], box["ymax"]] for box in item["bboxes"]]


def question_text(item: dict[str, Any], sub_task: str) -> str:
    """The question for one sub-task, with VIDEO_PAD where the frame captions belong (if anywhere)."""
    boxes = ground_truth_boxes(item)
    if sub_task == "vqa":
        return (
            f"Answer the question about the video: {item['question']}\n(If the answer is a person, you don't need to "
            f"identify the person.)\n{CLAUSES['what']}"
        )
    if sub_task == "temporal":
        video_length = round(item["frame_count"] / item["fps"], 2)
        return (
            f"This video is {video_length} seconds long. {VIDEO_PAD}\nAnswer the question about the video: "
            f"{item['temporal_question']}\n{CLAUSES['when']}"
        )
    if sub_task == "temporal_2":
        video_length = round(item["frame_count"] / item["fps"], 2)
        width, height = item["width"], item["height"]
        grid = [
            [
                int(round(box[0] / width * 1000)),
                int(round(box[1] / height * 1000)),
                int(round(box[2] / width * 1000)),
                int(round(box[3] / height * 1000)),
            ]
            for box in boxes
        ]
        grid_text = ", ".join("<box>[" + ",".join(str(value) for value in box) + "]</box>" for box in grid)
        return (
            f"This video is {video_length} seconds long. {VIDEO_PAD}\nAnswer the question about the video: "
            f"{item['temporal_question']}\nThere are {len(boxes)} boxes of the key object related to the question "
            f"(0-1000 grid, time unknown): {grid_text}.\n{CLAUSES['when']}"
        )
    if sub_task == "spatial":
        start, end = item["timestamps"][0], item["timestamps"][1]
        return (
            f"{VIDEO_PAD}\nAnswer the question about the video: {item['spatial_question']} The event occurs from "
            f"{format_seconds(start)}s to {format_seconds(end)}s; give the location of the object at every whole "
            f"second inside that window.\n{CLAUSES['where_box']}"
        )
    if sub_task == "spatial_2":
        return (
            f"{VIDEO_PAD}\nAnswer the question about the video: {item['spatial_question_2']} Find when the described "
            f"moment happens (about {len(boxes)} second(s)), then give the location of the object at every whole "
            f"second inside that window.\n{CLAUSES['where_nowindow']}"
        )
    raise ValueError(f"unknown sub-task {sub_task}")


def user_content(prompt: str, image_urls: list[str]) -> list[dict[str, Any]]:
    """Interleave the images at their markers; if the markers do not match the images, send the images first."""

    def image(url: str) -> dict[str, Any]:
        return {"type": "input_image", "detail": "auto", "image_url": url}

    def text(value: str) -> dict[str, Any]:
        return {"type": "input_text", "text": value}

    if image_urls and prompt.count(IMAGE_TOKEN) == len(image_urls):
        chunks = prompt.split(IMAGE_TOKEN)
        content: list[dict[str, Any]] = []
        for chunk, url in zip(chunks, image_urls):
            if chunk.strip():
                content.append(text(chunk))
            content.append(image(url))
        if chunks[-1].strip():
            content.append(text(chunks[-1]))
        return content
    return [image(url) for url in image_urls] + [text(prompt.replace(IMAGE_TOKEN, ""))]


def convert(
    item: dict[str, Any], index: int, sub_task: str, frames: list[Path], times: list[float], frames_root: Path
) -> dict[str, Any]:
    """One (item, sub-task) pair -> one benchmark row."""
    captions = frame_captions(times)
    prompt = question_text(item, sub_task)
    prompt = prompt.replace(VIDEO_PAD, captions) if VIDEO_PAD in prompt else captions + prompt
    urls = [frame_url(frame, frames_root) for frame in frames]
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content(prompt, urls)},
            ]
        },
        "verifier_metadata": {
            "item_index": index,
            "sub_task": sub_task,
            "vid": str(item["vid"]),
            "domain": item.get("domain"),
            "fps": float(item["fps"]),
            "frame_count": int(item["frame_count"]),
            "width": int(item["width"]),
            "height": int(item["height"]),
            "question": item["question"],
            "answer": item["answer"],
            "timestamps": item["timestamps"],
            "bboxes": item["bboxes"],
            "frame_times": times,
        },
    }


def find_videos(videos_dir: Path) -> dict[str, Path]:
    """vid -> path for every .mp4 under the videos directory (first one wins when a name repeats)."""
    found: dict[str, Path] = {}
    for directory, _, files in os.walk(videos_dir):
        for name in files:
            if name.endswith(".mp4"):
                found.setdefault(name[:-4], Path(directory) / name)
    return found


def prepare() -> Path:
    root = dataset_root()
    annotation_file = root / ANNOTATION_FILE
    videos_dir = root / "videos"
    if not annotation_file.is_file():
        raise FileNotFoundError(f"V-STaR annotations missing: {annotation_file}")
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"V-STaR videos missing: {videos_dir}")
    items = json.loads(annotation_file.read_text(encoding="utf-8"))
    videos = find_videos(videos_dir)
    missing = sorted({str(item["vid"]) for item in items} - set(videos))
    if missing:
        raise FileNotFoundError(
            f"V-STaR: {len(missing)} annotated videos are not under {videos_dir} (first: {missing[0]}.mp4). "
            "Refusing a partial evaluation."
        )

    frames_root = frame_dir()
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, tuple[list[Path], list[float]]] = {}
    written = 0
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as out:
        for index, item in enumerate(items):
            vid = str(item["vid"])
            if vid not in cache:
                cache[vid] = video_frames(vid, videos[vid], frames_root)
            frames, times = cache[vid]
            for sub_task in SUB_TASKS:
                out.write(json.dumps(convert(item, index, sub_task, frames, times, frames_root)) + "\n")
                written += 1
    print(
        f"Wrote {written} tasks ({len(items)} items x {len(SUB_TASKS)} sub-tasks, {len(cache)} videos) to {OUTPUT_FPATH}"
    )
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
