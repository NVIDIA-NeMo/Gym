# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Molmo2-VideoTrackEval verifier: text-referred point tracking, scored against masks.

Each task is one referring expression over one video clip, which may match
several objects. The prompt shows the clip's frames sampled 1 second apart
("Frame 1" .. "Frame N") and asks for the location of every matching object
that is visible in each frame, on a 0-1000 grid:

    <answer>{"tracks":[{"id":1,"frame":1,"xy":[x,y]},...]}</answer>

An object that is not visible in a frame is reported by leaving it out.

Scoring follows the Molmo2 paper's protocol (F1 of the points at 1 fps) and
needs no judge. For each frame:

  1. Predicted points are scaled to pixels and matched one-to-one to the
     ground-truth points of the visible objects, minimizing total distance.
  2. A matched prediction is correct when it lands inside its object's
     segmentation mask. Unmatched predictions and unmatched objects are errors.
  3. precision = correct / predicted, recall = correct / visible objects,
     F1 = their harmonic mean. A frame with predictions but no visible object,
     or visible objects but no prediction, scores 0. A frame with neither is
     skipped.

The per-task reward is the F1 averaged over the task's scored frames (1.0 when
no frame is scored: nothing to find and nothing reported). Ids are not scored;
matching is redone independently in every frame. `compute_metrics` reports the
mean precision, recall and F1 over tasks, overall and per category.
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Iterator, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linear_sum_assignment

from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture


GRID = 1000.0
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)

# Molmo2's own tracking markup, which that model emits instead of the requested JSON.
# Coordinates are already on the 0-1000 grid and the frame number precedes the colon:
#   <tracks coords="1:1 1 502 347;2:1 1 498 350">...</tracks>   frame:id visible x y
#   <tracks coords="1: [000,694];2: [010,690]">...</tracks>     frame: [x,y]
#   <points coords="1 1 502 347, 2 1 498 350">...</points>      frame id x y
NATIVE_COORDS_PATTERN = re.compile(r'<(?:points|tracks)[^>]*? coords="([^"]*)"')
# Tried in this order: a triple match on "id visible x y" would read (visible, x, y).
NATIVE_ID_VISIBLE_XY_PATTERN = re.compile(r"([0-9]+) ([01]) ([0-9]{3,4}) ([0-9]{3,4})")
NATIVE_ID_XY_PATTERN = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")
NATIVE_BRACKET_PATTERN = re.compile(r"\[\s*([0-9]{1,4})[\s,]+([0-9]{1,4})\s*\]")
NATIVE_FRAME_GROUP_PATTERN = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")

Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Molmo2Object(BaseModel):
    """One matching object that is visible in a frame."""

    # Ground-truth point in pixels, used only to match predictions to objects.
    xy: list[float] = Field(min_length=2, max_length=2)
    # COCO RLE segmentation mask: {"size": [height, width], "counts": "..."}.
    mask: dict[str, Any]


class Molmo2Frame(BaseModel):
    # Prompt-local frame number (1..N), the number the model is asked to answer with.
    frame: int = Field(ge=1)
    # 0-based frame index inside the source clip (bookkeeping only).
    source_frame: Optional[int] = None
    # Matching objects visible in this frame; empty when none is.
    objects: list[Molmo2Object] = []


class Molmo2VerifierMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    category: Optional[str] = None
    image_width: float = Field(gt=0)
    image_height: float = Field(gt=0)
    frames: list[Molmo2Frame] = Field(min_length=1)


class Molmo2FrameScore(BaseModel):
    frame: int
    predicted: int
    visible_objects: int
    correct: int
    precision: float
    recall: float
    f1: float


class Molmo2VideoTrackResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class Molmo2VideoTrackVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Molmo2VerifierMetadata


class Molmo2VideoTrackVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    category: Optional[str] = None
    precision: float
    recall: float
    f1: float
    # False when no point could be read from the answer at all.
    parse_ok: bool
    num_scored_frames: int
    frame_scores: list[Molmo2FrameScore]


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def _iter_json_objects(text: str) -> Iterator[Any]:
    """Yield every JSON object that can be decoded starting at a '{' in text."""
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        start = text.find("{", pos)
        if start == -1:
            return
        try:
            value, length = decoder.raw_decode(text[start:])
        except (ValueError, RecursionError):
            pos = start + 1
            continue
        yield value
        pos = start + length


def _load_payload(payload: str) -> Any:
    """The payload as JSON, or else the first JSON object embedded in it."""
    stripped = payload.strip()
    if stripped:
        try:
            return json.loads(stripped)
        except (ValueError, RecursionError):
            pass
    return next(_iter_json_objects(payload), None)


def _parse_native_markup(text: str) -> dict[int, list[Point]]:
    """Visible points per frame from Molmo2's <tracks>/<points> markup."""
    out: dict[int, list[Point]] = {}

    def add_points(segment: str, frame: int) -> None:
        points = [
            (float(m.group(3)), float(m.group(4)))
            for m in NATIVE_ID_VISIBLE_XY_PATTERN.finditer(segment)
            if m.group(2) == "1"
        ]
        if not points and not NATIVE_ID_VISIBLE_XY_PATTERN.search(segment):
            points = [(float(m.group(2)), float(m.group(3))) for m in NATIVE_ID_XY_PATTERN.finditer(segment)]
            if not points:
                points = [(float(m.group(1)), float(m.group(2))) for m in NATIVE_BRACKET_PATTERN.finditer(segment)]
        if points:
            out.setdefault(frame, []).extend(points)

    def frame_number(text: str) -> Optional[int]:
        try:
            return int(float(text.strip()))
        except ValueError:
            return None

    for coords in NATIVE_COORDS_PATTERN.finditer(text):
        payload = coords.group(1)
        if ":" in payload:
            for segment in re.split(r"[;\t]", payload):
                head, colon, rest = segment.partition(":")
                frame = frame_number(head) if colon else None
                if frame is not None:
                    add_points(rest, frame)
        else:
            for group in NATIVE_FRAME_GROUP_PATTERN.finditer(payload):
                frame = frame_number(group.group(1))
                if frame is not None:
                    add_points(group.group(2), frame)
    return out


def parse_tracks(text: str) -> dict[int, list[Point]]:
    """Recover {frame number: [(x, y), ...]} on the 0-1000 grid from the model's answer.

    Each <answer>...</answer> payload is tried in turn, then the whole text; the
    first that yields a point wins. A payload is read as JSON, or as the first
    JSON object found inside it, and may be {"tracks": [...]}, {"points": [...]}
    or a bare list. Entries need a "frame" and an "xy" (or "point") pair; ids
    are ignored. If nothing is found, Molmo2's own markup is tried.
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]

    out: dict[int, list[Point]] = {}
    for payload in [*ANSWER_PATTERN.findall(text), text]:
        loaded = _load_payload(payload)
        if isinstance(loaded, dict):
            entries = loaded.get("tracks") or loaded.get("points")
        else:
            entries = loaded
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            frame = entry.get("frame")
            xy = entry.get("xy") or entry.get("point")
            if frame is None or not isinstance(xy, (list, tuple)) or len(xy) < 2:
                continue
            try:
                frame = int(frame)
            except (TypeError, ValueError, OverflowError):
                continue
            # The frame is recorded even if its point turns out to be unusable, so an answer in
            # the right shape still counts as parsed and stops the search through later payloads.
            points = out.setdefault(frame, [])
            try:
                point = (float(xy[0]), float(xy[1]))
            except (TypeError, ValueError):
                continue
            # A non-finite coordinate cannot be matched or looked up in a mask.
            if math.isfinite(point[0]) and math.isfinite(point[1]):
                points.append(point)
        if out:
            return out
    return _parse_native_markup(text)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def rle_runs(counts: Any) -> list[int]:
    """Run lengths of a COCO RLE mask, from the compressed string or a plain list.

    The string packs each run as a base-48 varint; from the third run on, a run
    is stored as its difference from the run two places before it.
    """
    if not isinstance(counts, (str, bytes)):
        return [int(run) for run in counts]
    if isinstance(counts, bytes):
        counts = counts.decode()
    runs: list[int] = []
    pos = 0
    while pos < len(counts):
        value, shift, more = 0, 0, True
        while more:
            char = ord(counts[pos]) - 48
            value |= (char & 0x1F) << shift
            more = bool(char & 0x20)
            pos += 1
            shift += 5
            if not more and (char & 0x10):
                value |= -1 << shift
        if len(runs) > 2:
            value += runs[-2]
        runs.append(value)
    return runs


def point_in_mask(mask: dict[str, Any], x: float, y: float) -> bool:
    """Is the pixel nearest to (x, y) set in the mask? Points outside the image are not.

    Runs alternate background/foreground, starting with background, over the
    pixels in column-major order (index = column * height + row).
    """
    height, width = (int(value) for value in mask["size"])
    column, row = int(round(x)), int(round(y))
    if not (0 <= row < height and 0 <= column < width):
        return False
    index = column * height + row
    end = 0
    for run_index, run in enumerate(rle_runs(mask["counts"])):
        end += run
        if index < end:
            return run_index % 2 == 1
    return False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_frame(predicted: list[Point], objects: list[Molmo2Object]) -> Optional[tuple[int, float, float, float]]:
    """(correct, precision, recall, f1) for one frame, or None when the frame is skipped.

    `predicted` is in pixels.
    """
    if not predicted and not objects:
        return None
    if not predicted or not objects:
        return 0, 0.0, 0.0, 0.0
    targets = np.asarray([obj.xy for obj in objects], dtype=float)
    distances = np.linalg.norm(np.asarray(predicted, dtype=float)[:, None, :] - targets[None, :, :], axis=2)
    correct = sum(
        point_in_mask(objects[target_index].mask, *predicted[predicted_index])
        for predicted_index, target_index in zip(*linear_sum_assignment(distances))
    )
    precision = correct / len(predicted)
    recall = correct / len(objects)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return correct, precision, recall, f1


def score_expression(text: str, metadata: Molmo2VerifierMetadata) -> dict[str, Any]:
    width, height = float(metadata.image_width), float(metadata.image_height)
    predictions = parse_tracks(text)

    frame_scores: list[Molmo2FrameScore] = []
    for frame in metadata.frames:
        predicted = [(x * width / GRID, y * height / GRID) for x, y in predictions.get(frame.frame, [])]
        scored = score_frame(predicted, frame.objects)
        if scored is None:
            continue
        correct, precision, recall, f1 = scored
        frame_scores.append(
            Molmo2FrameScore(
                frame=frame.frame,
                predicted=len(predicted),
                visible_objects=len(frame.objects),
                correct=correct,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    if frame_scores:
        precision = float(np.mean([score.precision for score in frame_scores]))
        recall = float(np.mean([score.recall for score in frame_scores]))
        f1 = float(np.mean([score.f1 for score in frame_scores]))
    else:
        # Nothing to find in any frame and nothing reported.
        precision = recall = f1 = 1.0
    return {
        "reward": f1,
        "category": metadata.category,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "parse_ok": bool(predictions),
        "num_scored_frames": len(frame_scores),
        "frame_scores": frame_scores,
    }


def aggregate_metrics(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean precision/recall/F1 over rollouts, overall and per category."""
    if not rollouts:
        return {}

    def means(group: list[dict[str, Any]], prefix: str) -> dict[str, float]:
        return {
            f"{prefix}{name}": sum(rollout.get(name) or 0.0 for rollout in group) / len(group)
            for name in ("f1", "precision", "recall")
        }

    metrics: dict[str, Any] = means(rollouts, "")
    metrics["parse_fail_rate"] = sum(1 for rollout in rollouts if not rollout.get("parse_ok")) / len(rollouts)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rollout in rollouts:
        if rollout.get("category"):
            by_category[rollout["category"]].append(rollout)
    for category in sorted(by_category):
        metrics.update(means(by_category[category], f"{category}/"))
        metrics[f"{category}/samples"] = len(by_category[category])
    return metrics


def _assistant_text(body: BaseVerifyRequest) -> str:
    """Text of the assistant's message output (reasoning items are not read)."""
    texts: list[str] = []
    for output in body.response.output:
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        texts.extend(text for item in output.content if isinstance(text := getattr(item, "text", None), str))
    return "\n".join(texts)


class Molmo2VideoTrackVerifier:
    async def verify(self, body: Molmo2VideoTrackVerifyRequest) -> Molmo2VideoTrackVerifyResponse:
        result = score_expression(_assistant_text(body), body.verifier_metadata)
        return Molmo2VideoTrackVerifyResponse(**body.model_dump(), **result)


class Molmo2VideoTrackResourcesServer(Molmo2VideoTrackVerifier, SimpleResourcesServer):
    config: Molmo2VideoTrackResourcesServerConfig

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return aggregate_metrics([rollout for task in tasks for rollout in task])

    def compute_repeat_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return self.compute_metrics(tasks)

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            name: value
            for name, value in agent_metrics.items()
            if name in ("f1", "precision", "recall", "parse_fail_rate") or name.endswith("/f1")
        }


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=Molmo2VideoTrackVerifier,
    request_model=Molmo2VideoTrackVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    Molmo2VideoTrackResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = Molmo2VideoTrackResourcesServer.run_webserver()  # noqa: F401
