# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GOT-10k single-object tracking verifier (multi-image, 0-1000 grid protocol).

Each task is one GOT-10k sequence. The prompt shows the first frame ("Image 0")
with the target's box on a 0-1000 grid, then N later frames sampled from the
sequence ("Image 1" .. "Image N"). The model answers with one xyxy box per
requested image, on the same 0-1000 grid:

    <answer>{"boxes":[{"frame":1,"bbox":[x1,y1,x2,y2]},...]}</answer>

Scoring is deterministic and needs no judge:

  1. Boxes are recovered from the answer. Parsing is deliberately lenient (see
     `parse_boxes`): well-formed JSON first, then progressively looser fallbacks,
     so that a truncated or slightly malformed answer still earns credit for the
     frames it did localize.
  2. Each grid box is divided by 1000, clamped to [0, 1], converted to a pixel
     xywh box using the sequence's image size, and clipped to the image. A box
     with no positive area after clipping is discarded.
  3. Each requested frame scores the IoU between the predicted and ground-truth
     pixel boxes; a missing or discarded prediction scores 0.

Only frames whose GOT-10k `cover.label` is above 0 (target at least partly
visible) count. The per-task reward is the mean IoU over those frames. The
benchmark metric is frame-weighted rather than sequence-weighted, so
`compute_metrics` pools the per-frame sums across tasks:

  visible_gt_AO       mean IoU over all visible frames (GOT-10k average overlap)
  visible_gt_SR@0.5   fraction of visible frames with IoU > 0.5
  visible_gt_SR@0.75  fraction of visible frames with IoU > 0.75
"""

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, ClassVar, Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field

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
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
FRAME_BBOX_PAIR_PATTERN = re.compile(
    r"[\"']?frame[\"']?\s*[:=]?\s*(-?\d+)\s*,?\s*[\"']?bbox[\"']?\s*[:=]?\s*\[([^\]]+)\]", re.IGNORECASE
)
BOX_LIST_KEYS = ("boxes", "bboxes", "predictions", "answers")
BOX_VALUE_KEYS = ("bbox", "box", "bbox_2d", "bounding_box", "pred_bbox", "prediction", "answer")
XYXY_KEYS = ("x1", "y1", "x2", "y2")

RawBox = list[float]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Got10kTarget(BaseModel):
    """Ground truth for one requested image."""

    # Prompt-local image id (1..N), the id the model is asked to answer with.
    frame: int = Field(ge=1)
    # 1-based frame number inside the source sequence (bookkeeping only).
    source_frame: Optional[int] = None
    # Ground-truth box in pixels, [x, y, width, height].
    bbox: list[float] = Field(min_length=4, max_length=4)
    # GOT-10k cover.label value; the frame is scored only when this is above 0.
    visible: float = 1.0


class Got10kVerifierMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    sequence: Optional[str] = None
    image_width: float = Field(gt=0)
    image_height: float = Field(gt=0)
    targets: list[Got10kTarget] = Field(min_length=1)


class Got10kFrameScore(BaseModel):
    frame: int
    visible: bool
    iou: float
    # "ok", "missing" (no box for this frame), "non_finite", or "non_positive".
    status: str
    predicted_bbox: Optional[list[float]] = None


class Got10kResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class Got10kVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Got10kVerifierMetadata


class Got10kVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    # How the answer was read: "structured" (boxes recovered), "empty" (no text),
    # "ambiguous_numbers" (numbers but no recoverable boxes), or "parse_failed".
    parse_status: str
    frame_scores: list[Got10kFrameScore]
    target_frames: int
    parsed_frames: int
    visible_frames: int
    visible_iou_sum: float
    visible_sr50_count: int
    visible_sr75_count: int


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def _extract_numbers(text: str) -> list[float]:
    return [float(x) for x in NUMBER_PATTERN.findall(text)]


def _load_json_or_literal(text: str) -> Any:
    """json.loads, falling back to a Python literal (single quotes, tuples)."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
        return None


def _load_json_duplicate_aware(text: str) -> Any:
    """json.loads that keeps every repeat of a duplicated key.

    Models sometimes put every frame in one object, e.g.
    {"frame":1,"bbox":[...],"frame":2,"bbox":[...]}; plain json.loads keeps only
    the last pair. Such an object is split into a list of objects, starting a
    new one whenever a key repeats, so each frame keeps its own box.
    """

    def hook(pairs: list[tuple[str, Any]]) -> Any:
        keys = [key for key, _ in pairs]
        if len(keys) == len(set(keys)):
            return dict(pairs)
        groups: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        for key, value in pairs:
            if key in current:
                groups.append(current)
                current = {}
            current[key] = value
        if current:
            groups.append(current)
        return groups

    def flatten(value: Any) -> Any:
        if isinstance(value, list):
            out: list[Any] = []
            for item in value:
                item = flatten(item)
                if isinstance(item, list):
                    out.extend(item)
                else:
                    out.append(item)
            return out
        if isinstance(value, dict):
            return {key: flatten(item) for key, item in value.items()}
        return value

    try:
        return flatten(json.loads(text, object_pairs_hook=hook))
    except (ValueError, RecursionError):
        return None


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


def _coerce_box(value: Any) -> Optional[RawBox]:
    """Read four numbers out of a list, a dict holding a box, or a string of either."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) < 4:
            return None
        try:
            return [float(x) for x in value[:4]]
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        for key in BOX_VALUE_KEYS:
            if key in value:
                box = _coerce_box(value[key])
                if box is not None:
                    return box
        if all(key in value for key in XYXY_KEYS):
            try:
                return [float(value[key]) for key in XYXY_KEYS]
            except (TypeError, ValueError):
                return None
        return None
    if isinstance(value, str):
        loaded = _load_json_or_literal(value)
        return None if isinstance(loaded, str) else _coerce_box(loaded)
    return None


def _as_frame_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return -1


def _structured_boxes(value: Any, frame_ids: list[int]) -> dict[int, RawBox]:
    """Boxes keyed by frame id from a decoded answer, for the requested frames only."""
    if isinstance(value, str):
        value = _load_json_or_literal(value)
        if isinstance(value, str):
            return {}
    if value is None:
        return {}

    parsed: dict[int, RawBox] = {}
    if isinstance(value, dict):
        for key in BOX_LIST_KEYS:
            if key in value:
                return _structured_boxes(value[key], frame_ids)
        # {"1": [...], "frame 2": [...]} — the frame id is the first number in the key.
        for key, item in value.items():
            match = re.search(r"\d+", str(key))
            if match and int(match.group(0)) in frame_ids:
                box = _coerce_box(item)
                if box is not None:
                    parsed[int(match.group(0))] = box
        if len(frame_ids) == 1:
            box = _coerce_box(value)
            if box is not None:
                parsed[frame_ids[0]] = box
        return parsed

    if isinstance(value, (list, tuple)):
        if len(frame_ids) == 1:
            box = _coerce_box(value)
            if box is not None:
                return {frame_ids[0]: box}
        if all(isinstance(item, dict) for item in value):
            for position, item in enumerate(value):
                raw_frame = item.get("frame")
                if raw_frame is None:
                    # No frame id: fall back to the entry's position in the list.
                    raw_frame = frame_ids[position] if position < len(frame_ids) else -1
                frame_id = _as_frame_id(raw_frame)
                box = _coerce_box(item)
                if frame_id in frame_ids and box is not None:
                    parsed[frame_id] = box
            return parsed
        boxes = [_coerce_box(item) for item in value]
        if all(box is not None for box in boxes):
            return {frame_id: box for frame_id, box in zip(frame_ids, boxes) if box is not None}
    return parsed


def parse_boxes(text: str, frame_ids: list[int]) -> tuple[dict[int, RawBox], str]:
    """Recover {frame id: raw box} from the model's answer, plus a parse status.

    Tried in order, stopping at the first that yields any box:
      1. Each <answer>...</answer> payload, then the whole text, read as JSON
         (whole payload first, then every JSON object embedded in it).
      2. "frame": N ... "bbox": [...] pairs found by pattern, for JSON that was
         truncated or never closed. The first occurrence of a frame wins.
      3. "frame N" / "image N" followed by a bracketed list.
      4. A bare run of numbers: exactly 4 per frame, or 5 per frame when each
         group starts with its frame id.
    """
    text = text.strip()
    boxes: dict[int, RawBox] = {}

    for payload in [*ANSWER_PATTERN.findall(text), text]:
        loaded = _load_json_duplicate_aware(payload)
        if loaded is None:
            loaded = _load_json_or_literal(payload)
        boxes.update(_structured_boxes(loaded, frame_ids))
        if boxes:
            break
        for obj in _iter_json_objects(payload):
            boxes.update(_structured_boxes(obj, frame_ids))
        if boxes:
            break

    if not boxes:
        for match in FRAME_BBOX_PAIR_PATTERN.finditer(text):
            frame_id = int(match.group(1))
            numbers = _extract_numbers(match.group(2))
            if frame_id in frame_ids and frame_id not in boxes and len(numbers) >= 4:
                boxes[frame_id] = numbers[:4]

    if not boxes:
        for frame_id in frame_ids:
            match = re.search(rf"(?:frame|image)\s*{frame_id}\D+\[([^\]]+)\]", text, re.IGNORECASE | re.DOTALL)
            if match:
                numbers = _extract_numbers(match.group(1))
                if len(numbers) >= 4:
                    boxes[frame_id] = numbers[:4]

    if not boxes:
        numbers = _extract_numbers(text)
        if len(numbers) == len(frame_ids) * 4:
            boxes = {frame_id: numbers[i * 4 : i * 4 + 4] for i, frame_id in enumerate(frame_ids)}
        elif len(numbers) == len(frame_ids) * 5:
            groups = [numbers[i * 5 : i * 5 + 5] for i in range(len(frame_ids))]
            if all(math.isfinite(group[0]) and int(group[0]) == fid for group, fid in zip(groups, frame_ids)):
                boxes = {frame_id: group[1:] for frame_id, group in zip(frame_ids, groups)}

    if boxes:
        return boxes, "structured"
    if not text:
        return {}, "empty"
    return {}, "ambiguous_numbers" if _extract_numbers(text) else "parse_failed"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _round_box(box: list[float]) -> list[float]:
    return [round(float(x), 3) for x in box]


def grid_box_to_pixel_xywh(box: RawBox, width: float, height: float) -> list[float]:
    """0-1000 grid xyxy -> pixel xywh. Each coordinate is clamped to the grid first."""
    x1, y1, x2, y2 = [min(max(value / GRID, 0.0), 1.0) for value in box]
    return [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height]


def clip_box_to_image(box: list[float], width: float, height: float) -> tuple[Optional[list[float]], str]:
    """Clip a pixel xywh box to the image; None when nothing usable is left."""
    if any(not math.isfinite(x) for x in box):
        return None, "non_finite"
    x, y, w, h = box
    x = min(max(x, 0.0), width)
    y = min(max(y, 0.0), height)
    w = min(max(w, 0.0), max(width - x, 0.0))
    h = min(max(h, 0.0), max(height - y, 0.0))
    if w <= 0 or h <= 0:
        return None, "non_positive"
    return [x, y, w, h], "ok"


def iou_xywh(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    inter_w = max(min(ax1 + aw, bx1 + bw) - max(ax1, bx1), 0.0)
    inter_h = max(min(ay1 + ah, by1 + bh) - max(ay1, by1), 0.0)
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return min(max(intersection / union, 0.0), 1.0)


def score_sequence(text: str, metadata: Got10kVerifierMetadata) -> dict[str, Any]:
    frame_ids = [target.frame for target in metadata.targets]
    width, height = float(metadata.image_width), float(metadata.image_height)
    boxes, parse_status = parse_boxes(text, frame_ids)

    frame_scores: list[Got10kFrameScore] = []
    for target in metadata.targets:
        raw_box = boxes.get(target.frame)
        predicted: Optional[list[float]] = None
        status = "missing"
        if raw_box is not None:
            # Boxes are compared at 3-decimal pixel precision on both sides.
            predicted, status = clip_box_to_image(
                _round_box(grid_box_to_pixel_xywh(raw_box, width, height)), width, height
            )
        iou = iou_xywh(predicted, _round_box(target.bbox)) if predicted is not None else 0.0
        frame_scores.append(
            Got10kFrameScore(
                frame=target.frame,
                visible=target.visible > 0,
                iou=iou,
                status=status,
                predicted_bbox=_round_box(predicted) if predicted is not None else None,
            )
        )

    visible = [score for score in frame_scores if score.visible]
    iou_sum = sum(score.iou for score in visible)
    return {
        "reward": iou_sum / len(visible) if visible else 0.0,
        "parse_status": parse_status,
        "frame_scores": frame_scores,
        "target_frames": len(frame_scores),
        "parsed_frames": sum(1 for score in frame_scores if score.status == "ok"),
        "visible_frames": len(visible),
        "visible_iou_sum": iou_sum,
        "visible_sr50_count": sum(1 for score in visible if score.iou > 0.5),
        "visible_sr75_count": sum(1 for score in visible if score.iou > 0.75),
    }


def aggregate_frame_metrics(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Frame-weighted GOT-10k metrics pooled over rollouts."""

    def total(key: str) -> float:
        return sum(rollout.get(key) or 0 for rollout in rollouts)

    visible_frames = int(total("visible_frames"))
    target_frames = int(total("target_frames"))
    metrics: dict[str, Any] = {"visible_gt_samples": visible_frames}
    # No visible frame means no coverage: report no score rather than a zero.
    if visible_frames:
        metrics["visible_gt_AO"] = total("visible_iou_sum") / visible_frames
        metrics["visible_gt_SR@0.5"] = total("visible_sr50_count") / visible_frames
        metrics["visible_gt_SR@0.75"] = total("visible_sr75_count") / visible_frames
    if target_frames:
        metrics["parse_pass_ratio"] = total("parsed_frames") / target_frames
    return metrics


def _assistant_text(body: BaseVerifyRequest) -> str:
    """Text of the assistant's message output (reasoning items are not read)."""
    texts: list[str] = []
    for output in body.response.output:
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        texts.extend(text for item in output.content if isinstance(text := getattr(item, "text", None), str))
    return "\n".join(texts).strip()


class Got10kVerifier:
    async def verify(self, body: Got10kVerifyRequest) -> Got10kVerifyResponse:
        result = score_sequence(_assistant_text(body), body.verifier_metadata)
        return Got10kVerifyResponse(**body.model_dump(), **result)


class Got10kResourcesServer(Got10kVerifier, SimpleResourcesServer):
    config: Got10kResourcesServerConfig

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return aggregate_frame_metrics([rollout for task in tasks for rollout in task])

    def compute_repeat_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return self.compute_metrics(tasks)

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        names = ("visible_gt_AO", "visible_gt_SR@0.5", "visible_gt_SR@0.75", "parse_pass_ratio")
        return {name: agent_metrics[name] for name in names if name in agent_metrics}


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=Got10kVerifier,
    request_model=Got10kVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    Got10kResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = Got10kResourcesServer.run_webserver()  # noqa: F401
