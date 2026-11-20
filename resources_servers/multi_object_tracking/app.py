# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-object box tracking verifier (multi-image, 0-1000 grid protocol), built on the GOT-10k verifier.

GOT-10k tracks one object per sequence, and `resources_servers/got10k` scores one
box per frame. This server scores prompts that ask for several objects at once.
Each object is given an integer id in the prompt, and the answer carries that id
on every entry:

    <answer>{"tracks":[{"box":1,"frame":1,"bbox":[x1,y1,x2,y2],"visible":true},...]}</answer>

Every object is scored on its own with the single-object verifier's own
`score_sequence`, so the per-object numbers are exactly GOT-10k's: boxes on a
0-1000 grid are converted to clipped pixel boxes, each frame where the target is
visible scores the IoU with the ground truth (0 when there is no usable box), and
frames where the target is not visible are not scored.

  * The id is read from "box", "id", "object", "object_id", "track_id" or "track"
    (first one holding an integer). The box is read from "bbox", "bbox_2d",
    "bounding_box", "pred_bbox", or from "box" when "box" holds a list.
  * An entry without an id is accepted only when the task has a single object, so
    a single-object answer in the GOT-10k format ({"boxes":[{"frame":..,"bbox":..}]})
    scores the same here as in `got10k`.
  * An entry with "visible": false is the model saying the object is hidden: it is
    not used as a box, but it counts as an answer for that (object, frame) pair.
    Visibility claims earn no reward and cost none by themselves, as in GOT-10k; a
    hidden claim on a visible frame scores 0 for it.
  * Only entries for a requested (object, frame) pair are kept. The first box for
    a pair wins, and a box replaces a hidden claim for the same pair.
  * <answer> blocks are tried last one first, then the whole text; the first that
    answers a requested pair is used.
  * A truncated answer is salvaged entry by entry.

The per-task reward is the mean, over the objects that are visible in at least one
requested frame, of each object's mean IoU over its visible frames. Objects that
are never visible are left out; a task where no object is ever visible scores 0.
`compute_metrics` pools frames over all objects and tasks, giving the same
frame-weighted `visible_gt_AO` / `visible_gt_SR@*` as the single-object server.
"""

import json
import re
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture
from resources_servers.got10k.app import (
    ANSWER_PATTERN,
    _assistant_text,
    _coerce_box,
    _iter_json_objects,
    _load_json_or_literal,
    aggregate_frame_metrics,
    score_sequence,
)
from resources_servers.got10k.task_data import Got10kVerifierMetadata
from resources_servers.multi_object_tracking.task_data import MultiObjectTrackingVerifierMetadata


TRACK_LIST_KEYS = ("tracks", "boxes", "bboxes", "predictions", "answers")
OBJECT_ID_KEYS = ("box", "id", "object", "object_id", "track_id", "track")
BOX_KEYS = ("bbox", "bbox_2d", "bounding_box", "pred_bbox", "box")
FLAT_OBJECT_PATTERN = re.compile(r"\{[^{}]*\}")
SUMMED_FIELDS = (
    "target_frames",
    "parsed_frames",
    "visible_frames",
    "visible_iou_sum",
    "visible_sr50_count",
    "visible_sr75_count",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class MultiObjectTrackingObjectScore(BaseModel):
    id: int
    # Mean IoU over the object's visible frames; None when it is never visible (not scored).
    reward: float | None = None
    target_frames: int
    parsed_frames: int
    visible_frames: int


class MultiObjectTrackingResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class MultiObjectTrackingVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: MultiObjectTrackingVerifierMetadata


class MultiObjectTrackingVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    # "structured" (entries read from one JSON value), "salvaged" (read entry by entry
    # from a damaged answer), "empty" (no text) or "parse_failed".
    parse_status: str
    object_scores: list[MultiObjectTrackingObjectScore]
    scored_objects: int
    target_frames: int
    parsed_frames: int
    visible_frames: int
    visible_iou_sum: float
    visible_sr50_count: int
    visible_sr75_count: int


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        try:
            return int(value)
        except ValueError:
            # A digit string beyond Python's int conversion limit is not an id or a frame.
            return None
    return None


def _read_entry(entry: Any) -> tuple[int | None, int, list[float] | None] | None:
    """(object id or None, frame, grid box) of one answer entry, or None when the entry is unusable.

    The box is None for an entry with "visible": false, which answers the pair without giving a box.
    """
    if not isinstance(entry, dict):
        return None
    frame = _as_int(entry.get("frame"))
    if frame is None:
        return None
    object_id = next((i for key in OBJECT_ID_KEYS if (i := _as_int(entry.get(key))) is not None), None)
    if entry.get("visible") is False:
        return object_id, frame, None
    for key in BOX_KEYS:
        value = entry.get(key)
        if isinstance(value, (list, tuple, dict, str)) and (box := _coerce_box(value)) is not None:
            return object_id, frame, box
    return None


def _entries_of(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in TRACK_LIST_KEYS:
            if isinstance(value.get(key), list):
                return value[key]
        if "frame" in value:
            return [value]
    return []


def parse_tracks(text: str, requested: dict[int, set[int]]) -> tuple[dict[int, dict[int, list[float] | None]], str]:
    """Answers keyed by object id, then frame, for the requested frames of each object only.

    `requested` maps each object id to its requested frames. An entry without an id belongs to the only object, if
    there is one. The value for a pair is its box, or None when the answer only says the object is hidden there.

    Each <answer>...</answer> payload is tried, last one first (a later block corrects an earlier draft), then the
    whole text, stopping at the first that answers a requested pair. A payload is read as one JSON value, then as
    the JSON objects embedded in it, then entry by entry for an answer that was cut off.
    """
    tracks: dict[int, dict[int, list[float] | None]] = {}
    text = text.strip()
    if not text:
        return tracks, "empty"

    def add(entries: list[Any]) -> None:
        for entry in entries:
            read = _read_entry(entry)
            if read is None:
                continue
            object_id, frame, box = read
            if object_id is None:
                if len(requested) != 1:
                    continue
                object_id = next(iter(requested))
            if frame not in requested.get(object_id, ()):
                continue
            answers = tracks.setdefault(object_id, {})
            # The first box for a pair wins; a box replaces a hidden claim.
            if answers.get(frame) is None:
                answers[frame] = box

    for payload in [*reversed(ANSWER_PATTERN.findall(text)), text]:
        add(_entries_of(_load_json_or_literal(payload)))
        if tracks:
            return tracks, "structured"
        for value in _iter_json_objects(payload):
            add(_entries_of(value))
        if tracks:
            return tracks, "structured"
        # A cut-off or otherwise damaged answer: read every complete flat {...} entry on its own.
        add([_load_json_or_literal(chunk) for chunk in FLAT_OBJECT_PATTERN.findall(payload)])
        if tracks:
            return tracks, "salvaged"
    return tracks, "parse_failed"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_objects(text: str, metadata: MultiObjectTrackingVerifierMetadata) -> dict[str, Any]:
    requested = {obj.id: {target.frame for target in obj.targets} for obj in metadata.objects}
    tracks, parse_status = parse_tracks(text, requested)
    totals: dict[str, Any] = dict.fromkeys(SUMMED_FIELDS, 0)
    object_scores: list[MultiObjectTrackingObjectScore] = []
    for obj in metadata.objects:
        single = Got10kVerifierMetadata(
            image_width=metadata.image_width, image_height=metadata.image_height, targets=obj.targets
        )
        answers = tracks.get(obj.id, {})
        boxes = {frame: box for frame, box in answers.items() if box is not None}
        answer = (
            json.dumps({"boxes": [{"frame": frame, "bbox": box} for frame, box in boxes.items()]}) if boxes else ""
        )
        result = score_sequence(answer, single)
        # A hidden claim answers its pair without a box, so it counts as parsed but is never scored as a box.
        result["parsed_frames"] += len(answers) - len(boxes)
        for field in SUMMED_FIELDS:
            totals[field] += result[field]
        object_scores.append(
            MultiObjectTrackingObjectScore(
                id=obj.id,
                reward=result["reward"] if result["visible_frames"] else None,
                target_frames=result["target_frames"],
                parsed_frames=result["parsed_frames"],
                visible_frames=result["visible_frames"],
            )
        )
    scored = [score.reward for score in object_scores if score.reward is not None]
    return {
        "reward": sum(scored) / len(scored) if scored else 0.0,
        "parse_status": parse_status,
        "object_scores": object_scores,
        "scored_objects": len(scored),
        **totals,
    }


class MultiObjectTrackingVerifier:
    async def verify(self, body: MultiObjectTrackingVerifyRequest) -> MultiObjectTrackingVerifyResponse:
        result = score_objects(_assistant_text(body), body.verifier_metadata)
        return MultiObjectTrackingVerifyResponse(**body.model_dump(), **result)


class MultiObjectTrackingResourcesServer(MultiObjectTrackingVerifier, SimpleResourcesServer):
    config: MultiObjectTrackingResourcesServerConfig

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return aggregate_frame_metrics([rollout for task in tasks for rollout in task])

    def compute_repeat_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return self.compute_metrics(tasks)

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        names = ("visible_gt_AO", "visible_gt_SR@0.5", "visible_gt_SR@0.75", "parse_pass_ratio")
        return {name: agent_metrics[name] for name in names if name in agent_metrics}


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=MultiObjectTrackingVerifier,
    request_model=MultiObjectTrackingVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    MultiObjectTrackingResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = MultiObjectTrackingResourcesServer.run_webserver()  # noqa: F401
