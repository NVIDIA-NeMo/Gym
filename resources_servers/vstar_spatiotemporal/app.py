# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V-STaR (video spatio-temporal reasoning) verifier, frames-as-images protocol with 0-1000 grid boxes.

Every V-STaR item is asked as five independent tasks (`sub_task` in the verifier metadata):

    vqa         "what": a free-text answer, rated 0-3 by an LLM judge against the reference answer
    temporal    "when", chain 1: `From <t>a</t>s to <t>b</t>s`, scored by temporal IoU with the annotated span
    spatial     "where", chain 1: {"tracks":[{"id":1,"t":T,"box":[x1,y1,x2,y2]}]} given the span
    spatial_2   "where", chain 2: the same shape with no span given
    temporal_2  "when", chain 2: a span, given the annotated boxes without their times

Boxes are integers on a 0-1000 grid and are converted to video pixels. A box answer is scored against the annotated
box at every annotated whole second: the IoU at that second, or 0 when the answer has no box for it. The per-task
reward is 1.0 for a VQA rating of 2 or 3 (else 0.0), the temporal IoU, or the mean box IoU.

The benchmark metrics are defined per item, not per task, so `compute_metrics` folds the five tasks of an item back
together. As in the V-STaR evaluation, an item whose VQA answer is empty contributes nothing to the temporal and
spatial means, and still counts in the denominator of the overall VQA accuracy. For each chain k:

    AM_k  = mean(VQA accuracy, temporal mIoU_k, spatial mIoU_k)
    LGM_k = -mean(log(1 - VQA accuracy), log(1 - temporal mIoU_k), log(1 - spatial mIoU_k))

and `mAM`, `mLGM` are the means over the two chains.

The answer parsers and the metric definitions follow the V-STaR evaluation (https://github.com/V-STaR-Bench/V-STaR,
eval.py) and the 0-1000 grid answer format of the frames-as-images protocol this benchmark reproduces.
"""

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import JudgeError, VerifierFixture, call_judge


SUB_TASKS = ("vqa", "temporal", "spatial", "spatial_2", "temporal_2")
IOU_THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

JUDGE_SYSTEM_PROMPT = """
As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct 'groundtruth' answer, and the candidate answer will be provided to you.
Your assessment should range from 0 to 3, \
based solely on the semantic similarity between the groundtruth and the candidate answer, \
disregarding any grammatical differences.
A rating of 0 suggests no similarity, implying the candidate answer is entirely incorrect.
A rating of 1 suggests low similarity, meaning the candidate answer is largely incorrect.
A rating of 2 suggests high similarity, meaning the candidate answer is largely correct.
Lastly, a rating of 3 indicates complete similarity, which means the candidate answer is entirely correct.
Your response should be a single integer from 0, 1, 2, or 3.
"""
JUDGE_PROMPT = "Question: {}\nGroundtruth answer: {}\nCandidate answer: {}\nYour response: "

# Metrics also reported for each video-length and domain group.
GROUP_METRICS = (
    "items",
    "VQA_acc",
    "temporal_mIoU",
    "spatial_mIoU",
    "temporal_2_mIoU",
    "spatial_2_mIoU",
    "mAM",
    "mLGM",
)
KEY_METRICS = (
    "mAM",
    "mLGM",
    "VQA_acc",
    "temporal_mIoU",
    "spatial_mIoU",
    "temporal_2_mIoU",
    "spatial_2_mIoU",
    "parsed_rate_mean",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class VstarBox(BaseModel):
    model_config = ConfigDict(extra="allow")

    # The whole second this box is annotated at.
    timestamp: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class VstarVerifierMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Position of the item in the annotation file; the five tasks of one item share it.
    item_index: int
    sub_task: str = Field(pattern="^(vqa|temporal|spatial|spatial_2|temporal_2)$")
    domain: Optional[str] = None
    fps: float = Field(gt=0)
    frame_count: int = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    question: str
    answer: str
    # Annotated [start, end] of the event, in seconds.
    timestamps: list[float] = Field(min_length=2, max_length=2)
    bboxes: list[VstarBox]


class VstarResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    # Rates the VQA answers. Without it, a VQA task with a non-empty answer cannot be scored.
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(
            input=[], temperature=0.0, max_output_tokens=512
        )
    )


class VstarVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: VstarVerifierMetadata


class VstarVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    sub_task: str
    item_index: int
    # Whether the reply held something to score: a non-empty answer, a time span, or at least one box.
    parsed: bool
    # vqa: the answer text sent to the judge, whether it was non-empty, the judge's reply and its rating (-1 when the
    # reply was not a single integer). An empty model reply is rated 0 without calling the judge.
    answer_vqa: Optional[str] = None
    answered: Optional[bool] = None
    answer_tag_found: Optional[bool] = None
    judge_output: Optional[str] = None
    vqa_score: Optional[int] = None
    # temporal, temporal_2: the parsed [start, end] (empty when unreadable) and its IoU with the annotated span.
    predicted_span: Optional[list[float]] = None
    temporal_iou: Optional[float] = None
    # spatial, spatial_2: boxes in video pixels keyed by whole second, the mean IoU over the annotated seconds, and
    # the fraction of them at or above each of IOU_THRESHOLDS.
    predicted_boxes: Optional[dict[str, list[float]]] = None
    spatial_miou: Optional[float] = None
    spatial_aps: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def answer_text(text: str) -> str:
    """The content of the first <answer>...</answer>, or the whole text when there is none."""
    match = ANSWER_PATTERN.search(text)
    return match.group(1).strip() if match else text


def parse_span(text: str) -> list[float]:
    """[start, end] in seconds, or [] unless the answer holds exactly two numbers. `m:ss` times count as seconds."""
    text = answer_text(text)
    for stamp in re.findall(r"(\d+:\d+)", text):
        minutes, seconds = map(int, stamp.split(":"))
        text = text.replace(stamp, f"<t>{minutes * 60 + seconds}</t>s")
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    return [float(numbers[0]), float(numbers[1])] if len(numbers) == 2 else []


def balance_brackets(text: str) -> str:
    """Add the brackets a JSON string is short of, by count: closers at the end, openers at the start."""
    square = text.count("[") - text.count("]")
    text = text + "]" * square if square > 0 else "[" * -square + text
    curly = text.count("{") - text.count("}")
    return text + "}" * curly if curly > 0 else "{" * -curly + text


def close_truncated_json(text: str) -> str:
    """Repair JSON cut off mid-way: drop the unfinished last element, then close what is open, innermost first."""
    closers: list[str] = []
    in_string = escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            closers.append("]" if char == "[" else "}")
        elif char in "]}" and closers:
            closers.pop()
    if in_string:
        text += '"'
    text = re.sub(r',\s*("[^"]*"\s*(:\s*(\[[^\]]*)?)?)?$', "", text)
    text = re.sub(r":\s*$", ":null", text)
    return text + "".join(reversed(closers))


def _load_tracks_json(text: str) -> Any:
    """The first candidate span of the answer that loads as JSON, repaired if need be; None when none does."""
    candidates = []
    if text.lstrip().startswith("["):
        # A bare list: the outermost {...} would span several of its elements and fail to load.
        candidates.append(re.search(r"\[[\s\S]*\]", text))
    candidates += [re.search(r"\{[\s\S]*\}", text), re.search(r"\[[\s\S]*\]", text)]
    for match in candidates:
        if not match:
            continue
        raw = match.group(0)
        for repair in (lambda value: value, balance_brackets, close_truncated_json):
            try:
                return json.loads(repair(raw))
            except Exception:
                continue
    return None


def parse_tracks(text: str, width: float, height: float) -> Optional[dict[str, list[float]]]:
    """Boxes in video pixels keyed by whole second, from a 0-1000 grid tracks answer; None when there are none.

    Accepts {"tracks":[{"t":T,"box":[..]}]} entries, {"tracks":[{"boxes":{T:[..]}}]} entries and a bare list of
    either. A time is rounded to the nearest whole second; when several fall on one second the closest to it wins.
    Coordinates that are all at most 1 are read as fractions of the frame.
    """
    data = _load_tracks_json(answer_text(text or ""))
    tracks = data.get("tracks") if isinstance(data, dict) else data
    if not isinstance(tracks, list):
        return None
    best: dict[str, tuple[float, list[float]]] = {}

    def put(time: Any, box: Any) -> None:
        try:
            time = float(time)
            if box is None or len(box) != 4:
                return
            box = [float(value) for value in box]
            if max(box) <= 1.0:
                box = [value * 1000 for value in box]
            pixels = [box[0] / 1000 * width, box[1] / 1000 * height, box[2] / 1000 * width, box[3] / 1000 * height]
            second = str(int(round(time)))
        except Exception:
            return
        distance = abs(time - round(time))
        if second not in best or distance < best[second][0]:
            best[second] = (distance, pixels)

    for entry in tracks:
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("boxes"), dict):
            for time, box in entry["boxes"].items():
                put(time, box)
        elif "t" in entry:
            put(entry["t"], entry.get("box") or entry.get("xy"))
    return {second: value[1] for second, value in best.items()} or None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def temporal_iou(span: list[float], predicted: Any) -> float:
    if not predicted:
        return 0.0
    if isinstance(predicted, str):
        try:
            predicted = ast.literal_eval(predicted)
        except (ValueError, SyntaxError):
            return 0.0
    if (
        not isinstance(predicted, (list, tuple))
        or len(predicted) != 2
        or not all(isinstance(value, (int, float)) for value in predicted)
    ):
        return 0.0
    start, end = span
    predicted_start, predicted_end = predicted
    intersection = max(0, min(end, predicted_end) - max(start, predicted_start))
    union = max(end, predicted_end) - min(start, predicted_start)
    return intersection / union if union > 0 else 0.0


def box_iou(box: VstarBox, predicted: Any) -> float:
    if not isinstance(predicted, (list, tuple)) or len(predicted) != 4:
        return 0.0
    xmin, ymin, xmax, ymax = predicted
    x1 = max(box.xmin, xmin)
    y1 = max(box.ymin, ymin)
    x2 = min(box.xmax, xmax)
    y2 = min(box.ymax, ymax)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box.xmax - box.xmin) * (box.ymax - box.ymin) + (xmax - xmin) * (ymax - ymin) - intersection
    return intersection / union if union > 0 else 0.0


def spatial_scores(boxes: list[VstarBox], predicted: Optional[dict[str, list[float]]]) -> tuple[list[float], float]:
    """(fraction of annotated seconds with IoU at or above each threshold, mean IoU over the annotated seconds)."""
    if not predicted:
        return [0.0] * len(IOU_THRESHOLDS), 0.0
    ious = [box_iou(box, predicted[str(box.timestamp)]) if str(box.timestamp) in predicted else 0.0 for box in boxes]
    if not ious:
        return [0.0] * len(IOU_THRESHOLDS), 0.0
    return [_mean([1 if iou >= threshold else 0 for iou in ious]) for threshold in IOU_THRESHOLDS], _mean(ious)


def parse_judge_score(text: str) -> int:
    """The judge's rating, or -1 when its reply is not a single integer."""
    try:
        return int(str(text).strip())
    except (ValueError, TypeError):
        return -1


def score_reply(text: str, metadata: VstarVerifierMetadata) -> dict[str, Any]:
    """Everything about one task that needs no judge. A VQA task still needs its rating (`vqa_score`)."""
    sub_task = metadata.sub_task
    result: dict[str, Any] = {"sub_task": sub_task, "item_index": metadata.item_index}
    if sub_task == "vqa":
        answer = answer_text(text)
        # A reply with no text is kept as an answered item that rates 0, like any other wrong answer.
        result.update(
            answer_vqa=answer,
            answered=bool(answer) or not text,
            answer_tag_found=ANSWER_PATTERN.search(text) is not None,
            parsed=ANSWER_PATTERN.search(text) is not None,
            reward=0.0,
        )
    elif sub_task in ("temporal", "temporal_2"):
        span = parse_span(text)
        iou = temporal_iou(metadata.timestamps, span)
        result.update(predicted_span=span, temporal_iou=iou, parsed=bool(span), reward=iou)
    else:
        boxes = parse_tracks(text, metadata.width, metadata.height)
        aps, miou = spatial_scores(metadata.bboxes, boxes)
        result.update(predicted_boxes=boxes, spatial_aps=aps, spatial_miou=miou, parsed=bool(boxes), reward=miou)
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _new_stats() -> dict[str, Any]:
    return {
        "ratings": [],
        "valid_ratings": [],
        "correct": 0,
        "temporal": [],
        "temporal_2": [],
        "spatial": [],
        "spatial_2": [],
        "spatial_aps": [[] for _ in IOU_THRESHOLDS],
        "spatial_2_aps": [[] for _ in IOU_THRESHOLDS],
        "combined": {name: 0 for name in _COMBINED},
    }


# Combined-success counters: an item passes a part with a VQA rating of 2 or more, a temporal IoU of at least 0.3,
# and a mean box IoU of at least 0.1.
_COMBINED = (
    "VQA&Temp_1",
    "VQA&Temp_2",
    "VQA&Spat_1",
    "VQA&Spat_2",
    "Temp&Spat_1",
    "Temp&Spat_2",
    "VQA&Temp&Spat_1",
    "VQA&Temp&Spat_2",
)


def _summarize(stats: dict[str, Any], total: int) -> dict[str, float]:
    if total == 0:
        return {}
    accuracy = stats["correct"] / total
    temporal, temporal_2 = _mean(stats["temporal"]), _mean(stats["temporal_2"])
    spatial, spatial_2 = _mean(stats["spatial"]), _mean(stats["spatial_2"])

    def recall(ious: list[float], threshold: float) -> float:
        return _mean([1 if iou >= threshold else 0 for iou in ious])

    def log_miss(value: float) -> float:
        return math.log(1 - value) if value < 1 else float("-inf")

    am1 = (accuracy + temporal + spatial) / 3
    am2 = (accuracy + temporal_2 + spatial_2) / 3
    lgm1 = -(log_miss(accuracy) + log_miss(temporal) + log_miss(spatial)) / 3
    lgm2 = -(log_miss(accuracy) + log_miss(temporal_2) + log_miss(spatial_2)) / 3
    metrics: dict[str, float] = {
        "items": total,
        "VQA_avg_score": _mean(stats["ratings"]),
        "VQA_avg_valid_score": _mean(stats["valid_ratings"]),
        "VQA_acc": accuracy,
        "temporal_mIoU": temporal,
        "spatial_mIoU": spatial,
        "temporal_2_mIoU": temporal_2,
        "spatial_2_mIoU": spatial_2,
        "AM1": am1,
        "AM2": am2,
        "mAM": (am1 + am2) / 2,
        "LGM1": lgm1,
        "LGM2": lgm2,
        "mLGM": (lgm1 + lgm2) / 2,
    }
    for name, ious in (("temporal", stats["temporal"]), ("temporal_2", stats["temporal_2"])):
        for threshold in (0.3, 0.5, 0.7):
            metrics[f"{name}_R1@{threshold}"] = recall(ious, threshold)
    for name, aps in (("spatial", stats["spatial_aps"]), ("spatial_2", stats["spatial_2_aps"])):
        for threshold, values in zip(IOU_THRESHOLDS, aps):
            metrics[f"{name}_mAP@{threshold}"] = _mean(values)
    for name, count in stats["combined"].items():
        metrics[name] = count / total
    return metrics


def _duration_group(metadata: dict[str, Any]) -> str:
    length = round(metadata["frame_count"] / metadata["fps"], 2)
    return "Short" if length < 60 else ("Medium" if length < 180 else "Long")


def aggregate_metrics(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Benchmark metrics from one rollout per task: overall, and a subset per video length and per domain."""
    items: dict[int, dict[str, dict[str, Any]]] = {}
    for rollout in rollouts:
        if rollout.get("sub_task") in SUB_TASKS and rollout.get("item_index") is not None:
            items.setdefault(rollout["item_index"], {})[rollout["sub_task"]] = rollout
    if not items:
        return {}

    overall = _new_stats()
    lengths: dict[str, dict[str, Any]] = {}
    domains: dict[str, dict[str, Any]] = {}
    parsed = {sub_task: 0 for sub_task in SUB_TASKS}
    for index in sorted(items):
        tasks = items[index]
        metadata = next(iter(tasks.values()))["verifier_metadata"]
        length_stats = lengths.setdefault(_duration_group(metadata), _new_stats())
        domain_stats = domains.setdefault(metadata.get("domain") or "unknown", _new_stats())
        for sub_task in SUB_TASKS:
            parsed[sub_task] += bool((tasks.get(sub_task) or {}).get("parsed"))
        vqa = tasks.get("vqa") or {}
        # An item with no VQA answer contributes nothing further, though it stays in the overall denominator.
        if not vqa.get("answered"):
            continue
        score = vqa.get("vqa_score")
        score = -1 if score is None else score

        def value(sub_task: str, key: str) -> float:
            return (tasks.get(sub_task) or {}).get(key) or 0.0

        def aps(sub_task: str) -> list[float]:
            return (tasks.get(sub_task) or {}).get("spatial_aps") or [0.0] * len(IOU_THRESHOLDS)

        t1, t2 = value("temporal", "temporal_iou"), value("temporal_2", "temporal_iou")
        s1, s2 = value("spatial", "spatial_miou"), value("spatial_2", "spatial_miou")
        passed = {
            "VQA&Temp_1": score >= 2 and t1 >= 0.3,
            "VQA&Temp_2": score >= 2 and t2 >= 0.3,
            "VQA&Spat_1": score >= 2 and s1 >= 0.1,
            "VQA&Spat_2": score >= 2 and s2 >= 0.1,
            "Temp&Spat_1": t1 >= 0.3 and s1 >= 0.1,
            "Temp&Spat_2": t2 >= 0.3 and s2 >= 0.1,
            "VQA&Temp&Spat_1": score >= 2 and t1 >= 0.3 and s1 >= 0.1,
            "VQA&Temp&Spat_2": score >= 2 and t2 >= 0.3 and s2 >= 0.1,
        }
        for stats in (overall, length_stats, domain_stats):
            stats["ratings"].append(score if score != -1 else 0)
            if score != -1:
                stats["valid_ratings"].append(score)
            stats["correct"] += score >= 2
            stats["temporal"].append(t1)
            stats["temporal_2"].append(t2)
            stats["spatial"].append(s1)
            stats["spatial_2"].append(s2)
            for position, (ap1, ap2) in enumerate(zip(aps("spatial"), aps("spatial_2"))):
                stats["spatial_aps"][position].append(ap1)
                stats["spatial_2_aps"][position].append(ap2)
            for name, ok in passed.items():
                stats["combined"][name] += ok

    # Overall counts every item; a group counts its answered items only.
    metrics: dict[str, Any] = dict(_summarize(overall, len(items)))
    for sub_task in SUB_TASKS:
        metrics[f"parsed_rate_{sub_task}"] = parsed[sub_task] / len(items)
    metrics["parsed_rate_mean"] = sum(parsed.values()) / len(items) / len(SUB_TASKS)
    for prefix, groups in (("length", lengths), ("domain", domains)):
        for name, stats in groups.items():
            summary = _summarize(stats, len(stats["ratings"]))
            metrics.update({f"{prefix}/{name}/{key}": summary[key] for key in GROUP_METRICS if key in summary})
    # LGM is infinite when a component reaches 1; such a value is left out rather than reported.
    return {key: value for key, value in metrics.items() if math.isfinite(value)}


def aggregate_repeats(tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Metrics of each repeat (the i-th rollout of every task), averaged over the repeats that report them."""
    repeats = max((len(rollouts) for rollouts in tasks), default=0)
    per_repeat = [
        aggregate_metrics([rollouts[index] for rollouts in tasks if index < len(rollouts)]) for index in range(repeats)
    ]
    keys = list(dict.fromkeys(key for metrics in per_repeat for key in metrics))
    return {key: _mean([metrics[key] for metrics in per_repeat if key in metrics]) for key in keys}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def _assistant_text(response: Any) -> str:
    """Text of the assistant's message output (reasoning items are not read)."""
    texts: list[str] = []
    for output in response.output:
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        texts.extend(text for item in output.content if isinstance(text := getattr(item, "text", None), str))
    return "\n".join(texts).strip()


class VstarVerifier:
    """Scores one task. `rate_answer` returns the judge's reply for a VQA answer."""

    async def rate_answer(self, question: str, reference: str, candidate: str) -> str:
        raise JudgeError("Configure judge_model_server to rate V-STaR VQA answers")

    async def verify(self, body: VstarVerifyRequest) -> VstarVerifyResponse:
        metadata = body.verifier_metadata
        result = score_reply(_assistant_text(body.response), metadata)
        if metadata.sub_task == "vqa" and result["answered"]:
            if result["answer_vqa"]:
                result["judge_output"] = await self.rate_answer(
                    metadata.question, metadata.answer, result["answer_vqa"]
                )
                result["vqa_score"] = parse_judge_score(result["judge_output"])
            else:
                result["vqa_score"] = 0
            result["reward"] = 1.0 if result["vqa_score"] >= 2 else 0.0
        return VstarVerifyResponse(**body.model_dump(), **result)


class VstarResourcesServer(VstarVerifier, SimpleResourcesServer):
    config: VstarResourcesServerConfig

    async def rate_answer(self, question: str, reference: str, candidate: str) -> str:
        if self.config.judge_model_server is None:
            raise JudgeError("Configure judge_model_server to rate V-STaR VQA answers")
        params = self.config.judge_responses_create_params.model_copy(
            update={
                "input": [
                    NeMoGymEasyInputMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
                    NeMoGymEasyInputMessage(role="user", content=JUDGE_PROMPT.format(question, reference, candidate)),
                ]
            },
            deep=True,
        )
        judged = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=params,
            response_model=NeMoGymResponse,
        )
        return _assistant_text(judged)

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return aggregate_repeats(tasks)

    def compute_repeat_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return self.compute_metrics(tasks)

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        return {name: agent_metrics[name] for name in KEY_METRICS if name in agent_metrics}


class _ExactMatchVerifier(VstarVerifier):
    """Fixture verifier: rates a VQA answer 3 when it equals the reference and 0 otherwise, with no model call."""

    async def rate_answer(self, question: str, reference: str, candidate: str) -> str:
        return "3" if candidate.strip().lower() == reference.strip().lower() else "0"


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_ExactMatchVerifier,
    request_model=VstarVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    VstarResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = VstarResourcesServer.run_webserver()  # noqa: F401
