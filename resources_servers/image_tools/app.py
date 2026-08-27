# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image-tools pivot verifier for PivotRL on VLM tool-use trajectories.

Scores a single rollout tool call against the demonstrated (SFT) tool call at a
pivot turn. Layered, cheapest check first:

  Level 0: tool name match     — same tool?
  Level 1: image target match  — operating on the same image(s)?
  Level 2: argument match      — per tool family (IoU / point distance / value)

`label` is never compared: it is a free-text rationale string, not an action.

No LLM judge — purely computational.

Tool families in the bvstyle tool-call data:
  bbox    image_zoom_in_tool, image_crop_tool          -> bbox_2d, IoU
  point   color_at_tool                                -> point_2d, distance
  color   find_color_tool, count_objects_tool          -> color/tolerance/min_size
  pair    image_diff_tool, image_overlay_tool          -> img_idx_a/img_idx_b (+alpha)
  multi   image_side_by_side_tool                      -> img_indices
  scalar  image_rotate_tool, image_flip_tool           -> degrees / axis
"""

import json
import logging
from enum import Enum
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.image_tools.task_data import TaskData


# Reuse the exact parser and IoU the image-tools agent uses at rollout time, so
# the verifier can never disagree with the agent about what a tool call *is*.
#
# Two import paths on purpose. Under pytest this module is imported as
# ``resources_servers.image_tools.app``, so the package-relative import is the
# right one. As a server, Gym runs this file as a script, and a nemo-gym wheel
# built inside the training container can omit newly added asset trees -- the
# setuptools-scm file finder needs a working git checkout, and the submodule's
# gitdir is not mounted there. Job 1710074 died on exactly that
# (ModuleNotFoundError: No module named 'resources_servers.image_tools'). In
# script context sys.path[0] is this directory, so base.py resolves regardless
# of what the wheel contains.
try:
    from .base import bbox_iou, coerce_bbox, parse_image_tool_calls
except ImportError:
    from base import bbox_iou, coerce_bbox, parse_image_tool_calls


logger = logging.getLogger(__name__)

# Counter for sampling unparsed generations in the diagnostic log path.
_UNPARSED_SEEN = 0


# ---------------------------------------------------------------------------
# Tool families
# ---------------------------------------------------------------------------

BBOX_TOOLS = frozenset({"image_zoom_in_tool", "image_crop_tool"})
POINT_TOOLS = frozenset({"color_at_tool"})
COLOR_TOOLS = frozenset({"find_color_tool", "count_objects_tool"})
PAIR_TOOLS = frozenset({"image_diff_tool", "image_overlay_tool"})
MULTI_TOOLS = frozenset({"image_side_by_side_tool"})
SCALAR_TOOLS = frozenset({"image_rotate_tool", "image_flip_tool"})

# Free-text / cosmetic arguments that must never gate the reward.
IGNORED_ARGS = frozenset({"label", "labels"})

# Tools that take the same decision and differ only by a parameter this verifier
# already ignores. image_crop_tool IS image_zoom_in_tool at factor 1 -- the task
# system prompt says so verbatim ("Like image_zoom_in_tool but without
# magnification") -- and `factor` is in IGNORED_ARGS. Scoring zoom-vs-crop as a
# total failure while treating factor as irrelevant *within* zoom is
# self-contradictory, and it penalises a model that picked the right region.
#
# Only this pair qualifies. Do NOT collapse whole tool_family() groups: those
# exist to dispatch argument comparison, not to assert action equivalence.
# find_color_tool vs count_objects_tool (locate vs count) and image_rotate_tool
# vs image_flip_tool (different transforms) are genuinely different decisions.
REGION_INSPECT_TOOLS = frozenset({"image_zoom_in_tool", "image_crop_tool"})
_REGION_INSPECT_CANON = "<region_inspect>"


def tool_family(name: str) -> str:
    if name in BBOX_TOOLS:
        return "bbox"
    if name in POINT_TOOLS:
        return "point"
    if name in COLOR_TOOLS:
        return "color"
    if name in PAIR_TOOLS:
        return "pair"
    if name in MULTI_TOOLS:
        return "multi"
    if name in SCALAR_TOOLS:
        return "scalar"
    return "unknown"


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_point(value: Any) -> Optional[list[float]]:
    """Parse a point_2d argument into [x, y]."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def _coerce_index_list(value: Any) -> Optional[list[int]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, int):
        return [value]
    if not isinstance(value, (list, tuple)):
        return None
    out = []
    for v in value:
        iv = _as_int(v)
        if iv is None:
            return None
        out.append(iv)
    return out


def _coerce_color(value: Any) -> Optional[list[int]]:
    """Colors appear as [r, g, b] lists or as names/hex strings."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        value = parsed
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    out = []
    for v in value:
        iv = _as_int(v)
        if iv is None:
            return None
        out.append(iv)
    return out


# ---------------------------------------------------------------------------
# Verification levels
# ---------------------------------------------------------------------------


def canonical_tool_name(name: str, unify_region_inspect: bool = True) -> str:
    """Map interchangeable tools onto a shared key for Level 0 comparison."""
    stripped = name.strip()
    if unify_region_inspect and stripped in REGION_INSPECT_TOOLS:
        return _REGION_INSPECT_CANON
    return stripped


def verify_tool_name_match(rollout_name: str, expected_name: str, unify_region_inspect: bool = True) -> bool:
    """Level 0: same tool, treating zoom/crop as one decision."""
    return canonical_tool_name(rollout_name, unify_region_inspect) == canonical_tool_name(
        expected_name, unify_region_inspect
    )


def verify_target_match(family: str, r_args: dict, e_args: dict) -> bool:
    """Level 1: is the call operating on the same image(s)?"""
    if family in ("bbox", "point", "color", "scalar"):
        r_idx = _as_int(r_args.get("img_idx"))
        e_idx = _as_int(e_args.get("img_idx"))
        if r_idx is None or e_idx is None:
            # Missing on either side: cannot disprove, defer to Level 2.
            return True
        return r_idx == e_idx

    if family == "pair":
        for key in ("img_idx_a", "img_idx_b"):
            r_v = _as_int(r_args.get(key))
            e_v = _as_int(e_args.get(key))
            if r_v is None or e_v is None:
                continue
            if r_v != e_v:
                return False
        return True

    if family == "multi":
        r_ids = _coerce_index_list(r_args.get("img_indices"))
        e_ids = _coerce_index_list(e_args.get("img_indices"))
        if r_ids is None or e_ids is None:
            return True
        return sorted(r_ids) == sorted(e_ids)

    return True


def compute_argument_score(
    family: str,
    r_args: dict,
    e_args: dict,
    numeric_tolerance: float,
) -> tuple[float, str]:
    """Level 2: score the primary argument. Returns (score in [0,1], detail)."""
    if family == "bbox":
        r_box = coerce_bbox(r_args.get("bbox_2d"))
        e_box = coerce_bbox(e_args.get("bbox_2d"))
        if e_box is None:
            return 1.0, "expected_bbox_unparseable"
        if r_box is None:
            return 0.0, "rollout_bbox_unparseable"
        # Reproducing the demonstrated box exactly must always score 1.0.
        # Some demonstrations carry a degenerate box (e.g. [167, 1000, 1000,
        # 1000], zero height); IoU of a zero-area box with itself is 0/0 -> 0.0,
        # which would make those pivots unwinnable and feed the group nothing
        # but zeros. Short-circuit on exact equality before computing IoU.
        if r_box == e_box:
            return 1.0, "exact_bbox"
        return bbox_iou(r_box, e_box), "iou"

    if family == "point":
        r_pt = _coerce_point(r_args.get("point_2d"))
        e_pt = _coerce_point(e_args.get("point_2d"))
        if e_pt is None:
            return 1.0, "expected_point_unparseable"
        if r_pt is None:
            return 0.0, "rollout_point_unparseable"
        # Coordinates are normalized 0-1000; express closeness on that scale.
        dist = ((r_pt[0] - e_pt[0]) ** 2 + (r_pt[1] - e_pt[1]) ** 2) ** 0.5
        return max(0.0, 1.0 - dist / 1000.0), "point_distance"

    if family == "color":
        r_col = _coerce_color(r_args.get("color"))
        e_col = _coerce_color(e_args.get("color"))
        if e_col is None or r_col is None:
            # Fall back to comparing whatever raw values are present.
            return (1.0, "color_absent") if r_args.get("color") == e_args.get("color") else (0.0, "color_mismatch")
        # Channel-wise closeness in 0-255 space.
        diff = sum(abs(a - b) for a, b in zip(r_col, e_col)) / (3 * 255.0)
        return max(0.0, 1.0 - diff), "color_distance"

    if family == "scalar":
        for key in ("degrees", "axis"):
            if key not in e_args:
                continue
            e_v = e_args.get(key)
            r_v = r_args.get(key)
            e_f, r_f = _as_float(e_v), _as_float(r_v)
            if e_f is not None and r_f is not None:
                # Rotations are modular: 350 and -10 are the same action.
                if key == "degrees":
                    delta = abs((r_f - e_f) % 360.0)
                    delta = min(delta, 360.0 - delta)
                    return (1.0 if delta <= numeric_tolerance else 0.0), "degrees"
                return (1.0 if abs(r_f - e_f) <= numeric_tolerance else 0.0), key
            return (1.0 if str(r_v).strip().lower() == str(e_v).strip().lower() else 0.0), key
        return 1.0, "scalar_no_key"

    if family in ("pair", "multi"):
        # Fully determined by Level 1, except overlay alpha.
        e_alpha, r_alpha = _as_float(e_args.get("alpha")), _as_float(r_args.get("alpha"))
        if e_alpha is not None and r_alpha is not None:
            return (1.0 if abs(r_alpha - e_alpha) <= numeric_tolerance else 0.0), "alpha"
        return 1.0, "index_only"

    return 1.0, "unknown_family"


# ---------------------------------------------------------------------------
# Failure codes
# ---------------------------------------------------------------------------


class FailureCode(str, Enum):
    NONE = "none"
    EXPECTED_ACTION_INVALID = "expected_action_invalid"
    NO_TOOL_CALL_IN_ROLLOUT = "no_tool_call_in_rollout"
    TOOL_NAME_MISMATCH = "tool_name_mismatch"
    TARGET_MISMATCH = "target_mismatch"
    ARGUMENT_BELOW_THRESHOLD = "argument_below_threshold"
    UNKNOWN_ERROR = "unknown_error"


# ---------------------------------------------------------------------------
# Server config / IO models
# ---------------------------------------------------------------------------


class ImageToolsPivotResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "image_tools_pivot"
    # "binary" -> 1.0 iff score >= threshold; "continuous" -> score itself.
    reward_mode: str = "binary"
    enable_target_match: bool = True
    # IoU / point / color score required for full credit in binary mode.
    argument_threshold: float = 0.5
    numeric_tolerance: float = 1e-6
    # Reward when the rollout emits several calls but the first matches.
    penalize_extra_tool_calls: bool = True
    # Treat image_zoom_in_tool and image_crop_tool as the same decision at
    # Level 0. Set false to require the exact tool name.
    unify_region_inspect_tools: bool = True
    # Log the raw generation when no tool call parses. Off by default: it is a
    # diagnostic for "why is NO_TOOL_CALL_IN_ROLLOUT so high", not something to
    # carry in a training run.
    log_unparsed_generations: bool = False
    log_unparsed_every_n: int = 50


class ImageToolsPivotRunRequest(TaskData, BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ImageToolsPivotVerifyRequest(ImageToolsPivotRunRequest, BaseVerifyRequest):
    pass


class ImageToolsPivotVerifyResponse(BaseVerifyResponse):
    uuid: Optional[str | int] = None
    expected_action: Optional[dict[str, Any]] = None
    model_output: str = ""
    tool_name_match: bool = False
    # Recorded so tool-selection errors can be read as a confusion matrix
    # instead of a single aggregate mismatch count.
    rollout_tool_name: Optional[str] = None
    expected_tool_name: Optional[str] = None
    target_match: Optional[bool] = None
    argument_score: Optional[float] = None
    argument_detail: Optional[str] = None
    tool_family: Optional[str] = None
    num_rollout_tool_calls: int = 0
    failure_reason: Optional[FailureCode] = None
    metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_assistant_text(response: Any) -> str:
    """Concatenate assistant text from a Responses API payload."""
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        content = getattr(item, "content", None)
        if isinstance(content, str):
            parts.append(content)
            continue
        if isinstance(content, list):
            for chunk in content:
                text = getattr(chunk, "text", None)
                if text is None and isinstance(chunk, dict):
                    text = chunk.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts)


def extract_expected_action(body: ImageToolsPivotRunRequest) -> Optional[dict]:
    """Expected action is {"name": ..., "arguments": {...}} on the row."""
    candidates = []
    if body.expected_action:
        candidates.append(body.expected_action)
    if body.metadata:
        candidates.append(body.metadata.get("expected_action"))
    if body.expected_answer:
        try:
            candidates.append(json.loads(body.expected_answer))
        except (json.JSONDecodeError, TypeError):
            pass

    for cand in candidates:
        if isinstance(cand, dict) and cand.get("name"):
            args = cand.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return {"name": str(cand["name"]), "arguments": args if isinstance(args, dict) else {}}
    return None


def _strip_ignored(args: dict) -> dict:
    return {k: v for k, v in args.items() if k not in IGNORED_ARGS}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class ImageToolsPivotResourcesServer(SimpleResourcesServer):
    config: ImageToolsPivotResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(self, body: ImageToolsPivotVerifyRequest) -> ImageToolsPivotVerifyResponse:
        state: dict[str, Any] = {
            "reward": 0.0,
            "tool_name_match": False,
            "rollout_tool_name": None,
            "expected_tool_name": None,
            "target_match": None,
            "argument_score": None,
            "argument_detail": None,
            "tool_family": None,
            "num_rollout_tool_calls": 0,
            "failure_reason": FailureCode.NONE,
            "model_output": "",
        }

        try:
            expected = extract_expected_action(body)
            if expected is None:
                state["failure_reason"] = FailureCode.EXPECTED_ACTION_INVALID
                return self._build(body, state)

            text = extract_assistant_text(body.response)
            rollout_calls = parse_image_tool_calls(text)
            state["num_rollout_tool_calls"] = len(rollout_calls)

            if not rollout_calls:
                state["failure_reason"] = FailureCode.NO_TOOL_CALL_IN_ROLLOUT
                state["model_output"] = text[-500:]
                # Diagnostic: when nothing parses, the raw generation is the
                # only thing that distinguishes "wrong tool-call format" from
                # "tool call extracted into structured output, leaving empty
                # text" from "the model genuinely did not attempt a call".
                # Sampled, not every rollout, to keep the driver log usable.
                # Sample by counter, not by uuid: these rows carry uuid=None,
                # so hashing it would log all or nothing.
                global _UNPARSED_SEEN
                _UNPARSED_SEEN += 1
                if (
                    self.config.log_unparsed_generations
                    and _UNPARSED_SEEN % max(1, self.config.log_unparsed_every_n) == 1
                ):
                    logger.warning(
                        "image_tools_pivot UNPARSED len=%d head=%r tail=%r",
                        len(text),
                        text[:400],
                        text[-400:],
                    )
                return self._build(body, state)

            # The demonstrated pivot is a single action; score the first call.
            rollout = rollout_calls[0]
            r_name = str(rollout.get("name", ""))
            r_args = _strip_ignored(rollout.get("arguments") or {})
            e_name = expected["name"]
            e_args = _strip_ignored(expected["arguments"])

            state["model_output"] = f"{r_name}({json.dumps(r_args)[:400]})"
            state["rollout_tool_name"] = r_name
            state["expected_tool_name"] = e_name
            family = tool_family(e_name)
            state["tool_family"] = family

            # --- Level 0 ---
            state["tool_name_match"] = verify_tool_name_match(r_name, e_name, self.config.unify_region_inspect_tools)
            if not state["tool_name_match"]:
                state["failure_reason"] = FailureCode.TOOL_NAME_MISMATCH
                return self._build(body, state)

            # --- Level 1 ---
            if self.config.enable_target_match:
                state["target_match"] = verify_target_match(family, r_args, e_args)
                if not state["target_match"]:
                    state["failure_reason"] = FailureCode.TARGET_MISMATCH
                    return self._build(body, state)

            # --- Level 2 ---
            score, detail = compute_argument_score(family, r_args, e_args, self.config.numeric_tolerance)
            state["argument_score"] = score
            state["argument_detail"] = detail

            if self.config.reward_mode == "binary":
                if score >= self.config.argument_threshold:
                    state["reward"] = 1.0
                else:
                    state["reward"] = 0.0
                    state["failure_reason"] = FailureCode.ARGUMENT_BELOW_THRESHOLD
            else:
                state["reward"] = score
                if score < self.config.argument_threshold:
                    state["failure_reason"] = FailureCode.ARGUMENT_BELOW_THRESHOLD

            # A pivot is one decision: emitting a burst of calls is off-policy
            # behaviour even when the first one is right.
            if self.config.penalize_extra_tool_calls and len(rollout_calls) > 1:
                state["reward"] = 0.0

        except Exception as e:  # noqa: BLE001 - must never crash the rollout
            logger.error(f"image_tools_pivot verify error: {type(e).__name__} {e}")
            state["reward"] = 0.0
            state["failure_reason"] = FailureCode.UNKNOWN_ERROR

        return self._build(body, state)

    def _build(self, body: ImageToolsPivotVerifyRequest, state: dict[str, Any]) -> ImageToolsPivotVerifyResponse:
        logger.info(
            f"image_tools_pivot | uuid={body.uuid} reward={state['reward']:.3f} "
            f"family={state['tool_family']} name_match={state['tool_name_match']} "
            f"got={state['rollout_tool_name']} want={state['expected_tool_name']} "
            f"target={state['target_match']} arg={state['argument_score']} "
            f"failure={state['failure_reason']}"
        )
        return ImageToolsPivotVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=state["reward"],
            uuid=body.uuid,
            expected_action=body.expected_action,
            model_output=state["model_output"],
            tool_name_match=state["tool_name_match"],
            rollout_tool_name=state["rollout_tool_name"],
            expected_tool_name=state["expected_tool_name"],
            target_match=state["target_match"],
            argument_score=state["argument_score"],
            argument_detail=state["argument_detail"],
            tool_family=state["tool_family"],
            num_rollout_tool_calls=state["num_rollout_tool_calls"],
            failure_reason=state["failure_reason"],
            metadata=body.metadata,
        )


if __name__ == "__main__":
    ImageToolsPivotResourcesServer.run_webserver()
