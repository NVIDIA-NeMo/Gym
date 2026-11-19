# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nemotron OSWorld agent driven by a pinned canonical harness.

The harness is a directory holding ``mm_agents/nvidia/nemotron_agent.py`` (system
prompts and response parser) and, for execution feedback,
``desktop_env/execution_feedback.py``. Both files are loaded by path and checked
against sha256 pins, so a run names exactly the prompt and parser it trained with.

Observations use the RFC0015 layout: the first turn is ``Task: <instruction>`` and the
screenshot; later turns wrap the screenshot in ``<tool_response>`` after the rendered
execution feedback of the previous step. Every served assistant output is replayed;
screenshots follow the K/S window: the first image is pinned, and once there are more
than K images, the images from S-1 before the latest multiple of K onward are kept.
Each dropped screenshot is replaced in place by the omission marker.

Configure it through ``agent_kwargs`` of the ``nemotron_v3_nano_omni_agent`` runner::

    agent_class_path: responses_api_agents.osworld_agent.canonical_agent.CanonicalNemotronAgent
    agent_kwargs:
      canonical_harness:
        root: /path/to/harness
        agent_sha256: <sha256 of mm_agents/nvidia/nemotron_agent.py>
        feedback_sha256: <sha256 of desktop_env/execution_feedback.py>
        execution_feedback: true
      assistant_history_replay: raw_full
      legacy_response_parser: true
      context_overflow_policy: stop
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping

from responses_api_agents.osworld_agent.adapter_agents import (
    NemotronV3NanoOmniAgent,
    _encode_image,
    _response_parts,
)


LOG = logging.getLogger("nemo_gym.osworld_agent.canonical_agent")

DEFAULT_OMISSION_MARKER = "[Image omitted for text cleaning]"
_AGENT_SOURCE = Path("mm_agents") / "nvidia" / "nemotron_agent.py"
_FEEDBACK_SOURCE = Path("desktop_env") / "execution_feedback.py"
_SPECIAL_ACTIONS = frozenset({"WAIT", "FAIL", "DONE"})
_UNKNOWN_RESULT_PREFIX = 'Computer observation:\n<tool_response>\nresult:\n{"status": "unknown"}\n'
_HARNESS_KEYS = frozenset(
    {
        "root",
        "agent_sha256",
        "feedback_sha256",
        "execution_feedback",
        "image_block_k",
        "image_boundary_s",
        "omission_marker",
    }
)
_SHA256_HEX_LENGTH = 64


def normalize_canonical_harness(options: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate ``canonical_harness`` and return its contract form.

    The root path is deployment-specific and stays out of the contract; the pins
    identify the harness.
    """

    if not isinstance(options, Mapping):
        raise ValueError("canonical_harness must be a mapping")
    unknown = sorted(set(options) - _HARNESS_KEYS)
    if unknown:
        raise ValueError(f"Unknown canonical_harness setting(s): {unknown}")
    if not isinstance(options.get("root"), str) or not options["root"].strip():
        raise ValueError("canonical_harness.root must name the harness directory")
    execution_feedback = options.get("execution_feedback", False)
    if type(execution_feedback) is not bool:
        raise ValueError("canonical_harness.execution_feedback must be a boolean")
    pins = {"agent_sha256": options.get("agent_sha256")}
    if execution_feedback:
        pins["feedback_sha256"] = options.get("feedback_sha256")
    for field, pin in pins.items():
        if not isinstance(pin, str) or len(pin) != _SHA256_HEX_LENGTH or any(c not in "0123456789abcdef" for c in pin):
            raise ValueError(f"canonical_harness.{field} must be a lowercase sha256 hex digest")
    block_k = options.get("image_block_k", 8)
    boundary_s = options.get("image_boundary_s", 3)
    if type(block_k) is not int or type(boundary_s) is not int or not 1 <= boundary_s <= block_k:
        raise ValueError("canonical_harness needs integers image_block_k >= image_boundary_s >= 1")
    omission_marker = options.get("omission_marker", DEFAULT_OMISSION_MARKER)
    if not isinstance(omission_marker, str) or not omission_marker:
        raise ValueError("canonical_harness.omission_marker must be a non-empty string")
    return {
        **pins,
        "execution_feedback": execution_feedback,
        "image_block_k": block_k,
        "image_boundary_s": boundary_s,
        "omission_marker": omission_marker,
    }


def retained_image_indices(image_count: int, *, block_k: int, boundary_s: int) -> frozenset[int]:
    """Indices of the screenshots the K/S window keeps among ``image_count``."""

    if image_count <= block_k:
        return frozenset(range(image_count))
    boundary = ((image_count - 1) // block_k) * block_k
    window_start = max(1, boundary - (boundary_s - 1))
    return frozenset({0, *range(window_start, image_count)})


def load_pinned_module(path: Path, module_name: str, expected_sha256: str) -> Any:
    source = path.read_bytes()
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Canonical OSWorld source digest mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    cached = sys.modules.get(module_name)
    if cached is not None and getattr(cached, "__canonical_sha256__", None) == actual_sha256:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load canonical OSWorld source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    module.__canonical_sha256__ = actual_sha256
    LOG.info("Loaded canonical OSWorld source %s (sha256=%s)", path, actual_sha256)
    return module


class CanonicalNemotronAgent(NemotronV3NanoOmniAgent):
    """``NemotronV3NanoOmniAgent`` with the canonical harness's prompt, parser and feedback."""

    def __init__(self, *, canonical_harness: Mapping[str, Any], **kwargs: Any) -> None:
        harness = normalize_canonical_harness(canonical_harness)
        super().__init__(**kwargs)
        if self.assistant_history_replay != "raw_full":
            raise ValueError("CanonicalNemotronAgent replays served outputs and requires raw_full")
        if not self.legacy_response_parser:
            raise ValueError("CanonicalNemotronAgent parses with the harness parser; set legacy_response_parser")
        if self.history_layout != "osworld":
            raise ValueError("CanonicalNemotronAgent renders its own history; leave history_layout unset")
        if self.context_overflow_policy != "stop":
            raise ValueError("CanonicalNemotronAgent keeps its K/S window; set context_overflow_policy='stop'")
        root = Path(canonical_harness["root"])
        agent_module = load_pinned_module(
            root / _AGENT_SOURCE, "_nemo_gym_canonical_nemotron_agent", harness["agent_sha256"]
        )
        self._feedback = (
            load_pinned_module(
                root / _FEEDBACK_SOURCE, "_nemo_gym_canonical_execution_feedback", harness["feedback_sha256"]
            )
            if harness["execution_feedback"]
            else None
        )
        self._parser = agent_module.parse_response_to_cot_and_action
        self.system_prompt = (
            agent_module.SYSTEM_PROMPT_THINKING if self.thinking else agent_module.SYSTEM_PROMPT_NON_THINKING
        ).replace("{password}", self.client_password)
        self.block_k = harness["image_block_k"]
        self.boundary_s = harness["image_boundary_s"]
        self.omission_marker = harness["omission_marker"]
        self.behavior_options["canonical_harness"] = harness

    def reset(self, _logger: logging.Logger | None = None, **_kwargs: Any) -> None:
        super().reset(_logger, **_kwargs)
        # Feedback text shown in each turn's observation, and the execution results
        # recorded since the last observation was rendered.
        self.turn_feedback: List[str] = [""]
        self._pending_results: List[Any] = []
        self._model_commands: Deque[str] = deque()

    def predict(self, instruction: str, obs: Dict[str, Any], **kwargs: Any) -> tuple[str, List[str], Dict[str, Any]]:
        response, actions, info = super().predict(instruction, obs, **kwargs)
        self._model_commands = deque(
            action for action in actions if isinstance(action, str) and action not in _SPECIAL_ACTIONS
        )
        return response, actions, info

    def record_execution_result(self, command: str, result: Any) -> None:
        """Called by the runner with each executed command's guest ``/execute`` reply."""

        # Feedback carries the hash of the command the model wrote; DesktopEnv may
        # rewrite it before execution (its pyautogui '<' fix), so use the model's own.
        model_command = self._model_commands.popleft() if self._model_commands else command
        if self._feedback is not None and result is not None:
            self._pending_results.append(
                self._feedback.normalize_execution_result(result, model_command=model_command)
            )

    def _legacy_parse(self, response: Any) -> tuple[str, List[str], Dict[str, Any]]:
        raw_content = response.get("raw_content") if isinstance(response, Mapping) else None
        if isinstance(raw_content, str) and raw_content.startswith("<think>") and "</think>" in raw_content:
            reasoning, _, content = raw_content[len("<think>") :].partition("</think>")
        else:
            content, reasoning = _response_parts(response)
        return self._parser(
            {"content": content, "reasoning_content": reasoning},
            self.screen_size,
            self.coordinate_type,
            thinking=self.thinking,
        )

    def _feedback_for_turn(self, turn: int) -> str:
        if len(self.turn_feedback) == turn:
            text = self._feedback.render_feedback_batch(self._pending_results) if self._pending_results else ""
            self._pending_results = []
            self.turn_feedback.append(text)
        return self.turn_feedback[turn]

    def _observation(self, turn: int, screenshot: bytes, instruction: str, *, keep_image: bool) -> Dict[str, Any]:
        image: Dict[str, Any] = (
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_image(screenshot)}"}}
            if keep_image
            else {"type": "text", "text": self.omission_marker}
        )
        if turn == 0:
            return {"role": "user", "content": [{"type": "text", "text": f"Task: {instruction}\n"}, image]}
        feedback = self._feedback_for_turn(turn).strip() if self._feedback is not None else ""
        prefix = f"Computer observation:\n{feedback}\n<tool_response>\n" if feedback else _UNKNOWN_RESULT_PREFIX
        return {
            "role": "user",
            "content": [{"type": "text", "text": prefix}, image, {"type": "text", "text": "\n</tool_response>"}],
        }

    def _messages(
        self,
        instruction: str,
        obs: Dict[str, Any],
        *,
        max_images: int | None = None,
    ) -> List[Dict[str, Any]]:
        if max_images is not None:
            raise ValueError("CanonicalNemotronAgent does not shrink its K/S window")
        turn = len(self.actions)
        retained = retained_image_indices(turn + 1, block_k=self.block_k, boundary_s=self.boundary_s)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        for index in range(turn):
            messages.append(
                self._observation(
                    index, self.observations[index]["screenshot"], instruction, keep_image=index in retained
                )
            )
            messages.append({"role": "assistant", "content": self._assistant_history(self.cots[index])})
        messages.append(self._observation(turn, obs["screenshot"], instruction, keep_image=True))

        previous_count = int(self.snapshot_window.get("prompt_snapshot_count") or 0)
        compacted = previous_count > 0 and len(retained) < previous_count + 1
        epoch = int(self.snapshot_window.get("history_policy_compaction_epoch") or 0)
        intervals: List[List[int]] = []
        for index in sorted(retained):
            if intervals and intervals[-1][1] == index:
                intervals[-1][1] = index + 1
            else:
                intervals.append([index, index + 1])
        self.compacted_turns = tuple(index for index in range(turn + 1) if index not in retained)
        self.snapshot_window = {
            "prompt_snapshot_count": len(retained),
            "snapshot_window_start": min(retained - {0}, default=0),
            "snapshot_image_intervals": intervals,
            "snapshot_sink_size": 1,
            "snapshot_image_budget_clamped": False,
            "snapshot_compaction_triggered": compacted,
            "snapshot_window_min": self.boundary_s,
            "snapshot_window_max": self.block_k + self.boundary_s,
            "history_policy_id": f"canonical_k{self.block_k}_s{self.boundary_s}",
            "history_policy_name": "canonical_k_s",
            "history_policy_schema_version": 1,
            "history_policy_compaction_epoch": epoch + int(compacted),
        }
        return messages
