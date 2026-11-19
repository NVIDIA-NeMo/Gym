# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V4-only agent using prompt_format_repo's producer functions directly.

Commit: 35d35a7c38b356ff1549feda11572a4bceda4e49
The reference remains an external dependency; no external implementation is
copied here. Gym transport/execution declarations are separate from that
producer's prompt-format identity.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .coordinate_compat import COMPAT_PROTOCOL_ID, COORDINATE_POLICIES, STEP1500_COORDINATES, STRICT_COORDINATES
from .protocol import PROTOCOL_ID, Action, parse_action
from .reference import Reference, file_sha256
from .session import CURRENT_SESSION


FEEDBACK_KEY = "nemotron_v4_execution_feedback"


def stable_id(prefix: str, value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return prefix + "-" + hashlib.sha256(raw).hexdigest()[:24]


class ModelProtocol:
    # These two flags wire Gym's existing audit capture and safe Python-suite
    # execution. The model consumes the v4 environment-owned canonical payload,
    # not the legacy tool_execution_feedback audit record.
    execution_feedback = True
    python_execution = "exec"
    termination_tool_name = "computer.terminate"
    protocol_id = PROTOCOL_ID

    def __init__(
        self, reference: Reference, password: str = "password", coordinate_policy: str = STRICT_COORDINATES
    ) -> None:
        self.reference = reference
        self.password = password
        self.coordinate_policy = coordinate_policy
        self.protocol_id = PROTOCOL_ID if coordinate_policy == STRICT_COORDINATES else COMPAT_PROTOCOL_ID

    def to_contract(self, **_kwargs: Any) -> dict[str, Any]:
        root = Path(__file__).resolve().parent
        implementation = {
            str(path.relative_to(root)): file_sha256(path)
            for path in [
                *sorted(root.glob("*.py")),
                root / "templates/reference_chatml.jinja",
            ]
        }
        return {
            "protocol_id": self.protocol_id,
            "reference": self.reference.identity(),
            "system_prompt_sha256": hashlib.sha256(self.reference.system_prompt(self.password).encode()).hexdigest(),
            "tools": self.reference.schemas(),
            "coordinate_projection": "round(unit_coordinate * (screen_axis_size - 1))"
            if self.coordinate_policy == STRICT_COORDINATES
            else "VM-side unit-first; explicit space override; checkpoint grid1000 prior; native pixels outside grid; complete per-call audit",
            "coordinate_policy": self.coordinate_policy,
            "coordinate_policy_authority": "reference producer"
            if self.coordinate_policy == STRICT_COORDINATES
            else "User-authorized model compatibility; observed step1500 grid output and one actual training-mixture row; not claimed to be the frozen training protocol",
            "execution_feedback": "command_execution_feedback.v1",
            "feedback_owner": "responses_api_agents.osworld_agent.nano4.environment",
            "runtime_feedback_mapping": {
                "model_command_sha256": "SHA256 of the original UTF-8 code_block, before pixel projection",
                "failure_signature": ["model_command_sha256", "status", "returncode", "stderr", "message"],
                "failure_count": "1 for first failure; increment identical consecutive signatures; reset to 0 on success",
                "missing_or_invalid_returncode": "protocol_error",
                "authority": "explicit environment mapping; producer validates schema/text but does not define this backend mapping",
            },
            "message_renderer": "pinned_converter_build_conversations_and_tool_observation_text",
            "serving_template": "reference_chatml.jinja",
            "v4_implementation_sha256": implementation,
            "preserve_response": True,
            "python_execution": "exec",
        }


class Nano4Agent:
    def __init__(
        self,
        model: str,
        max_steps: int,
        *,
        reference_root: str | None = None,
        model_protocol_id: str | None = None,
        history_policy: dict[str, Any] | None = None,
        max_tokens: int = 16384,
        temperature: float = 0.6,
        top_p: float | None = 0.95,
        screen_size: tuple[int, int] = (1920, 1080),
        actions_per_chunk: int = 10,
        parse_retries: int = 5,
        coordinate_policy: str = STRICT_COORDINATES,
        parse_error_feedback: bool | None = None,
        client_password: str = "password",
        platform: str = "ubuntu",
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        log_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise ValueError(f"Unsupported v4 agent options; no legacy defaults are inherited: {sorted(kwargs)}")
        if coordinate_policy not in COORDINATE_POLICIES:
            raise ValueError(f"Unknown v4 coordinate policy: {coordinate_policy}")
        selected_protocol = PROTOCOL_ID if coordinate_policy == STRICT_COORDINATES else COMPAT_PROTOCOL_ID
        if model_protocol_id not in {None, selected_protocol}:
            raise ValueError("The v4 entrypoint protocol differs from its selected coordinate policy")
        if platform != "ubuntu" or action_space != "pyautogui" or observation_type != "screenshot":
            raise ValueError("The selected reference profile uses Ubuntu screenshot/PyAutoGUI observations")
        if not isinstance(actions_per_chunk, int) or isinstance(actions_per_chunk, bool) or actions_per_chunk < 1:
            raise ValueError("actions_per_chunk must be a positive integer")
        if len(screen_size) != 2 or any(type(v) is not int or v < 1 for v in screen_size):
            raise ValueError("screen_size must contain two positive integer dimensions")
        if type(parse_retries) is not int or parse_retries < 1:
            raise ValueError("parse_retries must be a positive integer")
        self.reference = Reference(reference_root)
        self.model_protocol = ModelProtocol(self.reference, client_password, coordinate_policy)
        self.model_protocol_id = selected_protocol
        self.coordinate_policy = coordinate_policy
        self.parse_error_feedback = (
            coordinate_policy == STEP1500_COORDINATES if parse_error_feedback is None else parse_error_feedback
        )
        if type(self.parse_error_feedback) is not bool:
            raise ValueError("parse_error_feedback must be a boolean")
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.screen_size = tuple(screen_size)
        self.password = client_password
        self.actions_per_chunk = actions_per_chunk
        self.parse_retries = parse_retries
        self.log_context = dict(log_context or {})
        self.history_contract = {
            "name": "reference_action_block_images",
            "params": {"actions_per_chunk": actions_per_chunk, "boundary_image_window": 1},
            "authority": self.reference.lock["commit"],
        }
        self.history_contract["history_policy_id"] = stable_id("v4-history", self.history_contract)
        if history_policy is not None and history_policy != self.history_contract:
            raise ValueError("Configured v4 history differs from the pinned action-block policy")
        self.behavior_options = {
            "thinking": True,
            "coordinate_type": "relative" if coordinate_policy == STRICT_COORDINATES else coordinate_policy,
            "screen_size": list(self.screen_size),
            "platform": platform,
            "action_space": action_space,
            "observation_type": observation_type,
            "parse_attempts": parse_retries,
            "parse_feedback_injection": self.parse_error_feedback,
            "prompt_shrinking": False,
            "runtime_options_authority": "prior evaluation attempt budget retained; not specified by the SFT producer",
        }
        self.reset()

    @property
    def agent_contract(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_protocol": self.model_protocol.to_contract(),
            "history_policy": copy.deepcopy(self.history_contract),
            "agent_options": dict(self.behavior_options),
        }

    def reset(self, logger: logging.Logger | None = None, **_kwargs: Any) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.observations: list[dict[str, Any]] = []
        self.reference_actions: list[Any] = []
        self.actions: list[Action] = []
        self._session = CURRENT_SESSION.get(None)
        if self._session is not None:
            self._session.reference = self.reference
            self._session.pending_action = None
            self._session.previous_failure_signature = None
            self._session.consecutive_identical_failure_count = 0

    def call_llm(self, payload: dict[str, Any], _model: str | None = None) -> Any:
        raise RuntimeError("Inject the Gym messages transport before calling predict")

    def _reference_action(self, action: Action) -> Any:
        terminal = action.name == "computer.terminate"
        return self.reference.converter.NormalizedAction(
            index=len(self.reference_actions),
            source={},
            screenshot_after=None,
            action_type=("done" if action.arguments["status"] == "success" else "fail") if terminal else "code",
            thought=action.reasoning,
            note="",
            action_text=action.summary or action.arguments.get("answer", ""),
            code=action.model_source if action.name == "exec_py" else action.desktop_action,
            source_family="canonical",  # Already-canonical runtime action, not relabeled Kimi/Holotron data.
            source_tool_call={"tool_name": action.name, **action.arguments},
        )

    def _receive_feedback(self, observation: dict[str, Any]) -> None:
        if not self.reference_actions:
            return
        previous = self.reference_actions[-1]
        if not self.reference.converter.action_requires_execution_feedback(previous):
            previous.source["execution_status"] = observation.get("nemotron_v4_control_status", "unknown")
            return
        feedback = observation.get(FEEDBACK_KEY)
        if not isinstance(feedback, dict):
            raise ValueError("V4 requires environment-owned command_execution_feedback.v1 before the next action")
        expected_hash = hashlib.sha256(self.actions[-1].model_source.encode()).hexdigest()
        if feedback["execution_feedback"].get("model_command_sha256") != expected_hash:
            raise ValueError("Execution feedback does not belong to the preceding model command")
        previous.source.update(copy.deepcopy(feedback))
        previous.source["execution_feedback_delivery"] = "delivered"
        # The producer checks exact keys, statuses, types, truncation bytes and
        # causal delivery. It is the authority, rather than a parallel validator.
        self.reference.converter.execution_feedback_text(previous)

    def canonical_prefix(self, instruction: str) -> tuple[list[dict[str, Any]], list[int]]:
        """Render already-observed actions through the producer's own functions."""
        count = len(self.actions)
        paths = [Path("/v4-observations") / f"{i}.png" for i in range(count + 1)]
        block_start = (count // self.actions_per_chunk) * self.actions_per_chunk
        markers, retained, _omitted = self.reference.converter.build_image_markers(
            [*paths, None],
            block_start,
            count + 1,
            1,
            Path("/v4-observations"),
            None,
        )
        turns = self.reference.converter.build_conversations(
            instruction,
            self.reference_actions,
            block_start,
            count,
            markers,
            None,
            self.reference.system_prompt(self.password),
            self.reference.profile,
        )
        if count:
            turns.append(
                {
                    "from": "human",
                    "value": self.reference.converter.tool_observation_text(
                        markers[count],
                        self.reference_actions[-1],
                        self.reference.profile,
                    ),
                }
            )
        image_turns = [int(Path(path).stem) for path in retained]
        return turns, image_turns

    def conversations(self, instruction: str, observation: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
        self._receive_feedback(observation)
        return self.canonical_prefix(instruction)

    def messages(self, instruction: str, observation: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        turns, image_turns = self.conversations(instruction, observation)
        observations = [*self.observations, observation]
        messages: list[dict[str, Any]] = []
        observation_index = 0
        for turn in turns:
            role = {"system": "system", "human": "user", "gpt": "assistant"}[turn["from"]]
            value = turn["value"]
            if role != "user":
                messages.append({"role": role, "content": value})
                continue
            if observation_index in image_turns:
                if value.count("<image>") != 1:
                    raise ValueError("The reference observation must contain exactly one media marker")
                # Keep an explicit runtime marker at the canonical location so
                # vLLM's string-content path cannot prepend it ahead of feedback.
                # <image> is the training cooker marker; <img> is the verified
                # Nemotron4 serving tokenizer alias at the same visual boundary.
                value = value.replace("<image>", "<img>")
                encoded = base64.b64encode(observations[observation_index]["screenshot"]).decode("ascii")
                content: Any = [
                    {"type": "text", "text": value},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
                ]
            else:
                content = value
            messages.append({"role": role, "content": content})
            observation_index += 1
        return messages, {
            "image_turns": image_turns,
            "image_count": len(image_turns),
            "block_start_action_index": (len(self.actions) // self.actions_per_chunk) * self.actions_per_chunk,
        }

    def predict(
        self, instruction: str, observation: dict[str, Any], **_kwargs: Any
    ) -> tuple[str, list[str], dict[str, Any]]:
        if self._session is not None and self._session.pending_action is not None:
            raise RuntimeError("The preceding v4 action has not been executed")
        messages, image_info = self.messages(instruction, observation)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tools": [{"type": "function", "function": schema} for schema in self.reference.schemas()],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "chat_template_kwargs": {"enable_thinking": True, "truncate_history_thinking": False},
            "skip_special_tokens": False,
            "_nemo_gym_return_message": True,
            "_nemo_gym_preserve_response": True,
            "_nemo_gym_require_stop": True,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        errors = []
        for attempt in range(self.parse_retries):
            # Keep the previous attempt budget without introducing an unowned
            # correction prompt into the reference conversation format.
            response = self.call_llm(payload, self.model)
            try:
                action = parse_action(
                    response,
                    reference=self.reference,
                    screen_size=self.screen_size,
                    coordinate_policy=self.coordinate_policy,
                )
                break
            except (ValueError, SyntaxError, TypeError) as exc:
                errors.append({"attempt": attempt + 1, "error": str(exc)})
                if self.parse_error_feedback and attempt + 1 < self.parse_retries:
                    rejected = (
                        response
                        if isinstance(response, str)
                        else json.dumps(response, ensure_ascii=False, default=str)
                    )
                    payload["messages"] = [
                        *payload["messages"],
                        {"role": "assistant", "content": rejected},
                        {
                            "role": "user",
                            "content": f"The previous response was not executed. Parser error: {exc}. Return one valid tool call. For exec_py prefer normalized [0,1] x/y; if you intend grid1000 or screenshot pixels, declare '# coordinate_space: grid1000' or '# coordinate_space: pixel' in the code. Screenshot size is {self.screen_size[0]}x{self.screen_size[1]}.",
                        },
                    ]
        else:
            return (
                errors[-1]["error"],
                [],
                {
                    "agent_outcome": "model_response_unparseable",
                    "stop_rollout": True,
                    "model_call_completed": True,
                    "parse_failure": {"attempt_count": len(errors), "all_errors": errors},
                    **image_info,
                },
            )
        self.observations.append(dict(observation))
        self.reference_actions.append(self._reference_action(action))
        self.actions.append(action)
        if self._session is not None:
            self._session.pending_action = action
        history = self.reference.converter.assistant_plain(
            self.reference_actions[-1], self.reference.profile.wait_tool_contract
        )
        return (
            history,
            [action.desktop_action],
            {
                "tool": action.name,
                "arguments": action.arguments,
                "original_code": action.model_source,
                "code": action.desktop_action,
                "canonical_assistant_history": history,
                "model_call_completed": True,
                "parse_attempts": attempt + 1,
                "previous_parse_errors": errors,
                "coordinate_policy": self.coordinate_policy,
                **image_info,
            },
        )
