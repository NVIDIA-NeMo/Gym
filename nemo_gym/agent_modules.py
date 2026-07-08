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
"""Agent module primitives.

Agent modules are internal Agent components. They may affect how the Agent
builds its turn context, and they may observe trajectory steps to update
Agent-owned artifacts. The Processor does not interpret these modules; it only
runs the normal interaction loop and routes typed trajectory steps to the Agent.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC
from collections import deque
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import BaseVerifyResponse
from nemo_gym.base_responses_api_agent import AgentArtifactRef, TrajectoryStep
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming


def _stable_hash(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(data).hexdigest()[:12]


class PromptArtifactModuleConfig(BaseModel):
    """Prompt artifact module.

    This is the artifact shape targeted by PR #1551-style DSPy/GEPA prompt
    optimization. The adaptive algorithm can update this module's prompt
    artifact; the Agent loop consumes it when constructing model input.
    """

    type: Literal["prompt_artifact"] = "prompt_artifact"
    name: str = "prompt"
    system_prompt: str
    prepend: bool = True


class GepaPromptModuleConfig(BaseModel):
    """GEPA-style reflective prompt adaptation module.

    This is a concrete AgentModule shape for PR #1551. It keeps the prompt as an
    Agent artifact, observes verified trajectory steps, and optionally calls a
    reflection model to evolve the prompt. It deliberately runs on the normal
    Processor `/run` path; it does not own rollout collection.

    A future implementation can swap the internal reflection/update loop for
    the DSPy `GEPA` teleprompter while preserving this module boundary.
    """

    type: Literal["gepa_prompt"] = "gepa_prompt"
    name: str = "gepa_prompt"
    system_prompt: str
    prepend: bool = True
    reflection_model: str | None = None
    reflection_base_url: str | None = None
    reflection_api_key: str | None = None
    min_observations: int = Field(default=8, ge=1)
    max_buffer_size: int = Field(default=64, ge=1)
    update_on_successes: bool = False


class AcePlaybookModuleConfig(BaseModel):
    """ACE-style playbook module.

    This is a small framework-level scaffold for PR #1706-style playbook state.
    A concrete ACE Agent can update the playbook from observations and inject it
    into context. The update algorithm remains Agent-owned.
    """

    type: Literal["ace_playbook"] = "ace_playbook"
    name: str = "playbook"
    items: list[str] = Field(default_factory=list)
    max_items: int = Field(default=64, ge=1)


AgentModuleConfig: TypeAlias = Annotated[
    PromptArtifactModuleConfig | GepaPromptModuleConfig | AcePlaybookModuleConfig,
    Field(discriminator="type"),
]


class AgentModule(ABC):
    """Base Agent module.

    Modules are intentionally small: they can alter Agent input, observe typed
    trajectory steps, and expose artifact refs for provenance when an Agent
    chooses to emit them elsewhere.
    """

    name: str

    def apply_to_responses_create_params(
        self, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        return body

    async def observe(self, step: TrajectoryStep) -> None:
        return None

    def artifact_refs(self) -> list[AgentArtifactRef]:
        return []


class PromptArtifactModule(AgentModule):
    def __init__(self, config: PromptArtifactModuleConfig):
        self.config = config
        self.name = config.name

    def apply_to_responses_create_params(
        self, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        if not self.config.prepend:
            return body

        body = body.model_copy(deep=True)
        current_input = body.input
        if isinstance(current_input, str):
            current_input = [NeMoGymEasyInputMessage(role="user", content=current_input)]

        body.input = [NeMoGymEasyInputMessage(role="system", content=self.config.system_prompt)] + list(current_input)
        return body

    def artifact_refs(self) -> list[AgentArtifactRef]:
        return [
            AgentArtifactRef(
                type="prompt",
                hash=_stable_hash({"system_prompt": self.config.system_prompt}),
                metadata={"name": self.name, "system_prompt": self.config.system_prompt},
            )
        ]


class GepaPromptModule(AgentModule):
    def __init__(self, config: GepaPromptModuleConfig):
        self.config = config
        self.name = config.name
        self.system_prompt = config.system_prompt
        self._observations: deque[BaseVerifyResponse] = deque(maxlen=config.max_buffer_size)
        self._updates = 0
        self._last_error: str | None = None

    def apply_to_responses_create_params(
        self, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        if not self.config.prepend:
            return body

        body = body.model_copy(deep=True)
        current_input = body.input
        if isinstance(current_input, str):
            current_input = [NeMoGymEasyInputMessage(role="user", content=current_input)]

        body.input = [NeMoGymEasyInputMessage(role="system", content=self.system_prompt)] + list(current_input)
        return body

    async def observe(self, step: TrajectoryStep) -> None:
        if step.kind not in {"terminated", "truncated"}:
            return None

        verify_response = self._coerce_verify_response(step.payload)
        if verify_response is None:
            return None
        if verify_response.reward > 0 and not self.config.update_on_successes:
            return None

        self._observations.append(verify_response)
        if len(self._observations) < self.config.min_observations:
            return None

        if not (self.config.reflection_model and self.config.reflection_base_url and self.config.reflection_api_key):
            return None

        try:
            candidate = await self._reflect_prompt()
        except Exception as exc:  # pragma: no cover - network/model dependent
            self._last_error = f"{type(exc).__name__}: {exc}"
            return None

        if candidate and candidate != self.system_prompt:
            self.system_prompt = candidate
            self._updates += 1
            self._observations.clear()
        return None

    def artifact_refs(self) -> list[AgentArtifactRef]:
        metadata = {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "updates": self._updates,
            "buffer_size": len(self._observations),
        }
        if self._last_error:
            metadata["last_error"] = self._last_error
        return [
            AgentArtifactRef(
                type="prompt",
                hash=_stable_hash({"system_prompt": self.system_prompt}),
                metadata=metadata,
            )
        ]

    @staticmethod
    def _coerce_verify_response(payload: object) -> BaseVerifyResponse | None:
        if isinstance(payload, BaseVerifyResponse):
            return payload
        if isinstance(payload, dict):
            try:
                return BaseVerifyResponse.model_validate(payload)
            except Exception:
                return None
        return None

    async def _reflect_prompt(self) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - dependency is required by project
            raise RuntimeError("GepaPromptModule requires the openai package.") from exc

        examples = []
        for item in self._observations:
            examples.append(
                {
                    "reward": item.reward,
                    "input": item.responses_create_params.model_dump(mode="json"),
                    "output_text": item.response.output_text[:4000],
                }
            )

        prompt = (
            "You are improving an instruction prompt for an agent. "
            "Given the current prompt and recent verified trajectories, return only a revised prompt. "
            "Keep all task-required output-format constraints explicit.\n\n"
            f"CURRENT PROMPT:\n{self.system_prompt}\n\n"
            f"RECENT TRAJECTORIES:\n{json.dumps(examples, indent=2)}\n\n"
            "REVISED PROMPT:"
        )

        client = AsyncOpenAI(api_key=self.config.reflection_api_key, base_url=self.config.reflection_base_url)
        response = await client.responses.create(
            model=self.config.reflection_model,
            input=[{"role": "user", "content": prompt}],
            temperature=1.0,
        )
        return response.output_text.strip()


class AcePlaybookModule(AgentModule):
    def __init__(self, config: AcePlaybookModuleConfig):
        self.config = config
        self.name = config.name
        self.items = list(config.items)

    def apply_to_responses_create_params(
        self, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        if not self.items:
            return body

        body = body.model_copy(deep=True)
        playbook = "\n".join(f"- {item}" for item in self.items)
        message = NeMoGymEasyInputMessage(
            role="system",
            content=f"ACE playbook strategies learned so far:\n{playbook}",
        )
        current_input = body.input
        if isinstance(current_input, str):
            current_input = [NeMoGymEasyInputMessage(role="user", content=current_input)]
        body.input = [message] + list(current_input)
        return body

    async def observe(self, step: TrajectoryStep) -> None:
        if step.kind != "custom":
            return None
        payload = step.payload if isinstance(step.payload, dict) else {}
        strategy = payload.get("ace_playbook_item")
        if not strategy or not isinstance(strategy, str):
            return None
        if strategy in self.items:
            return None
        self.items.append(strategy)
        if len(self.items) > self.config.max_items:
            self.items = self.items[-self.config.max_items :]
        return None

    def artifact_refs(self) -> list[AgentArtifactRef]:
        return [
            AgentArtifactRef(
                type="ace_playbook",
                hash=_stable_hash({"items": self.items}),
                metadata={"name": self.name, "num_items": len(self.items), "items": list(self.items)},
            )
        ]


def build_agent_modules(configs: list[AgentModuleConfig]) -> list[AgentModule]:
    modules: list[AgentModule] = []
    for config in configs:
        if isinstance(config, PromptArtifactModuleConfig):
            modules.append(PromptArtifactModule(config))
        elif isinstance(config, GepaPromptModuleConfig):
            modules.append(GepaPromptModule(config))
        elif isinstance(config, AcePlaybookModuleConfig):
            modules.append(AcePlaybookModule(config))
        else:  # pragma: no cover - discriminated union should prevent this
            raise TypeError(f"Unsupported agent module config: {config!r}")
    return modules
