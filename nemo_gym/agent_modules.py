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
"""Agent modules: behavior-shaping components on the Agent boundary.

Each module has a cognitive capability ``type`` discriminator (``working_memory``,
``skill_library``, ``planning``, …) and a shared lifecycle — ``activate``, ``adapt``,
``module_refs``. See research note ``agent-modules`` and
``issues/agent-modules-first-class-citizens.md``.

Run-level knobs (``prompt_config``, ``skills.path``) are migrating toward modules
declared on Agent config and activated inside ``/run``.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.global_config import SKILLS_REF_KEY_NAME
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.prompt import fill_prompt, load_prompt_config
from nemo_gym.skills import (
    SkillsRef,
    apply_skill_adaptation,
    format_skills_for_context,
    load_skill_bodies,
    load_skill_directory,
    stage_skills,
)


_HASH_PREFIX_LEN = 12

# Stamped on rollout results alongside legacy ``skills_ref``.
AGENT_MODULE_REFS_KEY = "agent_module_refs"
AGENT_UPDATE_EVENTS_KEY = "agent_update_events"

TrajectoryEventKind = Literal[
    "model_call",
    "tool_call",
    "tool_result",
    "verify",
    "terminated",
    "truncated",
]


class AgentModuleRef(BaseModel):
    """Provenance stamp for an active Agent module during a rollout."""

    type: str
    name: str
    hash: str
    path: Optional[str] = None
    uri: Optional[str] = None


class AgentUpdateEvent(BaseModel):
    """Structured before/after transition emitted by a module ``adapt`` hook."""

    module_type: str
    module_name: str
    update_type: Literal["replace", "append", "prune", "reweight", "checkpoint"] = "replace"
    before_hash: Optional[str] = None
    after_hash: Optional[str] = None
    reason: Optional[str] = None


class TrajectoryEvent(BaseModel):
    """Minimal trajectory signal for module ``adapt`` hooks (full contract: issue #1867)."""

    kind: TrajectoryEventKind
    reward: Optional[float] = None
    response: Optional[NeMoGymResponse] = None
    row: dict = Field(default_factory=dict)
    task_index: Optional[int] = None
    rollout_index: Optional[int] = None


class AgentContext(BaseModel):
    """Mutable per-rollout context passed through the module ``activate`` chain."""

    row: dict
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    module_refs: List[AgentModuleRef] = Field(default_factory=list)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:_HASH_PREFIX_LEN]


def hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


class AgentModule(ABC):
    """Behavior-shaping component on the Agent boundary."""

    def __init__(self, name: str, module_type: str) -> None:
        self.name = name
        self.module_type = module_type

    @abstractmethod
    def module_refs(self) -> List[AgentModuleRef]:
        pass

    async def activate(self, ctx: AgentContext) -> AgentContext:
        return ctx

    async def adapt(self, event: TrajectoryEvent) -> List[AgentUpdateEvent]:
        return []


class PromptAgentModule(AgentModule):
    """Inject a YAML prompt template into ``responses_create_params.input``.

    Wraps ``nemo_gym.prompt`` with content-hash provenance. When ``input`` is empty, builds
    messages from the template and row fields. When ``input`` is already set (e.g. Processor
    applied ``prompt_config``), only prepends a ``system`` message if the template defines one
    and the conversation does not already start with ``system``.
    """

    def __init__(self, name: str, path: str) -> None:
        super().__init__(name=name, module_type="working_memory")
        self.path = path
        self._resolved_path = _resolve_under_cwd_or_install(path)
        self._prompt_config = load_prompt_config(path)
        self._content_hash = hash_file(self._resolved_path)

    def module_refs(self) -> List[AgentModuleRef]:
        return [
            AgentModuleRef(
                type="working_memory",
                name=self.name,
                hash=self._content_hash,
                path=self.path,
            )
        ]

    async def activate(self, ctx: AgentContext) -> AgentContext:
        rcp = ctx.responses_create_params.model_copy(deep=True)
        messages = fill_prompt(self._prompt_config, ctx.row)
        easy_messages = [NeMoGymEasyInputMessage.model_validate(m) for m in messages]

        existing_input = rcp.input
        if existing_input is None or existing_input == "" or existing_input == []:
            rcp.input = easy_messages
        elif self._prompt_config.system is not None:
            normalized = self._coerce_input_list(existing_input)
            if not normalized or normalized[0].role != "system":
                rcp.input = [easy_messages[0]] + normalized
            else:
                rcp.input = normalized
        else:
            rcp.input = self._coerce_input_list(existing_input)

        ctx.responses_create_params = rcp
        return ctx

    @staticmethod
    def _coerce_input_list(
        existing_input: Union[str, List[NeMoGymEasyInputMessage], List[dict]],
    ) -> List[NeMoGymEasyInputMessage]:
        if isinstance(existing_input, str):
            return [NeMoGymEasyInputMessage(role="user", content=existing_input)]
        result: List[NeMoGymEasyInputMessage] = []
        for item in existing_input:
            if isinstance(item, NeMoGymEasyInputMessage):
                result.append(item)
            else:
                result.append(NeMoGymEasyInputMessage.model_validate(item))
        return result


class WorkingMemoryModuleSettings(BaseModel):
    path: str


class PromptModuleConfig(BaseModel):
    type: Literal["working_memory"] = "working_memory"
    name: str = "working_memory"
    config: WorkingMemoryModuleSettings


class SkillAdaptationConfig(BaseModel):
    """In-place skill mutation after rollouts (optimizer scaffold).

    When ``enabled``, failed rollouts (reward below ``reward_threshold``) append a lesson
    section to ``target_skill``'s ``SKILL.md``. The content hash changes so ``skills_ref`` and
    ``agent_module_refs`` distinguish variants (issue #1256).
    """

    enabled: bool = False
    target_skill: str = Field(description="Frontmatter ``name`` of the skill to adapt.")
    reward_threshold: float = Field(
        default=1.0,
        description="Adapt when terminal reward is strictly below this value.",
    )
    lesson_template: str = (
        "\n\n## Lesson (task={task_index}, rollout={rollout_index}, reward={reward})\n"
        "Previous attempt did not succeed. Re-read the task and apply the skill more carefully."
    )


class SkillLibraryModuleSettings(BaseModel):
    path: Optional[str] = None
    injection_mode: Literal["none", "context"] = Field(
        default="none",
        description=(
            "``none``: provenance only (native runtimes stage via ``stage_skills``). "
            "``context``: inject ``SKILL.md`` bodies into the system message (for ``simple_agent``)."
        ),
    )
    adaptation: Optional[SkillAdaptationConfig] = None


class SkillLibraryModuleConfig(BaseModel):
    type: Literal["skill_library"] = "skill_library"
    name: str = "skill_library"
    config: SkillLibraryModuleSettings = Field(default_factory=SkillLibraryModuleSettings)


class SkillLibraryAgentModule(AgentModule):
    """Agent Skills library: provenance, optional context injection, optional adaptation.

      - **Provenance:** ``skills_ref`` from the rollout row or module ``path``; stamped in
        ``module_refs``.
      - **Context injection:** for agents without native discovery, ``activate`` appends formatted
        ``SKILL.md`` bodies to the system message.
      - **Adaptation:** on failed rollouts, ``adapt`` can append a lesson to ``target_skill`` in
        place (hash changes; optimizers like ACE/EvoSkill can replace this with richer logic).
    - **Staging:** when ``staging_dir`` is set on the config, ``activate`` copies skills there for
        native runtimes (Claude Code pattern).
    """

    def __init__(
        self,
        name: str,
        path: Optional[str] = None,
        injection_mode: Literal["none", "context"] = "none",
        adaptation: Optional[SkillAdaptationConfig] = None,
        staging_dir: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, module_type="skill_library")
        self.path = path
        self.injection_mode = injection_mode
        self.adaptation = adaptation
        self.staging_dir = staging_dir
        self._skills_ref: Optional[SkillsRef] = None
        self._before_adapt_hash: Optional[str] = None

    @property
    def skills_path(self) -> Optional[str]:
        ref = self._resolve_skills_ref()
        return ref.path if ref else self.path

    def _resolve_skills_ref(self) -> Optional[SkillsRef]:
        if self._skills_ref is not None:
            return self._skills_ref
        if self.path is None:
            return None
        self._skills_ref = load_skill_directory(self.path)
        return self._skills_ref

    def bind_skills_ref(self, skills_ref: SkillsRef) -> None:
        """Prefer the run-level ``skills_ref`` stamped by rollout collection when present."""
        self._skills_ref = skills_ref
        if self.path is None:
            self.path = skills_ref.path

    def module_refs(self) -> List[AgentModuleRef]:
        skills_ref = self._resolve_skills_ref()
        if skills_ref is None:
            return []
        return [
            AgentModuleRef(
                type="skill_library",
                name=self.name,
                hash=skills_ref.hash,
                path=skills_ref.path,
            )
        ]

    async def activate(self, ctx: AgentContext) -> AgentContext:
        skills_path = self.skills_path
        if skills_path is None:
            return ctx

        if self.staging_dir is not None:
            dest = Path(self.staging_dir)
            if dest.exists():
                raise ValueError(f"staging_dir already exists: {dest}")
            stage_skills(skills_path, dest)

        if self.injection_mode != "context":
            return ctx

        bodies = load_skill_bodies(skills_path)
        skills_block = format_skills_for_context(bodies)
        rcp = ctx.responses_create_params.model_copy(deep=True)
        existing_input = rcp.input
        normalized = PromptAgentModule._coerce_input_list(existing_input if existing_input not in (None, "") else [])
        system_msg = NeMoGymEasyInputMessage(role="system", content=skills_block)
        if normalized and normalized[0].role == "system":
            merged = f"{normalized[0].content.rstrip()}\n\n{skills_block}"
            rcp.input = [NeMoGymEasyInputMessage(role="system", content=merged)] + normalized[1:]
        else:
            rcp.input = [system_msg] + normalized
        ctx.responses_create_params = rcp
        return ctx

    async def adapt(self, event: TrajectoryEvent) -> List[AgentUpdateEvent]:
        if event.kind != "terminated" or not self.adaptation or not self.adaptation.enabled:
            return []
        if event.reward is None or event.reward >= self.adaptation.reward_threshold:
            return []

        skills_path = self.skills_path
        if skills_path is None:
            return []

        before_ref = self._resolve_skills_ref()
        if before_ref is None:
            return []

        self._before_adapt_hash = before_ref.hash
        lesson = self.adaptation.lesson_template.format(
            reward=event.reward,
            task_index=event.task_index,
            rollout_index=event.rollout_index,
        )
        after_ref = apply_skill_adaptation(skills_path, self.adaptation.target_skill, lesson)
        self._skills_ref = after_ref

        return [
            AgentUpdateEvent(
                module_type=self.module_type,
                module_name=self.name,
                update_type="append",
                before_hash=self._before_adapt_hash,
                after_hash=after_ref.hash,
                reason=f"reward {event.reward} < {self.adaptation.reward_threshold}",
            )
        ]


AgentModuleConfig = Annotated[
    Union[PromptModuleConfig, SkillLibraryModuleConfig],
    Field(discriminator="type"),
]


def build_agent_modules(configs: Optional[Sequence[AgentModuleConfig]]) -> List[AgentModule]:
    if not configs:
        return []
    modules: List[AgentModule] = []
    for cfg in configs:
        if cfg.type == "working_memory":
            modules.append(PromptAgentModule(name=cfg.name, path=cfg.config.path))
        elif cfg.type == "skill_library":
            modules.append(
                SkillLibraryAgentModule(
                    name=cfg.name,
                    path=cfg.config.path,
                    injection_mode=cfg.config.injection_mode,
                    adaptation=cfg.config.adaptation,
                )
            )
    return modules


def bind_row_skills_ref(modules: Sequence[AgentModule], row: dict) -> None:
    skills_ref_data = row.get(SKILLS_REF_KEY_NAME)
    if not skills_ref_data:
        return
    skills_ref = SkillsRef.model_validate(skills_ref_data)
    for module in modules:
        if isinstance(module, SkillLibraryAgentModule):
            module.bind_skills_ref(skills_ref)


async def activate_agent_context(
    modules: Sequence[AgentModule],
    row: dict,
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
) -> AgentContext:
    bind_row_skills_ref(modules, row)
    ctx = AgentContext(row=row, responses_create_params=responses_create_params)
    for module in modules:
        ctx = await module.activate(ctx)
    ctx.module_refs = collect_module_refs(modules)
    return ctx


async def adapt_agent_modules(
    modules: Sequence[AgentModule],
    event: TrajectoryEvent,
) -> List[AgentUpdateEvent]:
    updates: List[AgentUpdateEvent] = []
    for module in modules:
        updates.extend(await module.adapt(event))
    return updates


def collect_module_refs(modules: Sequence[AgentModule]) -> List[AgentModuleRef]:
    refs: List[AgentModuleRef] = []
    for module in modules:
        refs.extend(module.module_refs())
    return refs
