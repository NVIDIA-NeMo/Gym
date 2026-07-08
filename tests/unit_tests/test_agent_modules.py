# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

from nemo_gym.agent_modules import (
    AgentContext,
    PromptAgentModule,
    PromptModuleConfig,
    SkillAdaptationConfig,
    SkillLibraryAgentModule,
    SkillLibraryModuleConfig,
    TrajectoryEvent,
    build_agent_modules,
    observe_agent_modules,
    prepare_agent_context,
)
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.prompt import load_prompt_config
from nemo_gym.skills import SKILL_MD_FILENAME, hash_skill_dir, load_skill_directory


@pytest.fixture(autouse=True)
def _clear_prompt_cache():
    load_prompt_config.cache_clear()
    yield
    load_prompt_config.cache_clear()


class TestPromptAgentModule:
    @pytest.mark.asyncio
    async def test_prepare_builds_input_from_row(self, tmp_path):
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"system": "Be concise.", "user": "Question: {question}"}))

        module = PromptAgentModule(name="test_prompt", path=str(prompt_path))
        row = {"question": "What is 2+2?"}
        rcp = NeMoGymResponseCreateParamsNonStreaming(input="")
        ctx = await module.prepare(AgentContext(row=row, responses_create_params=rcp))

        assert len(ctx.responses_create_params.input) == 2
        assert ctx.responses_create_params.input[0].role == "system"
        assert ctx.responses_create_params.input[1].content == "Question: What is 2+2?"

        refs = module.artifact_refs()
        assert refs[0].type == "prompt"
        assert refs[0].name == "test_prompt"
        assert len(refs[0].hash) == 12

    @pytest.mark.asyncio
    async def test_prepare_prepends_system_when_input_exists(self, tmp_path):
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"system": "Format: A/B/C/D", "user": "{question}"}))

        module = PromptAgentModule(name="fmt", path=str(prompt_path))
        rcp = NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content="Pick one.")]
        )
        ctx = await module.prepare(AgentContext(row={"question": "ignored"}, responses_create_params=rcp))

        assert ctx.responses_create_params.input[0].role == "system"
        assert ctx.responses_create_params.input[1].content == "Pick one."

    def test_artifact_ref_hash_changes_with_content(self, tmp_path):
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"user": "{q}"}))
        module_a = PromptAgentModule(name="p", path=str(prompt_path))
        hash_a = module_a.artifact_refs()[0].hash

        prompt_path.write_text(yaml.dump({"user": "{q} updated"}))
        load_prompt_config.cache_clear()
        module_b = PromptAgentModule(name="p", path=str(prompt_path))
        hash_b = module_b.artifact_refs()[0].hash

        assert hash_a != hash_b


class TestSkillLibraryAgentModule:
    def test_artifact_ref_from_path(self, tmp_path):
        skills_root = tmp_path / "skills"
        skill_dir = skills_root / "demo"
        skill_dir.mkdir(parents=True)
        (skill_dir / SKILL_MD_FILENAME).write_text("---\nname: demo\ndescription: test\n---\n")

        module = SkillLibraryAgentModule(name="baseline", path=str(skills_root))
        refs = module.artifact_refs()
        assert refs[0].type == "skill_library"
        assert refs[0].hash == hash_skill_dir(skills_root)[:12]

    def test_bind_skills_ref_from_row(self, tmp_path):
        skills_root = tmp_path / "skills"
        skill_dir = skills_root / "demo"
        skill_dir.mkdir(parents=True)
        (skill_dir / SKILL_MD_FILENAME).write_text("---\nname: demo\ndescription: test\n---\n")

        skills_ref = load_skill_directory(str(skills_root))
        module = SkillLibraryAgentModule(name="baseline")
        module.bind_skills_ref(skills_ref)

        refs = module.artifact_refs()
        assert refs[0].path == str(skills_root)

    @pytest.mark.asyncio
    async def test_context_injection_prepends_system_block(self, tmp_path):
        skills_root = tmp_path / "skills"
        skill_dir = skills_root / "cot"
        skill_dir.mkdir(parents=True)
        (skill_dir / SKILL_MD_FILENAME).write_text(
            "---\nname: cot\ndescription: Think step by step.\n---\n# CoT\nAlways reason first.\n"
        )

        module = SkillLibraryAgentModule(
            name="skills",
            path=str(skills_root),
            injection_mode="context",
        )
        rcp = NeMoGymResponseCreateParamsNonStreaming(input=[NeMoGymEasyInputMessage(role="user", content="Q?")])
        ctx = await module.prepare(AgentContext(row={}, responses_create_params=rcp))

        assert ctx.responses_create_params.input[0].role == "system"
        assert "cot" in ctx.responses_create_params.input[0].content.lower()
        assert "Always reason first" in ctx.responses_create_params.input[0].content
        assert ctx.responses_create_params.input[1].content == "Q?"

    @pytest.mark.asyncio
    async def test_adaptation_on_failed_rollout(self, tmp_path):
        skills_root = tmp_path / "skills"
        skill_dir = skills_root / "cot"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / SKILL_MD_FILENAME
        skill_md.write_text("---\nname: cot\ndescription: CoT\n---\n# Body\n")

        adaptation = SkillAdaptationConfig(enabled=True, target_skill="cot", reward_threshold=1.0)
        module = SkillLibraryAgentModule(
            name="skills",
            path=str(skills_root),
            adaptation=adaptation,
        )
        before_hash = module.artifact_refs()[0].hash

        event = TrajectoryEvent(kind="terminated", reward=0.0, task_index=1, rollout_index=2)
        updates = await module.observe(event)

        assert len(updates) == 1
        assert updates[0].update_type == "append"
        assert updates[0].before_hash == before_hash
        assert updates[0].after_hash != before_hash
        assert "Lesson" in skill_md.read_text()
        assert module.artifact_refs()[0].hash == updates[0].after_hash

    @pytest.mark.asyncio
    async def test_adaptation_skipped_on_success(self, tmp_path):
        skills_root = tmp_path / "skills"
        skill_dir = skills_root / "cot"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / SKILL_MD_FILENAME
        skill_md.write_text("---\nname: cot\ndescription: CoT\n---\n# Body\n")

        module = SkillLibraryAgentModule(
            name="skills",
            path=str(skills_root),
            adaptation=SkillAdaptationConfig(enabled=True, target_skill="cot"),
        )
        updates = await module.observe(TrajectoryEvent(kind="terminated", reward=1.0))
        assert updates == []
        assert skill_md.read_text().endswith("# Body\n")


class TestBuildAndPrepare:
    @pytest.mark.asyncio
    async def test_prepare_agent_context_runs_module_chain(self, tmp_path):
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"user": "Solve {problem}"}))

        modules = build_agent_modules([PromptModuleConfig(path=str(prompt_path))])
        row = {"problem": "1+1"}
        rcp = NeMoGymResponseCreateParamsNonStreaming(input="")

        ctx = await prepare_agent_context(modules, row, rcp)
        assert ctx.responses_create_params.input[0].content == "Solve 1+1"
        assert len(ctx.artifact_refs) == 1
        assert ctx.artifact_refs[0].type == "prompt"

    def test_build_skill_library_module_config(self, tmp_path):
        skills_root = tmp_path / "skills"
        skill_dir = skills_root / "demo"
        skill_dir.mkdir(parents=True)
        (skill_dir / SKILL_MD_FILENAME).write_text("---\nname: demo\ndescription: test\n---\n")

        modules = build_agent_modules([SkillLibraryModuleConfig(path=str(skills_root))])
        assert len(modules) == 1
        assert isinstance(modules[0], SkillLibraryAgentModule)


class TestObserveAgentModules:
    @pytest.mark.asyncio
    async def test_default_observe_returns_empty(self, tmp_path):
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"user": "{x}"}))
        modules = build_agent_modules([PromptModuleConfig(path=str(prompt_path))])
        event = TrajectoryEvent(kind="terminated", reward=1.0, row={"x": "hi"})
        updates = await observe_agent_modules(modules, event)
        assert updates == []
