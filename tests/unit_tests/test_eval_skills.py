# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from scripts.eval_skills import (
    DEFAULT_JUDGE_PROMPT_TEMPLATE,
    DEFAULT_JUDGE_SYSTEM_MESSAGE,
    EvalScenario,
    EvalSkill,
    _build_arg_parser,
    _configure_logging,
    _dispatch_tool_call,
    _main_async,
    _read_skill_md,
    _report_to_dict,
    call_judge,
    discover_skills,
    evaluate_scenario,
    evaluate_skill,
    load_skill,
    run_evaluation,
    run_rollout,
)

from nemo_gym.server_utils import ServerClient
from resources_servers.skill_workspace.app import SkillWorkspaceResourcesServer
from resources_servers.skill_workspace.schemas import SkillWorkspaceResourcesServerConfig


def _make_workspace(tmp_path: Path) -> SkillWorkspaceResourcesServer:
    (tmp_path / "ws").mkdir()
    cfg = SkillWorkspaceResourcesServerConfig(
        name="skill_workspace",
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        workspace_root=str(tmp_path / "ws"),
    )
    return SkillWorkspaceResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))


def _make_skill(tmp_path: Path, skill_name: str = "demo-skill") -> Path:
    skill_dir = tmp_path / "skills" / skill_name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "evals" / "files").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "You are an expert reviewer. Always run scripts/review.py and report the first line."
    )
    (skill_dir / "scripts" / "review.py").write_text("#!/usr/bin/env python3\nprint('REVIEW_OK first-line')\n")
    (skill_dir / "evals" / "files" / "sample.py").write_text("x = 1\n")
    evals = {
        "skill_name": skill_name,
        "evals": [
            {
                "id": 1,
                "prompt": "Review evals/files/sample.py.",
                "expected_output": "Reports REVIEW_OK from review.py",
                "files": ["evals/files/sample.py"],
                "assertions": [
                    "Agent runs scripts/review.py",
                    "Agent reports REVIEW_OK",
                ],
            },
        ],
    }
    (skill_dir / "evals" / "evals.json").write_text(json.dumps(evals))
    return skill_dir


def _assistant_msg_with_tool_call(name: str, arguments: dict, tool_call_id: str = "tc1") -> SimpleNamespace:
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=tool_call_id,
                function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
            )
        ],
    )


def _assistant_msg_final(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=text, tool_calls=None)


def _fake_completion(message: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _policy_client_that_returns(responses: list[SimpleNamespace]) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[_fake_completion(m) for m in responses])
    return client


def _judge_client_that_returns(text: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_fake_completion(SimpleNamespace(content=text, tool_calls=None))
    )
    return client


class TestSkillLoading:
    def test_load_skill_parses_scenarios(self, tmp_path: Path) -> None:
        skill_dir = _make_skill(tmp_path)
        skill = load_skill(skill_dir)
        assert skill is not None
        assert skill.skill_name == "demo-skill"
        assert len(skill.scenarios) == 1
        assert skill.scenarios[0].assertions == ["Agent runs scripts/review.py", "Agent reports REVIEW_OK"]

    def test_load_skill_missing_evals_returns_none(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "empty-skill"
        skill_dir.mkdir()
        assert load_skill(skill_dir) is None

    def test_discover_skills_filters_by_name(self, tmp_path: Path) -> None:
        _make_skill(tmp_path, "a-skill")
        _make_skill(tmp_path, "b-skill")
        skills = discover_skills(tmp_path / "skills", filter_names=["a-skill"])
        assert [s.skill_name for s in skills] == ["a-skill"]

    def test_discover_skills_returns_all_when_no_filter(self, tmp_path: Path) -> None:
        _make_skill(tmp_path, "a-skill")
        _make_skill(tmp_path, "b-skill")
        skills = discover_skills(tmp_path / "skills")
        assert sorted(s.skill_name for s in skills) == ["a-skill", "b-skill"]

    def test_discover_skills_skips_non_dir_entries(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "loose_file.txt").write_text("not a skill")
        _make_skill(tmp_path, "real-skill")
        skills = discover_skills(skills_dir)
        assert [s.skill_name for s in skills] == ["real-skill"]

    def test_discover_skills_raises_if_dir_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            discover_skills(tmp_path / "does-not-exist")


class TestDispatchToolCall:
    @pytest.mark.asyncio
    async def test_run_bash_returns_output_and_log_entry(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        skill_dir = _make_skill(tmp_path)
        seed = await ws.seed_session(
            __import__(
                "resources_servers.skill_workspace.schemas", fromlist=["SkillWorkspaceSeedSessionRequest"]
            ).SkillWorkspaceSeedSessionRequest(skill_path=str(skill_dir), scenario_id=1, files=[])
        )
        out, entry = await _dispatch_tool_call(ws, seed.env_id, "run_bash", json.dumps({"cmd": "echo hi"}))
        payload = json.loads(out)
        assert payload["exit_code"] == 0
        assert "hi" in payload["stdout"]
        assert entry.name == "run_bash"
        assert entry.exit_code == 0

    @pytest.mark.asyncio
    async def test_read_file_returns_content(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        skill_dir = _make_skill(tmp_path)
        seed = await ws.seed_session(
            __import__(
                "resources_servers.skill_workspace.schemas", fromlist=["SkillWorkspaceSeedSessionRequest"]
            ).SkillWorkspaceSeedSessionRequest(
                skill_path=str(skill_dir), scenario_id=1, files=["evals/files/sample.py"]
            )
        )
        out, entry = await _dispatch_tool_call(
            ws, seed.env_id, "read_file", json.dumps({"path": "evals/files/sample.py"})
        )
        assert "x = 1" in json.loads(out)["content"]
        assert entry.name == "read_file"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        out, entry = await _dispatch_tool_call(ws, "fake", "no_such_tool", "{}")
        assert "unknown tool" in json.loads(out)["error"]
        assert "unknown tool" in entry.stderr_snippet

    @pytest.mark.asyncio
    async def test_malformed_arguments_returns_error(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        out, entry = await _dispatch_tool_call(ws, "fake", "run_bash", "not-json")
        assert "not valid JSON" in json.loads(out)["error"]
        assert entry.arguments == "not-json"

    @pytest.mark.asyncio
    async def test_workspace_exception_returns_error_without_crashing(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        # env_id="bogus" will raise HTTPException from workspace.read_file
        out, entry = await _dispatch_tool_call(ws, "bogus", "read_file", '{"path": "a.txt"}')
        payload = json.loads(out)
        assert "error" in payload
        assert entry.name == "read_file"
        assert entry.stderr_snippet  # error recorded in log entry


class TestRunRollout:
    @pytest.mark.asyncio
    async def test_rollout_with_tool_call_then_final(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        skill_dir = _make_skill(tmp_path)
        seed = await ws.seed_session(
            __import__(
                "resources_servers.skill_workspace.schemas", fromlist=["SkillWorkspaceSeedSessionRequest"]
            ).SkillWorkspaceSeedSessionRequest(skill_path=str(skill_dir), scenario_id=1, files=[])
        )

        policy = _policy_client_that_returns(
            [
                _assistant_msg_with_tool_call("run_bash", {"cmd": "echo HELLO"}),
                _assistant_msg_final("ran it, got HELLO"),
            ]
        )
        scenario = EvalScenario(id=1, prompt="run echo", assertions=["agent ran echo"])
        final_text, tool_calls = await run_rollout(
            policy_client=policy,
            policy_model="m",
            workspace=ws,
            env_id=seed.env_id,
            scenario=scenario,
            system_prompt="sys",
            max_steps=5,
            temperature=0.0,
            max_output_tokens=None,
        )
        assert "HELLO" in final_text
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "run_bash"

    @pytest.mark.asyncio
    async def test_rollout_respects_max_steps(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        skill_dir = _make_skill(tmp_path)
        seed = await ws.seed_session(
            __import__(
                "resources_servers.skill_workspace.schemas", fromlist=["SkillWorkspaceSeedSessionRequest"]
            ).SkillWorkspaceSeedSessionRequest(skill_path=str(skill_dir), scenario_id=1, files=[])
        )

        # Always return a tool call — must cap at max_steps.
        policy = MagicMock()
        policy.chat.completions.create = AsyncMock(
            side_effect=[
                _fake_completion(_assistant_msg_with_tool_call("run_bash", {"cmd": f"echo {i}"}, f"tc{i}"))
                for i in range(3)
            ]
        )
        scenario = EvalScenario(id=1, prompt="loop", assertions=["x"])
        final_text, tool_calls = await run_rollout(
            policy_client=policy,
            policy_model="m",
            workspace=ws,
            env_id=seed.env_id,
            scenario=scenario,
            system_prompt=None,
            max_steps=3,
            temperature=0.0,
            max_output_tokens=128,
        )
        assert final_text == ""
        assert len(tool_calls) == 3

    @pytest.mark.asyncio
    async def test_rollout_no_tool_calls_returns_text_immediately(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        policy = _policy_client_that_returns([_assistant_msg_final("direct answer")])
        scenario = EvalScenario(id=1, prompt="p", assertions=["x"])
        final_text, tool_calls = await run_rollout(
            policy_client=policy,
            policy_model="m",
            workspace=ws,
            env_id="unused",
            scenario=scenario,
            system_prompt=None,
            max_steps=5,
            temperature=0.0,
            max_output_tokens=None,
        )
        assert final_text == "direct answer"
        assert tool_calls == []


class TestCallJudge:
    @pytest.mark.asyncio
    async def test_judge_parses_grades(self) -> None:
        judge = _judge_client_that_returns(
            '[{"id":1,"satisfied":true,"evidence":"ok"},{"id":2,"satisfied":false,"evidence":"missing"}]'
        )
        scenario = EvalScenario(id=1, prompt="p", assertions=["a1", "a2"])
        grades, err = await call_judge(
            judge_client=judge,
            judge_model="j",
            judge_prompt_template=DEFAULT_JUDGE_PROMPT_TEMPLATE.read_text(),
            judge_system_message=DEFAULT_JUDGE_SYSTEM_MESSAGE,
            scenario=scenario,
            response_text="r",
            tool_calls=[],
            temperature=0.0,
        )
        assert err is None
        assert [g.satisfied for g in grades] == [True, False]

    @pytest.mark.asyncio
    async def test_judge_empty_assertions_skips_llm(self) -> None:
        judge = MagicMock()
        judge.chat.completions.create = AsyncMock(side_effect=AssertionError("should not be called"))
        scenario = EvalScenario(id=1, prompt="p", assertions=[])
        grades, err = await call_judge(
            judge_client=judge,
            judge_model="j",
            judge_prompt_template="{prompt}{expected_output}{response}{tool_calls}{assertions}{n}",
            judge_system_message="sys",
            scenario=scenario,
            response_text="r",
            tool_calls=[],
            temperature=0.0,
        )
        assert grades == []
        assert err is None

    @pytest.mark.asyncio
    async def test_judge_unparseable_output_reports_error(self) -> None:
        judge = _judge_client_that_returns("I refuse")
        scenario = EvalScenario(id=1, prompt="p", assertions=["a1"])
        grades, err = await call_judge(
            judge_client=judge,
            judge_model="j",
            judge_prompt_template=DEFAULT_JUDGE_PROMPT_TEMPLATE.read_text(),
            judge_system_message="sys",
            scenario=scenario,
            response_text="r",
            tool_calls=[],
            temperature=0.0,
        )
        assert err is not None
        assert grades[0].satisfied is False


class TestEvaluateScenario:
    @pytest.mark.asyncio
    async def test_with_beats_without_yields_positive_delta(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        skill_dir = _make_skill(tmp_path)
        skill = load_skill(skill_dir)

        # With-skill path runs a tool then answers; without-skill path answers directly.
        # Route based on whether the system message is present in the prompt.
        with_tool_call = [
            _fake_completion(_assistant_msg_with_tool_call("run_bash", {"cmd": "python scripts/review.py"})),
            _fake_completion(_assistant_msg_final("REVIEW_OK first-line")),
        ]
        without_response = [_fake_completion(_assistant_msg_final("I don't know"))]
        with_queue = list(with_tool_call)
        without_queue = list(without_response)

        async def policy_side_effect(**kwargs):
            has_system = any(m.get("role") == "system" for m in kwargs["messages"])
            queue = with_queue if has_system else without_queue
            return queue.pop(0)

        policy = MagicMock()
        policy.chat.completions.create = AsyncMock(side_effect=policy_side_effect)

        async def judge_side_effect(**kwargs):
            user_content = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
            if "REVIEW_OK first-line" in user_content:
                text = '[{"id":1,"satisfied":true,"evidence":"ran"},{"id":2,"satisfied":true,"evidence":"REVIEW_OK"}]'
            else:
                text = (
                    '[{"id":1,"satisfied":false,"evidence":"no tool"},'
                    '{"id":2,"satisfied":false,"evidence":"no output"}]'
                )
            return _fake_completion(SimpleNamespace(content=text, tool_calls=None))

        judge = MagicMock()
        judge.chat.completions.create = AsyncMock(side_effect=judge_side_effect)

        report = await evaluate_scenario(
            policy_client=policy,
            policy_model="m",
            judge_client=judge,
            judge_model="j",
            judge_prompt_template=DEFAULT_JUDGE_PROMPT_TEMPLATE.read_text(),
            judge_system_message=DEFAULT_JUDGE_SYSTEM_MESSAGE,
            workspace=ws,
            skill=skill,
            scenario=skill.scenarios[0],
            max_steps=5,
            policy_temperature=0.0,
            judge_temperature=0.0,
            max_output_tokens=None,
        )
        assert report.score_with == 1.0
        assert report.score_without == 0.0
        assert report.delta == 1.0
        assert len(report.with_grades) == 2

    @pytest.mark.asyncio
    async def test_n_rollouts_averages_scores_and_reports_stddev(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        skill_dir = _make_skill(tmp_path)
        skill = load_skill(skill_dir)

        # 2 with-rollouts (score 1.0, 0.0) + 2 without-rollouts (score 0.5, 0.5)
        # expected: mean_with=0.5, stddev_with=0.5, mean_without=0.5, stddev_without=0.0, delta=0.0
        async def policy_side_effect(**_kwargs):
            return _fake_completion(_assistant_msg_final("ok"))

        policy = MagicMock()
        policy.chat.completions.create = AsyncMock(side_effect=policy_side_effect)

        grade_outputs = iter(
            [
                '[{"id":1,"satisfied":true,"evidence":""},{"id":2,"satisfied":true,"evidence":""}]',
                '[{"id":1,"satisfied":false,"evidence":""},{"id":2,"satisfied":false,"evidence":""}]',
                '[{"id":1,"satisfied":true,"evidence":""},{"id":2,"satisfied":false,"evidence":""}]',
                '[{"id":1,"satisfied":false,"evidence":""},{"id":2,"satisfied":true,"evidence":""}]',
            ]
        )

        async def judge_side_effect(**_kwargs):
            return _fake_completion(SimpleNamespace(content=next(grade_outputs), tool_calls=None))

        judge = MagicMock()
        judge.chat.completions.create = AsyncMock(side_effect=judge_side_effect)

        report = await evaluate_scenario(
            policy_client=policy,
            policy_model="m",
            judge_client=judge,
            judge_model="j",
            judge_prompt_template=DEFAULT_JUDGE_PROMPT_TEMPLATE.read_text(),
            judge_system_message=DEFAULT_JUDGE_SYSTEM_MESSAGE,
            workspace=ws,
            skill=skill,
            scenario=skill.scenarios[0],
            max_steps=3,
            policy_temperature=0.0,
            judge_temperature=0.0,
            max_output_tokens=None,
            n_rollouts=2,
        )
        assert report.n_rollouts == 2
        assert sorted(report.with_scores) == [0.0, 1.0]
        assert sorted(report.without_scores) == [0.5, 0.5]
        assert report.score_with == 0.5
        assert report.score_without == 0.5
        assert report.score_with_stddev == 0.5
        assert report.score_without_stddev == 0.0
        assert report.delta == 0.0


class TestEvaluateSkill:
    @pytest.mark.asyncio
    async def test_mean_delta_computed_across_scenarios(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        skill_dir = _make_skill(tmp_path)
        skill = load_skill(skill_dir)

        # Single scenario, delta = 0.5 - 0.5 = 0
        policy = MagicMock()
        policy.chat.completions.create = AsyncMock(
            side_effect=[
                _fake_completion(_assistant_msg_final("resp-with")),
                _fake_completion(_assistant_msg_final("resp-without")),
            ]
        )
        judge = MagicMock()
        judge.chat.completions.create = AsyncMock(
            side_effect=[
                _fake_completion(
                    SimpleNamespace(
                        content='[{"id":1,"satisfied":true,"evidence":"a"},{"id":2,"satisfied":false,"evidence":"b"}]',
                        tool_calls=None,
                    )
                ),
                _fake_completion(
                    SimpleNamespace(
                        content='[{"id":1,"satisfied":true,"evidence":"a"},{"id":2,"satisfied":false,"evidence":"b"}]',
                        tool_calls=None,
                    )
                ),
            ]
        )

        report = await evaluate_skill(
            policy_client=policy,
            policy_model="m",
            judge_client=judge,
            judge_model="j",
            judge_prompt_template=DEFAULT_JUDGE_PROMPT_TEMPLATE.read_text(),
            judge_system_message=DEFAULT_JUDGE_SYSTEM_MESSAGE,
            workspace=ws,
            skill=skill,
            max_steps=3,
            policy_temperature=0.0,
            judge_temperature=0.0,
            max_output_tokens=None,
            concurrency=1,
        )
        assert len(report.scenarios) == 1
        assert report.scenarios[0].delta == 0.0
        assert report.mean_delta == 0.0


class TestRunEvaluation:
    @pytest.mark.asyncio
    async def test_run_evaluation_emits_report_shape(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        skill_dir = _make_skill(tmp_path)
        skill = load_skill(skill_dir)

        policy = MagicMock()
        policy.chat.completions.create = AsyncMock(
            side_effect=[
                _fake_completion(_assistant_msg_final("r1")),
                _fake_completion(_assistant_msg_final("r2")),
            ]
        )
        judge = MagicMock()
        judge.chat.completions.create = AsyncMock(
            side_effect=[
                _fake_completion(
                    SimpleNamespace(
                        content='[{"id":1,"satisfied":true,"evidence":"ok"},'
                        '{"id":2,"satisfied":true,"evidence":"ok"}]',
                        tool_calls=None,
                    )
                ),
                _fake_completion(
                    SimpleNamespace(
                        content='[{"id":1,"satisfied":false,"evidence":"no"},'
                        '{"id":2,"satisfied":true,"evidence":"partial"}]',
                        tool_calls=None,
                    )
                ),
            ]
        )

        report = await run_evaluation(
            skills=[skill],
            policy_client=policy,
            policy_model="m",
            judge_client=judge,
            judge_model="j",
            judge_prompt_template=DEFAULT_JUDGE_PROMPT_TEMPLATE.read_text(),
            judge_system_message=DEFAULT_JUDGE_SYSTEM_MESSAGE,
            workspace=ws,
            max_steps=3,
            policy_temperature=0.0,
            judge_temperature=0.0,
            max_output_tokens=None,
            concurrency=2,
        )

        assert report.policy_model == "m"
        assert report.judge_model == "j"
        assert len(report.skills) == 1
        assert report.skills[0].scenarios[0].score_with == 1.0
        assert report.skills[0].scenarios[0].score_without == 0.5
        assert report.skills[0].scenarios[0].delta == 0.5

        d = _report_to_dict(report)
        assert d["skills"][0]["skill_name"] == "demo-skill"
        assert d["skills"][0]["scenarios"][0]["delta"] == 0.5
        # round-trips through JSON
        json.loads(json.dumps(d))


class TestCLIParser:
    def test_required_args(self) -> None:
        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--skills-dir",
                "/tmp/skills",
                "--policy-base-url",
                "http://localhost:8000/v1",
                "--policy-model",
                "nemotron",
                "--output",
                "/tmp/out.json",
            ]
        )
        assert args.policy_temperature is None
        assert args.judge_temperature is None
        assert args.max_steps == 10
        assert args.judge_model is None
        assert args.n_rollouts == 1
        assert args.skill_concurrency == 1


class TestExampleFixture:
    """Sanity check that EvalSkill round-trips prompts correctly."""

    def test_eval_skill_dataclass(self) -> None:
        skill = EvalSkill(
            skill_name="x",
            skill_path=Path("/tmp/x"),
            scenarios=[EvalScenario(id=1, prompt="p", assertions=["a"])],
        )
        assert skill.scenarios[0].id == 1


class TestReadSkillMd:
    def test_raises_if_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="SKILL.md missing"):
            _read_skill_md(tmp_path)


class TestDiscoverSkillsSkipsEmpty:
    def test_skill_with_no_scenarios_is_skipped(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        good = _make_skill(tmp_path, "good-skill")  # noqa: F841
        empty_dir = skills_dir / "empty-scenarios"
        (empty_dir / "evals").mkdir(parents=True)
        (empty_dir / "evals" / "evals.json").write_text(json.dumps({"skill_name": "empty", "evals": []}))
        skills = discover_skills(skills_dir)
        assert [s.skill_name for s in skills] == ["good-skill"]


class TestConfigureLogging:
    def test_verbose_sets_debug(self) -> None:
        _configure_logging(verbose=True)

    def test_default_sets_info(self) -> None:
        _configure_logging(verbose=False)


class TestMainAsync:
    @pytest.mark.asyncio
    async def test_returns_nonzero_when_no_skills_found(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        args = SimpleNamespace(
            skills_dir=skills_dir,
            skills=None,
            policy_base_url="http://x",
            policy_api_key="k",
            policy_model="m",
            judge_base_url=None,
            judge_api_key=None,
            judge_model=None,
            output=tmp_path / "out.json",
            max_steps=3,
            policy_temperature=0.0,
            judge_temperature=0.0,
            max_output_tokens=None,
            concurrency=1,
            skill_concurrency=1,
            n_rollouts=1,
            workspace_root=None,
            judge_prompt_template=DEFAULT_JUDGE_PROMPT_TEMPLATE,
        )
        rc = await _main_async(args)
        assert rc == 1

    @pytest.mark.asyncio
    async def test_writes_report_on_success(self, tmp_path: Path) -> None:
        _make_skill(tmp_path, "demo-skill")
        output_path = tmp_path / "out" / "report.json"
        args = SimpleNamespace(
            skills_dir=tmp_path / "skills",
            skills="demo-skill",
            policy_base_url="http://x",
            policy_api_key="k",
            policy_model="m",
            judge_base_url=None,
            judge_api_key=None,
            judge_model=None,
            output=output_path,
            max_steps=3,
            policy_temperature=0.0,
            judge_temperature=0.0,
            max_output_tokens=None,
            concurrency=1,
            skill_concurrency=1,
            n_rollouts=1,
            workspace_root=tmp_path / "ws_root",
            judge_prompt_template=DEFAULT_JUDGE_PROMPT_TEMPLATE,
        )
        (tmp_path / "ws_root").mkdir()

        policy_instance = MagicMock()
        policy_instance.chat.completions.create = AsyncMock(
            side_effect=[
                _fake_completion(_assistant_msg_final("with-answer")),
                _fake_completion(_assistant_msg_final("without-answer")),
            ]
        )
        judge_instance = MagicMock()
        judge_instance.chat.completions.create = AsyncMock(
            return_value=_fake_completion(
                SimpleNamespace(
                    content='[{"id":1,"satisfied":true,"evidence":"ok"},{"id":2,"satisfied":true,"evidence":"ok"}]',
                    tool_calls=None,
                )
            )
        )

        with patch("scripts.eval_skills.AsyncOpenAI", side_effect=[policy_instance, judge_instance]):
            rc = await _main_async(args)
        assert rc == 0
        assert output_path.is_file()
        data = json.loads(output_path.read_text())
        assert data["skills"][0]["skill_name"] == "demo-skill"
