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
"""Skill-evaluation harness — with-skill vs without-skill delta scoring via LLM judge.

Orchestrates rollouts per (skill, scenario) pair:
  1. Seed an ephemeral workspace with SKILL.md, scripts/, references/, and scenario fixtures.
  2. Run the policy model twice — once with SKILL.md as system message, once without.
  3. During each rollout, dispatch bash/file tool calls to the in-process skill_workspace.
  4. Call the judge LLM to grade each assertion against response + tool-call log.
  5. delta = score_with - score_without.

The harness is intentionally standalone: it does not require a running NeMo Gym HeadServer.
It talks to any OpenAI-compatible policy and judge endpoints (e.g. vLLM serving Nemotron Nano).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import hashlib
import json
import logging
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

from openai import AsyncOpenAI

from nemo_gym.server_utils import ServerClient
from resources_servers.skill_judge.app import (
    _extract_json_array,
    _format_assertions,
    _format_tool_calls,
    _normalize_grades,
)
from resources_servers.skill_judge.schemas import AssertionGrade, ToolCallLogEntry
from resources_servers.skill_workspace.app import SkillWorkspaceResourcesServer
from resources_servers.skill_workspace.schemas import (
    CloseRequest,
    ReadFileRequest,
    RunBashRequest,
    SkillWorkspaceResourcesServerConfig,
    SkillWorkspaceSeedSessionRequest,
)


logger = logging.getLogger(__name__)


DEFAULT_JUDGE_SYSTEM_MESSAGE = (
    "You grade whether an AI assistant's response satisfies a list of behavioral assertions. "
    "You judge only what is literally present in the response or tool-call log. "
    "Output a JSON array only — no prose, no markdown fences."
)

SNIPPET_CHARS = 500
DEFAULT_JUDGE_PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "resources_servers" / "skill_judge" / "prompt_templates" / "skill_judge.txt"
)

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Execute a bash command in the workspace CWD. "
                "Combined stdout+stderr is capped at 50KB. Default timeout 30s."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "The bash command to run."},
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout in seconds (max 120).",
                    },
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file from the workspace. Rejects absolute paths and paths that escape the workspace root."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative path."},
                },
                "required": ["path"],
            },
        },
    },
]


@dataclasses.dataclass
class EvalScenario:
    id: int
    prompt: str
    expected_output: str = ""
    files: list[str] = dataclasses.field(default_factory=list)
    assertions: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class EvalSkill:
    skill_name: str
    skill_path: Path
    scenarios: list[EvalScenario]
    skill_md_sha: str = ""


@dataclasses.dataclass
class RolloutResult:
    response_text: str
    tool_calls: list[ToolCallLogEntry]
    grades: list[AssertionGrade]
    score: float
    parse_error: Optional[str] = None


@dataclasses.dataclass
class ScenarioReport:
    scenario_id: int
    prompt: str
    with_grades: list[AssertionGrade]
    without_grades: list[AssertionGrade]
    score_with: float
    score_without: float
    delta: float
    response_with: str = ""
    response_without: str = ""
    tool_calls_with: list[ToolCallLogEntry] = dataclasses.field(default_factory=list)
    tool_calls_without: list[ToolCallLogEntry] = dataclasses.field(default_factory=list)
    parse_error_with: Optional[str] = None
    parse_error_without: Optional[str] = None
    n_rollouts: int = 1
    with_scores: list[float] = dataclasses.field(default_factory=list)
    without_scores: list[float] = dataclasses.field(default_factory=list)
    score_with_stddev: float = 0.0
    score_without_stddev: float = 0.0


@dataclasses.dataclass
class SkillReport:
    skill_name: str
    scenarios: list[ScenarioReport]
    mean_delta: float
    skill_md_sha: str = ""


@dataclasses.dataclass
class EvalReport:
    run_id: str
    started_at: str
    finished_at: str
    policy_model: str
    judge_model: str
    skills: list[SkillReport]


def load_skill(skill_path: Path) -> Optional[EvalSkill]:
    evals_path = skill_path / "evals" / "evals.json"
    if not evals_path.is_file():
        logger.warning("skill %s has no evals.json; skipping", skill_path.name)
        return None
    with evals_path.open() as f:
        data = json.load(f)
    scenarios = [
        EvalScenario(
            id=int(s["id"]),
            prompt=str(s["prompt"]),
            expected_output=str(s.get("expected_output", "")),
            files=list(s.get("files", [])),
            assertions=list(s.get("assertions", [])),
        )
        for s in data.get("evals", [])
    ]
    skill_md_path = skill_path / "SKILL.md"
    skill_md_sha = (
        hashlib.sha256(skill_md_path.read_text().encode("utf-8")).hexdigest()[:12] if skill_md_path.is_file() else ""
    )
    return EvalSkill(
        skill_name=str(data.get("skill_name", skill_path.name)),
        skill_path=skill_path,
        scenarios=scenarios,
        skill_md_sha=skill_md_sha,
    )


def discover_skills(skills_dir: Path, filter_names: Optional[list[str]] = None) -> list[EvalSkill]:
    if not skills_dir.is_dir():
        raise ValueError(f"skills-dir is not a directory: {skills_dir}")
    skills: list[EvalSkill] = []
    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_dir():
            continue
        if filter_names and entry.name not in filter_names:
            continue
        skill = load_skill(entry)
        if skill is None or not skill.scenarios:
            continue
        skills.append(skill)
    return skills


def _read_skill_md(skill_path: Path) -> str:
    skill_md = skill_path / "SKILL.md"
    if not skill_md.is_file():
        raise ValueError(f"SKILL.md missing at {skill_md}")
    return skill_md.read_text()


async def _dispatch_tool_call(
    workspace: SkillWorkspaceResourcesServer,
    env_id: str,
    name: str,
    arguments_json: str,
) -> tuple[str, ToolCallLogEntry]:
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        err = f"tool arguments were not valid JSON: {arguments_json!r}"
        return (
            json.dumps({"error": err}),
            ToolCallLogEntry(name=name, arguments=arguments_json, stderr_snippet=err),
        )

    if name == "run_bash":
        req = RunBashRequest(
            env_id=env_id,
            cmd=str(args.get("cmd", "")),
            timeout_seconds=args.get("timeout_seconds"),
        )
        try:
            resp = await workspace.run_bash(req)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            return (
                json.dumps({"error": err}),
                ToolCallLogEntry(name="run_bash", arguments=json.dumps(args), stderr_snippet=err[:SNIPPET_CHARS]),
            )
        tool_output = json.dumps(
            {
                "stdout": resp.stdout,
                "stderr": resp.stderr,
                "exit_code": resp.exit_code,
                "truncated": resp.truncated,
                "timed_out": resp.timed_out,
            }
        )
        entry = ToolCallLogEntry(
            name="run_bash",
            arguments=json.dumps(args),
            exit_code=resp.exit_code,
            stdout_snippet=resp.stdout[:SNIPPET_CHARS],
            stderr_snippet=resp.stderr[:SNIPPET_CHARS],
            truncated=resp.truncated,
        )
        return tool_output, entry

    if name == "read_file":
        req = ReadFileRequest(env_id=env_id, path=str(args.get("path", "")))
        try:
            resp = await workspace.read_file(req)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            return (
                json.dumps({"error": err}),
                ToolCallLogEntry(name="read_file", arguments=json.dumps(args), stderr_snippet=err[:SNIPPET_CHARS]),
            )
        tool_output = json.dumps({"content": resp.content, "truncated": resp.truncated})
        entry = ToolCallLogEntry(
            name="read_file",
            arguments=json.dumps(args),
            stdout_snippet=resp.content[:SNIPPET_CHARS],
            truncated=resp.truncated,
        )
        return tool_output, entry

    err = f"unknown tool: {name}"
    return (
        json.dumps({"error": err}),
        ToolCallLogEntry(name=name, arguments=arguments_json, stderr_snippet=err),
    )


async def run_rollout(
    policy_client: AsyncOpenAI,
    policy_model: str,
    workspace: SkillWorkspaceResourcesServer,
    env_id: str,
    scenario: EvalScenario,
    system_prompt: Optional[str],
    max_steps: int,
    temperature: Optional[float],
    max_output_tokens: Optional[int],
) -> tuple[str, list[ToolCallLogEntry]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": scenario.prompt})

    tool_call_log: list[ToolCallLogEntry] = []
    final_text = ""

    for _ in range(max_steps):
        kwargs: dict[str, Any] = {
            "model": policy_model,
            "messages": messages,
            "tools": TOOLS_SCHEMA,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens

        completion = await policy_client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        msg = choice.message
        assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "",
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        if not msg.tool_calls:
            final_text = msg.content or ""
            break

        for tc in msg.tool_calls:
            output, entry = await _dispatch_tool_call(workspace, env_id, tc.function.name, tc.function.arguments or "")
            tool_call_log.append(entry)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

    return final_text, tool_call_log


def _build_judge_user_prompt(
    template: str,
    scenario: EvalScenario,
    response_text: str,
    tool_calls: list[ToolCallLogEntry],
) -> str:
    return template.format(
        prompt=scenario.prompt,
        expected_output=scenario.expected_output or "(not provided)",
        response=response_text or "(empty response)",
        tool_calls=_format_tool_calls(tool_calls),
        assertions=_format_assertions(scenario.assertions),
        n=len(scenario.assertions),
    )


async def call_judge(
    judge_client: AsyncOpenAI,
    judge_model: str,
    judge_prompt_template: str,
    judge_system_message: str,
    scenario: EvalScenario,
    response_text: str,
    tool_calls: list[ToolCallLogEntry],
    temperature: Optional[float],
    max_output_tokens: Optional[int] = None,
) -> tuple[list[AssertionGrade], Optional[str]]:
    n = len(scenario.assertions)
    if n == 0:
        return [], None

    messages = [
        {"role": "system", "content": judge_system_message},
        {
            "role": "user",
            "content": _build_judge_user_prompt(judge_prompt_template, scenario, response_text, tool_calls),
        },
    ]

    kwargs: dict[str, Any] = {
        "model": judge_model,
        "messages": messages,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    completion = await judge_client.chat.completions.create(**kwargs)
    judge_text = completion.choices[0].message.content or ""
    raw_grades = _extract_json_array(judge_text)
    if raw_grades is None:
        return (
            [
                AssertionGrade(id=i, satisfied=False, evidence="judge output was not parseable JSON")
                for i in range(1, n + 1)
            ],
            "could not extract JSON array from judge output",
        )
    return _normalize_grades(raw_grades, n), None


async def _single_rollout_and_judge(
    policy_client: AsyncOpenAI,
    policy_model: str,
    judge_client: AsyncOpenAI,
    judge_model: str,
    judge_prompt_template: str,
    judge_system_message: str,
    workspace: SkillWorkspaceResourcesServer,
    skill_path: Path,
    scenario: EvalScenario,
    system_prompt: Optional[str],
    max_steps: int,
    policy_temperature: Optional[float],
    judge_temperature: Optional[float],
    max_output_tokens: Optional[int],
) -> RolloutResult:
    seed_resp = await workspace.seed_session(
        SkillWorkspaceSeedSessionRequest(
            skill_path=str(skill_path),
            scenario_id=scenario.id,
            files=list(scenario.files),
        )
    )
    env_id = seed_resp.env_id
    try:
        response_text, tool_calls = await run_rollout(
            policy_client=policy_client,
            policy_model=policy_model,
            workspace=workspace,
            env_id=env_id,
            scenario=scenario,
            system_prompt=system_prompt,
            max_steps=max_steps,
            temperature=policy_temperature,
            max_output_tokens=max_output_tokens,
        )
        grades, parse_error = await call_judge(
            judge_client=judge_client,
            judge_model=judge_model,
            judge_prompt_template=judge_prompt_template,
            judge_system_message=judge_system_message,
            scenario=scenario,
            response_text=response_text,
            tool_calls=tool_calls,
            temperature=judge_temperature,
        )
    finally:
        await workspace.close(CloseRequest(env_id=env_id))

    satisfied = sum(1 for g in grades if g.satisfied)
    score = satisfied / len(grades) if grades else 0.0
    return RolloutResult(
        response_text=response_text,
        tool_calls=tool_calls,
        grades=grades,
        score=score,
        parse_error=parse_error,
    )


async def evaluate_scenario(
    policy_client: AsyncOpenAI,
    policy_model: str,
    judge_client: AsyncOpenAI,
    judge_model: str,
    judge_prompt_template: str,
    judge_system_message: str,
    workspace: SkillWorkspaceResourcesServer,
    skill: EvalSkill,
    scenario: EvalScenario,
    max_steps: int,
    policy_temperature: Optional[float],
    judge_temperature: Optional[float],
    max_output_tokens: Optional[int],
    n_rollouts: int = 1,
) -> ScenarioReport:
    skill_md = _read_skill_md(skill.skill_path)

    def _one(system_prompt: Optional[str]) -> Any:
        return _single_rollout_and_judge(
            policy_client=policy_client,
            policy_model=policy_model,
            judge_client=judge_client,
            judge_model=judge_model,
            judge_prompt_template=judge_prompt_template,
            judge_system_message=judge_system_message,
            workspace=workspace,
            skill_path=skill.skill_path,
            scenario=scenario,
            system_prompt=system_prompt,
            max_steps=max_steps,
            policy_temperature=policy_temperature,
            judge_temperature=judge_temperature,
            max_output_tokens=max_output_tokens,
        )

    all_tasks = [_one(skill_md) for _ in range(n_rollouts)] + [_one(None) for _ in range(n_rollouts)]
    results = await asyncio.gather(*all_tasks)
    with_results = results[:n_rollouts]
    without_results = results[n_rollouts:]

    with_scores = [r.score for r in with_results]
    without_scores = [r.score for r in without_results]
    mean_with = statistics.fmean(with_scores)
    mean_without = statistics.fmean(without_scores)
    stddev_with = statistics.pstdev(with_scores) if len(with_scores) > 1 else 0.0
    stddev_without = statistics.pstdev(without_scores) if len(without_scores) > 1 else 0.0

    first_with = with_results[0]
    first_without = without_results[0]

    return ScenarioReport(
        scenario_id=scenario.id,
        prompt=scenario.prompt,
        with_grades=first_with.grades,
        without_grades=first_without.grades,
        score_with=mean_with,
        score_without=mean_without,
        delta=mean_with - mean_without,
        response_with=first_with.response_text,
        response_without=first_without.response_text,
        tool_calls_with=first_with.tool_calls,
        tool_calls_without=first_without.tool_calls,
        parse_error_with=first_with.parse_error,
        parse_error_without=first_without.parse_error,
        n_rollouts=n_rollouts,
        with_scores=with_scores,
        without_scores=without_scores,
        score_with_stddev=stddev_with,
        score_without_stddev=stddev_without,
    )


async def evaluate_skill(
    policy_client: AsyncOpenAI,
    policy_model: str,
    judge_client: AsyncOpenAI,
    judge_model: str,
    judge_prompt_template: str,
    judge_system_message: str,
    workspace: SkillWorkspaceResourcesServer,
    skill: EvalSkill,
    max_steps: int,
    policy_temperature: Optional[float],
    judge_temperature: Optional[float],
    max_output_tokens: Optional[int],
    concurrency: int,
    n_rollouts: int = 1,
) -> SkillReport:
    semaphore = asyncio.Semaphore(concurrency)

    async def _guarded(sc: EvalScenario) -> ScenarioReport:
        async with semaphore:
            return await evaluate_scenario(
                policy_client=policy_client,
                policy_model=policy_model,
                judge_client=judge_client,
                judge_model=judge_model,
                judge_prompt_template=judge_prompt_template,
                judge_system_message=judge_system_message,
                workspace=workspace,
                skill=skill,
                scenario=sc,
                max_steps=max_steps,
                policy_temperature=policy_temperature,
                judge_temperature=judge_temperature,
                max_output_tokens=max_output_tokens,
                n_rollouts=n_rollouts,
            )

    scenario_reports = await asyncio.gather(*(_guarded(sc) for sc in skill.scenarios))
    mean_delta = statistics.fmean(r.delta for r in scenario_reports) if scenario_reports else 0.0
    return SkillReport(
        skill_name=skill.skill_name,
        scenarios=scenario_reports,
        mean_delta=mean_delta,
        skill_md_sha=skill.skill_md_sha,
    )


def _report_to_dict(report: EvalReport) -> dict[str, Any]:
    def grade_dict(g: AssertionGrade) -> dict[str, Any]:
        return {"id": g.id, "satisfied": g.satisfied, "evidence": g.evidence}

    def tool_call_dict(tc: ToolCallLogEntry) -> dict[str, Any]:
        return {
            "name": tc.name,
            "arguments": tc.arguments,
            "exit_code": tc.exit_code,
            "stdout_snippet": tc.stdout_snippet,
            "stderr_snippet": tc.stderr_snippet,
            "truncated": tc.truncated,
        }

    return {
        "run_id": report.run_id,
        "started_at": report.started_at,
        "finished_at": report.finished_at,
        "policy_model": report.policy_model,
        "judge_model": report.judge_model,
        "skills": [
            {
                "skill_name": s.skill_name,
                "skill_md_sha": s.skill_md_sha,
                "mean_delta": s.mean_delta,
                "scenarios": [
                    {
                        "scenario_id": sc.scenario_id,
                        "prompt": sc.prompt,
                        "with_grades": [grade_dict(g) for g in sc.with_grades],
                        "without_grades": [grade_dict(g) for g in sc.without_grades],
                        "score_with": sc.score_with,
                        "score_without": sc.score_without,
                        "delta": sc.delta,
                        "n_rollouts": sc.n_rollouts,
                        "with_scores": sc.with_scores,
                        "without_scores": sc.without_scores,
                        "score_with_stddev": sc.score_with_stddev,
                        "score_without_stddev": sc.score_without_stddev,
                        "response_with": sc.response_with,
                        "response_without": sc.response_without,
                        "tool_calls_with": [tool_call_dict(tc) for tc in sc.tool_calls_with],
                        "tool_calls_without": [tool_call_dict(tc) for tc in sc.tool_calls_without],
                        "parse_error_with": sc.parse_error_with,
                        "parse_error_without": sc.parse_error_without,
                    }
                    for sc in s.scenarios
                ],
            }
            for s in report.skills
        ],
    }


async def run_evaluation(
    skills: list[EvalSkill],
    policy_client: AsyncOpenAI,
    policy_model: str,
    judge_client: AsyncOpenAI,
    judge_model: str,
    judge_prompt_template: str,
    judge_system_message: str,
    workspace: SkillWorkspaceResourcesServer,
    max_steps: int,
    policy_temperature: Optional[float],
    judge_temperature: Optional[float],
    max_output_tokens: Optional[int],
    concurrency: int,
    skill_concurrency: int = 1,
    n_rollouts: int = 1,
) -> EvalReport:
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    skill_semaphore = asyncio.Semaphore(skill_concurrency)

    async def _run_one(skill: EvalSkill) -> SkillReport:
        async with skill_semaphore:
            t0 = time.monotonic()
            report = await evaluate_skill(
                policy_client=policy_client,
                policy_model=policy_model,
                judge_client=judge_client,
                judge_model=judge_model,
                judge_prompt_template=judge_prompt_template,
                judge_system_message=judge_system_message,
                workspace=workspace,
                skill=skill,
                max_steps=max_steps,
                policy_temperature=policy_temperature,
                judge_temperature=judge_temperature,
                max_output_tokens=max_output_tokens,
                concurrency=concurrency,
                n_rollouts=n_rollouts,
            )
            logger.info(
                "skill=%s mean_delta=%+.3f scenarios=%d elapsed=%.1fs",
                report.skill_name,
                report.mean_delta,
                len(report.scenarios),
                time.monotonic() - t0,
            )
            return report

    skill_reports = list(await asyncio.gather(*(_run_one(s) for s in skills)))

    return EvalReport(
        run_id=str(uuid.uuid4()),
        started_at=started_at,
        finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        policy_model=policy_model,
        judge_model=judge_model,
        skills=skill_reports,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Skill-evaluation harness (with/without delta)")
    parser.add_argument("--skills-dir", type=Path, required=True, help="Directory containing skill subdirs.")
    parser.add_argument(
        "--skills",
        type=str,
        default=None,
        help="Comma-separated skill names to evaluate (default: all with evals.json).",
    )
    parser.add_argument("--policy-base-url", type=str, required=True)
    parser.add_argument("--policy-api-key", type=str, default="dummy")
    parser.add_argument("--policy-model", type=str, required=True)
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=None,
        help="Judge endpoint base URL (defaults to policy-base-url).",
    )
    parser.add_argument("--judge-api-key", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None, help="Defaults to policy-model.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report path.")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=None,
        help="Omitted from request if unset (required for some models, e.g. Claude Opus 4.7).",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=None,
        help="Omitted from request if unset.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4, help="Scenarios in flight per skill.")
    parser.add_argument("--skill-concurrency", type=int, default=1, help="Skills in flight simultaneously.")
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=1,
        help="Rollouts per scenario per condition. Averaging reduces variance at temperature>0.",
    )
    parser.add_argument("--workspace-root", type=Path, default=None, help="Parent dir for ephemeral workspaces.")
    parser.add_argument(
        "--judge-prompt-template",
        type=Path,
        default=DEFAULT_JUDGE_PROMPT_TEMPLATE,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


async def _main_async(args: argparse.Namespace) -> int:
    filter_names = [s.strip() for s in args.skills.split(",")] if args.skills else None
    skills = discover_skills(args.skills_dir, filter_names=filter_names)
    if not skills:
        logger.error("no skills found under %s (filter=%s)", args.skills_dir, filter_names)
        return 1
    logger.info("discovered %d skill(s): %s", len(skills), [s.skill_name for s in skills])

    judge_base_url = args.judge_base_url or args.policy_base_url
    judge_api_key = args.judge_api_key or args.policy_api_key
    judge_model = args.judge_model or args.policy_model

    policy_client = AsyncOpenAI(base_url=args.policy_base_url, api_key=args.policy_api_key)
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=judge_api_key)

    judge_prompt_template = Path(args.judge_prompt_template).read_text()

    workspace_root = str(args.workspace_root) if args.workspace_root is not None else None
    workspace = SkillWorkspaceResourcesServer(
        config=SkillWorkspaceResourcesServerConfig(
            name="skill_workspace",
            host="0.0.0.0",
            port=0,
            entrypoint="",
            workspace_root=workspace_root,
        ),
        server_client=MagicMock(spec=ServerClient),
    )

    report = await run_evaluation(
        skills=skills,
        policy_client=policy_client,
        policy_model=args.policy_model,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt_template=judge_prompt_template,
        judge_system_message=DEFAULT_JUDGE_SYSTEM_MESSAGE,
        workspace=workspace,
        max_steps=args.max_steps,
        policy_temperature=args.policy_temperature,
        judge_temperature=args.judge_temperature,
        max_output_tokens=args.max_output_tokens,
        concurrency=args.concurrency,
        skill_concurrency=args.skill_concurrency,
        n_rollouts=args.n_rollouts,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_report_to_dict(report), indent=2))
    logger.info("wrote report to %s", args.output)

    for s in report.skills:
        logger.info("skill=%s mean_delta=%+.3f", s.skill_name, s.mean_delta)

    return 0


def main() -> int:  # pragma: no cover
    parser = _build_arg_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
