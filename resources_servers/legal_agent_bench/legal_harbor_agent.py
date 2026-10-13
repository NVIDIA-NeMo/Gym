# SPDX-FileCopyrightText: Copyright (c) 2026 Harvey AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Harbor agent adapter for Legal Agent Bench inside NeMo Gym."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from resources_servers.legal_agent_bench.prepare import (
    DEFAULT_SKILLS_DIR,
    discover_harness_skills,
    resolve_repo_path,
    validate_harness_skills,
)


_PACKAGE_DIR = Path(__file__).resolve().parent
_VENDOR_ROOT = _PACKAGE_DIR / "vendor" / "harvey_labs"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from harness.adapters.base import ModelResponse  # noqa: E402
from harness.adapters.openai_compatible import OpenAICompatibleAdapter  # noqa: E402
from lab_harbor.tools import CONTAINER_TOOL_RUNNER_PATH, get_all_tool_definitions  # noqa: E402


try:  # pragma: no cover - Harbor is installed in the Harbor agent server venv.
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ImportError:  # pragma: no cover

    class BaseAgent:  # type: ignore[no-redef]
        def __init__(self, logs_dir: Path, model_name: str | None = None, logger=None, **_kwargs):
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name
            self.logger = logger or logging.getLogger(__name__)

    BaseEnvironment = Any  # type: ignore
    AgentContext = Any  # type: ignore


LAB_TASK_ID_MARKER = "lab_task_id:"
AGENT_VERSION = "0.1.0"
INITIAL_USER_PROMPT = "Please begin working on the task described in the system prompt."
REQUIRED_TASK_KEYS = {"title", "instructions", "criteria"}
REQUIRED_CRITERION_KEYS = {"id", "title", "match_criteria"}
SYSTEM_PROMPT_PATH = _VENDOR_ROOT / "harness" / "system-prompt.md"
SYSTEM_PROMPT_PREAMBLE = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


class HarborToolExecutor:
    """Execute LAB tools inside a Harbor environment."""

    def __init__(
        self,
        environment: BaseEnvironment,
        *,
        host_vdr_dir: Path,
        shell_timeout: int = 60,
        max_output_chars: int = 16_384,
    ) -> None:
        self.environment = environment
        self.host_vdr_dir = host_vdr_dir
        self.shell_timeout = shell_timeout
        self.max_output_chars = max_output_chars
        self.files_read: list[str] = []
        self.files_written = 0
        self.files_edited = 0
        self.bash_command_count = 0
        self.glob_count = 0
        self.grep_count = 0

    async def preflight(self) -> None:
        result, _metrics = await self._run_container_tool("preflight", {})
        if result.startswith("Error:"):
            raise RuntimeError(result)

    async def execute(self, tool_name: str, arguments: str | dict) -> str:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Error: invalid JSON arguments: {arguments}"

        if tool_name == "bash":
            return await self._bash(arguments.get("command", ""))
        if tool_name in {"read", "write", "write_docx", "edit", "glob", "grep"}:
            result, metrics = await self._run_container_tool(tool_name, arguments)
            self._apply_container_metrics(metrics)
            return result
        return f"Error: unknown tool: {tool_name}"

    async def _bash(self, command: str) -> str:
        if not command:
            return "Error: command is required"
        self.bash_command_count += 1
        result = await self.environment.exec(
            command,
            cwd="/workspace/output",
            env=self._tool_env(),
            timeout_sec=self.shell_timeout,
        )
        return self._format_exec_result(result)

    async def _run_container_tool(self, tool_name: str, arguments: dict) -> tuple[str, dict]:
        payload = shlex.quote(json.dumps(arguments))
        command = f"python {shlex.quote(CONTAINER_TOOL_RUNNER_PATH)} {shlex.quote(tool_name)} {payload}"
        result = await self.environment.exec(
            command,
            cwd="/workspace/output",
            env=self._tool_env(),
            timeout_sec=self.shell_timeout,
        )
        output = self._format_exec_result(result, merge_stderr=False, truncate=False)
        parsed = self._parse_container_tool_output(output)
        if parsed is None:
            return f"Error: invalid container tool response: {self._truncate(output)}", {}
        tool_result = str(parsed.get("result", ""))
        metrics = dict(parsed.get("metrics") or {})
        if tool_name == "read" and _full_read_requested(arguments.get("limit"), "limit" in arguments):
            return tool_result, metrics
        return self._truncate(tool_result), metrics

    def _tool_env(self) -> dict[str, str]:
        return {
            "VDR_DIR": "/workspace/vdr",
            "OUTPUT_DIR": "/workspace/output",
            "WORKSPACE_DIR": "/workspace/workspace",
            "SKILLS_DIR": "/workspace/skills",
            "HOME": "/workspace/workspace",
        }

    def _format_exec_result(self, result, *, merge_stderr: bool = True, truncate: bool = True) -> str:
        output = result.stdout or ""
        if result.stderr and merge_stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.return_code != 0:
            if result.stderr and not merge_stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            output += f"\n(exit code {result.return_code})"
        output = output or "(no output)"
        return self._truncate(output) if truncate else output

    def _parse_container_tool_output(self, output: str) -> dict | None:
        for line in reversed(output.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _apply_container_metrics(self, metrics: dict) -> None:
        if file_read := metrics.get("file_read"):
            self.files_read.append(str(file_read))
        self.files_written += int(metrics.get("files_written") or 0)
        self.files_edited += int(metrics.get("files_edited") or 0)
        self.glob_count += int(metrics.get("glob_count") or 0)
        self.grep_count += int(metrics.get("grep_count") or 0)

    def _truncate(self, output: str) -> str:
        if len(output) > self.max_output_chars:
            return output[: self.max_output_chars] + "\n[output truncated]"
        return output

    def get_metrics(self) -> dict:
        all_vdr_files = sorted(
            str(path.relative_to(self.host_vdr_dir)) for path in self.host_vdr_dir.rglob("*") if path.is_file()
        )
        unique_reads = list(dict.fromkeys(self.files_read))
        skipped = [path for path in all_vdr_files if path not in unique_reads]
        return {
            "documents_read": len(unique_reads),
            "documents_read_list": unique_reads,
            "documents_skipped": len(skipped),
            "documents_skipped_list": skipped,
            "total_vdr_files": len(all_vdr_files),
            "bash_commands": self.bash_command_count,
            "files_written": self.files_written,
            "files_edited": self.files_edited,
            "glob_searches": self.glob_count,
            "grep_searches": self.grep_count,
            "tool_runtime": "harbor",
        }


class LegalAgentBenchHarborAgent(BaseAgent):
    """NeMo Gym-specific Harbor agent for Legal Agent Bench."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, logger=logger)
        responses_create_params = dict(kwargs.pop("responses_create_params", {}) or {})
        api_base = kwargs.pop("api_base", None)

        self.agent_id = kwargs.pop("agent_id", "legal-agent-bench")
        self.agent_config_id = kwargs.pop("agent_config_id", self.agent_id)
        self.run_timestamp = kwargs.pop("run_timestamp", "nemo-gym")
        self.model = kwargs.pop("model", model_name)
        temperature = kwargs.pop("temperature", responses_create_params.get("temperature", 1.0))
        self.temperature = 1.0 if temperature is None else float(temperature)
        self.reasoning_effort = kwargs.pop("reasoning_effort", _reasoning_effort(responses_create_params))
        self.max_turns = int(kwargs.pop("max_turns", 60))
        self.shell_timeout = int(kwargs.pop("shell_timeout", 60))
        self.skills = kwargs.pop("skills", None)
        self.skills_dir = resolve_repo_path(kwargs.pop("skills_dir", DEFAULT_SKILLS_DIR))

        if api_base and not kwargs.get("agent_model_base_url"):
            kwargs["agent_model_base_url"] = api_base
        if responses_create_params.get("max_output_tokens") and not kwargs.get("agent_model_max_tokens"):
            kwargs["agent_model_max_tokens"] = responses_create_params["max_output_tokens"]
        if kwargs.get("agent_model_top_p") is None:
            kwargs["agent_model_top_p"] = responses_create_params.get("top_p", 0.95)

        self.adapter_kwargs = kwargs

    @staticmethod
    def name() -> str:
        return "legal-agent-bench"

    def version(self) -> str | None:
        return AGENT_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(
            "mkdir -p /workspace/vdr /workspace/output /workspace/workspace /workspace/skills /logs/agent",
            timeout_sec=60,
        )

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        task_id = _extract_task_id(instruction)
        task_dir = _task_dir_from_environment(environment)
        task = _load_task_from_mirror(task_dir, task_id=task_id)
        docs_dir = Path(task["docs_dir"])

        skill_names = self._skill_names()

        await self._hydrate_environment(environment, docs_dir)

        artifact_dir = Path(self.logs_dir) / "artifacts" / "lab-run"
        output_artifact = artifact_dir / "output"
        workspace_artifact = artifact_dir / "workspace"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        tools = get_all_tool_definitions()
        config = self._result_config(task_id, task_dir, tools, skill_names)
        (artifact_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        adapter = self._create_adapter()
        system_prompt = SYSTEM_PROMPT_PREAMBLE
        if skill_names:
            system_prompt += _load_skills(skill_names, self.skills_dir)
        system_prompt += "\n\n## Task\n\n" + task["system_prompt"]

        executor = HarborToolExecutor(environment, host_vdr_dir=docs_dir, shell_timeout=self.shell_timeout)
        await executor.preflight()

        transcript_path = artifact_dir / "transcript.jsonl"
        result = await _run_agent_async(
            adapter=adapter,
            system_prompt=system_prompt,
            tool_executor=executor,
            tools=tools,
            max_turns=self.max_turns,
            transcript_path=transcript_path,
        )

        await environment.download_dir("/workspace/output", output_artifact)
        await environment.download_dir("/workspace/workspace", workspace_artifact)

        metrics = {
            "model": self.model,
            "task": task_id,
            "run_id": self._run_id(task_id),
            "turn_count": result["turn_count"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["input_tokens"] + result["output_tokens"],
            "wall_clock_seconds": result["wall_clock_seconds"],
            "finished_cleanly": result["finished_cleanly"],
            "context_overflow": result["context_overflow"],
            "model_error": result.get("model_error"),
            **result["tool_metrics"],
        }
        (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _write_trajectory(
            logs_dir=Path(self.logs_dir),
            instruction=instruction,
            transcript_path=transcript_path,
            metrics=metrics,
            model_name=self.model,
            agent_name=self.name(),
        )
        _write_agent_error_flags(Path(self.logs_dir), metrics)
        _populate_context(context, result, metrics, task_id, self.agent_id, artifact_dir)

    async def _hydrate_environment(self, environment: BaseEnvironment, docs_dir: Path) -> None:
        validate_harness_skills(self.skills_dir)
        await environment.exec("rm -rf /workspace/vdr /workspace/skills && mkdir -p /workspace", timeout_sec=60)
        await environment.upload_dir(docs_dir, "/workspace/vdr")
        await environment.upload_dir(self.skills_dir, "/workspace/skills")
        await environment.exec("mkdir -p /workspace/output /workspace/workspace", timeout_sec=60)

    def _skill_names(self) -> list[str]:
        available_skills = discover_harness_skills(self.skills_dir)
        skill_names = available_skills if self.skills is None else list(self.skills)
        unknown_skills = sorted(set(skill_names) - set(available_skills))
        if unknown_skills:
            raise FileNotFoundError(
                f"Requested Legal Agent Bench skill(s) are missing from {self.skills_dir}: {', '.join(unknown_skills)}"
            )
        return skill_names

    def _create_adapter(self) -> OpenAICompatibleAdapter:
        kwargs = dict(self.adapter_kwargs)
        base_url = kwargs.pop("agent_model_base_url", None)
        if not base_url:
            raise ValueError("LegalAgentBenchHarborAgent requires api_base from the NeMo Harbor bridge")
        return OpenAICompatibleAdapter(
            model=self.model or "policy_model",
            base_url=base_url,
            api_key=kwargs.pop("agent_model_api_key", None),
            temperature=self.temperature,
            max_tokens=int(kwargs.pop("agent_model_max_tokens", 32768)),
            reasoning_effort=self.reasoning_effort,
            timeout_seconds=kwargs.pop("agent_model_timeout_seconds", None),
            omit_temperature=kwargs.pop("agent_model_omit_temperature", None),
            chat_template_kwargs=_coerce_json_object(kwargs.pop("agent_model_chat_template_kwargs", None)),
            top_p=kwargs.pop("agent_model_top_p", None),
        )

    def _result_config(self, task_id: str, task_dir: Path, tools: list[dict], skill_names: list[str]) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_config_id": self.agent_config_id,
            "model": self.model,
            "task": task_id,
            "task_dir": str(task_dir),
            "run_id": self._run_id(task_id),
            "max_turns": self.max_turns,
            "temperature": self.temperature,
            "shell_timeout": self.shell_timeout,
            "reasoning_effort": self.reasoning_effort,
            "agent_model_base_url": self.adapter_kwargs.get("agent_model_base_url"),
            "agent_model_max_tokens": self.adapter_kwargs.get("agent_model_max_tokens"),
            "agent_model_timeout_seconds": self.adapter_kwargs.get("agent_model_timeout_seconds"),
            "tool_runtime": "harbor",
            "skills": skill_names,
            "tool_count": len(tools),
            "tool_names": [tool["name"] for tool in tools],
            "tools": tools,
            "harbor": {"logs_dir": str(self.logs_dir)},
        }

    def _run_id(self, task_id: str) -> str:
        return f"{task_id}/{self.agent_id}/{self.run_timestamp}"


async def _run_agent_async(
    *,
    adapter: OpenAICompatibleAdapter,
    system_prompt: str,
    tool_executor: HarborToolExecutor,
    tools: list[dict],
    max_turns: int,
    transcript_path: Path,
) -> dict:
    messages = [adapter.make_system_message(system_prompt), adapter.make_user_message(INITIAL_USER_PROMPT)]
    total_input_tokens = 0
    total_output_tokens = 0
    turn_count = 0
    finished_cleanly = False
    context_overflow = False
    model_error = None
    empty_response_count = 0
    start_time = time.time()

    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    with transcript_path.open("w", encoding="utf-8") as transcript_file:
        for turn in range(max_turns):
            turn_count = turn + 1
            _log_model_input(transcript_file, turn_count, messages, tools, total_input_tokens, total_output_tokens)
            try:
                response = await _chat_with_timeout(adapter, messages, tools)
            except Exception as exc:
                err_msg = str(exc)
                _log_model_error(transcript_file, turn_count, exc)
                if _is_context_overflow_error(err_msg):
                    context_overflow = True
                    break
                model_error = err_msg
                break

            messages.append(response.message)
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            _log_turn(transcript_file, turn_count, "assistant", response)

            if not response.tool_calls:
                if not (response.text or "").strip():
                    empty_response_count += 1
                    if empty_response_count <= 2 and turn_count < max_turns:
                        messages.append(
                            adapter.make_user_message(
                                "Your last response was empty and did not call any tools. Continue the task. "
                                "Use the available tools to inspect the documents and write the required deliverables."
                            )
                        )
                        continue
                    break
                finished_cleanly = True
                break
            empty_response_count = 0

            tool_results = []
            for tool_call in response.tool_calls:
                result = await tool_executor.execute(tool_call.name, tool_call.arguments)
                _log_tool(transcript_file, turn_count, tool_call.id, tool_call.name, tool_call.arguments, result)
                tool_results.append((tool_call, result))

            result_messages = adapter.make_tool_result_messages(
                [(tool_call.id, result) for tool_call, result in tool_results]
            )
            _log_tool_result_messages(transcript_file, turn_count, result_messages)
            messages.extend(result_messages)

    elapsed = time.time() - start_time
    return {
        "messages": messages,
        "turn_count": turn_count,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "wall_clock_seconds": round(elapsed, 2),
        "finished_cleanly": (not context_overflow and finished_cleanly),
        "context_overflow": context_overflow,
        "model_error": model_error,
        "tool_metrics": tool_executor.get_metrics(),
    }


async def _chat_with_timeout(
    adapter: OpenAICompatibleAdapter, messages: list[dict], tools: list[dict]
) -> ModelResponse:
    timeout_seconds = getattr(adapter, "timeout_seconds", None)
    chat_call = adapter.chat(messages, tools)
    if not timeout_seconds:
        return await chat_call
    try:
        return await asyncio.wait_for(chat_call, timeout=float(timeout_seconds))
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"agent model request exceeded timeout of {float(timeout_seconds):g}s") from exc


def _load_task_from_mirror(task_dir: Path, *, task_id: str) -> dict:
    config_path = task_dir / "task.json"
    if not config_path.exists():
        raise FileNotFoundError(f"task.json not found in Harbor task mirror: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_task_config(config, config_path)
    docs_dir = task_dir / config.get("docs_dir", "documents")
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found in Harbor task mirror: {docs_dir}")
    system_prompt = config.get("instructions") or (task_dir / "instruction.md").read_text(encoding="utf-8")
    return {
        "name": task_id,
        "task_dir": str(task_dir),
        "docs_dir": str(docs_dir),
        "system_prompt": system_prompt,
        "config": config,
    }


def _validate_task_config(config: dict, task_path: Path) -> None:
    for key in REQUIRED_TASK_KEYS:
        if key not in config:
            raise ValueError(f"{task_path}: missing required key '{key}'")
    criteria = config["criteria"]
    if not isinstance(criteria, list) or not criteria:
        raise ValueError(f"{task_path}: 'criteria' must be a non-empty list")
    for i, criterion in enumerate(criteria):
        for key in REQUIRED_CRITERION_KEYS:
            if key not in criterion:
                raise ValueError(f"{task_path}: criterion {i} missing required key '{key}'")


def _task_dir_from_environment(environment: BaseEnvironment) -> Path:
    environment_dir = Path(getattr(environment, "environment_dir"))
    task_dir = environment_dir.parent
    if not (task_dir / "task.toml").exists():
        raise FileNotFoundError(f"Could not infer Harbor task mirror from environment_dir={environment_dir}")
    return task_dir


def _extract_task_id(instruction: str) -> str:
    pattern = rf"{re.escape(LAB_TASK_ID_MARKER)}\s*([^\s<]+)"
    match = re.search(pattern, instruction)
    if not match:
        raise ValueError("Generated Harbor instruction is missing lab_task_id marker")
    return match.group(1).strip()


def _load_skills(skill_names: list[str], skills_dir: Path) -> str:
    sections = []
    for name in skill_names:
        skill_path = skills_dir / name / "SKILL.md"
        if not skill_path.is_file():
            raise FileNotFoundError(f"Legal Agent Bench skill manual not found: {skill_path}")
        sections.append(f"\n\n## Skill: {name}\n\n{skill_path.read_text(encoding='utf-8')}")
    return "".join(sections)


def _coerce_json_object(value: Any) -> dict | None:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("agent_model_chat_template_kwargs must be a JSON object")


def _reasoning_effort(params: dict) -> str | None:
    reasoning = params.get("reasoning") or {}
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        return str(effort) if effort else None
    return None


def _is_context_overflow_error(message: str) -> bool:
    lower = message.lower()
    return any(
        phrase in lower
        for phrase in (
            "prompt is too long",
            "context_length_exceeded",
            "context length",
            "maximum input length",
            "too many tokens",
        )
    )


def _log_model_input(
    f, turn: int, messages: list[dict], tools: list[dict], prior_input_tokens: int, prior_output_tokens: int
) -> None:
    entry = {
        "turn": turn,
        "role": "model_input",
        "completed_turns": turn - 1,
        "messages": messages,
        "tools": tools,
        "prior_input_tokens": prior_input_tokens,
        "prior_output_tokens": prior_output_tokens,
    }
    f.write(json.dumps(entry) + "\n")
    f.flush()


def _log_turn(f, turn: int, role: str, response: ModelResponse) -> None:
    entry = {
        "turn": turn,
        "role": role,
        "message": response.message,
        "text": response.text[:500] if response.text else None,
        "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls]
        if response.tool_calls
        else None,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }
    f.write(json.dumps(entry) + "\n")
    f.flush()


def _log_tool(f, turn: int, tool_call_id: str, name: str, arguments: str, result: str) -> None:
    entry = {
        "turn": turn,
        "role": "tool",
        "tool_call_id": tool_call_id,
        "tool_name": name,
        "arguments": arguments if isinstance(arguments, str) else str(arguments),
        "result_preview": result[:1000],
    }
    f.write(json.dumps(entry) + "\n")
    f.flush()


def _log_tool_result_messages(f, turn: int, result_messages: list[dict]) -> None:
    f.write(json.dumps({"turn": turn, "role": "tool_results", "messages": result_messages}) + "\n")
    f.flush()


def _log_model_error(f, turn: int, exc: Exception) -> None:
    f.write(
        json.dumps({"turn": turn, "role": "model_error", "error": str(exc), "error_type": exc.__class__.__name__})
        + "\n"
    )
    f.flush()


def _write_trajectory(
    *,
    logs_dir: Path,
    instruction: str,
    transcript_path: Path,
    metrics: dict,
    model_name: str | None,
    agent_name: str,
) -> None:
    steps: list[dict[str, Any]] = [{"step_id": 1, "source": "user", "message": instruction}]
    entries = []
    if transcript_path.exists():
        entries = [
            json.loads(line) for line in transcript_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    tool_entries_by_turn: dict[int, list[dict]] = {}
    for entry in entries:
        if entry.get("role") == "tool":
            tool_entries_by_turn.setdefault(int(entry.get("turn") or 0), []).append(entry)
    for entry in entries:
        if entry.get("role") != "assistant":
            continue
        turn = int(entry.get("turn") or 0)
        message = entry.get("message") or {}
        content = message.get("content") or entry.get("text") or ""
        tool_calls = []
        for tc in entry.get("tool_calls") or []:
            tool_calls.append(
                {
                    "tool_call_id": tc.get("id"),
                    "function_name": tc.get("name"),
                    "arguments": _parse_tool_arguments(tc.get("arguments")),
                }
            )
        observations = [{"content": t.get("result_preview", "")} for t in tool_entries_by_turn.get(turn, [])]
        step: dict[str, Any] = {
            "step_id": len(steps) + 1,
            "source": "agent",
            "model_name": model_name,
            "message": content,
            "metrics": {
                "prompt_tokens": entry.get("input_tokens") or 0,
                "completion_tokens": entry.get("output_tokens") or 0,
            },
        }
        if tool_calls:
            step["tool_calls"] = tool_calls
            step["observation"] = {"results": observations}
        steps.append(step)

    trajectory = {
        "schema_version": "ATIF-v1.5",
        "session_id": logs_dir.parent.name,
        "agent": {"name": agent_name, "version": AGENT_VERSION, "model_name": model_name},
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": metrics.get("input_tokens", 0),
            "total_completion_tokens": metrics.get("output_tokens", 0),
            "total_cached_tokens": 0,
        },
    }
    (logs_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2), encoding="utf-8")


def _parse_tool_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _write_agent_error_flags(logs_dir: Path, metrics: dict) -> None:
    flags = {
        "context_length_exceeded": bool(metrics.get("context_overflow")),
        "memory_limit_exceeded": False,
    }
    (logs_dir / "agent_error_flags.json").write_text(json.dumps(flags), encoding="utf-8")


def _populate_context(
    context: AgentContext,
    result: dict,
    metrics: dict,
    task_id: str,
    agent_id: str,
    artifact_dir: Path,
) -> None:
    context.n_input_tokens = result["input_tokens"]
    context.n_output_tokens = result["output_tokens"]
    context.metadata = {
        "task": task_id,
        "agent_id": agent_id,
        "lab_run_id": metrics["run_id"],
        "artifact_dir": str(artifact_dir),
        "finished_cleanly": metrics["finished_cleanly"],
        "model_error": metrics.get("model_error"),
        "turn_count": metrics["turn_count"],
        "tool_metrics": {
            key: value
            for key, value in metrics.items()
            if key
            in {
                "documents_read",
                "documents_skipped",
                "total_vdr_files",
                "bash_commands",
                "files_written",
                "files_edited",
                "glob_searches",
                "grep_searches",
            }
        },
    }


def _full_read_requested(limit: int | None, limit_provided: bool) -> bool:
    if not limit_provided:
        return False
    if limit is None:
        return True
    return not isinstance(limit, bool) and limit == 0
