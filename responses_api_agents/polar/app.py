# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
from asyncio import Semaphore
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Body
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)


AGENT_DIR = Path(__file__).resolve().parent


class PolarAgentConfig(BaseResponsesAPIAgentConfig):
    """Configuration for running ProRL-Agent-Server Polar inference from Gym."""

    mode: Literal["direct", "slurm"] = Field(
        default="direct",
        description=(
            "direct runs simple_inference.py against an already-running OpenAI-compatible "
            "SGLang server; slurm submits submit_simple_inference.sh and waits for the result JSON."
        ),
    )
    prorl_root: str = Field(default="../ProRL-Agent-Server", description="Path to ProRL-Agent-Server.")
    inference_script: str = Field(
        default="examples/swegym_slime_grpo/simple_inference.py",
        description="simple_inference.py path, relative to prorl_root unless absolute.",
    )
    submit_script: str = Field(
        default="examples/swegym_slime_grpo/submit_simple_inference.sh",
        description="SLURM submit script path, relative to prorl_root unless absolute.",
    )
    base_url: str = Field(
        default="http://127.0.0.1:19000",
        description="OpenAI-compatible SGLang base URL used in direct mode.",
    )
    model_name: str = Field(default="Qwen/Qwen3.5-4B", description="Default model name for simple_inference.py.")
    max_tokens: int = Field(default=768, description="Default max tokens for simple_inference.py.")
    temperature: float = Field(default=0.2, description="Default temperature for simple_inference.py.")
    scenario: Literal["single_turn", "tool_multi_turn"] = Field(
        default="tool_multi_turn", description="simple_inference.py scenario."
    )
    top_logprobs: int = Field(default=3, description="top_logprobs requested from SGLang.")
    return_token_ids: bool = Field(default=True, description="Request token-id metadata when supported.")
    out_dir: str = Field(
        default="../ProRL-Agent-Server/tmp/gym_polar_inference",
        description="Directory for generated one-row JSONL files and Polar result JSON.",
    )
    concurrency: int = Field(default=1, description="Maximum concurrent Polar inference jobs.")
    request_timeout: float = Field(default=60 * 60, description="Timeout for one inference request, in seconds.")
    poll_interval: float = Field(default=10.0, description="Polling interval for SLURM result JSON.")
    success_reward: float = Field(
        default=0.0,
        description="Reward returned by /run for inference-only rollouts. No SWE-bench evaluation is run here.",
    )
    env: dict[str, str] = Field(default_factory=dict, description="Extra environment variables for direct mode.")
    slurm_env: dict[str, str] = Field(default_factory=dict, description="Extra environment variables for slurm mode.")


class PolarAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class PolarAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class PolarInferenceArtifacts:
    result: dict[str, Any]
    result_path: Path
    data_path: Path
    run_dir: Path
    row: dict[str, Any]
    stdout: str = ""
    stderr: str = ""


def _plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return dict(value)
    return dict(value)


def _resolve_existing_or_default(path: str, *, base: Optional[Path] = None) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw

    candidates = []
    if base is not None:
        candidates.append(base / raw)
    candidates.extend([PARENT_DIR / raw, Path.cwd() / raw, AGENT_DIR / raw])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_output_dir(path: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    return (PARENT_DIR / raw).resolve()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            item_dict = _plain_dict(item) if not isinstance(item, dict) else item
            text = item_dict.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts)
    return str(content)


def _input_to_chat_messages(body: NeMoGymResponseCreateParamsNonStreaming) -> list[dict[str, str]]:
    if isinstance(body.input, str):
        return [{"role": "user", "content": body.input}]

    messages: list[dict[str, str]] = []
    for item in body.input:
        item_dict = _plain_dict(item) if not isinstance(item, dict) else item
        if item_dict.get("type") != "message":
            continue
        role = item_dict.get("role")
        if role not in {"system", "user", "assistant", "developer"}:
            continue
        if role == "developer":
            role = "system"
        messages.append({"role": role, "content": _content_to_text(item_dict.get("content", ""))})
    return messages


def _metadata_issue_text(metadata: dict[str, Any], run_context: dict[str, Any]) -> str:
    for key in ("problem_statement", "issue", "prompt"):
        value = metadata.get(key, run_context.get(key))
        if isinstance(value, str) and value:
            return value

    instance = metadata.get("instance")
    if isinstance(instance, dict):
        value = instance.get("problem_statement")
        if isinstance(value, str) and value:
            return value

    return ""


def _body_to_swegym_row(
    body: NeMoGymResponseCreateParamsNonStreaming,
    run_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    run_context = run_context or {}

    body_metadata = _plain_dict(body.metadata)
    metadata = {}
    if isinstance(run_context.get("metadata"), dict):
        metadata.update(run_context["metadata"])
    metadata.update(body_metadata)

    for key in (
        "instance",
        "instance_id",
        "repo",
        "base_commit",
        "patch",
        "test_patch",
        "problem_statement",
        "hints_text",
        "created_at",
        "version",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "environment_setup_commit",
        "difficulty",
        "split",
        "base_image",
        "polar_image",
    ):
        if key in run_context and key not in metadata:
            metadata[key] = run_context[key]

    prompt = run_context.get("prompt")
    if not isinstance(prompt, list):
        prompt = _input_to_chat_messages(body)

    if not prompt:
        issue_text = _metadata_issue_text(metadata, run_context)
        if not issue_text:
            raise ValueError(
                "Polar inference needs either responses_create_params.input, a top-level prompt, "
                "or metadata.problem_statement."
            )
        prompt = [{"role": "user", "content": issue_text}]

    if "instance_id" not in metadata:
        metadata["instance_id"] = run_context.get("instance_id") or f"gym_polar_{uuid.uuid4().hex[:12]}"

    return {
        "prompt": prompt,
        "label": run_context.get("label", ""),
        "metadata": metadata,
    }


def _token_id(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"token_id:(-?\d+)", value)
        if match:
            return int(match.group(1))
    return None


def _logprobs_content(choice: dict[str, Any]) -> list[dict[str, Any]]:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return []
    content = logprobs.get("content")
    return [item for item in content if isinstance(item, dict)] if isinstance(content, list) else []


def _choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _message(response: dict[str, Any]) -> dict[str, Any]:
    message = _choice(response).get("message")
    return message if isinstance(message, dict) else {}


def _tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    calls = _message(response).get("tool_calls")
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


def _text(response: dict[str, Any]) -> str:
    content = _message(response).get("content")
    return content if isinstance(content, str) else ""


def _response_token_ids(response: dict[str, Any], choice: dict[str, Any]) -> list[int]:
    for candidate in (choice.get("token_ids"), response.get("token_ids")):
        if isinstance(candidate, list) and all(isinstance(item, int) for item in candidate):
            return list(candidate)

    token_ids: list[int] = []
    for item in _logprobs_content(choice):
        token_id = _token_id(item.get("token_id"))
        if token_id is None:
            token_id = _token_id(item.get("token"))
        if token_id is not None:
            token_ids.append(token_id)
    return token_ids


def _prompt_token_ids(response: dict[str, Any], choice: dict[str, Any]) -> list[int]:
    for candidate in (choice.get("input_token_ids"), response.get("prompt_token_ids")):
        if isinstance(candidate, list) and all(isinstance(item, int) for item in candidate):
            return list(candidate)
    return []


def _round_from_chat_response(name: str, response: dict[str, Any]) -> dict[str, Any]:
    choice = _choice(response)
    logprobs = _logprobs_content(choice)
    return {
        "name": name,
        "response": response,
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
        "text": _text(response),
        "tool_calls": _tool_calls(response),
        "response_token_ids": _response_token_ids(response, choice),
        "prompt_token_ids": _prompt_token_ids(response, choice),
        "logprobs_content": logprobs,
    }


def _normalized_rounds(result: dict[str, Any]) -> list[dict[str, Any]]:
    rounds = result.get("rounds")
    if isinstance(rounds, list) and rounds:
        normalized = []
        for item in rounds:
            if not isinstance(item, dict):
                continue
            if "response" in item:
                normalized.append(item)
        if normalized:
            return normalized

    response = result.get("response")
    if isinstance(response, dict):
        return [_round_from_chat_response(str(result.get("scenario") or "single_turn"), response)]
    return []


def _round_log_probs(round_data: dict[str, Any]) -> list[float]:
    logprobs: list[float] = []
    for item in round_data.get("logprobs_content") or []:
        if isinstance(item, dict) and isinstance(item.get("logprob"), (int, float)):
            logprobs.append(float(item["logprob"]))
    return logprobs


def _round_token_fields(round_data: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        "prompt_token_ids": list(round_data.get("prompt_token_ids") or []),
        "generation_token_ids": list(round_data.get("response_token_ids") or []),
        "generation_log_probs": _round_log_probs(round_data),
    }


def _tool_arguments(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    arguments = function.get("arguments") if isinstance(function, dict) else None
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    return "{}"


def _tool_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    return name if isinstance(name, str) and name else "unknown_tool"


def _usage_from_rounds(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    for round_data in rounds:
        usage = round_data.get("usage")
        if not isinstance(usage, dict):
            response = round_data.get("response")
            usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            continue
        input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)

    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def _default_response_object(model: str, output: list[dict[str, Any]], metadata: dict[str, str], usage: dict[str, Any]):
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "created_at": int(time.time()),
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": metadata,
        "model": model,
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "background": False,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "previous_response_id": None,
        "prompt": None,
        "reasoning": {
            "effort": None,
            "generate_summary": None,
            "summary": None,
        },
        "service_tier": "default",
        "status": "completed",
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "top_logprobs": 0,
        "truncation": "disabled",
        "usage": usage,
        "user": None,
        "prompt_cache_key": None,
        "safety_identifier": None,
        "store": True,
    }


def _build_response_from_result(
    result: dict[str, Any],
    *,
    result_path: Path,
    mode: str,
    model: str,
) -> NeMoGymResponse:
    rounds = _normalized_rounds(result)
    output_items: list[dict[str, Any]] = []
    tool_results = result.get("tool_results") if isinstance(result.get("tool_results"), dict) else {}

    for round_data in rounds:
        token_fields = _round_token_fields(round_data)
        for tool_call in round_data.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            call_id = str(tool_call.get("id") or f"call_{uuid.uuid4().hex}")
            name = _tool_name(tool_call)
            output_items.append(
                NeMoGymResponseFunctionToolCallForTraining(
                    arguments=_tool_arguments(tool_call),
                    call_id=call_id,
                    name=name,
                    id=f"fc_{uuid.uuid4().hex}",
                    status="completed",
                    **token_fields,
                ).model_dump()
            )

            if name in tool_results:
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=call_id,
                        output=json.dumps(tool_results[name], ensure_ascii=False),
                        status="completed",
                    ).model_dump()
                )

        text = round_data.get("text")
        if isinstance(text, str) and text:
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg_{uuid.uuid4().hex}",
                    content=[NeMoGymResponseOutputText(annotations=[], text=text)],
                    role="assistant",
                    status="completed",
                    **token_fields,
                ).model_dump()
            )

    if not output_items:
        output_items.append(
            NeMoGymResponseOutputMessageForTraining(
                id=f"msg_{uuid.uuid4().hex}",
                content=[NeMoGymResponseOutputText(annotations=[], text=str(result.get("text") or ""))],
                role="assistant",
                status="completed",
                prompt_token_ids=[],
                generation_token_ids=[],
                generation_log_probs=[],
            ).model_dump()
        )

    metadata = {
        "polar_mode": mode,
        "polar_result_path": str(result_path),
        "polar_instance_id": str(result.get("instance_id") or ""),
        "polar_scenario": str(result.get("scenario") or ""),
        "polar_token_summary": json.dumps(result.get("token_summary") or {}, ensure_ascii=False),
    }
    return NeMoGymResponse.model_validate(
        _default_response_object(
            model=model,
            output=output_items,
            metadata=metadata,
            usage=_usage_from_rounds(rounds),
        )
    )


class PolarAgent(SimpleResponsesAPIAgent):
    config: PolarAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        async with self.sem:
            artifacts = await self._execute_inference(body)
        model = body.model or self.config.model_name
        return _build_response_from_result(
            artifacts.result,
            result_path=artifacts.result_path,
            mode=self.config.mode,
            model=str(model),
        )

    async def run(self, body: PolarAgentRunRequest = Body()) -> PolarAgentVerifyResponse:
        run_context = body.model_dump(exclude={"responses_create_params"}, exclude_none=True)
        async with self.sem:
            artifacts = await self._execute_inference(body.responses_create_params, run_context=run_context)

        model = body.responses_create_params.model or self.config.model_name
        response = _build_response_from_result(
            artifacts.result,
            result_path=artifacts.result_path,
            mode=self.config.mode,
            model=str(model),
        )
        return PolarAgentVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=response,
            reward=self.config.success_reward,
            metadata={
                "mode": self.config.mode,
                "data_path": str(artifacts.data_path),
                "result_path": str(artifacts.result_path),
                "run_dir": str(artifacts.run_dir),
                "instance_id": artifacts.result.get("instance_id"),
                "scenario": artifacts.result.get("scenario"),
                "token_summary": artifacts.result.get("token_summary"),
                "stdout_tail": artifacts.stdout[-4000:],
                "stderr_tail": artifacts.stderr[-4000:],
            },
        )

    async def _execute_inference(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        run_context: Optional[dict[str, Any]] = None,
    ) -> PolarInferenceArtifacts:
        prorl_root = _resolve_existing_or_default(self.config.prorl_root)
        out_root = _resolve_output_dir(self.config.out_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        row = _body_to_swegym_row(body, run_context)
        instance_id = str(row.get("metadata", {}).get("instance_id") or "unknown")
        run_dir = out_root / f"{int(time.time())}_{_safe_name(instance_id)}_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)

        data_path = run_dir / "input.jsonl"
        data_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

        if self.config.mode == "direct":
            result, result_path, stdout, stderr = await self._run_direct(body, prorl_root, data_path, run_dir)
        else:
            result, result_path, stdout, stderr = await self._run_slurm(body, prorl_root, data_path, run_dir)

        return PolarInferenceArtifacts(
            result=result,
            result_path=result_path,
            data_path=data_path,
            run_dir=run_dir,
            row=row,
            stdout=stdout,
            stderr=stderr,
        )

    def _inference_args(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        data_path: Path,
        result_path: Path,
    ) -> list[str]:
        model = str(body.model or self.config.model_name)
        max_tokens = int(body.max_output_tokens or self.config.max_tokens)
        temperature = float(body.temperature if body.temperature is not None else self.config.temperature)
        top_logprobs = int(body.top_logprobs if body.top_logprobs is not None else self.config.top_logprobs)

        args = [
            "--data",
            str(data_path),
            "--index",
            "0",
            "--base-url",
            self.config.base_url,
            "--model",
            model,
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            str(temperature),
            "--scenario",
            self.config.scenario,
            "--top-logprobs",
            str(top_logprobs),
            "--out",
            str(result_path),
        ]
        if self.config.return_token_ids:
            args.append("--return-token-ids")
        return args

    async def _run_direct(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        prorl_root: Path,
        data_path: Path,
        run_dir: Path,
    ) -> tuple[dict[str, Any], Path, str, str]:
        script_path = _resolve_existing_or_default(self.config.inference_script, base=prorl_root)
        if not script_path.exists():
            raise FileNotFoundError(f"Polar inference script not found: {script_path}")

        result_path = run_dir / "result.json"
        cmd = [sys.executable, str(script_path), *self._inference_args(body, data_path=data_path, result_path=result_path)]
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in self.config.env.items()})

        stdout, stderr = await self._run_process(cmd, cwd=prorl_root, env=env, timeout=self.config.request_timeout)
        return self._read_result_json(result_path), result_path, stdout, stderr

    async def _run_slurm(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        prorl_root: Path,
        data_path: Path,
        run_dir: Path,
    ) -> tuple[dict[str, Any], Path, str, str]:
        submit_path = _resolve_existing_or_default(self.config.submit_script, base=prorl_root)
        if not submit_path.exists():
            raise FileNotFoundError(f"Polar SLURM submit script not found: {submit_path}")

        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in self.config.slurm_env.items()})
        env.update(
            {
                "DATA_PATH": str(data_path),
                "SAMPLE_INDEX": "0",
                "OUT_DIR": str(run_dir),
                "MODEL_NAME": str(body.model or self.config.model_name),
                "MAX_TOKENS": str(int(body.max_output_tokens or self.config.max_tokens)),
                "TEMPERATURE": str(
                    float(body.temperature if body.temperature is not None else self.config.temperature)
                ),
                "INFERENCE_SCENARIO": self.config.scenario,
                "TOP_LOGPROBS": str(int(body.top_logprobs if body.top_logprobs is not None else self.config.top_logprobs)),
                "RETURN_TOKEN_IDS": "1" if self.config.return_token_ids else "0",
            }
        )

        stdout, stderr = await self._run_process(
            ["bash", str(submit_path)],
            cwd=prorl_root,
            env=env,
            timeout=min(self.config.request_timeout, 300),
        )

        result_path = self._parse_slurm_result_path(stdout) or self._infer_slurm_result_path(stdout, run_dir)
        result = await self._wait_for_result_json(result_path)
        return result, result_path, stdout, stderr

    async def _run_process(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: float,
    ) -> tuple[str, str]:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TimeoutError(f"Polar command timed out after {timeout} seconds: {cmd}") from None

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Polar command failed with exit code {proc.returncode}: {cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        return stdout, stderr

    @staticmethod
    def _parse_slurm_result_path(stdout: str) -> Optional[Path]:
        match = re.search(r"^Result path:\s*(.+)$", stdout, flags=re.MULTILINE)
        if match:
            return Path(match.group(1).strip())
        return None

    @staticmethod
    def _infer_slurm_result_path(stdout: str, run_dir: Path) -> Path:
        match = re.search(r"Submitted SLURM job:\s*([^\s]+)", stdout)
        if not match:
            raise RuntimeError(f"Could not find SLURM job id or result path in submit output:\n{stdout}")
        return run_dir / f"result-{match.group(1)}.json"

    async def _wait_for_result_json(self, result_path: Path) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.request_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if result_path.exists():
                try:
                    return self._read_result_json(result_path)
                except json.JSONDecodeError as exc:
                    last_error = exc
            await asyncio.sleep(self.config.poll_interval)
        raise TimeoutError(f"Timed out waiting for Polar result JSON at {result_path}: {last_error}") from last_error

    @staticmethod
    def _read_result_json(result_path: Path) -> dict[str, Any]:
        with result_path.open(encoding="utf-8") as f:
            return json.load(f)


if __name__ == "__main__":
    PolarAgent.run_webserver()
