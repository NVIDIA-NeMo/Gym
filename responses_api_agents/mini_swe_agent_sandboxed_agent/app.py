# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""mini-swe-agent (v2.4.6) control loop running in the NeMo Gym task sandbox.

This is the harness Artificial Analysis runs Terminal-Bench with ("mini-SWE-agent v2.4.6, 500 steps,
30 s command timeout, interactive mini config and prompts, native bash tool, no compaction"). It is a
faithful asyncio port of three upstream classes:

* ``minisweagent.agents.default.DefaultAgent``      -> :class:`NeMoGymMiniSweAgent`
* ``minisweagent.models.litellm_response_model``    -> :class:`NeMoGymResponsesModel` (Gym Responses API)
* ``minisweagent.environments.local.LocalEnvironment`` -> :class:`NeMoGymSandboxShellEnvironment`

Prompts (``config/mini.yaml``), the bash tool schema, the tool-call parser and the observation formatter
are imported from the pinned ``mini-swe-agent`` package so they stay byte-identical to upstream. The task
sandbox is created by the resources server (``/seed_session``) and scored by it (``/verify``), exactly
like ``terminus_2_sandboxed_agent``.
"""

import os


# Must run before the first ``minisweagent`` import: it prints a banner and, without MSWEA_CONFIGURED,
# would try to launch the interactive first-time setup.
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
os.environ.setdefault("MSWEA_CONFIGURED", "true")

import asyncio  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402
from time import perf_counter  # noqa: E402
from traceback import format_exc  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402
from uuid import uuid4  # noqa: E402

import yaml  # noqa: E402
from fastapi import Request  # noqa: E402
from jinja2 import StrictUndefined, Template  # noqa: E402
from minisweagent import __version__ as MINI_SWE_AGENT_VERSION  # noqa: E402
from minisweagent.config import builtin_config_dir  # noqa: E402
from minisweagent.exceptions import (  # noqa: E402
    FormatError,
    InterruptAgentFlow,
    LimitsExceeded,
    Submitted,
    TimeExceeded,
)
from minisweagent.models.utils.actions_toolcall_response import (  # noqa: E402
    BASH_TOOL_RESPONSE_API,
    finish_reason_from_responses_api,
    format_toolcall_observation_messages,
    parse_toolcall_actions_response,
)
from minisweagent.utils.serialize import recursive_merge  # noqa: E402
from pydantic import ConfigDict, Field  # noqa: E402

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse  # noqa: E402
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent  # noqa: E402
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef  # noqa: E402
from nemo_gym.global_config import get_global_config_dict  # noqa: E402
from nemo_gym.openai_utils import (  # noqa: E402
    NeMoGymAsyncOpenAI,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.sandbox import AsyncSandbox, create_provider  # noqa: E402
from nemo_gym.sandbox.config import resolve_provider_config  # noqa: E402
from nemo_gym.server_utils import (  # noqa: E402
    SESSION_ID_KEY,
    get_response_json,
    get_server_url,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)
from responses_api_agents.mini_swe_agent_sandboxed_agent.episode_export import (  # noqa: E402
    episode_export,
    request_snapshot,
    snapshot_hash,
)
from responses_api_agents.mini_swe_agent_sandboxed_agent.sandbox_identity import (  # noqa: E402
    SandboxIdentityMismatch,
    checked_exec,
)


ROUTING_KEY_HEADER = "x-session-affinity"

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
# Upstream's tool dict verbatim plus ``strict``: Gym's Responses request schema (FunctionToolParam) requires the
# field, and OpenAI treats a missing ``strict`` as false, so this is wire-equivalent to what litellm sends.
BASH_TOOL = {**BASH_TOOL_RESPONSE_API, "strict": False}


class ModelCallTimeout(RuntimeError):
    """The model service exhausted its call deadlines before the episode deadline."""


class MiniSweAgentSandboxedConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    # mini-swe-agent config. ``None`` loads the packaged ``config/mini.yaml``; overrides are merged on top
    # with mini-swe-agent's own ``recursive_merge`` (same semantics as ``mini-swe-agent -c key=value``).
    mini_config_path: Optional[str] = None
    mini_config_overrides: Dict[str, Any] = Field(default_factory=dict)
    # AgentConfig fields (upstream defaults: step_limit 0, cost_limit 3.0). Artificial Analysis: 500 / off.
    step_limit: int = 500
    cost_limit: float = 0.0
    wall_time_limit_seconds: int = 0
    max_consecutive_format_errors: int = 3
    # LocalEnvironmentConfig.timeout
    command_timeout_s: int = 30
    # Upstream runs actions through ``subprocess.Popen(shell=True)`` -> ``/bin/sh -c``.
    shell: str = "/bin/sh"
    # Upstream's Responses-API model echoes every output item back as input, reasoning included. Some
    # endpoints 500 on echoed reasoning items, so this is opt-in. Reasoning is always kept in the rollout.
    replay_reasoning_items: bool = False
    dump_trajectory_dir: Optional[str] = None
    debug: bool = False

    sandbox_provider: str
    sandbox_config: Dict[str, Any] = Field(default_factory=dict)
    sandbox_timeout: float = Field(gt=0, allow_inf_nan=False)
    # OpenSandbox Kubernetes deployments use <sandbox-id>-0; opt in for that backend.
    sandbox_hostname_suffix: Optional[str] = None
    # TB4 exposes cancellation to release a seeded resources-server session after failures.
    cancel_session_on_error: bool = False
    model_call_timeout_s: float = 600.0
    model_call_max_attempts: int = 3


def _slug(text: str) -> str:
    """Filesystem-safe form of a task name (they look like ``tb4-ml-synth/foo-bar``)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:120]


class MiniSweAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    agent_timeout_sec: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)


class MiniSweAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class MiniSweAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    # Upstream exit_status: Submitted | LimitsExceeded | TimeExceeded | RepeatedFormatError | <ExceptionName>,
    # plus Gym's own SandboxTimeout when ``sandbox_timeout`` elapses.
    mini_swe_exit_status: str
    # The analogue of terminus2_completed: the agent itself decided it was done.
    mini_swe_completed: bool
    mini_swe_submission: str
    mini_swe_n_model_calls: int
    mini_swe_n_format_errors: int
    # Untruncated command outputs, one per tool call in order (see execute_actions).
    mini_swe_shell_records: List[Dict[str, Any]]
    mini_swe_trajectory_path: Optional[str]
    mini_swe_tool_outputs_raw: List[Dict[str, Any]]
    mini_swe_agent_version: str
    command_exec_times: List[float]
    model_call_times: List[float]
    average_command_exec_time: float
    average_model_call_time: float
    total_command_exec_time: float
    total_model_call_time: float
    command_exec_time_pct: float
    model_call_time_pct: float
    mini_swe_time_taken: float
    model_calls_gt_timeout: int


@lru_cache(maxsize=8)
def _load_mini_config_file(path: Optional[str]) -> Dict[str, Any]:
    config_path = Path(path) if path else builtin_config_dir / "mini.yaml"
    return yaml.safe_load(config_path.read_text())


def _instruction(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value
    messages: list[str] = []
    for item in input_value or []:
        value = item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        if not isinstance(value, dict):
            messages.append(str(value))
            continue
        content = value.get("content", "")
        if isinstance(content, str):
            messages.append(content)
        elif isinstance(content, list):
            messages.extend(
                str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("text")
            )
    return "\n\n".join(messages)


def _text_of(content: Any) -> str:
    """Flatten a mini-swe-agent message ``content`` (str or Responses-API part list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


class NeMoGymSandboxShellEnvironment:
    """``LocalEnvironment`` semantics on top of ``AsyncSandbox.exec``.

    Each action runs in a fresh ``<shell> -c`` with stderr merged into stdout (upstream uses
    ``stderr=subprocess.STDOUT``), under ``command_timeout_s``. A timed-out or failed execution returns the
    partial output with ``returncode=-1`` and an ``exception_info`` string, like upstream's except path.
    """

    UNAME_KEYS = ("system", "release", "version", "machine", "node")

    def __init__(
        self,
        sandbox: AsyncSandbox,
        *,
        timeout: int,
        env: Dict[str, str],
        shell: str,
        cwd: str = "",
        user=None,
        expected_hostname=None,
    ):
        self._sandbox = sandbox
        self.expected_hostname = expected_hostname
        self.user = user
        self.shell_records = []
        self.timeout = timeout
        self.env = dict(env)
        self.shell = shell
        self.cwd = cwd
        # Filled by prepare(); empty defaults keep the instance template renderable if that never ran.
        self._uname: Dict[str, str] = {key: "" for key in self.UNAME_KEYS}
        self.exec_times: List[float] = []

    async def _exec(self, command, **kwargs):
        return await checked_exec(self._sandbox, command, self.expected_hostname, **kwargs)

    async def prepare(self) -> None:
        """Fetch ``platform.uname()`` equivalents from inside the sandbox for the instance template."""
        result = await self._exec(
            "uname -s; uname -r; uname -v; uname -m; uname -n", timeout_s=self.timeout, user=self.user
        )
        lines = (result.stdout or "").splitlines()
        self._uname = {key: (lines[i].strip() if i < len(lines) else "") for i, key in enumerate(self.UNAME_KEYS)}

    async def execute(self, action: Dict[str, Any], cwd: str = "") -> Dict[str, Any]:
        command = action.get("command", "")
        wrapped = f"{self.shell} -c {shlex.quote(command)} 2>&1"
        start = perf_counter()
        try:
            result = await self._exec(
                wrapped,
                timeout_s=self.timeout,
                env=self.env or None,
                cwd=(cwd or self.cwd) or None,
                user=self.user,
            )
        except SandboxIdentityMismatch:
            raise
        except Exception as e:  # sandbox unreachable, hard-cap trips, ...
            self.exec_times.append(perf_counter() - start)
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
            self.shell_records.append({"command": command, **output})
            self._check_finished(output)
            return output
        self.exec_times.append(perf_counter() - start)

        stdout = result.stdout or ""
        if result.error_type is not None:
            detail = result.stderr or result.error_type
            if result.error_type == "timeout" or "timeout" in detail.lower() or "timed out" in detail.lower():
                exception_type = "TimeoutExpired"
                exception = f"Command '{command}' timed out after {self.timeout} seconds"
            else:
                exception_type = "SandboxExecError"
                exception = detail
            output = {
                "output": stdout,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {exception}",
                "extra": {"exception_type": exception_type, "exception": exception},
            }
        else:
            merged = stdout
            if result.stderr:  # stderr is already redirected; anything here is the sandbox runtime talking
                merged = merged + ("" if not merged or merged.endswith("\n") else "\n") + result.stderr
            output = {"output": merged, "returncode": result.return_code, "exception_info": ""}
        self.shell_records.append({"command": command, **output})
        self._check_finished(output)
        return output

    @staticmethod
    def _check_finished(output: Dict[str, Any]) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == SUBMIT_MARKER and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> Dict[str, Any]:
        # Upstream also merges os.environ of the process running the agent; the shipped templates only
        # read the uname fields, and the sandbox env is not the agent process env, so it is left out.
        return recursive_merge({"cwd": self.cwd, "env": self.env, "timeout": self.timeout}, self._uname, kwargs)

    def serialize(self) -> Dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": {
                        "cwd": self.cwd,
                        "env": self.env,
                        "timeout": self.timeout,
                        "shell": self.shell,
                        "user": self.user,
                    },
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


class NeMoGymResponsesModel:
    """``LitellmResponseModel`` semantics against NeMo Gym's Responses API model server."""

    def __init__(
        self,
        client: NeMoGymAsyncOpenAI,
        model_name: str,
        *,
        observation_template: str,
        format_error_template: str,
        model_kwargs: Dict[str, Any],
        call_timeout_s: float,
        max_attempts: int,
        replay_reasoning_items: bool,
    ):
        self._client = client
        self.model_name = model_name
        self.observation_template = observation_template
        self.format_error_template = format_error_template
        self.model_kwargs = model_kwargs
        self._call_timeout_s = call_timeout_s
        self._max_attempts = max_attempts
        self._replay_reasoning_items = replay_reasoning_items
        self.responses: List[NeMoGymResponse] = []
        self.call_times: List[float] = []
        self.calls_gt_timeout = 0
        self.first_request = None
        self.request_attempt_count = 0
        self.omitted_request_parameters = []

    def _prepare_messages_for_api(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten stored response objects into their output items for a stateless call (upstream logic)."""
        result: List[Dict[str, Any]] = []
        for message in messages:
            if message.get("object") == "response":
                for item in message.get("output", []):
                    if not self._replay_reasoning_items and item.get("type") == "reasoning":
                        continue
                    result.append({k: v for k, v in item.items() if k != "extra"})
            else:
                result.append({k: v for k, v in message.items() if k != "extra"})
        return result

    async def query(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        input_items = self._prepare_messages_for_api(messages)
        response: Optional[NeMoGymResponse] = None
        start = perf_counter()
        for attempt in range(self._max_attempts):
            try:
                async with asyncio.timeout(self._call_timeout_s):
                    parameters = dict(model=self.model_name, input=input_items, tools=[BASH_TOOL], **self.model_kwargs)
                    if self.first_request is None:
                        self.first_request = request_snapshot(parameters)
                        self.omitted_request_parameters = sorted(set(parameters) - set(self.first_request))
                    self.request_attempt_count += 1
                    raw = await self._client.create_response(**parameters)
                response = NeMoGymResponse.model_validate(raw)
                break
            except TimeoutError:
                self.calls_gt_timeout += 1
                print(
                    f"Model call exceeded {self._call_timeout_s:.0f}s, attempt {attempt + 1} / {self._max_attempts}",
                    file=sys.stderr,
                )
        self.call_times.append(perf_counter() - start)
        if response is None:
            raise ModelCallTimeout(
                f"Failed to query model endpoint due to timeouts after {self._max_attempts} attempts!"
            )
        self.responses.append(response)

        response_dict = response.model_dump(mode="json")
        model_error = (response.metadata or {}).get("upstream_error") or (response.metadata or {}).get(
            "h3_upstream_error"
        )
        if model_error:
            failure = json.loads(model_error)
            if failure.get("kind") == "context_window_exceeded":
                raise LimitsExceeded(
                    {
                        "role": "exit",
                        "content": failure["raw_response_body"],
                        "extra": {"exit_status": "ContextWindowExceeded", "submission": "", "model_error": failure},
                    }
                )
            raise RuntimeError("Unrecognized preserved model error: " + model_error)
        cost_output = {"cost": 0.0}
        try:
            actions = parse_toolcall_actions_response(
                response_dict.get("output", []),
                format_error_template=self.format_error_template,
                template_kwargs={"finish_reason": finish_reason_from_responses_api(response_dict)},
            )
        except FormatError as e:
            # Upstream contract: the response is persisted on the error message so nothing is lost.
            e.messages[0]["extra"].update(cost_output)
            e.messages[0]["extra"]["response"] = response_dict
            raise
        message = response_dict
        message["extra"] = {"actions": actions, **cost_output, "timestamp": time.time()}
        return message

    def request_capture(self) -> Dict[str, Any]:
        state = "no_model_call"
        if self.first_request is not None:
            state = "first_response_received" if self.responses else "attempted_no_response"
            if self.responses and not self.responses[0].output:
                state = "first_response_empty"
        return {
            "state": state,
            "first_request": request_snapshot(self.first_request) if self.first_request is not None else None,
            "first_request_sha256": snapshot_hash(self.first_request),
            "request_attempt_count": self.request_attempt_count,
            "returned_response_count": len(self.responses),
            "omitted_request_parameters": list(self.omitted_request_parameters),
        }

    def format_message(self, **kwargs) -> Dict[str, Any]:
        return kwargs

    def format_observation_messages(
        self, message: Dict[str, Any], outputs: List[Dict[str, Any]], template_vars: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=self.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_kwargs": self.model_kwargs,
            "observation_template": self.observation_template,
            "format_error_template": self.format_error_template,
        }

    def serialize(self) -> Dict[str, Any]:
        return {
            "model_request_capture": self.request_capture(),
            "info": {
                "config": {
                    "model": self.get_template_vars()
                    | {
                        "model_kwargs": request_snapshot(self.model_kwargs),
                        "replay_reasoning_items": self._replay_reasoning_items,
                    },
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            },
        }


class NeMoGymMiniSweAgent:
    """Asyncio port of ``minisweagent.agents.default.DefaultAgent`` (v2.4.6).

    Besides the upstream ``messages`` list it keeps ``trajectory``: the Responses-API item sequence that
    becomes the full audit history (system + task message, every model output item incl. reasoning, every
    function_call_output / format-error message).
    """

    def __init__(
        self,
        model: NeMoGymResponsesModel,
        env: NeMoGymSandboxShellEnvironment,
        *,
        system_template: str,
        instance_template: str,
        step_limit: int,
        cost_limit: float,
        wall_time_limit_seconds: int,
        max_consecutive_format_errors: int,
    ):
        self.model = model
        self.env = env
        self.config: Dict[str, Any] = {
            "system_template": system_template,
            "instance_template": instance_template,
            "step_limit": step_limit,
            "cost_limit": cost_limit,
            "wall_time_limit_seconds": wall_time_limit_seconds,
            "max_consecutive_format_errors": max_consecutive_format_errors,
            "output_path": None,
        }
        self.messages: List[Dict[str, Any]] = []
        self.trajectory: List[NeMoGymResponseOutputItem] = []
        self.raw_tool_outputs: List[Dict[str, Any]] = []
        self.extra_template_vars: Dict[str, Any] = {}
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self.n_format_errors = 0
        self._start_time = time.time()

    def get_template_vars(self, **kwargs) -> Dict[str, Any]:
        return recursive_merge(
            self.config,
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.messages.extend(messages)
        return list(messages)

    def _record_user_text(self, message: Dict[str, Any]) -> None:
        self.trajectory.append(NeMoGymEasyInputMessage(role="user", content=_text_of(message.get("content", ""))))

    def handle_uncaught_exception(self, e: Exception) -> List[Dict[str, Any]]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": format_exc(),
                },
            )
        )

    async def run(self, task: str = "", **kwargs) -> Dict[str, Any]:
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        system_text = self._render_template(self.config["system_template"])
        instance_text = self._render_template(self.config["instance_template"])
        self.add_messages(
            self.model.format_message(role="system", content=system_text),
            self.model.format_message(role="user", content=instance_text),
        )
        self.trajectory.extend(
            [
                NeMoGymEasyInputMessage(role="system", content=system_text),
                NeMoGymEasyInputMessage(role="user", content=instance_text),
            ]
        )
        while True:
            try:
                await self.step()
                self.n_consecutive_format_errors = 0
            except FormatError as e:
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                self.n_format_errors += 1
                # Upstream drops the offending response from the conversation (the model never sees it
                # again); the stored rollout keeps it because it is real generation with real usage.
                failed = e.messages[0].get("extra", {}).get("response")
                if failed:
                    self.trajectory.extend(NeMoGymResponse.model_validate(failed).output)
                for message in e.messages:
                    self._record_user_text(message)
                if 0 < self.config["max_consecutive_format_errors"] <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    async def step(self) -> List[Dict[str, Any]]:
        return await self.execute_actions(await self.query())

    async def query(self) -> Dict[str, Any]:
        if 0 < self.config["step_limit"] <= self.n_calls or 0 < self.config["cost_limit"] <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config["wall_time_limit_seconds"] <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {"role": "exit", "content": "TimeExceeded", "extra": {"exit_status": "TimeExceeded", "submission": ""}}
            )
        self.n_calls += 1
        message = await self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        self.trajectory.extend(self.model.responses[-1].output)
        return message

    async def execute_actions(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        outputs = []
        for action in message.get("extra", {}).get("actions", []):
            outputs.append(await self.env.execute(action))
        observations = self.model.format_observation_messages(message, outputs, self.get_template_vars())
        # Upstream's observation template elides the middle of any command output longer than 10,000
        # chars (head 5000 + tail 5000), and that elided text is what the model sees and what the
        # function_call_output item stores. Keep the untouched output alongside, one entry per tool
        # call in order, so the rollout retains the full text without changing harness behaviour.
        tool_observations = [o for o in observations if o.get("type") == "function_call_output"]
        for observation, raw in zip(tool_observations, outputs):
            self.raw_tool_outputs.append(
                {
                    "call_id": observation["call_id"],
                    "returncode": raw.get("returncode"),
                    "output": raw.get("output", ""),
                    "elided_in_observation": "elided_chars" in observation["output"],
                }
            )
        for observation in observations:
            if observation.get("type") == "function_call_output":
                self.trajectory.append(
                    NeMoGymFunctionCallOutput(call_id=observation["call_id"], output=observation["output"])
                )
            else:
                self._record_user_text(observation)
        return self.add_messages(*observations)

    def serialize(self, *extra_dicts: Dict[str, Any]) -> Dict[str, Any]:
        last_extra = (self.messages[-1] if self.messages else {}).get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {"instance_cost": self.cost, "api_calls": self.n_calls},
                "config": {
                    "agent": self.config,
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": MINI_SWE_AGENT_VERSION,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
            "gym_full_trajectory": [item.model_dump(mode="json") for item in self.trajectory],
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Optional[Path]) -> Dict[str, Any]:
        data = self.serialize()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2, default=str))
        return data


class MiniSweAgentSandboxedAgent(SimpleResponsesAPIAgent):
    config: MiniSweAgentSandboxedConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_sandboxes: dict[str, AsyncSandbox] = {}
        # session id -> task_name, only so dumped trajectories are identifiable without
        # cross-referencing rollouts.jsonl. Absent for a bare /responses call.
        self._session_task_names: dict[str, str] = {}

    def _mini_config(self) -> Dict[str, Any]:
        return recursive_merge(_load_mini_config_file(self.config.mini_config_path), self.config.mini_config_overrides)

    async def _connect_sandbox(self, sandbox_id: str) -> AsyncSandbox:
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, get_global_config_dict()))
        return await AsyncSandbox.connect({"sandbox_id": sandbox_id}, provider=provider)

    async def _execute(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming,
        sandbox: AsyncSandbox,
        agent_user=None,
        episode_id=None,
        task_name="",
        timeout_s=None,
    ) -> Tuple[NeMoGymResponse, Dict[str, Any]]:
        start_time = perf_counter()
        episode_id = episode_id or uuid4().hex
        instruction = _instruction(body.input)
        mini_config = self._mini_config()
        agent_section = mini_config.get("agent", {})
        model_section = mini_config.get("model", {})
        env_section = mini_config.get("environment", {})

        model_base_url = (
            self.base_url_for_run(base_url=get_server_url(self.config.model_server.name), body=await request.json())
            + "/v1"
        )
        # One routing key per episode pins every turn to the same model endpoint (see terminus_2_sandboxed_agent).
        model = NeMoGymResponsesModel(
            client=NeMoGymAsyncOpenAI(
                base_url=model_base_url,
                api_key="dummy",
                internal=True,
                default_headers={
                    ROUTING_KEY_HEADER: episode_id,
                    "x-nemo-gym-log-task-id": task_name,
                    "x-nemo-gym-log-run-id": os.environ.get("NEMO_GYM_RUN_ID", ""),
                },
            ),
            model_name=self.config.model_server.name,
            observation_template=model_section["observation_template"],
            format_error_template=model_section["format_error_template"],
            # ``drop_params`` and friends are litellm client knobs, not request parameters.
            model_kwargs={k: v for k, v in model_section.get("model_kwargs", {}).items() if k != "drop_params"},
            call_timeout_s=self.config.model_call_timeout_s,
            max_attempts=self.config.model_call_max_attempts,
            replay_reasoning_items=self.config.replay_reasoning_items,
        )
        env = NeMoGymSandboxShellEnvironment(
            sandbox,
            timeout=self.config.command_timeout_s,
            env={k: str(v) for k, v in env_section.get("env", {}).items()},
            shell=self.config.shell,
            cwd=env_section.get("cwd", ""),
            user=agent_user,
            expected_hostname=(sandbox._handle.sandbox_id + self.config.sandbox_hostname_suffix)
            if self.config.sandbox_hostname_suffix is not None
            else None,
        )
        agent = NeMoGymMiniSweAgent(
            model,
            env,
            system_template=agent_section["system_template"],
            instance_template=agent_section["instance_template"],
            step_limit=self.config.step_limit,
            cost_limit=self.config.cost_limit,
            wall_time_limit_seconds=self.config.wall_time_limit_seconds,
            max_consecutive_format_errors=self.config.max_consecutive_format_errors,
        )

        exit_info: Dict[str, Any] = {}
        try:
            async with asyncio.timeout(timeout_s if timeout_s is not None else self.config.sandbox_timeout):
                await env.prepare()
                exit_info = await agent.run(instruction)
        except SandboxIdentityMismatch as exc:
            exit_info = {"exit_status": "SandboxIdentityMismatch", "submission": "", "infrastructure_error": str(exc)}
        except TimeoutError:
            exit_info = {"exit_status": "SandboxTimeout", "submission": ""}
        except Exception as exc:
            exit_info = {"exit_status": type(exc).__name__, "submission": "", "error": str(exc)}
            print(f"Hit exception while running mini-swe-agent: {format_exc()}", file=sys.stderr)

        # Persist the actual terminal status for external deadline/crash outcomes too.
        if not agent.messages or agent.messages[-1].get("role") != "exit":
            agent.add_messages(
                {"role": "exit", "content": exit_info.get("exit_status", "Unknown"), "extra": exit_info}
            )
        captured_request = model.request_capture()
        exported_request, rollout_items, export_metadata = episode_export(agent.trajectory, captured_request)
        trajectory_path = None
        if self.config.dump_trajectory_dir:
            try:
                session_id = episode_id
                label = _slug(task_name)
                name = f"{label}__{session_id}.traj.json" if label else f"{session_id}.traj.json"
                trajectory_path = str(Path(self.config.dump_trajectory_dir) / name)
                saved = agent.serialize(
                    {
                        "shell_records": env.shell_records,
                        "model_responses": [x.model_dump(mode="json") for x in model.responses],
                        "gym_export": export_metadata,
                        "responses_create_params": exported_request,
                        "original_responses_create_params": body.model_dump(mode="json"),
                    }
                )
                Path(trajectory_path).parent.mkdir(parents=True, exist_ok=True)
                Path(trajectory_path).write_text(json.dumps(saved, default=str))
            except Exception:
                print(f"Failed to dump mini-swe-agent trajectory: {format_exc()}", file=sys.stderr)

        input_tokens = output_tokens = cached_tokens = reasoning_tokens = 0
        for response in model.responses:
            usage = response.usage
            if usage is None:
                continue
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens
            if usage.input_tokens_details is not None:
                cached_tokens += usage.input_tokens_details.cached_tokens or 0
            if usage.output_tokens_details is not None:
                reasoning_tokens += usage.output_tokens_details.reasoning_tokens or 0
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time.time()),
            model=self.config.model_server.name,
            object="response",
            output=rollout_items,
            tool_choice=exported_request.get("tool_choice"),
            tools=exported_request.get("tools", []),
            parallel_tool_calls=exported_request.get("parallel_tool_calls"),
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning_tokens),
                total_tokens=input_tokens + output_tokens,
            ),
        )

        total_time = perf_counter() - start_time
        total_command_exec_time = sum(env.exec_times)
        total_model_call_time = sum(model.call_times)
        exit_status = str(exit_info.get("exit_status", "") or "Unknown")
        metrics = {
            "responses_create_params": exported_request,
            "mini_swe_export": export_metadata,
            "mini_swe_model_request_capture": captured_request,
            "mini_swe_original_responses_create_params": body.model_dump(mode="json"),
            "mini_swe_exit_status": exit_status,
            "mini_swe_completed": exit_status == "Submitted",
            "mini_swe_submission": str(exit_info.get("submission", "") or ""),
            "mini_swe_n_model_calls": agent.n_calls,
            "mini_swe_n_format_errors": agent.n_format_errors,
            "mini_swe_shell_records": env.shell_records,
            "mini_swe_trajectory_path": trajectory_path,
            "mini_swe_tool_outputs_raw": agent.raw_tool_outputs,
            "mini_swe_agent_version": MINI_SWE_AGENT_VERSION,
            "mini_swe_error": exit_info.get("error") or exit_info.get("infrastructure_error"),
            "mini_swe_agent_timeout_s": timeout_s if timeout_s is not None else self.config.sandbox_timeout,
            "command_exec_times": env.exec_times,
            "model_call_times": model.call_times,
            "average_command_exec_time": total_command_exec_time / max(len(env.exec_times), 1),
            "average_model_call_time": total_model_call_time / max(len(model.call_times), 1),
            "total_command_exec_time": total_command_exec_time,
            "total_model_call_time": total_model_call_time,
            "command_exec_time_pct": 100 * total_command_exec_time / max(total_time, 1e-9),
            "model_call_time_pct": 100 * total_model_call_time / max(total_time, 1e-9),
            "mini_swe_time_taken": total_time,
            "model_calls_gt_timeout": model.calls_gt_timeout,
            "mini_swe_sandbox_identity_guard": self.config.sandbox_hostname_suffix is not None,
            "mini_swe_sandbox_id": getattr(getattr(sandbox, "_handle", None), "sandbox_id", None),
            "mini_swe_sandbox_hostname": env._uname.get("node"),
        }
        return response, metrics

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        sandbox = self._session_sandboxes[request.session[SESSION_ID_KEY]]
        response, metrics = await self._execute(request, body, sandbox)
        # A bare /responses route has no outer run envelope; retain its export request as an extension.
        return NeMoGymResponse.model_validate(
            response.model_dump()
            | {
                "responses_create_params": metrics["responses_create_params"],
                "mini_swe_export": metrics["mini_swe_export"],
                "mini_swe_model_request_capture": metrics["mini_swe_model_request_capture"],
            }
        )

    async def run(self, request: Request, body: MiniSweAgentRunRequest) -> MiniSweAgentVerifyResponse:
        cookies = request.cookies
        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = cookies | seed_session_response.cookies
        seed_session_result = await seed_session_response.json()

        sandbox = None
        verification_completed = False
        session_key = uuid4().hex
        self._session_task_names[session_key] = getattr(body, "task_name", "") or ""

        try:
            sandbox = await self._connect_sandbox(seed_session_result["sandbox_handle"])
            self._session_sandboxes[session_key] = sandbox
            response, metrics = await self._execute(
                request,
                body.responses_create_params,
                sandbox,
                agent_user=seed_session_result.get("agent_user"),
                episode_id=session_key,
                task_name=getattr(body, "task_name", "") or "",
                timeout_s=body.agent_timeout_sec,
            )
            if metrics.get("mini_swe_exit_status") == "SandboxIdentityMismatch":
                result = body.model_dump() | {
                    "response": response.model_dump(),
                    "reward": 0.0,
                    "evaluation_completed": False,
                    "verification_time_taken": 0.0,
                    "test_output": "Sandbox identity mismatch; verification skipped",
                    "golden_patch_output": None,
                }
                result.update(metrics)
                return MiniSweAgentVerifyResponse.model_validate(result)
            verification = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump()
                | {"response": response.model_dump(), "responses_create_params": metrics["responses_create_params"]},
                cookies=cookies,
            )
            await raise_for_status(verification)
            result = await get_response_json(verification)
            result.update(metrics)
            parsed = MiniSweAgentVerifyResponse.model_validate(result)
            verification_completed = True
            return parsed
        finally:
            self._session_sandboxes.pop(session_key, None)
            self._session_task_names.pop(session_key, None)
            if not verification_completed and self.config.cancel_session_on_error:
                try:
                    cancelled = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/cancel_session",
                        json={},
                        cookies=cookies,
                    )
                    await raise_for_status(cancelled)
                except Exception:
                    print("Failed to cancel resources session", format_exc(), file=sys.stderr)
            if sandbox is not None:
                try:
                    await sandbox.stop()
                except Exception:
                    print("Failed to settle agent sandbox", format_exc(), file=sys.stderr)


if __name__ == "__main__":
    MiniSweAgentSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = MiniSweAgentSandboxedAgent.run_webserver()  # noqa: F401
