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

import asyncio
import copy
import json
import logging
import math
import os
import re
import shlex
import shutil
import tempfile
from asyncio import Semaphore
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from shlex import quote
from time import monotonic, time
from typing import Any, ClassVar, Literal, Optional, Self
from uuid import uuid4

from fastapi import HTTPException, Request
from openai.types.responses.response import IncompleteDetails
from pydantic import ConfigDict, Field, JsonValue, PrivateAttr, model_validator

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AGENT_TIMEOUT_METADATA_KEY,
    HARNESS_COMMAND_NOT_STARTED,
    HARNESS_RUNNER_FAILED,
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
    HARNESS_STEP_LIMIT_METADATA_KEY,
    AgentCloseSessionResponse,
    AgentContinuationCapabilities,
    AgentOutcome,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    Finished,
    RestoredLaunch,
    SandboxNetworkAccess,
    SandboxSession,
    SimpleResponsesAPIAgent,
    Stopped,
    abandon_launch,
    check_restored,
    exec_idempotent,
    resolve_agent_timeout,
    restrict_sandbox_network,
    run_until_parked,
    verification_failure_fields,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef, ResourcesServerRef, get_global_config_dict
from nemo_gym.core import (
    AGENT_ROUTE_FAILED,
    MODEL_CONTEXT_LIMIT,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    EpisodeId,
    MediaFile,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
    NgFailureEvent,
    SandboxFailureDetails,
    UnsupportedContentError,
    observe_failure,
    split_text_and_media,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxExecResult,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
)
from nemo_gym.server_runtime import Activation, BaseRunRequest, BaseVerifyResponse, MCPToolAccess, RestoredAgentSession
from nemo_gym.trajectory import (
    NG_AGENT_SANDBOX_OBSERVATION_KEY,
    AgentEpisode,
    AgentInvocation,
    AgentObservationBundle,
    ObservationGap,
    SandboxObservation,
    TrajectoryRecord,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.opencode_agent.artifacts import (
    _assistant_part_items,
    _empty_answer_part,
    _parse_opencode_session,
    parse_opencode_session,
)
from responses_api_agents.opencode_agent.continuation import OpenCodeRunOutcome, classify_opencode_run
from responses_api_agents.opencode_agent.observability import scope_opencode_trajectory
from responses_api_agents.opencode_agent.pinned import (
    COPY_PINNED,
    REJECTED,
    STAGE_EXECUTABLE,
    UNPACK_ARCHIVE,
    check_archive,
)
from responses_api_agents.opencode_agent.resume import resume_decision
from responses_api_agents.opencode_agent.runtime import (
    ASSISTANT_MESSAGE_PLUGIN_FILE,
    ASSISTANT_MESSAGE_PLUGIN_SOURCE,
    IMAGE_INPUT_MODEL_OPTIONS,
    OPENCODE_VERSION,
    REQUIRED_MCP_PLUGIN_FILE,
    RUNTIME_INSTALLER_FILE,
    RUNTIME_INSTALLER_SOURCE,
    TASK_ENVIRONMENT_PLUGIN_FILE,
    TASK_ENVIRONMENT_PLUGIN_SOURCE,
    UNRESTRICTED_IMAGE_ATTACHMENT,
    apply_observability_patch,
    command_failure_reason,
)
from responses_api_agents.opencode_agent.sandbox import (
    MODEL_RELAY_SCRIPT,
    RELAY_STATUS_FILE,
    OpenCodeSandboxSession,
    cli_environment,
    format_error_message,
)
from responses_api_agents.opencode_agent.setup_opencode import ensure_opencode
from responses_api_agents.opencode_agent.snapshot import stored_message_ids, stored_sessions


LOG = logging.getLogger(__name__)
RESUME_SCRIPT = Path(__file__).with_name("resume_opencode.sh")
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"
_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"
_TESTED_OUTCOME_VERSION = "1.17.11"
# Restore outcomes that are expected rather than faults: the checkpoint froze the launch before its relay started.
# They are logged at INFO; every other reason a launch cannot resume in place, a launch without a relay included, at
# WARNING: the relay runs on OpenCode's own binary, so every image can run it.
_EXPECTED_RESTORE_FALLBACKS = frozenset({"relay_gone"})
_API_ERROR = "APIError"
_CONTEXT_OVERFLOW_ERROR = "ContextOverflowError"


def _extract_instruction(body_input) -> tuple[str, Optional[str]]:
    """Return (user_message, system_message) from a responses body input list."""
    items = list(body_input)
    system_message: Optional[str] = None

    if items:
        first = items[0]
        role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        if role == "system":
            content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            system_message = content or ""
            items = items[1:]

    user_message = ""
    for item in reversed(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        if role == "user":
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            user_message = content or ""
            break

    return user_message, system_message


def _sandbox_prepare_command(workdir: str, directory: str, runtime: str, *, export_dir: str | None = None) -> str:
    # Bootstrap with POSIX sh; the runtime installer itself requires Bash.
    bootstrap = """set -eu
set --
command -v bash >/dev/null 2>&1 || set -- "$@" bash
if [ "$#" -gt 0 ]; then
    [ "$(id -u)" = 0 ] || { echo "OpenCode sandbox execution requires $*: preinstall these tools or use a root image." >&2; exit 1; }
    if command -v apk >/dev/null 2>&1; then
        apk add --no-cache "$@"
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
    else
        echo "OpenCode sandbox execution requires $*: preinstall these tools (automatic installation requires apt-get or apk)." >&2
        exit 1
    fi
fi
"""
    # Resolve symlinks like realpath -m: storage roots need not exist yet.
    validate_paths = """resolve() {
    if [ -e "$1" ]; then (cd -P "$1" && pwd -P); else
        parent=${1%/*}
        parent=$(resolve "${parent:-/}") && printf '%s/%s\\n' "${parent%/}" "${1##*/}"
    fi
}
[ -d "$1" ] || { echo 'OpenCode task workdir is missing' >&2; exit 1; }
workdir=$(resolve "$1")
shift
for root in "$@"; do
    root=$(resolve "$root")
    case "$root/" in "${workdir%/}"/*) overlap=1 ;; esac
    case "$workdir/" in "${root%/}"/*) overlap=1 ;; esac
done
[ -z "${overlap-}" ] || { echo 'OpenCode runtime/session storage overlaps the task workdir' >&2; exit 1; }
"""
    command = (
        bootstrap
        + f"set -- {quote(workdir)} {quote(str(PurePosixPath(directory).parent))} {quote(runtime)}\n"
        + validate_paths
    )
    if export_dir is not None:
        # Exports are written after the agent stops; in the workdir they would become part of the task's result.
        command += f"set -- {quote(export_dir)}\n" + _CHECK_EXPORT_DIR
    return command + f"mkdir -p {quote(directory)}"


def _is_sandbox_path(path: object) -> bool:
    """Whether ``path`` is an absolute, normalized POSIX path that a shell can receive."""
    return (
        isinstance(path, str)
        and "\x00" not in path
        and PurePosixPath(path).is_absolute()
        and ".." not in PurePosixPath(path).parts
    )


# Exit status of the plugin check and the pinned-file steps when an owner-supplied input is unusable (a missing plugin
# file, pinned bytes failing their SHA-256 or health check): a retry cannot succeed.
_INPUT_REJECTED = REJECTED
# Prints each owner-staged plugin path ("$@") that is not a regular file and exits _INPUT_REJECTED if any.
_CHECK_PLUGIN_FILES = f"""missing=
for path in "$@"; do
    [ -f "$path" ] || {{ printf '%s\\n' "$path"; missing=1; }}
done
[ -z "$missing" ] || exit {_INPUT_REJECTED}
"""
# Runs after the prepare step's path checks, whose resolve() and workdir it uses: exits non-zero when the owner's export
# directory "$1" resolves to the task workdir, a folder inside it, or one above it.
_CHECK_EXPORT_DIR = """export_dir=$(resolve "$1")
case "$export_dir/" in "${workdir%/}"/*) overlap=1 ;; esac
case "$workdir/" in "${export_dir%/}"/*) overlap=1 ;; esac
[ -z "${overlap-}" ] || { echo 'OpenCode export_dir overlaps the task workdir' >&2; exit 1; }
"""


# Built-in agents by OpenCode version. Each sets its own sampling (1.17.11's ``title`` fixes temperature 0.5) and none
# inherits from ``build``, so request sampling goes to all of them. A name OpenCode does not define would add an
# agent, so other versions refuse request sampling, as the adapter did before it applied any.
_OPENCODE_AGENTS = {"1.17.11": ("build", "plan", "general", "explore", "compaction", "title", "summary")}
# Isolation shared by every OpenCode process of a session: run, export and import.
_OPENCODE_ISOLATION = {
    "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
    "OPENCODE_DISABLE_AUTOUPDATE": "true",
    "OPENCODE_DISABLE_MODELS_FETCH": "true",
}


def _private_directories(root: str) -> dict[str, str]:
    """OpenCode's private HOME and XDG directories under ``root``; tool shells get the task's back through a hook."""
    return {
        "HOME": f"{root}/home",
        "XDG_DATA_HOME": f"{root}/data",
        "XDG_CONFIG_HOME": f"{root}/config",
        "XDG_CACHE_HOME": f"{root}/cache",
        "XDG_STATE_HOME": f"{root}/state",
    }


def _without_empty_tools(opencode_config: dict[str, Any]) -> dict[str, Any]:
    """Drop a null or empty ``tools`` map so ``permission`` is the only rule source, exactly as given.

    OpenCode 1.17.11 turns ``tools`` into base permission rules and merges ``permission`` on top.
    """
    return {key: value for key, value in opencode_config.items() if key != "tools" or value}


@dataclass(frozen=True, kw_only=True)
class _PinnedFile:
    """A file inside the sandbox that the session must use byte for byte."""

    digest: str
    sandbox_path: str


def _pinned_copy(target: str) -> str:
    """Where a pinned file is copied and checked before it becomes ``target``: a hidden file beside it."""
    path = PurePosixPath(target)
    return str(path.with_name(f".pinned-{path.name}"))


class _PauseClock:
    """An activation's park handle that adds up the time spent parked, so a relaunch keeps the unspent budget."""

    def __init__(self, activation: Activation) -> None:
        self._activation = activation
        self.paused_s = 0.0

    @property
    def park_requested(self) -> bool:
        return self._activation.park_requested

    async def wait_park_requested(self) -> None:
        await self._activation.wait_park_requested()

    async def boundary(self, snapshot: Callable[[], dict[str, JsonValue]]) -> None:
        started = monotonic()
        try:
            await self._activation.boundary(snapshot)
        finally:
            self.paused_s += monotonic() - started

    def queued(self) -> AbstractAsyncContextManager[None]:
        return self._activation.queued()


@asynccontextmanager
async def _slot(semaphore: asyncio.Semaphore, *, deadline: float | None) -> AsyncIterator[None]:
    """Hold one of the agent's concurrency slots for the body when checkpointing is off; wait until ``deadline``."""
    async with asyncio.timeout_at(deadline):
        await semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


class _ActivationDeadlineExpired(Exception):
    """The immutable activation window elapsed before process launch."""


def _dump_with_serialized_member(value: dict[str, Any], member: str, serialized: str) -> str:
    return (
        "{"
        + ",".join(
            json.dumps(key) + ":" + (serialized if key == member else json.dumps(item)) for key, item in value.items()
        )
        + "}"
    )


def _serialize_replacement_config(config: dict[str, Any]) -> str:
    """Keep only the replacement prompt literal through OpenCode's pre-JSON substitutions."""
    agent = config["agent"]
    build = agent["build"]
    prompt = json.dumps(build["prompt"]).replace("{", "\\u007b")
    build_json = _dump_with_serialized_member(build, "prompt", prompt)
    agent_json = _dump_with_serialized_member(agent, "build", build_json)
    return _dump_with_serialized_member(config, "agent", agent_json)


def _session_opencode_config_error(opencode_config: Mapping[str, Any]) -> str | None:
    """Return why a sandbox session cannot use ``opencode_config``, or ``None``.

    The adapter owns the model, provider, prompt and plugins of a sandbox session, so it accepts only:

    - ``permission`` and ``tools``, OpenCode's tool access.
    - ``attachment``, OpenCode's image limits. A session that declares image input lifts them, and this restores them
      for an endpoint with image limits.
    - ``compaction.auto`` (a boolean). Automatic compaction summarizes the session and continues from the
      summary, in model calls that do not extend the scored call's prefix. With it off, a run that fills the
      context ends with OpenCode's context-overflow error, which the adapter reports as a context limit.
    - ``agent.build.steps`` (a positive integer): the step limit of OpenCode's ``build`` agent, which the session
      runs, for each prompt. From the call at the limit on, OpenCode adds an assistant message to each request that
      asks for a text answer. It still offers the tools, so a model that keeps calling them runs past the limit.
      Other agent fields would replace the adapter's prompt, model or sampling. Token capture recognizes that
      message (``nemo_gym.models.token_capture.harness_notes``) and keeps the calls after it chained.
    """
    unsupported = sorted(opencode_config.keys() - {"permission", "tools", "attachment", "compaction", "agent"})
    if unsupported:
        return (
            "supports permission, tools, attachment, compaction.auto and agent.build.steps only; "
            f"unsupported: {unsupported}"
        )
    compaction = opencode_config.get("compaction", {})
    if not isinstance(compaction, Mapping) or compaction.keys() - {"auto"}:
        return "compaction supports only auto"
    if "auto" in compaction and not isinstance(compaction["auto"], bool):
        return "compaction.auto must be true or false"
    agent = opencode_config.get("agent", {})
    if not isinstance(agent, Mapping) or agent.keys() - {"build"}:
        return "agent supports only build, the agent a session runs"
    build = agent.get("build", {})
    if not isinstance(build, Mapping) or build.keys() - {"steps"}:
        return "agent.build supports only steps"
    if "steps" in build and (type(build["steps"]) is not int or build["steps"] <= 0):
        return "agent.build.steps must be a positive integer"
    return None


def _build_step_limit(opencode_config: Mapping[str, Any]) -> int | None:
    """The configured ``agent.build.steps``, or None when OpenCode runs without a step limit."""
    agent = opencode_config.get("agent")
    build = agent.get("build") if isinstance(agent, Mapping) else None
    steps = build.get("steps") if isinstance(build, Mapping) else None
    return steps if type(steps) is int and steps > 0 else None


def _continues_the_prompt(messages: Sequence[Mapping[str, Any]], index: int) -> bool:
    """Whether the user message at ``index`` is one that OpenCode's compaction adds, not a new prompt."""
    if any(part.get("type") == "compaction" for part in messages[index].get("parts") or []):
        return True
    # After an automatic compaction's summary, OpenCode continues the prompt with a user message: a synthetic
    # "Continue if you have next steps" text, or a replay of the prompt when the compaction was for an overflow.
    return index > 0 and messages[index - 1]["info"].get("summary") is True


def reached_step_limit(export: Mapping[str, Any], steps: int | None) -> bool:
    """Whether the last prompt of an OpenCode export reached the step limit that ``steps`` sets.

    OpenCode 1.17.11's ``opencode run`` runs the session loop in packages/opencode/src/session/prompt.ts. Its counter
    ``step`` grows by one on each pass of the loop, and ``isLastStep = step >= maxSteps``: from that pass on, each
    build call appends MAX_STEPS_PROMPT as an assistant message but still offers the tools and leaves toolChoice
    unset. So the note asks for a final answer and does not force one, and the model may keep calling tools.

    Compaction does not reset the counter, and its passes count too (packages/opencode/src/session/compaction.ts).
    A build call and a compaction's summary call each take a pass and save an assistant message. A pass can also
    only start a compaction, when the last finished call overflowed; it saves a user message with a compaction part
    and no ``overflow`` field. A compaction that a build call starts sets ``overflow`` and takes no pass of its own.
    The user messages that compaction adds are not new prompts (``_continues_the_prompt``).
    """
    if steps is None:
        return False
    messages = export.get("messages") or []
    prompt = max(
        (
            index
            for index, message in enumerate(messages)
            if message["info"].get("role") == "user" and not _continues_the_prompt(messages, index)
        ),
        default=-1,
    )
    passes = 0
    for message in messages[prompt + 1 :]:
        parts = message.get("parts") or []
        if message["info"].get("role") == "assistant" or any(
            part.get("type") == "compaction" and "overflow" not in part for part in parts
        ):
            passes += 1
    return passes >= steps


class OpenCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef | None = None
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    command: str = "opencode"
    model: str = "openai/gpt-4o-mini"
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    # extra env vars for the subprocess e.g. API keys
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/opencode_agent/workspaces"
    repo_dir: Optional[str] = None
    thinking: bool = True
    image_input: bool = Field(
        default=False,
        description=(
            "Declare image input on the Gym model in sandbox sessions even when the prompt has no images, so that "
            "OpenCode sends the model images its tools return, such as task files it reads. A prompt with images "
            "always declares it, for that activation and every later one in the session."
        ),
    )
    system_prompt: Optional[str] = None
    system_prompt_mode: Literal["append", "replace"] = "append"
    timeout: int = 900
    extra_args: list[str] = []
    opencode_config: dict[str, Any] = Field(default_factory=dict)
    context_window: int = 262144
    max_output_tokens: int = 131072
    opencode_version: str = OPENCODE_VERSION

    # Optional verified, preinstalled runtime for sandboxes without download egress.
    sandbox_path_prefix: list[str] = Field(default_factory=list)
    sandbox_executable_path: str | None = None
    sandbox_executable_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    # The native model ID affects OpenCode's model-family prompts and tools.
    # Preserve the existing alias unless a benchmark explicitly qualifies another ID.
    sandbox_model_id: str = "dummy_model"

    # Used only when Resources does not supply a borrowed sandbox.
    sandbox_provider: str | None = None
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    network_access: SandboxNetworkAccess = Field(
        default="inherit",
        description=(
            "Egress of a sandbox OpenCode creates: `model_only` allows only the model server's host, "
            "`model_and_tools` also each MCP grant's host. A restricted sandbox cannot download OpenCode, so set "
            "`sandbox_executable_path` (or use an image with the runtime already at the cache path). A borrowed task "
            "sandbox keeps its owner's network, so a session that borrows one rejects any other value than `inherit`."
        ),
    )
    sandbox_install_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)
    report_agent_sandbox_oom: bool = False
    # Connection and per-call timeout of a sandbox session's MCP tool grants. Unset, OpenCode connects within 30 s
    # and each call gets the MCP SDK's 60 s.
    mcp_tool_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    # Where each session keeps its supervisor files and OpenCode's HOME and data. With checkpointing it must be on
    # the sandbox's snapshotted disk, so a restore finds the saved conversation.
    sandbox_session_root: str = "/tmp/nemo-gym-opencode-sessions"
    # How long a launch may stay paused for checkpoints in all, beyond `timeout`. Paused time does not count against
    # `timeout`; a launch paused for longer fails, so size it to cover the longest checkpoint commit.
    checkpoint_pause_allowance_seconds: float = Field(default=3600, ge=0, allow_inf_nan=False)
    # The sandbox port of the model relay OpenCode calls the model through, on a sandbox whose snapshots keep
    # processes: a restore re-points the relay at the replacement attempt while OpenCode keeps running.
    sandbox_model_relay_port: int = Field(default=18431, gt=0, lt=65536)

    # Sandbox sessions only. Absolute sandbox paths of OpenCode plugin files that the sandbox owner staged,
    # loaded after Gym's own plugins in this order. Seed fails with 422 if one is not a regular file.
    plugins: list[str] = Field(default_factory=list)
    # Sandbox sessions only. Offline runtime bundle beside the binary, each pinned the same way: an rg for OpenCode's
    # grep/glob tools, and a tar archive of the plugin dependencies OpenCode otherwise installs at every start.
    # Either is a file already inside the sandbox and requires its SHA-256, which is checked there.
    sandbox_ripgrep_path: str | None = None
    sandbox_ripgrep_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    sandbox_plugin_dependencies_path: str | None = None
    sandbox_plugin_dependencies_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    # Sandbox sessions only. Absolute sandbox directory, outside the task workdir, where each activation's capture
    # also writes `opencode export` of every OpenCode session and the event log, for the sandbox owner to read after
    # agent close. A sandbox the agent owns keeps them only as long as the provider keeps its files after close.
    export_dir: str | None = None
    # Sandbox sessions only. Absolute sandbox path of an OpenCode session export staged by the owner. When the file
    # exists at seed, it is imported and the first activation continues that session; its messages stay history.
    session_import_path: str | None = None
    # Sandbox sessions only. Close also returns an ordinary session's own turns (one per saved assistant reply,
    # compaction summaries left out), as continuation sessions always do, so rollout collection joins each captured
    # call to its step and the harness evidence checks can match them. Without them, collection builds one turn per
    # captured call with no message ID, every call is reported as an unmatched assistant message, and the summary call
    # of a compaction is counted as a step too.
    return_session_trajectory: bool = True
    # Sandbox sessions only. Let OpenCode's run load the task repository's project config and instruction files
    # (AGENTS.md or CLAUDE.md, opencode.json, .opencode/), as it does when started in that repository. The configuration
    # Gym passes is merged last and still wins. Off by default, so a repository cannot add instructions, agents, plugins
    # or tools.
    session_project_config: bool = False

    @model_validator(mode="after")
    def _validate_prompt_agent(self) -> Self:
        if self.system_prompt_mode == "replace" and any(
            arg == "--agent" or arg.startswith("--agent=") for arg in self.extra_args
        ):
            raise ValueError(
                "OpenCode replacement selects the build agent; remove --agent from extra_args "
                "or use system_prompt_mode='append'."
            )
        return self

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class OpenCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class OpenCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None, exclude_if=lambda value: value is None
    )


class OpenCodeAgent(SimpleResponsesAPIAgent):
    """Run OpenCode locally or in a borrowed or agent-owned sandbox session."""

    @staticmethod
    def _reported_model_error(native_result: Mapping[str, Any], cleanup: dict | None) -> bool:
        """Recognize a completed API error or context limit; tested on OpenCode 1.17.11.

        Independent worker failures, cleanup errors and timeouts are not provider reports.
        """
        finishes = {_API_ERROR: None, _CONTEXT_OVERFLOW_ERROR: "error"}
        name = (native_result.get("error") or {}).get("name")
        return (
            cleanup is not None
            and cleanup["return_code"] == 1
            and not cleanup["timed_out"]
            and not cleanup["error"]
            and name in finishes
            and native_result.get("finish") == finishes[name]
            and bool(native_result.get("time", {}).get("completed"))
        )

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """Read native APIError status or transport retry marker; tested on OpenCode 1.17.11.

        OpenCode marks connection resets, header timeouts, decompression failures and
        interrupted streams retryable without an HTTP status. Preserve that evidence;
        the Environment Server still decides whether to retry the episode.
        """
        error = native_result.get("error") or {}
        if error.get("name") == _CONTEXT_OVERFLOW_ERROR:
            return AgentOutcome(failure_kind=MODEL_CONTEXT_LIMIT)
        if error.get("name") != _API_ERROR:
            return AgentOutcome(failure_kind="unknown")
        data = error.get("data") or {}
        status = data.get("statusCode")
        if type(status) is int and 400 <= status <= 599:
            return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=status)
        if status is None and data.get("isRetryable") is True:
            return AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)
        return AgentOutcome(failure_kind="unknown")

    config: OpenCodeAgentConfig
    # Sandbox sessions run the CLI inside the task sandbox, which calls the model server from there.
    calls_model_from_sandbox: ClassVar[bool] = True
    # A one-activation session in a sandbox the agent borrows pauses OpenCode in place at a checkpoint and resumes
    # its saved conversation after a crash. Sessions the agent cannot capture restart (``_seed_restarts``).
    checkpoint_mode: ClassVar[str] = "exported"
    # OpenCode takes no resources tool grants; a re-bound session renders its configuration again anyway.
    rebinds_tool_accesses: ClassVar[bool] = True
    _local_setup_task: asyncio.Task[None] | None = PrivateAttr(default=None)
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.sem = Semaphore(self.config.concurrency)

    async def _ensure_local_runtime(self) -> None:
        if self._local_setup_task is None:
            self._local_setup_task = asyncio.create_task(
                asyncio.to_thread(ensure_opencode, self.config.opencode_version)
            )
            self._local_setup_task.add_done_callback(self._clear_failed_local_setup)
        setup = self._local_setup_task
        try:
            await asyncio.shield(setup)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._local_setup_task is setup:
                self._local_setup_task = None
            raise

    def _clear_failed_local_setup(self, task: asyncio.Task[None]) -> None:
        # All shielded waiters may have gone away before the installer fails.
        failed = task.cancelled() or task.exception() is not None
        if failed and self._local_setup_task is task:
            self._local_setup_task = None

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                OpenCodeAgent._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _workspace_root(self) -> Path:
        """Create a fresh per-rollout workspace directory and return it.

        The full uuid plus exist_ok=False: a name collision must fail this
        rollout loudly rather than silently merge two live rollouts' trees.
        """
        root = Path(self.config.workspace_root).expanduser() / f"opencode_{uuid4().hex}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=False)
        return root

    def _repo_dir(self, fallback: Path) -> Path:
        """Return the configured persistent repository or the temporary fallback."""
        if not self.config.repo_dir:
            return fallback
        root = Path(self.config.repo_dir).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _resolve_model_base_url(self, rollout_id: Optional[str] = None) -> str:
        if self.config.model_server is None:
            return ""
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    def _effective_model(self) -> str:
        return f"nemo/{self.config.model}" if self.config.model_server else self.config.model

    def _build_opencode_config(self, rollout_id: Optional[str] = None) -> dict[str, Any]:
        config = self._deep_merge({}, copy.deepcopy(_without_empty_tools(self.config.opencode_config)))
        if self.config.model_server:
            providers = config.setdefault("provider", {})
            nemo = providers.setdefault("nemo", {"npm": "@ai-sdk/openai-compatible"})
            nemo.setdefault("options", {}).update(
                {"baseURL": self._resolve_model_base_url(rollout_id), "apiKey": "EMPTY"}  # pragma: allowlist secret
            )
            model = nemo.setdefault("models", {}).get(self.config.model, {})
            model.setdefault("interleaved", {"field": "reasoning_content"})
            self._deep_merge(
                model,
                {
                    "name": self.config.model,
                    "limit": {"context": self.config.context_window, "output": self.config.max_output_tokens},
                },
            )
            nemo["models"] = {self.config.model: model}
        if self._model_call_capture_enabled():
            apply_observability_patch(config)
        return config

    def _write_opencode_config(self, work_dir: Path, rollout_id: Optional[str] = None) -> None:
        config = self._build_opencode_config(rollout_id)
        if not config:
            return
        (work_dir / "opencode.json").write_text(json.dumps(config, indent=2))

    def _env(self, data_home: str, rollout_id: Optional[str] = None) -> dict[str, str]:
        env = {**os.environ, "XDG_DATA_HOME": data_home}
        base_url = (
            self._resolve_model_base_url(rollout_id) if self.config.model_server else self.config.openai_base_url
        )
        api_key = "EMPTY" if self.config.model_server else self.config.openai_api_key  # pragma: allowlist secret
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        env.update({k: v for k, v in self.config.env.items() if v})
        if self.config.model_server is not None:
            env["OPENAI_BASE_URL"] = base_url
            env["OPENAI_API_KEY"] = "EMPTY"  # pragma: allowlist secret
        return env

    async def _run_opencode(
        self,
        instruction: str,
        system_prompt: Optional[str],
        *,
        rollout_id: Optional[str] = None,
        collect_observations: bool = True,
        trajectory: Optional[TrajectoryRecord] = None,
        timeout: float | None = None,
    ) -> tuple[list[Any], dict[str, int], str, AgentObservationBundle]:
        """Run one headless OpenCode session and read its persisted artifact."""
        try:
            self.config._validate_prompt_agent()
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        timeout = self.config.timeout if timeout is None else timeout
        await self._ensure_local_runtime()
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        project_dir = self._repo_dir(work_dir)
        data_home = work_dir / ".opencode-data"
        data_home.mkdir(parents=True, exist_ok=True)
        self._write_opencode_config(project_dir, rollout_id)
        env = self._env(str(data_home), rollout_id)
        if system_prompt and self.config.system_prompt_mode == "replace":
            # Keep request-specific instructions out of a persistent repo's config file.
            overrides = json.loads(env.get("OPENCODE_CONFIG_CONTENT", "{}"))
            self._deep_merge(overrides, {"agent": {"build": {"prompt": system_prompt}}})
            env["OPENCODE_CONFIG_CONTENT"] = _serialize_replacement_config(overrides)
            prompt = instruction

        cmd = [*self.config.command_parts, "run", "-m", self._effective_model(), "--dir", str(project_dir)]
        if self.config.system_prompt_mode == "replace":
            cmd.extend(["--agent", "build"])
        if self.config.thinking:
            cmd.append("--thinking")
        cmd.extend(self.config.extra_args)
        cmd.append(prompt)

        try:
            timed_out = False
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                _, stderr = await proc.communicate()
                timed_out = True
                LOG.warning("opencode timed out after %ss", timeout)

            if proc.returncode not in (0, None):
                LOG.warning("opencode exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

            db_path = data_home / "opencode" / "opencode.db"
            invocation_id = rollout_id or f"opencode-{uuid4().hex}"
            output_items, usage = (
                ([], {"input_tokens": 0, "output_tokens": 0})
                if timed_out
                else parse_opencode_session(db_path, gym_reasoning=self.config.model_server is not None)
            )
            observations = AgentObservationBundle(schema_version="2.0", source="opencode")
            if collect_observations:
                try:
                    observations = _parse_opencode_session(
                        db_path,
                        invocation_id,
                        trajectory,
                        model_ref=self.config.model_server,
                        # This fresh local database is written by one OpenCode launch.
                        clock_domain=f"opencode-wall:{uuid4().hex}",
                        agent_instance=f"opencode-agent:{uuid4().hex}",
                        observation_schema_version="2.0",
                    )
                except Exception:
                    LOG.exception("failed to read OpenCode session artifact")
                    if trajectory is not None:
                        trajectory.gaps.append(ObservationGap(code="turns_unavailable"))
                    observations = AgentObservationBundle(
                        schema_version="2.0",
                        source="opencode",
                        records=[AgentInvocation(invocation_id=invocation_id)],
                        gaps=[
                            ObservationGap(code="agent_artifact_unavailable"),
                            ObservationGap(code="agent_transcript_unavailable"),
                            ObservationGap(code="model_call_ownership_unavailable"),
                        ],
                    )
            run_status = "incomplete" if timed_out else "completed" if proc.returncode == 0 else "failed"
            for invocation in observations.records:
                if not isinstance(invocation, AgentInvocation):
                    continue
                if invocation.parent_invocation_id is None:
                    invocation.status = run_status
            if timed_out:
                observations.gaps.append(ObservationGap(code="agent_run_timeout"))
            return output_items, usage, self.config.model, observations
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: Optional[str] = None,
        collect_observations: bool = True,
    ) -> AgentEpisode:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [
            p
            for p in (
                self.config.system_prompt,
                body.instructions if self.config.system_prompt_mode == "replace" else None,
                input_system,
            )
            if p
        ]
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        prompt = user_message if system_prompt is None else f"{system_prompt}\n\n{user_message}"
        trajectory = (
            TrajectoryRecord(task_id="unscoped", rollout_id=rollout_id or "unscoped") if collect_observations else None
        )
        timeout = resolve_agent_timeout(body.metadata, ceiling=self.config.timeout)

        output_items, usage, model_name, observations = await self._run_opencode(
            user_message,
            system_prompt,
            rollout_id=rollout_id,
            collect_observations=collect_observations,
            trajectory=trajectory,
            timeout=timeout,
        )
        if collect_observations:
            observations.gaps.append(ObservationGap(code="no_sandbox_runtime"))

        if collect_observations:
            root = next(
                (
                    record
                    for record in observations.records
                    if isinstance(record, AgentInvocation) and record.parent_invocation_id is None
                ),
                None,
            )
            if root is not None and not any(
                getattr(item, "role", None) in {"user", "system", "developer"} for item in root.conversation
            ):
                conversation = [NeMoGymEasyInputMessage(role="user", content=prompt)]
                if system_prompt and self.config.system_prompt_mode == "replace":
                    conversation = [
                        NeMoGymEasyInputMessage(role="system", content=system_prompt),
                        NeMoGymEasyInputMessage(role="user", content=user_message),
                    ]
                root.conversation = [*conversation, *root.conversation]

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("OpenCode produced no assistant message. Padding empty output")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(
                    reasoning_tokens=usage.get("reasoning_tokens", 0)
                ),
                total_tokens=input_tokens + output_tokens,
            ),
            metadata={AGENT_TIMEOUT_METADATA_KEY: str(timeout)},
        )
        if trajectory is not None:
            response = response.model_copy(update={_INTERNAL_TRAJECTORY_KEY: trajectory.model_dump(mode="json")})
        return AgentEpisode(response=response, observations=observations)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is not None:
            state = self._require_agent_session(session_id)
            assert isinstance(state, OpenCodeSandboxSession)
            if state.is_continuation:
                response = await self.activate_agent_session(request, body)
                verification_failure_fields(response)
                return response
            if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
                raise HTTPException(409, "OpenCode activation does not match the seeded session and rollout")
            participant = self.checkpoint_participant
            index = self._activation_index(request)
            if participant is not None:
                episode_id = EpisodeId.from_capture_key(request.path_params["rollout_id"])
                participant.check_session(session_id, episode_id)
                if index is not None and index == state.completed_activation and state.completed_reply is not None:
                    # The caller's checkpoint precedes the reply this activation already gave: give it again.
                    participant.answered_from_saved_reply(session_id, episode_id)
                    return NeMoGymResponse.model_validate(state.completed_reply)
                if index is not None and index != state.completed_activation + 1:
                    raise HTTPException(409, f"OpenCode activation {index} is out of order")
            if state.task is None:
                if state.session is None:
                    raise HTTPException(409, "OpenCode session was restored and not re-bound: seed it again first")
                # Parse only for a new activation: a retry is compared with the stored request instead,
                # so its media are never decoded again.
                prompt, system, media = self._session_input(body)
                state.activation_request = body.model_copy(deep=True)
                state.task = asyncio.create_task(
                    self._activation_task(
                        state, state.activation_request, prompt=prompt, system=system, media=media, index=index
                    )
                )
            elif body != state.activation_request:
                raise HTTPException(409, "OpenCode sandbox sessions support one activation; retry the same request")
            response = (await asyncio.shield(state.task)).model_copy(deep=True)
            verification_failure_fields(response)
            return response
        # Only genuinely unseeded calls reach local execution; invalid or closed
        # session markers are rejected above, never retried on the host.
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        participant = self.checkpoint_participant
        # Nothing of a local run is in a checkpoint: with checkpointing on it is a restart.
        async with participant.restart_call(rollout_id) if participant is not None else nullcontext():
            episode = await self._create_episode(
                body,
                rollout_id=rollout_id,
                collect_observations=isinstance(rollout_id, str),
            )
        if not isinstance(rollout_id, str):
            return episode.response
        return episode.response.model_copy(
            update={_INTERNAL_OBSERVATIONS_KEY: episode.observations.model_dump(mode="json")}
        )

    async def run(self, request: Request, body: OpenCodeAgentRunRequest) -> OpenCodeAgentVerifyResponse:
        if self._agent_session_id_from_request(request) is not None:
            raise HTTPException(409, "OpenCode sandbox sessions must use EnvironmentServer /run")
        if self.config.resources_server is None:
            raise HTTPException(422, "Local OpenCode /run requires resources_server")
        participant = self.checkpoint_participant
        # Nothing of a local run is in a checkpoint: with checkpointing on it is a restart.
        async with (
            participant.restart_call(self.rollout_id_from_run(body)) if participant is not None else nullcontext()
        ):
            return await self._run_locally(request, body)

    async def _run_locally(self, request: Request, body: OpenCodeAgentRunRequest) -> OpenCodeAgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            rollout_id = self.rollout_id_from_run(body)
            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)
            raw_trajectory = agent_resp_json.pop(_INTERNAL_TRAJECTORY_KEY, None)
            trajectory = (
                scope_opencode_trajectory(TrajectoryRecord.model_validate(raw_trajectory), body, rollout_id)
                if isinstance(raw_trajectory, dict) and rollout_id is not None
                else None
            )
            raw_observations = (
                agent_resp_json.pop(_INTERNAL_OBSERVATIONS_KEY, None) if rollout_id is not None else None
            )
            observations = (
                AgentObservationBundle.model_validate(raw_observations) if isinstance(raw_observations, dict) else None
            )

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump()
                | {"response": agent_resp_json}
                | verification_failure_fields(NeMoGymResponse.model_validate(agent_resp_json)),
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
            turns = sum(
                1
                for item in gym_resp.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            last = gym_resp.output[-1] if gym_resp.output else None
            naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"

            return OpenCodeAgentVerifyResponse.model_validate(
                verify_json
                | {"turns_used": turns, "finished_naturally": naturally}
                | ({"ng_agent_observations": observations} if observations is not None else {})
                | ({"ng_trajectory": trajectory.model_dump(mode="json")} if trajectory is not None else {})
            )

    # -- partial-rollout checkpoints ----------------------------------------------------------------------------

    def _seed_restarts(self, body: AgentSeedSessionRequest) -> bool:
        # A sandbox the agent owns is in no checkpoint, and a multi-turn continuation session keeps its turns'
        # evidence outside one: their episodes start over after a crash.
        return body.sandbox_access is None or body.continuation is not None

    async def export_agent_sessions(
        self, session_keys: list[str], *, checkpoint_id: str
    ) -> dict[str, dict[str, JsonValue]]:
        """Each session's bookmark: its seed, where its files are, and what its launches so far produced.

        Every session is parked, its harness paused or not started, so nothing here changes while it is read.
        The sandbox itself is the resources server's to snapshot.
        """
        exported: dict[str, dict[str, JsonValue]] = {}
        for key in session_keys:
            state = self._session_state(key)
            if not isinstance(state, OpenCodeSandboxSession):
                raise TypeError(f"no open OpenCode session {key!r}")
            if state.session is None:
                # Restored and not re-bound yet: what the checkpoint it came from kept is still its state.
                exported[key] = dict(state.restored or {})
                continue
            exported[key] = {
                "request": state.request.model_dump(mode="json"),
                "session": dict(state.session.checkpoint_state()),
                "runtime": state.runtime,
                "sandbox_id": state.sandbox_id,
                "harness_session_id": state.harness_session_id,
                "launch_count": state.launch_count,
                "launch_records": list(state.launch_records),
                "observations": state.observations.model_dump(mode="json") if state.observations else None,
                "completed_activation": state.completed_activation,
                "completed_reply": state.completed_reply,
            }
            if state.keeps_processes:
                exported[key] |= {
                    "keeps_processes": True,
                    "relay_gen": state.relay_gen,
                }
        return exported

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        """Install every session without a sandbox connection; the replacement attempt's seed re-binds it."""
        restored: dict[str, OpenCodeSandboxSession] = {}
        for session in sessions:
            saved = session.session
            request = AgentSeedSessionRequest.model_validate(
                saved["request"] | {"episode_id": session.episode_id.model_dump(mode="json")}
            )
            observations = saved.get("observations")
            restored[session.session_key] = OpenCodeSandboxSession(
                request=request,
                session=None,
                runtime=str(saved["runtime"]),
                sandbox_id=saved.get("sandbox_id"),
                model_ref=self.config.model_server,
                observations=AgentObservationBundle.model_validate(observations) if observations else None,
                harness_session_id=saved.get("harness_session_id"),
                checkpointed=True,
                launch_count=int(saved["launch_count"]),
                launch_records=list(saved.get("launch_records") or []),
                completed_activation=int(saved.get("completed_activation") or 0),
                completed_reply=saved.get("completed_reply"),
                restored=dict(saved),
                keeps_processes=bool(saved.get("keeps_processes", False)),
                relay_gen=int(saved.get("relay_gen") or 0),
            )
        for key, state in restored.items():
            self._install_restored_session(key, state)

    async def _rebind_restored_agent_session_state(
        self, state: AgentSessionState, body: AgentSeedSessionRequest
    ) -> AgentSessionState:
        """Connect a restored session to the sandbox its replacement attempt borrows, a fork of the snapshot.

        The fork holds the session's files as of the checkpoint: the runtime, the supervisor files and the saved
        conversation. Files the snapshot does not hold are put back; a conversation it does not hold makes the next
        launch start the task afresh. A fork that also kept the session's processes may still run the launch the
        checkpoint held: it is taken over in place when it can be, and otherwise ended before a relaunch.
        """
        assert isinstance(state, OpenCodeSandboxSession)
        if state.session is not None:
            return state
        access = body.sandbox_access
        if access is None or not isinstance(access.connection, DirectSandboxConnection):
            raise AgentSeedRejectedError("A restored OpenCode session needs direct SandboxAccess to re-bind")
        provider = create_provider(
            resolve_provider_config(access.connection.provider_config_ref, get_global_config_dict())
        )
        try:
            sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        saved_session = state.restored["session"]
        session = SandboxSession.reattach(
            sandbox,
            saved_session,
            workdir=access.workdir,
            harness="OpenCode",
            pausable=True,
            pause_mode="hold" if state.keeps_processes else "stop",
        )
        root = session.session_root
        files = (f"{state.runtime}/opencode", f"{root}/{HARNESS_RUNNER_FILE}", f"{root}/{RESUME_SCRIPT.name}")
        if state.keeps_processes:
            files += (f"{root}/{MODEL_RELAY_SCRIPT.name}",)
        present = await sandbox.exec(
            " && ".join(f"[ -e {quote(path)} ]" for path in files),
            cwd="/",
            timeout_s=self.config.session_close_timeout_seconds,
        )
        if present.return_code != 0 or present.error_type:
            LOG.warning("Restored OpenCode session %s lacks its runtime or files; installing them again", root)
            command = _sandbox_prepare_command(access.workdir, root, state.runtime, export_dir=self.config.export_dir)
            result = await exec_idempotent(
                sandbox,
                command,
                what="OpenCode sandbox preparation",
                timeout_s=self.config.sandbox_install_timeout_seconds,
            )
            self._check_sandbox_setup(command, result)
            await self._install_session_files(
                sandbox,
                state,
                root,
                access.workdir,
                state.runtime,
                checkpointed=True,
                model_relay=state.keeps_processes,
            )
        state.session = session
        state.sandbox_id = sandbox.sandbox_id
        state.restored = None
        if state.keeps_processes:
            await self._check_restored_launch(state, saved_session)
        return state

    async def _check_restored_launch(self, state: OpenCodeSandboxSession, saved: Mapping[str, Any]) -> None:
        """Decide what becomes of the launch a process-keeping restore brought back: take it over, or end it.

        Only a launch whose supervisor held a confirmed pause at the checkpoint can still run. It is taken over
        when its supervisor, its pause and its model relay all survived the restore, and also when it had already
        ended (its cleanup receipt is there): the take-over then just collects what it left. Otherwise it is
        ended, without letting it run again, and the next launch resumes the saved conversation as after a
        files-only restore.
        """
        pause_seq, identity = saved.get("pause_seq"), saved.get("supervisor_identity")
        if pause_seq is None or identity is None:
            return
        session = state.require_session()
        verdict = await check_restored(
            session.sandbox,
            session.session_dir,
            pause_seq=pause_seq,
            identity=identity,
            relay_status_path=f"{session.session_dir}/{RELAY_STATUS_FILE}",
            timeout=self.config.session_close_timeout_seconds,
        )
        if isinstance(verdict, RestoredLaunch) or verdict.reason == "launch_finished":
            state.adoption = RestoredLaunch(pause_seq=pause_seq, identity=identity)
            return
        LOG.log(
            logging.INFO if verdict.reason in _EXPECTED_RESTORE_FALLBACKS else logging.WARNING,
            "OpenCode cannot resume in place: session %s, reason %s, detail %s; relaunching",
            state.request.agent_session_id,
            verdict.reason,
            verdict.detail,
        )
        await self._abandon_restored_launch(state, identity, reason=verdict.reason)

    async def _abandon_restored_launch(self, state: OpenCodeSandboxSession, identity: str, *, reason: str) -> None:
        """End a restored launch that will not be taken over, so nothing of it runs beside its relaunch."""
        session = state.require_session()
        try:
            receipt = await abandon_launch(
                session.sandbox,
                session.session_dir,
                identity=identity,
                timeout=self.config.session_close_timeout_seconds,
                harness="OpenCode",
            )
        except Exception:
            LOG.exception(
                "OpenCode could not abandon the restored launch of session %s (reason %s)",
                state.request.agent_session_id,
                reason,
            )
            return
        LOG.log(
            logging.INFO if reason in _EXPECTED_RESTORE_FALLBACKS else logging.WARNING,
            "OpenCode abandoned a restored launch: session %s, reason %s, receipt %s",
            state.request.agent_session_id,
            reason,
            json.dumps(dict(receipt), sort_keys=True),
        )

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> OpenCodeSandboxSession:
        """Prepare only the harness runtime; Resources owns borrowed task setup."""
        if self.config.model_server is None:
            raise HTTPException(422, "OpenCode sandbox execution requires model_server")
        owns_sandbox = body.sandbox_access is None
        if owns_sandbox:
            if not self.config.sandbox_provider:
                raise HTTPException(422, "OpenCode requires sandbox_access or a configured sandbox_provider")
            spec = SandboxSpec(**{"workdir": "/app", **self.config.sandbox_config})
            workdir = spec.workdir
            provider_ref = self.config.sandbox_provider
        else:
            if not isinstance(body.sandbox_access.connection, DirectSandboxConnection):
                raise HTTPException(422, "OpenCode requires direct SandboxAccess")
            if self.config.network_access != "inherit":
                raise AgentSeedRejectedError(
                    "OpenCode network_access applies only to a sandbox it creates, not a borrowed one"
                )
            workdir = body.sandbox_access.workdir
            provider_ref = body.sandbox_access.connection.provider_config_ref
        if not isinstance(workdir, str):
            raise HTTPException(422, "OpenCode sandbox workdir must be absolute")
        normalized_workdir = PurePosixPath(workdir)
        if (
            not normalized_workdir.is_absolute()
            or "\x00" in workdir
            or ".." in normalized_workdir.parts
            or normalized_workdir in (PurePosixPath("/"), PurePosixPath("/tmp"))
            or str(normalized_workdir).startswith("/tmp/nemo-gym-opencode")
        ):
            raise HTTPException(
                422, "OpenCode sandbox workdir must be absolute and separate from adapter runtime/session storage"
            )
        unsupported = [
            access.name
            for access in self.effective_tool_accesses(body)
            if access.required and not isinstance(access, MCPToolAccess)
        ]
        if unsupported:
            raise HTTPException(
                422,
                "OpenCode in the task sandbox supports only MCP tool grants; required grants it cannot use: "
                + ", ".join(sorted(unsupported)),
            )
        if not isinstance(self.config.opencode_version, str) or not re.fullmatch(
            r"\d+\.\d+\.\d+", self.config.opencode_version
        ):
            raise HTTPException(422, "OpenCode sandbox execution requires an exact opencode_version")
        if self.config.opencode_version != _TESTED_OUTCOME_VERSION:
            LOG.warning(
                "OpenCode %s is untested: error classification is validated on %s",
                self.config.opencode_version,
                _TESTED_OUTCOME_VERSION,
            )
        if self.config.context_window <= 0 or self.config.timeout <= 0:
            raise HTTPException(422, "OpenCode context window and execution timeout must be positive")
        if not 0 < self.config.max_output_tokens <= self.config.context_window:
            raise HTTPException(
                422, "OpenCode sandbox max_output_tokens must be positive and not exceed context_window"
            )
        config_error = _session_opencode_config_error(self.config.opencode_config)
        if config_error is not None:
            raise HTTPException(422, f"OpenCode sandbox config: {config_error}")
        if self.config.env or self.config.extra_args or self.config.command != "opencode":
            raise HTTPException(422, "env, extra_args and command overrides are supported only by local OpenCode")
        if not all(_is_sandbox_path(path) for path in self.config.plugins):
            raise AgentSeedRejectedError("OpenCode plugins must be absolute, normalized paths inside the sandbox")
        if self.config.export_dir is not None and (
            not _is_sandbox_path(self.config.export_dir)
            or PurePosixPath(self.config.export_dir) == PurePosixPath("/")
            or str(PurePosixPath(self.config.export_dir)).startswith("/tmp/nemo-gym-opencode")
        ):
            raise AgentSeedRejectedError(
                "OpenCode export_dir must be an absolute sandbox path outside adapter runtime/session storage"
            )
        # The prepare step repeats this check on resolved paths, for links inside the sandbox.
        if self.config.export_dir is not None and (
            normalized_workdir == PurePosixPath(self.config.export_dir)
            or normalized_workdir in PurePosixPath(self.config.export_dir).parents
            or PurePosixPath(self.config.export_dir) in normalized_workdir.parents
        ):
            raise AgentSeedRejectedError(
                "OpenCode export_dir must be neither the task workdir, inside it nor above it"
            )
        if self.config.session_import_path is not None and not _is_sandbox_path(self.config.session_import_path):
            raise AgentSeedRejectedError("OpenCode session_import_path must be an absolute, normalized sandbox path")
        ripgrep = self._pinned_file("ripgrep")
        plugin_dependencies = self._pinned_file("plugin_dependencies")
        provider = create_provider(resolve_provider_config(provider_ref, get_global_config_dict()))
        if owns_sandbox:
            try:
                spec = restrict_sandbox_network(
                    spec,
                    network_access=self.config.network_access,
                    provider=provider,
                    model_base_url=self.resolve_model_base_url(self.config.model_server.name, trace_context=False),
                    tool_urls=[
                        str(access.connection.url)
                        for access in self.effective_tool_accesses(body)
                        if isinstance(access, MCPToolAccess)
                    ],
                    # model_only blocks these hosts, so a required grant is refused before the sandbox exists.
                    required_tool_urls=[
                        str(access.connection.url)
                        for access in self.effective_tool_accesses(body)
                        if isinstance(access, MCPToolAccess) and access.required
                    ],
                )
            except BaseException:
                await provider.aclose()
                raise
            sandbox = AsyncSandbox(provider)
        else:
            try:
                sandbox = await AsyncSandbox.connect(body.sandbox_access.connection.descriptor, provider=provider)
            except BaseException:
                await provider.aclose()
                raise
        # Caller-assigned IDs are wire identifiers, never filesystem paths.
        directory = f"{self.config.sandbox_session_root.rstrip('/')}/{uuid4().hex}"
        runtime = f"/tmp/nemo-gym-opencode-runtime-{self.config.opencode_version}"
        checkpointed = self.checkpoint_participant is not None and not self._seed_restarts(body)
        # A restore of this sandbox brings its processes back: the harness keeps running through checkpoints and
        # calls the model through a relay a restore can re-point.
        keeps_processes = checkpointed and bool(getattr(provider, "snapshot_keeps_processes", False))
        state = OpenCodeSandboxSession(
            request=body,
            session=SandboxSession(
                sandbox=sandbox,
                session_dir=directory,
                workdir=workdir,
                harness="OpenCode",
                owns_sandbox=owns_sandbox,
                pausable=checkpointed,
                pause_mode="hold" if keeps_processes else "stop",
            ),
            runtime=runtime,
            model_ref=self.config.model_server,
            export_dir=self.config.export_dir,
            opencode_environment=cli_environment(
                workdir=workdir,
                env={**_private_directories(directory), **_OPENCODE_ISOLATION},
                path_prefix=self.config.sandbox_path_prefix,
            ),
            checkpointed=checkpointed,
            keeps_processes=keeps_processes,
            relay_gen=1 if keeps_processes else 0,
        )
        prepared_directory = False
        try:
            if owns_sandbox:
                await sandbox.start(spec)
                # Providers need not create SandboxSpec.workdir. Never prepare a borrowed task here.
                workspace = await exec_idempotent(
                    sandbox,
                    f"mkdir -p -- {shlex.quote(workdir)}",
                    what="OpenCode workdir creation",
                    cwd="/",
                    timeout_s=30,
                )
                if workspace.return_code != 0 or workspace.error_type:
                    raise NgFailureEvent(
                        boundary="harness",
                        failure_kind=AGENT_ROUTE_FAILED,
                        operation="opencode.setup",
                        stage="seed",
                        failure_reason=f"Cannot create OpenCode sandbox workdir {workdir}: {workspace.stderr or workspace.stdout}",
                    )
            state.sandbox_id = sandbox.sandbox_id
            if state.sandbox_id is None:
                raise NgFailureEvent(
                    boundary="harness",
                    failure_kind=AGENT_ROUTE_FAILED,
                    operation="opencode.setup",
                    stage="seed",
                    failure_reason="OpenCode sandbox provider did not return a sandbox_id",
                )
            command = _sandbox_prepare_command(workdir, directory, runtime, export_dir=self.config.export_dir)
            # Checks before it installs, so running it again is safe.
            result = await exec_idempotent(
                sandbox,
                command,
                what="OpenCode sandbox preparation",
                timeout_s=self.config.sandbox_install_timeout_seconds,
            )
            self._check_sandbox_setup(command, result)
            prepared_directory = True
            await self._install_session_files(
                sandbox,
                state,
                directory,
                workdir,
                runtime,
                checkpointed=checkpointed,
                ripgrep=ripgrep,
                plugin_dependencies=plugin_dependencies,
                model_relay=keeps_processes,
            )
            if self.config.session_import_path is not None:
                await self._import_session(state)
        except BaseException as error:
            try:
                if prepared_directory or owns_sandbox:
                    await state.close(self.config.session_close_timeout_seconds)
                else:
                    await sandbox.disconnect()
            except BaseException:
                LOG.exception("OpenCode seed cleanup failed; retaining session %s", body.agent_session_id)
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    def _pinned_file(self, kind: str) -> _PinnedFile | None:
        """Validate ``sandbox_<kind>_path`` and ``sandbox_<kind>_sha256`` before any sandbox is touched."""
        path_field, digest_field = f"sandbox_{kind}_path", f"sandbox_{kind}_sha256"
        path, digest = getattr(self.config, path_field), getattr(self.config, digest_field)
        if not path:
            if digest:
                raise AgentSeedRejectedError(f"{digest_field} pins a supplied file; set {path_field}")
            return None
        if not digest:
            raise AgentSeedRejectedError(f"A supplied file requires {digest_field}")
        if not _is_sandbox_path(path):
            raise AgentSeedRejectedError(f"{path_field} must be an absolute, normalized sandbox path")
        return _PinnedFile(digest=digest.lower(), sandbox_path=path)

    async def _stage_executable(
        self, state: OpenCodeSandboxSession, pin: _PinnedFile, *, target: str, what: str
    ) -> None:
        """Install a pinned executable at ``target`` once a private copy matches its SHA-256 and answers ``--version``."""
        command = (
            f"set -- {quote(pin.sandbox_path)} {pin.digest} {quote(_pinned_copy(target))} {quote(target)}\n"
            + STAGE_EXECUTABLE
        )
        result = await state.session.sandbox.exec(
            command, cwd="/", timeout_s=self.config.sandbox_install_timeout_seconds
        )
        self._check_sandbox_setup(command, result, pinned=what)

    async def _stage_archive(self, state: OpenCodeSandboxSession, pin: _PinnedFile, *, target: str, what: str) -> None:
        """Unpack a pinned tar archive into ``target`` once a private copy matches its SHA-256 and its members pass
        ``check_archive``, which runs here on a download of that copy."""
        sandbox, timeout = state.session.sandbox, self.config.sandbox_install_timeout_seconds
        copy = _pinned_copy(target)
        command = f"set -- {quote(pin.sandbox_path)} {pin.digest} {quote(copy)}\n" + COPY_PINNED
        self._check_sandbox_setup(command, await sandbox.exec(command, cwd="/", timeout_s=timeout), pinned=what)
        with tempfile.TemporaryDirectory(prefix="opencode-pinned-") as directory:
            archive = Path(directory) / "archive"
            async with asyncio.timeout(timeout):
                await sandbox.download(copy, archive)
            try:
                await asyncio.to_thread(check_archive, archive, pin.digest)
            except ValueError as error:
                raise AgentSeedRejectedError(f"Pinned {what} rejected: {error}") from error
        command = f"set -- {quote(copy)} {quote(target)}\n" + UNPACK_ARCHIVE
        self._check_sandbox_setup(command, await sandbox.exec(command, cwd="/", timeout_s=timeout))

    async def _import_session(self, state: OpenCodeSandboxSession) -> None:
        """Import the owner's staged export, if present, so the first activation continues that session.

        ``opencode import`` runs with the session's private directories and without plugins, from the task workdir:
        OpenCode binds an imported session to the importing process's directory. The prefix then stays out of
        output, usage and trajectory through ``previous_message_ids``, as for continuations. Only an export that
        OpenCode refuses, or one that leaves other than one root session, refuses the seed; a signal, a timeout or a
        command that never started stays retryable.
        """
        sandbox, path = state.session.sandbox, self.config.session_import_path
        assert path is not None
        timeout = self.config.sandbox_install_timeout_seconds
        command = f"test -f {quote(path)}"
        staged = await sandbox.exec(command, cwd="/", timeout_s=30)
        if staged.return_code == 1 and staged.error_type is None:
            return
        self._check_sandbox_setup(command, staged)
        private = _private_directories(state.persistent_directory).values()
        command = (
            f"mkdir -p -- {' '.join(map(quote, private))} || exit {HARNESS_RUNNER_FAILED}\n"
            + state.opencode_environment
            + f"exec {quote(f'{state.runtime}/opencode')} import {quote(path)} </dev/null"
        )
        result = await sandbox.exec(command, cwd="/", timeout_s=timeout)
        code = result.return_code
        if (
            result.error_type is None
            and 0 < code < 128
            and code not in (HARNESS_RUNNER_FAILED, *HARNESS_COMMAND_NOT_STARTED)
        ):
            raise AgentSeedRejectedError(
                format_error_message(
                    f"OpenCode could not import session_import_path (exit {code})",
                    stderr=result.stderr or result.stdout or "",
                )
            )
        self._check_sandbox_setup(command, result)
        with tempfile.TemporaryDirectory(prefix="opencode-import-") as directory:
            async with asyncio.timeout(timeout):
                names = await state.list_store(timeout=timeout)
                if "opencode.db" not in names:
                    raise RuntimeError("OpenCode import left no session store")
                await state.download_store(names, Path(directory))
            database = Path(directory) / "opencode.db"
            sessions = await asyncio.to_thread(stored_sessions, database)
            roots = [session["id"] for session in sessions if session["parent_id"] is None]
            if len(roots) != 1:
                raise AgentSeedRejectedError(f"Expected one imported root session, found {len(roots)}")
            state.harness_session_id = roots[0]
            state.message_ids = await asyncio.to_thread(stored_message_ids, database)

    def _granted_mcp_servers(self, request: AgentSeedSessionRequest) -> tuple[dict[str, Any], dict[str, list[str]]]:
        """OpenCode ``mcp`` entries for a session's MCP tool grants, and the grant names by requirement.

        The model sees each grant's tools as ``<grant>_<tool>``. OpenCode connects when the run starts and leaves a
        server that fails to connect out of the tool list, so ``required_mcp.mjs`` checks the grants before the first
        model call and ``_check_mcp_grants`` reads the result.
        """
        servers = {}
        grants: dict[str, list[str]] = {"required": [], "optional": []}
        for access in self.effective_tool_accesses(request):
            if not isinstance(access, MCPToolAccess):
                continue
            entry: dict[str, Any] = {
                "type": "remote",
                "url": str(access.connection.url),
                "enabled": True,
                # The grant's headers authenticate the session; a headless run must not start an OAuth flow.
                "oauth": False,
            }
            if access.connection.headers:
                entry["headers"] = dict(access.connection.headers)
            if self.config.mcp_tool_timeout_seconds is not None:
                entry["timeout"] = round(self.config.mcp_tool_timeout_seconds * 1000)
            servers[access.name] = entry
            grants["required" if access.required else "optional"].append(access.name)
        return servers, grants

    @staticmethod
    async def _check_mcp_grants(
        state: OpenCodeSandboxSession, grants: dict[str, list[str]], observations: AgentObservationBundle
    ) -> tuple[str | None, str | None]:
        """Read the status ``required_mcp.mjs`` recorded, and add a gap for each optional grant that did not connect.

        Returns the error for required grants it recorded as not connected, and the error to use when no status
        confirms the required grants: the plugin writes it before the first model call, so a missing, null or
        malformed status means the required tools were never confirmed.
        """
        unconfirmed = (
            "Required Gym MCP tools were not confirmed before the model call: " + ", ".join(grants["required"])
            if grants["required"]
            else None
        )
        try:
            statuses = json.loads(await state.read_text("mcp-status.json"))
        except Exception:
            LOG.warning("OpenCode MCP grant status unavailable", exc_info=True)
            statuses = None
        if not isinstance(statuses, dict):
            observations.gaps.append(ObservationGap(code="mcp_status_unavailable"))
            return None, unconfirmed
        missing = []
        for name in [*grants["required"], *grants["optional"]]:
            entry = statuses.get(name) if isinstance(statuses.get(name), dict) else {}
            if entry.get("status") == "connected":
                continue
            detail = f"{name}: {entry.get('status', 'unavailable')}" + (
                f" ({entry['error']})" if entry.get("error") else ""
            )
            if name in grants["required"]:
                missing.append(detail)
            else:
                observations.gaps.append(ObservationGap(code="mcp_server_skipped", detail=detail))
        if missing:
            return "Required Gym MCP tools could not be initialized: " + "; ".join(missing), None
        return None, None

    async def _install_session_files(
        self,
        sandbox: AsyncSandbox,
        state: OpenCodeSandboxSession,
        directory: str,
        workdir: str,
        runtime: str,
        *,
        checkpointed: bool,
        ripgrep: "_PinnedFile | None" = None,
        plugin_dependencies: "_PinnedFile | None" = None,
        model_relay: bool = False,
    ) -> None:
        """Install the OpenCode runtime if the sandbox lacks it, and upload the session's helper files.

        ``ripgrep`` and ``plugin_dependencies`` are the pinned files a seed resolved, staged under the session's
        persistent directory. A checkpointed session also gets the script that resumes its saved conversation.
        """
        if self.config.plugins:
            # Owner-staged plugins are checked before installation.
            command = f"set -- {' '.join(map(quote, self.config.plugins))}\n{_CHECK_PLUGIN_FILES}"
            result = await sandbox.exec(command, cwd="/", timeout_s=30)
            if result.return_code == _INPUT_REJECTED and result.error_type is None:
                raise AgentSeedRejectedError(
                    f"OpenCode plugin files are missing from the sandbox: {(result.stdout or '').strip()}"
                )
            self._check_sandbox_setup(command, result)
        installer = f"{directory}/{RUNTIME_INSTALLER_FILE}"
        await sandbox.upload(RUNTIME_INSTALLER_SOURCE, installer)
        # A pinned rg (below) replaces the installer's best-effort package install of ripgrep.
        command = "NG_OPENCODE_SKIP_RIPGREP=1 " if ripgrep is not None else ""
        command += "bash " + " ".join(
            quote(arg)
            for arg in (
                installer,
                runtime,
                self.config.opencode_version,
                self.config.sandbox_executable_path or "",
                self.config.sandbox_executable_sha256 or "",
                ":".join(self.config.sandbox_path_prefix),
            )
        )
        # The installer serializes concurrent runs and skips a runtime already in place, so running it again is safe.
        result = await exec_idempotent(
            sandbox,
            command,
            what="OpenCode runtime installation",
            cwd=workdir,
            timeout_s=self.config.sandbox_install_timeout_seconds,
        )
        self._check_sandbox_setup(command, result)
        await sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")
        if ripgrep is not None:
            # OpenCode 1.17.11 runs the rg on PATH, else <XDG_CACHE_HOME>/opencode/bin/rg, else downloads one.
            # The private cache keeps the pin off task shells' PATH; an rg on the task PATH still takes precedence.
            await self._stage_executable(
                state,
                ripgrep,
                target=f"{state.persistent_directory}/cache/opencode/bin/rg",
                what="ripgrep executable",
            )
        if plugin_dependencies is not None:
            # With any plugin configured, OpenCode installs @opencode-ai/plugin into XDG_CONFIG_HOME/opencode
            # unless node_modules exists and package-lock.json covers package.json.
            await self._stage_archive(
                state,
                plugin_dependencies,
                target=f"{state.persistent_directory}/config/opencode",
                what="plugin dependency archive",
            )
        await sandbox.upload(TASK_ENVIRONMENT_PLUGIN_SOURCE, f"{directory}/{TASK_ENVIRONMENT_PLUGIN_FILE}")
        if self._model_call_capture_enabled():
            await sandbox.upload(ASSISTANT_MESSAGE_PLUGIN_SOURCE, f"{directory}/{ASSISTANT_MESSAGE_PLUGIN_FILE}")
        if checkpointed:
            await sandbox.upload(RESUME_SCRIPT, f"{directory}/{RESUME_SCRIPT.name}")
        if model_relay:
            await sandbox.upload(MODEL_RELAY_SCRIPT, f"{directory}/{MODEL_RELAY_SCRIPT.name}")

    @staticmethod
    def _check_sandbox_setup(command: str, result: SandboxExecResult, *, pinned: str | None = None) -> None:
        """Raise on a failed setup step; a ``pinned`` file that fails its checks refuses the seed (a terminal 422)."""
        if pinned is not None and result.return_code == _INPUT_REJECTED and result.error_type is None:
            raise AgentSeedRejectedError(format_error_message(f"Pinned {pinned} rejected", stderr=result.stderr or ""))
        if result.return_code != 0 or result.error_type is not None:
            raise NgFailureEvent(
                boundary="harness",
                failure_kind=AGENT_ROUTE_FAILED,
                operation="opencode.setup",
                stage="seed",
                failure_reason=command_failure_reason(f"OpenCode setup failed: command={command!r}", result),
            )

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Finalize and release the connection; retain release failures for retry."""
        assert isinstance(state, OpenCodeSandboxSession)
        if state.session is None:
            # Restored from a checkpoint and closed before its replacement attempt re-bound it to a sandbox:
            # the evidence is what the checkpoint kept, and there is no connection to release.
            LOG.warning(
                "Closing OpenCode session %s, which was restored and never re-bound; returning its saved evidence",
                state.request.agent_session_id,
            )
            return AgentCloseSessionResponse(
                agent_session_id=state.request.agent_session_id,
                agent_observations=state.observations
                or AgentObservationBundle(
                    source="opencode", gaps=[ObservationGap(code="agent_activation_interrupted")]
                ),
            )
        await state.close(self.config.session_close_timeout_seconds)
        observations = state.observations or AgentObservationBundle(
            schema_version="2.0", source="opencode", gaps=[ObservationGap(code="agent_activation_interrupted")]
        )
        # A capture that close finished after the activation returned reports its export failure here.
        observations.gaps.extend(state.take_export_gaps())
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=observations,
            trajectory=state.trajectory if state.is_continuation or self.config.return_session_trajectory else None,
            cleanup_confirmed=state.session.cleanup_confirmed if state.is_continuation else None,
        )

    def _session_input(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str, list[MediaFile]]:
        """Return the user prompt, the system instructions and the prompt's media files."""
        # Sampling is applied to every OpenCode agent (see _session_response); other controls are rejected.
        supported = {"input", "instructions", "model", "temperature", "top_p"}
        sampled = [name for name in ("temperature", "top_p") if getattr(body, name) is not None]
        for name in sampled:
            if not math.isfinite(getattr(body, name)):
                raise HTTPException(422, f"OpenCode sandbox execution requires a finite {name}")
        if sampled and self.config.opencode_version not in _OPENCODE_AGENTS:
            raise HTTPException(
                422,
                f"OpenCode sandbox execution applies request {' and '.join(sampled)} only with OpenCode "
                f"{', '.join(_OPENCODE_AGENTS)}, whose built-in agents it knows; configure Gym model sampling",
            )
        for key, field in type(body).model_fields.items():
            if key in supported:
                continue
            value = getattr(body, key)
            if key in ("stream", "background") and value is False:
                continue
            if key == "metadata" and value and set(value) <= {AGENT_TIMEOUT_METADATA_KEY}:
                continue
            # Gym agents read a request's max_output_tokens as the limit for each model call (simple_agent copies it
            # into every call), and NeMo RL sets it on every row as its generation limit. OpenCode's own limit for
            # each call (config.max_output_tokens) already stays within a value at least that large. A tighter one is
            # refused below. Nothing limits the sum over a session's calls, as in simple_agent.
            if key == "max_output_tokens" and value is not None and value >= self.config.max_output_tokens:
                continue
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(
                    422,
                    f"OpenCode sandbox execution does not support request field {key}; configure Gym model limits/sampling",
                )
        if body.model not in (None, "", "dummy_model", self.config.model_server.name):
            raise HTTPException(422, "OpenCode sandbox execution uses the configured Gym model_server")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        roles = [getattr(item, "role", None) for item in items]
        if roles not in (["user"], ["system", "user"], ["developer", "user"]):
            raise HTTPException(
                422, "OpenCode sandbox execution accepts one user prompt and an optional system or developer message"
            )
        texts = []
        media: list[MediaFile] = []
        for item in items:
            try:
                text, item_media = split_text_and_media(item.content)
            except UnsupportedContentError as error:
                raise HTTPException(422, f"OpenCode sandbox execution: {error}") from error
            if item_media and item is not items[-1]:
                raise HTTPException(422, "OpenCode sandbox execution accepts media only in the user prompt")
            texts.append(text)
            media = item_media
        if not texts[-1].strip():
            raise HTTPException(422, "OpenCode sandbox execution requires a nonempty user prompt")
        system = "\n\n".join(text for text in [self.config.system_prompt, body.instructions, *texts[:-1]] if text)
        return texts[-1], system, media

    def _session_output(
        self, export: dict[str, Any], observations: AgentObservationBundle
    ) -> list[NeMoGymResponseOutputItem]:
        output = []
        gym_reasoning = self.config.model_server is not None
        for message in export.get("messages", []):
            if message.get("info", {}).get("role") != "assistant":
                continue
            parts = message.get("parts", [])
            # The empty-answer decision needs the whole message; each part still converts on its own.
            empty_answer = _empty_answer_part(parts) if gym_reasoning else None
            for index, part in enumerate(parts):
                if part.get("type") in (
                    "step-start",
                    "step-finish",
                    "patch",
                    "snapshot",
                    "compaction",
                    "agent",
                    "retry",
                ):
                    continue
                try:
                    items = _assistant_part_items(
                        part, gym_reasoning=gym_reasoning, keep_empty_answer=index == empty_answer
                    )
                    if part.get("type") == "tool":
                        state = part.get("state") or {}
                        for item in items:
                            item.status = "completed" if state.get("status") == "completed" else "incomplete"
                            if isinstance(item, NeMoGymFunctionCallOutput):
                                item.output = state.get("output") or state.get("error") or ""
                    output.extend(items)
                except Exception:
                    # A malformed or newer artifact must not discard already captured turns.
                    observations.gaps.append(
                        ObservationGap(code="agent_artifact_record_unparseable", detail=str(part.get("type")))
                    )
        return output

    @staticmethod
    def _session_usage(export: dict[str, Any]) -> NeMoGymResponseUsage | None:
        # OpenCode 1.17.11 getUsage subtracts cache from input and reasoning from output.
        # Restore inclusive OpenAI counters across the root and subagent model turns.
        infos = export.get("usage_messages", [message["info"] for message in export.get("messages", [])])
        usages = []
        missing_usage = False
        for info in infos:
            if info.get("role") != "assistant":
                continue
            tokens = info.get("tokens")
            if not isinstance(tokens, dict) or not tokens:
                missing_usage = True
                continue
            cache = tokens.get("cache")
            cache = cache if isinstance(cache, dict) else {}

            def count(value: object) -> int:
                return value if type(value) is int and value >= 0 else 0

            cache_read = count(cache.get("read"))
            reasoning = count(tokens.get("reasoning"))
            input_tokens = int(tokens.get("input") or 0) + cache_read + count(cache.get("write"))
            output_tokens = int(tokens.get("output") or 0) + reasoning
            # Pinned 1.17.11 Session.getUsage defaults absent/invalid optional counters
            # to zero before persistence. Only positive values establish measured details;
            # a stored zero cannot prove a backend-reported zero.
            usages.append(
                NeMoGymResponseUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cache_read or None),
                    output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning or None),
                )
            )
        total = NeMoGymResponseUsage.sum_from_list(usages) if usages else None
        if total is not None and missing_usage:
            total.input_tokens_details.cached_tokens = None
            total.output_tokens_details.reasoning_tokens = None
        return total

    def _agent_continuation_capabilities(self) -> AgentContinuationCapabilities:
        return AgentContinuationCapabilities(
            observations=["trajectory", "timing", "harness_tool_calls", "reasoning", "compaction"],
        )

    def _continuation_close_timeout(self) -> float:
        return self.config.session_close_timeout_seconds

    def _validate_agent_activation(
        self, state: AgentSessionState, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> None:
        assert isinstance(state, OpenCodeSandboxSession)
        prompt, system, media = self._session_input(body)
        # The base class passes this same request object to _activate_agent_session_state,
        # which reuses the result so the media are decoded once.
        state.validated_input = (body, prompt, system, media)
        supplied_instructions = body.instructions is not None or (
            not isinstance(body.input, str)
            and any(getattr(item, "role", None) in {"system", "developer"} for item in body.input)
        )
        if state.system_instructions is not None and supplied_instructions and system != state.system_instructions:
            raise HTTPException(422, "OpenCode continuation cannot replace session instructions")

    async def _activate_agent_session_state(
        self, state: AgentSessionState, body: NeMoGymResponseCreateParamsNonStreaming, *, activation_id: int
    ) -> NeMoGymResponse:
        """Run a delta under one budget, including preparation and the concurrency queue."""
        assert isinstance(state, OpenCodeSandboxSession)
        validated, state.validated_input = state.validated_input, None
        if validated is not None and validated[0] is body:
            _, prompt, system, media = validated
        else:
            # Another request validated after this one, so its result does not apply here.
            prompt, system, media = self._session_input(body)
        if state.system_instructions is None:
            state.system_instructions = system
        timeout = resolve_agent_timeout(body.metadata, ceiling=self.config.timeout)
        started = monotonic()
        deadline = started + timeout
        failure: BaseException | None = None
        try:
            async with asyncio.timeout_at(deadline):
                await state.prepare_activation(
                    activation_id, timeout=min(timeout, self.config.session_close_timeout_seconds)
                )
            return await self._session_response(
                state,
                body,
                prompt=prompt,
                system=state.system_instructions,
                media=media,
                timeout=timeout,
                deadline=deadline,
            )
        except (TimeoutError, _ActivationDeadlineExpired) as error:
            if state.session.launch_started:
                failure = error
                raise
            # No OpenCode session is required until a launch persists one. A
            # pre-launch timeout therefore does not prevent the next activation.
            invocation_id = state.invocation_prefix
            if state.harness_session_id is not None:
                invocation_id += f":{state.harness_session_id}"
            invocation = AgentInvocation(
                invocation_id=invocation_id,
                activation_id=activation_id,
                harness_session_id=state.harness_session_id,
                status="incomplete",
                error_type="timeout",
                stop_reason="timeout",
                duration_ms=(monotonic() - started) * 1000,
            )
            state.activation_observations = AgentObservationBundle(
                schema_version="2.0", source="opencode", records=[invocation]
            )
            state.activation_trajectory = TrajectoryRecord(
                task_id=state.request.task_id.task_id,
                rollout_id=state.request.episode_id.capture_key,
            )
            return self._make_session_response(
                state,
                body,
                timeout=timeout,
                outcome=OpenCodeRunOutcome(status="incomplete", error_type="timeout"),
            )
        except BaseException as error:
            failure = error
            raise
        finally:
            # execute() finishes capture before propagating cancellation. Retain
            # once here, including preparation failures and interrupted launches.
            if failure is not None:
                observations = state.activation_observations
                if observations is None:
                    observations = AgentObservationBundle(schema_version="2.0", source="opencode")
                    state.activation_observations = observations
                if not any(
                    isinstance(r, AgentInvocation) and r.parent_invocation_id is None for r in observations.records
                ):
                    invocation_id = state.invocation_prefix
                    if state.harness_session_id is not None:
                        invocation_id += f":{state.harness_session_id}"
                    cancelled = isinstance(failure, asyncio.CancelledError)
                    observations.records.append(
                        AgentInvocation(
                            invocation_id=invocation_id,
                            activation_id=activation_id,
                            harness_session_id=state.harness_session_id,
                            status="incomplete" if cancelled else "failed",
                            error_type="cancelled" if cancelled else "server_error",
                            duration_ms=(monotonic() - started) * 1000,
                        )
                    )
            state.retain_activation()

    def _make_session_response(
        self,
        state: OpenCodeSandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        timeout: float,
        outcome: OpenCodeRunOutcome,
        output: list[NeMoGymResponseOutputItem] | None = None,
        usage: NeMoGymResponseUsage | None = None,
        error_message: str | None = None,
    ) -> NeMoGymResponse:
        runtime = state.runtime_info
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model_server.name,
            object="response",
            output=output or [],
            usage=usage,
            status=outcome.status,
            error={"code": "server_error", "message": error_message} if error_message is not None else None,
            incomplete_details=IncompleteDetails(reason=outcome.incomplete_reason)
            if state.is_continuation and outcome.incomplete_reason
            else None,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            metadata={
                "harness_execution": "sandbox",
                "opencode_version": self.config.opencode_version,
                **({"harness_launches": str(state.launch_count)} if state.checkpointed else {}),
                **({"harness_hostname": runtime.hostname, "harness_pid": str(runtime.pid)} if runtime else {}),
                AGENT_TIMEOUT_METADATA_KEY: str(timeout),
            },
        )
        if state.is_continuation:
            response.metadata.update({"activation_id": str(state.activation_id)})
            if state.harness_session_id is not None:
                response.metadata["harness_session_id"] = state.harness_session_id
            response = response.model_copy(
                update={
                    _INTERNAL_OBSERVATIONS_KEY: state.activation_observations.model_dump(mode="json"),
                    _INTERNAL_TRAJECTORY_KEY: state.activation_trajectory.model_dump(mode="json")
                    if state.activation_trajectory is not None
                    else None,
                }
            )
        fields = verification_failure_fields()
        return response.model_copy(update=fields) if fields["failure_report"] is not None else response

    async def _session_response(
        self,
        state: OpenCodeSandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        prompt: str,
        system: str,
        media: Sequence[MediaFile] = (),
        timeout: float | None = None,
        deadline: float | None = None,
        activation: Activation | None = None,
    ) -> NeMoGymResponse:
        timeout = timeout if timeout is not None else resolve_agent_timeout(body.metadata, ceiling=self.config.timeout)
        started = monotonic()
        # A text-only deployment rejects a request that carries an image, so image input is declared only for prompt
        # images or when the configuration asks for it. A resumed session keeps declaring it, because OpenCode resends
        # earlier images with the history and would otherwise replace them with its error text.
        state.declares_image_input = state.declares_image_input or bool(media) or self.config.image_input
        previous_session_id = state.harness_session_id
        failure = None
        try:
            # The slot goes back on every path, including a retire or close while a checkpoint holds it parked.
            async with (
                activation.slot(self.sem, deadline=deadline)
                if activation is not None
                else _slot(self.sem, deadline=deadline)
            ):
                if deadline is None:
                    # A first activation's budget starts once it holds a slot, as Pi's does: staging spends it,
                    # and OpenCode runs for what remains.
                    deadline = monotonic() + timeout
                if state.checkpointed and activation is not None:
                    await self._run_launches(
                        state,
                        activation,
                        lambda: self._launch_payload(state, body, prompt=prompt, system=system),
                        prompt=prompt,
                        media=media,
                        timeout=timeout,
                        deadline=deadline,
                    )
                else:
                    await state.execute(
                        self._launch_payload(state, body, prompt=prompt, system=system),
                        media=media,
                        timeout=timeout,
                        close_timeout=self.config.session_close_timeout_seconds,
                        deadline=deadline,
                    )
        except TimeoutError as exc:
            if state.is_continuation and not state.session.launch_started:
                raise _ActivationDeadlineExpired from None
            failure = exc
        except BaseException as exc:
            failure = exc
        return await self._finish_session_response(
            state, body, timeout=timeout, started=started, previous_session_id=previous_session_id, failure=failure
        )

    async def _activation_task(
        self,
        state: OpenCodeSandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        prompt: str,
        system: str,
        media: Sequence[MediaFile] = (),
        index: int | None,
    ) -> NeMoGymResponse:
        """Run the session's activation; with checkpointing, as the participant's activation of the session.

        Entered inside the task every request of the activation awaits, so a close or a retire that cancels the
        participant's activation stops the launch too.
        """
        participant = self.checkpoint_participant
        if participant is None:
            return await self._session_response(state, body, prompt=prompt, system=system, media=media)
        async with participant.activation(
            state.request.agent_session_id, state.request.episode_id, index=index
        ) as activation:
            response = await self._session_response(
                state, body, prompt=prompt, system=system, media=media, activation=activation
            )
            # Recorded before the activation ends, so a checkpoint exports the session with its reply.
            state.completed_activation = index if index is not None else state.completed_activation + 1
            state.completed_reply = response.model_dump(mode="json")
            return response

    async def _run_launches(
        self,
        state: OpenCodeSandboxSession,
        activation: Activation,
        payload: Callable[[], dict[str, JsonValue]],
        *,
        prompt: str,
        media: Sequence[MediaFile] = (),
        timeout: float,
        deadline: float | None,
    ) -> None:
        """Launch OpenCode until a launch finishes, pausing it in place at every checkpoint.

        A launch whose pause or continue failed was stopped; the next launch resumes the saved conversation, and so
        does the first launch of a restored session that had started one. Each launch has what is left of the
        activation's budget, not counting time paused for checkpoints. A launch that resumes the conversation attaches
        no media: the saved user message already holds the prompt's files. A restored session whose launch survived
        the restore (``adoption``) takes that launch over first, and relaunches only if the take-over fails.
        """
        close_timeout = self.config.session_close_timeout_seconds
        clock = _PauseClock(activation)
        while True:
            adoption, state.adoption = state.adoption, None
            if adoption is not None:
                launch = state.launch_count
                try:
                    outcome = await self._adopt_restored_launch(
                        state, adoption, clock, timeout=timeout, close_timeout=close_timeout
                    )
                except Exception as error:
                    LOG.warning(
                        "OpenCode cannot resume in place: session %s, reason adopt_failed, detail %r; relaunching",
                        state.request.agent_session_id,
                        error,
                    )
                    if not state.require_session().cleanup_confirmed:
                        await self._abandon_restored_launch(state, adoption.identity, reason="adopt_failed")
                    continue
                resumed = "in_place"
            else:
                resumed = state.launch_count > 0
                if resumed:
                    await state.prepare_launch(state.launch_count + 1, timeout=close_timeout)
                launch_payload = payload()
                if resumed:
                    launch_payload = await self._relaunch_payload(state, launch_payload, prompt=prompt)
                state.launch_count += 1
                launch = state.launch_count
                launch_deadline = None if deadline is None else deadline + clock.paused_s
                outcome = await run_until_parked(
                    state.require_session(),
                    clock,
                    launch=lambda: state.execute(
                        launch_payload,
                        media=() if launch_payload.get("resume") is not None else media,
                        timeout=timeout,
                        close_timeout=close_timeout,
                        deadline=launch_deadline,
                        pause_allowance_s=self.config.checkpoint_pause_allowance_seconds,
                    ),
                    on_parked=lambda: {"launch": launch, "harness_session_id": state.harness_session_id},
                    stop_timeout=close_timeout,
                )
            receipt = state.session.cleanup or {}
            state.launch_records.append(
                {
                    "launch": launch,
                    "resumed": resumed,
                    "outcome": "finished" if isinstance(outcome, Finished) else "stopped",
                    "return_code": receipt.get("return_code"),
                    "paused_s": round(clock.paused_s, 3),
                }
            )
            if isinstance(outcome, Finished):
                return
            assert isinstance(outcome, Stopped)
            LOG.warning(
                "OpenCode launch %d of session %s stopped at a checkpoint (%s); relaunching it",
                launch,
                state.request.agent_session_id,
                outcome.error or "pause not confirmed",
            )

    async def _adopt_restored_launch(
        self,
        state: OpenCodeSandboxSession,
        adoption: RestoredLaunch,
        clock: _PauseClock,
        *,
        timeout: float,
        close_timeout: float,
    ) -> Finished[str] | Stopped:
        """Point the restored launch's relay at this attempt's model URL, then continue the launch and finish it.

        The relay is re-pointed before the continue: it holds the launch's model calls until then, and re-sends the
        one the checkpoint cut, byte for byte, under the replacement attempt.
        """
        launch = state.launch_count
        upstream = self.resolve_model_base_url(self.config.model_server.name, state.request.episode_id.capture_key)
        await state.point_relay(upstream)
        LOG.warning(
            "OpenCode resumed in place: session %s, launch %d, pause %d, relay generation %d",
            state.request.agent_session_id,
            launch,
            adoption.pause_seq,
            state.relay_gen,
        )
        return await run_until_parked(
            state.require_session(),
            clock,
            launch=lambda: state.adopt(
                pause_seq=adoption.pause_seq,
                timeout=timeout,
                close_timeout=close_timeout,
                pause_allowance_s=self.config.checkpoint_pause_allowance_seconds,
            ),
            on_parked=lambda: {"launch": launch, "harness_session_id": state.harness_session_id},
            stop_timeout=close_timeout,
        )

    async def _relaunch_payload(
        self, state: OpenCodeSandboxSession, payload: dict[str, JsonValue], *, prompt: str
    ) -> dict[str, JsonValue]:
        """Resume the saved conversation without a new user turn when it is usable; otherwise start afresh."""
        timeout = self.config.session_close_timeout_seconds
        decision = resume_decision(await state.inspect_conversation(timeout=timeout), prompt)
        if decision.resume:
            LOG.info(
                "OpenCode launch %d of session %s resumes conversation %s",
                state.launch_count + 1,
                state.request.agent_session_id,
                decision.session_id,
            )
            state.harness_session_id = decision.session_id
            command = [
                "bash",
                f"{state.persistent_directory}/{RESUME_SCRIPT.name}",
                f"{state.runtime}/opencode",
                decision.session_id,
                f"{state.session.session_dir}/resume.json",
            ]
            return {**payload, "command": command, "resume": decision.payload}
        LOG.warning(
            "OpenCode launch %d of session %s starts the task afresh: %s",
            state.launch_count + 1,
            state.request.agent_session_id,
            decision.reason,
        )
        await state.discard_conversation(timeout=timeout)
        state.harness_session_id = None
        state.message_ids = []
        return payload

    def _launch_payload(
        self,
        state: OpenCodeSandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        prompt: str,
        system: str,
    ) -> dict[str, JsonValue]:
        """The runner's input for the next launch of OpenCode, in the session's current launch directory."""
        base_url = self.resolve_model_base_url(self.config.model_server.name, state.request.episode_id.capture_key)
        config = {
            "model": "nemo_gym/" + self.config.sandbox_model_id,
            "small_model": "nemo_gym/" + self.config.sandbox_model_id,
            "enabled_providers": ["nemo_gym"],
            "autoupdate": False,
            "share": "disabled",
            # A session's opencode_config "attachment" replaces this, so a run on an endpoint with image limits can
            # restore one.
            **({"attachment": copy.deepcopy(UNRESTRICTED_IMAGE_ATTACHMENT)} if state.declares_image_input else {}),
            "provider": {
                "nemo_gym": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": base_url,
                        "apiKey": "dummy_key",
                        "timeout": False,
                    },  # pragma: allowlist secret
                    "models": {
                        self.config.sandbox_model_id: {
                            **(
                                {"reasoning": self.config.thinking}
                                if self.config.sandbox_model_id != "dummy_model"
                                else {}
                            ),
                            "interleaved": {"field": "reasoning_content"},
                            "limit": {
                                "context": self.config.context_window,
                                "input": self.config.context_window,
                                "output": self.config.max_output_tokens,
                            },
                            **(copy.deepcopy(IMAGE_INPUT_MODEL_OPTIONS) if state.declares_image_input else {}),
                        }
                    },
                }
            },
            # A copy: the replacement prompt and the request sampling below are per session and must not reach
            # the shared configuration.
            **copy.deepcopy(_without_empty_tools(self.config.opencode_config)),
        }
        sampling = {name: value for name in ("temperature", "top_p") if (value := getattr(body, name)) is not None}
        if sampling:
            # Merged into each agent, so a configured agent.build.steps stays.
            agents = config.setdefault("agent", {})
            for agent in _OPENCODE_AGENTS[self.config.opencode_version]:
                agents.setdefault(agent, {}).update(sampling)
            if "temperature" in sampling:
                # OpenCode sends temperature only for models that declare the capability.
                config["provider"]["nemo_gym"]["models"][self.config.sandbox_model_id]["temperature"] = True
        mcp_servers, mcp_grants = self._granted_mcp_servers(state.request)
        if mcp_servers:
            config["mcp"] = mcp_servers
        # System instructions are separate from the user's task and applied to every OpenCode model turn.
        if system:
            if self.config.system_prompt_mode == "replace":
                # Added to the build agent, beside the request sampling and a configured agent.build.steps.
                config.setdefault("agent", {}).setdefault("build", {})["prompt"] = system
            else:
                config["instructions"] = [f"{state.session.session_dir}/instructions.md"]
        if self._model_call_capture_enabled():
            apply_observability_patch(
                config, plugin_path=Path(state.persistent_directory) / ASSISTANT_MESSAGE_PLUGIN_FILE
            )
        config.setdefault("plugin", []).append(
            (Path(state.session.session_dir) / TASK_ENVIRONMENT_PLUGIN_FILE).as_uri()
        )
        if mcp_servers:
            config["plugin"].append((Path(state.session.session_dir) / REQUIRED_MCP_PLUGIN_FILE).as_uri())
        # Owner-staged plugins load after Gym's; a plugin that matters must fail closed through its own receipt.
        config["plugin"].extend(PurePosixPath(path).as_uri() for path in self.config.plugins)
        command = [f"{state.runtime}/opencode", "run", "--format", "json", "--title", "NeMo Gym"]
        if self.config.system_prompt_mode == "replace":
            command.extend(["--agent", "build"])
        if self.config.thinking:
            command.append("--thinking")
        if state.harness_session_id:
            command.extend(["--session", state.harness_session_id])
        activation_id = {}
        if state.is_continuation:
            # A relaunch within one activation tags its calls with its launch; they join the activation's invocation.
            launch = f":launch-{state.launch_count + 1}" if state.launch_count else ""
            activation_id = {"NEMO_GYM_OPENCODE_ACTIVATION_ID": f"{state.invocation_prefix}{launch}"}

        def serialize(launch: dict[str, Any]) -> str:
            replace = system and self.config.system_prompt_mode == "replace"
            return _serialize_replacement_config(launch) if replace else json.dumps(launch)

        relay: dict[str, JsonValue] | None = None
        launch_config = config
        if state.keeps_processes:
            # OpenCode calls the model through the relay, which a restore re-points; without a relay in the sandbox,
            # the runner's fallback environment points it at the model server directly.
            relay_url = f"http://127.0.0.1:{self.config.sandbox_model_relay_port}/v1"
            provider = config["provider"]["nemo_gym"]
            launch_config = {
                **config,
                "provider": {"nemo_gym": {**provider, "options": {**provider["options"], "baseURL": relay_url}}},
            }
            relay = {
                "port": self.config.sandbox_model_relay_port,
                "upstream": base_url,
                "fallback_env": [f"OPENCODE_CONFIG_CONTENT={serialize(config)}"],
            }
        payload = {
            "instructions": system if self.config.system_prompt_mode == "append" else "",
            "path_prefix": self.config.sandbox_path_prefix,
            "cwd": state.session.workdir,
            "command": command,
            "prompt": prompt,
            "mcp_grants": mcp_grants if mcp_servers else None,
            **({"relay": relay} if relay is not None else {}),
            "env": {
                **_private_directories(state.persistent_directory),
                **activation_id,
                "OPENCODE_CONFIG_CONTENT": serialize(launch_config),
                **_OPENCODE_ISOLATION,
                "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": str(self.config.max_output_tokens),
            },
        }
        if self.config.session_project_config:
            del payload["env"]["OPENCODE_DISABLE_PROJECT_CONFIG"]
        return payload

    async def _finish_session_response(
        self,
        state: OpenCodeSandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        timeout: float,
        started: float,
        previous_session_id: str | None,
        failure: BaseException | None,
    ) -> NeMoGymResponse:
        """Build the reply from the last launch's artifacts, or raise what the activation failed with."""
        error = None
        export = {}
        observations = state.activation_observations if state.is_continuation else state.observations
        if observations is None:
            confirmed_oom = isinstance(failure, NgFailureEvent) and state.session.execution_oom is failure
            invocation_id = state.invocation_prefix if state.is_continuation else state.request.episode_id.capture_key
            if state.is_continuation and state.harness_session_id is not None:
                invocation_id += f":{state.harness_session_id}"
            observations = AgentObservationBundle(
                schema_version="2.0",
                source="opencode",
                records=(
                    [
                        AgentInvocation(
                            invocation_id=invocation_id,
                            activation_id=state.activation_id if state.is_continuation else None,
                            harness_session_id=state.harness_session_id if state.is_continuation else None,
                        )
                    ]
                    if confirmed_oom
                    else []
                ),
                gaps=[ObservationGap(code="observation_capture_failed")],
            )
        if state.is_continuation:
            state.activation_observations = observations
        else:
            state.observations = observations
        observations.gaps.extend(state.take_export_gaps())
        mcp_unconfirmed = None
        mcp_servers, mcp_grants = self._granted_mcp_servers(state.request)
        if mcp_servers and not isinstance(failure, asyncio.CancelledError):
            error, mcp_unconfirmed = await self._check_mcp_grants(state, mcp_grants, observations)
        if state.session.artifacts is not None:
            try:
                parsed = json.loads(state.session.artifacts)
                if not isinstance(parsed, dict) or not isinstance(parsed.get("messages", []), list):
                    raise ValueError("expected a session object with a messages list")
                if any(
                    not isinstance(message, dict) or not isinstance(message.get("info"), dict)
                    for message in parsed.get("messages", [])
                ):
                    raise ValueError("expected message info objects")
                export = parsed
            except (ValueError, TypeError) as exc:
                error = error or f"OpenCode output parse failed: {exc}"
        output = []
        usage = None
        if export.get("messages"):
            try:
                output = self._session_output(export, observations)
                usage = self._session_usage(export)
                if (
                    usage is None
                    or usage.input_tokens_details.cached_tokens is None
                    or usage.output_tokens_details.reasoning_tokens is None
                ):
                    observations.gaps.append(
                        ObservationGap(
                            code="token_usage_detail_unavailable",
                            detail="Persisted optional counters are absent, invalid, or runtime-defaulted zeros",
                        )
                    )
            except Exception as exc:
                error = error or f"OpenCode output parse failed: {exc}"
        result = state.session.cleanup
        assistants = [m["info"] for m in export.get("messages", []) if m["info"].get("role") == "assistant"]
        native_result = assistants[-1] if assistants else {}
        reported_error = error is None and failure is None and self._reported_model_error(native_result, result)
        context_limit = reported_error and (native_result.get("error") or {}).get("name") == _CONTEXT_OVERFLOW_ERROR
        runtime = state.runtime_info
        if runtime is None:
            observations.gaps.append(ObservationGap(code="runtime_info_unavailable"))
        if result is not None and result["return_code"] is None:
            observations.gaps.append(ObservationGap(code="worker_exit_code_unavailable"))
        outcome = classify_opencode_run(
            export,
            cleanup=result,
            previous_session_id=previous_session_id,
            session_id=state.harness_session_id,
            stdout=state.activation_log,
            continuation=state.is_continuation,
        )
        error = error or outcome.error
        # Without a recorded status, work the model did cannot count: the required tools were never confirmed.
        # A run that failed for another reason, or made no model call and ended early (timeout), keeps its cause.
        # OpenCode saves the assistant message before it connects MCP; parts or a finish show a model step ran.
        model_called = any(
            message["info"].get("role") == "assistant" and (message.get("parts") or message["info"].get("finish"))
            for message in export.get("messages", [])
        )
        if (
            mcp_unconfirmed
            and not (error or failure or reported_error)
            and (model_called or outcome.status == "completed")
        ):
            error = mcp_unconfirmed
        if context_limit:
            outcome = OpenCodeRunOutcome(status="incomplete")
        status = outcome.status
        if (error or failure) and not context_limit:
            status = "failed"
        # Artifact message completion records a model turn, not the entire invocation.
        for record in observations.records:
            if isinstance(record, AgentInvocation) and record.parent_invocation_id is None:
                record.status = "incomplete" if isinstance(failure, asyncio.CancelledError) else status
                if state.is_continuation:
                    record.duration_ms = (monotonic() - started) * 1000
                    record.stop_reason = "context_limit" if context_limit else outcome.error_type
                record.error_type = outcome.error_type
                if isinstance(failure, asyncio.CancelledError):
                    record.error_type = "cancelled"
                elif context_limit:
                    record.error_type = "context_limit"
                elif reported_error:
                    record.error_type = "provider_error"
                elif error or failure:
                    record.error_type = "server_error"
        cancelled = isinstance(failure, asyncio.CancelledError)
        timed_out = result is not None and result["timed_out"]
        # The enclosing process deadline does not establish a tool's own terminal outcome.
        # Persisted starts without native completion remain incomplete through normalization.
        if result is not None:
            sandbox_outcome = (
                "timeout" if timed_out else "cancelled" if cancelled else "failed" if error or failure else "completed"
            )
            observations.records.append(
                SandboxObservation(
                    role="agent",
                    stage="agent",
                    provider=(
                        self.config.sandbox_provider
                        if state.session.owns_sandbox
                        else state.request.sandbox_access.connection.provider_config_ref
                    ),
                    sandbox_id=state.sandbox_id,
                    outcome=sandbox_outcome,
                    exit_code=result["return_code"] if not timed_out else None,
                    # The same category the root invocation carries for the stop or failure.
                    error_type="process_exit"
                    if reported_error
                    else {"cancelled": "cancelled", "failed": "server_error"}.get(sandbox_outcome),
                )
            )
        execution_oom = (
            failure
            if self.config.report_agent_sandbox_oom
            and isinstance(failure, NgFailureEvent)
            and state.session.execution_oom is failure
            else None
        )
        if execution_oom is not None:
            observe_failure(execution_oom)
            details = execution_oom.details
            observation = SandboxObservation(
                role="agent",
                provider=details.provider if isinstance(details, SandboxFailureDetails) else None,
                sandbox_id=details.sandbox_id if isinstance(details, SandboxFailureDetails) else None,
                outcome="oom",
                stage=execution_oom.stage or "agent",
                failure_id=execution_oom.event_id,
            )
            observations.records.append(observation)
        observations.gaps.append(
            ObservationGap(
                code="model_usage_reconciliation_unavailable",
                detail="Usage comes from OpenCode persisted assistant messages; compare against captured Gym model calls.",
            )
        )
        if execution_oom is not None:
            response = self._make_session_response(
                state,
                body,
                timeout=timeout,
                outcome=OpenCodeRunOutcome(status="failed", error_type="server_error"),
            )
            response.metadata[NG_AGENT_SANDBOX_OBSERVATION_KEY] = observation.model_dump_json(exclude_none=True)
            response.metadata[AGENT_OUTCOME_METADATA_KEY] = AgentOutcome(
                failure_kind=execution_oom.failure_kind
            ).model_dump_json()
            return response
        if failure is not None:
            raise failure
        if error and not reported_error:
            raise HTTPException(502, format_error_message(error, stderr=state.stderr))
        reported_message = None
        if reported_error and not context_limit:
            error_data = native_result["error"].get("data") or {}
            reported_message = error_data.get("message")
            if not isinstance(reported_message, str) or not reported_message:
                reported_message = "OpenCode provider request failed"
            upstream_status = error_data.get("statusCode")
            if type(upstream_status) is int and 400 <= upstream_status <= 599:
                reported_message = f"HTTP {upstream_status}: {reported_message}"
            reported_message = format_error_message(reported_message)
        response = self._make_session_response(
            state,
            body,
            timeout=timeout,
            outcome=outcome,
            output=output,
            usage=usage,
            error_message=reported_message,
        )
        if reached_step_limit(export, _build_step_limit(self.config.opencode_config)):
            # The environment decides whether an answer given at or after the step limit is scored.
            response = response.model_copy(
                update={"metadata": {**(response.metadata or {}), HARNESS_STEP_LIMIT_METADATA_KEY: "true"}}
            )
        return self.report_outcome(response, native_result)


if __name__ == "__main__":
    OpenCodeAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeAgent.run_webserver()  # noqa: F401
