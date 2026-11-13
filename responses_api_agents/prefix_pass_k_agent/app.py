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

"""Prefix pass@K agent: replay a captured trajectory, then hand over at turn T.

The rollout replays the captured turns 1..T-1 into a fresh task container — the
assistant text verbatim, each action re-executed so the repo reaches the state
the trajectory was in — and from turn T the candidate model drives the agent to
finish. The resources server scores whatever patch results, so K rollouts of one
row give mean reward (n_pass/K) and true pass@K (n_pass >= 1) from Gym's own
`num_repeats` aggregation.

This reimplements mini-swe-agent's loop rather than driving the real CLI
through a replay proxy, in either of its two wires: backticks (one bash command
per turn inside a ```mswea_bash_command fence, an `<returncode>/<output>`
observation back) or function calling (`bash` tool calls, a JSON observation
per call), with a sentinel echo to submit. The loop is small; the part that
must not drift is the *wire* — see `ACTION_FENCE`, `render_observation` and
`render_tool_observation`, which mirror the captured harness exactly.
"""

import asyncio
import json
import re
import sys
from shlex import quote
from time import monotonic, time
from traceback import format_exc
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from aiohttp import ClientConnectionError, ClientResponseError
from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.failure_kinds import AGENT_RUN_ERROR, TRANSPORT_PEER_DROP
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)


# The action envelope of mini-swe-agent's backticks mode. The dataset builder
# uses the identical pattern; if these two ever diverge the replay silently
# stops matching the trajectory it is replaying.
ACTION_FENCE = re.compile(r"```mswea_bash_command\s*\n(.*?)\n?```", re.DOTALL)

# Echoing this is how the harness's prompt tells the model to submit.
SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# A transcript too long for the model's context. vLLM has worded this as "This
# model's maximum context length is N tokens" and as "the model's context length
# is only N tokens"; the model server passes the body through. A request whose
# max_tokens alone exceeds the window is a config error and must still raise --
# vLLM words that one without "context length".
CONTEXT_OVERFLOW = re.compile(r"context length", re.IGNORECASE)

# `offline` points the sandbox's resolver at a port nothing listens on, so every
# lookup fails at once, as it does in a container with no network.
RESOLV_CONF = "/etc/resolv.conf"
OFFLINE_RESOLVER = "nameserver 127.0.0.1\n"

# The captured harness elides observations longer than this, keeping a head and
# a tail. Reproduced exactly: an observation that differs in length from the
# captured one changes what the model reads.
OBSERVATION_LIMIT = 10000
OBSERVATION_HEAD = 5000
OBSERVATION_TAIL = 5000


def parse_action(content: str) -> Optional[str]:
    """The single bash command in an assistant turn, or None."""
    matches = ACTION_FENCE.findall(content or "")
    if len(matches) != 1:
        return None
    return matches[0].strip() or None


def is_submission(returncode: Any, output: str) -> bool:
    """mini-swe-agent's own test: the sentinel alone on the first line of a clean
    exit. A substring match would also end the rollout on, say, a `cat` of a file
    that happens to mention the sentinel."""
    lines = (output or "").lstrip().splitlines()
    return bool(lines) and lines[0].strip() == SUBMIT_SENTINEL and returncode == 0


def render_observation(returncode: Any, output: str, exception_info: Optional[str] = None) -> str:
    """Render a command result the way the captured harness did."""
    output = output or ""
    exception = f"<exception>{exception_info}</exception>\n" if exception_info else ""
    if len(output) < OBSERVATION_LIMIT:
        return f"{exception}<returncode>{returncode}</returncode>\n<output>\n{output}</output>"
    elided = len(output) - OBSERVATION_LIMIT
    return (
        f"{exception}<returncode>{returncode}</returncode>\n"
        "<warning>\n"
        "The output of your last command was too long.\n"
        "Please try a different command that produces less output.\n"
        "If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\n"
        "If you're using grep or find and it produced too much output, you can use a more selective search pattern.\n"
        "If you really need to see something from the full command's output, "
        "you can redirect output to a file and then search in that file.\n"
        "</warning>\n"
        f"<output_head>\n{output[:OBSERVATION_HEAD]}\n</output_head>\n"
        f"<elided_chars>\n{elided} characters elided\n</elided_chars>\n"
        f"<output_tail>\n{output[-OBSERVATION_TAIL:]}\n</output_tail>"
    )


def format_error_observation(n_actions: int) -> str:
    """The harness's reply to a turn that did not carry exactly one action."""
    return (
        "Format error:\n\n"
        "<error>\n"
        f"Expected exactly 1 action, found {n_actions}.\n"
        "</error>\n\n"
        "Here is general guidance on how to format your response:\n\n"
        f"Please always provide EXACTLY ONE action in triple backticks, found {n_actions} actions.\n\n"
        "Please format your action in triple backticks as shown in <response_example>.\n\n"
        "<response_example>\n"
        "Here are some thoughts about why you want to perform the action.\n\n"
        "```mswea_bash_command\n"
        "<action>\n"
        "```\n"
        "</response_example>\n\n"
        "If you have completed your assignment, please consult the first message about how to\n"
        "submit your solution (you will not be able to continue working on this task after that)."
    )


def html_safe_json(value: Any) -> str:
    """Jinja's built-in `tojson`, which the function-calling harness's
    observation template goes through: ASCII-only json.dumps with <, >, & and '
    escaped to \\uXXXX. Measured on the captured DeepSWE traces: 617,715 escaped
    apostrophes against 4 literal ones, and those 4 are harness-authored text,
    not program output. Plain json.dumps would put characters into a replayed
    prefix that the trajectory never contained.
    """
    return (
        json.dumps(value, sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("'", "\\u0027")
    )


def render_tool_observation(returncode: Any, output: str, exception_info: Optional[str] = None) -> str:
    """A command result as the function-calling harness renders it: JSON, not XML.

    `exception_info` (a timeout) is appended to the last field's line, not given
    its own -- the template's whitespace control puts it there.
    """
    output = output or ""
    exception = f', "exception_info": {html_safe_json(exception_info)}' if exception_info else ""
    if len(output) < OBSERVATION_LIMIT:
        return f'{{\n  "returncode": {returncode},\n  "output": {html_safe_json(output)}{exception}\n}}'
    return (
        "{\n"
        f'  "returncode": {returncode},\n'
        f'  "output_head": {html_safe_json(output[:OBSERVATION_HEAD])},\n'
        f'  "output_tail": {html_safe_json(output[-OBSERVATION_TAIL:])},\n'
        f'  "elided_chars": {len(output) - OBSERVATION_LIMIT},\n'
        f'  "warning": "Output too long."{exception}\n'
        "}"
    )


def tool_call_error_observation() -> str:
    """The function-calling harness's reply to a captured turn with no tool call.

    Only replay uses it. A LIVE generation with no usable call is never answered:
    the reference bridge returns a 502 and mini retries the identical request, so
    the failure costs its forward and leaves the transcript untouched.
    """
    return (
        "Tool call error:\n\n"
        "<error>\n"
        "No tool calls found in the response. Every response MUST include at least one tool call.\n"
        "</error>\n\n"
        "Here is general guidance on how to submit correct toolcalls:\n\n"
        "Every response needs to use the 'bash' tool at least once to execute commands.\n\n"
        "Call the bash tool with your command as the argument:\n"
        "- Tool: bash\n"
        '- Arguments: {"command": "your_command_here"}\n\n'
        "If you want to end the task, please issue the following command: "
        "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\n"
        "without any other command."
    )


def git_state(output: str, base_commit: Optional[str]) -> Dict[str, Any]:
    """Parse `git rev-parse HEAD && git status --porcelain`."""
    lines = output.splitlines()
    head = lines[0].strip() if lines else ""
    entries = [line for line in lines[1:] if len(line) >= 3]
    return {
        "head_moved": bool(base_commit) and bool(head) and head != base_commit,
        "staged_files": sum(1 for line in entries if line[0] not in " ?"),
        "untracked_files": sum(1 for line in entries if line.startswith("??")),
    }


def chat_content(content: Any) -> str:
    """Message content as the plain string chat completions takes. Content-part
    lists are joined with newlines, as the reference bridge normalised them."""
    if isinstance(content, list):
        parts = [part.get("text") or "" if isinstance(part, dict) else str(part) for part in content]
        return "\n".join(part for part in parts if part)
    return content or ""


def tool_call_commands(tool_calls: Optional[List[Dict[str, Any]]]) -> List[tuple]:
    """(call_id, command) for every `bash` call, in order.

    The harness executes EVERY call in a turn and answers each with its own tool
    message; replaying only the first silently skips commands (39 of the 120
    broad120 trials have a multi-call prefix turn). Blank placeholder entries
    some gateways leave in the array are skipped.
    """
    pairs = []
    for call in tool_calls or []:
        function = call.get("function") or {}
        if (function.get("name") or "").strip() != "bash":
            continue
        try:
            arguments = json.loads(function.get("arguments") or "")
        except (json.JSONDecodeError, TypeError):
            continue
        command = arguments.get("command") if isinstance(arguments, dict) else None
        if isinstance(command, str) and command.strip():
            pairs.append((call.get("id") or "", command))
    return pairs


class PrefixPassKAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    # Must match the wire the captured traces were shaped on (see the base-model
    # server). backticks: a fenced command in the assistant text, XML-ish
    # observations back as user turns. function_calling: a `bash` tool call, JSON
    # observations back as tool turns, spoken over chat completions because that
    # is the traces' native message shape.
    wire: Literal["backticks", "function_calling"] = "backticks"

    # Only the provider is needed: the container is created by the resources
    # server and this agent attaches to the handle it returns.
    sandbox_provider: str

    # Exported before every replayed command. SWE-bench images run Python under
    # conda, and `Path.read_text()` uses the locale's preferred encoding: with no
    # UTF-8 locale it falls back to ASCII and dies on the first non-ASCII byte
    # ("UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2"). The captured
    # trajectories ran under Docker where the image's own LANG applied, but the
    # resources server sets an explicit env on the sandbox spec, which drops it.
    # Without this, an edit command fails and the rollout scores a silent 0.
    exec_env: Dict[str, str] = Field(default_factory=lambda: {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})

    # How each command is run. None: directly in the sandbox's shell. A list such
    # as ["bash", "-c"]: exec'd as `<interpreter> <command>`, the way
    # mini-swe-agent's DockerEnvironment runs it. That matters for BASH_ENV: a
    # fresh interpreter sources it at startup, which an in-shell export cannot.
    interpreter: Optional[List[str]] = None

    # Turns the candidate may take after the handover. The reference arms use 3:
    # the decisive turn plus a little room to recover from a format error.
    max_forwards: int = 3
    step_timeout: int = 60

    # Gold check: replay 1..T INCLUSIVE and take no model turns, which must
    # reproduce the trajectory's resolving patch. Run this before trusting any
    # score — a pass@K of 0 is more often a broken replay than a weak model.
    replay_through_target: bool = False

    # Commit the working tree before verification. DeepSWE v1.1 grading reads
    # only COMMITTED work (`git diff <base> HEAD`), but the decisive turns were
    # located by scoring the working tree (`git add -A && git diff --cached
    # <base>`), and only 4 of the 120 broad120 trials had committed before T.
    # Without this the gold check fails on nearly every row and an arm scores ~0.
    commit_worktree: bool = False

    # The opposite collector: the SWE-bench server grades `git diff` -- unstaged
    # edits to tracked files -- so a rollout that commits, stages or creates its
    # fix is graded as if it never made it. Before grading, fold everything since
    # the starting commit back into unstaged edits (new files as intent-to-add),
    # the same set `git add -A && git diff --cached <base>` located the decisive
    # turns with. Without it, arms that commit or stage often score low.
    uncommit_worktree: bool = False

    # Record the repository's git state before grading: whether HEAD moved off
    # where the sandbox started, and how many files are staged or untracked. The
    # SWE-bench server grades `git diff` -- unstaged edits to tracked files --
    # while the decisive turns were located with `git add -A && git diff
    # --cached <base>`, which also counts commits, staged and new files. These
    # counts show whether that difference costs a rollout its patch.
    record_git_state: bool = False

    # Take the sandbox offline before the replay, for traces captured with no
    # network (SWE-bench Pro's ran under `docker run --network none`). Their
    # prefixes try to fetch the upstream fix -- `pip download`, `git clone` --
    # which must fail as it did, not leave the solution in the sandbox. The
    # resources server owns this sandbox and grades in a fresh one that needs
    # the network, so no sandbox-wide policy fits; instead hostnames stop
    # resolving ("Could not resolve host", the capture's own error), while
    # localhost and the exec channel keep working.
    offline: bool = False

    debug: bool = False


class PrefixPassKRunRequest(BaseRunRequest):
    # Rows carry the SWE-bench schema plus a `prefix_pass_k` block.
    model_config = ConfigDict(extra="allow")


class PrefixPassKVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class PrefixPassKVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SandboxExecError(Exception):
    """The sandbox failed to run a command at all (e.g. a connection reset).

    Not program output, and not retryable -- the command may already have run --
    so the attempt can no longer replay the trajectory faithfully: it is aborted
    and its sample masked rather than scored.
    """

    def __init__(self, failure_kind: str, reason: str) -> None:
        super().__init__(reason)
        self.failure_kind = failure_kind


class PrefixPassKAgent(SimpleResponsesAPIAgent):
    config: PrefixPassKAgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: Dict[str, AsyncSandbox] = {}
        self._session_id_to_stats: Dict[str, Dict[str, Any]] = {}

    async def _connect_sandbox(self, seed_session_result: Dict[str, Any]) -> AsyncSandbox:
        provider = create_provider(
            resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
        )
        # Prefer the full descriptor: it carries the workdir commands must run in
        # (the harness ran every DeepSWE command in /app). A bare id falls back
        # to the image's WORKDIR, which happens to agree on the images probed but
        # is not guaranteed to.
        descriptor = seed_session_result.get("sandbox_descriptor") or {
            "sandbox_id": seed_session_result["sandbox_handle"]
        }
        return await AsyncSandbox.connect(descriptor, provider=provider)

    def _wrap(self, command: str) -> str:
        """The command as the sandbox shell must run it to behave like the harness.

        Env vars are exported rather than passed as the sandbox's `env`, which
        would replace the container's own environment (PATH, conda) rather than
        add to it. stderr is merged into stdout in order, as the harness's
        subprocess did (stderr=STDOUT), so the observation never depends on how
        a provider splits the two streams.
        """
        exports = "".join(f"export {name}={quote(value)}; " for name, value in (self.config.exec_env or {}).items())
        if self.config.interpreter:
            interpreter = " ".join(quote(part) for part in self.config.interpreter)
            return f"{exports}exec {interpreter} {quote(command)} 2>&1"
        return f"{exports}exec 2>&1; {command}"

    def _render(self, returncode: Any, output: str, exception_info: Optional[str] = None) -> str:
        if self.config.wire == "function_calling":
            return render_tool_observation(returncode, output, exception_info)
        return render_observation(returncode, output, exception_info)

    async def _run(self, sandbox: AsyncSandbox, command: str) -> tuple[Any, str, Optional[str]]:
        """Run one action; return (returncode, output, exception_info) the way the
        harness's environment reports them."""
        # The harness reports a timeout as Python's subprocess.TimeoutExpired,
        # quoting the RAW command -- not the wrapped one actually executed.
        timeout_info = (
            "An error occurred while executing the command: "
            f"Command '{command}' timed out after {self.config.step_timeout} seconds"
        )
        started = monotonic()
        try:
            result = await sandbox.exec(self._wrap(command), timeout_s=self.config.step_timeout)
        except TimeoutError:
            return -1, "", timeout_info
        except Exception as exc:
            # An infrastructure failure, not program output: answering it as a
            # failed command would let the attempt go on and be scored.
            print("prefix_pass_k: exec failed", format_exc(), file=sys.stderr)
            kind = (
                TRANSPORT_PEER_DROP if isinstance(exc, (ConnectionError, ClientConnectionError)) else AGENT_RUN_ERROR
            )
            raise SandboxExecError(kind, f"sandbox exec failed: {type(exc).__name__}: {exc}"[:500]) from exc
        # Providers may return an infrastructure failure instead of raising it.
        if result.error_type and result.error_type != "timeout":
            raise SandboxExecError(
                AGENT_RUN_ERROR,
                f"sandbox exec failed ({result.error_type}): {result.stderr or 'no details'}"[:500],
            )
        # stdout is the command's merged output. stderr is the SANDBOX's status
        # text -- "exit status 1", "signal: killed" -- never program output: the
        # harness would not show it, and appending it put "exit status 1" under
        # every grep that matched nothing.
        output = result.stdout or ""
        if self.config.debug and result.stderr:
            print(f"prefix_pass_k: sandbox status {result.stderr!r}", file=sys.stderr)
        # Not every provider types a timeout: OpenSandbox kills the command and
        # reports rc -1 with "signal: killed", like any other killed command.
        # Elapsed time is what tells the two apart.
        if result.error_type == "timeout" or (
            result.return_code == -1 and monotonic() - started >= self.config.step_timeout
        ):
            return -1, output, timeout_info
        return result.return_code, output, None

    async def _exec(self, sandbox: AsyncSandbox, command: str) -> tuple[str, bool]:
        """Run one action; return (its observation for this wire, whether it submitted)."""
        returncode, output, exception_info = await self._run(sandbox, command)
        return self._render(returncode, output, exception_info), is_submission(returncode, output)

    async def _replay_step(
        self,
        sandbox: AsyncSandbox,
        messages: List[Dict[str, Any]],
        step: Dict[str, Any],
        stats: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Replay one captured turn; return whether it submitted.

        Used for the prefix and, in the gold check, for the decisive turn, so the
        two can never be replayed differently.
        """
        action = step.get("action")
        if self.config.wire == "function_calling":
            tool_calls = step.get("tool_calls") or []
            commands = tool_call_commands(tool_calls)
            if not commands:
                # No call ran; the harness answered with a format error as a
                # user turn, taken from the trace when it was captured.
                messages.append({"role": "assistant", "content": step.get("content") or ""})
                messages.append({"role": "user", "content": step.get("observation") or tool_call_error_observation()})
                return False
            # The captured calls verbatim, ids and argument strings included:
            # they are rendered back into every later prompt.
            messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
            captured = dict(zip([call.get("id") for call in tool_calls], step.get("captured_observations") or []))
            overrides = step.get("replay_commands") or {}
            for call_id, command in commands:
                if call_id in overrides:
                    # The capture timed out partway through this command, and
                    # finishing it would change the repository: run only what
                    # completed, and show what the harness showed.
                    await self._run(sandbox, overrides[call_id])
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": captured[call_id]})
                    continue
                observation, submitted = await self._exec(sandbox, command)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": observation})
                if stats is not None and call_id in captured:
                    # Fidelity gauge: the replayed environment answering exactly
                    # as the captured one did. Timings and temp paths legitimately
                    # differ; a low rate means the replay is off.
                    stats["prefix_observations_compared"] += 1
                    stats["prefix_observations_identical"] += observation == captured[call_id]
                if submitted:
                    return True
            return False

        messages.append({"role": "assistant", "content": step.get("content") or ""})
        submitted = False
        if action:
            observation, submitted = await self._exec(sandbox, action)
        else:
            # A format-error turn ran nothing, so its observation is harness
            # text captured from the trace rather than program output.
            observation = step.get("observation") or format_error_observation(0)
        messages.append({"role": "user", "content": observation})
        return submitted

    async def _replay_prefix(
        self,
        sandbox: AsyncSandbox,
        messages: List[Dict[str, Any]],
        prefix: List[Dict[str, Any]],
        stats: Dict[str, Any],
    ) -> None:
        """Re-run turns 1..T-1 so the container reaches the trajectory's state."""
        for step in prefix:
            await self._replay_step(sandbox, messages, step, stats)

    async def _head(self, sandbox: AsyncSandbox) -> Optional[str]:
        try:
            result = await sandbox.exec("git rev-parse HEAD", timeout_s=60)
        except Exception:
            return None
        return (result.stdout or "").strip() or None

    async def _git_state(self, sandbox: AsyncSandbox, base_commit: Optional[str]) -> Dict[str, Any]:
        """HEAD movement and staged/untracked file counts, for `record_git_state`."""
        try:
            result = await sandbox.exec("git rev-parse HEAD && git status --porcelain", timeout_s=120)
        except Exception:
            print("prefix_pass_k: git state failed", format_exc(), file=sys.stderr)
            return {}
        return git_state(result.stdout or "", base_commit)

    async def _commit_worktree(self, sandbox: AsyncSandbox) -> None:
        """Commit whatever `git add -A` picks up -- it honours .gitignore, as the
        working-tree extraction the decisive turns were located with did -- so a
        committed-only collector scores the working tree."""
        try:
            result = await sandbox.exec(
                "git add -A && git -c commit.gpgsign=false commit -q --no-verify --allow-empty "
                "-m 'prefix pass@K: working tree'",
                timeout_s=300,
            )
        except Exception:
            print("prefix_pass_k: worktree commit failed", format_exc(), file=sys.stderr)
            return
        if result.return_code != 0:
            print(
                f"prefix_pass_k: worktree commit exited {result.return_code}: {(result.stdout or '')[-500:]}",
                file=sys.stderr,
            )

    async def _worktree_patch(self, sandbox: AsyncSandbox, base: Optional[str]) -> Optional[str]:
        """Everything the rollout changed since `base` -- commits, staged and
        unstaged edits, new files (.gitignore honoured) -- as one binary patch:
        `git add -A && git diff --cached <base>`, taken on a scratch copy of the
        index so the repository the grader reads is untouched."""
        if not base:
            return None
        try:
            result = await sandbox.exec(
                'idx=$(mktemp) && cp "$(git rev-parse --git-dir)/index" "$idx" && '
                f'GIT_INDEX_FILE="$idx" git add -A && GIT_INDEX_FILE="$idx" git diff --cached --binary {quote(base)}; '
                'status=$?; rm -f "$idx"; exit $status',
                timeout_s=300,
            )
        except Exception:
            print("prefix_pass_k: worktree patch failed", format_exc(), file=sys.stderr)
            return None
        if result.return_code != 0:
            print(f"prefix_pass_k: worktree patch exited {result.return_code}: {result.stderr!r}", file=sys.stderr)
            return None
        # Providers report empty output as None; an unchanged tree is an empty patch.
        return result.stdout or ""

    async def _uncommit_worktree(self, sandbox: AsyncSandbox, base: Optional[str]) -> None:
        """Make every change since `base` an unstaged edit a `git diff` collector
        sees: commits folded back into the working tree, the index cleared, and
        untracked files (.gitignore honoured, as `git add -A` does) added as
        intent-to-add so they diff as new files."""
        if not base:
            return
        try:
            result = await sandbox.exec(
                f"git reset -q --soft {quote(base)} && git reset -q && "
                "git ls-files -z --others --exclude-standard | xargs -0 -r git add -N --",
                timeout_s=300,
            )
        except Exception:
            print("prefix_pass_k: worktree uncommit failed", format_exc(), file=sys.stderr)
            return
        if result.return_code != 0:
            print(
                f"prefix_pass_k: worktree uncommit exited {result.return_code}: {(result.stdout or '')[-500:]}",
                file=sys.stderr,
            )

    async def _take_offline(self, sandbox: AsyncSandbox) -> None:
        """Make hostnames unresolvable in the sandbox, for `offline`."""
        result = await sandbox.exec(f"printf %s {quote(OFFLINE_RESOLVER)} > {quote(RESOLV_CONF)}", timeout_s=60)
        if result.return_code != 0:
            # An online replay can hand the candidate the upstream fix; fail the rollout instead.
            raise RuntimeError(
                f"prefix_pass_k: could not take the sandbox offline: {(result.stdout or result.stderr or '')[-500:]}"
            )

    async def _generate_chat(
        self,
        request: Request,
        messages: List[Dict[str, Any]],
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """One model turn over chat completions. Returns (assistant message, finish_reason),
        or (None, "context_overflow") when the transcript no longer fits the model."""
        params: Dict[str, Any] = {"model": body.model or self.config.model_server.name, "messages": messages}
        if body.temperature is not None:
            params["temperature"] = body.temperature
        if body.top_p is not None:
            params["top_p"] = body.top_p
        if body.max_output_tokens:
            params["max_tokens"] = body.max_output_tokens
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_request("/v1/chat/completions", request),
            json=params,
            cookies=request.cookies,
        )
        try:
            await raise_for_status(model_response)
        except ClientResponseError as error:
            content = getattr(error, "response_content", None) or b""
            if CONTEXT_OVERFLOW.search(content.decode(errors="replace")):
                return None, "context_overflow"
            raise
        data = await get_response_json(model_response)
        choice = (data.get("choices") or [{}])[0]
        return choice.get("message") or {}, choice.get("finish_reason")

    async def _generate(
        self,
        request: Request,
        messages: List[Dict[str, Any]],
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> tuple[str, Any]:
        """One model turn. Returns (assistant text, usage)."""
        params = body.model_copy(update={"input": messages})
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_request("/v1/responses", request),
            json=params,
            cookies=request.cookies,
        )
        await raise_for_status(model_response)
        parsed = NeMoGymResponse.model_validate(await get_response_json(model_response))
        text = ""
        for item in parsed.output or []:
            dumped = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            for part in dumped.get("content") or []:
                if isinstance(part, dict) and part.get("text"):
                    text += part["text"]
        return text, parsed.usage

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_key = request.session[SESSION_ID_KEY]
        sandbox = self._session_id_to_sandbox[session_key]
        row: Dict[str, Any] = getattr(request.state, "_ng_prefix_pass_k_row", {}) or {}
        spec = row.get("prefix_pass_k") or {}

        messages: List[Dict[str, Any]] = [dict(item) for item in (body.input or [])]
        if self.config.wire == "function_calling":
            # Spoken over chat completions, whose schema rejects the extra
            # fields Responses input items carry (`type`, `phase`).
            messages = [{"role": item.get("role"), "content": chat_content(item.get("content"))} for item in messages]
        stats: Dict[str, Any] = {
            "target_turn": spec.get("target_turn"),
            "n_prefix_turns": len(spec.get("prefix") or []),
            "forwards": 0,
            # Forwards whose output carried no usable action. The reference
            # measured ~20% on this wire at 1024 tokens, almost all truncations;
            # a rate far above that points at the wire, not the model.
            "unparsed_forwards": 0,
            # Unparsed forwards cut off at max_tokens: the reference found 97% of
            # parse failures were truncations, so this should track the above.
            "truncated_forwards": 0,
            "context_overflow": False,
            "prefix_observations_compared": 0,
            "prefix_observations_identical": 0,
            "submitted": False,
            "replay_through_target": self.config.replay_through_target,
            # Every forward's raw output, unparseable ones included. With the
            # row's prefix, the rollout replays without the model -- e.g. to
            # re-grade it under a different verifier.
            "candidate_turns": [],
            # Commands the sandbox failed to run at all; the first one aborts the attempt.
            "exec_errors": 0,
        }

        try:
            await self._replay_prefix(sandbox, messages, spec.get("prefix") or [], stats)

            if self.config.replay_through_target:
                # Gold check: play the captured decisive turn instead of generating.
                stats["submitted"] = await self._replay_step(sandbox, messages, spec.get("target") or {})
            elif self.config.wire == "function_calling":
                for _ in range(self.config.max_forwards):
                    message, finish_reason = await self._generate_chat(request, messages, body)
                    stats["forwards"] += 1
                    stats["candidate_turns"].append(
                        {"context_overflow": True}
                        if message is None
                        else {
                            "content": message.get("content"),
                            "tool_calls": message.get("tool_calls"),
                            "finish_reason": finish_reason,
                        }
                    )
                    if message is None:
                        # The transcript no longer fits the model's context, and a
                        # retry cannot shrink it: the rollout ends on the prefix's
                        # state. The reference got there by failing every forward
                        # until the budget ran out -- five broad120 trials overflow
                        # at handover, and all five scored 0/32 there.
                        stats["context_overflow"] = True
                        break
                    tool_calls = message.get("tool_calls")
                    commands = tool_call_commands(tool_calls)
                    if not commands:
                        # Resample from the SAME transcript: the reference bridge
                        # answers an unparseable generation with a 502 and mini
                        # retries the identical request, so the failure costs its
                        # forward and the model never sees it.
                        stats["unparsed_forwards"] += 1
                        stats["truncated_forwards"] += finish_reason == "length"
                        continue
                    messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
                    for call_id, command in commands:
                        observation, submitted = await self._exec(sandbox, command)
                        messages.append({"role": "tool", "tool_call_id": call_id, "content": observation})
                        if submitted:
                            # The harness stops the turn at the submission; later
                            # calls in it never run.
                            stats["submitted"] = True
                            break
                    if stats["submitted"]:
                        break
            else:
                for _ in range(self.config.max_forwards):
                    text, _usage = await self._generate(request, messages, body)
                    stats["forwards"] += 1
                    stats["candidate_turns"].append({"content": text})
                    messages.append({"role": "assistant", "content": text})
                    action = parse_action(text)
                    if action is None:
                        stats["unparsed_forwards"] += 1
                        n_actions = len(ACTION_FENCE.findall(text or ""))
                        messages.append({"role": "user", "content": format_error_observation(n_actions)})
                        continue
                    observation, submitted = await self._exec(sandbox, action)
                    messages.append({"role": "user", "content": observation})
                    if submitted:
                        stats["submitted"] = True
                        break
        except SandboxExecError as exc:
            # Stop at the first command the sandbox could not run: nothing after it
            # replays the trajectory faithfully, so `run` masks the sample.
            stats["exec_errors"] = 1
            stats["failure_kind"] = exc.failure_kind
            stats["failure_reason"] = str(exc)

        self._session_id_to_stats[session_key] = stats
        if self.config.debug:
            print(f"prefix_pass_k: {stats}", file=sys.stderr)

        # The transcript is the rollout; the patch itself is read out of the
        # container by the resources server.
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model or self.config.model_server.name,
            object="response",
            output=[
                {
                    "id": f"msg_{uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (messages[-1].get("content") or "") if messages else "",
                            "annotations": [],
                        }
                    ],
                }
            ],
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
        )

    async def _release(self, session_key: str, sandbox: AsyncSandbox) -> None:
        """Forget the session and stop its sandbox.

        The bookkeeping goes first, so it happens even if the stop is cancelled; the
        stop is shielded, so a cancelled rollout still releases the sandbox rather
        than abandoning the stop halfway. A failed stop is logged, not raised: it
        must not replace the error that ended the attempt.
        """
        self._session_id_to_sandbox.pop(session_key, None)
        self._session_id_to_stats.pop(session_key, None)
        try:
            await asyncio.shield(sandbox.stop())
        except Exception:
            print("prefix_pass_k: failed to stop sandbox", format_exc(), file=sys.stderr)

    async def run(self, request: Request, body: PrefixPassKRunRequest) -> PrefixPassKVerifyResponse:
        cookies = request.cookies
        session_key = request.session[SESSION_ID_KEY]

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = cookies | seed_session_response.cookies
        seed_session_result = await seed_session_response.json()

        sandbox = await self._connect_sandbox(seed_session_result)
        self._session_id_to_sandbox[session_key] = sandbox
        try:
            if self.config.offline:
                await self._take_offline(sandbox)

            # The starting commit, not the row's `base_commit`: task images need not
            # sit at that hash, and the question is whether the rollout committed.
            initial_head = await self._head(sandbox)

            request._cookies = cookies
            # The row carries the prefix; `responses` is the Responses-API surface
            # and only receives create params, so hand it over on request state.
            request.state._ng_prefix_pass_k_row = body.model_dump()

            try:
                response = await self.responses(request, body.responses_create_params)
            finally:
                request.state._ng_prefix_pass_k_row = None

            stats = self._session_id_to_stats.setdefault(session_key, {})
            aborted = stats.get("exec_errors", 0) > 0
            if not aborted:
                # Skipped on an aborted attempt: its sandbox just failed to run a command.
                if self.config.record_git_state:
                    stats.update(await self._git_state(sandbox, initial_head))
                # The full change set, before any grading-side git steps: re-grading the
                # rollout later, under a different collector, needs neither model nor sandbox.
                stats["worktree_patch"] = await self._worktree_patch(sandbox, initial_head)
                if self.config.commit_worktree:
                    await self._commit_worktree(sandbox)
                if self.config.uncommit_worktree:
                    # After record_git_state, so the diagnostic still shows what the model did.
                    await self._uncommit_worktree(sandbox, initial_head)

            verify_request = PrefixPassKVerifyRequest.model_validate(body.model_dump() | {"response": response})
            try:
                # Called on an aborted attempt too, so the resources server releases its sandboxes.
                verify_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/verify",
                    json=verify_request.model_dump(),
                    cookies=cookies,
                )
                await raise_for_status(verify_response)
                verify_json = await get_response_json(verify_response)
            except Exception:
                if not aborted:
                    raise
                print("prefix_pass_k: verify failed on an aborted attempt", format_exc(), file=sys.stderr)
                verify_json = verify_request.model_dump()

            stats = self._session_id_to_stats.pop(session_key, {})
            if aborted:
                # The infrastructure failed, not the candidate: whatever the verifier saw is
                # not a measurement of it, so the sample is masked out of pass@k.
                verify_json = verify_json | {"reward": 0.0, "mask_sample": True}
            return PrefixPassKVerifyResponse.model_validate(verify_json | stats)
        finally:
            # Whatever ended the attempt -- a model or verifier error, a cancelled
            # rollout -- the session is forgotten and its sandbox stopped.
            await self._release(session_key, sandbox)


if __name__ == "__main__":
    PrefixPassKAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PrefixPassKAgent.run_webserver()  # noqa: F401
