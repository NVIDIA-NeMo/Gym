# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenCode agent variants that record agent-process filesystem accesses."""

import base64
import json
import shlex
import sqlite3
from pathlib import Path, PurePath
from typing import Annotated, Any, Literal, cast, override

from harbor.agents.installed.opencode import OpenCode, OpenCodeOptions
from harbor.agents.options import Cli
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Step, Trajectory
from pydantic import BaseModel, ConfigDict, Field, field_validator

from responses_api_agents.harbor_agent_general.alerts import (
    AlertContext,
    AlertScheduleConfig,
    build_time_remaining_scheduler,
)


POLICY_FILE_TRACE_FILENAME = ".policy-fs.trace"
JUDGE_FILE_TRACE_FILENAME = ".judge-fs.trace"
_OPENCODE_PORT = 4096
_OPENCODE_DATABASE_RELATIVE_PATH = Path("opencode/xdg-data/opencode/opencode.db")
_PREINSTALLED_OPENCODE_CHECK = (
    'set -euo pipefail; test -r "$HOME/.nvm/nvm.sh"; '
    '. "$HOME/.nvm/nvm.sh"; command -v opencode >/dev/null; opencode --version'
)


class OpenCodeProcessRLimitConfig(BaseModel):
    """Per-process resource limits inherited by OpenCode and its children."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    address_space_mib: int = Field(gt=0)


class TitledOpenCodeOptions(OpenCodeOptions):
    """Preserve upstream options while avoiding unsolicited title-model calls."""

    # A non-default session title makes OpenCode skip its title-model request.
    session_title: Annotated[str | None, Cli("--title")] = "nemo-gym"


class PreinstalledOpenCodeOptions(TitledOpenCodeOptions):
    """Options for an agent that always uses the image's OpenCode installation."""

    use_preinstalled: Literal[True] = True


class AlertedOpenCodeOptions(TitledOpenCodeOptions):
    """Runtime controls validated before Harbor queues any trials."""

    use_preinstalled: bool = False
    alert_schedule: AlertScheduleConfig | None = None
    opencode_port: int = Field(default=_OPENCODE_PORT, ge=1, le=65535)
    process_rlimits: OpenCodeProcessRLimitConfig | None = None


class AuditedOpenCodeOptions(AlertedOpenCodeOptions):
    """Runtime controls and the name of the agent-process filesystem trace."""

    trace_filename: str = POLICY_FILE_TRACE_FILENAME

    @field_validator("trace_filename")
    @classmethod
    def validate_trace_filename(cls, value: str) -> str:
        if not value or PurePath(value).name != value:
            raise ValueError("trace_filename must be a filename, not a path")
        return value


def _is_policy_command(command: str) -> bool:
    return "opencode --model=" in command and " run " in command and "--format=json" in command


def _load_session_events(database_path: Path, session_id: str | None) -> list[dict[str, Any]]:
    """Reconstruct OpenCode's complete persisted session as CLI-shaped events."""
    connection = sqlite3.connect(
        f"{database_path.resolve().as_uri()}?mode=ro",
        uri=True,
        timeout=10,
    )
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        if session_id is None:
            row = connection.execute(
                "SELECT id FROM session WHERE parent_id IS NULL ORDER BY time_created, id LIMIT 1"
            ).fetchone()
            session_id = str(row[0]) if row is not None else None
        if session_id is None:
            return []

        message_rows = connection.execute(
            "SELECT id, time_created, data FROM message WHERE session_id = ? ORDER BY time_created, id",
            (session_id,),
        ).fetchall()
        part_rows = connection.execute(
            "SELECT message_id, time_created, data FROM part WHERE session_id = ? ORDER BY time_created, id",
            (session_id,),
        ).fetchall()
    finally:
        connection.close()

    parts_by_message: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for message_id, timestamp, raw_data in part_rows:
        try:
            part = json.loads(raw_data)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid OpenCode part JSON for {message_id}") from error
        parts_by_message.setdefault(str(message_id), []).append((int(timestamp), part))

    events: list[dict[str, Any]] = []
    event_type_by_part_type = {
        "reasoning": "reasoning",
        "step-finish": "step_finish",
        "step-start": "step_start",
        "text": "text",
        "tool": "tool_use",
    }
    for message_id, message_timestamp, raw_data in message_rows:
        try:
            info = json.loads(raw_data)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid OpenCode message JSON for {message_id}") from error
        message_id = str(message_id)
        message_parts = parts_by_message.get(message_id, [])
        common = {
            "sessionID": session_id,
            "messageID": message_id,
            "parentID": info.get("parentID"),
            "assistantSummary": bool(info.get("summary")),
        }
        if info.get("role") == "user":
            events.append(
                {
                    **common,
                    "type": "user",
                    "timestamp": int(message_timestamp),
                    "parts": [part for _timestamp, part in message_parts],
                }
            )
            for timestamp, part in message_parts:
                if part.get("type") == "compaction":
                    events.append(
                        {
                            **common,
                            "type": "compaction",
                            "timestamp": timestamp,
                            "part": part,
                        }
                    )
            continue

        if info.get("role") != "assistant":
            continue
        for timestamp, part in message_parts:
            event_type = event_type_by_part_type.get(part.get("type"))
            if event_type is None:
                continue
            events.append(
                {
                    **common,
                    "type": event_type,
                    "timestamp": timestamp,
                    "part": part,
                }
            )
        if info.get("error"):
            events.append(
                {
                    **common,
                    "type": "error",
                    "timestamp": int(info.get("time", {}).get("completed") or message_timestamp),
                    "error": info["error"],
                }
            )
    return events


def _run_with_opencode_server(command: str, port: int) -> str:
    """Run the policy through a loopback OpenCode server for live user turns."""
    if not _is_policy_command(command):
        return command

    server_url = f"http://127.0.0.1:{port}"
    attached_command = command.replace(
        " run ",
        f" run --attach={server_url} ",
        1,
    )
    readiness_script = f"""
import http.client
import time

deadline = time.monotonic() + 60
last_error = None
while True:
    connection = http.client.HTTPConnection('127.0.0.1', {port}, timeout=1)
    try:
        connection.request('GET', '/global/health')
        response = connection.getresponse()
        response.read()
        if response.status == 200:
            break
        last_error = RuntimeError(f'OpenCode health returned HTTP {{response.status}}')
    except (OSError, http.client.HTTPException) as error:
        last_error = error
    finally:
        connection.close()
    if time.monotonic() >= deadline:
        raise RuntimeError('OpenCode server did not become healthy') from last_error
    time.sleep(0.1)
"""
    return (
        'set -e; . "$HOME/.nvm/nvm.sh"; '
        f"opencode serve --hostname=127.0.0.1 --port={port} "
        ">/logs/agent/opencode-server.log 2>&1 & "
        "__nemo_opencode_server_pid=$!; "
        'trap \'kill "$__nemo_opencode_server_pid" 2>/dev/null || true; '
        'wait "$__nemo_opencode_server_pid" 2>/dev/null || true\' EXIT; '
        f"python3 -c {shlex.quote(readiness_script)}; "
        f"{attached_command}"
    )


def _alert_delivery_command(*, port: int, message: str) -> str:
    """Build a sandbox-local request without putting the message in shell syntax."""
    encoded_message = base64.b64encode(message.encode()).decode()
    script = f"""
import base64
import json
import pathlib
import urllib.parse
import urllib.request

events = []
for line in pathlib.Path('/logs/agent/opencode.txt').read_text().splitlines():
    try:
        events.append(json.loads(line))
    except json.JSONDecodeError:
        pass
session_id = next(
    (event.get('sessionID') for event in reversed(events) if event.get('sessionID')),
    None,
)
if not session_id:
    raise RuntimeError('OpenCode session id is not available yet')
message = base64.b64decode('{encoded_message}').decode()
payload = json.dumps({{
    'parts': [{{'type': 'text', 'text': message}}],
    'noReply': False,
}}).encode()
request = urllib.request.Request(
    'http://127.0.0.1:{port}/session/' + urllib.parse.quote(session_id, safe='') + '/prompt_async',
    data=payload,
    headers={{
        'Content-Type': 'application/json',
        'x-opencode-directory': str(pathlib.Path.cwd()),
    }},
    method='POST',
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
with opener.open(request, timeout=30) as response:
    response.read()
"""
    encoded_script = base64.b64encode(script.encode()).decode()
    return f"python3 -c \"import base64; exec(base64.b64decode('{encoded_script}'))\""


def _trace_command(command: str, trace_filename: str) -> str:
    trace_path = f"/logs/agent/{trace_filename}"
    return (
        "strace -f -qq -yy -s 4096 "
        f"-e trace=%file -o {shlex.quote(trace_path)} "
        f"bash -o pipefail -c {shlex.quote(command)} >/dev/null"
    )


def _process_limit_command(command: str, config: OpenCodeProcessRLimitConfig) -> str:
    address_space_kib = config.address_space_mib * 1024
    return f"set -e; ulimit -S -v {address_space_kib}; ulimit -H -v {address_space_kib}; {command}"


class _OpenCodeTracingEnvironment:
    """Delegate a Harbor environment while wrapping one OpenCode run."""

    def __init__(self, environment: BaseEnvironment, trace_filename: str):
        self._environment = environment
        self._trace_filename = trace_filename
        self.command_wrapped = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    async def exec(self, command: str, **kwargs: Any) -> Any:
        if _is_policy_command(command):
            if self.command_wrapped:
                raise RuntimeError("OpenCode run command was issued more than once")
            self.command_wrapped = True
            command = _trace_command(command, self._trace_filename)
        return await self._environment.exec(command=command, **kwargs)


class _OpenCodeProcessLimitEnvironment:
    """Apply inherited per-process limits to one complete OpenCode run."""

    def __init__(
        self,
        environment: BaseEnvironment,
        config: OpenCodeProcessRLimitConfig,
    ) -> None:
        self._environment = environment
        self._config = config
        self.command_wrapped = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    async def exec(self, command: str, **kwargs: Any) -> Any:
        if _is_policy_command(command):
            if self.command_wrapped:
                raise RuntimeError("OpenCode run command was issued more than once")
            self.command_wrapped = True
            command = _process_limit_command(command, self._config)
        return await self._environment.exec(command=command, **kwargs)


class _OpenCodeAlertingEnvironment:
    """Run OpenCode with its local API enabled and deliver scheduled user turns."""

    def __init__(
        self,
        environment: BaseEnvironment,
        alert_schedule: AlertScheduleConfig,
        port: int,
    ) -> None:
        self._environment = environment
        self._alert_schedule = alert_schedule
        self._port = port
        self.scheduler = None
        self.command_wrapped = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    async def exec(self, command: str, **kwargs: Any) -> Any:
        if not _is_policy_command(command):
            return await self._environment.exec(command=command, **kwargs)
        if self.command_wrapped:
            raise RuntimeError("OpenCode run command was issued more than once")
        self.command_wrapped = True

        async def deliver(message: str, _context: AlertContext) -> None:
            result = await self._environment.exec(
                command=_alert_delivery_command(port=self._port, message=message),
                timeout_sec=45,
            )
            if result.return_code != 0:
                outputs = [output.strip() for output in (result.stdout, result.stderr) if output and output.strip()]
                output = "\n".join(outputs) or "no output"
                raise RuntimeError(f"OpenCode alert delivery failed: {output}")

        self.scheduler = build_time_remaining_scheduler(
            self._alert_schedule,
            deliver,
        )
        return await self.scheduler.run_until_complete(
            self._environment.exec(
                command=_run_with_opencode_server(command, self._port),
                **kwargs,
            )
        )


class _PolicyTracingEnvironment(_OpenCodeTracingEnvironment):
    """Backward-compatible policy tracing environment used by older callers."""

    def __init__(self, environment: BaseEnvironment):
        super().__init__(environment, POLICY_FILE_TRACE_FILENAME)

    @property
    def policy_command_wrapped(self) -> bool:
        return self.command_wrapped


class PreinstalledOpenCode(OpenCode):
    """Use the OpenCode installation baked into the sandbox image."""

    options_model = PreinstalledOpenCodeOptions

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_agent(environment, command=_PREINSTALLED_OPENCODE_CHECK)


class AlertedOpenCode(OpenCode):
    """Inject configured time-left warnings as native OpenCode user turns."""

    options_model = AlertedOpenCodeOptions
    options: AlertedOpenCodeOptions

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._use_preinstalled = self.options.use_preinstalled
        self._alert_schedule = self.options.alert_schedule
        self._process_rlimits = self.options.process_rlimits
        self._opencode_port = self.options.opencode_port
        self._runtime_alert_status: list[dict[str, Any]] | None = None
        self._session_events: list[dict[str, Any]] | None = None
        self._session_capture_status: dict[str, Any] | None = None

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if self._use_preinstalled:
            await self.exec_as_agent(
                environment,
                command=_PREINSTALLED_OPENCODE_CHECK,
            )
            return
        await super().install(environment)

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._runtime_alert_status = None
        self._session_events = None
        self._session_capture_status = None
        limiting_environment = None
        run_environment = environment
        if self._process_rlimits is not None:
            limiting_environment = _OpenCodeProcessLimitEnvironment(
                environment,
                self._process_rlimits,
            )
            run_environment = cast(BaseEnvironment, limiting_environment)
        if self._alert_schedule is None:
            await super().run(instruction, run_environment, context)
            if limiting_environment is not None and not limiting_environment.command_wrapped:
                raise RuntimeError("OpenCode run command was not resource limited")
            return

        alerting_environment = _OpenCodeAlertingEnvironment(
            run_environment,
            self._alert_schedule,
            self._opencode_port,
        )
        try:
            await super().run(
                instruction,
                cast(BaseEnvironment, alerting_environment),
                context,
            )
        finally:
            if alerting_environment.scheduler is not None:
                # Harbor only invokes populate_context_post_run() while the
                # context is empty, so attach metadata in that hook instead.
                self._runtime_alert_status = alerting_environment.scheduler.status()
        if limiting_environment is not None and not limiting_environment.command_wrapped:
            raise RuntimeError("OpenCode run command was not resource limited")

    @override
    def _parse_stdout(self) -> list[dict[str, Any]]:
        stdout_events = super()._parse_stdout()
        if self._alert_schedule is None:
            return stdout_events

        session_id = next(
            (event.get("sessionID") for event in stdout_events if event.get("sessionID")),
            None,
        )
        database_path = self.logs_dir / _OPENCODE_DATABASE_RELATIVE_PATH
        if not database_path.is_file():
            self._session_capture_status = {
                "source": "stdout",
                "event_count": len(stdout_events),
                "error": "database_missing",
            }
            return stdout_events

        try:
            session_events = _load_session_events(database_path, session_id)
        except (OSError, sqlite3.Error, ValueError) as error:
            self._session_capture_status = {
                "source": "stdout",
                "event_count": len(stdout_events),
                "error": f"{type(error).__name__}: {error}",
            }
            return stdout_events
        if not session_events:
            self._session_capture_status = {
                "source": "stdout",
                "event_count": len(stdout_events),
                "error": "database_session_empty",
            }
            return stdout_events

        self._session_events = session_events
        self._session_capture_status = {
            "source": "database",
            "event_count": len(session_events),
            "error": None,
        }
        return session_events

    def _annotate_alert_outcomes(self) -> None:
        if self._runtime_alert_status is None or self._alert_schedule is None:
            return
        events = self._session_events or []
        messages_by_name = {
            f"remaining_{alert.remaining_seconds:g}s": alert.message for alert in self._alert_schedule.alerts
        }
        for status in self._runtime_alert_status:
            expected_message = messages_by_name.get(str(status.get("name")))
            user_event = (
                next(
                    (
                        event
                        for event in events
                        if event.get("type") == "user" and self._user_event_text(event) == expected_message
                    ),
                    None,
                )
                if expected_message is not None
                else None
            )
            user_message_id = user_event.get("messageID") if user_event is not None else None
            user_event_index = events.index(user_event) if user_event is not None else -1
            direct_finishes = [
                event
                for event in events
                if event.get("type") == "step_finish"
                and event.get("parentID") == user_message_id
                and not event.get("assistantSummary")
            ]
            direct_errors = [
                event for event in events if event.get("type") == "error" and event.get("parentID") == user_message_id
            ]
            direct_finish = direct_finishes[0] if direct_finishes else None
            finish_part = direct_finish.get("part", {}) if direct_finish is not None else {}
            tokens = finish_part.get("tokens", {}) if isinstance(finish_part, dict) else {}
            input_tokens = int(tokens.get("input", 0) or 0) if isinstance(tokens, dict) else 0
            output_tokens = int(tokens.get("output", 0) or 0) if isinstance(tokens, dict) else 0
            finish_reason = finish_part.get("reason") if isinstance(finish_part, dict) else None
            direct_zero_token_length = finish_reason == "length" and input_tokens == 0 and output_tokens == 0

            # A typed input-overflow error causes OpenCode to append a new
            # compaction user message, summarize the prior session, and then
            # clone the interrupted user turn for replay. Follow those message
            # IDs rather than treating the summary itself as the alert reply.
            direct_error_text = json.dumps(
                [event.get("error") for event in direct_errors],
                sort_keys=True,
            ).lower()
            direct_context_overflow = any(
                marker in direct_error_text
                for marker in (
                    "context_length_exceeded",
                    "contextoverflow",
                    "context overflow",
                    "maximum context length",
                )
            )
            compaction_event = None
            if direct_context_overflow:
                compaction_event = next(
                    (event for event in events[user_event_index + 1 :] if event.get("type") == "compaction"),
                    None,
                )
            compaction_message_id = compaction_event.get("messageID") if compaction_event is not None else None
            summary_finishes = [
                event
                for event in events
                if event.get("type") == "step_finish"
                and event.get("parentID") == compaction_message_id
                and event.get("assistantSummary")
            ]
            replay_user_event = None
            if compaction_event is not None:
                compaction_event_index = events.index(compaction_event)
                replay_user_event = next(
                    (
                        event
                        for event in events[compaction_event_index + 1 :]
                        if event.get("type") == "user" and self._user_event_text(event) == expected_message
                    ),
                    None,
                )
            replay_user_message_id = replay_user_event.get("messageID") if replay_user_event is not None else None
            replay_finishes = [
                event
                for event in events
                if event.get("type") == "step_finish"
                and event.get("parentID") == replay_user_message_id
                and not event.get("assistantSummary")
            ]
            replay_finish = replay_finishes[0] if replay_finishes else None
            replay_finish_part = replay_finish.get("part", {}) if replay_finish is not None else {}
            replay_tokens = replay_finish_part.get("tokens", {}) if isinstance(replay_finish_part, dict) else {}
            replay_input_tokens = int(replay_tokens.get("input", 0) or 0) if isinstance(replay_tokens, dict) else 0
            replay_output_tokens = int(replay_tokens.get("output", 0) or 0) if isinstance(replay_tokens, dict) else 0
            replay_finish_reason = replay_finish_part.get("reason") if isinstance(replay_finish_part, dict) else None
            replay_zero_token_length = (
                replay_finish_reason == "length" and replay_input_tokens == 0 and replay_output_tokens == 0
            )
            direct_processed = direct_finish is not None and not direct_zero_token_length
            replay_processed = replay_finish is not None and not replay_zero_token_length
            status.update(
                {
                    "session_user_turn_recorded": user_event is not None,
                    "session_assistant_turn_recorded": any(
                        user_message_id is not None
                        and event.get("parentID") == user_message_id
                        and event.get("type") in {"step_start", "step_finish", "error"}
                        for event in events
                    ),
                    "session_alert_processed": direct_processed or replay_processed,
                    "session_direct_finish_reason": finish_reason,
                    "session_direct_prompt_tokens": input_tokens,
                    "session_direct_completion_tokens": output_tokens,
                    "session_direct_context_overflow": direct_context_overflow,
                    "session_alert_replayed": replay_user_event is not None,
                    "session_replay_finish_reason": replay_finish_reason,
                    "session_replay_prompt_tokens": replay_input_tokens,
                    "session_replay_completion_tokens": replay_output_tokens,
                    "session_zero_token_length": direct_zero_token_length or replay_zero_token_length,
                    "session_compaction_recorded": compaction_event is not None,
                    "session_compaction_completed": bool(summary_finishes),
                }
            )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)
        if self._runtime_alert_status is None:
            return
        self._annotate_alert_outcomes()
        metadata = dict(context.metadata or {})
        metadata["runtime_alerts"] = self._runtime_alert_status
        metadata["opencode_session_capture"] = self._session_capture_status
        context.metadata = metadata
        if (
            any(status.get("delivered") for status in self._runtime_alert_status)
            and (self._session_capture_status or {}).get("source") != "database"
        ):
            raise RuntimeError("OpenCode accepted a runtime alert but its canonical session database was not captured")

    @override
    def _convert_events_to_trajectory(
        self,
        events: list[dict[str, Any]],
    ) -> Trajectory | None:
        trajectory = super()._convert_events_to_trajectory(events)
        if trajectory is None or self._alert_schedule is None or self._instruction is None:
            return trajectory

        assistant_steps = [step for step in trajectory.steps if step.source == "agent"]
        rebuilt_steps = [Step(step_id=1, source="user", message=self._instruction)]
        assistant_index = 0
        turn_open = False
        deferred_alerts: list[Step] = []
        alert_messages = {alert.message for alert in self._alert_schedule.alerts}
        for event in events:
            event_type = event.get("type")
            if event_type == "step_start":
                turn_open = True
            elif event_type == "user":
                message = self._user_event_text(event)
                if message in alert_messages:
                    user_step = Step(
                        step_id=1,
                        timestamp=self._millis_to_iso(event.get("timestamp")),
                        source="user",
                        message=message,
                    )
                    if turn_open:
                        deferred_alerts.append(user_step)
                    else:
                        rebuilt_steps.append(user_step)
            elif event_type == "step_finish":
                if assistant_index < len(assistant_steps):
                    rebuilt_steps.append(assistant_steps[assistant_index])
                    assistant_index += 1
                rebuilt_steps.extend(deferred_alerts)
                deferred_alerts.clear()
                turn_open = False

        rebuilt_steps.extend(assistant_steps[assistant_index:])
        rebuilt_steps.extend(deferred_alerts)
        for step_id, step in enumerate(rebuilt_steps, start=1):
            step.step_id = step_id
        trajectory.steps = rebuilt_steps
        if trajectory.final_metrics is not None:
            trajectory.final_metrics.total_steps = len(rebuilt_steps)
        return trajectory


class AuditedOpenCode(AlertedOpenCode):
    """Trace actual file accesses made by OpenCode and its child processes.

    The trace records generic file syscalls. It intentionally has no knowledge
    of which paths are sensitive; classification happens after the sandbox has
    stopped so the audited path is never exposed to the policy process.
    """

    options_model = AuditedOpenCodeOptions
    options: AuditedOpenCodeOptions

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._trace_filename = self.options.trace_filename

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if self._use_preinstalled:
            await self.exec_as_agent(
                environment,
                command=(f"set -euo pipefail; command -v strace >/dev/null; {_PREINSTALLED_OPENCODE_CHECK}"),
            )
            return
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y strace",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await super().install(environment)

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        tracing_environment = _OpenCodeTracingEnvironment(environment, self._trace_filename)
        await super().run(
            instruction,
            cast(BaseEnvironment, tracing_environment),
            context,
        )
        if not tracing_environment.command_wrapped:
            raise RuntimeError("OpenCode run command was not traced")
