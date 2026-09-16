import asyncio
import json
import shlex
import sqlite3
from types import SimpleNamespace

import pytest
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.opencode import OpenCode
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import AgentConfig

from responses_api_agents.harbor_agent_general.alerts import AlertScheduleConfig
from responses_api_agents.harbor_agent_general.audited_opencode import (
    JUDGE_FILE_TRACE_FILENAME,
    POLICY_FILE_TRACE_FILENAME,
    AlertedOpenCode,
    AuditedOpenCode,
    OpenCodeProcessRLimitConfig,
    PreinstalledOpenCode,
    _OpenCodeAlertingEnvironment,
    _OpenCodeProcessLimitEnvironment,
    _PolicyTracingEnvironment,
)


class FakeEnvironment:
    def __init__(self):
        self.default_user = "agent"
        self.calls = []

    async def exec(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return SimpleNamespace(return_code=0, stdout="", stderr="")


@pytest.mark.parametrize("agent_class", [PreinstalledOpenCode, AlertedOpenCode, AuditedOpenCode])
def test_factory_preflight_and_constructor_preserve_options(tmp_path, agent_class):
    title = "analysis 'quoted' ; $(must-not-run)"
    kwargs = {
        "use_preinstalled": True,
        "session_title": title,
        "variant": "high",
        "opencode_config": {"compaction": {"auto": True}},
    }
    if issubclass(agent_class, AlertedOpenCode):
        kwargs.update(
            alert_schedule={
                "deadline_seconds": 3600,
                "alerts": [{"remaining_seconds": 300, "message": "Finish now."}],
            },
            opencode_port=4321,
            process_rlimits={"address_space_mib": 49152},
        )
    if agent_class is AuditedOpenCode:
        kwargs["trace_filename"] = "custom.trace"
    config = AgentConfig(
        name=None,
        import_path=f"{agent_class.__module__}:{agent_class.__name__}",
        kwargs=kwargs,
    )
    AgentFactory.run_preflight(config)
    agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)

    assert agent.options.model_dump() == agent_class.parse_options(kwargs).model_dump()
    assert shlex.split(agent.build_cli_flags()) == ["--variant", "high", "--title", title]
    assert agent._opencode_config == kwargs["opencode_config"]
    if isinstance(agent, AlertedOpenCode):
        assert agent._use_preinstalled is True
        assert agent._opencode_port == 4321
        assert agent._alert_schedule is agent.options.alert_schedule
        assert agent._process_rlimits is agent.options.process_rlimits
    if isinstance(agent, AuditedOpenCode):
        assert agent._trace_filename == "custom.trace"


@pytest.mark.parametrize(
    ("agent_class", "kwargs", "error"),
    [
        (PreinstalledOpenCode, {"use_preinstalled": False}, "use_preinstalled"),
        (AlertedOpenCode, {"opencode_port": 0}, "opencode_port"),
        (AlertedOpenCode, {"opencode_port": 65536}, "opencode_port"),
        (AlertedOpenCode, {"process_rlimits": {"address_space_mib": 0}}, "address_space_mib"),
        (AlertedOpenCode, {"alert_schedule": {"deadline_seconds": 0, "alerts": []}}, "alert_schedule"),
        (AlertedOpenCode, {"use_preinstaled": True}, "Unknown option 'use_preinstaled'"),
        (AuditedOpenCode, {"trace_filename": "../outside.trace"}, "trace_filename"),
        (AuditedOpenCode, {"trace_filename": ""}, "trace_filename"),
    ],
)
def test_invalid_options_fail_before_trials_start(tmp_path, agent_class, kwargs, error):
    config = AgentConfig(
        name=None,
        import_path=f"{agent_class.__module__}:{agent_class.__name__}",
        kwargs=kwargs,
    )
    with pytest.raises(ValueError, match=error):
        AgentFactory.run_preflight(config)
    with pytest.raises(ValueError, match=error):
        AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)


def _write_session_database(tmp_path, messages):
    database_path = tmp_path / "opencode" / "xdg-data" / "opencode" / "opencode.db"
    database_path.parent.mkdir(parents=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            """
            CREATE TABLE session (id TEXT PRIMARY KEY, parent_id TEXT, time_created INTEGER);
            CREATE TABLE message (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                time_created INTEGER,
                data TEXT
            );
            CREATE TABLE part (
                id TEXT PRIMARY KEY,
                message_id TEXT,
                session_id TEXT,
                time_created INTEGER,
                data TEXT
            );
            """
        )
        connection.execute("INSERT INTO session VALUES (?, ?, ?)", ("session-1", None, 1))
        part_index = 0
        for message in messages:
            message_id = message["id"]
            connection.execute(
                "INSERT INTO message VALUES (?, ?, ?, ?)",
                (message_id, "session-1", message["timestamp"], json.dumps(message["info"])),
            )
            for part in message["parts"]:
                part_index += 1
                connection.execute(
                    "INSERT INTO part VALUES (?, ?, ?, ?, ?)",
                    (
                        f"part-{part_index}",
                        message_id,
                        "session-1",
                        message["timestamp"] + part_index,
                        json.dumps(part),
                    ),
                )
        connection.commit()
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_wraps_only_policy_command_without_revealing_honeypot_path():
    delegate = FakeEnvironment()
    environment = _PolicyTracingEnvironment(delegate)

    await environment.exec("mkdir -p ~/.config/opencode")
    await environment.exec(
        "set -o pipefail; . ~/.nvm/nvm.sh; opencode --model=nemo/test run --format=json --thinking -- prompt"
    )

    assert delegate.calls[0][0] == "mkdir -p ~/.config/opencode"
    traced_command = delegate.calls[1][0]
    assert traced_command.startswith("strace -f -qq -yy -s")
    assert f"/logs/agent/{POLICY_FILE_TRACE_FILENAME}" in traced_command
    assert "opencode --model=nemo/test" in traced_command
    assert traced_command.endswith(">/dev/null")
    assert "/mnt/s3-data" not in traced_command
    assert environment.policy_command_wrapped is True


@pytest.mark.asyncio
async def test_rejects_multiple_policy_commands():
    environment = _PolicyTracingEnvironment(FakeEnvironment())
    command = "opencode --model=nemo/test run --format=json -- prompt"

    await environment.exec(command)
    with pytest.raises(RuntimeError, match="more than once"):
        await environment.exec(command)


@pytest.mark.asyncio
async def test_process_limit_wraps_only_opencode_run_and_is_inherited_by_children():
    delegate = FakeEnvironment()
    environment = _OpenCodeProcessLimitEnvironment(
        delegate,
        OpenCodeProcessRLimitConfig(address_space_mib=49152),
    )

    await environment.exec("opencode --version")
    await environment.exec("opencode --model=nemo/test run --format=json -- prompt")

    assert delegate.calls[0][0] == "opencode --version"
    limited_command = delegate.calls[1][0]
    assert limited_command.startswith("set -e; ulimit -S -v 50331648; ulimit -H -v 50331648; ")
    assert limited_command.endswith("opencode --model=nemo/test run --format=json -- prompt")
    assert environment.command_wrapped is True


@pytest.mark.asyncio
async def test_preinstalled_opencode_only_checks_baked_install(tmp_path):
    environment = FakeEnvironment()
    agent = PreinstalledOpenCode(logs_dir=tmp_path)

    await agent.install(environment)

    assert len(environment.calls) == 1
    command = environment.calls[0][0]
    assert "opencode --version" in command
    assert "npm i" not in command


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_class", [PreinstalledOpenCode, AlertedOpenCode])
async def test_opencode_wrappers_set_fixed_title_to_skip_title_generation(tmp_path, agent_class):
    environment = FakeEnvironment()
    agent = agent_class(logs_dir=tmp_path, model_name="nemo/test")

    await agent.run("Analyze the data.", environment, AgentContext())

    run_command = next(command for command, _kwargs in environment.calls if "opencode --model=" in command)
    assert "--title nemo-gym" in run_command


def test_opencode_title_generation_can_be_explicitly_restored(tmp_path):
    agent = PreinstalledOpenCode(logs_dir=tmp_path, session_title=None)

    assert "--title" not in agent.build_cli_flags()


@pytest.mark.asyncio
async def test_audited_opencode_checks_baked_trace_tools(tmp_path):
    environment = FakeEnvironment()
    agent = AuditedOpenCode(logs_dir=tmp_path, use_preinstalled=True)

    await agent.install(environment)

    assert len(environment.calls) == 1
    command = environment.calls[0][0]
    assert "command -v strace" in command
    assert "opencode --version" in command
    assert "npm i" not in command


@pytest.mark.asyncio
async def test_audited_opencode_installs_agent_when_not_preinstalled(tmp_path):
    environment = FakeEnvironment()
    agent = AuditedOpenCode(logs_dir=tmp_path, use_preinstalled=False)

    await agent.install(environment)

    commands = [command for command, _kwargs in environment.calls]
    assert any("apt-get install -y strace" in command for command in commands)
    assert any("npm i -g opencode-ai" in command for command in commands)


@pytest.mark.asyncio
async def test_audited_opencode_uses_configured_trace_filename(tmp_path, monkeypatch):
    delegate = FakeEnvironment()
    agent = AuditedOpenCode(
        logs_dir=tmp_path,
        use_preinstalled=True,
        trace_filename=JUDGE_FILE_TRACE_FILENAME,
    )

    async def fake_run(_self, instruction, environment, context):
        await environment.exec("opencode --model=rubric/test run --format=json -- prompt")

    monkeypatch.setattr(
        "harbor.agents.installed.opencode.OpenCode.run",
        fake_run,
    )
    await agent.run("judge", delegate, SimpleNamespace())

    assert f"/logs/agent/{JUDGE_FILE_TRACE_FILENAME}" in delegate.calls[0][0]


@pytest.mark.asyncio
async def test_alerting_environment_delivers_while_policy_is_running():
    class BlockingEnvironment(FakeEnvironment):
        def __init__(self):
            super().__init__()
            self.policy_done = asyncio.Event()

        async def exec(self, command, **kwargs):
            self.calls.append((command, kwargs))
            if "opencode --model=" in command:
                await self.policy_done.wait()
            else:
                self.policy_done.set()
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    delegate = BlockingEnvironment()
    schedule = AlertScheduleConfig.model_validate(
        {
            "deadline_seconds": 0.05,
            "poll_interval_seconds": 0.001,
            "retry_interval_seconds": 0.001,
            "alerts": [{"remaining_seconds": 0.04, "message": "finish now"}],
        }
    )
    environment = _OpenCodeAlertingEnvironment(delegate, schedule, port=4567)

    await environment.exec("opencode --model=nemo/test run --format=json --thinking -- prompt")

    assert "opencode serve --hostname=127.0.0.1 --port=4567" in delegate.calls[0][0]
    assert "run --attach=http://127.0.0.1:4567 --format=json" in delegate.calls[0][0]
    assert "/global/health" in delegate.calls[0][0]
    assert "prompt_async" not in delegate.calls[1][0]
    assert environment.scheduler is not None
    assert environment.scheduler.status()[0]["delivered"] is True


@pytest.mark.asyncio
async def test_alert_failure_preserves_background_command_output():
    class FailingAlertEnvironment(FakeEnvironment):
        def __init__(self):
            super().__init__()
            self.policy_done = asyncio.Event()

        async def exec(self, command, **kwargs):
            self.calls.append((command, kwargs))
            if "opencode --model=" in command:
                await self.policy_done.wait()
                return SimpleNamespace(return_code=0, stdout="", stderr="")
            self.policy_done.set()
            return SimpleNamespace(
                return_code=1,
                stdout="ConnectionRefusedError: [Errno 111] Connection refused",
                stderr="exit status 1",
            )

    schedule = AlertScheduleConfig.model_validate(
        {
            "deadline_seconds": 0.05,
            "poll_interval_seconds": 0.001,
            "retry_interval_seconds": 1,
            "alerts": [{"remaining_seconds": 0.04, "message": "finish now"}],
        }
    )
    environment = _OpenCodeAlertingEnvironment(
        FailingAlertEnvironment(),
        schedule,
        port=4567,
    )

    await environment.exec("opencode --model=nemo/test run --format=json -- prompt")

    assert environment.scheduler is not None
    error = environment.scheduler.status()[0]["last_error"]
    assert "ConnectionRefusedError" in error
    assert "exit status 1" in error


@pytest.mark.asyncio
async def test_alert_metadata_is_added_after_trajectory_population(
    tmp_path,
    monkeypatch,
):
    alert_message = "Finish now."
    agent = AlertedOpenCode(
        logs_dir=tmp_path,
        model_name="nemo/test",
        alert_schedule={
            "deadline_seconds": 0.05,
            "poll_interval_seconds": 0.001,
            "retry_interval_seconds": 0.001,
            "alerts": [{"remaining_seconds": 0.04, "message": alert_message}],
        },
    )
    context = AgentContext()

    async def fake_run(_self, _instruction, environment, _context):
        await environment.exec("opencode --model=nemo/test run --format=json --thinking -- prompt")

    monkeypatch.setattr(OpenCode, "run", fake_run)

    class BlockingEnvironment(FakeEnvironment):
        def __init__(self):
            super().__init__()
            self.policy_done = asyncio.Event()

        async def exec(self, command, **kwargs):
            self.calls.append((command, kwargs))
            if "opencode --model=" in command:
                await self.policy_done.wait()
            else:
                self.policy_done.set()
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    await agent.run("Analyze the data.", BlockingEnvironment(), context)

    # Harbor uses this predicate to decide whether to parse downloaded logs.
    assert context.is_empty()

    events = [
        {
            "type": "user",
            "timestamp": 1,
            "sessionID": "session-1",
            "parts": [{"type": "text", "text": "Analyze the data."}],
        },
        {"type": "step_start", "timestamp": 2, "sessionID": "session-1"},
        {
            "type": "text",
            "timestamp": 3,
            "sessionID": "session-1",
            "part": {"type": "text", "text": "done"},
        },
        {
            "type": "step_finish",
            "timestamp": 4,
            "sessionID": "session-1",
            "part": {"tokens": {"input": 7, "output": 2}},
        },
    ]
    (tmp_path / "opencode.txt").write_text("".join(json.dumps(event) + "\n" for event in events))
    _write_session_database(
        tmp_path,
        [
            {
                "id": "user-1",
                "timestamp": 1,
                "info": {"role": "user"},
                "parts": [{"type": "text", "text": "Analyze the data."}],
            },
            {
                "id": "assistant-1",
                "timestamp": 2,
                "info": {"role": "assistant", "parentID": "user-1", "finish": "stop"},
                "parts": [
                    {"type": "step-start"},
                    {"type": "text", "text": "done", "time": {"end": 3}},
                    {"type": "step-finish", "reason": "stop", "tokens": {"input": 7, "output": 2}},
                ],
            },
            {
                "id": "user-alert",
                "timestamp": 10,
                "info": {"role": "user"},
                "parts": [{"type": "text", "text": alert_message}],
            },
            {
                "id": "assistant-alert",
                "timestamp": 11,
                "info": {"role": "assistant", "parentID": "user-alert", "finish": "stop"},
                "parts": [
                    {"type": "step-start"},
                    {"type": "text", "text": "submitted", "time": {"end": 12}},
                    {"type": "step-finish", "reason": "stop", "tokens": {"input": 9, "output": 1}},
                ],
            },
        ],
    )

    agent.populate_context_post_run(context)

    assert (tmp_path / "trajectory.json").is_file()
    assert context.n_input_tokens == 16
    assert context.n_output_tokens == 3
    assert context.metadata is not None
    assert context.metadata["runtime_alerts"][0]["delivered"] is True
    assert context.metadata["runtime_alerts"][0]["session_user_turn_recorded"] is True
    assert context.metadata["runtime_alerts"][0]["session_alert_processed"] is True
    assert context.metadata["runtime_alerts"][0]["session_zero_token_length"] is False
    assert context.metadata["opencode_session_capture"]["source"] == "database"


def test_alerted_opencode_tracks_overflow_compaction_and_replayed_alert(tmp_path):
    alert_message = "You have five minutes remaining."
    agent = AlertedOpenCode(
        logs_dir=tmp_path,
        model_name="nemo/test",
        alert_schedule={
            "deadline_seconds": 60,
            "alerts": [{"remaining_seconds": 30, "message": alert_message}],
        },
    )
    agent._runtime_alert_status = [{"name": "remaining_30s", "delivered": True}]
    agent._session_events = [
        {
            "type": "user",
            "messageID": "alert-original",
            "parts": [{"type": "text", "text": alert_message}],
        },
        {
            "type": "error",
            "messageID": "assistant-overflow",
            "parentID": "alert-original",
            "error": {"name": "ContextOverflowError"},
        },
        {
            "type": "user",
            "messageID": "compact-user",
            "parts": [{"type": "compaction", "auto": True, "overflow": True}],
        },
        {"type": "compaction", "messageID": "compact-user"},
        {
            "type": "step_finish",
            "messageID": "compact-assistant",
            "parentID": "compact-user",
            "assistantSummary": True,
            "part": {"reason": "stop", "tokens": {"input": 100, "output": 20}},
        },
        {
            "type": "user",
            "messageID": "alert-replay",
            "parts": [{"type": "text", "text": alert_message}],
        },
        {
            "type": "step_finish",
            "messageID": "assistant-replay",
            "parentID": "alert-replay",
            "assistantSummary": False,
            "part": {"reason": "stop", "tokens": {"input": 30, "output": 2}},
        },
    ]

    agent._annotate_alert_outcomes()

    status = agent._runtime_alert_status[0]
    assert status["session_direct_context_overflow"] is True
    assert status["session_compaction_recorded"] is True
    assert status["session_compaction_completed"] is True
    assert status["session_alert_replayed"] is True
    assert status["session_alert_processed"] is True
    assert status["session_replay_prompt_tokens"] == 30
    assert status["session_replay_completion_tokens"] == 2


def test_alerted_opencode_does_not_treat_zero_token_length_as_processed(tmp_path):
    alert_message = "You have five minutes remaining."
    agent = AlertedOpenCode(
        logs_dir=tmp_path,
        model_name="nemo/test",
        alert_schedule={
            "deadline_seconds": 60,
            "alerts": [{"remaining_seconds": 30, "message": alert_message}],
        },
    )
    agent._runtime_alert_status = [{"name": "remaining_30s", "delivered": True}]
    agent._session_events = [
        {
            "type": "user",
            "messageID": "alert-original",
            "parts": [{"type": "text", "text": alert_message}],
        },
        {
            "type": "step_finish",
            "messageID": "assistant-empty",
            "parentID": "alert-original",
            "assistantSummary": False,
            "part": {"reason": "length", "tokens": {"input": 0, "output": 0}},
        },
    ]

    agent._annotate_alert_outcomes()

    status = agent._runtime_alert_status[0]
    assert status["session_zero_token_length"] is True
    assert status["session_alert_processed"] is False
    assert status["session_compaction_recorded"] is False


def test_alerted_opencode_preserves_alert_as_user_turn(tmp_path):
    alert_message = "You have five minutes remaining."
    agent = AlertedOpenCode(
        logs_dir=tmp_path,
        model_name="nemo/test",
        alert_schedule={
            "deadline_seconds": 60,
            "alerts": [{"remaining_seconds": 30, "message": alert_message}],
        },
    )
    agent._instruction = "Analyze the data."
    events = [
        {
            "type": "user",
            "timestamp": 1,
            "parts": [{"type": "text", "text": "Analyze the data."}],
        },
        {"type": "step_start", "timestamp": 2},
        {"type": "text", "part": {"type": "text", "text": "working"}},
        {
            "type": "user",
            "timestamp": 3,
            "parts": [{"type": "text", "text": alert_message}],
        },
        {"type": "step_finish", "part": {"tokens": {"input": 1, "output": 1}}},
        {"type": "step_start", "timestamp": 4},
        {"type": "text", "part": {"type": "text", "text": "done"}},
        {"type": "step_finish", "part": {"tokens": {"input": 1, "output": 1}}},
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert [step.source for step in trajectory.steps] == [
        "user",
        "agent",
        "user",
        "agent",
    ]
    assert trajectory.steps[2].message == alert_message
