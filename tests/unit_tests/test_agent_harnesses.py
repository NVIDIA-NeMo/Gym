# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import tomllib
from pathlib import Path
from threading import Event
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from nemo_gym import agents
from nemo_gym.agents import (
    AgentHarnessConfig,
    AgentModelConfig,
    ClaudeCodeHarness,
    ClineHarness,
    CodexHarness,
    HermesHarness,
    KiloCodeHarness,
    OpenClawHarness,
    PiHarness,
    PrimeAgentHarness,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


def test_harnesses_share_one_public_config() -> None:
    config = AgentHarnessConfig(model=AgentModelConfig(model="test-model"))

    for harness_class in (
        ClaudeCodeHarness,
        ClineHarness,
        CodexHarness,
        HermesHarness,
        KiloCodeHarness,
        OpenClawHarness,
        PiHarness,
        PrimeAgentHarness,
    ):
        assert harness_class(config).config is config


def test_optional_harness_is_not_imported_by_star() -> None:
    assert "Terminus2Harness" not in agents.__all__


def test_terminus_extra_pins_harbor() -> None:
    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    dependencies = project["project"]["optional-dependencies"]["terminus-2"]
    assert "harbor==0.1.42" in dependencies
    assert all(" @ " not in dependency for dependency in dependencies)
    assert project["tool"]["uv"]["sources"]["harbor"] == {
        "git": "https://github.com/laude-institute/harbor.git",
        "rev": "9dddd797b57ab8a0" + "f9d6352a20fce73abbb29573",
    }


@pytest.mark.parametrize(
    "harness_class",
    [ClineHarness, CodexHarness, KiloCodeHarness, OpenClawHarness, PiHarness, PrimeAgentHarness],
)
def test_unsupported_max_turns_is_rejected(harness_class) -> None:
    config = AgentHarnessConfig(model=AgentModelConfig(model="test-model"), max_turns=3)

    with pytest.raises(ValueError, match="does not support max_turns"):
        harness_class(config)


def test_normalized_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AgentHarnessConfig.model_validate({"model": {"model": "test-model"}, "timeout": 10})


@pytest.mark.parametrize("field", ["timeout_seconds", "max_turns"])
def test_normalized_limits_must_be_positive(field) -> None:
    with pytest.raises(ValidationError):
        AgentHarnessConfig.model_validate({"model": {"model": "test-model"}, field: 0})


async def test_pi_preserves_state_in_caller_workspace(tmp_path) -> None:
    existing_home = tmp_path / ".pi-home"
    existing_home.mkdir()
    (existing_home / "keep").write_text("mine")
    harness = PiHarness(AgentHarnessConfig(model=AgentModelConfig(model="provider/model"), workspace=tmp_path))
    captured = {}

    class FakeProcess:
        returncode = 0
        stdout = object()
        stderr = object()

        async def communicate(self):
            return b"", b""

    async def fake_subprocess(*args, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    with patch("nemo_gym.agents.pi.asyncio.create_subprocess_exec", fake_subprocess):
        await harness._run_pi("solve", None, collect_observations=False)

    assert captured["cwd"] == str(tmp_path)
    assert captured["env"]["HOME"] != str(existing_home)
    assert (existing_home / "keep").read_text() == "mine"


async def test_hermes_scopes_process_environment_to_rollout(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", "caller-home")
    monkeypatch.setenv("TERMINAL_ENV", "caller-backend")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "caller-timeout")
    harness = HermesHarness(
        AgentHarnessConfig(
            model=AgentModelConfig(model="test-model"),
            settings={"terminal_backend": "local", "terminal_timeout": 17},
        )
    )
    assert os.environ["HERMES_HOME"] == "caller-home"

    seen = []

    class FakeAgent:
        def __init__(self, **kwargs) -> None:
            self._build_api_kwargs = lambda _messages: {}
            seen.append(tuple(os.environ[key] for key in ("HERMES_HOME", "TERMINAL_ENV", "TERMINAL_TIMEOUT")))

        def run_conversation(self, *args, **kwargs):
            seen.append(tuple(os.environ[key] for key in ("HERMES_HOME", "TERMINAL_ENV", "TERMINAL_TIMEOUT")))
            return {
                "completed": True,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"},
                ],
            }

    monkeypatch.setattr("nemo_gym.agents.hermes._load_ai_agent", lambda: FakeAgent)
    monkeypatch.setattr(harness, "_ensure_sigterm_handler", lambda: None)
    await harness.run(NeMoGymResponseCreateParamsNonStreaming(input="hi"))

    expected = (harness.hermes_home, "local", "17")
    assert seen == [expected, expected]
    assert tuple(os.environ[key] for key in ("HERMES_HOME", "TERMINAL_ENV", "TERMINAL_TIMEOUT")) == (
        "caller-home",
        "caller-backend",
        "caller-timeout",
    )


async def test_hermes_queues_overlapping_harnesses(monkeypatch) -> None:
    first = HermesHarness(AgentHarnessConfig(model=AgentModelConfig(model="first")))
    second = HermesHarness(AgentHarnessConfig(model=AgentModelConfig(model="second")))
    first_started = Event()
    release_first = Event()
    second_started = Event()
    homes = []

    class FakeAgent:
        def __init__(self, **kwargs) -> None:
            self._build_api_kwargs = lambda _messages: {}
            self.home = os.environ["HERMES_HOME"]

        def run_conversation(self, *args, **kwargs):
            homes.append(self.home)
            if self.home == first.hermes_home:
                first_started.set()
                release_first.wait(timeout=5)
            else:
                second_started.set()
            return {
                "completed": True,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"},
                ],
            }

    monkeypatch.setattr("nemo_gym.agents.hermes._load_ai_agent", lambda: FakeAgent)
    monkeypatch.setattr(first, "_ensure_sigterm_handler", lambda: None)
    monkeypatch.setattr(second, "_ensure_sigterm_handler", lambda: None)
    body = NeMoGymResponseCreateParamsNonStreaming(input="hi")

    first_task = asyncio.create_task(first.run(body))
    assert await asyncio.to_thread(first_started.wait, 1)
    second_task = asyncio.create_task(second.run(body))
    await asyncio.sleep(0.05)
    assert not second_started.is_set()

    release_first.set()
    await asyncio.gather(first_task, second_task)
    assert second_started.is_set()
    assert homes == [first.hermes_home, second.hermes_home]
