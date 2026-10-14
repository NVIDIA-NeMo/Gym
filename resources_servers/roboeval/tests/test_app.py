# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.verifier_fixture import exercise_verifier_fixture

from ..app import (
    VERIFIER_FIXTURE,
    RoboEvalResourcesServer,
    RoboEvalResourcesServerConfig,
    RoboEvalVerifier,
    RoboEvalVerifyRequest,
    extract_program,
)


needs_roboeval = pytest.mark.skipif(
    not os.environ.get("ROBOEVAL_ROOT"), reason="set ROBOEVAL_ROOT to a CodeBotler checkout"
)


class TestExtractProgram:
    def test_takes_last_fenced_block(self):
        text = "Plan:\n```python\nx = 1\n```\nFinal:\n```python\ndef task_program():\n    say('hi')\n```"
        assert extract_program(text) == "def task_program():\n    say('hi')"

    def test_adds_missing_header(self):
        assert extract_program("    say('hi')\n") == "def task_program():\n    say('hi')"

    def test_keeps_unfenced_program(self):
        assert extract_program("\ndef task_program():\n    say('hi')\n") == "def task_program():\n    say('hi')"


@needs_roboeval
def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
        )
    )


@needs_roboeval
def test_timed_out_run_keeps_its_partial_trace() -> None:
    config = RoboEvalResourcesServerConfig(
        host="", port=0, entrypoint="", name="", roboeval_root=os.environ["ROBOEVAL_ROOT"], simulation_timeout_s=1.0
    )
    verifier = RoboEvalVerifier(config)
    program = 'def task_program():\n    go_to("elevator")\n    while True:\n        is_in_room("person")\n        time.sleep(1)'
    trace, status, _ = asyncio.run(verifier._simulate("ElevatorTour", program, 1))
    assert status == "timeout"
    assert [element.element_name for element in trace[:2]] == ["GoTo", "CheckEntity"]


def _request(task: str, text: str) -> RoboEvalVerifyRequest:
    response = NeMoGymResponse(
        id="r",
        created_at=0,
        model="m",
        object="response",
        output=[
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )
    return RoboEvalVerifyRequest(responses_create_params={"input": "x"}, response=response, task=task)


@needs_roboeval
def test_simulator_failure_is_masked() -> None:
    config = RoboEvalResourcesServerConfig(
        host="", port=0, entrypoint="", name="", roboeval_root=os.environ["ROBOEVAL_ROOT"]
    )
    verifier = RoboEvalVerifier(config)

    async def _cannot_start(*args, **kwargs):
        raise OSError("Resource temporarily unavailable")

    verifier._simulate = _cannot_start
    result = asyncio.run(verifier.verify(_request("SayGoodDay", "say('hi')")))
    assert result.mask_sample is True
    assert result.evaluation_completed is False
    assert result.failure_kind == "verifier_error"
    assert "Resource temporarily unavailable" in result.failure_reason


@needs_roboeval
def test_program_crash_is_a_valid_measurement() -> None:
    config = RoboEvalResourcesServerConfig(
        host="", port=0, entrypoint="", name="", roboeval_root=os.environ["ROBOEVAL_ROOT"]
    )
    result = asyncio.run(RoboEvalVerifier(config).verify(_request("SayGoodDay", "raise ValueError('bad program')")))
    assert (result.reward, result.mask_sample, result.failure_kind) == (0.0, False, None)
    assert result.evaluation_completed is True


def test_key_metrics_report_pass_at_1_averaged_over_repeats() -> None:
    tasks = [[{"reward": 1.0}, {"reward": 0.0}], [{"reward": 1.0}, {"reward": 1.0}]]
    metrics = RoboEvalResourcesServer.compute_metrics(None, tasks)
    key = RoboEvalResourcesServer.get_key_metrics(None, metrics)
    assert key["pass@1[avg-of-2]/accuracy"] == 75.0


@needs_roboeval
def test_program_exiting_at_the_timeout_is_not_a_verifier_error(monkeypatch) -> None:
    config = RoboEvalResourcesServerConfig(
        host="", port=0, entrypoint="", name="", roboeval_root=os.environ["ROBOEVAL_ROOT"], simulation_timeout_s=1.0
    )
    verifier = RoboEvalVerifier(config)

    real_killpg = os.killpg

    def _exits_just_before_the_kill(pid, sig):
        real_killpg(pid, sig)
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(os, "killpg", _exits_just_before_the_kill)
    program = 'def task_program():\n    while True:\n        time.sleep(1)'
    _, status, _ = asyncio.run(verifier._simulate("ElevatorTour", program, 1))
    assert status == "timeout"
