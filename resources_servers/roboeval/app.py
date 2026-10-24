# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RoboEval (CodeBotler) verifier for service-mobile-robot programs.

The model writes ``task_program()`` against a small robot API (``go_to``, ``ask``, ``say``, ``is_in_room``, ...). The
program runs in RoboEval's symbolic simulator once per world state defined for the task, and each run's trace of
robot actions is checked by that state's temporal-logic test. Reward is 1 only if every state passes, the criterion
behind RoboEval's pass@1.

RoboEval is not vendored: its task, simulator, and tracer modules are imported from a CodeBotler checkout, which is
``roboeval_root`` if set and otherwise fetched at a pinned commit on startup. Its license forbids using the benchmark
contents as ML training data, so this verifier is for evaluation only.

Failures of the verifier itself (simulator processes that cannot start, an unknown task, broken imports) are masked
with ``failure_kind="verifier_error"``. A program that crashes, loops, or fails a check is a valid measurement and is
not masked, as in RoboEval.
"""

import asyncio
import importlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.roboeval.setup_codebotler import ensure_codebotler


_LOGGER = logging.getLogger(__name__)


class RoboEvalResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    roboeval_root: Optional[str] = Field(
        default=None,
        description="CodeBotler checkout (https://github.com/ut-amrl/codebotler); fetched at a pinned commit if unset.",
    )
    # RoboEval's default is 1 s per simulation, which includes interpreter start-up; that misfires under concurrent
    # load, so the default here is larger. Programs that should run forever (waiting loops) still time out.
    simulation_timeout_s: float = Field(default=5.0, gt=0)
    max_concurrent_simulations: int = Field(default=16, ge=1)


class RoboEvalVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    task: str
    prompt_index: int = 0


class RoboEvalVerifyResponse(BaseVerifyResponse):
    task: str
    # False when the verifier failed and the sample is masked; required by harness conformance (TE-6).
    evaluation_completed: bool = True
    program: str = ""
    # One entry per world state: state index, passed, status (success/timeout/error), and error message.
    state_results: list[dict] = []


# RoboEval's simple_tracer keeps at most this much simulator output per run.
_MAX_OUTPUT_BYTES = 1_000_000


async def _drain(stream: asyncio.StreamReader, buffer: bytearray) -> None:
    while chunk := await stream.read(65536):
        if len(buffer) < _MAX_OUTPUT_BYTES:
            buffer.extend(chunk[: _MAX_OUTPUT_BYTES - len(buffer)])


def extract_program(text: str) -> str:
    """Return the ``task_program`` source from a chat answer.

    Takes the last fenced code block if there is one. Like CodeBotler's chat path, prepends the
    ``def task_program():`` header when the model returned only the body.
    """
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
    code = blocks[-1] if blocks else text
    code = code.strip("\n")
    if "def task_program" not in code:
        code = "def task_program():\n" + code
    return code


class RoboEvalVerifier:
    """Runs RoboEval simulations and temporal checks; holds no per-request state."""

    def __init__(self, config: RoboEvalResourcesServerConfig) -> None:
        self._root = ensure_codebotler(config.roboeval_root)
        self._timeout_s = config.simulation_timeout_s
        self._semaphore = asyncio.Semaphore(config.max_concurrent_simulations)
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))
        tracer = importlib.import_module("benchmark.simple_tracer")
        self._code_replace: Callable[[str], str] = tracer.code_replace
        self._parse_trace: Callable[[str], list] = tracer.construct_trace_element_from_str
        self._tests: dict[str, list[dict]] = {}

    def _task_tests(self, task: str) -> list[dict]:
        if task not in self._tests:
            if not re.fullmatch(r"[A-Za-z0-9_]+", task):
                raise ValueError(f"invalid RoboEval task name {task!r}")
            self._tests[task] = importlib.import_module(f"benchmark.tasks.{task}").tests
        return self._tests[task]

    async def _simulate(self, task: str, program: str, state_index: int) -> tuple[list, str, str]:
        """Run the program on one world state; returns (trace, status, error) like simple_tracer.run_simulation."""
        script = (
            "import sys\nimport time\n"
            "from benchmark.simulator import Robot\n"
            f"from benchmark.tasks.{task} import tests\n"
            f"robot = Robot(tests[{state_index}]['state'])\n"
            f"{self._code_replace(program)}\n"
        )
        async with self._semaphore:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                script,
                cwd=self._root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            # Drain into our own buffers so a timed-out run keeps the trace it printed before being killed.
            stdout, stderr = bytearray(), bytearray()
            readers = asyncio.gather(_drain(process.stdout, stdout), _drain(process.stderr, stderr))
            timed_out = False
            try:
                await asyncio.wait_for(process.wait(), timeout=self._timeout_s)
            except TimeoutError:
                timed_out = True
                try:
                    os.killpg(process.pid, 9)
                except ProcessLookupError:  # The program exited between the timeout and the kill.
                    pass
            await readers
            await process.wait()
        trace = self._parse_trace(stdout.decode(errors="replace"))
        if timed_out:
            return trace, "timeout", ""
        if process.returncode != 0:
            lines = stderr.decode(errors="replace").strip().splitlines()
            return [], "error", lines[-1] if lines else f"exit code {process.returncode}"
        return trace, "success", ""

    async def _check_state(self, task: str, program: str, state_index: int, test: dict) -> dict:
        trace, status, error = await self._simulate(task, program, state_index)
        passed = False
        if status == "error":
            error = f"SIM: {error}"
        elif status == "timeout" and not test["timeout"]:
            error = "SIM: program timed out but should terminate"
        else:
            try:
                passed = bool(test["test"](trace))
            except Exception as exc:  # Task checks are third-party code; a crash fails this state, as in RoboEval.
                error = f"EVAL: {exc!r}"
        return {"state": state_index, "passed": passed, "status": status, "error": error}

    async def verify(self, body: RoboEvalVerifyRequest) -> RoboEvalVerifyResponse:
        program = extract_program(body.response.output_text or "")
        try:
            tests = self._task_tests(body.task)
            state_results = await asyncio.gather(
                *(self._check_state(body.task, program, i, test) for i, test in enumerate(tests))
            )
        except Exception as exc:  # Request boundary: the verifier failed, so the reward is not a measurement.
            _LOGGER.exception("RoboEval verification failed for task %s", body.task)
            return RoboEvalVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                program=program,
                mask_sample=True,
                evaluation_completed=False,
                failure_kind="verifier_error",
                failure_reason=f"RoboEval verifier failed: {exc!r}"[:2000],
            )
        reward = float(all(result["passed"] for result in state_results))
        return RoboEvalVerifyResponse(
            **body.model_dump(), reward=reward, program=program, state_results=list(state_results)
        )


class RoboEvalResourcesServer(SimpleResourcesServer):
    ray_enabled = False

    config: RoboEvalResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._verifier = RoboEvalVerifier(self.config)

    async def verify(self, body: RoboEvalVerifyRequest) -> RoboEvalVerifyResponse:
        return await self._verifier.verify(body)

    def compute_metrics(self, tasks: list[list[dict]]) -> dict:
        """RoboEval pass@1: the mean reward over repeated samples, as ``pass@1[avg-of-k]/accuracy``."""
        return compute_pass_majority_metrics(tasks, score_fn=lambda r: {"accuracy": r["reward"]})[0]

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        key = {
            name: agent_metrics[name] for name in ("mean/input_tokens", "mean/output_tokens") if name in agent_metrics
        }
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
        return key


def _fixture_verifier() -> RoboEvalVerifier:
    config = RoboEvalResourcesServerConfig(
        host="", port=0, entrypoint="", name="", roboeval_root=os.environ.get("ROBOEVAL_ROOT")
    )
    return RoboEvalVerifier(config)


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_fixture_verifier,
    request_model=RoboEvalVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    RoboEvalResourcesServer.run_webserver()
