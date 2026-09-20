# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import shlex
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from time import perf_counter, time
from traceback import format_exc
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import Request
from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_2 import Command
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.llms.base import BaseLLM, ContextLengthExceededError, LLMResponse
from harbor.models.agent.context import AgentContext
from harbor.models.metric.usage_info import UsageInfo
from harbor.utils.logger import logger as harbor_logger
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    TrajectoryRecord,
)
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    get_server_url,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)
from responses_api_agents.terminus_2_sandboxed_agent.observability import TerminusObservations


class Terminus2AgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_turns: int | None
    parser_name: str = "json"
    enable_summarize: bool
    proactive_summarization_threshold: int
    tmux_pane_width: int
    tmux_pane_height: int
    dump_trajectory: bool = False
    debug: bool = False
    model_context_limit: int
    model_output_limit: int | None
    interleaved_thinking: bool
    recover_stalled_interrupts: bool = False

    llm_request_timeout: int

    sandbox_provider: str
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    sandbox_timeout: float
    remote_tmux_binary_path: Optional[str]


class Terminus2AgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class Terminus2AgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class Terminus2AgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None, exclude_if=lambda value: value is None
    )

    ng_trajectory: Optional[TrajectoryRecord] = Field(default=None, exclude_if=lambda value: value is None)

    terminus2_completed: bool
    command_exec_times: List[float]
    model_call_times: List[float]
    average_command_exec_time: float
    average_model_call_time: float
    total_command_exec_time: float
    total_model_call_time: float
    command_exec_time_pct: float
    model_call_time_pct: float
    terminus2_time_taken: float
    model_calls_gt_10min: int
    usages: List[Optional[NeMoGymResponseUsage]]
    num_proactive_compactions: int
    num_compactions: int
    terminal_interrupt_drains: int = 0
    terminal_interrupt_drain_errors: int = 0
    error: Optional[str]


class NeMoGymSandboxEnvironment:
    """The Harbor environment surface used by Terminus 2, backed by AsyncSandbox."""

    def __init__(self, sandbox: AsyncSandbox, logs_dir: Path, session_id: str):
        self._sandbox = sandbox
        self.default_user = None
        self.trial_paths = SimpleNamespace(agent_dir=logs_dir)
        self.session_id = session_id

    async def exec(
        self,
        command: str,
        timeout_sec: float | None = None,
        user: str | int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **_: Any,
    ) -> Any:
        result = await self._sandbox.exec(command, timeout_s=timeout_sec, cwd=cwd, user=user, env=env)

        return SimpleNamespace(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            return_code=result.return_code,
        )

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        result = await self._sandbox.exec(f"test -d {json.dumps(path)}", user=user)
        return result.return_code == 0


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


class NeMoGymLLM(BaseLLM):
    """Responses-only Harbor LLM adapter backed by NeMo Gym's aiohttp client."""

    def __init__(
        self,
        client: NeMoGymAsyncOpenAI,
        model_name: str,
        model_context_limit: int,
        model_output_limit: int | None,
        llm_request_timeout: int,
        observations: TerminusObservations | None = None,
    ):
        super().__init__()
        self._client = client
        self.observations = observations
        self._model_name = model_name
        self._model_context_limit = model_context_limit
        self._model_output_limit = model_output_limit
        self._llm_request_timeout = llm_request_timeout
        self.trajectory: list[NeMoGymResponseOutputItem] = []
        self.usages: list[NeMoGymResponseUsage] = []
        self._times_spent = []
        self._last_input_items = []
        self._model_calls_gt_10min = 0
        self._num_compactions = 0

        self._is_compacting = False
        self._just_compacted = False

    @staticmethod
    def _input_items(message_history: list[dict[str, Any]], prompt: str) -> list[NeMoGymEasyInputMessage]:
        messages = [*message_history, {"role": "user", "content": prompt}]

        res = []
        for message in messages:
            if message.get("role") in ("user", "system"):
                res.append(
                    NeMoGymEasyInputMessage(role=message.get("role", "user"), content=message.get("content", ""))
                )
            elif message.get("role") == "assistant":
                if message.get("reasoning_content"):
                    res.append(
                        NeMoGymResponseReasoningItem(
                            id="",
                            summary=[NeMoGymSummary(text=message.get("reasoning_content"), type="summary_text")],
                            type="reasoning",
                        )
                    )
                res.append(
                    NeMoGymResponseOutputMessage(
                        id="",
                        content=[NeMoGymResponseOutputText(annotations=[], text=message.get("content", ""))],
                    )
                )
            else:
                raise NotImplementedError(f"Found an unknown role in messages: {messages}")

        return res

    @staticmethod
    def _response_text(response: NeMoGymResponse) -> tuple[str, str | None]:
        content: list[str] = []
        reasoning: list[str] = []
        for item in response.output:
            if isinstance(item, NeMoGymResponseOutputMessage):
                content.extend(part.text for part in item.content if isinstance(part, NeMoGymResponseOutputText))
            elif isinstance(item, NeMoGymResponseReasoningItem):
                reasoning.extend(summary.text for summary in item.summary)
        return "".join(content), "\n".join(reasoning) or None

    async def call(self, prompt: str, **kwargs: Any) -> LLMResponse:
        message_history = kwargs.pop("message_history", [])
        kwargs.pop("previous_response_id", None)
        kwargs.pop("logging_path", None)
        if kwargs:
            raise NotImplementedError(f"NeMoGymLLM does not support call options: {sorted(kwargs)}")

        input_items = self._input_items(message_history, prompt)
        request_input = [item.model_dump(mode="json", exclude_none=True) for item in input_items]
        observed_input = deepcopy(request_input) if self.observations is not None else None
        response = None
        start_time = perf_counter()
        max_attempts = 10  # Harbor does 3 by default and Litellm does 3 by default. Hardcode 10 attempts for now.
        for _ in range(max_attempts):
            try:
                async with asyncio.timeout(delay=self._llm_request_timeout):
                    response = NeMoGymResponse.model_validate(
                        await self._client.create_response(
                            model=self._model_name,
                            input=request_input,
                        )
                    )
                    break
            except TimeoutError:
                self._model_calls_gt_10min += 1
                if self.observations is not None:
                    self.observations.gap("model_attempt_without_response", "TimeoutError")
            except BaseException as exc:
                if self.observations is not None:
                    self.observations.gap("model_attempt_without_response", type(exc).__name__)
                raise

        self._times_spent.append(perf_counter() - start_time)
        if not response:
            raise TimeoutError(f"Failed to query model endpoint due to timeouts after {max_attempts} attempts!")

        observed = (
            self.observations.record_response(observed_input, response, time())
            if self.observations is not None
            else None
        )

        if self._is_compacting:
            self.trajectory.extend([input_items[-1], *response.output])
            self._just_compacted = True
        elif self._just_compacted:
            self.trajectory.extend([*input_items, *response.output])
            self._just_compacted = False
        else:
            self.trajectory.extend([input_items[-1], *response.output])

        self.usages.append(response.usage)
        self._last_input_items = input_items.copy()

        usage = response.usage
        usage_info = None
        if usage is not None:
            usage_info = UsageInfo(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                cache_tokens=usage.input_tokens_details.cached_tokens or 0,
                cost_usd=0.0,
            )
        content, reasoning_content = self._response_text(response)

        # @bxyu-nvidia: Gym will return an empty model response when context length is exceeded
        if not (content or reasoning_content) or response.incomplete_details:
            self._num_compactions += 1
            if self.observations is not None:
                self.observations.gap("model_response_rejected", response.id)
            raise ContextLengthExceededError

        result = LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            model_name=response.model,
            usage=usage_info,
            response_id=response.id,
        )
        if observed is not None:
            observed.harbor_response = result
        return result

    def get_model_context_limit(self) -> int:
        return self._model_context_limit

    def get_model_output_limit(self) -> int | None:
        return self._model_output_limit


class NeMoGymTerminus2(Terminus2):
    """Terminus 2 with NeMo Gym model calls and optional Harbor file trajectories."""

    def __init__(
        self,
        *args: Any,
        llm: NeMoGymLLM,
        dump_trajectory: bool,
        recover_stalled_interrupts: bool = False,
        **kwargs: Any,
    ):
        self._nemo_gym_llm = llm
        self._dump_trajectory_enabled = dump_trajectory
        self._recover_stalled_interrupts = recover_stalled_interrupts
        self._terminal_interrupt_drains = 0
        self._terminal_interrupt_drain_errors = 0
        self._times_spent = []
        self._num_proactive_compactions = 0
        self._completed_command_batches = 0
        self._is_check_proactive_summarization = False
        super().__init__(*args, **kwargs)

    def _init_llm(self, *args: Any, **kwargs: Any) -> BaseLLM:
        return self._nemo_gym_llm

    def _dump_trajectory_with_continuation_index(self, continuation_index: int) -> None:
        if self._dump_trajectory_enabled:
            super()._dump_trajectory_with_continuation_index(continuation_index)

    async def _handle_llm_interaction(self, *args, **kwargs):
        observations = self._nemo_gym_llm.observations
        if observations is None:
            return await super()._handle_llm_interaction(*args, **kwargs)
        observations.begin_decision()
        try:
            return await super()._handle_llm_interaction(*args, **kwargs)
        finally:
            # Record before terminal commands execute, including when parsing raises.
            observations.finish_decision(self._completed_command_batches)

    async def _query_llm(self, *args, **kwargs):
        response = await super()._query_llm(*args, **kwargs)
        if self._nemo_gym_llm.observations is not None:
            self._nemo_gym_llm.observations.decision_response = response
        return response

    async def _execute_commands(self, commands: list[Command], session: TmuxSession) -> tuple[bool, str]:
        start_time = perf_counter()
        if self._recover_stalled_interrupts:
            res = await self._execute_commands_with_interrupt_recovery(commands, session)
        else:
            res = await super()._execute_commands(commands, session)
        if commands:
            self._completed_command_batches += 1
        self._times_spent.append(perf_counter() - start_time)

        return res

    async def _recover_stalled_interrupt(self, session: TmuxSession) -> None:
        # A full canonical input queue can block tmux's queued Ctrl+C. Reading the
        # pending input lets that key reach the terminal's normal signal handler.
        # Use a separate exec channel; never enqueue recovery behind the blocked key.
        command = """
pane_tty=$(tmux display-message -p -t SESSION '#{pane_tty}') || exit 1
terminal_modes=$(LC_ALL=C stty -F "$pane_tty" -a) || exit 1
case "$terminal_modes" in
    *-icanon*|*-isig*) echo skipped; exit 0 ;;
esac
case "$terminal_modes" in
    *'intr = ^C;'*) ;;
    *) echo skipped; exit 0 ;;
esac
case "$terminal_modes" in
    *-noflsh*) ;;
    *) echo skipped; exit 0 ;;
esac
timeout --kill-after=1 0.5 dd if="$pane_tty" of=/dev/null bs=4096 status=none
drain_status=$?
case "$drain_status" in
    0|124) echo drained ;;
    *) exit "$drain_status" ;;
esac
""".replace("SESSION", shlex.quote(session._session_name))
        try:
            result = await session.environment.exec(command, timeout_sec=5, user=session._user)
        except Exception:
            # Recovery is best effort. Preserve the ordinary terminal observation
            # if this separate control operation fails; cancellation still propagates.
            self.logger.warning("Terminal interrupt input drain failed", exc_info=True)
            self._terminal_interrupt_drain_errors += 1
            return
        if result.return_code != 0:
            self._terminal_interrupt_drain_errors += 1
            self.logger.warning("Terminal interrupt input drain failed: %s", result.stderr)
        else:
            if result.stdout.strip() == "drained":
                self._terminal_interrupt_drains += 1
            self.logger.info("Terminal interrupt input drain: %s", result.stdout.strip())

    async def _execute_commands_with_interrupt_recovery(
        self, commands: list[Command], session: TmuxSession
    ) -> tuple[bool, str]:
        # Keep Harbor's batch ordering, timeout feedback, and single observation.
        # Recover before the next command in the batch can enter the input queue.
        for command in commands:
            try:
                await session.send_keys(command.keystrokes, block=False, min_timeout_sec=command.duration_sec)
            except TimeoutError:
                return True, self._timeout_template.format(
                    timeout_sec=command.duration_sec,
                    command=command.keystrokes,
                    terminal_state=self._limit_output_length(await session.get_incremental_output()),
                )
            if command.keystrokes == "C-c":
                await self._recover_stalled_interrupt(session)
        return False, self._limit_output_length(await session.get_incremental_output())

    def _count_total_tokens(self, *args, **kwargs):
        if self._is_check_proactive_summarization and self._nemo_gym_llm.usages:
            return self._nemo_gym_llm.usages[-1].total_tokens
        return super()._count_total_tokens(*args, **kwargs)

    async def _check_proactive_summarization(self, *args, **kwargs):
        self._is_check_proactive_summarization = True
        try:
            res = await super()._check_proactive_summarization(*args, **kwargs)
        finally:
            self._is_check_proactive_summarization = False
        if res:
            self._num_proactive_compactions += 1
        return res

    async def _summarize(self, *args, **kwargs):
        self._nemo_gym_llm._is_compacting = True
        observations = self._nemo_gym_llm.observations
        if observations is not None:
            observations.gap("compaction_outside_main_turns")
            observations.compaction = ContextCompactionObservation(
                invocation_id=observations.invocation_id, observed_at=time(), trigger="harbor_summarize"
            )
        try:
            res = await super()._summarize(*args, **kwargs)
            if observations is not None:
                observations.compaction.outcome = "completed"
            return res
        except asyncio.CancelledError:
            if observations is not None:
                observations.compaction.outcome = "aborted"
            raise
        except BaseException:
            if observations is not None:
                observations.compaction.outcome = "failed"
            raise
        finally:
            self._nemo_gym_llm._is_compacting = False
            if observations is not None:
                observations.compactions.append(observations.compaction)
                observations.compaction = None


class Terminus2Agent(SimpleResponsesAPIAgent):
    config: Terminus2AgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_sandboxes: dict[str, AsyncSandbox] = {}

        if not self.config.debug:
            harbor_logger.setLevel(logging.WARNING)

    async def _connect_sandbox(self, sandbox_id: str) -> AsyncSandbox:
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, get_global_config_dict()))
        sandbox = await AsyncSandbox.connect({"sandbox_id": sandbox_id}, provider=provider)
        return sandbox

    async def _execute(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming,
        sandbox: AsyncSandbox,
    ) -> Tuple[NeMoGymResponse, Dict[str, Any]]:
        start_time = perf_counter()
        instruction = _instruction(body.input)
        run_body = await request.json()
        rollout_id = self.rollout_id_from_run(run_body)
        invocation_id = f"terminus2_{uuid4().hex}" if rollout_id is not None else None
        observations = None
        if invocation_id is not None:
            task_id = next(
                (
                    str(run_body[key])
                    for key in ("task_id", "problem_id", "instance_id")
                    if run_body.get(key) is not None
                ),
                str(run_body.get("_ng_task_index", rollout_id)),
            )
            observations = TerminusObservations(invocation_id, task_id, rollout_id, self.config.model_server)

        model_base_url = (
            self.base_url_for_run(base_url=get_server_url(self.config.model_server.name), body=run_body) + "/v1"
        )
        llm = NeMoGymLLM(
            client=NeMoGymAsyncOpenAI(
                base_url=model_base_url,
                api_key="dummy",
                internal=True,
                default_headers={"x-session-id": invocation_id} if invocation_id is not None else {},
            ),
            model_name=self.config.model_server.name,
            model_context_limit=self.config.model_context_limit,
            model_output_limit=self.config.model_output_limit,
            llm_request_timeout=self.config.llm_request_timeout,
            observations=observations,
        )

        with tempfile.TemporaryDirectory(prefix="nemo-gym-terminus-2-") as log_dir:
            environment = NeMoGymSandboxEnvironment(sandbox, Path(log_dir), request.session[SESSION_ID_KEY])
            context = AgentContext()
            agent = NeMoGymTerminus2(
                logs_dir=Path(log_dir),
                model_name=self.config.model_server.name,
                max_turns=self.config.max_turns,
                parser_name=self.config.parser_name,
                enable_summarize=self.config.enable_summarize,
                proactive_summarization_threshold=self.config.proactive_summarization_threshold,
                tmux_pane_width=self.config.tmux_pane_width,
                tmux_pane_height=self.config.tmux_pane_height,
                record_terminal_session=False,
                llm=llm,
                dump_trajectory=self.config.dump_trajectory,
                interleaved_thinking=self.config.interleaved_thinking,
                recover_stalled_interrupts=self.config.recover_stalled_interrupts,
            )

            await environment.exec("mkdir -p /logs/agent", user="root")
            if self.config.remote_tmux_binary_path:
                # We add the /usr/local/bin path at the end to not supersede and pre-existing orderings.
                tmux_install_result = await sandbox.exec(
                    f"""mkdir -p /usr/local/bin \
&& cp {self.config.remote_tmux_binary_path} /usr/local/bin/tmux \
&& chmod +x /usr/local/bin/tmux \
&& export PATH=$PATH:/usr/local/bin \
&& tmux -V""",
                )
                assert tmux_install_result.return_code == 0, tmux_install_result
            else:
                print(
                    "Downloading and installing tmux in the sandbox. Please consider mounting or uploading the appropriate tmux binary instead!",
                    file=sys.stderr,
                )
            await agent.setup(environment)

            try:
                async with asyncio.timeout(self.config.sandbox_timeout):
                    await agent.run(instruction, environment, context)
                terminus2_completed = True
                error = None
                invocation_status = "completed"
                error_type = None
            except TimeoutError:
                terminus2_completed = False
                error = format_exc()
                invocation_status = "incomplete"
                error_type = "TimeoutError"
            except asyncio.CancelledError:
                terminus2_completed = False
                error = format_exc()
                invocation_status = "incomplete"
                error_type = "CancelledError"
            except BaseException as exc:
                terminus2_completed = False
                error = format_exc()
                invocation_status = "failed"
                error_type = type(exc).__name__
                print(f"Hit exception while running Terminus2: {format_exc()}", file=sys.stderr)
            finally:
                pass

        usage = NeMoGymResponseUsage(
            input_tokens=context.n_input_tokens or 0,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=context.n_cache_tokens or 0),
            output_tokens=context.n_output_tokens or 0,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=(context.n_input_tokens or 0) + (context.n_output_tokens or 0),
        )
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model_server.name,
            object="response",
            output=llm.trajectory,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=usage,
        )

        total_time = perf_counter() - start_time
        total_command_exec_time = sum(agent._times_spent)
        total_model_call_time = sum(llm._times_spent)
        metrics = {
            "terminus2_completed": terminus2_completed,
            "command_exec_times": agent._times_spent,
            "model_call_times": llm._times_spent,
            "average_command_exec_time": total_command_exec_time / max(len(agent._times_spent), 1),
            "average_model_call_time": total_model_call_time / max(len(llm._times_spent), 1),
            "total_command_exec_time": total_command_exec_time,
            "total_model_call_time": total_model_call_time,
            "command_exec_time_pct": 100 * total_command_exec_time / total_time,
            "model_call_time_pct": 100 * total_model_call_time / total_time,
            "terminus2_time_taken": total_time,
            "model_calls_gt_10min": llm._model_calls_gt_10min,
            "num_proactive_compactions": agent._num_proactive_compactions,
            "num_compactions": llm._num_compactions,
            "terminal_interrupt_drains": agent._terminal_interrupt_drains,
            "terminal_interrupt_drain_errors": agent._terminal_interrupt_drain_errors,
            "error": error,
            "usages": llm.usages,
        }
        if observations is not None:
            metrics["ng_trajectory"] = observations.finish()
            metrics["ng_agent_observations"] = AgentObservationBundle(
                source="terminus2",
                records=[
                    AgentInvocation(
                        invocation_id=invocation_id,
                        status=invocation_status,
                        duration_ms=total_time * 1000,
                        error_type=error_type,
                    ),
                    *observations.compactions,
                ],
            )
        return response, metrics

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        session_key = request.session[SESSION_ID_KEY]
        sandbox = self._session_sandboxes[session_key]
        response, _ = await self._execute(request, body, sandbox)
        return response

    async def run(self, request: Request, body: Terminus2AgentRunRequest) -> Terminus2AgentVerifyResponse:
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

        sandbox_id = seed_session_result["sandbox_handle"]

        sandbox = await self._connect_sandbox(sandbox_id)
        session_key = request.session[SESSION_ID_KEY]
        self._session_sandboxes[session_key] = sandbox

        response, metrics = await self._execute(request, body.responses_create_params, sandbox)

        verification = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": response.model_dump()},
            cookies=cookies,
        )
        await raise_for_status(verification)

        self._session_sandboxes.pop(session_key)
        try:
            await sandbox.stop()
        except:
            print("Failed to stop sandbox", format_exc(), file=sys.stderr)

        result = await get_response_json(verification)
        result.update(metrics)
        return Terminus2AgentVerifyResponse.model_validate(result)


if __name__ == "__main__":
    Terminus2Agent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = Terminus2Agent.run_webserver()  # noqa: F401
