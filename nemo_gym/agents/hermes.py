# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import atexit
import logging
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from time import time
from typing import Any, Callable, ClassVar, Iterator, Optional
from uuid import uuid4

from nemo_gym.agents.config import AgentHarnessConfig
from nemo_gym.agents.hermes_observability import HermesAgentObserver
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentObservationBundle,
    ObservationGap,
)


def _trajectory_to_output_items(messages, n_input):
    output_items = []
    for item in messages[n_input:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "") or ""
        if isinstance(content, list):
            content = "".join(c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "") for c in content)
        if role == "assistant":
            reasoning_text = item.get("reasoning") or ""
            if reasoning_text:
                content = ResponsesConverter._parse_think_tags(content)[1]
                output_items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rsn-{len(output_items)}",
                        summary=[NeMoGymSummary(type="summary_text", text=reasoning_text)],
                        type="reasoning",
                    )
                )
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg-{len(output_items)}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=content, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=item.get("prompt_token_ids") or [],
                    generation_token_ids=item.get("generation_token_ids") or [],
                    generation_log_probs=item.get("generation_log_probs") or [],
                    routed_experts=item.get("routed_experts"),
                )
            )
            for tc in item.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if not fn:
                    continue
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=fn.get("arguments", ""),
                        call_id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        type="function_call",
                        id=tc.get("id"),
                        status="completed",
                    )
                )
        elif role == "tool":
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=item.get("tool_call_id", ""),
                    output=content,
                    status="completed",
                )
            )
    return output_items


LOG = logging.getLogger(__name__)

_HERMES_ENV_KEYS = ("HERMES_HOME", "TERMINAL_ENV", "TERMINAL_TIMEOUT")
_HERMES_RUN_LOCK = asyncio.Lock()


@contextmanager
def _hermes_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in _HERMES_ENV_KEYS}
    os.environ.update(values)

    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _load_ai_agent():
    from run_agent import AIAgent  # from hermes-agent on path  # pyright: ignore[reportMissingImports]

    return AIAgent


# Ray can close sys.stderr mid-request.
class _SafeStderrHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = sys.__stderr__
            if stream is None:
                return
            stream.write(msg + "\n")
            stream.flush()
        except Exception:
            pass


if not LOG.handlers:
    LOG.addHandler(_SafeStderrHandler(level=logging.WARNING))


def _split_input_to_user_and_history(input_items) -> tuple[str, list[dict], Optional[str]]:
    items = list(input_items)
    system_message: Optional[str] = None
    if items:
        first = items[0]
        first_role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        first_content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
        if first_role == "system":
            if isinstance(first_content, list):
                first_content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in first_content
                )
            system_message = first_content or ""
            items = items[1:]

    user_message = ""
    history: list[dict] = []
    for idx, item in enumerate(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        if isinstance(content, list):
            content = "".join((p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content)
        content = content or ""
        if idx == len(items) - 1 and role == "user":
            user_message = content
        else:
            history.append({"role": role, "content": content})
    return user_message, history, system_message


class HermesHarness:
    active_agents: ClassVar[set[Any]] = set()
    interrupted_agents: ClassVar[set[int]] = set()
    _sigterm_loop: ClassVar[asyncio.AbstractEventLoop | None] = None

    def __init__(self, config: AgentHarnessConfig):
        if config.model.model is None:
            raise ValueError("HermesHarness requires a model")
        self.config = config
        terminal_backend = self.config.settings.get("terminal_backend", "local")
        self.hermes_home = tempfile.mkdtemp(prefix="hermes_agent_")
        atexit.register(shutil.rmtree, self.hermes_home, True)
        with open(os.path.join(self.hermes_home, "config.yaml"), "w") as config_file:
            config_file.write(self._build_config())
        self._environment = {
            "HERMES_HOME": self.hermes_home,
            "TERMINAL_ENV": terminal_backend,
            "TERMINAL_TIMEOUT": str(self.config.settings.get("terminal_timeout", 180)),
        }

    @property
    def sigterm_installed(self) -> bool:
        try:
            return self._sigterm_loop is asyncio.get_event_loop()
        except RuntimeError:
            return False

    def _ensure_sigterm_handler(self) -> None:
        # Event loops replace existing signal handlers, so one dispatcher owns SIGTERM.
        import signal

        loop = asyncio.get_event_loop()
        if self._sigterm_loop is loop:
            return

        def _dispatch():
            for ag in list(self.active_agents):
                self.interrupted_agents.add(id(ag))
                if hasattr(ag, "interrupt"):
                    ag.interrupt("timeout")

        try:
            loop.add_signal_handler(signal.SIGTERM, _dispatch)
            HermesHarness._sigterm_loop = loop
        except (NotImplementedError, OSError, RuntimeError):
            pass

    def _build_config(self) -> str:
        import yaml

        config: dict[str, Any] = {
            "model": self._model_name(),
            "provider": "auto",
            "toolsets": ["hermes-cli"],
            "agent": {"max_turns": self.config.max_turns or 90},
            "memory": {
                "memory_enabled": False,
                "user_profile_enabled": False,
            },
            "compression": {
                "enabled": self.config.settings.get("compression_enabled", True),
                "threshold": self.config.settings.get("compression_threshold", 0.85),
            },
            "terminal": {
                "backend": self.config.settings.get("terminal_backend", "local"),
                "timeout": self.config.settings.get("terminal_timeout", 180),
            },
            "delegation": {
                "max_iterations": self.config.settings.get("delegation_max_iterations", 50),
            },
            "checkpoints": {
                "enabled": self.config.settings.get("checkpoints_enabled", False),
            },
        }
        return yaml.dump(config, default_flow_style=False)

    def _model_name(self) -> str:
        return self.config.model.model or ""

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_base_url: Optional[str] = None,
        model_ref: Optional[ModelServerRef] = None,
        observation_collector: Optional[Callable[[AgentObservationBundle], None]] = None,
    ) -> NeMoGymResponse:
        # Hermes uses a process-global interrupt flag. Parallel runs can cancel each other.
        async with _HERMES_RUN_LOCK:
            with _hermes_environment(self._environment):
                return await self._create_response_in_environment(
                    body,
                    model_base_url=model_base_url,
                    model_ref=model_ref,
                    observation_collector=observation_collector,
                )

    async def _create_response_in_environment(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_base_url: Optional[str] = None,
        model_ref: Optional[ModelServerRef] = None,
        observation_collector: Optional[Callable[[AgentObservationBundle], None]] = None,
    ) -> NeMoGymResponse:
        AIAgent = _load_ai_agent()

        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, history, input_system = _split_input_to_user_and_history(body.input)
        system_message = self.config.system_prompt or input_system

        base_url = model_base_url or self.config.model.base_url or ""
        model_name = self._model_name()

        agent = AIAgent(
            base_url=base_url,
            api_key=self.config.model.api_key or os.environ.get("OPENAI_API_KEY", "gym"),  # pragma: allowlist secret
            model=model_name,
            use_streaming=False,
            temperature=self.config.model.settings.get("temperature"),
            insert_reasoning=True,
            max_iterations=self.config.max_turns or 90,
            max_tokens=self.config.model.settings.get("max_tokens"),
            enabled_toolsets=self.config.settings.get("enabled_toolsets"),
            disabled_toolsets=self.config.settings.get("disabled_toolsets"),
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            persist_session=False,
            save_trajectories=False,
        )
        _original_build_api_kwargs = agent._build_api_kwargs

        def _patched_build_api_kwargs(api_messages):
            kw = _original_build_api_kwargs(api_messages)
            if not self.config.settings.get("chat_template_kwargs_enabled", True):
                return kw
            ctk = kw.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})
            ctk.setdefault("enable_thinking", True)
            ctk["truncate_history_thinking"] = False
            return kw

        agent._build_api_kwargs = _patched_build_api_kwargs
        observer = None
        if observation_collector is not None:
            try:
                observer = HermesAgentObserver(model_ref=model_ref).instrument(agent)
            except Exception:
                LOG.exception("failed to initialize Hermes observability")

        self._ensure_sigterm_handler()
        agent_id = id(agent)
        self.active_agents.add(agent)

        result = None
        agent_error: Optional[BaseException] = None
        interrupted_by_dispatch = False
        try:
            conversation = asyncio.create_task(
                asyncio.to_thread(
                    agent.run_conversation,
                    user_message,
                    system_message,
                    history,
                )
            )
            done, _ = await asyncio.wait({conversation}, timeout=self.config.timeout_seconds)
            if not done:
                self.interrupted_agents.add(agent_id)
                agent.interrupt("timeout")
            result = await conversation
        except BaseException as exc:
            agent_error = exc
            raise
        finally:
            self.active_agents.discard(agent)
            interrupted_by_dispatch = agent_id in self.interrupted_agents
            self.interrupted_agents.discard(agent_id)
            if observation_collector is not None:
                try:
                    observations = (
                        observer.finish(result, error=agent_error)
                        if observer is not None
                        else AgentObservationBundle(
                            source="hermes",
                            gaps=[ObservationGap(code="observation_capture_failed")],
                        )
                    )
                except Exception:
                    LOG.exception("failed to finish Hermes observability")
                    observations = AgentObservationBundle(
                        source="hermes",
                        gaps=[ObservationGap(code="observation_capture_failed")],
                    )
                try:
                    observation_collector(observations)
                except Exception:
                    LOG.exception("failed to return Hermes observations")

        messages = result.get("messages") or []
        n_input = len(history) + 1

        output_items = _trajectory_to_output_items(messages, n_input)

        has_assistant_message = any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        )
        if not has_assistant_message:
            LOG.warning(
                "Hermes agent ended without an assistant message. Padding empty assistant message. This should not happen often, investigate: error=%r",
                result.get("error"),
            )
            last_valid = next(
                (
                    m
                    for m in reversed(messages)
                    if isinstance(m, dict) and m.get("role") == "assistant" and m.get("generation_token_ids")
                ),
                None,
            )
            pti = last_valid["prompt_token_ids"] if last_valid else [0]
            gti = last_valid["generation_token_ids"] if last_valid else [0]
            glp = (last_valid.get("generation_log_probs") if last_valid else None) or [0.0]
            routed_experts = last_valid.get("routed_experts") if last_valid else None
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text=result.get("error") or "", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=pti,
                    generation_token_ids=gti,
                    generation_log_probs=glp,
                    routed_experts=routed_experts,
                )
            )

        # Preserve incomplete status so evaluators can mask exhausted runs.
        agent_completed = bool(result.get("completed", True))

        was_interrupted = bool(result.get("interrupted")) or interrupted_by_dispatch

        harness_error = result.get("error")
        agent_failed = bool(harness_error) or bool(result.get("failed"))
        metadata: dict[str, str] = {
            "interrupted": "true" if was_interrupted else "false",
            "failed": "true" if result.get("failed") else "false",
            "partial": "true" if result.get("partial") else "false",
        }
        if isinstance(result.get("api_calls"), int):
            metadata["turns"] = str(result["api_calls"])
        if harness_error:
            metadata["hermes_error"] = str(harness_error)[:2000]

        # ResponseError restricts code values, so preserve details in the message.
        response_error = None
        if harness_error:
            from openai.types.responses import ResponseError  # pyright: ignore[reportMissingImports]

            response_error = ResponseError(code="server_error", message=str(harness_error)[:2000])

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            status="failed" if agent_failed else ("completed" if agent_completed else "incomplete"),
            error=response_error,
            metadata=metadata,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=0,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=0,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=0,
            ),
        )

    async def run(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_base_url: Optional[str] = None,
    ) -> NeMoGymResponse:
        return await self._create_response(body, model_base_url=model_base_url)

    async def run_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_base_url: Optional[str] = None,
        model_ref: Optional[ModelServerRef] = None,
    ) -> AgentEpisode:
        observations: Optional[AgentObservationBundle] = None

        def collect(bundle: AgentObservationBundle) -> None:
            nonlocal observations
            observations = bundle

        response = await self._create_response(
            body,
            model_base_url=model_base_url,
            model_ref=model_ref,
            observation_collector=collect,
        )
        if observations is None:
            observations = AgentObservationBundle(
                source="hermes",
                gaps=[ObservationGap(code="observation_capture_failed")],
            )
        observations.gaps.append(
            ObservationGap(
                code=(
                    "no_sandbox_runtime"
                    if self.config.settings.get("terminal_backend", "local") == "local"
                    else "sandbox_observation_unavailable"
                ),
                detail=(
                    None
                    if self.config.settings.get("terminal_backend", "local") == "local"
                    else f"terminal_backend={self.config.settings.get('terminal_backend')}"
                ),
            )
        )
        return AgentEpisode(response=response, observations=observations)
