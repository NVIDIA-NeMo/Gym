# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
import json
import os
import shutil
import signal
import subprocess
import traceback
from pathlib import Path
from typing import Any, AsyncGenerator

import ssa
from omegaconf import OmegaConf
from ssa.agent import StrandsResolverAgent
from ssa.callbacks.throttling import ThrottlingCallback
from ssa.conversation_manager.conversation_manager import AdaptiveConversationManager
from ssa.environments import LocalEnvironment, create_environment
from ssa.hooks import initialize_hooks
from ssa.metrics import MetricsCollector
from ssa.models.openai import SROpenAIModel
from ssa.prompts.prompt_gen import PromptGenerator
from ssa.tools import load_tools
from strands.handlers.callback_handler import CompositeCallbackHandler
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec


importlib.import_module("ssa.utils.monkey_patch")


class QuietCallback:
    def __call__(self, **kwargs: Any) -> None:
        pass


class GymOpenAIModel(SROpenAIModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.training_turns: list[dict[str, Any]] = []

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        request = self.format_request(
            messages,
            tool_specs,
            system_prompt,
            tool_choice,
            system_prompt_content=system_prompt_content,
            **kwargs,
        )
        request["stream"] = False
        request.pop("stream_options", None)
        async with self._get_client() as client:
            response = await client.chat.completions.create(**request)
        self.training_turns.append(_training_fields(response))
        for event in self._format_non_streaming_response(response):
            yield event


def _dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _token_ids(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    result = []
    for token in value:
        if isinstance(token, str) and token.startswith("token_id:"):
            token = token.removeprefix("token_id:")
        try:
            result.append(int(token))
        except (TypeError, ValueError):
            return None
    return result


def _training_fields(response: Any) -> dict[str, Any]:
    response_data = _dump(response)
    choices = response_data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    prompt_ids = _token_ids(message.get("prompt_token_ids") or response_data.get("prompt_token_ids"))
    generation_ids = _token_ids(message.get("generation_token_ids"))
    log_probs = message.get("generation_log_probs")
    if log_probs is None:
        content_log_probs = (choice.get("logprobs") or {}).get("content") or []
        log_probs = [item.get("logprob") for item in content_log_probs]
    if (
        not prompt_ids
        or not generation_ids
        or not isinstance(log_probs, list)
        or len(log_probs) != len(generation_ids)
    ):
        return {}
    try:
        generation_log_probs = [float(value) for value in log_probs]
    except (TypeError, ValueError):
        return {}
    fields = {
        "prompt_token_ids": prompt_ids,
        "generation_token_ids": generation_ids,
        "generation_log_probs": generation_log_probs,
    }
    if message.get("routed_experts") is not None:
        fields["routed_experts"] = message["routed_experts"]
    return fields


def _portable_execute_bash(
    environment: LocalEnvironment,
    command: str,
    workdir: str | None = None,
    timeout: float | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    timeout = environment.timeout if timeout is None else timeout
    process = subprocess.Popen(
        ["bash", "-c", command],
        cwd=workdir or environment.workdir,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        return {
            "command": command,
            "status": "error",
            "exit_code": 124,
            "output": output,
            "error": "",
        }
    return {
        "command": command,
        "status": "success" if process.returncode == 0 else "error",
        "exit_code": process.returncode,
        "output": output,
        "error": "" if process.returncode == 0 else output,
    }


def _build_config(payload: dict[str, Any], output_dir: Path):
    cfg = OmegaConf.load(Path(ssa.__file__).parent / "configs" / "default.yaml")
    cfg.max_llm_iterations = payload["max_turns"]
    cfg.env.env_type = "local"
    cfg.env.local.workdir = payload["work_dir"]
    cfg.env.timeout = payload["shell_timeout"]
    cfg.dataset.name = "gym"
    cfg.dataset.identifier = payload["rollout_id"]
    cfg.dataset.issue_description = payload["instruction"]
    cfg.agent.model = payload["model"]
    cfg.agent.prompt_tag = payload["prompt_tag"]
    cfg.agent.invoker = "openai"
    params: dict[str, Any] = {"max_tokens": payload["max_output_tokens"]}
    if payload.get("temperature") is not None:
        params["temperature"] = payload["temperature"]
    cfg.agent.invoker_params = OmegaConf.create(
        {
            **params,
            "use_responses_api": False,
            "cache_client": False,
            "client_args": {
                "base_url": payload["model_base_url"],
                "api_key": payload["api_key"],
                "timeout": payload["model_timeout"],
                "max_retries": 0,
            },
        }
    )
    cfg.agent.tools = OmegaConf.create({name: None for name in payload["tools"]})
    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    (hydra_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    return cfg


def _prompts(cfg, payload: dict[str, Any]) -> tuple[str, str]:
    prompt_generator = PromptGenerator(base_dir=str(Path(ssa.__file__).parent / "prompts"))
    native_system = prompt_generator.get_system_prompt(
        cfg.agent.agent_id,
        prompt_tag=cfg.agent.prompt_tag,
        project_path=payload["work_dir"],
    )
    extra_system = payload.get("system_prompt")
    system_prompt = f"{native_system}\n\n{extra_system}" if extra_system else native_system
    if not payload["native_user_prompt"]:
        return system_prompt, payload["instruction"]
    user_prompt = prompt_generator.get_user_prompt(
        cfg.agent.agent_id,
        prompt_tag=cfg.agent.prompt_tag,
        project_path=payload["work_dir"],
        git_issue=payload["instruction"],
    )
    return system_prompt, user_prompt


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _build_config(payload, output_dir)
    system_prompt, user_prompt = _prompts(cfg, payload)
    conversation_manager = AdaptiveConversationManager(
        window_size=payload["conversation_window"],
        should_truncate_results=True,
        per_turn=True,
    )
    callbacks = CompositeCallbackHandler(ThrottlingCallback(), QuietCallback())
    hooks = initialize_hooks(cfg, str(output_dir))
    invoker_params = OmegaConf.to_container(cfg.agent.invoker_params, resolve=True)
    client_args = invoker_params.pop("client_args")
    invoker_params.pop("use_responses_api")
    invoker_params.pop("cache_client")
    model = GymOpenAIModel(
        model_id=payload["model"].removeprefix("openai/"),
        params=invoker_params,
        client_args=client_args,
        use_responses_api=False,
        cache_client=False,
    )

    tools, tool_params = load_tools(cfg)
    environment = create_environment(cfg, str(output_dir))
    if isinstance(environment, LocalEnvironment) and shutil.which("timeout") is None:
        environment.execute_bash = _portable_execute_bash.__get__(environment, LocalEnvironment)

    with environment as env, MetricsCollector(str(output_dir)) as metrics_collector:
        agent = StrandsResolverAgent(
            system_prompt=system_prompt,
            model=model,
            tools=tools,
            conversation_manager=conversation_manager,
            hooks=hooks,
            callback_handler=callbacks,
        )
        metrics_collector.bind(agent)
        result = None
        try:
            while True:
                result = agent(user_prompt, environment=env, show_panel=False, tool_params=tool_params)
                retry_feedback = env.retry_feedback(agent, result)
                if not retry_feedback:
                    break
                user_prompt = retry_feedback
        finally:
            env.collect_submission(agent=agent, result=result)
            metrics_collector.dump(agent)

    usage = dict(agent.event_loop_metrics.accumulated_usage or {})
    return {
        "messages": agent.messages,
        "training_turns": model.training_turns,
        "usage": usage,
        "stop_reason": result.stop_reason if result else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.request.read_text())
    try:
        result = _run(payload)
    except BaseException as error:
        args.result.write_text(json.dumps({"error": str(error), "traceback": traceback.format_exc()}, default=str))
        raise
    args.result.write_text(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
