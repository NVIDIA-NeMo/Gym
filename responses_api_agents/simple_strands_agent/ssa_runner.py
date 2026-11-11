# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from shlex import quote
from typing import Any

import ssa
from omegaconf import OmegaConf
from ssa.agent import StrandsResolverAgent
from ssa.callbacks.throttling import ThrottlingCallback
from ssa.conversation_manager.conversation_manager import AdaptiveConversationManager
from ssa.environments import Environment, LocalEnvironment, create_environment
from ssa.hooks import initialize_hooks
from ssa.metrics import MetricsCollector
from ssa.models import sr_model
from ssa.prompts.prompt_gen import PromptGenerator
from ssa.tools import load_tools
from strands.handlers.callback_handler import CompositeCallbackHandler


importlib.import_module("ssa.utils.monkey_patch")


class QuietCallback:
    def __call__(self, **kwargs: Any) -> None:
        pass


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


class SandboxEnvironment(Environment):
    """Run SSA's tools in a Gym task sandbox: every command and file operation goes through the sandbox provider.

    The agent loop and its model calls stay in this process; the sandbox is borrowed from the Resources Server (or
    owned by the agent server), and this process only connects to it.
    """

    def __init__(self, spec: dict[str, Any], workdir: str, default_timeout: float | None):
        from nemo_gym.sandbox import AsyncSandbox, create_provider

        self._workdir = workdir
        self._default_timeout = default_timeout
        # The environment the task's owner resolved for harness commands, over the sandbox's own.
        self._env = spec.get("env") or None
        # A provider failure, which has no command exit code. SSA turns tool exceptions into tool results, so _run
        # checks this after SSA returns and fails the run.
        self.infra_error: str | None = None
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        provider = create_provider(spec["provider"])
        self._sandbox = self._call(AsyncSandbox.connect(spec["descriptor"], provider=provider))

    def _call(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    @property
    def workdir(self) -> str:
        return self._workdir

    @property
    def timeout(self) -> float | None:
        return self._default_timeout

    def execute_bash(self, command, workdir: str | None = None, timeout: float | None = None, verbose: bool = False):
        timeout = self.timeout if timeout is None else timeout
        # Like LocalEnvironment: SIGTERM at the deadline, SIGKILL 30 s later, and exit 124, where the image has
        # coreutils timeout; the provider deadline bounds the command either way.
        script = f"bash -c {quote(command)}"
        if timeout:
            script = (
                f"if command -v timeout >/dev/null 2>&1; then exec timeout -k 30s {float(timeout)}s {script}; "
                f"else exec {script}; fi"
            )
        try:
            result = self._call(
                self._sandbox.exec(
                    f"bash -c {quote(script)}",
                    cwd=workdir or self.workdir,
                    env=self._env,
                    timeout_s=float(timeout) + 60 if timeout else None,
                )
            )
        except Exception as error:
            self.infra_error = f"Sandbox exec failed: {error}"
            raise
        if result.error_type and result.error_type != "timeout":
            self.infra_error = f"Sandbox exec failed ({result.error_type}): {result.stderr or result.stdout}"
            raise RuntimeError(self.infra_error)
        exit_code = 124 if result.error_type == "timeout" else result.return_code
        output = (result.stdout or "") + (result.stderr or "")
        return {
            "command": command,
            "status": "success" if exit_code == 0 else "error",
            "exit_code": exit_code,
            "output": output,
            "error": "" if exit_code in (0, 124) else output,
        }

    def _test(self, flag: str, path: str) -> bool:
        return self.execute_bash(f"test {flag} {quote(path)}", timeout=30)["exit_code"] == 0

    def dir_exists(self, path: str) -> bool:
        return self._test("-d", path)

    def file_exists(self, path: str) -> bool:
        return self._test("-f", path)

    def write_file(self, content: str, dest_path: str) -> None:
        from nemo_gym.sandbox import upload_text

        try:
            self._call(upload_text(self._sandbox, path=dest_path, text=content))
        except Exception as error:
            self.infra_error = f"Sandbox upload failed: {error}"
            raise

    def retry_feedback(self, *args, **kwargs) -> str | None:
        return None

    def collect_submission(self, *args, **kwargs) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        # Disconnect only: the sandbox's owner stops it.
        try:
            self._call(self._sandbox.disconnect())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
        return False


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
    if payload.get("reasoning_effort") is not None:
        params["reasoning_effort"] = payload["reasoning_effort"]
    cfg.agent.invoker_params = OmegaConf.create(
        {
            **params,
            "use_responses_api": False,
            "cache_client": False,
            "client_args": {
                "base_url": payload["model_base_url"],
                "api_key": "gym",  # pragma: allowlist secret
                "timeout": payload["model_timeout"],
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


def _run(payload: dict[str, Any], sandbox: dict[str, Any] | None = None) -> dict[str, Any]:
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
    model = sr_model(cfg)

    tools, tool_params = load_tools(cfg)
    if sandbox is not None:
        environment = SandboxEnvironment(sandbox, payload["work_dir"], cfg.env.timeout)
    else:
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
        stop_reason = None
        try:
            while True:
                try:
                    result = agent(user_prompt, environment=env, show_panel=False, tool_params=tool_params)
                except Exception as error:
                    # max_llm_iterations ends the run with this exception, raw or wrapped in EventLoopException
                    # when it fires in a nested cycle; the transcript so far is the result.
                    if "exceeded the set threshold" not in str(error):
                        raise
                    stop_reason = "max_iterations"
                    break
                retry_feedback = env.retry_feedback(agent, result)
                if not retry_feedback:
                    break
                user_prompt = retry_feedback
        finally:
            env.collect_submission(agent=agent, result=result)
            metrics_collector.dump(agent)
    if sandbox is not None and environment.infra_error:
        raise RuntimeError(environment.infra_error)

    usage = dict(agent.event_loop_metrics.accumulated_usage or {})
    return {
        "messages": agent.messages,
        "usage": usage,
        "stop_reason": stop_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    # The task sandbox's provider config and descriptor arrive on stdin, so provider credentials never touch disk.
    parser.add_argument("--sandbox-stdin", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.request.read_text())
    sandbox = json.loads(sys.stdin.read()) if args.sandbox_stdin else None
    try:
        result = _run(payload, sandbox)
    except BaseException as error:
        # The HTTP status of a failed model request, from the exception or one it wraps.
        status_code, cause = None, error
        while cause is not None and status_code is None:
            status_code = getattr(cause, "status_code", None)
            cause = cause.__cause__ or cause.__context__
        failure = {"error": str(error), "status_code": status_code, "traceback": traceback.format_exc()}
        args.result.write_text(json.dumps(failure, default=str))
        raise
    args.result.write_text(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
