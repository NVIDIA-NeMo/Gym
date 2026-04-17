# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generic Stirrup-based agent wrapper with pluggable task strategies.

The ``StirrupAgentWrapper`` owns all Stirrup mechanics (agent creation,
Ray execution, history conversion).  Task-specific behaviour (prompt
construction, scoring, response building) is delegated to a
``TaskStrategy`` instance selected via the ``task`` config field.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from asyncio import Semaphore
from pathlib import Path
from typing import Any, Dict, Optional

import ray
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from responses_api_agents.stirrup_agent.task_strategy import TaskSampleSkipError, TaskStrategy


# ---------------------------------------------------------------------------
# Registry of known task strategies (add new tasks here)
# ---------------------------------------------------------------------------

_TASK_REGISTRY: Dict[str, type] = {}


def _load_task_registry() -> Dict[str, type]:
    """Lazily populate the registry so imports only happen when needed."""
    if not _TASK_REGISTRY:
        from responses_api_agents.stirrup_agent.tasks.gdpval import GDPValTask

        _TASK_REGISTRY["gdpval"] = GDPValTask
    return _TASK_REGISTRY


def get_task_strategy(name: str) -> TaskStrategy:
    """Instantiate a ``TaskStrategy`` by its registered name."""
    registry = _load_task_registry()
    if name not in registry:
        raise ValueError(f"Unknown task '{name}'. Available tasks: {sorted(registry.keys())}")
    return registry[name]()


# ---------------------------------------------------------------------------
# Stirrup agent runner (executed in a Ray worker)
# ---------------------------------------------------------------------------


_GDPVAL_PROMPT_TEMPLATE: Optional[str] = None


def _build_gdpval_user_prompt(task_prompt: str, input_files_dir: Optional[str] = None) -> str:
    """Build the full GDPVal user prompt from our template.

    Replaces the former ``gdpval_mode`` fork feature by constructing the prompt
    externally before passing to Stirrup.  File paths are listed relative
    to the parent of *input_files_dir* (e.g. ``gdpval_ref_files_xxx/file.pdf``)
    to match the fork's ``state.uploaded_file_paths`` format.
    """
    global _GDPVAL_PROMPT_TEMPLATE
    if _GDPVAL_PROMPT_TEMPLATE is None:
        template_path = Path(__file__).parent / "prompts" / "gdpval_user_prompt.txt"
        _GDPVAL_PROMPT_TEMPLATE = template_path.read_text(encoding="utf-8")

    if input_files_dir:
        import os

        ref_dir = input_files_dir.rstrip("/")
        parent = os.path.dirname(ref_dir)
        files_section = ""
        for root, _dirs, fnames in os.walk(ref_dir):
            for fname in sorted(fnames):
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, parent)
                files_section += f"- {rel}\n"
        if not files_section:
            files_section = "None"
    else:
        files_section = "None"

    return _GDPVAL_PROMPT_TEMPLATE.format(task=task_prompt, reference_files=files_section)


@ray.remote(scheduling_strategy="SPREAD")
def run_stirrup_agent_remote(params: dict[str, Any]) -> Any:
    return asyncio.run(_run_stirrup_agent(**params))


async def _run_stirrup_agent(
    task_prompt: str,
    system_prompt: str,
    model_base_url: str,
    model_name: str,
    api_key: str = "dummy",
    max_turns: int = 100,
    temperature: float = 0.6,
    max_tokens: int = 262144,
    input_files_dir: Optional[str] = None,
    exec_provider_class: Optional[str] = None,
    exec_provider_kwargs: Optional[Dict[str, Any]] = None,
    persist_deliverables_dir: Optional[str] = None,
    task_id: Optional[str] = None,
    is_gdpval: bool = False,
) -> Dict[str, Any]:
    """Run a Stirrup agent session and return history + metadata.

    If *exec_provider_class* is given (as a dotted import path), it is
    used instead of the default ``LocalCodeExecToolProvider``.
    """
    from stirrup.clients.chat_completions_client import ChatCompletionsClient

    from responses_api_agents.stirrup_agent.nemo_agent import NeMoAgent
    from stirrup.tools import DEFAULT_TOOLS
    from stirrup.tools.code_backends.base import SHELL_TIMEOUT, CodeExecToolProvider, CommandResult
    from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

    class _SandboxTolerantExecProvider(LocalCodeExecToolProvider):
        """LocalCodeExecToolProvider that ensures venv Python is on PATH,
        tolerates file_exists calls outside the sandbox, and creates a
        /workspace symlink so the model can use absolute paths."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            import os
            import sys

            self._venv_bin = os.environ.get("NEMO_GYM_VENV_BIN", str(Path(sys.executable).parent))
            venv_root = str(Path(self._venv_bin).parent)
            self._env_prefix = f'export PATH="{self._venv_bin}:$PATH"; export VIRTUAL_ENV="{venv_root}"; '

        async def start(self):
            await super().start()
            # Create /workspace and /working_dir symlinks to the sandbox temp dir
            # so the model can use absolute paths like /workspace/report.pdf
            if self._temp_dir:
                import os
                for alias in ("/workspace", "/working_dir"):
                    try:
                        os.symlink(str(self._temp_dir), alias)
                    except (OSError, FileExistsError):
                        pass  # Already exists or no permission (non-root)

        async def file_exists(self, path: str) -> bool:
            try:
                return await super().file_exists(path)
            except ValueError:
                return False

        async def run_command(self, cmd: str, *, timeout: int = SHELL_TIMEOUT) -> CommandResult:
            return await super().run_command(self._env_prefix + cmd, timeout=timeout)

    client = ChatCompletionsClient(
        model=model_name,
        base_url=model_base_url,
        api_key=api_key,
        max_tokens=max_tokens,
    )

    if exec_provider_class:
        import importlib

        module_path, class_name = exec_provider_class.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        provider_cls = getattr(mod, class_name)
        exec_provider = provider_cls(**(exec_provider_kwargs or {}))
    else:
        exec_provider = _SandboxTolerantExecProvider()

    tools = [exec_provider if isinstance(t, CodeExecToolProvider) else t for t in DEFAULT_TOOLS]

    # Replace Stirrup's WebToolProvider with TavilyToolProvider when TAVILY_API_KEY is set
    import os as _os

    from stirrup.tools.web import WebToolProvider

    if _os.environ.get("TAVILY_API_KEY"):
        from responses_api_agents.stirrup_agent.tavily_search import TavilyToolProvider

        tools = [TavilyToolProvider() if isinstance(t, WebToolProvider) else t for t in tools]

    agent_kwargs: Dict[str, Any] = {
        "client": client,
        "name": "stirrup_agent",
        "max_turns": max_turns,
        "tools": tools,
        "tool_response_as_user": True,
        "skip_input_file_listing": is_gdpval,
    }
    if system_prompt:
        agent_kwargs["system_prompt"] = system_prompt
    agent = NeMoAgent(**agent_kwargs)

    start_time = time.time()

    input_files = f"{input_files_dir}/" if input_files_dir else None

    output_dir = tempfile.mkdtemp(prefix="stirrup_output_")
    try:
        _aexit_failed = False
        try:
            async with agent.session(output_dir=output_dir, input_files=input_files) as session:
                if is_gdpval:
                    # Build GDPVal prompt with input-dir-relative file paths (matches fork behavior)
                    task_prompt = _build_gdpval_user_prompt(task_prompt, input_files_dir)
                finish_params, history, metadata = await session.run(task_prompt)
        except TypeError as _session_err:
            # Stirrup's session __aexit__ may crash in _log_finish when the exit
            # reason is not a string (e.g. a tuple).  If session.run() completed
            # successfully the results are still available on `session`.
            if "finish_params" not in dir():
                raise
            _aexit_failed = True
            print(f"[stirrup] warning: session __aexit__ raised {_session_err!r}, continuing with results", flush=True)

        # Recover deliverable files that Stirrup's save_output_files may have missed.
        # Only needed for local sandbox (path translation issues with /workspace/ symlinks).
        # Skipped for Apptainer — clean container paths don't need recovery, and skipping
        # ensures fair benchmarking (no artificial ELO boost from path recovery).
        uses_container = exec_provider_class is not None
        if hasattr(finish_params, "paths") and finish_params.paths and not uses_container:
            sandbox_dir = getattr(exec_provider, "_temp_dir", None) or getattr(exec_provider, "temp_dir", None)
            output_dir_path = Path(output_dir)
            existing_files = {f.name for f in output_dir_path.iterdir() if f.is_file()} if output_dir_path.exists() else set()

            for src_path_str in finish_params.paths:
                src = Path(src_path_str)
                filename = src.name

                # Skip if already in output_dir
                if filename in existing_files:
                    continue

                # Try to find the file:
                # 1. Exact path (works for relative paths)
                # 2. Relative to sandbox temp dir (strip any prefix)
                # 3. Search sandbox dir by filename (handles /workspace/, /working_dir/, etc.)
                candidates = [src]
                if sandbox_dir and sandbox_dir.exists():
                    # Strip known prefixes to get relative path
                    path_str = str(src)
                    for prefix in ("/workspace/", "/working_dir/", "/tmp/local_exec_env_"):
                        if path_str.startswith(prefix):
                            # For /tmp/local_exec_env_xxxxx/file.pdf, strip everything up to the ID dir
                            rel = path_str.split("/", 4)[-1] if prefix == "/tmp/local_exec_env_" else path_str[len(prefix):]
                            candidates.append(sandbox_dir / rel)
                            break
                    # Also try just the filename in sandbox root
                    candidates.append(sandbox_dir / filename)
                    # Deep search: find by name anywhere in sandbox
                    for found in sandbox_dir.rglob(filename):
                        candidates.append(found)

                for candidate in candidates:
                    if candidate.is_file():
                        dest = output_dir_path / filename
                        shutil.copy2(str(candidate), str(dest))
                        existing_files.add(filename)
                        print(f"[stirrup] recovered deliverable: {filename} (from {candidate})", flush=True)
                        break
                else:
                    print(f"[stirrup] could not find deliverable: {src_path_str}", flush=True)

        # Stirrup's session __aexit__ saves files from finish_params.paths to output_dir.
        # Read their text content so the judge can score actual deliverables.
        from responses_api_agents.stirrup_agent.file_reader import (
            convert_deliverables_to_content_blocks,
            read_deliverable_files,
        )

        file_contents = read_deliverable_files(output_dir)

        # Build multimodal content blocks (base64 PDFs/images) for visual judging.
        # These are serializable dicts that cross the Ray boundary.
        deliverable_content_blocks = convert_deliverables_to_content_blocks(output_dir)

        # Optionally persist full task artifacts for comparison judging / human review.
        if persist_deliverables_dir:
            import json as _persist_json
            import pickle as _persist_pickle
            import uuid

            dir_name = f"task_{task_id}" if task_id else f"task_{uuid.uuid4().hex[:8]}"
            task_dir = Path(persist_deliverables_dir) / dir_name
            task_dir.mkdir(parents=True, exist_ok=True)

            # 1. Deliverable files
            for f in Path(output_dir).iterdir():
                if f.is_file():
                    shutil.copy2(f, task_dir / f.name)

            # 2. finish_params.json
            try:
                from pydantic import BaseModel as _BM

                fp_data = finish_params.model_dump() if isinstance(finish_params, _BM) else finish_params
                with open(task_dir / "finish_params.json", "w") as fp_f:
                    _persist_json.dump(fp_data, fp_f, indent=2, default=str)
            except Exception as e:
                print(f"[stirrup] warning: could not persist finish_params: {e}", flush=True)

            # 3. history.json (JSON-serializable form)
            try:
                full_history = []
                for msgs in history:
                    for msg in msgs:
                        full_history.append(msg.model_dump() if hasattr(msg, "model_dump") else msg)
                with open(task_dir / "history.json", "w") as h_f:
                    _persist_json.dump(full_history, h_f, indent=2, default=str)
            except Exception as e:
                print(f"[stirrup] warning: could not persist history.json: {e}", flush=True)

            # 4. history.pkl (preserves Stirrup types)
            try:
                with open(task_dir / "history.pkl", "wb") as pkl_f:
                    _persist_pickle.dump(history, pkl_f)
            except Exception as e:
                print(f"[stirrup] warning: could not persist history.pkl: {e}", flush=True)

            # 5. metadata.json
            try:
                with open(task_dir / "metadata.json", "w") as m_f:
                    _persist_json.dump(metadata, m_f, indent=2, default=str)
            except Exception as e:
                print(f"[stirrup] warning: could not persist metadata.json: {e}", flush=True)

            # 6. Reference files
            if input_files_dir and Path(input_files_dir).is_dir():
                ref_dest = task_dir / "reference_files"
                shutil.copytree(input_files_dir, ref_dest, dirs_exist_ok=True)

            print(f"[stirrup] persisted task artifacts to {task_dir}", flush=True)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    elapsed = time.time() - start_time

    # Capture patch from Apptainer provider (saved in __aexit__)
    model_patch = None
    if hasattr(exec_provider, "patch"):
        model_patch = exec_provider.patch
    patch_len = len(model_patch) if model_patch else 0
    print(
        f"[stirrup] model_patch captured: {model_patch is not None} "
        f"(len={patch_len}), provider_type={type(exec_provider).__name__}",
        flush=True,
    )

    # Convert Stirrup objects to plain dicts *inside* the Ray worker so that
    # no Stirrup-specific types cross the Ray serialization boundary (avoids
    # ``Can't get attribute 'SummaryMessage'`` errors from version mismatches).
    from responses_api_agents.stirrup_agent.stirrup_utils import (
        convert_stirrup_history_to_output_items,
        extract_deliverable_text,
    )

    input_items, output_items = convert_stirrup_history_to_output_items(history)
    deliverable_text = extract_deliverable_text(history, finish_params)
    if file_contents:
        deliverable_text = deliverable_text + "\n\n" + file_contents

    # Serialize finish_params to a plain dict
    finish_reason = None
    if finish_params and hasattr(finish_params, "reason"):
        finish_reason = finish_params.reason

    return {
        "input_items": input_items,
        "output_items": output_items,
        "deliverable_text": deliverable_text,
        "deliverable_content_blocks": deliverable_content_blocks,
        "finish_reason": finish_reason,
        "metadata": metadata,
        "elapsed_seconds": elapsed,
        "model_patch": model_patch,
    }


# ---------------------------------------------------------------------------
# Config / request / response types
# ---------------------------------------------------------------------------


class StirrupAgentWrapperConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef

    task: str = Field(
        description="Name of the task strategy to use (e.g. 'gdpval'). Must match a key in the task registry.",
    )

    agent_max_turns: int = Field(default=100, description="Maximum turns for the Stirrup agent")
    concurrency: int = Field(default=32, description="Maximum concurrent runs")
    temperature: float = Field(default=0.6, description="Sampling temperature for the agent model")

    system_prompt_template: Optional[str] = Field(
        default=None, description="Path to the system prompt Jinja2 template"
    )
    user_prompt_template: Optional[str] = Field(default=None, description="Path to the user prompt Jinja2 template")
    judge_prompt_template: Optional[str] = Field(default=None, description="Path to the judge prompt Jinja2 template")
    judge_model_name: Optional[str] = Field(
        default=None,
        description="Model to use for judging. If None, uses the same model as the agent.",
    )
    judge_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the judge model API. If None, uses the policy model's URL.",
    )
    judge_api_key: Optional[str] = Field(
        default=None,
        description="API key for the judge model. If None, uses 'dummy'.",
    )

    container_formatter: Optional[Any] = Field(
        default=None,
        description="Container path template(s) for tasks that need Apptainer execution. "
        "Can be a string or list of strings with {instance_id} placeholder.",
    )
    apptainer_memory_limit_mb: Optional[int] = Field(
        default=None,
        description="Memory limit in MB for Apptainer containers.",
    )
    gdpval_container_path: Optional[str] = Field(
        default=None,
        description="Path to GDPVal Apptainer .sif container. When set, code execution runs inside the container.",
    )
    swebench_tests_timeout: int = Field(
        default=30 * 60,
        description="Timeout in seconds for SWE-bench test evaluation.",
    )
    persist_deliverables_dir: Optional[str] = Field(
        default=None,
        description="Directory to persist deliverable files for human review. "
        "If None (default), files are deleted after text extraction for scoring.",
    )
    reward_mode: str = Field(
        default="rubric",
        description="Reward mode: 'rubric' (default) for LLM judge scoring against rubric, "
        "or 'comparison' for pairwise comparison against a reference model's outputs.",
    )
    reference_model_dir: Optional[str] = Field(
        default=None,
        description="Path to reference model output directory for comparison reward mode. "
        "Must contain task_<id>/ subdirectories with deliverable files.",
    )
    num_judge_trials: int = Field(
        default=4,
        description="Number of judge trials per task (comparison mode: alternating swapped/unswapped, "
        "structured_rubric mode: independent scoring rounds averaged).",
    )
    formatting_retries: int = Field(
        default=3,
        description="Number of times to retry a judge request when the response doesn't contain "
        "parseable score tags (structured_rubric mode only).",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="HuggingFace model ID for tokenizer (enables dynamic token sizing). "
        "E.g. 'Qwen/Qwen3-Coder-30B-A3B-Instruct'. If None, completion_token_buffer is ignored.",
    )
    completion_token_buffer: int = Field(
        default=5000,
        description="Token buffer reserved for completion (prevents context overflow). "
        "Only used when model_id is set.",
    )


class StirrupRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class StirrupVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Generic wrapper
# ---------------------------------------------------------------------------


class StirrupAgentWrapper(SimpleResponsesAPIAgent):
    """Generic Stirrup agent wrapper — task logic is pluggable via config."""

    config: StirrupAgentWrapperConfig
    sem: Semaphore = None
    task_strategy: TaskStrategy = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self.task_strategy = get_task_strategy(self.config.task)
        print(f"Stirrup agent initialized with task={self.config.task!r}", flush=True)

    # -- helpers ----------------------------------------------------------

    def _get_model_base_url(self) -> str:
        from nemo_gym.global_config import get_first_server_config_dict
        from nemo_gym.server_utils import ServerClient

        global_config_dict = ServerClient.load_from_global_config().global_config_dict
        model_server_config = get_first_server_config_dict(global_config_dict, self.config.model_server.name)
        return f"http://{model_server_config['host']}:{model_server_config['port']}/v1"

    # -- /v1/responses ----------------------------------------------------

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        task_info = self.task_strategy.extract_task_info(body.metadata)

        model_base_url = self._get_model_base_url()

        input_files_dir = self.task_strategy.prepare_input_files(task_info)

        if self.config.task == "gdpval":
            system_prompt = None
            # Raw task prompt — _run_stirrup_agent wraps it in GDPVal template when is_gdpval=True
            user_prompt = f"Sector: {task_info['sector']}\nOccupation: {task_info['occupation']}\n\n{task_info['prompt']}"
        else:
            system_prompt = self.task_strategy.build_system_prompt(task_info, self.config)
            user_prompt = self.task_strategy.build_user_prompt(task_info, self.config)

        model_name = getattr(body, "model", None) or "default"
        temperature = getattr(body, "temperature", None) or self.config.temperature
        max_tokens = getattr(body, "max_output_tokens", 262144) or 262144

        exec_provider = self.task_strategy.get_exec_provider(task_info, self.config)
        exec_provider_class = None
        exec_provider_kwargs = None
        if exec_provider is not None:
            cls = type(exec_provider)
            exec_provider_class = f"{cls.__module__}.{cls.__qualname__}"
            exec_provider_kwargs = exec_provider._serializable_kwargs()

        try:
            params = {
                "task_prompt": user_prompt,
                "system_prompt": system_prompt,
                "model_base_url": model_base_url,
                "model_name": model_name,
                "api_key": "dummy",
                "max_turns": self.config.agent_max_turns,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "input_files_dir": input_files_dir,
                "exec_provider_class": exec_provider_class,
                "exec_provider_kwargs": exec_provider_kwargs,
                "persist_deliverables_dir": self.config.persist_deliverables_dir,
                "task_id": task_info.get("task_id"),
                "is_gdpval": self.config.task == "gdpval",
            }

            future = run_stirrup_agent_remote.remote(params)
            result = await future
        finally:
            if input_files_dir:
                shutil.rmtree(input_files_dir, ignore_errors=True)

        input_items = result["input_items"]
        output_items = result["output_items"]
        deliverable_text = result["deliverable_text"]

        if not output_items:
            output_items = [
                NeMoGymResponseOutputMessage(
                    id=self.task_strategy.fallback_message_id(task_info),
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text="No output produced by agent.",
                            annotations=[],
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ]

        metadata = self.task_strategy.build_response_metadata(
            task_info=task_info,
            deliverable_text=deliverable_text,
            elapsed_seconds=result.get("elapsed_seconds", 0),
        )

        if result.get("model_patch") is not None:
            metadata["model_patch"] = result["model_patch"]

        if result.get("deliverable_content_blocks"):
            import json as _json

            metadata["deliverable_content_blocks"] = _json.dumps(result["deliverable_content_blocks"])

        return NeMoGymResponse(
            id=self.task_strategy.response_id(task_info),
            created_at=int(time.time()),
            model=model_name,
            object="response",
            output=input_items + output_items,
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            metadata=metadata,
        )

    # -- /run -------------------------------------------------------------

    async def run(self, body: StirrupRunRequest):
        async with self.sem:
            fixed_params = body.responses_create_params
            if fixed_params.tool_choice is None:
                fixed_params = fixed_params.model_copy(update={"tool_choice": "auto"})

            try:
                response = await self.responses(fixed_params)
            except Exception as exc:
                task_info = self.task_strategy.extract_task_info(body.responses_create_params.metadata)
                label = "skipped" if isinstance(exc, TaskSampleSkipError) else "failed"
                instance_hint = task_info.get("instance_id", "unknown")
                print(
                    f"[stirrup-{label}] {instance_hint}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                return self.task_strategy.build_skipped_verify_response(
                    responses_create_params=fixed_params,
                    task_info=task_info,
                    reason=f"{type(exc).__name__}: {exc}",
                )

            metadata = response.metadata or {}
            response_clean = response.model_copy(update={"metadata": None})

            task_info = self.task_strategy.extract_task_info(body.responses_create_params.metadata)
            deliverable_text = metadata.get("deliverable_text", "")

            # Enrich task_info with runtime artifacts for reward computation
            if metadata.get("model_patch"):
                task_info["model_patch"] = metadata["model_patch"]

            judge_base_url = self.config.judge_base_url or self._get_model_base_url()
            judge_model_name = self.config.judge_model_name or fixed_params.model
            judge_api_key = self.config.judge_api_key or "dummy"

            import json as _json

            _blocks_str = metadata.get("deliverable_content_blocks", "[]")
            deliverable_content_blocks = _json.loads(_blocks_str) if isinstance(_blocks_str, str) else _blocks_str

            reward_result = await self.task_strategy.compute_reward(
                deliverable_text=deliverable_text,
                task_info=task_info,
                config=self.config,
                model_base_url=judge_base_url,
                model_name=judge_model_name,
                api_key=judge_api_key,
                deliverable_content_blocks=deliverable_content_blocks,
            )
            # compute_reward returns (float, dict|None) or plain float for backward compat
            if isinstance(reward_result, tuple):
                reward, judge_response = reward_result
            else:
                reward, judge_response = reward_result, None

            input_messages = [
                item
                for item in response_clean.output
                if hasattr(item, "role") and item.role in ("system", "user", "developer")
            ]
            output_only = [item for item in response_clean.output if item not in input_messages]

            response_clean = response_clean.model_copy(update={"output": output_only})
            params_with_input = fixed_params.model_copy(
                update={"input": input_messages, "tools": response_clean.tools or []}
            )

            elapsed_seconds = float(metadata.get("elapsed_seconds", 0))

            verify_response = self.task_strategy.build_verify_response(
                responses_create_params=params_with_input,
                response=response_clean,
                reward=reward,
                task_info=task_info,
                deliverable_text=deliverable_text,
                elapsed_seconds=elapsed_seconds,
                judge_response=judge_response,
            )
            # Return as dict to bypass FastAPI's response_model filtering,
            # which would drop subclass fields like judge_response.
            return verify_response.model_dump(mode="json")


if __name__ == "__main__":
    StirrupAgentWrapper.run_webserver()
