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
"""GDPVal task strategy for the generic Stirrup agent wrapper."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseVerifyResponse
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseOutputMessage, NeMoGymResponseOutputText
from responses_api_agents.stirrup_agent.task_strategy import TaskStrategy


def _render_template(template_path: str, **kwargs) -> str:
    from jinja2 import Environment

    path = Path(template_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Template not found at '{template_path}'. "
            f"Directory exists: {path.parent.is_dir()}, "
            f"Directory contents: {list(path.parent.iterdir()) if path.parent.is_dir() else 'N/A'}"
        )
    template_source = path.read_text()
    template = Environment().from_string(template_source)
    return template.render(**kwargs)


def _parse_json_str(value: Any, default: Any = None):
    """Parse a value that may be a JSON-encoded string."""
    if default is None:
        default = value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default
    return value


def _download_reference_files(
    reference_files: list[str],
    reference_file_urls: list[str],
    dest_dir: Path,
) -> list[str]:
    import urllib.request

    if not reference_files or not reference_file_urls:
        return []

    downloaded = []
    for file_path, url in zip(reference_files, reference_file_urls):
        rel_path = file_path.lstrip("/")
        local_path = dest_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, local_path)
            downloaded.append(rel_path)
        except Exception as e:
            print(f"Warning: failed to download {url} -> {rel_path}: {e}", flush=True)

    return downloaded


# ---------------------------------------------------------------------------
# GDPVal-specific verify response
# ---------------------------------------------------------------------------


class GDPValVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    rubric_score: Optional[float] = None
    deliverable_text: Optional[str] = None
    task_id: Optional[str] = None
    sector: Optional[str] = None
    occupation: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    judge_response: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------


class GDPValTask(TaskStrategy):
    """GDPVal benchmark — professional knowledge-work tasks scored via rubric."""

    def extract_task_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": metadata["task_id"],
            "sector": metadata.get("sector", ""),
            "occupation": metadata.get("occupation", ""),
            "prompt": metadata["prompt"],
            "reference_files": _parse_json_str(metadata.get("reference_files", "[]"), []),
            "reference_file_urls": _parse_json_str(metadata.get("reference_file_urls", "[]"), []),
            "rubric_json": _parse_json_str(metadata.get("rubric_json", "{}"), {}),
            "rubric_pretty": metadata.get("rubric_pretty", ""),
        }

    def build_system_prompt(self, task_info: Dict[str, Any], config: Any) -> str:
        if config.system_prompt_template:
            return _render_template(config.system_prompt_template, task=task_info)
        return ""

    def build_user_prompt(self, task_info: Dict[str, Any], config: Any) -> str:
        if config.user_prompt_template:
            return _render_template(config.user_prompt_template, task=task_info)
        return task_info["prompt"]

    def prepare_input_files(self, task_info: Dict[str, Any]) -> Optional[str]:
        ref_files = task_info.get("reference_files")
        ref_urls = task_info.get("reference_file_urls")
        if not ref_files or not ref_urls:
            return None

        input_files_dir = tempfile.mkdtemp(prefix="gdpval_ref_files_")
        downloaded = _download_reference_files(ref_files, ref_urls, Path(input_files_dir))
        if downloaded:
            print(f"Downloaded {len(downloaded)} reference files to {input_files_dir}", flush=True)
            return input_files_dir

        return None

    def get_exec_provider(self, task_info: Dict[str, Any], config: Any) -> Any:
        container_path = getattr(config, "gdpval_container_path", None)
        if not container_path:
            return None

        if not os.path.exists(container_path):
            print(
                f"[gdpval] WARNING: container not found at {container_path}, "
                f"falling back to local sandbox",
                flush=True,
            )
            return None

        from responses_api_agents.stirrup_agent.apptainer_provider import ApptainerCodeExecToolProvider

        print(f"[gdpval] Using Apptainer container {container_path} for task {task_info.get('task_id', '?')}", flush=True)

        return ApptainerCodeExecToolProvider(
            sif_path=container_path,
            working_dir="/workspace",
            memory_limit_mb=getattr(config, "apptainer_memory_limit_mb", None),
            capture_git_diff=False,
            env_passthrough=["HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy"],
        )

    def build_response_metadata(
        self,
        task_info: Dict[str, Any],
        deliverable_text: str,
        elapsed_seconds: float,
    ) -> Dict[str, str]:
        return {
            "task_id": task_info["task_id"],
            "sector": task_info["sector"],
            "occupation": task_info["occupation"],
            "deliverable_text": deliverable_text,
            "elapsed_seconds": str(elapsed_seconds),
        }

    def response_id(self, task_info: Dict[str, Any]) -> str:
        return f"gdpval-{task_info['task_id']}"

    async def compute_reward(
        self,
        deliverable_text: str,
        task_info: Dict[str, Any],
        config: Any,
        model_base_url: str,
        model_name: str,
        api_key: str = "dummy",
        deliverable_content_blocks: list | None = None,
    ) -> tuple[float, dict | None]:
        reward_mode = getattr(config, "reward_mode", "rubric")

        if reward_mode == "comparison":
            return self._compute_comparison_reward(task_info, config, model_base_url, model_name, api_key)

        if reward_mode == "structured_rubric":
            return await self._compute_structured_rubric_reward(
                deliverable_text, task_info, config, model_base_url, model_name, api_key,
                deliverable_content_blocks,
            )

        # --- Rubric-based scoring (default) ---
        if not (task_info.get("rubric_json") or task_info.get("rubric_pretty")):
            return 0.0, None

        if not config.judge_prompt_template:
            print("Warning: rubric present but no judge_prompt_template configured", flush=True)
            return 0.0, None

        # Use visual scoring when content blocks are available and judge is multimodal (Gemini)
        if deliverable_content_blocks and "gemini" in model_name.lower():
            from responses_api_agents.stirrup_agent.tasks._gdpval_scoring import score_with_rubric_visual

            print(
                f"[gdpval] Using visual judge ({model_name}) with {len(deliverable_content_blocks)} content blocks",
                flush=True,
            )
            return await score_with_rubric_visual(
                deliverable_content_blocks=deliverable_content_blocks,
                rubric_json=task_info.get("rubric_json", {}),
                rubric_pretty=task_info.get("rubric_pretty", ""),
                task_prompt=task_info["prompt"],
                judge_prompt_template=config.judge_prompt_template,
                model_base_url=model_base_url,
                model_name=model_name,
                api_key=api_key,
            )

        from responses_api_agents.stirrup_agent.tasks._gdpval_scoring import score_with_rubric

        return await score_with_rubric(
            deliverable_text=deliverable_text,
            rubric_json=task_info.get("rubric_json", {}),
            rubric_pretty=task_info.get("rubric_pretty", ""),
            task_prompt=task_info["prompt"],
            judge_prompt_template=config.judge_prompt_template,
            model_base_url=model_base_url,
            model_name=model_name,
            api_key=api_key,
        )

    async def _compute_structured_rubric_reward(
        self,
        deliverable_text: str,
        task_info: Dict[str, Any],
        config: Any,
        model_base_url: str,
        model_name: str,
        api_key: str,
        deliverable_content_blocks: list | None,
    ) -> tuple[float, dict | None]:
        """Structured rubric scoring with tagged output, multi-trial averaging, and format retries."""
        if not (task_info.get("rubric_json") or task_info.get("rubric_pretty")):
            return 0.0, None

        from responses_api_agents.stirrup_agent.tasks._gdpval_scoring import score_with_rubric_structured

        judge_base_url = getattr(config, "judge_base_url", None) or model_base_url
        judge_model = getattr(config, "judge_model_name", None) or model_name
        judge_key = getattr(config, "judge_api_key", None) or api_key
        num_trials = getattr(config, "num_judge_trials", 2)
        formatting_retries = getattr(config, "formatting_retries", 3)

        return await score_with_rubric_structured(
            deliverable_text=deliverable_text,
            rubric_json=task_info.get("rubric_json", {}),
            rubric_pretty=task_info.get("rubric_pretty", ""),
            task_prompt=task_info["prompt"],
            model_base_url=judge_base_url,
            model_name=judge_model,
            api_key=judge_key,
            num_trials=num_trials,
            formatting_retries=formatting_retries,
            deliverable_content_blocks=deliverable_content_blocks,
        )

    def _compute_comparison_reward(
        self,
        task_info: Dict[str, Any],
        config: Any,
        model_base_url: str,
        model_name: str,
        api_key: str,
    ) -> tuple[float, dict | None]:
        """Pairwise comparison reward: compare eval model output vs reference model output."""
        from openai import OpenAI

        from responses_api_agents.stirrup_agent.tasks._gdpval_comparison import (
            build_file_section,
            compute_comparison_reward,
            run_trials,
            task_attempted,
        )

        reference_model_dir = getattr(config, "reference_model_dir", None)
        persist_dir = getattr(config, "persist_deliverables_dir", None)

        if not reference_model_dir:
            print("[gdpval] comparison mode requires reference_model_dir in config", flush=True)
            return 0.0, None
        if not persist_dir:
            print("[gdpval] comparison mode requires persist_deliverables_dir in config", flush=True)
            return 0.0, None

        task_id = task_info["task_id"]
        ref_task_dir = os.path.join(reference_model_dir, f"task_{task_id}")
        eval_task_dir = os.path.join(persist_dir, f"task_{task_id}")

        if not task_attempted(ref_task_dir):
            print(f"[gdpval] reference model has no output for task {task_id}, score=0.0", flush=True)
            return 0.0, None
        if not task_attempted(eval_task_dir):
            print(f"[gdpval] eval model has no output for task {task_id}, score=0.0", flush=True)
            return 0.0, None

        refs_dir = os.path.join(ref_task_dir, "reference_files")
        if not os.path.isdir(refs_dir):
            refs_dir = None

        refs = build_file_section(refs_dir)
        ref_submission = build_file_section(ref_task_dir)
        eval_submission = build_file_section(eval_task_dir)

        judge_base_url = getattr(config, "judge_base_url", None) or model_base_url
        judge_model = getattr(config, "judge_model_name", None) or model_name
        judge_key = getattr(config, "judge_api_key", None) or api_key
        num_trials = getattr(config, "num_judge_trials", 4)

        client = OpenAI(base_url=judge_base_url, api_key=judge_key)

        print(
            f"[gdpval] comparison mode: judging task {task_id} ({num_trials} trials, judge={judge_model})",
            flush=True,
        )
        result = run_trials(
            client=client,
            model=judge_model,
            task_prompt=task_info["prompt"],
            refs=refs,
            submission_a=ref_submission,
            submission_b=eval_submission,
            num_trials=num_trials,
        )

        reward = compute_comparison_reward(result["winner"])
        print(
            f"[gdpval] comparison result: winner={result['winner']}, "
            f"A={result['win_count_a']}, B={result['win_count_b']}, "
            f"ties={result['tie_count']}, reward={reward}",
            flush=True,
        )
        return reward, result

    def build_skipped_verify_response(
        self,
        *,
        responses_create_params: Any,
        task_info: Dict[str, Any],
        reason: str,
    ) -> GDPValVerifyResponse:
        import time

        dummy_output = NeMoGymResponseOutputMessage(
            id=f"msg-{self.response_id(task_info)}-skipped",
            content=[
                NeMoGymResponseOutputText(
                    type="output_text",
                    text=f"Skipped: {reason}",
                    annotations=[],
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        )
        dummy_response = NeMoGymResponse(
            id=f"{self.response_id(task_info)}-skipped",
            created_at=int(time.time()),
            model="skipped",
            object="response",
            output=[dummy_output],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        return GDPValVerifyResponse(
            responses_create_params=responses_create_params,
            response=dummy_response,
            reward=0.0,
            rubric_score=0.0,
            deliverable_text="",
            task_id=task_info.get("task_id"),
            sector=task_info.get("sector", ""),
            occupation=task_info.get("occupation", ""),
            elapsed_seconds=0.0,
            judge_response={"skipped": True, "reason": reason},
        )

    def build_verify_response(
        self,
        *,
        responses_create_params: Any,
        response: NeMoGymResponse,
        reward: float,
        task_info: Dict[str, Any],
        deliverable_text: str,
        elapsed_seconds: float,
        judge_response: dict | None = None,
    ) -> GDPValVerifyResponse:
        return GDPValVerifyResponse(
            responses_create_params=responses_create_params,
            response=response,
            reward=reward,
            rubric_score=reward,
            deliverable_text=deliverable_text,
            judge_response=judge_response,
            task_id=task_info["task_id"],
            sector=task_info["sector"],
            occupation=task_info["occupation"],
            elapsed_seconds=elapsed_seconds,
        )
