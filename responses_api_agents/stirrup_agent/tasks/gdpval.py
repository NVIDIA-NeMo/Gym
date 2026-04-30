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
    """Materialize reference files into ``dest_dir``.

    Each entry in ``reference_file_urls`` is one of:
    - an HTTP(S) URL — downloaded with HF auth + retry-on-429/5xx (anonymous
      HuggingFace downloads hit an aggressive rate limit when the agent runs
      tasks concurrently; the previous urlretrieve-based version silently
      dropped files, leaving subsequent judging/scoring without the reference
      context — we now pass ``HF_TOKEN`` as a bearer token and retry with
      exponential backoff);
    - an absolute filesystem path or ``file://`` URL — copied directly with
      ``shutil.copy2``, bypassing the urllib retry loop. Benchmarks whose
      reference files already live on the agent's filesystem (rather than on
      the HuggingFace Hub) use this path.
    """
    import shutil as _shutil
    import time
    import urllib.error
    import urllib.request

    if not reference_files or not reference_file_urls:
        return []

    token = os.environ.get("HF_TOKEN")
    max_attempts = 6  # backoffs: 1, 2, 4, 8, 16, 30 seconds
    downloaded = []
    for file_path, url in zip(reference_files, reference_file_urls):
        rel_path = file_path.lstrip("/")
        local_path = dest_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not (url.startswith("http://") or url.startswith("https://")):
            src = url[len("file://") :] if url.startswith("file://") else url
            try:
                _shutil.copy2(src, local_path)
                downloaded.append(rel_path)
            except Exception as e:
                print(f"Warning: failed to copy local file {src} -> {rel_path}: {e}", flush=True)
            continue

        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")

        last_err: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp, open(local_path, "wb") as f_out:
                    _shutil.copyfileobj(resp, f_out)
                downloaded.append(rel_path)
                last_err = None
                break
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code != 429 and not (500 <= e.code < 600):
                    break  # non-retryable (404, 403, etc.)
                if attempt < max_attempts - 1:
                    time.sleep(min(2**attempt, 30))
            except Exception as e:  # URLError, timeout, socket errors
                last_err = e
                if attempt < max_attempts - 1:
                    time.sleep(min(2**attempt, 30))

        if last_err is not None:
            print(f"Warning: failed to download {url} -> {rel_path}: {last_err}", flush=True)

    return downloaded


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
                f"[gdpval] WARNING: container not found at {container_path}, falling back to local sandbox",
                flush=True,
            )
            return None

        from responses_api_agents.stirrup_agent.apptainer_provider import ApptainerCodeExecToolProvider

        print(
            f"[gdpval] Using Apptainer container {container_path} for task {task_info.get('task_id', '?')}", flush=True
        )

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
