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
from typing import Any, Dict

from responses_api_agents.stirrup_agent.task_strategy import TaskStrategy


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

    def get_exec_provider(self, task_info: Dict[str, Any], config: Any) -> Any:
        container_path = getattr(config, "gdpval_container_path", None)

        # GDPval MUST run inside the Apptainer sandbox built from
        # resources_servers/gdpval/containers/gdpval.def. The sandbox carries the heavy dependency set
        # the task prompt advertises (TeX Live, the full data/ML/document/audio
        # stack, ...). We deliberately do NOT install these into the
        # evaluation/agent container — that would bloat the eval image by many
        # GB. Consequently the local (non-sandbox) backend cannot provide the
        # advertised environment, and silently falling back to it would run
        # every task in a crippled sandbox and yield invalid results. Throw
        # early instead of degrading silently.
        if not container_path:
            raise RuntimeError(
                "GDPval requires the Apptainer sandbox: set `gdpval_container_path` to the "
                ".sif built from resources_servers/gdpval/containers/gdpval.def. "
                "The local (non-sandbox) backend is rejected because the heavy sandbox "
                "dependencies are not — and must not be — installed in the evaluation container."
            )

        if not os.path.exists(container_path):
            raise RuntimeError(
                f"GDPval Apptainer container not found at {container_path}. Build the .sif from "
                "resources_servers/gdpval/containers/gdpval.def. Refusing to fall back "
                "to the local backend, which lacks the sandbox dependencies."
            )

        from responses_api_agents.stirrup_agent.apptainer_provider import ApptainerCodeExecToolProvider

        print(
            f"[gdpval] Using Apptainer container {container_path} for task {task_info.get('task_id', '?')}", flush=True
        )

        # ``working_dir="/root"`` rather than ``/workspace``: ``/root`` is
        # guaranteed by the ``--home /root`` flag we pass to apptainer, while
        # ``/workspace`` only exists if the container's ``%post`` actually ran
        # ``mkdir -p /workspace`` — and the previous ``%post`` could silently
        # skip that step on apt failure. Using ``/root`` makes the per-task
        # apptainer flow robust to a partial container build.
        return ApptainerCodeExecToolProvider(
            sif_path=container_path,
            working_dir="/root",
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
