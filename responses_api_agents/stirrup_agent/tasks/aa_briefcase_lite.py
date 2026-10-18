# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AA-Briefcase-Lite strategy for the shared Stirrup agent wrapper."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

from responses_api_agents.stirrup_agent.task_strategy import TaskStrategy


def _dataset_path(root: Path, relative: str, *, directory: bool = False) -> Path:
    candidate = (root / relative.rstrip("/")).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes AA-Briefcase-Lite dataset: {relative}")
    exists = candidate.is_dir() if directory else candidate.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"AA-Briefcase-Lite {kind} not found: {candidate}")
    return candidate


def _render(path: Path, values: Dict[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    unresolved = sorted(set(re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", text)))
    if unresolved:
        raise ValueError(f"Unresolved prompt placeholders in {path}: {unresolved}")
    return text


class AABriefcaseLiteTask(TaskStrategy):
    """Public four-task AA-Briefcase-Lite execution protocol."""

    def extract_task_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        root = Path(metadata["dataset_dir"]).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"AA-Briefcase-Lite dataset not found: {root}")
        for relative in metadata["shared_files"]:
            _dataset_path(root, relative, directory=relative.endswith("/"))
            if not relative.startswith("source_files/shared/"):
                raise ValueError(f"Invalid shared input path: {relative}")
        for relative in metadata["week_files"]:
            _dataset_path(root, relative, directory=relative.endswith("/"))
            if not relative.startswith("source_files/week/"):
                raise ValueError(f"Invalid week input path: {relative}")
        for filename in metadata["deliverable_filenames"]:
            if Path(filename).name != filename or filename in {"", ".", ".."}:
                raise ValueError(f"Invalid deliverable filename: {filename}")
        return {
            "task_id": metadata["task_id"],
            "week": int(metadata["week"]),
            "dataset_dir": str(root),
            "dataset_revision": metadata["dataset_revision"],
            "task_md_path": metadata["task_md_path"],
            "deliverable_filenames": list(metadata["deliverable_filenames"]),
            "scenario_overview_path": metadata["scenario_overview_path"],
            "week_overview_path": metadata["week_overview_path"],
            "shared_files": list(metadata["shared_files"]),
            "week_files": list(metadata["week_files"]),
        }

    def build_system_prompt(self, task_info: Dict[str, Any], config: Any) -> str:
        root = Path(task_info["dataset_dir"])
        return _render(
            _dataset_path(root, "prompts/eval_system.txt"),
            {
                "max_turns": str(config.agent_max_turns),
                "finish_tool_name": "finish",
                "abandon_task_finish": "abandon_task_finish",
            },
        )

    def build_user_prompt(self, task_info: Dict[str, Any], config: Any) -> str:
        root = Path(task_info["dataset_dir"])
        values = {
            "scenario_overview": _dataset_path(root, task_info["scenario_overview_path"]).read_text(encoding="utf-8"),
            "week_overview": _dataset_path(root, task_info["week_overview_path"]).read_text(encoding="utf-8"),
            "task": _dataset_path(root, task_info["task_md_path"]).read_text(encoding="utf-8"),
            "expected_output_filenames": "\n".join(
                f"- `{filename}`" for filename in task_info["deliverable_filenames"]
            ),
            "finish_tool_name": "finish",
            "abandon_task_finish": "abandon_task_finish",
        }
        return _render(_dataset_path(root, "prompts/eval_submission.txt"), values)

    def get_exec_provider(self, task_info: Dict[str, Any], config: Any) -> Any:
        image = getattr(config, "aa_briefcase_container_path", None)
        if not image or not os.path.isfile(image):
            raise RuntimeError("AA-Briefcase-Lite requires a valid `aa_briefcase_container_path`")

        root = Path(task_info["dataset_dir"])
        shared = _dataset_path(root, "source_files/shared", directory=True)
        week = _dataset_path(root, "source_files/week", directory=True)

        from responses_api_agents.stirrup_agent.apptainer_provider import ApptainerCodeExecToolProvider

        return ApptainerCodeExecToolProvider(
            sif_path=image,
            working_dir="/home/user",
            home_dir="/home/user",
            isolated_home=True,
            disable_network=True,
            extra_mounts=[
                f"--mount type=bind,src={shared},dst=/home/user/shared,ro",
                f"--mount type=bind,src={week},dst=/home/user/week,ro",
            ],
            capture_git_diff=False,
            env_passthrough=[],
        )

    def runtime_options(self) -> Dict[str, bool]:
        return {
            "allow_web_tools": False,
            "require_exec_provider": True,
            "use_abandon_finish_tool": True,
            "skip_input_file_listing": True,
            "tool_response_as_user": False,
        }

    def build_response_metadata(
        self,
        task_info: Dict[str, Any],
        deliverable_text: str,
        elapsed_seconds: float,
    ) -> Dict[str, str]:
        return {
            "task_id": task_info["task_id"],
            "week": str(task_info["week"]),
            "dataset_revision": task_info["dataset_revision"],
            "deliverable_text": deliverable_text,
            "elapsed_seconds": str(elapsed_seconds),
        }

    def response_id(self, task_info: Dict[str, Any]) -> str:
        return f"aa-briefcase-lite-{task_info['task_id']}"
