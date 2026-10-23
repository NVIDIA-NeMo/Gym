# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a Gym task from conversation, candidate responses, and verifier metadata."""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, TextIO

from nemo_gym.base_resources_server import BaseRunRequest


# Shared by benchmark preparation scripts to build Gym evaluation inputs.
def make_task(
    task_id: str, turns: list[dict[str, str]], first: str, second: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    if not turns:
        raise ValueError(f"Row {task_id} has no conversation")
    return {
        "id": task_id,
        "responses_create_params": {
            "input": turns,
            "metadata": {"response_1": first, "response_2": second},
            "temperature": 1.0,
            "top_p": 0.95,
            "max_output_tokens": 16384,
        },
        "verifier_metadata": metadata,
    }


@contextmanager
def atomic_output(output: Path) -> Iterator[TextIO]:
    """Replace the destination only after preparation succeeds."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, suffix=".jsonl", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            yield stream
            stream.close()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


def download_prepared_dataset(
    dataset_name: str,
    version: str,
    artifact_fpath: str,
    output_fpath: Path | str,
    *,
    validate_metadata: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Fetch preprocessed Gym tasks from the private GitLab registry."""
    from nemo_gym.config_types import DownloadJsonlDatasetGitlabConfig
    from nemo_gym.gitlab_utils import download_jsonl_dataset

    output = Path(output_fpath)
    with TemporaryDirectory() as directory:
        artifact = Path(directory) / "dataset.jsonl"
        download_jsonl_dataset(
            DownloadJsonlDatasetGitlabConfig(
                dataset_name=dataset_name, version=version, artifact_fpath=artifact_fpath, output_fpath=str(artifact)
            )
        )
        ids: set[str] = set()
        with artifact.open() as source, atomic_output(output) as destination:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a Gym task object")
                task_id = row.get("id")
                metadata = row.get("verifier_metadata")
                if not isinstance(task_id, str) or not task_id.strip() or task_id in ids:
                    raise ValueError("Expected a nonempty, unique task ID")
                if not isinstance(metadata, dict) or not metadata:
                    raise ValueError("Expected nonempty verifier metadata")
                params = BaseRunRequest.model_validate(row).responses_create_params
                if not isinstance(params.input, list) or not params.input:
                    raise ValueError("Expected conversation messages")
                candidates = params.metadata or {}
                if not all(isinstance(candidates.get(key), str) for key in ("response_1", "response_2")):
                    raise ValueError("Expected two candidate response strings")
                if validate_metadata is not None:
                    validate_metadata(metadata)
                ids.add(task_id)
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            if not ids:
                raise ValueError("Downloaded dataset is empty")
    return output
