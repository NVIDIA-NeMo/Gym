# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``task_input.py``: the task input the episode request validates, loadable without the server's dependencies."""

import ast
import sys
from pathlib import Path

from environment_servers.bird_interact import task_data, task_input
from environment_servers.bird_interact.app import BirdInteractRequest


SERVER_DIR = Path(__file__).resolve().parents[1]


def test_task_input_is_what_the_episode_request_validates() -> None:
    assert (
        BirdInteractRequest.model_fields["task"].annotation.model_fields["task_input"].annotation
        is task_input.TaskInput
    )
    assert task_data.TaskData is task_input.TaskInput


def test_task_input_imports_only_the_standard_library_and_pydantic() -> None:
    tree = ast.parse((SERVER_DIR / "task_input.py").read_text())
    imported = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    imported += [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all(name.split(".")[0] in sys.stdlib_module_names or name.split(".")[0] == "pydantic" for name in imported)
