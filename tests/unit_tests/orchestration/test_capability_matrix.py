# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import inspect

import pytest

from nemo_gym.orchestration import submit
from nemo_gym.orchestration.capability_matrix import (
    COMMAND_CAPABILITIES,
    DOCS_PAGE,
    END_MARKER,
    START_MARKER,
    render_capability_matrix,
    render_generated_block,
    replace_generated_block,
)
from nemo_gym.orchestration.executors.kubernetes.executor import KubernetesExecutor
from nemo_gym.orchestration.executors.slurm.executor import SlurmExecutor


def test_docs_page_matches_the_executor_classes() -> None:
    page = DOCS_PAGE.read_text()
    assert replace_generated_block(page, render_generated_block()) == page, (
        f"The capability matrix in {DOCS_PAGE} is out of date. Run "
        "`python -m nemo_gym.orchestration.capability_matrix --write` and commit the result."
    )


def test_matrix_reflects_what_each_executor_implements() -> None:
    rows = {
        line.split("|")[1].strip(" `"): [cell.strip() for cell in line.split("|")[2:-1]]
        for line in render_capability_matrix().splitlines()[2:]
    }
    assert rows[SlurmExecutor.name] == ["Yes"] * (len(COMMAND_CAPABILITIES) + 1)
    assert rows[KubernetesExecutor.name] == ["No"] * (len(COMMAND_CAPABILITIES) + 1)


def test_command_capabilities_match_what_submit_requires() -> None:
    """`submit.py` asks for a capability by name at each command; the table must name the same one."""
    required = {}
    for node in ast.walk(ast.parse(inspect.getsource(submit))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "require" and len(node.args) == 3:
            _, capability, operation = node.args
            required.setdefault(operation.value, set()).add(capability.id)
    documented = {operation: {capability.__name__} for capability, operation in COMMAND_CAPABILITIES.values()}
    assert required == documented


def test_replace_generated_block_keeps_the_text_around_the_markers() -> None:
    page = f"before\n{START_MARKER}\nold\n{END_MARKER}\nafter\n"
    assert replace_generated_block(page, f"{START_MARKER}\nnew\n{END_MARKER}") == (
        f"before\n{START_MARKER}\nnew\n{END_MARKER}\nafter\n"
    )


def test_replace_generated_block_rejects_a_page_without_markers() -> None:
    with pytest.raises(ValueError, match="must contain"):
        replace_generated_block("no markers here", "x")
