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

"""The table of what each executor supports, generated from the executor classes for the docs.

    python -m nemo_gym.orchestration.capability_matrix            # print the table
    python -m nemo_gym.orchestration.capability_matrix --write    # update the docs page in place

`tests/unit_tests/orchestration/test_capability_matrix.py` fails when the docs page differs from this output.
"""

import argparse
from pathlib import Path

from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.executors.capabilities import Inspectable, LogReadable, Resubmittable, Terminable
from nemo_gym.orchestration.submit import _EXECUTORS


DOCS_PAGE = Path(__file__).resolve().parents[2] / "fern/versions/latest/pages/orchestration/executors.mdx"
START_MARKER = "{/* capability-matrix:start */}"
END_MARKER = "{/* capability-matrix:end */}"

# `gym job` command to the capability and the `require` operation name `submit.py` asks for it with.
# `gym job list` reads the local index and needs no executor.
COMMAND_CAPABILITIES: dict[str, tuple[type, str]] = {
    "gym job status": (Inspectable, "status"),
    "gym job info": (Inspectable, "info"),
    "gym job cancel": (Terminable, "cancel"),
    "gym job logs": (LogReadable, "logs"),
    "gym job resubmit": (Resubmittable, "resubmit"),
}


def _executors() -> list[type[BaseExecutor]]:
    return sorted(_EXECUTORS.values(), key=lambda executor: executor.name)


def render_capability_matrix() -> str:
    """A Markdown table with a row per executor: whether it takes each `gym job` command and `resumable`."""
    header = ["Executor", *(f"`{command}`" for command in COMMAND_CAPABILITIES), "`resumable`"]
    rows = [header, ["---"] * len(header)]
    for executor in _executors():
        cells = [f"`{executor.name}`"]
        cells += ["Yes" if issubclass(executor, cap) else "No" for cap, _ in COMMAND_CAPABILITIES.values()]
        cells.append("Yes" if executor.supports_resumable else "No")
        rows.append(cells)
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def render_capability_list() -> str:
    """A bullet list of each capability interface, the commands that need it, and the executors that have it."""
    lines = []
    for capability in dict.fromkeys(cap for cap, _ in COMMAND_CAPABILITIES.values()):
        commands = ", ".join(f"`{cmd}`" for cmd, (cap, _) in COMMAND_CAPABILITIES.items() if cap is capability)
        executors = ", ".join(f"`{e.name}`" for e in _executors() if issubclass(e, capability)) or "none"
        lines.append(f"- `{capability.__name__}`: {commands}. Implemented by {executors}.")
    return "\n".join(lines)


def render_generated_block() -> str:
    return f"{START_MARKER}\n\n{render_capability_matrix()}\n\n{render_capability_list()}\n\n{END_MARKER}"


def replace_generated_block(page: str, block: str) -> str:
    """`page` with the text between the markers (markers included) replaced by `block`."""
    start, end = page.find(START_MARKER), page.find(END_MARKER)
    if start == -1 or end < start:
        raise ValueError(f"{DOCS_PAGE} must contain {START_MARKER} followed by {END_MARKER}")
    return page[:start] + block + page[end + len(END_MARKER) :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help=f"update {DOCS_PAGE} instead of printing")
    args = parser.parse_args()
    if args.write:
        DOCS_PAGE.write_text(replace_generated_block(DOCS_PAGE.read_text(), render_generated_block()))
    else:
        print(render_generated_block())


if __name__ == "__main__":
    main()
