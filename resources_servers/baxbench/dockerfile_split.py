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
"""Split an upstream BaxBench Dockerfile at the COPY of the generated code.

Every upstream framework Dockerfile installs its dependencies first and copies the generated code with
its last COPY. Everything before that COPY becomes the prebuilt framework image (``sandbox/images``);
the COPY and the steps after it run in the grading sandbox for each sample, as upstream runs them in
``docker build`` (``sandbox_runner.py``). One parser serves both, so the image and the per-sample build
cannot disagree about where the split is. Stdlib-only: this file is uploaded into the sandbox.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Instruction:
    keyword: str
    argument: str


@dataclass(frozen=True)
class SplitDockerfile:
    # Raw lines before the code COPY, ready to be written as the framework image's Dockerfile.
    image_lines: list[str]
    # WORKDIR and ENV in effect where the code COPY starts.
    workdir: str
    env: dict[str, str]
    # The code COPY and every instruction after it, in order.
    sample_steps: list[Instruction] = field(default_factory=list)


def _logical_lines(text: str) -> list[str]:
    lines, pending = [], ""
    for raw in text.splitlines():
        if pending:
            pending += "\n" + raw
        else:
            pending = raw
        if pending.rstrip().endswith("\\"):
            continue
        lines.append(pending)
        pending = ""
    if pending:
        lines.append(pending)
    return lines


def parse_env(argument: str) -> dict[str, str]:
    """``ENV KEY=VALUE`` (several pairs allowed) or the legacy ``ENV KEY VALUE``."""
    if "=" not in argument.split()[0]:
        key, _, value = argument.partition(" ")
        return {key: value.strip()}
    pairs = {}
    for token in argument.split():
        key, _, value = token.partition("=")
        pairs[key] = value.strip("\"'")
    return pairs


def split_dockerfile(text: str) -> SplitDockerfile:
    lines = _logical_lines(text)
    instructions: list[tuple[int, Instruction]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        keyword, _, argument = stripped.partition(" ")
        instructions.append((index, Instruction(keyword.upper(), argument.strip())))

    copies = [position for position, (_, instruction) in enumerate(instructions) if instruction.keyword == "COPY"]
    if not copies:
        raise ValueError("Dockerfile has no COPY of the generated code")
    code_copy = copies[-1]

    workdir, env = "/", {}
    for _, instruction in instructions[:code_copy]:
        if instruction.keyword == "WORKDIR":
            workdir = instruction.argument
        elif instruction.keyword == "ENV":
            env.update(parse_env(instruction.argument))
    first_line = instructions[code_copy][0]
    return SplitDockerfile(
        image_lines=lines[:first_line],
        workdir=workdir,
        env=env,
        sample_steps=[instruction for _, instruction in instructions[code_copy:]],
    )
