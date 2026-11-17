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
"""A Harbor task's own skills, put where a harness's CLI discovers them.

A task that declares ``skills_dir`` has its skills in the task sandbox: the ``harbor`` Environment Server uploads
``environment/skills/`` there at seed, unless the image already holds them, and writes the folder's absolute path to
``/tmp/.nemo-gym/task.json``. A harness that runs its CLI with a private configuration folder (Claude Code's
``CLAUDE_CONFIG_DIR``, Codex's ``CODEX_HOME``) does not see that folder, so at seed it copies the skills into the
CLI's own discovery folder with :func:`stage_task_skills`, and the model finds them through the CLI's skill mechanism.
"""

import json
import shlex

from nemo_gym.agents.base_responses_api_agent import AgentSeedRejectedError
from nemo_gym.sandbox.api import AsyncSandbox


# Where the Harbor Environment Server leaves the task's MCP servers and skills directory at seed.
TASK_CONTEXT_FILE = "/tmp/.nemo-gym/task.json"
_READ_TIMEOUT_S = 30
# A skills folder can hold supporting files, so the copy has more time than a read.
_COPY_TIMEOUT_S = 120


async def stage_task_skills(sandbox: AsyncSandbox, destination: str) -> str | None:
    """Copy the skills folder the task declares to ``destination``, inside the sandbox.

    Reads ``skills_dir`` from ``/tmp/.nemo-gym/task.json`` in the sandbox and copies that folder's contents into
    ``destination``, created if it is missing. Returns the ``skills_dir`` it copied, or ``None`` when the task
    declares none, so a task without skills (or a sandbox the Environment Server did not seed) is left untouched.

    An unusable task is refused with :class:`AgentSeedRejectedError`: a ``task.json`` that is not a JSON object, a
    ``skills_dir`` that is not an absolute path (the server makes a relative one absolute from the working directory,
    so only one a shell resolves reaches here, such as ``~/skills`` or ``$HOME/skills``, which cannot be copied from
    safely), or a folder that is not in the sandbox or cannot be copied. A declared folder that is empty is copied
    all the same, so the task counts as one that declares skills.
    A read or copy that did not report an exit code (a timeout or a provider failure) raises ``RuntimeError``,
    which a seed answers as a retryable failure.
    """
    context_path = shlex.quote(TASK_CONTEXT_FILE)
    # A file that exists but cannot be read fails the read instead of passing for an absent one.
    read = await sandbox.exec(
        f"if [ -e {context_path} ]; then cat {context_path}; fi", cwd="/", timeout_s=_READ_TIMEOUT_S
    )
    if read.error_type is not None or read.return_code != 0:
        raise RuntimeError(f"The sandbox could not read {TASK_CONTEXT_FILE}: {read.error_type or read.return_code}")
    text = (read.stdout or "").strip()
    if not text:
        return None
    try:
        context = json.loads(text)
    except json.JSONDecodeError:
        raise AgentSeedRejectedError(f"{TASK_CONTEXT_FILE} is not JSON") from None
    if not isinstance(context, dict):
        raise AgentSeedRejectedError(f"{TASK_CONTEXT_FILE} is not a JSON object")
    skills_dir = context.get("skills_dir")
    if skills_dir is None or skills_dir == "":
        return None
    if not isinstance(skills_dir, str) or not skills_dir.startswith("/"):
        raise AgentSeedRejectedError(
            f"The task's skills_dir {skills_dir!r} is not an absolute path, so its skills cannot be staged for the CLI"
        )
    source, target = shlex.quote(skills_dir), shlex.quote(destination)
    copied = await sandbox.exec(
        f"[ -d {source} ] || {{ echo 'not a folder in the sandbox' >&2; exit 2; }}; "
        f"mkdir -p -- {target} && cp -R -- {source}/. {target}/",
        cwd="/",
        timeout_s=_COPY_TIMEOUT_S,
    )
    if copied.error_type is not None:
        raise RuntimeError(f"Could not copy the task's skills_dir {skills_dir!r}: {copied.error_type}")
    if copied.return_code != 0:
        details = (copied.stderr or copied.stdout or "").strip()
        raise AgentSeedRejectedError(
            f"Could not stage the task's skills_dir {skills_dir!r} to {destination!r} "
            f"(exit {copied.return_code}): {details[-2000:]}"
        )
    return skills_dir
