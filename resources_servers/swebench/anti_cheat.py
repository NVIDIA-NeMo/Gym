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
"""Shared anti-cheating git-history scrub, run during ``seed_session`` before an agent gets
control of its sandbox.

Many SWE-bench-style images are built by checking out a repo that still has its full git
history -- including the commit(s) that contain the actual fix/tests used to build the "golden"
answer -- even though the working tree is reset to a pre-fix commit. Without this, an agent could
just run ``git log --all`` / ``git show <future-commit>`` and read the answer directly instead of
solving the task. ``anti_cheat_setup.sh`` (a single shared script, not duplicated per server)
rewrites HEAD onto an orphan-ish branch, deletes every other ref (tags, remotes, other branches,
packed-refs), expires the reflog, and runs ``git gc --prune=now`` to physically delete the
now-unreachable objects.

Originally inline in resources_servers/swebench_pro/app.py; factored out here once multiple
servers needed the identical upload+exec+cleanup sequence rather than each carrying its own copy.
"""

import asyncio
import sys
from pathlib import Path
from traceback import format_exc
from typing import Any


ANTI_CHEAT_SCRIPT_FPATH = Path(__file__).parent / "anti_cheat_setup.sh"
# The scrub is idempotent, so a provider error (a sandbox the gateway briefly could not reach) is retried: this many
# attempts in all, waiting ANTI_CHEAT_RETRY_DELAY_S, then twice that, between them.
ANTI_CHEAT_ATTEMPTS = 3
ANTI_CHEAT_RETRY_DELAY_S = 2.0


class AntiCheatSetupError(RuntimeError):
    """The scrub did not complete, so the sandbox's git history may still hold the task's solution."""


async def apply_anti_cheat_setup(
    sandbox: Any, workdir: str, instance_id: str, log_prefix: str, *, required: bool = False
) -> None:
    """Scrub the repository's git history before the agent gets the sandbox.

    A provider error is retried. By default a scrub that still fails is logged and the caller hands the (unscrubbed)
    sandbox to the agent; with ``required`` it raises :class:`AntiCheatSetupError` instead, so the seed fails and the
    rollout is retried rather than run against a history that holds the answer.
    """
    remote_fpath = f"{workdir.rstrip('/')}/anti_cheat_setup.sh"
    delay = ANTI_CHEAT_RETRY_DELAY_S
    for attempt in range(1, ANTI_CHEAT_ATTEMPTS + 1):
        try:
            await sandbox.upload(ANTI_CHEAT_SCRIPT_FPATH, remote_fpath)
            result = await sandbox.exec(
                f"git reset --hard && WORKING_DIRECTORY={workdir} bash anti_cheat_setup.sh && rm -f anti_cheat_setup.sh",
                cwd=workdir,
                timeout_s=600,
            )
        except Exception as error:
            if attempt < ANTI_CHEAT_ATTEMPTS:
                print(
                    f"[{log_prefix}] {instance_id}: anti-cheat setup raised (attempt {attempt}/{ANTI_CHEAT_ATTEMPTS}),"
                    f" retrying in {delay:g}s: {error!r}",
                    file=sys.stderr,
                )
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if required:
                raise AntiCheatSetupError(f"{instance_id}: anti-cheat setup raised: {error!r}") from error
            print(f"[{log_prefix}] {instance_id}: anti-cheat setup raised (non-fatal)", format_exc(), file=sys.stderr)
            return
        if result.return_code == 0:
            return
        message = (
            f"{instance_id}: anti-cheat setup failed. Return code: {result.return_code}\n"
            f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )
        if required:
            raise AntiCheatSetupError(message)
        print(f"[{log_prefix}] {message} (non-fatal)", file=sys.stderr)
        return
