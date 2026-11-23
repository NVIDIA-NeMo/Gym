# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed the agent's sandbox as ``SWERebenchResourcesServer._start_task_sandbox`` does, before the agent starts.

``setup.sh`` runs this file with the interpreter it recorded in ``/var/lib/nemo_gym/python_bin``. It calls the
server's own seed steps, vendored unchanged beside it, through ``local_sandbox.LocalSandbox``:

1. ``apply_anti_cheat_setup(sandbox, workdir, instance_id, "swe_rebench", required=...)``
   (``resources_servers/swebench/anti_cheat.py``) with the server's ``apply_anti_cheating`` and
   ``anti_cheat_required``. With ``required``, a scrub that did not complete raises, and the seed fails, as at the
   server (where the rollout is then retried);
2. ``prepare_git_for_commits`` (``resources_servers/swebench/patch_capture.py``): a committer identity, never fatal;
3. ``_pristine_untracked_files`` (``resources_servers/swebench/repo_task_sessions.py``, restated: it imports
   ``nemo_gym``): ``git ls-files --others --exclude-standard`` in the workdir with the server's 180 s budget; a
   listing that does not run or that git refuses fails the seed. The list goes to ``pristine_untracked.txt`` for the
   capture hook.

Exit 0 on success, 1 when the seed must fail.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shlex
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
# repo_task_sessions.SETUP_COMMAND_TIMEOUT_S
SETUP_COMMAND_TIMEOUT_S = 180


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


async def _seed() -> int:
    sys.path.insert(0, str(HERE))
    from local_sandbox import LocalSandbox

    config = json.loads((HERE / "config.json").read_text())
    anti_cheat = _load("nemo_gym_anti_cheat", HERE / "resources_servers/swebench/anti_cheat.py")
    patch_capture = _load("nemo_gym_patch_capture", HERE / "resources_servers/swebench/patch_capture.py")
    sandbox = LocalSandbox()
    workdir = config["workdir"]

    if config["apply_anti_cheating"]:
        try:
            await anti_cheat.apply_anti_cheat_setup(
                sandbox, workdir, config["instance_id"], "swe_rebench", required=config["anti_cheat_required"]
            )
        except anti_cheat.AntiCheatSetupError as error:
            print(f"[nemo_gym seed] {error}", file=sys.stderr, flush=True)
            return 1
    await patch_capture.prepare_git_for_commits(sandbox, workdir, "swe_rebench")

    result = await sandbox.exec(
        f"git -C {shlex.quote(workdir)} ls-files --others --exclude-standard", timeout_s=SETUP_COMMAND_TIMEOUT_S
    )
    if result.error_type is not None or result.return_code != 0:
        reason = result.error_type or f"exit {result.return_code}"
        print(
            f"[nemo_gym seed] could not list the pristine untracked files: {reason}: {(result.stderr or '')[-500:]}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    pristine = sorted({line.strip() for line in (result.stdout or "").splitlines() if line.strip()})
    (HERE / "pristine_untracked.txt").write_text("".join(f"{path}\n" for path in pristine))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_seed()))
