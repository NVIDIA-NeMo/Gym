# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a BioML-bench task's public data in its OpenSandbox sandbox before the agent starts.

The sandbox follows BioML-bench's container: the public split at ``/home/data`` (read-only for the
``nonroot`` agent) and a submission checker on ``localhost:5000``. Unlike upstream, the private
split never stays here: a process in an OpenSandbox sandbox can run commands as root through the
sandbox's exec daemon, so the answers exist only in the separate verifier sandbox.
"""

from __future__ import annotations

import json
import os
import shlex

from responses_api_agents.harbor_agent.custom_envs.nemo_gym_sandbox.environment import NemoGymSandboxEnvironment


PYTHON = "/opt/conda/envs/biomlb/bin/python"
RUNTIME = "/opt/biomlbench/biomlbench_task.py"
TASK_JSON = "/opt/biomlbench/task.json"
DATA_DIR = "/private/data"
# OpenProblems preparers call `aws s3 cp --no-sign-request`; the image ships no AWS CLI.
AWSCLI = "awscli==1.35.0"
SANDBOX_CONFIG_ENV_VAR = "BIOMLBENCH_SANDBOX_CONFIG"


class BioMLBenchSandboxEnvironment(NemoGymSandboxEnvironment):
    """Start the upstream image, then prepare the task's public data and its checker as root."""

    def __init__(self, *args, setup_timeout_sec: int = 1800, **kwargs):
        self._setup_timeout_sec = setup_timeout_sec
        if "sandbox_provider" not in kwargs:
            # Passed only to the Ray task, never serialized into Harbor job/trial configs.
            kwargs["sandbox_provider"] = json.loads(os.environ[SANDBOX_CONFIG_ENV_VAR])
        super().__init__(*args, **kwargs)

    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        try:
            await self._prepare_task()
        except BaseException:
            await self.stop(delete=True)
            raise

    async def _run_as_root(self, command: str, what: str) -> None:
        result = await self.exec(command, timeout_sec=self._setup_timeout_sec, user="root")
        if result.return_code != 0:
            output = (result.stderr or "") + (result.stdout or "")
            raise RuntimeError(f"BioML-bench {what} failed (exit {result.return_code}): {output[-4000:]}")

    async def _prepare_task(self) -> None:
        task_id = json.loads((self.environment_dir / "task.json").read_text())["task_id"]
        await self._run_as_root("mkdir -p /opt/biomlbench", "runtime folder setup")
        await self.upload_file(self.environment_dir / "biomlbench_task.py", RUNTIME)
        await self.upload_file(self.environment_dir / "task.json", TASK_JSON)
        task = shlex.quote(task_id)
        if task_id.startswith("manual/open-problems-"):
            await self._run_as_root(f"{PYTHON} -m pip install --quiet {AWSCLI}", "AWS CLI install")
        # The verifier prepares the task again from the same pins; this copy keeps only the public split.
        await self._run_as_root(
            f"PATH=/opt/conda/envs/biomlb/bin:$PATH {PYTHON} {RUNTIME} prepare --task-id {task} "
            f"--data-dir {DATA_DIR} --public-dir /home/data --task-json {TASK_JSON} && rm -rf {DATA_DIR}",
            "data preparation",
        )
        # As BioML-bench's entrypoint.sh: everything in /home but the data belongs to the agent user.
        # The agent also writes its Harbor logs, which Harbor's Docker backend mounts writable.
        await self._run_as_root(
            "chmod -R a+rX,go-w /home/data && "
            r"find /home -path /home/data -prune -o -exec chown nonroot:nonroot {} + && "
            "mkdir -p /home/submission /home/logs && chown nonroot:nonroot /home/submission /home/logs /logs/agent",
            "permission setup",
        )
        await self._run_as_root(
            f"nohup {PYTHON} {RUNTIME} serve-validation --public-dir /home/data --port 5000 "
            "> /var/log/biomlbench-validation.log 2>&1 < /dev/null & "
            "for _ in $(seq 120); do curl -sf http://localhost:5000/health > /dev/null && exit 0; sleep 1; done; "
            "tail -n 50 /var/log/biomlbench-validation.log; exit 1",
            "validation server start",
        )
