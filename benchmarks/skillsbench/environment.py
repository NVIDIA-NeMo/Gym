# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install official SkillsBench skills into a prebuilt OpenSandbox task image."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from harbor.environments.base import ExecResult
    from harbor.models.task.config import NetworkPolicy

from benchmarks.skillsbench.observations import VERIFIER_EXECUTION_FILE, verifier_command
from responses_api_agents.harbor_agent.custom_envs.nemo_gym_sandbox.environment import NemoGymSandboxEnvironment


class SkillsBenchSandboxEnvironment(NemoGymSandboxEnvironment):
    """Use upstream task images and stage skills when a sandbox starts."""

    def __init__(self, *args, skill_mode: str = "with-skill", **kwargs):
        if skill_mode not in {"with-skill", "no-skill"}:
            raise ValueError(f"Invalid SkillsBench skill mode: {skill_mode!r}")
        self._skill_mode = skill_mode
        if "sandbox_provider" not in kwargs:
            # Passed only to the Ray task, never serialized into Harbor job/trial configs.
            kwargs["sandbox_provider"] = json.loads(os.environ["SKILLSBENCH_SANDBOX_CONFIG"])
        super().__init__(*args, **kwargs)

    def validate_network_policy_support(self, network_policy: NetworkPolicy | None = None) -> None:
        policy = network_policy or self.network_policy
        if policy.network_mode == "no-network" and self._allow_unenforced_internet_isolation:
            self.logger.warning("SkillsBench no-network task is running with unenforced isolation by explicit opt-in")
            return
        super().validate_network_policy_support(policy)

    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        try:
            await self._stage_skills()
        except BaseException:
            await self.stop(delete=True)
            raise

    async def exec(self, command: str, *args, **kwargs) -> ExecResult:
        # The agent executor wraps every action in its own shell. Only Harbor's
        # direct verifier invocation matches this command, including redirection.
        # Keep the witness on the host, outside downloaded sandbox log trees.
        verifying = command == verifier_command()
        if verifying:
            witness = self.trial_paths.result_path.parent / VERIFIER_EXECUTION_FILE
            witness.unlink(missing_ok=True)
        started = time.time()
        result = await super().exec(command, *args, **kwargs)
        if verifying and result.return_code == -1:
            await self.stop(delete=True)
            raise RuntimeError("Verifier command outcome is unknown; remote termination is unconfirmed")
        if verifying:
            witness.write_text(
                json.dumps(
                    {
                        "command": command,
                        "return_code": result.return_code,
                        "started_at": started,
                        "finished_at": time.time(),
                    }
                )
            )
        return result

    async def _stage_skills(self) -> None:
        if self._skill_mode == "no-skill":
            # Published task images are shared by both conditions. Some task
            # Dockerfiles bake skills into agent paths, so remove those copies.
            result = await self.exec(
                "rm -rf /root/.agents/skills /root/.codex/skills "
                "/root/.claude/skills /root/.gemini/skills "
                "/root/.opencode/skills /root/.opencode/skill "
                "/root/.goose/skills /root/.factory/skills "
                "/root/.openhands/skills",
                timeout_sec=300,
            )
            if result.return_code != 0:
                raise RuntimeError(f"Failed to remove preinstalled skills: {result.stderr or result.stdout}")
            return

        skills_dir = self.environment_dir / "skills"
        if not skills_dir.is_dir():
            raise FileNotFoundError(f"SkillsBench skills directory is missing: {skills_dir}")

        await self.upload_dir(skills_dir, "/root/.agents/skills")
        result = await self.exec(
            "for location in /root/.codex/skills /root/.claude/skills "
            "/root/.gemini/skills /root/.opencode/skills; do "
            'mkdir -p "$location" && cp -a /root/.agents/skills/. "$location/" || exit; '
            "done",
            timeout_sec=300,
        )
        if result.return_code != 0:
            raise RuntimeError(f"Failed to install SkillsBench skills: {result.stderr or result.stdout}")
