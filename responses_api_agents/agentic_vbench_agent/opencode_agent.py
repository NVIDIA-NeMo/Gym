# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode agent that uses an install baked into the task image when present.

Harbor's stock OpenCode agent downloads nvm, Node 22 and the opencode-ai package at
the start of every trial (~100 MB from GitHub and npm). At 128 concurrent episodes
that burst starved the backend's network and most trials hit the 360 s setup
timeout; it also makes every episode depend on two external services being up.
The SDG base images carry a prebaked install (scripts/agentic_vbench/sdg/
opencode_layer); this agent skips the download when the image has the requested
version and falls back to the stock install otherwise (benchmark tasks built from
public images, or a version bump before the images are rebuilt).
"""

from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment


class PrebakedOpenCode(OpenCode):
    async def _baked_version_matches(self, environment: BaseEnvironment) -> bool:
        probe = (
            '[ -s "$HOME/.nvm/nvm.sh" ] && . "$HOME/.nvm/nvm.sh" >/dev/null 2>&1 '
            "&& command -v opencode >/dev/null && opencode --version"
        )
        result = await environment.exec(command=probe)
        if result.return_code != 0:
            return False
        found = (result.stdout or "").strip().splitlines()
        installed = found[-1].strip() if found else ""
        if not installed:
            return False
        if self._version and installed != self._version:
            self.logger.info(f"Image has opencode {installed}, job wants {self._version}; installing")
            return False
        self.logger.info(f"Using opencode {installed} baked into the task image")
        return True

    async def install(self, environment: BaseEnvironment) -> None:
        if await self._baked_version_matches(environment):
            return
        await super().install(environment)
