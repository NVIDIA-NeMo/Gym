# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional npm installer for Pier's otherwise unchanged Claude Code agent."""

from pier.agents.installed.claude_code import ClaudeCode
from pier.models.agent.install import AgentInstallSpec, InstallStep


class ClaudeCodeNpmInstall(ClaudeCode):
    """Use Claude Code's exact-version npm package on CPUs unsupported by Bun."""

    def install_spec(self) -> AgentInstallSpec:
        package_version = f"@{self._version}" if self._version else ""
        ensure_npm = (
            "if ! command -v npm >/dev/null 2>&1; then "
            "if command -v apk >/dev/null 2>&1; then apk add --no-cache bash nodejs npm; "
            "elif command -v apt-get >/dev/null 2>&1; then "
            "export DEBIAN_FRONTEND=noninteractive; apt-get update && apt-get install -y nodejs npm; "
            "elif command -v yum >/dev/null 2>&1; then yum install -y nodejs npm; "
            "else echo 'npm is required for Claude Code installation' >&2; exit 1; fi; fi"
        )
        install = f"set -euo pipefail; npm install -g @anthropic-ai/claude-code{package_version}; claude --version"
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(user="root", run=ensure_npm),
                InstallStep(user="agent", run=install),
            ],
            verification_command=self.get_version_command(),
        )


__all__ = ["ClaudeCodeNpmInstall"]
