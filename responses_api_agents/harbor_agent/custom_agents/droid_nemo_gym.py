# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""NeMo Gym Droid wrapper for TB2 evaluation via Harbor.

Droid is a BaseInstalledAgent — it runs as a binary inside Docker containers and
calls the model server directly.
"""

import json
import logging
import os
import shlex
import socket
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


logger = logging.getLogger(__name__)


# Safety denylist from Droid's recommended configuration.
_COMMAND_DENYLIST = [
    "rm -rf /",
    "rm -rf /*",
    "rm -rf .",
    "rm -rf ~",
    "rm -rf ~/*",
    "rm -rf $HOME",
    "rm -r /",
    "rm -r /*",
    "rm -r ~",
    "rm -r ~/*",
    "mkfs",
    "mkfs.ext4",
    "mkfs.ext3",
    "mkfs.vfat",
    "mkfs.ntfs",
    "dd if=/dev/zero of=/dev",
    "dd of=/dev",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init 0",
    "init 6",
    ":(){ :|: & };:",
    ":() { :|:& };:",
    "chmod -R 777 /",
    "chmod -R 000 /",
    "chown -R",
    "Format-Volume",
    "format.com",
    "powershell Remove-Item -Recurse -Force",
]


def _get_host_ip() -> str:
    """Get the host IP so containers can reach host services."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def _rewrite_localhost(settings_json: str) -> str:
    """Replace localhost/loopback/wildcard URLs with host IP so Docker containers can reach host services."""
    host_ip = _get_host_ip()
    return (
        settings_json
        .replace("localhost", host_ip)
        .replace("127.0.0.1", host_ip)
        .replace("0.0.0.0", host_ip)
    )


class DroidNemoGym(BaseInstalledAgent):
    """Droid agent wrapper for NeMo Gym TB2 evaluation.

    Dynamically generates Droid's ~/.factory/settings.json from api_base and
    model_name provided by NeMo Gym's agent config, instead of relying on a
    host-side settings file.
    """

    # Minimum droid version that supports FACTORY_API_KEY=EMPTY for custom models.
    _DEFAULT_DROID_VERSION = "0.65.0"

    def __init__(
        self,
        *args: Any,
        api_base: str | None = None,
        api_key: str = "not-needed",
        droid_custom_model_id: str | None = None,
        droid_version: str | None = None,
        droid_bundle_path: str | None = None,
        droid_autonomy_mode: str = "auto-low",
        enable_thinking: bool = True,
        max_output_tokens: int = 32768,
        max_context_limit: int = 131072,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._api_base = api_base
        self._api_key = api_key
        self._droid_custom_model_id = droid_custom_model_id
        self._droid_version = droid_version or self._DEFAULT_DROID_VERSION
        self._droid_bundle_path = droid_bundle_path
        self._droid_autonomy_mode = droid_autonomy_mode
        self._enable_thinking = enable_thinking
        self._max_output_tokens = max_output_tokens
        self._max_context_limit = max_context_limit

    @staticmethod
    def name() -> str:
        return "droid-nemo-gym"

    def version(self) -> str | None:
        return self._droid_version

    @property
    def _install_agent_template_path(self) -> Path:
        return Path(__file__).parent / "droid-setup.sh.j2"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install droid inside the container.

        When ``droid_bundle_path`` is set, uploads a pre-built tarball
        containing nvm + Node + droid and extracts it (~5s vs ~2-3min for a
        fresh npm install).  Falls back to the default template-based install
        if no bundle is configured.
        """
        if self._droid_bundle_path:
            bundle = Path(self._droid_bundle_path)
            if not bundle.exists():
                raise FileNotFoundError(f"Droid bundle not found: {bundle}")

            logger.info("Uploading droid bundle (%s) to container", bundle)
            await environment.upload_file(
                source_path=bundle,
                target_path="/tmp/droid-bundle.tar.gz",
            )
            result = await environment.exec(
                command=(
                    "apt-get update -qq && apt-get install -y -qq curl xdg-utils procps > /dev/null 2>&1 ; "
                    "tar xzf /tmp/droid-bundle.tar.gz -C $HOME && rm /tmp/droid-bundle.tar.gz"
                ),
            )

            setup_dir = self.logs_dir / "setup"
            setup_dir.mkdir(parents=True, exist_ok=True)
            (setup_dir / "return-code.txt").write_text(str(result.return_code))
            if result.stdout:
                (setup_dir / "stdout.txt").write_text(result.stdout)
            if result.stderr:
                (setup_dir / "stderr.txt").write_text(result.stderr)
            if result.return_code != 0:
                raise RuntimeError(
                    f"Droid bundle extraction failed with exit code {result.return_code}. See logs in {setup_dir}"
                )
        else:
            await super().setup(environment)

    def populate_context_post_run(self, context: AgentContext) -> None:
        pass

    def _get_droid_custom_id(self) -> str:
        """Return the Droid custom model ID for CLI matching.

        Droid resolves ``--model custom:<id>`` by matching against the ``id``
        field in settings.json (must start with ``custom:``).  The ``model``
        field is separate — it's sent as the ``model`` parameter in API requests.
        """
        if self._droid_custom_model_id:
            return self._droid_custom_model_id
        return self.model_name or "nemo-gym-model"

    def _build_settings_json(self) -> str:
        """Generate Droid settings.json from NeMo Gym config."""
        model_name = self.model_name or "nemo-gym-model"
        custom_id = self._get_droid_custom_id()
        base_url = self._api_base or "http://localhost:8000/v1"

        settings = {
            "cloudSessionSync": False,
            "includeCoAuthoredByDroid": False,
            "enableDroidShield": False,
            "commandDenylist": _COMMAND_DENYLIST,
            "modelPolicy": {
                "allowedModelIds": [],
                "blockedModelIds": [],
                "allowCustomModels": True,
                "allowAllFactoryModels": False,
            },
            "customModels": [
                {
                    "id": f"custom:{custom_id}",
                    "model": model_name,
                    "baseUrl": base_url,
                    "apiKey": self._api_key,
                    "displayName": f"NeMo Gym: {model_name}",
                    "maxContextLimit": self._max_context_limit,
                    "enableThinking": self._enable_thinking,
                    "maxOutputTokens": self._max_output_tokens,
                    "noImageSupport": True,
                    "provider": "generic-chat-completion-api",
                },
            ],
            "sessionDefaultSettings": {
                "model": f"custom:{custom_id}",
                "autonomyMode": self._droid_autonomy_mode,
            },
        }

        settings_json = json.dumps(settings, indent=2)
        return _rewrite_localhost(settings_json)

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        escaped_instruction = shlex.quote(instruction)
        # Droid CLI model ID: custom:<id> matching the id field in settings.json.
        custom_id = self._get_droid_custom_id()
        cli_model_id = f"custom:{custom_id}"

        env = {
            "FACTORY_API_KEY": os.environ.get("FACTORY_API_KEY", "EMPTY"),
        }

        settings_json = self._build_settings_json()
        # Use base64 encoding to avoid triggering command safety filters
        # (the JSON denylist contains words like "shutdown" that match patterns).
        import base64

        b64_settings = base64.b64encode(settings_json.encode()).decode()
        setup_command = f"mkdir -p $HOME/.factory && echo {b64_settings} | base64 -d > $HOME/.factory/settings.json"

        # Source nvm so droid is on PATH (nvm is installed in setup but the
        # exec shell doesn't inherit that environment).  Use ``--`` before the
        # instruction to prevent droid's CLI parser from treating leading
        # hyphens in task text as flags.
        #
        # Use stream-json output format to capture the full conversation trace
        # (tool calls, messages, completions) as streaming JSONL in stdout.
        # After droid exits, also dump session files from ~/.factory/sessions/.
        droid_and_trace = (
            f"source $HOME/.nvm/nvm.sh 2>/dev/null ; "
            f"droid exec "
            f"--skip-permissions-unsafe "
            f"--output-format stream-json "
            f"--model {shlex.quote(cli_model_id)} "
            f"-- {escaped_instruction}"
            f" ; echo '\\n=== DROID_SESSION_TRACE ===' ; "
            f"latest_session=$(ls -td $HOME/.factory/sessions/*/ 2>/dev/null | head -1) && "
            f'if [ -n "$latest_session" ]; then '
            f'for f in "$latest_session"*.json; do '
            f'echo "=== $(basename "$f") ===" && cat "$f"; '
            f"done; fi"
        )

        return [
            ExecInput(
                command=setup_command,
                env=env,
            ),
            ExecInput(
                command=droid_and_trace,
                env=env,
            ),
        ]
