# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""mini-swe-agent environment adapter backed by the Gym sandbox API."""

import os
import shlex
from dataclasses import dataclass, field
from typing import Any


try:
    from minisweagent.exceptions import Submitted
except ModuleNotFoundError:

    class Submitted(Exception):
        """Compatibility shim for local mini-swe-agent versions before v2."""

        def __init__(self, *messages: dict[str, Any]) -> None:
            self.messages = messages
            super().__init__()


from nemo_gym.sandbox import Sandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.utils import rewrite_image


@dataclass
class MiniSWESandboxEnvironmentConfig:
    """Configuration for mini-swe-agent runs inside a sandbox."""

    image: str
    cwd: str = "/workspace"
    env: dict[str, str] = field(default_factory=dict)
    forward_env: list[str] = field(default_factory=list)
    timeout: int = 60
    step_timeout: int = 600
    eval_timeout: int = 1800
    interpreter: list[str] = field(default_factory=lambda: ["bash", "-c"])
    executable: str = "sandbox"
    run_args: list[str] = field(default_factory=list)
    start_args: list[str] = field(default_factory=list)
    container_timeout: str = "2h"
    instance_id: str | None = None
    provider: dict[str, Any] = field(default_factory=dict)
    spec: dict[str, Any] = field(default_factory=dict)
    conda_env: str | None = None
    activate_conda: bool = False
    user: str | int | None = "root"


class MiniSWESandboxEnvironment:
    """mini-swe-agent sync environment implemented with ``nemo_gym.sandbox.Sandbox``."""

    def __init__(
        self,
        *,
        config_class: type = MiniSWESandboxEnvironmentConfig,
        **kwargs: Any,
    ) -> None:
        self.config = config_class(**kwargs)
        if not self.config.provider:
            raise ValueError("MiniSWESandboxEnvironment requires provider")

        self._sandbox: Sandbox | None = None
        self._closed = False

        spec_config = dict(self.config.spec)
        image = spec_config.pop("image", None) or self.config.image
        image = rewrite_image(image, spec_config.pop("image_rewrites", []))
        provider_options = dict(spec_config.pop("provider_options", {}))

        env = dict(spec_config.pop("env", {}))
        for key in self.config.forward_env:
            value = os.getenv(key)
            if value is not None:
                env[key] = value
        env.update(self.config.env)

        self._sandbox = Sandbox(self.config.provider).start(
            SandboxSpec(
                image=image,
                ttl_s=spec_config.pop("ttl_s", None),
                ready_timeout_s=spec_config.pop("ready_timeout_s", None),
                workdir=spec_config.pop("workdir", self.config.cwd),
                env=env,
                files=spec_config.pop("files", {}),
                metadata={
                    **spec_config.pop("metadata", {}),
                    "nemo_gym_agent": "mini_swe_agent_2",
                    "instance_id": (self.config.instance_id or "unknown")[:63],
                },
                resources=SandboxResources.from_mapping(spec_config.pop("resources", {})),
                entrypoint=spec_config.pop("entrypoint", None),
                provider_options=provider_options,
            )
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {**self.config.__dict__, **kwargs}

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": self.config.__dict__,
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def _command(self, command: str) -> str:
        if not self.config.activate_conda or not self.config.conda_env:
            return command
        quoted_env = shlex.quote(self.config.conda_env)
        # Resolve conda from common install roots before activating. The sandbox exec
        # shell is non-login (apptainer/docker/ECS alike), so `conda` may not be on PATH
        # and `conda info --base` can't be relied on. The grouped loop (not an `&&` chain)
        # keeps a missing root from aborting the command; cwd is handled by exec(cwd=...),
        # so we don't `cd` here.
        return (
            '{ for __base in /opt/miniconda3 /opt/conda "$HOME/miniconda3" '
            '"$(command -v conda >/dev/null 2>&1 && conda info --base 2>/dev/null)"; do '
            '[ -n "$__base" ] && [ -f "$__base/etc/profile.d/conda.sh" ] && '
            '. "$__base/etc/profile.d/conda.sh" && break; done; } && '
            f"conda activate {quoted_env} && {command}"
        )

    def execute(
        self,
        action: dict[str, Any] | str,
        cwd: str = "",
        is_eval: bool = False,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        command = action.get("command", "") if isinstance(action, dict) else action
        timeout_s = timeout or (self.config.eval_timeout if is_eval else self.config.step_timeout)
        exec_cwd = cwd or self.config.cwd
        if self._sandbox is None:
            raise RuntimeError("Sandbox is not available")

        result = self._sandbox.exec(
            self._command(command),
            timeout_s=timeout_s,
            cwd=exec_cwd,
            user=self.config.user,
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        response = {
            "output": output,
            "returncode": result.return_code,
            "exception_info": "",
        }
        self._check_finished(response)
        return response

    def _check_finished(self, output: dict[str, Any]) -> None:
        """Match mini-swe-agent's submit sentinel handling for sandbox-backed runs."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._sandbox is not None:
            self._sandbox.stop()
            self._sandbox = None

    def __enter__(self) -> "MiniSWESandboxEnvironment":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    def __del__(self) -> None:  # pragma: no cover
        if hasattr(self, "_closed") and not self._closed:
            try:
                self.cleanup()
            except Exception:
                pass
