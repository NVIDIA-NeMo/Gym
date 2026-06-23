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
        """Signals that the agent has submitted its final output, carrying the submission messages.

        Used when the installed mini-swe-agent does not provide its own ``Submitted`` exception.
        """

        def __init__(self, *messages: dict[str, Any]) -> None:
            """Store the submission messages on the exception.

            Args:
                *messages: The message dicts describing the submission.
            """
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
        """Build the environment config and start the backing sandbox.

        Resolves the image (with rewrites), assembles the environment variables and resources from
        the spec and config, and starts a sandbox for the task.

        Args:
            config_class: The dataclass used to construct the environment config from ``kwargs``.
            **kwargs: Fields forwarded to ``config_class`` (image, cwd, env, provider, spec, etc.).

        Raises:
            ValueError: If no sandbox provider is configured.
        """
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
        """Return the variables available for prompt/command templating.

        Args:
            **kwargs: Extra variables to merge over the config fields.

        Returns:
            A dict combining the config fields with the provided overrides.
        """
        return {**self.config.__dict__, **kwargs}

    def serialize(self) -> dict[str, Any]:
        """Serialize the environment configuration for trajectory records.

        Returns:
            A nested dict describing the environment config and its fully qualified type name.
        """
        return {
            "info": {
                "config": {
                    "environment": self.config.__dict__,
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def _command(self, command: str) -> str:
        """Wrap a command to activate the configured conda env when enabled.

        Args:
            command: The shell command to run.

        Returns:
            The command prefixed with conda activation, or the unchanged command when conda
            activation is disabled or no env is configured.
        """
        if not self.config.activate_conda or not self.config.conda_env:
            return command
        quoted_env = shlex.quote(self.config.conda_env)
        return f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {quoted_env} && {command}"

    def execute(
        self,
        action: dict[str, Any] | str,
        cwd: str = "",
        is_eval: bool = False,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute a command in the sandbox and return its combined output and return code.

        Args:
            action: The command to run, either a string or a dict with a ``command`` key.
            cwd: The working directory; defaults to the configured cwd when empty.
            is_eval: Whether this is an eval command, which selects the eval timeout.
            timeout: An explicit timeout in seconds; overrides the step/eval default when set.

        Returns:
            A dict with ``output`` (merged stdout and stderr), ``returncode``, and ``exception_info``.

        Raises:
            RuntimeError: If the sandbox is not available.
            Submitted: If the command output signals a final submission.
        """
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
        """Raise ``Submitted`` when the command output begins with the submit sentinel.

        Args:
            output: The execute-result dict whose ``output`` and ``returncode`` are inspected.

        Raises:
            Submitted: If the first output line is the submit sentinel and the return code is zero.
        """
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
        """Stop the backing sandbox and mark the environment closed.

        Idempotent: subsequent calls return immediately once the environment is closed.
        """
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
