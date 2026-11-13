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

"""Harbor's Terminus-2 loop over a local shell, shared by the host path and the sandbox runner.

The sandbox runner imports this module with only Harbor installed, so it must not import nemo_gym.
"""

import asyncio
import os
import shlex
import shutil
import signal
import sys
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext


try:
    from .llm import NemoGymLLM
except ImportError:  # The sandbox runner loads these modules from one directory, outside any package.
    from llm import NemoGymLLM


class LocalEnvironment:
    """Execute Harbor environment commands in one local workspace.

    ``base_env`` replaces this process's environment for the commands, so a runner can hand its tools the task's
    environment as it started. ``login_shell`` runs each command in a login shell.
    """

    def __init__(
        self,
        cwd: Path,
        command_timeout_sec: float,
        *,
        base_env: Mapping[str, str] | None = None,
        login_shell: bool = True,
    ) -> None:
        self.cwd = cwd.resolve()
        self.command_timeout_sec = command_timeout_sec
        self._base_env = dict(base_env) if base_env is not None else None
        self._shell_flags = "-lc" if login_shell else "-c"

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | float | None = None,
    ) -> ExecResult:
        if cwd:
            run_cwd = Path(cwd).resolve()
            if not run_cwd.is_dir():
                raise FileNotFoundError(f"Terminus-2 working directory does not exist: {run_cwd}")
        else:
            # A command typed in the terminal can remove the workdir; Terminus's tmux commands do not need it.
            run_cwd = self.cwd if self.cwd.is_dir() else Path("/")

        process_env = dict(self._base_env) if self._base_env is not None else os.environ.copy()
        if env:
            process_env.update({str(key): str(value) for key, value in env.items()})

        proc = await asyncio.create_subprocess_exec(
            "bash",
            self._shell_flags,
            command,
            cwd=str(run_cwd),
            env=process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        timeout = self.command_timeout_sec if timeout_sec is None else float(timeout_sec)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.CancelledError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.communicate()
            raise
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = await proc.communicate()
            message = stderr.decode(errors="replace")
            message += f"\nCommand timed out after {timeout:g} seconds"
            return ExecResult(stdout=stdout.decode(errors="replace"), stderr=message.strip(), return_code=124)

        return ExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=int(proc.returncode or 0),
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path).resolve()
        target = Path(target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source != target:
            await asyncio.to_thread(shutil.copy2, source, target)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self.upload_file(source_path, str(target_path))


class LocalTmuxSession(TmuxSession):
    """Use Harbor's Linux launcher and a BSD-compatible macOS launcher."""

    @property
    def _tmux_start_session(self) -> str:
        if sys.platform != "darwin":
            return super()._tmux_start_session

        session_name = shlex.quote(self._session_name)
        login_shell = shlex.quote("bash --login")
        pipe_command = shlex.quote(f"cat > {shlex.quote(str(self._logging_path))}")
        return (
            "export TERM=xterm-256color && "
            "export SHELL=/bin/bash && "
            f"tmux new-session -x {self._pane_width} -y {self._pane_height} -d "
            f"-s {session_name} {login_shell} \\; pipe-pane -t {session_name} {pipe_command}"
        )


class StandaloneTerminus2(Terminus2):
    """Run Terminus-2 in a unique tmux session."""

    # Harbor sends a turn again up to three times on any error, 4xx included, around NemoGymLLM.call's own retries.
    # Its undecorated method leaves NemoGymLLM.call's retry of transient model errors as the only layer.
    _query_llm = Terminus2._query_llm.__wrapped__

    async def _handle_llm_interaction(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Track Terminus-2's consecutive two-response completion handshake."""
        result = await super()._handle_llm_interaction(*args, **kwargs)
        _, is_task_complete, feedback = result[:3]
        if is_task_complete:
            self._consecutive_completion_claims = getattr(self, "_consecutive_completion_claims", 0) + 1
        elif "ERROR:" not in (feedback or ""):
            # Harbor skips a reply it could not parse without clearing a pending completion claim, so a claim, an
            # unparseable reply and a second claim still complete the task.
            self._consecutive_completion_claims = 0
        return result

    @property
    def finished_naturally(self) -> bool:
        return getattr(self, "_consecutive_completion_claims", 0) >= 2

    @property
    def run_error(self) -> Exception | None:
        """The error that ended the run early, which ``run`` logs and keeps instead of raising."""
        return getattr(self, "_run_error", None)

    @property
    def compactions(self) -> list[dict[str, Any]]:
        """Each Harbor summarization: when it started, how it ended, and the model responses it used."""
        return self.__dict__.setdefault("_compactions", [])

    async def run(self, instruction: str, environment: LocalEnvironment, context: AgentContext) -> None:
        """Keep completed turns when Terminus fails so Gym can score the partial trajectory."""
        self.context_length_exceeded = False
        self._run_error = None
        try:
            await super().run(instruction, environment, context)
        except Exception as exc:
            self._run_error = exc
            self.logger.info("Agent error: %s: %s. Returning completed turns.", type(exc).__name__, exc)
        finally:
            self._attach_routed_experts_to_trajectory()
            llm = getattr(self, "_llm", None)
            if isinstance(llm, NemoGymLLM):
                self.context_length_exceeded = llm.context_length_exceeded
                try:
                    await llm.aclose()
                except Exception:
                    pass

    async def _summarize(self, *args: Any, **kwargs: Any) -> Any:
        llm = getattr(self, "_llm", None)
        first_response = len(llm.response_ids) if isinstance(llm, NemoGymLLM) else 0
        record: dict[str, Any] = {"observed_at": time(), "outcome": "unknown", "response_ids": []}
        self.compactions.append(record)
        try:
            result = await super()._summarize(*args, **kwargs)
        except asyncio.CancelledError:
            record["outcome"] = "aborted"
            raise
        except BaseException:
            record["outcome"] = "failed"
            raise
        finally:
            if isinstance(llm, NemoGymLLM):
                record["response_ids"] = llm.response_ids[first_response:]
        record["outcome"] = "completed"
        return result

    def _attach_routed_experts_to_trajectory(self) -> None:
        llm = getattr(self, "_llm", None)
        if not isinstance(llm, NemoGymLLM):
            return

        modified = False
        for step in getattr(self, "_trajectory_steps", []):
            if getattr(step, "source", None) != "agent" or step.metrics is None:
                continue
            routed_experts = llm.pop_routed_experts_for_rollout_details(
                step.metrics.prompt_token_ids,
                step.metrics.completion_token_ids,
                step.metrics.logprobs,
            )
            if routed_experts is None:
                continue
            extra = step.metrics.extra or {}
            extra["routed_experts"] = routed_experts
            step.metrics.extra = extra
            modified = True

        if modified:
            self._dump_trajectory()

    async def setup(self, environment: LocalEnvironment) -> None:
        self._consecutive_completion_claims = 0
        self._standalone_session_name = f"terminus-2-{uuid4().hex[:12]}"
        recording_path = self.logs_dir / "recording.cast" if self._record_terminal_session else None
        self._session = LocalTmuxSession(
            session_name=self._standalone_session_name,
            environment=environment,
            logging_path=self.logs_dir / "terminus_2.pane",
            local_asciinema_recording_path=recording_path,
            remote_asciinema_recording_path=recording_path,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
        )
        await self._session.start()

    async def close(self, environment: LocalEnvironment) -> None:
        if self._session is not None:
            try:
                await self._session.stop()
            except Exception as exc:
                self.logger.warning("Could not stop Terminus-2 recording cleanly: %s", exc)
        session_name = getattr(self, "_standalone_session_name", None)
        if session_name:
            await environment.exec(
                f"tmux kill-session -t {shlex.quote(session_name)}",
                timeout_sec=30,
            )
