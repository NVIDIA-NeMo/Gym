# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a ``[[steps]]`` turn needs in the sandbox before its agent session, as Harbor prepares a step.

Harbor 0.22.0's ``MultiStepTrial._prepare_step`` (trial/multi_step.py) runs, as the step's agent user: the step's
``steps/<name>/workdir/`` uploaded into the environment's working directory (``_upload_step_workdir``), then
``bash <workdir>/setup.sh`` when the folder ships one (``_run_step_setup``), then the step's own healthcheck
(``_run_step_healthcheck``). A setup or healthcheck that fails is the step's ``exception_info``: the step runs no
agent and no verifier, and the trial stops after it. The mixin here does the same for one turn of the environment
server; :class:`StepPrepareError` is that failure, which the turn loop records as a failed turn.
"""

import logging
import shlex

from environment_servers.harbor.session import HarborSession
from nemo_gym.sandbox.transfer import SandboxProbeError, SandboxTransferError, upload_dir
from nemo_gym.tasks.harbor.digest import is_ignored
from nemo_gym.tasks.harbor.task import HarborTask, HarborTurn


LOGGER = logging.getLogger(__name__)

# The step's `workdir/` could not be merged into the working directory (an entry the sandbox refused to overwrite).
STEP_WORKDIR_FAILED_KIND = "harbor:step_workdir_failed"
# The step's `workdir/setup.sh` exited non-zero (or ran past its bound) before the turn's agent session.
STEP_SETUP_FAILED_KIND = "harbor:step_setup_failed"


class StepPrepareError(Exception):
    """A step's own preparation failed: the turn is recorded failed, with ``kind`` as its ``failure_kind``, and the
    episode stops there, as Harbor stops a trial after a step whose preparation raised."""

    def __init__(self, kind: str, reason: str) -> None:
        super().__init__(reason)
        self.kind = kind
        self.reason = reason


class HarborStepsMixin:
    """The per-turn preparation Harbor's multi-step trial does between the previous step and the agent."""

    def _step_setup_timeout_s(self, task: HarborTask, turn: HarborTurn) -> float:
        """The bound on a step's ``setup.sh``: ``step_setup_timeout_s``, else the step's own agent budget
        (``[steps.agent].timeout_sec``, else ``[agent].timeout_sec``), the task's own bound on that step's time.
        Harbor has no bound of its own there but the trial's timeout."""
        if self.config.step_setup_timeout_s is not None:
            return self.config.step_setup_timeout_s
        return task.config.step_agent_timeout_sec(turn.config)

    def _step_setup_budget_s(self, task: HarborTask, turn: HarborTurn) -> float:
        """What a step's setup adds to the episode budget: its bound when the step ships a ``setup.sh``, else 0."""
        return self._step_setup_timeout_s(task, turn) if (task.path / turn.setup_script).is_file() else 0.0

    async def _prepare_step_workdir(self, session: HarborSession, turn: HarborTurn) -> None:
        """Merge ``steps/<name>/workdir/`` into the working directory and run its ``setup.sh``, as Harbor does.

        The folder is uploaded over the working directory as it stands (a merge: what the previous turns left
        there stays unless an entry of the same name replaces it), extracted as root with the folder's own modes,
        as Harbor's ``docker cp`` upload lands it, and the names the task digest skips are left out as for every
        task folder. The working directory is the previous turns' agents' own, so a symbolic link one of them
        planted at the name of a shipped entry (``ln -s /etc/profile.d scripts`` where the step ships
        ``scripts/``) is removed, as root, right before the extraction (``upload_dir(replace_links=True)``):
        ``docker cp`` and GNU tar replace such a link with the entry, BusyBox tar (Alpine images) would follow it
        and write the shipped files as root wherever it points. The removal and the extraction are two commands of
        one shell line, so a process an earlier turn left running can plant the link again between them: a window
        of the same shape as ``docker cp``'s, which moves where the shipped files land and not what they hold
        (:func:`tar_extract_command` says why a staged copy would not close it). ``setup.sh`` then runs as ``bash <workdir>/setup.sh`` from the working directory, as the
        sandbox's default user at that point, which is the turn's agent user (Harbor runs both under
        ``with_default_user(step agent user)``), within :meth:`_step_setup_timeout_s` (Harbor bounds it by nothing
        but the trial's own timeout). A non-zero exit, or a run past that bound, is the step's failure
        (:class:`StepPrepareError`, kind :data:`STEP_SETUP_FAILED_KIND`), as Harbor records it in the step's
        ``exception_info``; a sandbox that cannot run the script at all is an outage and raises. An entry of the
        folder the sandbox refuses to unpack over what is there (a non-empty directory where a file is shipped) is
        the step's failure too (kind :data:`STEP_WORKDIR_FAILED_KIND`), as Harbor's ``docker cp`` refuses it and the
        trial errors; the sandbox itself is sound, so it is not an outage to retry.

        The script runs as the turn's agent user in a directory that user owns, after the previous turns' agents
        ran there: a process an earlier turn left running can replace the uploaded ``setup.sh`` before this exec
        reads it, so a step whose agent is root after a step whose agent was not runs, in that window, whatever that
        earlier agent left in its place (Harbor shares the window: ``docker cp``, then ``exec``; the loader warns
        about such a task). The step's own agent gets root right after, so the window gains the agent nothing it
        is not about to be given; it is noted here so that a task author does not rely on ``setup.sh`` to undo
        what an earlier, less privileged turn did.
        """
        task = session.task
        source = task.path / turn.workdir_dir
        if not source.is_dir():
            return
        try:
            await upload_dir(session.sandbox, source, session.workdir, ignore=is_ignored, replace_links=True)
        except SandboxTransferError as exc:
            # tar refused an entry: a non-empty directory at the name of a shipped file, which the previous turn's
            # agent may have planted in the working directory it owns. `docker cp` refuses the same mismatch
            # (Harbor's trial errors there and is not retried), so it is the step's failure, not a sandbox outage:
            # retrying would hit the same tree. (GNU tar lets a shipped directory replace an existing file, where
            # `docker cp` refuses; the folder then lands as declared.)
            raise StepPrepareError(
                STEP_WORKDIR_FAILED_KIND,
                f"step {turn.name!r} workdir/ could not be merged into {session.workdir}: {exc}",
            ) from exc
        if not (task.path / turn.setup_script).is_file():
            return
        script = f"{session.workdir.rstrip('/')}/{turn.setup_script.name}"
        bound = self._step_setup_timeout_s(task, turn)
        result = await session.sandbox.exec(f"bash {shlex.quote(script)}", cwd=session.workdir, timeout_s=bound)
        if result.error_type == "timeout":
            raise StepPrepareError(
                STEP_SETUP_FAILED_KIND,
                f"step {turn.name!r} setup.sh ran past its {bound:g}s bound (step_setup_timeout_s, else the step's "
                "agent budget)",
            )
        if result.error_type:
            raise SandboxProbeError(f"step {turn.name!r} setup.sh could not run: {result.error_type}")
        if result.return_code != 0:
            raise StepPrepareError(
                STEP_SETUP_FAILED_KIND,
                f"step {turn.name!r} setup.sh exited with code {result.return_code}: "
                f"{(result.stderr or result.stdout or '')[-600:]}",
            )
