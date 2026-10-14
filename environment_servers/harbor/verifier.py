# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grade an episode: the test.sh wrapper protocol, the reward files and the verifier sandbox.

The verifier runs Harbor's tests/test.sh under a timeout wrapper that writes an exit record,
reads reward.json or reward.txt, applies Harbor's reward selection rule and turns the exit into
a measured or masked sample. Module level are the in-sandbox paths the protocol uses, the
failure kinds, the wrapper command, the reward and exit record parsers, the timeout rule, and
the choice of image and user the verifier runs as in shared and separate mode. The mixin holds
the shared and separate verification choreography unchanged, the reward and log collection,
the collect hooks and artifact download, and the scoring methods. Creating the verifier
sandbox stays with the other provisioning code in sandboxes.py.
"""

import asyncio
import errno
import json
import logging
import math
import shlex
import tarfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from environment_servers.harbor.overlay import OVERLAY_FAILED_KIND, HarborOverlayError
from environment_servers.harbor.session import (
    AGENT_LOGS_DIR,
    ARTIFACTS_DIR,
    VERIFIER_LOGS_DIR,
    HarborSession,
    resolve_env_templates,
)
from nemo_gym import failure_kinds
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from nemo_gym.sandbox.transfer import (
    SandboxProbeError,
    SandboxTransferError,
    download_dir,
    download_path,
    exec_as_root,
    remote_kind,
    upload_archive,
    upload_dir,
)
from nemo_gym.tasks.harbor import HarborTask
from nemo_gym.tasks.harbor.models import HarborArtifact


LOGGER = logging.getLogger(__name__)

TESTS_DIR = "/tests"

VERIFIER_STDOUT = f"{VERIFIER_LOGS_DIR}/test-stdout.txt"
# Written by the verifier wrapper once `timeout` returns: the exit code and the start and end times
# (epoch seconds) of the run, so the server can tell the timer firing from a test.sh that exits 137.
EXIT_RECORD_NAME = ".ng-exit"
# Runs before any redirection: an image without `timeout` (or `bash`) exits 127 here, leaving neither a
# stdout file nor an exit record, which is how the caller tells it from a script that exits 127 itself.
TIMEOUT_WRAPPER_GUARD = "command -v timeout >/dev/null 2>&1 && command -v bash >/dev/null 2>&1 || exit 127"

# Namespaced failure kinds for outcomes the shared vocabulary does not name.
VERIFIER_TIMEOUT_KIND = "harbor:verifier_timeout"
MISSING_REWARD_KIND = "harbor:missing_reward"
INVALID_REWARD_KIND = "harbor:invalid_reward"


def parse_reward_file(directory: Path) -> tuple[dict[str, float] | None, str | None]:
    """Read reward.json when present; only a missing JSON file falls back to reward.txt."""
    for path in (directory / "reward.json", directory / "reward.txt"):
        if path.is_file():
            return _parse_one_reward_file(path)
    return None, "no reward.json or reward.txt was written"


def _parse_one_reward_file(path: Path) -> tuple[dict[str, float] | None, str | None]:
    # The verifier wrote these bytes; never let a stray non-UTF-8 byte escape as an exception.
    text = path.read_text(errors="replace").strip()
    if not text:
        return None, f"{path.name} is empty"
    try:
        raw = json.loads(text) if path.suffix == ".json" else {"reward": float(text)}
    except ValueError as exc:
        return None, f"{path.name} is not valid: {exc}"
    if not isinstance(raw, dict) or not raw:
        return None, f"{path.name} must hold a non-empty JSON object"
    rewards: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(value, (int, float)):
            return None, f"{path.name}: {key!r} is not a finite number"
        try:
            value = float(value)
        except OverflowError:
            return None, f"{path.name}: {key!r} is not a finite number"
        if not math.isfinite(value):
            return None, f"{path.name}: {key!r} is not a finite number"
        rewards[str(key)] = value
    return rewards, None


def verifier_command(budget: int) -> str:
    """One in-container run of ``tests/test.sh`` under ``timeout``, with the exit record written after it."""
    return (
        f"{TIMEOUT_WRAPPER_GUARD}; start=$(date +%s); "
        f"timeout -s KILL {budget} bash {TESTS_DIR}/test.sh > {VERIFIER_STDOUT} 2>&1; rc=$?; "
        f'echo "$rc $start $(date +%s)" > {VERIFIER_LOGS_DIR}/{EXIT_RECORD_NAME}; exit $rc'
    )


async def _make_test_script_executable(sandbox: AsyncSandbox) -> None:
    """``chmod +x`` on ``test.sh`` as root before it runs, as Harbor's verifier does.

    Harbor uploads ``tests/`` as ``docker cp`` leaves it (root-owned, the package's modes), chmods the test
    script alone because it is a ``.sh`` (``needs_chmod``), and runs it as the verifier user; it chowns
    nothing and widens nothing, so a package built under ``umask 077`` is readable by a root verifier and by
    no other user, there as here. The chmod's exit status is not read, as Harbor does not read it.
    """
    await exec_as_root(sandbox, f"chmod +x {TESTS_DIR}/test.sh", timeout_s=60)


def parse_exit_record(directory: Path) -> tuple[int, int | None] | None:
    """``(exit code, elapsed seconds)`` from the wrapper's record, or None when the wrapper never finished.

    The elapsed time is None when the record carries no usable timestamps (an image without ``date``).
    """
    path = directory / EXIT_RECORD_NAME
    if not path.is_file():
        return None
    parts = path.read_text(errors="replace").split()
    try:
        return_code = int(parts[0])
    except (IndexError, ValueError):
        return None
    try:
        return return_code, int(parts[2]) - int(parts[1])
    except (IndexError, ValueError):
        return return_code, None


def verifier_timed_out(result: SandboxExecResult, exit_record: tuple[int, int | None] | None, budget: int) -> bool:
    """Whether the timer stopped test.sh: the provider gave up first, or ``timeout`` killed it at the budget.

    GNU ``timeout -s KILL`` exits 137 when the timer fired, but so does a script whose own helper was
    killed; the timer is the cause only once the run has lasted the whole budget. An image without
    ``date`` leaves no elapsed time, and 137 then means the timer.
    """
    if result.error_type == "timeout":
        return True
    if exit_record is None or exit_record[0] != 137:
        return False
    elapsed = exit_record[1]
    return elapsed is None or elapsed >= budget


def select_reward(rewards: dict[str, float]) -> float | None:
    """Harbor's rule: the ``reward`` key, else the only key. Several keys and none named ``reward`` is ambiguous."""
    if "reward" in rewards:
        return rewards["reward"]
    if len(rewards) == 1:
        return next(iter(rewards.values()))
    return None


def _verifier_image(task: HarborTask) -> str | None:
    """The prebuilt image a separate verifier runs in: ``[verifier.environment].docker_image``, else the agent's."""
    if task.config.is_shared_verifier:
        return None
    environment = task.config.verifier.environment
    if environment is not None and environment.docker_image:
        return environment.docker_image
    if environment is None:
        return task.image
    return None


def _verifier_key(episode_key: str) -> str:
    """The ownership key of an episode's separate-verifier sandbox."""
    return f"{episode_key}/verifier"


def _is_main_service(name: str | None) -> bool:
    """``None`` or ``"main"`` name the agent's own container; anything else is a Compose sidecar."""
    return name in (None, "main")


def _collected_artifacts(task: HarborTask) -> list[HarborArtifact]:
    """``/logs/artifacts`` first, then the task's own ``artifacts`` entries."""
    entries = list(task.config.artifacts)
    if not any(a.source.rstrip("/") == ARTIFACTS_DIR and a.service in (None, "main") for a in entries):
        entries.insert(0, HarborArtifact(source=ARTIFACTS_DIR))
    return entries


def _stdout_tail(logs_dir: Path, limit: int = 600) -> str:
    path = logs_dir / Path(VERIFIER_STDOUT).name
    if not path.is_file():
        return "(no test-stdout.txt)"
    return path.read_text(errors="replace")[-limit:]


def _verifier_gets_overlay(task: HarborTask) -> bool:
    """A separate verifier that runs in the task's own image needs the same Dockerfile overlay as the agent."""
    return bool(task.overlay) and not task.config.is_shared_verifier and task.config.verifier.environment is None


class HarborVerificationMixin:
    """Grade an episode: the test.sh wrapper protocol, the reward files and the verifier sandbox.

    The verifier runs Harbor's tests/test.sh under a timeout wrapper that writes an exit record,
    reads reward.json or reward.txt, applies Harbor's reward selection rule and turns the exit into
    a measured or masked sample. Module level are the in-sandbox paths the protocol uses, the
    failure kinds, the wrapper command, the reward and exit record parsers, the timeout rule, and
    the choice of image and user the verifier runs as in shared and separate mode. The mixin holds
    the shared and separate verification choreography unchanged, the reward and log collection,
    the collect hooks and artifact download, and the scoring methods. Creating the verifier
    sandbox stays with the other provisioning code in sandboxes.py.

    """

    async def _run_verifier(self, session: HarborSession) -> dict[str, Any]:
        """Run ``tests/test.sh`` and read the reward it wrote, in the agent's sandbox or a separate one."""
        if session.task.config.is_shared_verifier:
            return await self._run_shared_verifier(session) | {"verifier_mode": "shared"}
        return await self._run_separate_verifier(session) | {"verifier_mode": "separate"}

    def _reserve_logs_dir(self) -> Path:
        """A fresh host directory for one verification's logs.

        Episode IDs are opaque caller input, never host paths. Reserve a fresh directory so
        neither traversal nor stale output from a reused ID can supply verifier files.
        """
        if self._artifacts_dir.resolve() != self._artifacts_dir:
            raise RuntimeError("Verifier artifact root was redirected after server startup")
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = self._artifacts_dir / f"session-{uuid4().hex}"
        # Exclusive mkdir keeps the directory fresh while honoring the process umask and parent ACLs.
        logs_dir.mkdir()
        logs_dir = logs_dir.resolve()
        if not logs_dir.is_relative_to(self._artifacts_dir):
            raise RuntimeError("Verifier log directory escapes the configured artifact root")
        return logs_dir

    async def _collect_reward(
        self, sandbox: AsyncSandbox, task: HarborTask, logs_dir: Path
    ) -> tuple[tuple[int, int | None] | None, dict[str, float] | None, str | None]:
        """Download the reward files and the wrapper's exit record, then whatever else the verifier wrote.

        The reward and the exit record are the required transfer, whatever the exit code: a grader may
        have written its reward before a late exit, and a timed-out run's partial logs are still worth
        keeping. Optional diagnostics cannot invalidate a measured score. Returns the exit record (None
        when the wrapper never reached its end), the rewards and the reward problem.
        """
        await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir, names=("reward.json", "reward.txt", EXIT_RECORD_NAME))
        exit_record = parse_exit_record(logs_dir)
        rewards, problem = parse_reward_file(logs_dir)
        try:
            await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir, skip_unsafe=True)
        except Exception:
            LOGGER.warning("Could not collect optional verifier logs for %s", task.task_id, exc_info=True)
        return exit_record, rewards, problem

    async def _run_separate_verifier(self, session: HarborSession) -> dict[str, Any]:
        """Harbor's separate mode: collect the agent's artifacts, stop its sandbox, verify in a fresh one.

        The verifier sandbox comes from ``[verifier.environment]`` (or the task image when the
        mode is ``separate`` without one). ``/logs/artifacts`` and every ``artifacts`` entry are
        collected as the archives the agent's sandbox packs, staged through the host (the filtered
        host copy is the safety gate and a diagnostic) and restored from those archives into the same
        paths, with the modes the agent left (see :func:`upload_archive`), then ``tests/`` is uploaded
        and ``test.sh`` runs there. Collection is phased as Harbor's is: the main container's hooks and
        artifacts, its stop, then the sidecars' hooks and artifacts, then the rest of the group is stopped.
        """
        task = session.task
        settings = task.config.verifier
        started = asyncio.get_running_loop().time()
        try:
            logs_dir = self._reserve_logs_dir()
        except (OSError, RuntimeError) as exc:
            LOGGER.exception("Could not create verifier logs under %s", self._artifacts_dir)
            return self._masked(
                failure_kinds.PERSISTENCE_FAILED,
                f"Could not create verifier logs under {self._artifacts_dir}: {type(exc).__name__}: {exc}",
            )
        artifacts_dir = logs_dir / "artifacts"
        verifier: AsyncSandbox | None = None
        try:
            main_hooks = [hook for hook in settings.collect if _is_main_service(hook.service)]
            sidecar_hooks = [hook for hook in settings.collect if not _is_main_service(hook.service)]
            # Each artifact stages under its own numbered root (see ``_download_artifacts``), across both phases.
            artifacts = list(enumerate(_collected_artifacts(task)))
            main_artifacts = [(index, artifact) for index, artifact in artifacts if _is_main_service(artifact.service)]
            sidecar_artifacts = [(i, artifact) for i, artifact in artifacts if not _is_main_service(artifact.service)]
            await self._run_collect_hooks(session, main_hooks)
            restored = await self._download_artifacts(session, main_artifacts, artifacts_dir)
            # Harbor's phased collection: the agent's own container is stopped before the sidecars' hooks
            # run and their artifacts are read, so no agent process can change a sidecar's state meanwhile.
            await session.stop_main()
            await self._run_collect_hooks(session, sidecar_hooks)
            restored += await self._download_artifacts(session, sidecar_artifacts, artifacts_dir)
            # Harbor stops the agent's containers before the verifier starts; nothing may keep running.
            # The episode key stays reserved until the episode's final cleanup, which releases it without
            # stopping the agent side again.
            await session.stop_agent_side()

            verifier = await self._create_verifier_sandbox(task, episode_key=session.session_id)
            if _verifier_gets_overlay(task):
                # The verifier runs in the task's own image, so it needs the same overlay as the agent did.
                await self._apply_overlay(verifier, task)
            prepare = await exec_as_root(
                verifier,
                f"mkdir -p {TESTS_DIR} {VERIFIER_LOGS_DIR} {ARTIFACTS_DIR} && chmod 777 {TESTS_DIR} {VERIFIER_LOGS_DIR}",
                timeout_s=60,
            )
            if prepare.return_code != 0:
                raise RuntimeError(f"Could not prepare verifier directories: {prepare.stderr or prepare.stdout}")
            # A prebuilt verifier image usually ships /tests already, with the modes its tests expect.
            # Only upload the package's tests when the image has none, as the old server did.
            if await remote_kind(verifier, f"{TESTS_DIR}/test.sh") is None:
                await upload_dir(verifier, task.path / "tests", TESTS_DIR)
            for archive, kind, source in restored:
                await upload_archive(verifier, archive, kind, source)
                # The staged tree stays as the diagnostic; the archive served its purpose.
                archive.unlink(missing_ok=True)
            # A sub-second task budget still gets one whole second: `timeout 0` would mean no limit.
            budget = max(1, math.ceil(settings.timeout_sec))
            await _make_test_script_executable(verifier)
            result = await verifier.exec(
                verifier_command(budget),
                env=resolve_env_templates(settings.env),
                timeout_s=settings.timeout_sec + self.config.verifier_grace_s,
                user=settings.user,
            )
            if result.error_type not in (None, "timeout"):
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            exit_record, rewards, problem = await self._collect_reward(verifier, task, logs_dir)
            if exit_record is None and result.error_type != "timeout":
                # The wrapper never reached its end: the guard found no `timeout` or `bash` (exit 127
                # before any redirection), or the shell itself could not run. test.sh never started.
                return self._masked(
                    failure_kinds.VERIFIER_ERROR,
                    f"test.sh could not be started: timeout wrapper or bash missing (exit {result.return_code})",
                )
            timed_out = verifier_timed_out(result, exit_record, budget)
        except HTTPException:
            raise
        except HarborOverlayError as exc:
            # The rollout is complete; losing the verifier's sandbox to its overlay must not discard it.
            LOGGER.error(f"Dockerfile overlay failed in the verifier sandbox for {task.task_id}: {exc}")
            return self._masked(OVERLAY_FAILED_KIND, f"Dockerfile overlay failed in the verifier sandbox: {exc}")
        except Exception as exc:
            LOGGER.exception(f"Separate verification infrastructure failed for {task.task_id}")
            return self._masked(failure_kinds.PROVIDER_UNAVAILABLE, f"{type(exc).__name__}: {exc}")
        finally:
            # Owned from before start(): a create that failed or timed out is released here too.
            if _verifier_key(session.session_id) in self._sandboxes:
                try:
                    await self._stop_sandbox(_verifier_key(session.session_id))
                except Exception:
                    # Still owned: shutdown retries the stop.
                    LOGGER.exception(f"Could not stop the verifier sandbox for {task.task_id}")
        return self._score(
            task, settings, result, logs_dir, started, timed_out=timed_out, rewards=rewards, problem=problem
        )

    async def _run_collect_hooks(self, session: HarborSession, hooks: list[Any]) -> None:
        """Run ``[[verifier.collect]]`` hooks in their services, before anything is collected from them.

        Harbor runs them in both verifier modes. A hook that exits non-zero, or runs past its
        ``timeout_sec``, is the task's business and only warns, as Harbor's best-effort hooks do; a hook
        the sandbox could not run at all is an outage and raises.
        """
        task = session.task
        for hook in hooks:
            try:
                sandbox = session.service(hook.service)
            except KeyError as exc:
                LOGGER.warning(f"{task.task_id}: collect hook skipped: {exc}")
                continue
            main = _is_main_service(hook.service)
            result = await sandbox.exec(
                hook.command if main else f"sh -c {shlex.quote(hook.command)}",
                cwd=session.workdir if main else None,
                timeout_s=hook.timeout_sec + 30,
                user=hook.user,
            )
            if result.error_type == "timeout":
                LOGGER.warning(f"{task.task_id}: collect hook timed out after {hook.timeout_sec:g}s: {hook.command!r}")
                continue
            if result.error_type:
                raise SandboxProbeError(f"collect hook could not run: {result.error_type}: {hook.command!r}")
            if result.return_code:
                LOGGER.warning(f"{task.task_id}: collect hook exited {result.return_code}: {hook.command!r}")

    async def _download_artifacts(
        self, session: HarborSession, artifacts: list[tuple[int, HarborArtifact]], artifacts_dir: Path
    ) -> list[tuple[Path, str, str]]:
        """Pull the numbered ``artifacts`` out of their services; returns ``(archive, kind, sandbox path)`` for the
        ones found.

        Each artifact stages under its own numbered root, and the archive its sandbox packed is kept beside
        that root (``NN.tar.gz``) for the restore: the staged copy went through the host's filter and carries
        widened modes, the archive carries the modes the agent left. Links are kept (below), so one
        artifact's tree may hold a link to anywhere; with a shared root a later artifact's path could run
        through it.
        """
        task = session.task
        restored: list[tuple[Path, str, str]] = []
        for index, artifact in artifacts:
            try:
                sandbox = session.service(artifact.service)
            except KeyError as exc:
                LOGGER.warning(f"{task.task_id}: artifact {artifact.source!r} skipped: {exc}")
                continue
            host = artifacts_dir / f"{index:02d}" / artifact.host_path
            archive = artifacts_dir / f"{index:02d}.tar.gz"
            # Best effort per artifact, as Harbor collects them: what the agent left behind (an
            # unreadable tree, a link that escapes the folder, a device node, a file where a folder was
            # expected, a path too long for this host) cannot mask the sample. Symbolic links travel as
            # links, because the restore target is another sandbox, not the host. Anything else (a lost
            # sandbox, a full disk, a dropped connection) is Gym's outage and still masks.
            try:
                kind = await download_path(
                    sandbox,
                    artifact.source,
                    host,
                    links="keep",
                    exclude=tuple(artifact.exclude),
                    archive_copy=archive,
                )
            except (
                SandboxTransferError,
                tarfile.TarError,
                FileExistsError,
                IsADirectoryError,
                NotADirectoryError,
            ) as exc:
                LOGGER.warning(f"{task.task_id}: artifact {artifact.source!r} not collected: {exc}")
                continue
            except OSError as exc:
                # Only the name-length error is the agent's tree (a path this host's filesystem cannot
                # hold); ENOSPC, EIO and the rest are this host's failure and must still mask, so no
                # blanket OSError clause.
                if exc.errno != errno.ENAMETOOLONG:
                    raise
                LOGGER.warning(f"{task.task_id}: artifact {artifact.source!r} not collected: {exc}")
                continue
            if kind is not None and host.is_symlink():
                # The artifact itself is a link: nothing to restore without following it on this host.
                LOGGER.warning(f"{task.task_id}: artifact {artifact.source!r} is a symbolic link, not restored")
                archive.unlink(missing_ok=True)
                continue
            if kind is not None:
                restored.append((archive, kind, artifact.source))
        return restored

    async def _run_shared_verifier(self, session: HarborSession) -> dict[str, Any]:
        """Run ``tests/test.sh`` in the agent's sandbox and read the reward it wrote.

        ``[[verifier.collect]]`` hooks run first, as in Harbor: what they produce under ``/logs/artifacts``
        is there for ``test.sh``, and for the run's logs.
        """
        task = session.task
        sandbox = session.sandbox
        settings = task.config.verifier
        started = asyncio.get_running_loop().time()
        try:
            logs_dir = self._reserve_logs_dir()
        except (OSError, RuntimeError) as exc:
            LOGGER.exception("Could not create verifier logs under %s", self._artifacts_dir)
            return self._masked(
                failure_kinds.PERSISTENCE_FAILED,
                f"Could not create verifier logs under {self._artifacts_dir}: {type(exc).__name__}: {exc}",
            )
        try:
            await self._run_collect_hooks(session, settings.collect)
            # Replace agent-planted directory links before cleanup: find does not follow its starting
            # symlink, while tar does. Unlink /logs first without touching its former target.
            roots = (str(Path(VERIFIER_LOGS_DIR).parent), TESTS_DIR, VERIFIER_LOGS_DIR, AGENT_LOGS_DIR)
            unlink_roots = [f"if test -L {shlex.quote(path)}; then rm -- {shlex.quote(path)}; fi" for path in roots]
            prepare = await exec_as_root(
                sandbox,
                " && ".join(
                    unlink_roots
                    + [
                        f"mkdir -p {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR}",
                        f"find {TESTS_DIR} {VERIFIER_LOGS_DIR} -mindepth 1 -delete",
                        f"chmod 777 {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR}",
                    ]
                ),
                timeout_s=60,
            )
            if prepare.return_code != 0:
                raise RuntimeError(f"Could not prepare verifier directories: {prepare.stderr or prepare.stdout}")
            await upload_dir(sandbox, task.path / "tests", TESTS_DIR)
            await _make_test_script_executable(sandbox)
            # One plain exec for the whole run. The provider keeps it alive for as long as
            # `timeout_s` says (OpenSandbox polls a background command, Docker streams), and
            # `timeout` inside the container is the cap that stops the tests themselves.
            # Commands must not be left running after the exec returns: OpenSandbox reaps
            # them when the command completes, so a background launch never finishes.
            # A sub-second task budget still gets one whole second: `timeout 0` would mean no limit.
            budget = max(1, math.ceil(settings.timeout_sec))
            result = await sandbox.exec(
                verifier_command(budget),
                cwd=session.workdir,
                env=resolve_env_templates(settings.env),
                timeout_s=settings.timeout_sec + self.config.verifier_grace_s,
                user=settings.user,
            )
            if result.error_type not in (None, "timeout"):
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            exit_record, rewards, problem = await self._collect_reward(sandbox, task, logs_dir)
            if exit_record is None and result.error_type != "timeout":
                # The wrapper never reached its end: the guard found no `timeout` or `bash` (exit 127
                # before any redirection), or the shell itself could not run. test.sh never started.
                return self._masked(
                    failure_kinds.VERIFIER_ERROR,
                    f"test.sh could not be started: timeout wrapper or bash missing (exit {result.return_code})",
                )
            timed_out = verifier_timed_out(result, exit_record, budget)
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception(f"Verification infrastructure failed for {task.task_id}")
            return self._masked(failure_kinds.PROVIDER_UNAVAILABLE, f"{type(exc).__name__}: {exc}")
        return self._score(
            task, settings, result, logs_dir, started, timed_out=timed_out, rewards=rewards, problem=problem
        )

    def _score(
        self,
        task: HarborTask,
        settings: Any,
        result: SandboxExecResult,
        logs_dir: Path,
        started: float,
        *,
        timed_out: bool,
        rewards: dict[str, float] | None,
        problem: str | None,
    ) -> dict[str, Any]:
        """Turn the verifier's exit and the reward it wrote into the reward fields."""
        # Absolute, so the row names a folder that exists from wherever the run was launched; the server
        # process runs from its own folder, where a relative artifacts_dir would otherwise resolve.
        extras = {
            "verifier_return_code": result.return_code,
            "verifier_logs_dir": str(Path(logs_dir).resolve()),
            "verifier_seconds": round(asyncio.get_running_loop().time() - started, 1),
        }
        if rewards is None and timed_out:
            # A reward written before the kill still counts; only a run that left none is the timeout.
            return (
                self._measured(
                    0.0,
                    VERIFIER_TIMEOUT_KIND,
                    f"test.sh exceeded [verifier].timeout_sec={settings.timeout_sec}; tail: {_stdout_tail(logs_dir)}",
                )
                | extras
            )
        if rewards is None:
            kind = MISSING_REWARD_KIND if "written" in (problem or "") else INVALID_REWARD_KIND
            return self._measured(0.0, kind, f"{problem}; tail: {_stdout_tail(logs_dir)}") | extras
        reward = select_reward(rewards)
        if reward is None:
            raise HTTPException(
                422,
                f"reward.json for {task.task_id!r} has several keys and none named 'reward': {sorted(rewards)}",
            )
        return {"reward": reward, "verifier_rewards": rewards} | extras

    @staticmethod
    def _measured(reward: float, kind: str, reason: str) -> dict[str, Any]:
        return {"reward": reward, "mask_sample": False, "failure_kind": kind, "failure_reason": reason[:2000]}

    @staticmethod
    def _masked(kind: str, reason: str) -> dict[str, Any]:
        return {"reward": 0.0, "mask_sample": True, "failure_kind": kind, "failure_reason": reason[:2000]}
