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

import json
import logging
import math
import shlex
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from environment_servers.harbor.session import AGENT_LOGS_DIR, VERIFIER_LOGS_DIR, HarborSession, resolve_env_templates
from nemo_gym import failure_kinds
from nemo_gym.sandbox.transfer import download_dir, exec_as_root, upload_dir


LOGGER = logging.getLogger(__name__)

TESTS_DIR = "/tests"

VERIFIER_STDOUT = f"{VERIFIER_LOGS_DIR}/test-stdout.txt"

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


def select_reward(rewards: dict[str, float]) -> float | None:
    """Harbor's rule: the ``reward`` key, else the only key. Several keys and none named ``reward`` is ambiguous."""
    if "reward" in rewards:
        return rewards["reward"]
    if len(rewards) == 1:
        return next(iter(rewards.values()))
    return None


def _stdout_tail(logs_dir: Path, limit: int = 600) -> str:
    path = logs_dir / Path(VERIFIER_STDOUT).name
    if not path.is_file():
        return "(no test-stdout.txt)"
    return path.read_text(errors="replace")[-limit:]


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
        """Run ``tests/test.sh`` in the agent's sandbox and read the reward it wrote."""
        task = session.task
        sandbox = session.sandbox
        settings = task.config.verifier
        try:
            # Episode IDs are opaque caller input, never host paths. Reserve a fresh directory
            # so neither traversal nor stale output from a reused ID can supply verifier files.
            if self._artifacts_dir.resolve() != self._artifacts_dir:
                raise RuntimeError("Verifier artifact root was redirected after server startup")
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            logs_dir = self._artifacts_dir / f"session-{uuid4().hex}"
            # Exclusive mkdir keeps the directory fresh while honoring the process umask and parent ACLs.
            logs_dir.mkdir()
            logs_dir = logs_dir.resolve()
            if not logs_dir.is_relative_to(self._artifacts_dir):
                raise RuntimeError("Verifier log directory escapes the configured artifact root")
        except (OSError, RuntimeError) as exc:
            LOGGER.exception("Could not create verifier logs under %s", self._artifacts_dir)
            return self._masked(
                failure_kinds.PERSISTENCE_FAILED,
                f"Could not create verifier logs under {self._artifacts_dir}: {type(exc).__name__}: {exc}",
            )
        try:
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
            result = await sandbox.exec(
                f"bash {TESTS_DIR}/test.sh > {VERIFIER_STDOUT} 2>&1",
                cwd=session.workdir,
                env=resolve_env_templates(settings.env),
                timeout_s=settings.timeout_sec,
                user=settings.user,
            )
            if result.error_type not in (None, "timeout"):
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            if result.error_type != "timeout":
                # Reward transfer is required; optional diagnostics cannot invalidate a measured score.
                await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir, names=("reward.json", "reward.txt"))
                rewards, problem = parse_reward_file(logs_dir)
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception(f"Verification infrastructure failed for {task.task_id}")
            return self._masked(failure_kinds.PROVIDER_UNAVAILABLE, f"{type(exc).__name__}: {exc}")

        # Absolute, so the row names a folder that exists from wherever the run was launched; the server
        # process runs from its own folder, where a relative artifacts_dir would otherwise resolve.
        extras = {"verifier_return_code": result.return_code, "verifier_logs_dir": str(Path(logs_dir).resolve())}
        try:
            await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir, skip_unsafe=True)
        except Exception:
            LOGGER.warning("Could not collect optional verifier logs for %s", task.task_id, exc_info=True)
        if result.error_type == "timeout":
            return (
                self._measured(
                    0.0, VERIFIER_TIMEOUT_KIND, f"test.sh exceeded [verifier].timeout_sec={settings.timeout_sec}"
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
