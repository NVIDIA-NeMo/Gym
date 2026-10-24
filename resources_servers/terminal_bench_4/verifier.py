# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore task artifacts and run the official baked-in verifier."""

import asyncio
import json
import math
import shlex
from pathlib import Path, PurePosixPath

from resources_servers.terminal_bench_4.models import VerifierTestCounts
from resources_servers.terminal_bench_4.task import resolve_env
from resources_servers.terminal_bench_4.transfers import download_dir, prepare_directory, upload_dir, upload_file


class RewardFileNotFoundError(FileNotFoundError):
    pass


class RewardFileEmptyError(ValueError):
    pass


class VerifierOutputParseError(ValueError):
    pass


class VerifierTimeoutError(TimeoutError):
    pass


def parse_reward(directory):
    directory = Path(directory)
    json_path, text_path = directory / "reward.json", directory / "reward.txt"
    path = json_path if json_path.exists() else text_path
    if not path.exists():
        raise RewardFileNotFoundError(f"No official reward file found in {directory}")
    text = path.read_text()
    if not text:
        raise RewardFileEmptyError(f"Reward file is empty: {path}")
    try:
        rewards = json.loads(text) if path == json_path else {"reward": float(text)}
        if not isinstance(rewards, dict):
            raise ValueError("Reward JSON must be an object")
        for value in rewards.values():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Official rewards must be finite numbers")
        return rewards
    except (ValueError, TypeError) as exc:
        raise VerifierOutputParseError(f"Invalid official reward in {path}: {exc}") from exc


def parse_test_counts(directory: Path, diagnostics: list[dict[str, object]]) -> VerifierTestCounts:
    """Read optional canonical CTRF or case-count metadata without affecting grading.

    Only ctrf.json is read: numbered reports may repeat the same tests, and the
    canonical file can also be a copy of the final numbered report.
    """
    for name in ("ctrf.json", "trace_results.json"):
        path = directory / name
        if not path.exists():
            continue
        try:
            report = json.loads(path.read_text())
            if name == "ctrf.json":
                summary = report["results"]["summary"]
                total = summary["tests"]
                outcomes = {key: summary.get(key, 0) for key in ("passed", "failed", "skipped", "pending", "other")}
                # Passed and failed are required; the other statuses can be omitted.
                outcomes["passed"] = summary["passed"]
                outcomes["failed"] = summary["failed"]
                if any(type(value) is not int or value < 0 for value in (total, *outcomes.values())):
                    raise ValueError("Test counts must be nonnegative integers")
                if sum(outcomes.values()) != total:
                    raise ValueError("Test outcome counts do not sum to the reported total")
                passed, failed, skipped = outcomes["passed"], outcomes["failed"], outcomes["skipped"]
            else:
                total, passed = report["total_cases"], report["passed_cases"]
                if any(type(value) is not int or value < 0 for value in (total, passed)) or passed > total:
                    raise ValueError("Case counts must be nonnegative integers with passed <= total")
                failed, skipped = total - passed, 0
            counts = VerifierTestCounts(
                num_tests_passed=passed,
                num_tests_total=total,
                num_tests_failed=failed,
                num_tests_skipped=skipped,
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            diagnostics.append({"operation": "verifier_test_counts", "source": name, "error": str(exc)})
            continue
        diagnostics.append({"operation": "verifier_test_counts", "source": name})
        return counts
    return VerifierTestCounts()


async def restore(environment, artifacts_dir):
    await prepare_directory(environment, "/logs/verifier", empty=True)
    shared_logs = getattr(environment, "shared_logs", None)
    for artifact in environment.task.config.collected_artifacts:
        host = Path(artifacts_dir) / artifact.host_path
        if not host.exists():
            continue
        target = artifact.source
        if host.is_dir():
            await prepare_directory(environment, target, empty=True)
            if shared_logs and shared_logs.restored_archive and target == "/logs/artifacts":
                archive = shlex.quote(shared_logs.restored_archive)
                result = await environment.main.exec(
                    f"tar -xzf {archive} -C /logs/artifacts; status=$?; rm -f {archive}; exit $status",
                    timeout_s=600,
                )
                if result.return_code == 0:
                    continue
                # Retain the normal transfer fallback if remote extraction is
                # unavailable, clearing any partial extraction first.
                await prepare_directory(environment, target, empty=True)
            await upload_dir(environment.main, host, target)
        else:
            parent = str(PurePosixPath(target).parent)
            if parent and parent != target:
                await prepare_directory(environment, parent)
            await upload_file(environment.main, host, target)


def _require_command_outcome(result, step):
    """A sandbox that reports no command outcome (``error_type`` set) failed the verifier, so the reward files it
    left say nothing about the task. Test doubles without ``error_type`` ran the command."""
    error_type = getattr(result, "error_type", None)
    if error_type:
        raise RuntimeError(f"Verifier {step} failed in the sandbox: {error_type}: {result.stderr}")


async def run_verifier(environment, directory, diagnostics, timeout_sec=None):
    settings = environment.task.config.verifier
    timeout_sec = settings.timeout_sec if timeout_sec is None else timeout_sec
    logs = Path(directory) / "verifier"
    logs.mkdir(parents=True, exist_ok=True)

    async def execute():
        _require_command_outcome(await environment.exec("chmod +x /tests/test.sh", user="root"), "preparation")
        result = await environment.exec(
            "/tests/test.sh > /logs/verifier/test-stdout.txt 2>&1",
            env=resolve_env(settings.env),
            user=settings.user,
        )
        diagnostics.append({"operation": "verifier_command", "return_code": result.return_code})
        _require_command_outcome(result, "execution")
        await download_dir(environment.main, "/logs/verifier", logs)
        rewards = parse_reward(logs)
        counts = parse_test_counts(logs, diagnostics)
        return {"rewards": rewards, **counts.model_dump(exclude_none=True)}

    # Only the verifier's own budget makes a VerifierTimeoutError. A provider call that times out within it is an
    # infrastructure failure and propagates as raised.
    deadline = asyncio.timeout(timeout_sec)
    try:
        async with deadline:
            return await execute()
    except TimeoutError as exc:
        if not deadline.expired():
            raise
        # Preserve diagnostics on timeout without turning partial reward files
        # into a completed evaluation (the reference aborts this verifier).
        try:
            await download_dir(environment.main, "/logs/verifier", logs)
        except Exception as download_exc:
            diagnostics.append({"operation": "verifier_logs", "error": str(download_exc)})
        raise VerifierTimeoutError(f"Verifier execution timed out after {timeout_sec} seconds") from exc
