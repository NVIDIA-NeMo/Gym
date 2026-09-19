# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore task artifacts and run the official baked-in verifier."""

import asyncio
import json
import math
import shlex
from pathlib import Path, PurePosixPath

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


async def run_verifier(environment, directory, diagnostics):
    settings = environment.task.config.verifier
    logs = Path(directory) / "verifier"
    logs.mkdir(parents=True, exist_ok=True)

    async def execute():
        await environment.exec("chmod +x /tests/test.sh", user="root")
        result = await environment.exec(
            "/tests/test.sh > /logs/verifier/test-stdout.txt 2>&1",
            env=resolve_env(settings.env),
            user=settings.user,
        )
        diagnostics.append({"operation": "verifier_command", "return_code": result.return_code})
        await download_dir(environment.main, "/logs/verifier", logs)
        return {"rewards": parse_reward(logs)}

    try:
        return await asyncio.wait_for(execute(), timeout=settings.timeout_sec)
    except TimeoutError as exc:
        # Preserve diagnostics on timeout without turning partial reward files
        # into a completed evaluation (the reference aborts this verifier).
        try:
            await download_dir(environment.main, "/logs/verifier", logs)
        except Exception as download_exc:
            diagnostics.append({"operation": "verifier_logs", "error": str(download_exc)})
        raise VerifierTimeoutError(f"Verifier execution timed out after {settings.timeout_sec} seconds") from exc
