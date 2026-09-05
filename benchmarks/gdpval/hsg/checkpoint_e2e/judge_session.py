#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run bounded GDPVal resume passes against one already-live Gym service set.

``gym eval run`` owns both the servers and one rollout-collection pass, so a
missing-row retry historically paid the complete Ray/component startup cost
again.  This helper deliberately owns neither lifecycle.  ``judge.sbatch``
starts ``gym env start`` once, this process imports its resolved config from the
job-local head server, and each invocation runs only the configured GDPVal
rollout-collection driver.

The shell wrapper remains the scientific authority: it applies the exact
campaign result gate after every pass and fails closed after its bounded
concurrency ladder.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence


EXPECTED_DRIVER = "resources_servers.gdpval.multistage_orchestrator:run_rollout_collection"
CONFIG_ENV = "NEMO_GYM_CONFIG_DICT"


class JudgeSessionError(RuntimeError):
    """Raised when the persistent judge-session contract cannot be proved."""


def _url_json(url: str, *, timeout: float = 10.0) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def fetch_config_yaml(head_url: str) -> str:
    """Fetch the exact resolved config exported by the live Gym head server."""
    value = _url_json(f"{head_url.rstrip('/')}/global_config_dict_yaml")
    if not isinstance(value, str) or not value.strip():
        raise JudgeSessionError("Gym head server returned no resolved config")
    return value


def services_ready(head_url: str, expected_services: Sequence[str]) -> tuple[bool, str]:
    """Return whether the head server advertises exactly the expected live services."""
    instances = _url_json(f"{head_url.rstrip('/')}/server_instances")
    if not isinstance(instances, list):
        return False, "server_instances is not a list"

    by_path: dict[str, dict[str, Any]] = {}
    for row in instances:
        if not isinstance(row, dict) or not isinstance(row.get("config_path"), str):
            return False, "server_instances contains a malformed row"
        config_path = row["config_path"]
        if config_path in by_path:
            return False, f"duplicate service {config_path}"
        by_path[config_path] = row

    expected = set(expected_services)
    actual = set(by_path)
    if actual != expected:
        return False, f"service set mismatch missing={sorted(expected - actual)} extra={sorted(actual - expected)}"

    for config_path in sorted(expected):
        url = by_path[config_path].get("url")
        if not isinstance(url, str) or not url.startswith("http://"):
            return False, f"service {config_path} has no local HTTP URL"
        try:
            # Gym considers any HTTP response proof that the server is live.
            urllib.request.urlopen(url, timeout=5).close()
        except urllib.error.HTTPError:
            pass
        except (OSError, urllib.error.URLError) as exc:
            return False, f"service {config_path} is not ready: {exc}"
    return True, f"all {len(expected)} services ready"


def wait_for_services(
    head_url: str,
    expected_services: Sequence[str],
    *,
    timeout_seconds: int,
    poll_seconds: float,
    owner_pid: int | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_status = "head server not yet reachable"
    while time.monotonic() < deadline:
        if owner_pid is not None:
            try:
                os.kill(owner_pid, 0)
            except ProcessLookupError as exc:
                raise JudgeSessionError(f"Gym service owner exited before readiness: pid={owner_pid}") from exc
            proc_stat = Path(f"/proc/{owner_pid}/stat")
            if proc_stat.is_file() and proc_stat.read_text().split()[2] == "Z":
                raise JudgeSessionError(f"Gym service owner became a zombie before readiness: pid={owner_pid}")
        try:
            ready, last_status = services_ready(head_url, expected_services)
            if ready:
                print(last_status, flush=True)
                return
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_status = f"head server not ready: {exc}"
        time.sleep(poll_seconds)
    raise JudgeSessionError(f"judge services did not become ready within {timeout_seconds}s: {last_status}")


def _run_pass(head_url: str, concurrency: int) -> None:
    """Run one resume pass without starting or stopping any Gym services."""
    config_yaml = fetch_config_yaml(head_url)
    # Set this before importing Gym. Its global-config singleton is initialized
    # on first use, and must see the exact server-resolved config including the
    # live Ray address and job-local ports.
    os.environ[CONFIG_ENV] = config_yaml

    from omegaconf import OmegaConf, open_dict

    from nemo_gym.rollout_collection import E2ERolloutCollectionConfig, RolloutCollectionConfig
    from nemo_gym.train_data_utils import TrainDataProcessor

    global_config = OmegaConf.create(config_yaml)
    with open_dict(global_config):
        global_config["num_samples_in_parallel"] = concurrency
        global_config["resume_from_cache"] = True

    e2e_config = E2ERolloutCollectionConfig.model_validate(global_config)
    if e2e_config.rollout_collection_driver != EXPECTED_DRIVER:
        raise JudgeSessionError(
            f"unexpected rollout_collection_driver: {e2e_config.rollout_collection_driver!r}"
        )

    output_path = Path(e2e_config.output_jsonl_fpath)
    data_output_dir = output_path.parent / "preprocessed_datasets"
    input_path = data_output_dir / f"{e2e_config.split}.jsonl"
    journal_path = output_path.with_name(f"{output_path.stem}_multistage_state.jsonl")

    reuse_prepared = e2e_config.reuse_existing_data_preparation or (
        output_path.is_file() and journal_path.is_file()
    )
    if input_path.exists():
        if input_path.is_symlink() or not input_path.is_file():
            raise JudgeSessionError(f"prepared input is not a regular non-symlink file: {input_path}")
    if not (reuse_prepared and input_path.is_file()):
        data_config = deepcopy(global_config)
        with open_dict(data_config):
            data_config["should_download"] = True
            data_config["mode"] = "train_preparation"
            data_config["output_dirpath"] = str(data_output_dir)
        TrainDataProcessor().run(data_config)
    else:
        print(f"reusing prepared input: {input_path}", flush=True)

    if not input_path.is_file() or input_path.is_symlink():
        raise JudgeSessionError(f"data preparation did not publish a regular input: {input_path}")

    rollout_config_dict = deepcopy(global_config)
    with open_dict(rollout_config_dict):
        rollout_config_dict["input_jsonl_fpath"] = str(input_path)
    rollout_config = RolloutCollectionConfig.model_validate(OmegaConf.to_container(rollout_config_dict))

    module_name, separator, function_name = EXPECTED_DRIVER.partition(":")
    if not separator:
        raise JudgeSessionError(f"invalid pinned driver: {EXPECTED_DRIVER}")
    driver = getattr(importlib.import_module(module_name), function_name)
    resolved_config = OmegaConf.to_container(global_config, resolve=True)
    print(f"judge resume pass concurrency={concurrency} output={output_path}", flush=True)
    asyncio.run(driver(rollout_config, resolved_config))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    wait = subcommands.add_parser("wait", help="wait for the exact persistent service set")
    wait.add_argument("--head-url", required=True)
    wait.add_argument("--expected-service", action="append", required=True)
    wait.add_argument("--timeout-seconds", type=_positive_int, default=1800)
    wait.add_argument("--poll-seconds", type=float, default=3.0)
    wait.add_argument("--owner-pid", type=_positive_int)

    run_pass = subcommands.add_parser("run-pass", help="run one resume pass against live services")
    run_pass.add_argument("--head-url", required=True)
    run_pass.add_argument("--concurrency", type=_positive_int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "wait":
            if args.poll_seconds <= 0 or args.poll_seconds > 60:
                raise JudgeSessionError("poll interval must be in (0, 60]")
            wait_for_services(
                args.head_url,
                args.expected_service,
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
                owner_pid=args.owner_pid,
            )
        else:
            _run_pass(args.head_url, args.concurrency)
    except (JudgeSessionError, OSError, ValueError) as exc:
        print(f"CHECKPOINT_E2E_JUDGE_SESSION_FAIL: {exc}", file=sys.stderr, flush=True)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
