# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write Terminal-Bench-Science rows that carry the official instruction as their prompt.

Harnesses behind an Environment Server (``codex.yaml``) receive the row's
``responses_create_params``, not the resources seed, so each row embeds the task's
``instruction.md`` as one user message, read from the pinned package exactly as the
resources server reads it. Loading packages verifies their content hashes with ``pathspec``,
which Gym's own environment leaves out (it is MPL-2.0). Without it, ``prepare`` runs this
module in the terminal_bench_4 server's environment, set up as ``gym env start`` sets it up,
so ``gym eval prepare --benchmark terminal_bench_science/codex`` works from Gym's environment.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path

from omegaconf import OmegaConf

from benchmarks.terminal_bench_science.prepare import BENCHMARK_DIR, load_manifest, select_tasks, task_row
from nemo_gym import PARENT_DIR
from nemo_gym.config import global_config


OUTPUT_PATH = BENCHMARK_DIR / "data" / "benchmark_codex.jsonl"
# Every v0.1 task gives its agent 8 hours; codex.yaml sets the harness timeout to match.
AGENT_TIMEOUT_SEC = 28800
MODULE = "benchmarks.terminal_bench_science.prepare_codex"
SERVER_DIR = PARENT_DIR / "resources_servers" / "terminal_bench_4"
# Set for the run in the server's environment, which must not hand off again.
IN_SERVER_ENVIRONMENT_ENV_VAR = "NEMO_GYM_TBSCI_PREPARE_IN_SERVER_ENVIRONMENT"


def _needs_server_environment() -> bool:
    return importlib.util.find_spec("pathspec") is None and not os.environ.get(IN_SERVER_ENVIRONMENT_ENV_VAR)


def _prepare_in_server_environment(arguments: list[str]) -> Path:
    """Run this module with the terminal_bench_4 server's Python, setting up its environment first."""
    from nemo_gym.cli.setup_command import get_venv_path, run_command, setup_env_command
    from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    setup_config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    python = get_venv_path(SERVER_DIR, setup_config) / "bin" / "python"
    prefix = "resources_servers/terminal_bench_4".replace("/", "\\/")
    command = shlex.join([str(python), "-m", MODULE, *arguments])
    process = run_command(
        f"{setup_env_command(SERVER_DIR, setup_config, prefix)} && {command}",
        SERVER_DIR,
        server_name="terminal_bench_4",
        project_root=PARENT_DIR,
        global_config_dict=setup_config,
        stdout_target=sys.stderr,
        stderr_target=sys.stderr,
        extra_env={IN_SERVER_ENVIRONMENT_ENV_VAR: "1"},
    )
    return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"Preparing the rows in the terminal_bench_4 server's environment failed (exit {return_code})"
        )
    return OUTPUT_PATH


async def _load_tasks(names_and_refs: list[tuple[str, str]], download_dir: Path | None) -> list:
    try:
        from resources_servers.terminal_bench_4.task import PackageLoader
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Loading pinned task packages needs the terminal_bench_4 server's requirements "
            "(pathspec), which this environment lacks"
        ) from error
    loader = PackageLoader(download_dir)
    return await asyncio.gather(*(loader.load(name, ref) for name, ref in names_and_refs))


def prepare(
    *,
    task_names: list[str] | None = None,
    category: str | None = None,
    domain: str | None = None,
    task_download_dir: str | None = None,
) -> Path:
    """Write one prompt-bearing row per selected task; filters combine as in ``prepare.py``."""
    manifest = load_manifest()
    tasks = select_tasks(manifest, task_names=task_names, category=category, domain=domain)
    if _needs_server_environment():
        arguments = ["--task-names", *(task["name"] for task in tasks)]
        if task_download_dir:
            arguments += ["--task-download-dir", str(Path(task_download_dir).absolute())]
        return _prepare_in_server_environment(arguments)
    rows = [task_row(manifest, task) for task in tasks]
    packages = asyncio.run(
        _load_tasks(
            [(row["task_name"], row["task_ref"]) for row in rows],
            Path(task_download_dir) if task_download_dir else None,
        )
    )
    for row, package in zip(rows, packages):
        if package.config.agent.timeout_sec != AGENT_TIMEOUT_SEC:
            raise ValueError(
                f"{row['task_name']} has a {package.config.agent.timeout_sec}s agent budget; "
                f"codex.yaml assumes {AGENT_TIMEOUT_SEC}s"
            )
        row["responses_create_params"] = {"input": [{"role": "user", "content": package.instruction}]}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")
    return OUTPUT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-names", nargs="+")
    parser.add_argument("--category")
    parser.add_argument("--domain")
    parser.add_argument("--task-download-dir")
    args = parser.parse_args()
    # Package downloads use Gym's shared HTTP client, which would otherwise parse this
    # script's arguments as a Gym run config.
    if global_config._GLOBAL_CONFIG_DICT is None:
        global_config._GLOBAL_CONFIG_DICT = OmegaConf.create({})
    print(
        prepare(
            task_names=args.task_names,
            category=args.category,
            domain=args.domain,
            task_download_dir=args.task_download_dir,
        )
    )


if __name__ == "__main__":
    main()
