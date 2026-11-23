# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a Harbor-backed benchmark for the in-process API: the rows and the configuration that serves them.

A benchmark whose config sets `harbor_target` has no `datasets:` of its own. `gym eval run --benchmark NAME` resolves
the target into one row per task, writes a run config that binds those rows to the `harbor` environment server, and
starts the servers from that config. `prepare_harbor_run` does the same preparation for Python callers
(`plan_benchmark` and `serve_benchmark` in `nemo_gym.environment.in_process`) and records what it produced in a
`HarborRun`.
"""

import glob
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from nemo_gym.config.config_types import ConfigError
from nemo_gym.data.tasks.harbor.build_plan import BuildError
from nemo_gym.data.tasks.harbor.errors import HarborTaskError
from nemo_gym.data.tasks.harbor.hub import HubError
from nemo_gym.data.tasks.harbor.manifest import ManifestError
from nemo_gym.data.tasks.harbor.prepare import (
    HarborBenchmark,
    PreparedTaskset,
    _listed,
    harbor_benchmark,
    prepare_target,
    write_run_configs,
)
from nemo_gym.data.tasks.harbor.registry import ImageConfigError
from nemo_gym.data.tasks.harbor.settings import pairing_override_from_overrides, preparation_settings
from nemo_gym.environment.assets import asset_config_path
from nemo_gym.environment.benchmark_preparation import read_jsonl_rows


@dataclass(frozen=True)
class HarborRun:
    """What preparing a Harbor-backed benchmark produced, beyond the rows.

    `dataset_digest` and `dataset_toml_digest` identify the dataset the task folders came from and the Gym settings
    applied to it; `images` lists every image a run starts (task, verifier and Compose service images, as recorded).
    A plan's task digests already change with a task folder's content; these also change when the dataset or its
    settings move, or an image reference does.

    `config_paths` are the files, in merge order, that start the servers. They live in the `work_dir` of the process
    that prepared the run, so they describe that process's files only: another process prepares again.
    """

    target: str
    agent: str
    sandbox: str | None
    dataset_digest: str | None
    dataset_toml_digest: str | None
    images: tuple[str, ...]
    config_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "agent": self.agent,
            "sandbox": self.sandbox,
            "dataset_digest": self.dataset_digest,
            "dataset_toml_digest": self.dataset_toml_digest,
            "images": list(self.images),
            "config_paths": list(self.config_paths),
        }

    @classmethod
    def from_dict(cls, saved: Mapping[str, Any]) -> "HarborRun":
        return cls(
            target=saved["target"],
            agent=saved["agent"],
            sandbox=saved["sandbox"],
            dataset_digest=saved["dataset_digest"],
            dataset_toml_digest=saved["dataset_toml_digest"],
            images=tuple(saved["images"]),
            config_paths=tuple(saved["config_paths"]),
        )


@dataclass(frozen=True)
class PreparedHarborRun:
    run: HarborRun
    rows: list[dict[str, Any]]
    taskset: str


def prepare_harbor_run(
    benchmark: HarborBenchmark,
    *,
    task_ids: Sequence[str] | None,
    overrides: Mapping[str, Any],
    sandbox: str | None,
    work_dir: Path,
) -> PreparedHarborRun:
    """Resolve the benchmark's Harbor target into rows and write the run config under `work_dir`.

    `task_ids` selects tasks by exact id (None: every task). `overrides` and `sandbox` play the roles of the
    command-line overrides and `--sandbox NAME` of `gym eval run`: the `builds:`, sandbox and Compose overlay
    settings preparation reads come from the benchmark's config with `overrides` merged on top. A task that cannot
    load, or a selected id that is not a task of the target, is a `ConfigError`.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    overrides_path = work_dir / "overrides.yaml"
    overrides_path.write_text(yaml.safe_dump(dict(overrides)))
    entries = [json.dumps(str(path), ensure_ascii=False) for path in (benchmark.config_path, overrides_path.resolve())]
    tokens = [f"+config_paths=[{','.join(entries)}]"]
    agent = benchmark.agent
    only = None if task_ids is None else [glob.escape(task_id) for task_id in task_ids]
    try:
        prepared = prepare_target(
            benchmark.target,
            only=only,
            output_root=work_dir,
            rows_name=f"tasks_{agent.instance_name}.jsonl",
            agent=agent.impl_name,
            allow_unsupported_pairing=pairing_override_from_overrides(tokens),
            **preparation_settings(tokens, sandbox),
        )
        _require_selection(prepared, task_ids)
        _, config_paths = write_run_configs(prepared, agent, sandbox=sandbox)
    except (HarborTaskError, HubError, BuildError, ImageConfigError, ManifestError, ValueError) as error:
        raise ConfigError(f"Harbor target {benchmark.target!r}: {error}") from error
    run = HarborRun(
        target=benchmark.target,
        agent=agent.instance_name,
        sandbox=sandbox,
        dataset_digest=prepared.dataset_digest,
        dataset_toml_digest=prepared.dataset_toml_digest,
        images=prepared.images,
        config_paths=tuple(str(path) for path in config_paths),
    )
    return PreparedHarborRun(run=run, rows=read_jsonl_rows(prepared.rows_path), taskset=prepared.taskset)


def _require_selection(prepared: PreparedTaskset, task_ids: Sequence[str] | None) -> None:
    if prepared.skipped:
        raise ValueError(
            f"{len(prepared.skipped)} task(s) could not load (see {prepared.output_dir / 'skipped.json'}):"
            f"{_listed(prepared.skipped)}"
        )
    if task_ids is not None:
        unknown = sorted(set(task_ids) - {task.task_id for task in prepared.tasks})
        if unknown:
            raise ValueError(f"no tasks with ids {unknown}")


def require_run_files(run: HarborRun) -> None:
    """Refuse a `HarborRun` whose configuration files are not on this machine (it was prepared elsewhere)."""
    missing = [path for path in run.config_paths if not Path(path).is_file()]
    if missing:
        raise ConfigError(
            f"The configuration files of Harbor target {run.target!r} are missing: {missing}. "
            "They belong to the process that planned the benchmark; plan it again here before serving."
        )


def resolve_harbor_benchmark(benchmark: str) -> Optional[HarborBenchmark]:
    """The Harbor target the benchmark's config sets, or None for a benchmark that composes its own servers."""
    try:
        return harbor_benchmark(asset_config_path("benchmark", benchmark))
    except ValueError as error:
        raise ConfigError(str(error)) from error
