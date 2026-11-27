# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan a benchmark: resolve it, prepare its data, build its plan and collate its rows."""

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from nemo_gym.config.config_types import ConfigError, ignoring_command_line
from nemo_gym.config.global_config import (
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    scoped_global_config_dict,
)
from nemo_gym.data.task_selection import row_task_id
from nemo_gym.environment.assets import asset_config_path
from nemo_gym.environment.benchmark_preparation import PrepareBenchmarkConfig, plan_for_config, read_jsonl_rows
from nemo_gym.environment.in_process.overrides import _checked_overrides
from nemo_gym.eval.eval_plan import EvalPlan
from nemo_gym.utils.decorators import experimental


@dataclass(frozen=True)
class PlannedBenchmark:
    """A prepared benchmark: its plan (task ids and input digests) and the collated rows to run."""

    plan: EvalPlan
    rows: list[dict[str, Any]]

    def task_ids(self) -> list[str]:
        return [task.task_id.task_id for task in self.plan.tasks]

    def task_digests(self) -> dict[str, str]:
        """Each planned task's id mapped to the digest of its input, which changes when the input changes."""
        return {task.task_id.task_id: task.digest for task in self.plan.tasks}

    @property
    def gym_commit(self) -> Optional[str]:
        """The Gym commit the plan was made with, or None when Gym was not installed from a git checkout."""
        return self.plan.gym.commit

    def to_json(self) -> str:
        """The planned benchmark as JSON text, for handing it to another process; `from_json` reads it back."""
        return json.dumps(
            {"plan": self.plan.model_dump(mode="json", by_alias=True), "rows": self.rows}, allow_nan=False
        )

    @classmethod
    def from_json(cls, text: str) -> "PlannedBenchmark":
        saved = json.loads(text)
        return cls(plan=EvalPlan.model_validate(saved["plan"]), rows=saved["rows"])


@experimental(once=True)
def plan_benchmark(
    benchmark: str,
    *,
    task_ids: Optional[Sequence[str]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    work_dir: Optional[Path] = None,
) -> PlannedBenchmark:
    """Resolve the benchmark `NAME[/FLAVOR]`, prepare its data, build its plan and collate its rows.

    `overrides` are config keys merged on top, as a nested dict like the config file's; a dotted key is refused
    (`ConfigError`), because it would be one literal key here but a path on the command line.
    `work_dir` receives the collated files; a temporary directory is used when it is None.

    The caller's command line and active exporters are left alone. The benchmark's prepare script sees
    `overrides` as its Gym configuration, which is this process's configuration only while it runs.
    """
    global_config_dict = _benchmark_config(asset_config_path("benchmark", benchmark), overrides)
    with ignoring_command_line(), scoped_global_config_dict(global_config_dict):
        plan = plan_for_config(global_config_dict, PrepareBenchmarkConfig.model_validate(global_config_dict), task_ids)
        if work_dir is None:
            with tempfile.TemporaryDirectory(prefix="gym-collate-") as directory:
                rows = _collated_rows(global_config_dict, Path(directory))
        else:
            rows = _collated_rows(global_config_dict, work_dir)
    if task_ids is not None:
        rows = [row for row in rows if row_task_id(row) in set(task_ids)]
    return PlannedBenchmark(plan=plan, rows=rows)


def _benchmark_config(config_path: str, overrides: Optional[Mapping[str, Any]]) -> DictConfig:
    """The resolved config for one benchmark, built without reading sys.argv or caching it process-wide."""
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": [config_path]},
        _checked_overrides(overrides),
    )
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(initial_global_config_dict=initial, skip_load_from_cli=True, skip_exporters=True)
    )


def _collated_rows(global_config_dict: DictConfig, output_dir: Path) -> list[dict[str, Any]]:
    from nemo_gym.data.dataset_preparation import TrainDataProcessor

    TrainDataProcessor().run(
        OmegaConf.merge(global_config_dict, {"mode": "train_preparation", "output_dirpath": str(output_dir)})
    )
    collated = output_dir / "benchmark.jsonl"
    if not collated.exists():
        raise ConfigError(f"Collation did not produce {collated}")
    return read_jsonl_rows(collated)
