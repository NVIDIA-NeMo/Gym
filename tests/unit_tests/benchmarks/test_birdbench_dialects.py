# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BIRD dev's dialects: each is a benchmark of its own, whose manifest name, config and taskset share one ID."""

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.birdbench import prepare as birdbench_prepare
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest, resolve_manifest_config_path


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "birdbench"
# The ID of each dialect's benchmark: its manifest's name, which is the manifest's folder under benchmarks/.
BENCHMARK_IDS = {"sqlite": "birdbench", "spark": "birdbench/spark", "postgres": "birdbench/postgres"}


def _config_path(dialect: str) -> Path:
    manifest_path = REPO_ROOT / "benchmarks" / BENCHMARK_IDS[dialect] / "manifest.yaml"
    return resolve_manifest_config_path(manifest_path, load_manifest(manifest_path))


def test_every_dialect_has_a_benchmark():
    assert set(BENCHMARK_IDS) == set(birdbench_prepare.DIALECTS)


@pytest.mark.parametrize("dialect", birdbench_prepare.DIALECTS)
def test_a_dialect_s_manifest_config_and_taskset_share_its_id(dialect):
    manifest = load_manifest(REPO_ROOT / "benchmarks" / BENCHMARK_IDS[dialect] / "manifest.yaml")
    benchmark = BenchmarkConfig.from_config_path(_config_path(dialect))
    assert manifest.name == benchmark.dataset.taskset == BENCHMARK_IDS[dialect]
    assert benchmark.environment_server == "birdbench_bird_sql_environment_server"
    (declared,) = manifest.datasets
    declared_fields = declared.model_dump(mode="json", exclude_none=True)
    assert benchmark.dataset.model_dump(mode="json", include=set(declared_fields)) == declared_fields


@pytest.mark.parametrize("dialect", birdbench_prepare.DIALECTS)
def test_a_dialect_s_config_prepares_and_grades_in_that_dialect(dialect):
    config = GlobalConfigDictParser().parse_no_environment(
        initial_global_config_dict=OmegaConf.merge(
            OmegaConf.load(_config_path(dialect)), GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT
        )
    )
    assert config["prepare_script_args"]["dialect"] == dialect
    # config.yaml keeps bird_sql's default dialect, sqlite.
    assert (
        config["birdbench_bird_sql_resources_server"]["resources_servers"]["bird_sql"].get("dialect", "sqlite")
        == dialect
    )
    (dataset,) = config["birdbench_bird_sql_simple_agent"]["responses_api_agents"]["simple_agent"]["datasets"]
    # `gym eval prepare` requires prepare()'s output path to equal the dataset's jsonl_fpath.
    assert REPO_ROOT / dataset["jsonl_fpath"] == birdbench_prepare.output_fpath_for(dialect, "dev")
    assert REPO_ROOT / dataset["prompt_config"] == birdbench_prepare.prompt_config_fpath_for(dialect)
    assert dataset["num_repeats"] == 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="eval.sh is a bash script")
@pytest.mark.parametrize("split", birdbench_prepare.SPLITS)
@pytest.mark.parametrize("dialect", birdbench_prepare.DIALECTS)
def test_eval_sh_prepares_and_runs_the_dialect_s_benchmark(dialect, split, tmp_path):
    calls = tmp_path / "gym_calls.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # gym records each call; java passes eval.sh's JDK check; ps lists no process, so the exit trap stops none.
    stubs = {
        "gym": f"#!/bin/sh\nexec python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' \"$@\" >> {shlex.quote(str(calls))}\n",
        "java": "#!/bin/sh\nexit 0\n",
        "ps": "#!/bin/sh\nexit 0\n",
    }
    for name, text in stubs.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key not in ("JAVA_HOME", "BIRD_PG_ADMIN_DSN")}
    env.update(PATH=f"{bin_dir}{os.pathsep}{env['PATH']}", BIRD_PG_DSN="postgresql://reader@db.invalid/bird")
    command = ["bash", str(BENCHMARK_DIR / "eval.sh"), "--model", "m", "--dialect", dialect, "--split", split]
    subprocess.run([*command, "--output_dir", str(tmp_path / "out")], env=env, check=True, capture_output=True)

    benchmark = BENCHMARK_IDS[dialect]
    prepare, run = (json.loads(line) for line in calls.read_text().splitlines())
    assert (prepare[:2], run[:2]) == (["eval", "prepare"], ["eval", "run"])
    for call in (prepare, run):
        assert call[call.index("--benchmark") + 1] == benchmark
        datasets = [arg for arg in call if ".simple_agent.datasets=" in arg]
        if split == "dev":
            # The dialect's own config declares its dataset.
            assert datasets == []
        else:
            (override,) = datasets
            assert f"jsonl_fpath:benchmarks/birdbench/data/birdbench_{dialect}_train.jsonl," in override
            assert override.endswith(f",taskset:{benchmark}}}]")
        assert len([arg for arg in call if ".bird_sql.postgres_dsn=" in arg]) == (dialect == "postgres")
