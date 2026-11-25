# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``benchmarks/birdbench/config.yaml`` and its manifest."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError

from benchmarks.birdbench import prepare as birdbench_prepare
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.environment import benchmark_preparation
from nemo_gym.environment.benchmark_preparation import PrepareBenchmarkConfig, datasets_to_prepare, prepare_datasets
from nemo_gym.server_runtime.supervisor import _prepare_worker


BENCHMARK_DIR = Path(__file__).parent.parent
REPO_ROOT = BENCHMARK_DIR.parent.parent


def _dataset():
    """config.yaml on its own, with its interpolations resolved by OmegaConf (the defaults)."""
    cfg = OmegaConf.to_container(OmegaConf.load(BENCHMARK_DIR / "config.yaml"), resolve=True)
    (dataset,) = cfg["birdbench_bird_sql_simple_agent"]["responses_api_agents"]["simple_agent"]["datasets"]
    return cfg, dataset


def _gym_config(**overrides):
    """config.yaml composed by Gym's own parser, as `gym eval prepare --benchmark birdbench` composes it (no model, no
    command line or env.yaml, offline); ``overrides`` stand for the command line's `+<key>=<value>`."""
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": [str(BENCHMARK_DIR / "config.yaml")]},
        overrides,
    )
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )


def _gym_dataset(**prepare_script_args):
    """The dataset `gym eval prepare` prepares and looks up in the cache, given `+prepare_script_args.<name>=<value>`."""
    (benchmark,) = datasets_to_prepare(_gym_config(prepare_script_args=prepare_script_args))
    return benchmark.dataset


def test_dataset_is_never_repeated():
    # `gym eval run` multiplies the dataset-level num_repeats by --num-repeats, so any value above 1 turns the
    # 1534-row prepared files into 1534 x N rows.
    assert _dataset()[1]["num_repeats"] == 1
    assert yaml.safe_load((BENCHMARK_DIR / "manifest.yaml").read_text())["datasets"][0]["num_repeats"] == 1


def test_default_dataset_is_the_sqlite_one_and_prepare_writes_exactly_that_path():
    # `gym eval prepare` raises ConfigError unless prepare()'s output path equals the dataset's jsonl_fpath.
    cfg, dataset = _dataset()
    assert dataset["jsonl_fpath"] == "benchmarks/birdbench/data/birdbench_sqlite_dev.jsonl"
    assert cfg["prepare_script_args"]["dialect"] == "sqlite"
    assert birdbench_prepare.output_fpath_for("sqlite", "dev") == REPO_ROOT / dataset["jsonl_fpath"]


def test_every_dialect_has_its_own_prompt_named_by_dialect():
    # eval.sh builds `prompts/default_<dialect>.yaml`; the default dataset uses the sqlite one.
    assert (REPO_ROOT / _dataset()[1]["prompt_config"]) == birdbench_prepare.prompt_config_fpath_for("sqlite")
    assert (
        REPO_ROOT / yaml.safe_load((BENCHMARK_DIR / "manifest.yaml").read_text())["standard_prompt_config"]
    ).is_file()
    for dialect in birdbench_prepare.DIALECTS:
        assert (
            birdbench_prepare.prompt_config_fpath_for(dialect) == BENCHMARK_DIR / "prompts" / f"default_{dialect}.yaml"
        )
        assert birdbench_prepare.prompt_config_fpath_for(dialect).is_file()


def test_default_prepare_script_args_are_dev_sqlite():
    cfg, _ = _dataset()
    assert (cfg["prepare_script_args"]["split"], cfg["prepare_script_args"]["dialect"]) == ("dev", "sqlite")
    assert (cfg["prepare_script_args"]["version"], cfg["prepare_script_args"]["prompt_format"]) == (
        "dev_20240627",
        "default",
    )


@pytest.mark.parametrize("prompt_format", birdbench_prepare.PROMPT_FORMATS)
@pytest.mark.parametrize("version", list(birdbench_prepare.VERSIONS))
def test_direct_version_and_prompt_format_select_their_file_and_prompt(version, prompt_format):
    # Without eval.sh, `+prepare_script_args.version=...` / `.prompt_format=...` alone must select the file prepare()
    # writes for them (the path `gym eval prepare` checks its output against and looks up in the cache), never the
    # default's, and the matching prompt. Checked as `gym eval prepare` reads the dataset, and as collation
    # (`gym eval run`) reads it.
    config = _gym_config(prepare_script_args={"version": version, "prompt_format": prompt_format})
    (benchmark,) = datasets_to_prepare(config)
    (collated,) = [
        dataset
        for server in GlobalConfigDictParser().filter_for_server_instance_configs(config)
        for dataset in server.datasets or []
        if dataset.name == "birdbench"
    ]
    for dataset in (benchmark.dataset, collated):
        assert REPO_ROOT / dataset.jsonl_fpath == birdbench_prepare.output_fpath_for(
            "sqlite", "dev", version, prompt_format
        )
        assert str(dataset.prompt_config) == f"benchmarks/birdbench/prompts/{prompt_format}_sqlite.yaml"
        assert (REPO_ROOT / dataset.prompt_config).is_file()


def test_the_file_name_tables_list_exactly_the_accepted_values():
    tables = OmegaConf.load(BENCHMARK_DIR / "config.yaml").birdbench_prepared_file
    assert set(tables.name_for_version) == set(birdbench_prepare.VERSIONS)
    assert set(tables.suffix_for_prompt_format) == set(birdbench_prepare.PROMPT_FORMATS)


def test_eval_sh_accepts_exactly_the_known_versions():
    # eval.sh builds the file name from --version itself, so an unlisted value could name another cached file. It is
    # read, not run: its exit trap reaps orphaned server processes.
    # The whole case block, so an extra accepting arm before the rejecting `*)` fails too.
    script = (BENCHMARK_DIR / "eval.sh").read_text()
    (block,) = re.findall(r'case "\$\{VERSION\}" in\n(.*?)\nesac', script, re.S)
    accepted, rejecting = re.findall(r"^\s*([^\s()]+)\)", block, re.M)
    assert set(accepted.split("|")) == set(birdbench_prepare.VERSIONS)
    assert rejecting == "*"


@pytest.mark.parametrize(
    "prepare_script_args",
    [
        {"version": "dev"},  # the default version's file-name part, not a version
        {"version": "dev_schema_only"},  # would name the schema-only file but keep the default prompt
        {"version": "train"},
        {"version": "dev_2000"},
        {"prompt_format": "fancy"},
        {"dialect": "spark"},  # dialect and split also change the server: eval.sh only
        {"split": "train"},
    ],
)
def test_other_direct_values_fail_before_the_cache_lookup(prepare_script_args):
    # `gym eval prepare` resolves the dataset before it looks for a cached file, so no other file is reused.
    with pytest.raises(InterpolationKeyError, match="birdbench_prepared_file"):
        _gym_dataset(**prepare_script_args)


@pytest.fixture
def gym_eval_prepare(tmp_path, monkeypatch):
    """`gym eval prepare --benchmark birdbench +use_cached_prepared_benchmarks=<cached> +prepare_script_args...`, run in
    process from tmp_path: Gym's own composition, cache lookup and output-path check, with prepare() replaced by a stub
    that writes an empty file where the real one would. Returns the (version, prompt_format) of each prepare() call."""
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "benchmarks" / "birdbench" / "data"
    calls = []

    def fake_prepare(**kwargs):
        calls.append((kwargs["version"], kwargs["prompt_format"]))
        name = birdbench_prepare.output_fpath_for(
            kwargs["dialect"], kwargs["split"], kwargs["version"], kwargs["prompt_format"]
        ).name
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / name).write_text("{}\n")
        return data_dir / name

    monkeypatch.setattr(benchmark_preparation, "_prepare_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(_prepare_worker, "load_preparer", lambda reference: SimpleNamespace(prepare=fake_prepare))

    def _run(cached=True, **prepare_script_args):
        calls.clear()
        config = _gym_config(prepare_script_args=prepare_script_args, use_cached_prepared_benchmarks=cached)
        prepare_datasets(datasets_to_prepare(config), PrepareBenchmarkConfig.model_validate(config))
        return list(calls)

    return _run


def test_a_cached_default_file_is_reused_for_the_default_only(gym_eval_prepare):
    assert gym_eval_prepare() == [("dev_20240627", "default")]  # cold cache
    assert gym_eval_prepare() == []  # the default file is now cached
    assert gym_eval_prepare(version="minidev_20261028") == [("minidev_20261028", "default")]
    assert gym_eval_prepare(prompt_format="schema_only") == [("dev_20240627", "schema_only")]
    assert gym_eval_prepare(version="minidev_20261028", prompt_format="schema_only") == [
        ("minidev_20261028", "schema_only")
    ]
    assert gym_eval_prepare(version="minidev_20261028") == []  # its own file is now cached


def test_without_the_cache_a_direct_version_prepares_and_passes_the_path_check(gym_eval_prepare):
    assert gym_eval_prepare(cached=False, version="dev_20251106") == [("dev_20251106", "default")]


def test_every_split_and_dialect_has_a_distinct_prepared_file():
    paths = {
        birdbench_prepare.output_fpath_for(dialect, split)
        for split in birdbench_prepare.SPLITS
        for dialect in birdbench_prepare.DIALECTS
    }
    assert len(paths) == len(birdbench_prepare.SPLITS) * len(birdbench_prepare.DIALECTS)
    assert birdbench_prepare.output_fpath_for("spark", "train").name == "birdbench_spark_train.jsonl"
