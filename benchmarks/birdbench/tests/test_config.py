# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``benchmarks/birdbench/config.yaml`` and its manifest."""

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from benchmarks.birdbench import prepare as birdbench_prepare


BENCHMARK_DIR = Path(__file__).parent.parent
REPO_ROOT = BENCHMARK_DIR.parent.parent


def _dataset(**prepare_script_args):
    """config.yaml with its interpolations resolved, as Gym resolves it; keyword arguments set prepare_script_args
    as `+prepare_script_args.<name>=<value>` does on the command line."""
    cfg = OmegaConf.merge(OmegaConf.load(BENCHMARK_DIR / "config.yaml"), {"prepare_script_args": prepare_script_args})
    cfg = OmegaConf.to_container(cfg, resolve=True)
    (dataset,) = cfg["birdbench_bird_sql_simple_agent"]["responses_api_agents"]["simple_agent"]["datasets"]
    return cfg, dataset


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
    # default's, and the matching prompt.
    _, dataset = _dataset(version=version, prompt_format=prompt_format)
    assert REPO_ROOT / dataset["jsonl_fpath"] == birdbench_prepare.output_fpath_for(
        "sqlite", "dev", version, prompt_format
    )
    assert dataset["prompt_config"] == f"benchmarks/birdbench/prompts/{prompt_format}_sqlite.yaml"
    assert (REPO_ROOT / dataset["prompt_config"]).is_file()


def test_every_split_and_dialect_has_a_distinct_prepared_file():
    paths = {
        birdbench_prepare.output_fpath_for(dialect, split)
        for split in birdbench_prepare.SPLITS
        for dialect in birdbench_prepare.DIALECTS
    }
    assert len(paths) == len(birdbench_prepare.SPLITS) * len(birdbench_prepare.DIALECTS)
    assert birdbench_prepare.output_fpath_for("spark", "train").name == "birdbench_spark_train.jsonl"
