# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``benchmarks/birdbench/config.yaml`` and its manifest."""

from pathlib import Path

import yaml

from benchmarks.birdbench import prepare as birdbench_prepare


BENCHMARK_DIR = Path(__file__).parent.parent
REPO_ROOT = BENCHMARK_DIR.parent.parent


def _dataset():
    cfg = yaml.safe_load((BENCHMARK_DIR / "config.yaml").read_text())
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
    assert dataset["jsonl_fpath"] == "benchmarks/birdbench/data/birdbench_sqlite_benchmark.jsonl"
    assert cfg["prepare_script_args"]["dialect"] == "sqlite"
    assert birdbench_prepare.output_fpath_for("sqlite") == REPO_ROOT / dataset["jsonl_fpath"]


def test_every_dialect_has_its_own_prompt_and_file_named_by_dialect():
    # eval.sh builds `prompts/<dialect>.yaml` for non-default dialects; sqlite keeps prompts/default.yaml.
    assert (REPO_ROOT / _dataset()[1]["prompt_config"]).is_file()
    for dialect in birdbench_prepare.DIALECTS:
        assert birdbench_prepare.output_fpath_for(dialect).name == f"birdbench_{dialect}_benchmark.jsonl"
        if dialect != "sqlite":
            assert (BENCHMARK_DIR / "prompts" / f"{dialect}.yaml").is_file()


def test_default_prepare_script_args_are_dev_sqlite():
    cfg, _ = _dataset()
    assert (cfg["prepare_script_args"]["split"], cfg["prepare_script_args"]["dialect"]) == ("dev", "sqlite")


def test_every_split_and_dialect_has_a_distinct_prepared_file():
    paths = {
        birdbench_prepare.output_fpath_for(dialect, split)
        for split in birdbench_prepare.SPLITS
        for dialect in birdbench_prepare.DIALECTS
    }
    assert len(paths) == len(birdbench_prepare.SPLITS) * len(birdbench_prepare.DIALECTS)
    assert birdbench_prepare.output_fpath_for("spark", "train").name == "birdbench_train_spark.jsonl"
