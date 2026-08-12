# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.biomysterybench import prepare_official, prepare_test
from nemo_gym.benchmarks import BenchmarkConfig


@pytest.mark.parametrize(
    ("directory", "dataset_name", "repeats", "prepare_script"),
    [
        ("biomysterybench", "biomysterybench", 5, "prepare_official.py"),
        ("biomysterybench_test", "biomysterybench_test", 1, "prepare_test.py"),
        ("biomysterybench_v11", "biomysterybench_v11", 5, "prepare.py"),
    ],
)
def test_benchmark_variant_resolves_one_expected_dataset(
    directory: str,
    dataset_name: str,
    repeats: int,
    prepare_script: str,
) -> None:
    repository_root = Path(__file__).parents[3]
    config_path = repository_root / "benchmarks" / directory / "config.yaml"
    benchmark = BenchmarkConfig.from_config_path(config_path)
    assert benchmark is not None
    assert benchmark.name == dataset_name
    assert benchmark.num_repeats == repeats
    assert benchmark.dataset.prepare_script.name == prepare_script


def test_official_entrypoint_pins_published_release() -> None:
    prepared = Path("benchmarks/biomysterybench/data/biomysterybench_official_99.jsonl")
    with patch.object(prepare_official, "_prepare", return_value=prepared) as inner:
        assert prepare_official.prepare() == prepared
    inner.assert_called_once_with(release_name="official-99")


def test_test_entrypoint_pins_one_official_task() -> None:
    prepared = prepare_test.DATA_DIR / "biomysterybench_official_test.jsonl"
    with patch.object(prepare_test, "_prepare", return_value=prepared) as inner:
        assert prepare_test.prepare() == prepared
    inner.assert_called_once_with(
        release_name="official-99",
        task_ids=["hb013"],
        output=prepared,
    )
