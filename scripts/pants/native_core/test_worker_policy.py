# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep sampled CI worker scheduling and resource limits authoritative."""

import os
from pathlib import Path

import pytest
from run import core_worker_count, core_worker_environment


def source_script(root: Path, options: str, *, helper: str | None = None) -> None:
    scripts = root / "scripts/ci"
    scripts.mkdir(parents=True)
    script = f"pytest_addopts='{options}'\n"
    if helper is not None:
        script += 'source "${ci_dir}/test_worker_env.sh"\n'
        (scripts / "test_worker_env.sh").write_text(helper)
    (scripts / "core_unit_tests.sh").write_text(script)


def test_historical_serial_source_does_not_acquire_workers(tmp_path):
    source_script(tmp_path, '-m "not sandbox" --cov-report=')
    assert core_worker_count(tmp_path) is None
    environment = {"OMP_NUM_THREADS": "7"}
    assert core_worker_environment(tmp_path, environment) == environment


def test_sampled_worker_count_and_thread_limits_override_host_settings(tmp_path):
    source = tmp_path / "sample with spaces"
    source_script(
        source,
        '-m "not sandbox" -n 4 --dist=loadfile --color=yes',
        helper="export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=5 NUMEXPR_NUM_THREADS=7\n",
    )
    assert core_worker_count(source) == 4
    before = dict(os.environ, OMP_NUM_THREADS="99", RETAINED_ENV="yes")
    after = core_worker_environment(source, before)
    assert {
        name: after[name]
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    } == {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "3", "MKL_NUM_THREADS": "5", "NUMEXPR_NUM_THREADS": "7"}
    assert after["RETAINED_ENV"] == "yes"
    assert before["OMP_NUM_THREADS"] == "99"


@pytest.mark.parametrize(
    "options", ["-n auto --dist=loadfile", "-n 4 --dist=loadscope", "--numprocesses=4", "-n4", "-n"]
)
def test_changed_worker_contract_requires_inspection(tmp_path, options):
    source_script(tmp_path, options)
    with pytest.raises(ValueError, match="worker"):
        core_worker_count(tmp_path)
