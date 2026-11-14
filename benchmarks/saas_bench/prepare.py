# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Delegate task conversion to the provisioned benchmark source."""

import os
import runpy
from pathlib import Path


def prepare(output_fpath=None, **kwargs):
    source = Path(os.environ["SAAS_BENCH_ROOT"]).resolve(strict=True)
    exporter = source / "integrations/gym/benchmarks/saas_bench/prepare.py"
    return runpy.run_path(str(exporter))["prepare"](output_fpath=output_fpath, **kwargs)
