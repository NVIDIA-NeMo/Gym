# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import shim: ``benchmarks/nemotron_3.5_super`` is not an importable package name (it has a dot)."""

import importlib.util
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    "nemotron_3_5_super_derived_metrics",
    Path(__file__).parent / "nemotron_3.5_super" / "analysis" / "derived_metrics.py",
)
derived_metrics = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(derived_metrics)

derive = derived_metrics.derive
summarize = derived_metrics.summarize
union = derived_metrics.union
total = derived_metrics.total
subtract = derived_metrics.subtract
intersect = derived_metrics.intersect
