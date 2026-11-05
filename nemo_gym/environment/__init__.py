# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment authoring contracts and tooling."""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.environment.agent_registry import discover_agents
    from nemo_gym.environment.benchmarks import BENCHMARKS_DIR, BenchmarkConfig
    from nemo_gym.environment.deliverables import IGNORE_FILES, is_deliverable
    from nemo_gym.environment.in_process import (
        EvalPlan,
        PlannedBenchmark,
        TaskResult,
        TaskRunner,
        plan_benchmark,
        serve_benchmark,
    )
    from nemo_gym.environment.manifest import EnvironmentManifest, load_manifest, resolve_manifest_config_path
    from nemo_gym.environment.registry import discover_environment_catalog


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "BENCHMARKS_DIR": "benchmarks",
    "BenchmarkConfig": "benchmarks",
    "EnvironmentManifest": "manifest",
    "EvalPlan": "in_process",
    "IGNORE_FILES": "deliverables",
    "PlannedBenchmark": "in_process",
    "TaskResult": "in_process",
    "TaskRunner": "in_process",
    "discover_agents": "agent_registry",
    "discover_environment_catalog": "registry",
    "is_deliverable": "deliverables",
    "load_manifest": "manifest",
    "plan_benchmark": "in_process",
    "resolve_manifest_config_path": "manifest",
    "serve_benchmark": "in_process",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
