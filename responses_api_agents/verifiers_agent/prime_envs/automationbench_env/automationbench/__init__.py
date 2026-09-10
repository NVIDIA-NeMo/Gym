# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""AutomationBench - A benchmark for evaluating AI agents on realistic business workflows."""

from automationbench.schema.world import WorldState

try:
    # Optional import: runner depends on external evaluation stack (verifiers/agents).
    from automationbench.runner import AutomationBenchEnv  # type: ignore

except Exception:  # pragma: no cover
    # Keep lightweight imports (e.g., schema + task builders) usable even when the
    # evaluation stack isn't available or isn't permitted in the runtime.
    AutomationBenchEnv = None  # type: ignore

__all__ = ["AutomationBenchEnv", "WorldState"]
