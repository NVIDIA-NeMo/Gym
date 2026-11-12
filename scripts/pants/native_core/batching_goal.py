# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only Pants goal: use the running engine's real Python test partitions."""

import json
from collections.abc import Iterable

from pants.backend.python.goals.pytest_runner import PyTestRequest, partition_python_tests
from pants.backend.python.subsystems.pytest import PythonTestFieldSet
from pants.core.goals.test import TestFieldSet
from pants.engine.console import Console
from pants.engine.goal import Goal, GoalSubsystem
from pants.engine.internals import native_engine
from pants.engine.internals.specs_rules import find_valid_field_sets_for_target_roots
from pants.engine.rules import Rule, collect_rules, goal_rule, implicitly
from pants.engine.target import NoApplicableTargetsBehavior, TargetRootsToFieldSetsRequest
from pants.util.collections import partition_sequentially
from pants.version import VERSION
from scripts.pants.native_core.batching import single_batch_plan


class NativeBatchPlanSubsystem(GoalSubsystem):
    name = "native-batch-plan"
    help = "Verify a single forced core batch with the installed Pants partitioners."


class NativeBatchPlan(Goal):
    subsystem_cls = NativeBatchPlanSubsystem
    environment_behavior = Goal.EnvironmentBehavior.LOCAL_ONLY


@goal_rule
async def native_batch_plan(console: Console) -> NativeBatchPlan:
    roots = await find_valid_field_sets_for_target_roots(
        TargetRootsToFieldSetsRequest(
            TestFieldSet,
            goal_description="The native batch planner",
            no_applicable_targets_behavior=NoApplicableTargetsBehavior.error,
        ),
        **implicitly(),
    )
    fields = roots.field_sets
    if any(not isinstance(field, PythonTestFieldSet) for field in fields):
        raise ValueError("Native batch planner only supports Python test targets")
    partitions = await partition_python_tests(PyTestRequest.PartitionRequest(fields), **implicitly())
    plan = single_batch_plan(
        [[str(field.address) for field in group.elements] for group in partitions],
        pants_version=VERSION,
        hash_prefix=native_engine.hash_prefix_zero_bits,
        partition=partition_sequentially,
    )
    metadata = next(iter(partitions)).metadata
    plan["compatibility"] = {
        "resolve": metadata.resolve,
        "environment": metadata.environment,
        "interpreter_constraints": str(metadata.interpreter_constraints),
        "extra_env_vars": list(metadata.extra_env_vars),
        "xdist_concurrency": metadata.xdist_concurrency,
        "tag": metadata.compatability_tag,
    }
    console.print_stdout(json.dumps(plan, sort_keys=True))
    return NativeBatchPlan(exit_code=0)


def rules() -> Iterable[Rule]:
    return collect_rules()
