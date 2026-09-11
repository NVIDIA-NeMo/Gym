# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which statistical test `gym eval stat-test --test <name>` runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type

from nemo_gym.cli.utils import did_you_mean
from nemo_gym.config_types import ConfigError
from nemo_gym.statistical_tests import paired_t_test
from nemo_gym.statistical_tests.common import render_markdown, report_stem, resolve_output_dir, write_reports
from nemo_gym.statistical_tests.schema import StatTestConfig, StatTestReport


@dataclass(frozen=True)
class StatTest:
    config_type: Type[StatTestConfig]
    build_report: Callable[[StatTestConfig, str], StatTestReport]


STAT_TESTS: Dict[str, StatTest] = {
    "paired-t-test": StatTest(
        config_type=paired_t_test.PairedTTestConfig,
        build_report=paired_t_test.build_report,
    ),
}


def resolve_stat_test(name: str) -> StatTest:
    if name not in STAT_TESTS:
        raise ConfigError(
            f"Unknown statistical test '{name}'. Available: {', '.join(sorted(STAT_TESTS))}."
            + did_you_mean(name, STAT_TESTS)
        )
    return STAT_TESTS[name]


def build_config(test: StatTest, config_dict: Any) -> StatTestConfig:
    """Validate `config_dict` against `test`, rejecting another test's flags instead of ignoring them.

    Pydantic drops unknown keys, so without this a flag belonging to a different test (`--margin` under
    a test with no margin) would be accepted and silently have no effect. Derived from the registry, so
    a new test needs no edit here.
    """
    own_fields = set(test.config_type.model_fields)
    other_fields = {f for t in STAT_TESTS.values() for f in t.config_type.model_fields} - own_fields
    unsupported = sorted(f for f in other_fields if config_dict.get(f) is not None)
    if unsupported:
        flags = ", ".join(f"--{f.replace('_', '-')}" for f in unsupported)
        name = test.config_type.model_fields["test"].default
        raise ConfigError(f"{flags} {'is' if len(unsupported) == 1 else 'are'} not a parameter of '{name}'.")
    return test.config_type.model_validate(config_dict)


def run_stat_test(test: StatTest, config: StatTestConfig, command: str) -> Tuple[StatTestReport, List[Path]]:
    report = test.build_report(config, command)
    return report, write_reports(
        resolve_output_dir(config),
        report_stem(config, report),
        report_format=config.report_format,
        markdown=render_markdown(report),
        payload=report.model_dump(mode="json"),
    )
