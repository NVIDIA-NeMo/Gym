# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which statistical test `gym eval stat-test --test <name>` runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Type

from nemo_gym.cli.utils import did_you_mean
from nemo_gym.config_types import ConfigError
from nemo_gym.statistical_tests import paired
from nemo_gym.statistical_tests.common import report_stem, resolve_output_dir, write_reports
from nemo_gym.statistical_tests.schema import StatTestConfig, StatTestReport


@dataclass(frozen=True)
class StatTest:
    config_type: Type[StatTestConfig]
    build_report: Callable[[StatTestConfig, str], StatTestReport]
    render_markdown: Callable[[StatTestReport], str]
    summary: Callable[[StatTestReport, Sequence[Path]], Sequence[str]]


STAT_TESTS: Dict[str, StatTest] = {
    "paired": StatTest(
        config_type=paired.PairedTestConfig,
        build_report=paired.build_report,
        render_markdown=paired.render_markdown,
        summary=paired.summary,
    ),
}


def resolve_stat_test(name: str) -> StatTest:
    if name not in STAT_TESTS:
        raise ConfigError(
            f"Unknown statistical test '{name}'. Available: {', '.join(sorted(STAT_TESTS))}."
            + did_you_mean(name, STAT_TESTS)
        )
    return STAT_TESTS[name]


def run_stat_test(test: StatTest, config: StatTestConfig, command: str) -> Tuple[StatTestReport, List[Path]]:
    report = test.build_report(config, command)
    return report, write_reports(
        resolve_output_dir(config),
        report_stem(config),
        report_format=config.report_format,
        markdown=test.render_markdown(report),
        payload=report.model_dump(mode="json"),
    )
