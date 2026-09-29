# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse

import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.statistical_tests import paired_t_test
from nemo_gym.statistical_tests.paired_t_test import PairedTTestConfig
from nemo_gym.statistical_tests.registry import STAT_TESTS, StatTest, build_config, resolve_stat_test
from nemo_gym.statistical_tests.schema import (
    DEFAULT_STAT_TEST,
    STATS_SUBDIR_NAME,
    StatTestConfig,
    StatTestReport,
    StatTestResult,
)


BASE = {"baseline_rollouts_jsonl_fpath": "a.jsonl", "candidate_rollouts_jsonl_fpaths": ["b.jsonl"]}


class TestStatTestRegistry:
    @pytest.mark.parametrize("name", sorted(STAT_TESTS))
    def test_every_registered_test_is_fully_wired(self, name):
        test = STAT_TESTS[name]
        assert issubclass(test.config_type, StatTestConfig)
        assert callable(test.build_report)
        assert test.config_type.model_fields["test"].default == name

    def test_paired_is_the_default_and_resolves_to_the_paired_implementation(self):
        assert DEFAULT_STAT_TEST == "paired-t-test"
        assert StatTestConfig.model_fields["test"].default == DEFAULT_STAT_TEST

        test = resolve_stat_test(DEFAULT_STAT_TEST)
        assert test.config_type is PairedTTestConfig
        assert test.build_report is paired_t_test.build_report

    def test_unknown_test_name_lists_what_exists_and_suggests_the_close_one(self):
        with pytest.raises(ConfigError) as excinfo:
            resolve_stat_test("paired-t-tes")
        message = str(excinfo.value)
        assert "Unknown statistical test 'paired-t-tes'" in message
        assert "Did you mean `paired-t-test`?" in message

    def test_stat_test_runs_the_test_the_name_selected(self, monkeypatch, capsys, tmp_path):
        """A stub entry must be dispatched to instead of the paired t-test implementation.

        Also the whole of what a new test supplies: a result that knows its own `line()`, and a
        `build_report`. Rendering, summarising, naming and writing come from the shared layer.
        """
        from nemo_gym.statistical_tests import registry
        from nemo_gym.statistical_tests.common import stat_test_from_config_dict

        class StubResult(StatTestResult):
            def line(self) -> str:
                return f"{self.metric} (stub test): ran"

        stub_report = StatTestReport(
            results=[StubResult(metric="stub-metric")],
            generated_at="2026-01-01T00:00:00+00:00",
            nemo_gym_version="0.0.0",
            command="gym eval stat-test ...",
            test="paired-t-test",
            baseline_rollouts_jsonl_fpath="a.jsonl",
            baseline_aggregate_metrics_fpath="a_aggregate_metrics.json",
            candidate_rollouts_jsonl_fpath="b.jsonl",
            candidate_aggregate_metrics_fpath="b_aggregate_metrics.json",
            baseline_agent="agent",
            candidate_agent="agent",
            baseline_task_count=1,
            candidate_task_count=1,
        )
        calls = []
        monkeypatch.setitem(
            registry.STAT_TESTS,
            "paired-t-test",
            StatTest(
                config_type=PairedTTestConfig,
                build_report=lambda config, command: (calls.append(command), stub_report)[1],
            ),
        )

        stat_test_from_config_dict({**BASE, "output_dirpath": str(tmp_path)}, "stat-test")

        assert calls, "the registered build_report was never called -- dispatch is still hardcoded"
        assert calls[0].startswith("gym eval stat-test")
        assert "stub-metric (stub test): ran" in capsys.readouterr().out
        # --output-dir is the parent: the report always lands in a statistical_tests/ inside it.
        stem = "paired-t-test__a__b__agent-agent__two-sided__alpha-0.05__*"
        (written,) = (tmp_path / STATS_SUBDIR_NAME).glob(f"{stem}.md")
        # The shared renderer wrote it, using only the stub's own `line()`.
        markdown = written.read_text()
        assert markdown.startswith("# Statistical tests results")
        assert "- stub-metric (stub test): ran" in markdown

    def test_cli_test_flag_choices_match_the_registry(self):
        from nemo_gym.cli.main import COMMANDS

        parser = argparse.ArgumentParser()
        for flag in COMMANDS["eval stat-test"].flags:
            flag.register(parser)
        action = next(a for a in parser._actions if "--test" in a.option_strings)
        assert set(action.choices) == set(STAT_TESTS)


class TestBuildConfig:
    """`build_config` rejects another test's flags instead of letting pydantic drop them."""

    def test_a_tests_own_flags_are_kept(self):
        config = build_config(resolve_stat_test("paired-t-test"), {**BASE, "metric": ["reward"], "margin": [0.01]})
        assert config.metric == ["reward"]
        assert config.margin == [0.01]

    def test_a_flag_the_selected_test_does_not_declare_is_an_error_not_a_silent_drop(self, monkeypatch):
        from nemo_gym.statistical_tests import registry

        class NoMarginConfig(StatTestConfig):
            test: str = "no_margin"

        no_margin = StatTest(
            config_type=NoMarginConfig,
            build_report=lambda config, command: None,
        )
        monkeypatch.setitem(registry.STAT_TESTS, "no_margin", no_margin)

        # Without the guard pydantic would ignore both keys and report a two-sided test on every metric.
        with pytest.raises(ConfigError) as excinfo:
            build_config(no_margin, {**BASE, "test": "no_margin", "margin": 0.01, "metric": ["reward"]})
        assert "--margin, --metric are not a parameter of 'no_margin'" in str(excinfo.value)

        assert build_config(no_margin, {**BASE, "test": "no_margin"}).test == "no_margin"

    def test_unset_flags_from_other_tests_are_ignored(self, monkeypatch):
        from nemo_gym.statistical_tests import registry

        class NoMarginConfig(StatTestConfig):
            test: str = "no_margin"

        no_margin = StatTest(
            config_type=NoMarginConfig,
            build_report=lambda config, command: None,
        )
        monkeypatch.setitem(registry.STAT_TESTS, "no_margin", no_margin)

        # `gym eval compare` always relays metric/margin, unset as None -- that must not trip the guard.
        assert build_config(no_margin, {**BASE, "test": "no_margin", "margin": None, "metric": None})
