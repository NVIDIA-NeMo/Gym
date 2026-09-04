# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse

import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.statistical_tests import paired
from nemo_gym.statistical_tests.paired import PairedTestConfig
from nemo_gym.statistical_tests.registry import STAT_TESTS, StatTest, resolve_stat_test
from nemo_gym.statistical_tests.schema import DEFAULT_STAT_TEST, StatTestConfig, StatTestReport


BASE = {"baseline_rollouts_jsonl_fpath": "a.jsonl", "candidate_rollouts_jsonl_fpaths": ["b.jsonl"]}


class TestStatTestRegistry:
    @pytest.mark.parametrize("name", sorted(STAT_TESTS))
    def test_every_registered_test_is_fully_wired(self, name):
        test = STAT_TESTS[name]
        assert issubclass(test.config_type, StatTestConfig)
        assert all(callable(fn) for fn in (test.build_report, test.render_markdown, test.summary))
        assert test.config_type.model_fields["test"].default == name

    def test_paired_is_the_default_and_resolves_to_the_paired_implementation(self):
        assert DEFAULT_STAT_TEST == "paired"
        assert StatTestConfig.model_fields["test"].default == DEFAULT_STAT_TEST

        test = resolve_stat_test(DEFAULT_STAT_TEST)
        assert test.config_type is PairedTestConfig
        assert test.build_report is paired.build_report
        assert test.render_markdown is paired.render_markdown
        assert test.summary is paired.summary

    def test_unknown_test_name_lists_what_exists_and_suggests_the_close_one(self):
        with pytest.raises(ConfigError) as excinfo:
            resolve_stat_test("paried")
        message = str(excinfo.value)
        assert "Unknown statistical test 'paried'" in message
        assert "Did you mean `paired`?" in message

    def test_stat_test_runs_the_test_the_name_selected(self, monkeypatch, capsys, tmp_path):
        """A stub entry must be dispatched to instead of the paired implementation."""
        from nemo_gym.cli.eval import stat_test
        from nemo_gym.statistical_tests import registry

        stub_report = StatTestReport(
            generated_at="2026-01-01T00:00:00+00:00",
            nemo_gym_version="0.0.0",
            command="gym eval stat-test ...",
            test="paired",
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
            "paired",
            StatTest(
                config_type=PairedTestConfig,
                build_report=lambda config, command: (calls.append(command), stub_report)[1],
                render_markdown=lambda report: "stub markdown",
                summary=lambda report, written: ("stub ran",),
            ),
        )

        stat_test(PairedTestConfig.model_validate({**BASE, "output_dirpath": str(tmp_path)}))

        assert calls, "the registered build_report was never called -- dispatch is still hardcoded"
        assert calls[0].startswith("gym eval compare")
        assert "stub ran" in capsys.readouterr().out
        assert (tmp_path / "paired__two-sided__alpha-0.05.md").read_text() == "stub markdown"

    def test_cli_test_flag_choices_match_the_registry(self):
        from nemo_gym.cli.main import COMMANDS

        parser = argparse.ArgumentParser()
        for flag in COMMANDS["eval stat-test"].flags:
            flag.register(parser)
        action = next(a for a in parser._actions if "--test" in a.option_strings)
        assert set(action.choices) == set(STAT_TESTS)
