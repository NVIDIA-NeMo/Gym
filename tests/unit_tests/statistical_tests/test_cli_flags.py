# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List

from nemo_gym.statistical_tests.paired import PairedTestConfig


class TestCliFlagTranslation:
    """`gym eval stat-test`'s flags must survive the argparse -> Hydra override -> pydantic round trip."""

    def _overrides(self, argv: List[str]) -> List[str]:
        from nemo_gym.cli.main import build_parser

        args, unknown = build_parser().parse_known_args(argv)
        assert unknown == [], f"flags left unparsed: {unknown}"
        return [token for flag in args._command.flags for token in flag.translate_to_hydra(args)]

    def _config(self, argv: List[str]) -> PairedTestConfig:
        from hydra.core.override_parser.overrides_parser import OverridesParser

        parsed = OverridesParser.create().parse_overrides(self._overrides(argv))
        return PairedTestConfig.model_validate({o.key_or_group: o.value() for o in parsed})

    def test_paths_round_trip_through_hydra(self):
        config = self._config(
            ["eval", "stat-test", "--baseline", "runs/a/rollouts.jsonl", "--candidates", "runs/b/rollouts.jsonl"]
        )
        assert config.baseline_rollouts_jsonl_fpath == "runs/a/rollouts.jsonl"
        assert config.candidate_rollouts_jsonl_fpaths == ["runs/b/rollouts.jsonl"]
        assert config.metric is None
        assert config.margin is None
        assert config.alpha == 0.05

    def test_metric_margin_alpha_translate(self):
        config = self._config(
            [
                "eval",
                "stat-test",
                "--baseline",
                "a.jsonl",
                "--candidates",
                "b.jsonl",
                "--metric",
                "reward,output_tokens",
                "--margin",
                "0.01",
                "--alpha",
                "0.1",
            ]
        )
        assert config.metric == ["reward", "output_tokens"]
        assert config.margin == 0.01
        assert config.alpha == 0.1

    def test_unset_flags_contribute_no_overrides(self):
        overrides = self._overrides(["eval", "stat-test", "--baseline", "a.jsonl", "--candidates", "b.jsonl"])
        assert not [token for token in overrides if "metric" in token or "margin" in token]

    def test_test_selector_round_trips_and_defaults_to_paired(self):
        argv = ["eval", "stat-test", "--baseline", "a.jsonl", "--candidates", "b.jsonl"]
        assert self._config([*argv, "--test", "paired"]).test == "paired"
        # Unset, the flag contributes no override and the pydantic default supplies the same value.
        assert not [token for token in self._overrides(argv) if token.startswith("+test=")]
        assert self._config(argv).test == "paired"

    def test_eval_compare_gets_the_same_statistical_flags(self):
        overrides = self._overrides(
            [
                "eval",
                "compare",
                "--baseline",
                "a.jsonl",
                "--candidates",
                "b.jsonl",
                "--metric",
                "reward",
                "--margin",
                "0.01",
                "--stats-output-dir",
                "elsewhere",
                "--no-stats",
            ]
        )
        assert '+metric=["reward"]' in overrides
        assert "+margin=0.01" in overrides
        assert '+stats_output_dirpath="elsewhere"' in overrides
        assert "+no_stats=true" in overrides
