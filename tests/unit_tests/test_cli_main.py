# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys

import pytest
from pytest import MonkeyPatch

import nemo_gym.cli.main as cli_main
from nemo_gym.cli.main import main


def _dispatch_for(monkeypatch: MonkeyPatch, argv: list[str]) -> tuple[str, list[str]]:
    """Run the gym router for `argv` and return the (target, overrides) handed to dispatch."""
    captured: dict = {}

    def fake_dispatch(target: str, overrides: list[str]) -> None:
        captured["target"] = target
        captured["overrides"] = overrides

    monkeypatch.setattr(cli_main, "dispatch", fake_dispatch)
    monkeypatch.setattr(sys, "argv", ["gym", *argv])
    main()
    return captured["target"], captured["overrides"]


# `gym <command>` -> the legacy ng_<command> function it dispatches to, for the config-accepting commands.
CONFIG_COMMANDS = [
    (["env", "run"], "nemo_gym.cli.env:run"),
    (["env", "resolve"], "nemo_gym.cli.env:dump_config"),
    (["eval", "prepare"], "nemo_gym.cli.eval:prepare_benchmark"),
    (["eval", "aggregate"], "nemo_gym.cli.eval:aggregate_rollouts"),
    (["eval", "run"], "nemo_gym.cli.eval:e2e_rollout_collection"),
    (["dataset", "collate"], "nemo_gym.cli.dataset:prepare_data"),
]


class TestConfigFlag:
    @pytest.mark.parametrize("command, expected_target", CONFIG_COMMANDS)
    def test_config_becomes_config_paths(self, monkeypatch: MonkeyPatch, command, expected_target) -> None:
        """`gym <command> --config X` dispatches to ng_<command> with +config_paths=[X]."""
        target, overrides = _dispatch_for(monkeypatch, [*command, "--config", "my.yaml"])
        assert target == expected_target
        assert overrides == ["+config_paths=[my.yaml]"]

    def test_repeated_config_joined_into_one_list(self, monkeypatch: MonkeyPatch) -> None:
        _, overrides = _dispatch_for(monkeypatch, ["env", "run", "--config", "a.yaml", "--config", "b.yaml"])

        # We have this set of asserts to avoid asserting configs order in the string
        assert len(overrides) == 1
        override = overrides[0]
        assert override.startswith("+config_paths=[")
        assert override.endswith("]")
        assert "a.yaml" in override
        assert "b.yaml" in override

    def test_config_is_prepended_before_passthrough_overrides(self, monkeypatch: MonkeyPatch) -> None:
        _, overrides = _dispatch_for(monkeypatch, ["env", "run", "--config", "a.yaml", "+foo=bar"])
        assert len(overrides) == 2
        assert "+config_paths=[a.yaml]" in overrides
        assert "+foo=bar" in overrides

    def test_without_config_no_config_paths_added(self, monkeypatch: MonkeyPatch) -> None:
        _, overrides = _dispatch_for(monkeypatch, ["env", "run", "+foo=bar"])
        assert overrides == ["+foo=bar"]

    def test_config_rejected_on_non_config_command(self, monkeypatch: MonkeyPatch) -> None:
        # `dataset rm` does not declare --config, so the router must reject it rather than leak it downstream.
        monkeypatch.setattr(cli_main, "dispatch", lambda target, overrides: None)
        monkeypatch.setattr(sys, "argv", ["gym", "dataset", "rm", "--config", "x.yaml"])
        with pytest.raises(SystemExit):
            main()


class TestStorageFlag:
    @pytest.mark.parametrize(
        "argv, expected_target",
        [
            (["dataset", "upload"], "nemo_gym.cli.dataset:upload_jsonl_dataset_to_hf_cli"),
            (["dataset", "upload", "--storage", "hf"], "nemo_gym.cli.dataset:upload_jsonl_dataset_to_hf_cli"),
            (["dataset", "upload", "--storage", "gitlab"], "nemo_gym.cli.dataset:upload_jsonl_dataset_cli"),
            (["dataset", "download"], "nemo_gym.cli.dataset:download_jsonl_dataset_from_hf_cli"),
            (["dataset", "download", "--storage", "hf"], "nemo_gym.cli.dataset:download_jsonl_dataset_from_hf_cli"),
            (["dataset", "download", "--storage", "gitlab"], "nemo_gym.cli.dataset:download_jsonl_dataset_cli"),
        ],
    )
    def test_storage_selects_backend(self, monkeypatch: MonkeyPatch, argv, expected_target) -> None:
        target, _ = _dispatch_for(monkeypatch, argv)
        assert target == expected_target

    def test_storage_does_not_leak_into_overrides(self, monkeypatch: MonkeyPatch) -> None:
        # --storage only selects the target; it must not appear in the Hydra overrides.
        _, overrides = _dispatch_for(monkeypatch, ["dataset", "upload", "--storage", "gitlab", "+foo=bar"])
        assert overrides == ["+foo=bar"]

    def test_invalid_storage_value_is_rejected(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["gym", "dataset", "upload", "--storage", "s3"])
        with pytest.raises(SystemExit):
            main()


class TestEvalRunFlags:
    @pytest.mark.parametrize(
        "flag_argv, expected_override",
        [
            (["--agent", "my_agent"], "+agent_name=my_agent"),
            (["-a", "my_agent"], "+agent_name=my_agent"),
            (["--input", "in.jsonl"], "+input_jsonl_fpath=in.jsonl"),
            (["-i", "in.jsonl"], "+input_jsonl_fpath=in.jsonl"),
            (["--output", "out.jsonl"], "+output_jsonl_fpath=out.jsonl"),
            (["-o", "out.jsonl"], "+output_jsonl_fpath=out.jsonl"),
            (["--limit", "1024"], "+limit=1024"),
            (["--num-repeats", "4"], "+num_repeats=4"),
            (["--concurrency", "10"], "+num_samples_in_parallel=10"),
            (["--prompt-config", "p.yaml"], "+prompt_config=p.yaml"),
            (["--split", "benchmark"], "+split=benchmark"),
            (["--model-name", "openai/gpt-oss-120b"], "+policy_model_name=openai/gpt-oss-120b"),
            (["--model-url", "http://0.0.0.0:10240/v1"], "+policy_base_url=http://0.0.0.0:10240/v1"),
            (["--model-api-key", "sk-your-api-key"], "+policy_api_key=sk-your-api-key"),
            (["--temperature", "1.0"], "+responses_create_params.temperature=1.0"),
            (["--top-p", "1.0"], "+responses_create_params.top_p=1.0"),
            (["--max-output-tokens", "4096"], "+responses_create_params.max_output_tokens=4096"),
            (["--resume"], "+resume_from_cache=true"),
        ],
    )
    def test_flag_maps_to_single_override(self, monkeypatch: MonkeyPatch, flag_argv, expected_override) -> None:
        _, overrides = _dispatch_for(monkeypatch, ["eval", "run", *flag_argv])
        assert overrides == [expected_override]

    def test_unset_flags_contribute_nothing(self, monkeypatch: MonkeyPatch) -> None:
        _, overrides = _dispatch_for(monkeypatch, ["eval", "run", "--agent", "x"])
        assert overrides == ["+agent_name=x"]

    def test_default_dispatches_e2e(self, monkeypatch: MonkeyPatch) -> None:
        target, _ = _dispatch_for(monkeypatch, ["eval", "run"])
        assert target == "nemo_gym.cli.eval:e2e_rollout_collection"

    def test_no_serve_dispatches_collect_without_override(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(monkeypatch, ["eval", "run", "--no-serve"])
        assert target == "nemo_gym.cli.eval:collect_rollouts"
        assert overrides == []

    def test_readme_collect_rollouts_example(self, monkeypatch: MonkeyPatch) -> None:
        # From resources_servers/my_weather_tool README:
        #   ng_collect_rollouts +agent_name=... +input_jsonl_fpath=... +output_jsonl_fpath=... +limit=1024 +num_repeats=1
        target, overrides = _dispatch_for(
            monkeypatch,
            [
                "eval",
                "run",
                "--no-serve",
                "--agent",
                "my_weather_tool_simple_agent",
                "--input",
                "resources_servers/my_weather_tool/data/example.jsonl",
                "--output",
                "resources_servers/my_weather_tool/data/example_rollouts.jsonl",
                "--limit",
                "1024",
                "--num-repeats",
                "1",
            ],
        )
        assert target == "nemo_gym.cli.eval:collect_rollouts"
        assert set(overrides) == {
            "+agent_name=my_weather_tool_simple_agent",
            "+input_jsonl_fpath=resources_servers/my_weather_tool/data/example.jsonl",
            "+output_jsonl_fpath=resources_servers/my_weather_tool/data/example_rollouts.jsonl",
            "+limit=1024",
            "+num_repeats=1",
        }

    def test_readme_model_and_sampling_example(self, monkeypatch: MonkeyPatch) -> None:
        # From the gpt-oss eval example: ++policy_* and ++responses_create_params.* overrides.
        _, overrides = _dispatch_for(
            monkeypatch,
            [
                "eval",
                "run",
                "--model-name",
                "openai/gpt-oss-120b",
                "--model-url",
                "http://0.0.0.0:10240/v1",
                "--model-api-key",
                "dummy_key",
                "--temperature",
                "1.0",
                "--top-p",
                "1.0",
            ],
        )
        assert set(overrides) == {
            "+policy_model_name=openai/gpt-oss-120b",
            "+policy_base_url=http://0.0.0.0:10240/v1",
            "+policy_api_key=dummy_key",
            "+responses_create_params.temperature=1.0",
            "+responses_create_params.top_p=1.0",
        }

    def test_flags_compose_with_config_and_passthrough(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(
            monkeypatch,
            [
                "eval",
                "run",
                "--no-serve",
                "--config",
                "b.yaml",
                "--agent",
                "a",
                "+responses_create_params.tool_choice=auto",
            ],
        )
        assert target == "nemo_gym.cli.eval:collect_rollouts"
        assert "+config_paths=[b.yaml]" in overrides
        assert "+agent_name=a" in overrides
        assert "+responses_create_params.tool_choice=auto" in overrides  # unknown +override passes through


class TestEnvTestResourceServerFlag:
    def test_no_resource_server_runs_all(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(monkeypatch, ["env", "test"])
        assert target == "nemo_gym.cli.env:test_all"
        assert overrides == []

    def test_resource_server_name_translates_to_entrypoint(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(monkeypatch, ["env", "test", "--resource-server", "gpqa"])
        assert target == "nemo_gym.cli.env:test"
        assert overrides == ["+entrypoint=resources_servers/gpqa"]

    def test_direct_entrypoint_override_also_runs_single(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(monkeypatch, ["env", "test", "+entrypoint=resources_servers/gpqa"])
        assert target == "nemo_gym.cli.env:test"
        assert overrides == ["+entrypoint=resources_servers/gpqa"]
