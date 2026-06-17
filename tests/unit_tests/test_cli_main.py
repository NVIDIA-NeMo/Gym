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
import logging
import sys

import pytest
from pytest import MonkeyPatch

import nemo_gym.cli.main as cli_main
import nemo_gym.global_config as gc
from nemo_gym.cli.main import main
from nemo_gym.global_config import NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME


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


class TestDatasetFlags:
    def test_upload_hf_default(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(
            monkeypatch,
            ["dataset", "upload", "-i", "data/train.jsonl", "--name", "my_ds", "--split", "train", "--create-pr"],
        )
        assert target == "nemo_gym.cli.dataset:upload_jsonl_dataset_to_hf_cli"
        assert set(overrides) == {
            "+input_jsonl_fpath=data/train.jsonl",
            "+dataset_name=my_ds",
            "+split=train",
            "+create_pr=true",
        }

    def test_upload_gitlab(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(
            monkeypatch,
            [
                "dataset",
                "upload",
                "--storage",
                "gitlab",
                "-i",
                "data/train.jsonl",
                "--name",
                "my_ds",
                "--revision",
                "0.0.1",
            ],
        )
        assert target == "nemo_gym.cli.dataset:upload_jsonl_dataset_cli"
        overrides.remove(
            "+revision=0.0.1"
        )  # we set both version and revision because GitLab and HF use different keys
        assert set(overrides) == {
            "+input_jsonl_fpath=data/train.jsonl",
            "+dataset_name=my_ds",
            "+version=0.0.1",
        }

    def test_download_hf_default(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(
            monkeypatch,
            [
                "dataset",
                "download",
                "--repo-id",
                "org/my_ds",
                "--artifact",
                "train.jsonl",
                "--output-dir",
                "./data",
                "--split",
                "train",
            ],
        )
        assert target == "nemo_gym.cli.dataset:download_jsonl_dataset_from_hf_cli"
        assert set(overrides) == {
            "+repo_id=org/my_ds",
            "+artifact_fpath=train.jsonl",
            "+output_dirpath=./data",
            "+split=train",
        }

    def test_download_gitlab(self, monkeypatch: MonkeyPatch) -> None:
        # On download, --revision is GitLab-only and maps to +version (HF download has no revision field).
        target, overrides = _dispatch_for(
            monkeypatch,
            [
                "dataset",
                "download",
                "--storage",
                "gitlab",
                "--name",
                "my_ds",
                "--revision",
                "0.0.1",
                "--artifact",
                "train.jsonl",
                "-o",
                "./train.jsonl",
            ],
        )
        assert target == "nemo_gym.cli.dataset:download_jsonl_dataset_cli"
        assert set(overrides) == {
            "+dataset_name=my_ds",
            "+version=0.0.1",
            "+artifact_fpath=train.jsonl",
            "+output_fpath=./train.jsonl",
        }

    def test_rm(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(monkeypatch, ["dataset", "rm", "--name", "my_ds"])
        assert target == "nemo_gym.cli.dataset:delete_jsonl_dataset_from_gitlab_cli"
        assert overrides == ["+dataset_name=my_ds"]

    def test_migrate_revision_maps_to_hf_revision(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(
            monkeypatch,
            ["dataset", "migrate", "-i", "data/train.jsonl", "--name", "my_ds", "--revision", "r1", "--create-pr"],
        )
        assert target == "nemo_gym.cli.dataset:upload_jsonl_dataset_to_hf_and_delete_gitlab_cli"
        assert set(overrides) == {
            "+input_jsonl_fpath=data/train.jsonl",
            "+dataset_name=my_ds",
            "+revision=r1",
            "+create_pr=true",
        }

    def test_render(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(
            monkeypatch, ["dataset", "render", "-i", "raw.jsonl", "--prompt-config", "p.yaml", "-o", "out.jsonl"]
        )
        assert target == "nemo_gym.cli.dataset:materialize_prompts_cli"
        assert overrides == ["+input_jsonl_fpath=raw.jsonl", "+prompt_config=p.yaml", "+output_jsonl_fpath=out.jsonl"]

    def test_collate(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(
            monkeypatch,
            [
                "dataset",
                "collate",
                "--config",
                "c.yaml",
                "--mode",
                "train_preparation",
                "--output-dir",
                "./prep",
                "--download",
            ],
        )
        assert target == "nemo_gym.cli.dataset:prepare_data"
        assert set(overrides) == {
            "+config_paths=[c.yaml]",
            "+mode=train_preparation",
            "+output_dirpath=./prep",
            "+should_download=true",
        }

    def test_bool_flags_omitted_when_unset(self, monkeypatch: MonkeyPatch) -> None:
        # --create-pr not passed -> no +create_pr override leaks in.
        _, overrides = _dispatch_for(monkeypatch, ["dataset", "upload", "--name", "my_ds"])
        assert overrides == ["+dataset_name=my_ds"]

    def test_collate_mode_rejects_invalid_choice(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["gym", "dataset", "collate", "--mode", "bogus"])
        with pytest.raises(SystemExit):
            main()


class TestEvalAggregateFlags:
    def test_output_flag(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(monkeypatch, ["eval", "aggregate", "-o", "out.jsonl"])
        assert target == "nemo_gym.cli.eval:aggregate_rollouts"
        assert overrides == ["+output_jsonl_fpath=out.jsonl"]


class TestEvalProfileFlags:
    def test_profile_flags(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(
            monkeypatch, ["eval", "profile", "--inputs", "in.jsonl", "--rollouts", "r.jsonl"]
        )
        assert target == "nemo_gym.cli.eval:reward_profile"
        assert set(overrides) == {
            "+materialized_inputs_jsonl_fpath=in.jsonl",
            "+rollouts_jsonl_fpath=r.jsonl",
        }

    def test_profile_does_not_accept_config(self, monkeypatch: MonkeyPatch) -> None:
        # reward_profile reads file paths, not config_paths, so --config is not offered and is rejected.
        monkeypatch.setattr(cli_main, "dispatch", lambda target, overrides: None)
        monkeypatch.setattr(sys, "argv", ["gym", "eval", "profile", "--config", "x.yaml"])
        with pytest.raises(SystemExit):
            main()


class TestEnvRunFlags:
    def test_model_flags(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(
            monkeypatch,
            [
                "env",
                "run",
                "--config",
                "c.yaml",
                "--model-name",
                "gpt",
                "--model-url",
                "http://x",
                "--model-api-key",
                "k",
            ],
        )
        assert target == "nemo_gym.cli.env:run"
        assert set(overrides) == {
            "+config_paths=[c.yaml]",
            "+policy_model_name=gpt",
            "+policy_base_url=http://x",
            "+policy_api_key=k",
        }


class TestEnvInitFlags:
    def test_resource_server_translates_to_entrypoint(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(monkeypatch, ["env", "init", "--resource-server", "my_server"])
        assert target == "nemo_gym.cli.env:init_resources_server"
        assert overrides == ["+entrypoint=resources_servers/my_server"]


class TestEnvPackagesFlags:
    def test_flags(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(monkeypatch, ["env", "packages", "--resource-server", "gpqa", "--outdated"])
        assert target == "nemo_gym.cli.env:pip_list"
        assert set(overrides) == {
            "+entrypoint=resources_servers/gpqa",
            "+outdated=true",
        }


class TestVerboseFlag:
    @pytest.mark.parametrize("flag", ["-v", "--verbose"])
    def test_verbose_injects_config_override(self, monkeypatch: MonkeyPatch, flag: str) -> None:
        # --verbose flows through the config (so it reaches servers), not just the local logger.
        _, overrides = _dispatch_for(monkeypatch, ["env", "status", flag])
        assert overrides == ["+verbose=true"]

    def test_no_verbose_no_override(self, monkeypatch: MonkeyPatch) -> None:
        _, overrides = _dispatch_for(monkeypatch, ["env", "status"])
        assert overrides == []

    def test_verbose_prepended_before_other_overrides(self, monkeypatch: MonkeyPatch) -> None:
        _, overrides = _dispatch_for(monkeypatch, ["eval", "run", "--verbose", "--agent", "a", "+x=1"])
        assert "+verbose=true" in overrides
        assert "+agent_name=a" in overrides
        assert "+x=1" in overrides

    def test_config_verbose_sets_debug_on_load(self, monkeypatch: MonkeyPatch) -> None:
        # The server-side path: a config carrying `verbose` (forwarded via env var) raises the log level.
        monkeypatch.setattr(gc, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, "verbose: true\nsome_server: {}\n")
        root = logging.getLogger()
        original = root.level
        try:
            root.setLevel(logging.WARNING)
            gc.get_global_config_dict()
            assert root.level == logging.DEBUG
        finally:
            root.setLevel(original)

    def test_config_without_verbose_keeps_level(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(gc, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, "some_server: {}\n")
        root = logging.getLogger()
        original = root.level
        try:
            root.setLevel(logging.WARNING)
            gc.get_global_config_dict()
            assert root.level == logging.WARNING
        finally:
            root.setLevel(original)
