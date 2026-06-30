# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import types

import pytest
from pytest import MonkeyPatch

import nemo_gym.cli.main as cli_main
import nemo_gym.global_config as gc
from nemo_gym import WORKING_DIR
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


def _split_overrides(overrides: list[str]) -> tuple[set[str], set[str]]:
    """Split overrides into (config paths, other overrides) as sets, so tests never assert ordering."""
    prefix = "+config_paths=["
    config_tokens = [o for o in overrides if o.startswith(prefix) and o.endswith("]")]
    assert len(config_tokens) <= 1  # --config and the asset selectors coalesce into a single token
    paths = set(config_tokens[0][len(prefix) : -1].split(",")) if config_tokens else set()
    others = {o for o in overrides if o not in config_tokens}
    return paths, others


# `gym <command>` -> the legacy ng_<command> function it dispatches to, for the config-accepting commands.
CONFIG_COMMANDS = [
    (["env", "start"], "nemo_gym.cli.env:run"),
    (["env", "resolve"], "nemo_gym.cli.env:dump_config"),
    (["env", "validate"], "nemo_gym.cli.env:validate"),
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
        _, overrides = _dispatch_for(monkeypatch, ["env", "start", "--config", "a.yaml", "--config", "b.yaml"])

        # We have this set of asserts to avoid asserting configs order in the string
        assert len(overrides) == 1
        override = overrides[0]
        assert override.startswith("+config_paths=[")
        assert override.endswith("]")
        assert "a.yaml" in override
        assert "b.yaml" in override

    def test_config_is_prepended_before_passthrough_overrides(self, monkeypatch: MonkeyPatch) -> None:
        _, overrides = _dispatch_for(monkeypatch, ["env", "start", "--config", "a.yaml", "+foo=bar"])
        assert len(overrides) == 2
        assert "+config_paths=[a.yaml]" in overrides
        assert "+foo=bar" in overrides

    def test_without_config_no_config_paths_added(self, monkeypatch: MonkeyPatch) -> None:
        _, overrides = _dispatch_for(monkeypatch, ["env", "start", "+foo=bar"])
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
            (["--model", "openai/gpt-oss-120b"], "+policy_model_name=openai/gpt-oss-120b"),
            (["-m", "openai/gpt-oss-120b"], "+policy_model_name=openai/gpt-oss-120b"),
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
                "--model",
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
        target, overrides = _dispatch_for(monkeypatch, ["env", "test", "--resources-server", "gpqa"])
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
        assert set(overrides) == {
            "+input_jsonl_fpath=raw.jsonl",
            "+prompt_config=p.yaml",
            "+output_jsonl_fpath=out.jsonl",
        }

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
    def test_aggregate_flags(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(
            monkeypatch, ["eval", "aggregate", "-i", "results/rollouts-*.jsonl", "-o", "out.jsonl"]
        )
        assert target == "nemo_gym.cli.eval:aggregate_rollouts"
        assert set(overrides) == {"+input_glob=results/rollouts-*.jsonl", "+output_jsonl_fpath=out.jsonl"}


class TestDispatch:
    """Exercise the real `dispatch` (every router test above stubs it), so argv rewriting and
    module/attribute resolution are actually covered."""

    def test_dispatch_imports_target_and_rewrites_argv(self, monkeypatch: MonkeyPatch) -> None:
        # Direct unit test of dispatch (no parser): it splits "module:func", imports the module,
        # resolves the attribute, rewrites sys.argv to argv[0] + overrides, then calls the target.
        # A synthetic module registered in sys.modules keeps this decoupled from any real target.
        captured: dict = {}

        def recorder() -> None:
            captured["argv"] = list(sys.argv)

        fake_module = types.ModuleType("my_module.my_submodule")
        fake_module.my_function = recorder
        monkeypatch.setitem(sys.modules, "my_module.my_submodule", fake_module)
        # argv starts with the parsed command tokens; dispatch must drop them but keep argv[0].
        monkeypatch.setattr(sys, "argv", ["my_command", "subcommand", "--some-flag", "some-value"])
        cli_main.dispatch("my_module.my_submodule:my_function", ["+a=1", "+b=2"])
        assert captured["argv"] == ["my_command", "+a=1", "+b=2"]

    def test_main_to_dispatch_end_to_end(self, monkeypatch: MonkeyPatch) -> None:
        # Drive dispatch from main() with only the leaf target stubbed: the target string is split,
        # the module is imported, the attribute is resolved, and sys.argv is rewritten for real.
        # Mix `--flag` asset selectors (translated and coalesced into +config_paths) with raw
        # `+`/`++` Hydra passthroughs.
        captured: dict = {}

        def recorder() -> None:
            captured["argv"] = list(sys.argv)

        monkeypatch.setattr("nemo_gym.cli.eval.e2e_rollout_collection", recorder)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "gym",
                "eval",
                "run",
                "--benchmark",
                "aime24",
                "--model-type",
                "openai_model",
                "++responses_create_params.reasoning.effort=low",
                "+wandb_project=gym-dev",
            ],
        )
        main()
        # dispatch drops the parsed command tokens, keeping argv[0] and the resolved overrides.
        assert captured["argv"][0] == "gym"
        config_paths, others = _split_overrides(captured["argv"][1:])
        assert any(p.endswith("benchmarks/aime24/config.yaml") for p in config_paths)
        assert any(p.endswith("responses_api_models/openai_model/configs/openai_model.yaml") for p in config_paths)
        assert others == {"++responses_create_params.reasoning.effort=low", "+wandb_project=gym-dev"}

    def test_dispatch_raises_on_unresolvable_attribute(self, monkeypatch: MonkeyPatch) -> None:
        # A typo in a Command target must surface as a resolution error, not silently no-op.
        monkeypatch.setattr(sys, "argv", ["gym"])
        with pytest.raises(AttributeError):
            cli_main.dispatch("nemo_gym.cli.eval:no_such_function", [])


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
                "start",
                "--config",
                "c.yaml",
                "--model",
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


class TestModelFlag:
    """`--model` is the single served-model identifier (name, HF id, or local checkpoint path) for any backend."""

    def test_local_vllm_deployment_flow(self, monkeypatch: MonkeyPatch) -> None:
        # The deployment invocation: select the local vLLM server type and pass the checkpoint to serve via --model.
        _, overrides = _dispatch_for(
            monkeypatch,
            ["eval", "run", "--model-type", "local_vllm_model", "--model", "Qwen/Qwen3-8B"],
        )
        paths, others = _split_overrides(overrides)
        assert paths == {str(WORKING_DIR / "responses_api_models/local_vllm_model/configs/local_vllm_model.yaml")}
        assert others == {"+policy_model_name=Qwen/Qwen3-8B"}

    def test_short_alias_on_env_run(self, monkeypatch: MonkeyPatch) -> None:
        _, overrides = _dispatch_for(monkeypatch, ["env", "start", "-m", "/ckpt/path"])
        assert overrides == ["+policy_model_name=/ckpt/path"]


class TestEnvInitFlags:
    def test_resource_server_translates_to_entrypoint(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(monkeypatch, ["env", "init", "--resources-server", "my_server"])
        assert target == "nemo_gym.cli.env:init_resources_server"
        assert overrides == ["+entrypoint=resources_servers/my_server"]


class TestEnvPackagesFlags:
    def test_flags(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(monkeypatch, ["env", "packages", "--resources-server", "gpqa", "--outdated"])
        assert target == "nemo_gym.cli.env:pip_list"
        assert set(overrides) == {
            "+entrypoint=resources_servers/gpqa",
            "+outdated=true",
        }


class TestJsonFlag:
    @pytest.mark.parametrize(
        "argv, expected_target",
        [
            (["list", "benchmarks", "--json"], "nemo_gym.cli.eval:list_benchmarks"),
            (["list", "agents", "--json"], "nemo_gym.cli.agents:list_agents"),
            (["env", "status", "--json"], "nemo_gym.cli.env:status"),
        ],
    )
    def test_json_becomes_config_override(self, monkeypatch: MonkeyPatch, argv, expected_target) -> None:
        # Reporting commands surface --json as the reserved `json` config key, read centrally by cli.output.emit.
        target, overrides = _dispatch_for(monkeypatch, argv)
        assert target == expected_target
        assert overrides == ["+json=true"]

    def test_no_json_no_override(self, monkeypatch: MonkeyPatch) -> None:
        _, overrides = _dispatch_for(monkeypatch, ["list", "benchmarks"])
        assert overrides == []


class TestSearch:
    def test_search_routes_to_list_with_query(self, monkeypatch: MonkeyPatch) -> None:
        # `gym search <query>` reuses the benchmarks listing, passing the query as the `query` config key.
        target, overrides = _dispatch_for(monkeypatch, ["search", "math"])
        assert target == "nemo_gym.cli.eval:list_benchmarks"
        assert overrides == ["+query=math"]

    def test_search_json(self, monkeypatch: MonkeyPatch) -> None:
        _, overrides = _dispatch_for(monkeypatch, ["search", "math", "--json"])
        assert set(overrides) == {"+query=math", "+json=true"}

    def test_version_json_dispatches_with_override(self, monkeypatch: MonkeyPatch) -> None:
        # `gym --version --json` is the top-level path; it still forwards +json=true to the version command.
        target, overrides = _dispatch_for(monkeypatch, ["--version", "--json"])
        assert target == "nemo_gym.cli.general:version"
        assert overrides == ["+json=true"]

    def test_env_packages_json_maps_to_uv_format(self, monkeypatch: MonkeyPatch) -> None:
        # env packages delegates to `uv pip list`, so --json maps onto its own --format=json rather than +json=true.
        target, overrides = _dispatch_for(monkeypatch, ["env", "packages", "--resources-server", "mcqa", "--json"])
        assert target == "nemo_gym.cli.env:pip_list"
        assert set(overrides) == {"+entrypoint=resources_servers/mcqa", "+format=json"}

    def test_env_packages_without_json(self, monkeypatch: MonkeyPatch) -> None:
        _, overrides = _dispatch_for(monkeypatch, ["env", "packages", "--resources-server", "mcqa"])
        assert overrides == ["+entrypoint=resources_servers/mcqa"]


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


class TestAssetSelectors:
    """Named selectors (--benchmark, --resources-server, --model-type) that resolve a name to a default config path.

    Each example mirrors a real invocation from the docs/READMEs, so the sugar stays faithful to the documented
    config paths it replaces. The legacy `+config_paths=[...]` form each one is derived from is cited inline.
    """

    @pytest.mark.parametrize(
        "argv, expected_config",
        [
            # benchmarks/gsm8k/README.md: ng_prepare_benchmark "+config_paths=[benchmarks/gsm8k/config.yaml]"
            (["eval", "prepare", "--benchmark", "gsm8k"], "benchmarks/gsm8k/config.yaml"),
            # benchmarks/aime25-x/README.md: ng_prepare_benchmark "+config_paths=[benchmarks/aime25-x/config.yaml]"
            (["eval", "prepare", "--benchmark", "aime25-x"], "benchmarks/aime25-x/config.yaml"),
            # benchmarks/gpqa/README.md: ng_run "+config_paths=[benchmarks/gpqa/config.yaml]" (start a benchmark's servers)
            (["env", "start", "--benchmark", "gpqa"], "benchmarks/gpqa/config.yaml"),
            # README.md / quickstart.mdx: resources_servers/mcqa/configs/mcqa.yaml
            (["env", "start", "--resources-server", "mcqa"], "resources_servers/mcqa/configs/mcqa.yaml"),
            # model-server/vllm.mdx: resources_servers/example_multi_step/configs/example_multi_step.yaml
            (
                ["env", "start", "--resources-server", "example_multi_step"],
                "resources_servers/example_multi_step/configs/example_multi_step.yaml",
            ),
            # README.md / quickstart.mdx: responses_api_models/openai_model/configs/openai_model.yaml
            (
                ["env", "start", "--model-type", "openai_model"],
                "responses_api_models/openai_model/configs/openai_model.yaml",
            ),
            # model-server/vllm.mdx: responses_api_models/vllm_model/configs/vllm_model.yaml
            (
                ["env", "start", "--model-type", "vllm_model"],
                "responses_api_models/vllm_model/configs/vllm_model.yaml",
            ),
        ],
    )
    def test_name_resolves_to_config_path(self, monkeypatch: MonkeyPatch, argv, expected_config) -> None:
        _, overrides = _dispatch_for(monkeypatch, argv)
        assert overrides == [f"+config_paths=[{WORKING_DIR / (expected_config)}]"]

    def test_quickstart_resource_server_plus_model(self, monkeypatch: MonkeyPatch) -> None:
        # README.md / quickstart.mdx:
        #   ng_run "+config_paths=[resources_servers/mcqa/configs/mcqa.yaml,
        #                          responses_api_models/openai_model/configs/openai_model.yaml]"
        target, overrides = _dispatch_for(
            monkeypatch, ["env", "start", "--resources-server", "mcqa", "--model-type", "openai_model"]
        )
        assert target == "nemo_gym.cli.env:run"
        paths, others = _split_overrides(overrides)
        assert paths == {
            str(WORKING_DIR / "resources_servers/mcqa/configs/mcqa.yaml"),
            str(WORKING_DIR / "responses_api_models/openai_model/configs/openai_model.yaml"),
        }
        assert others == set()

    def test_gpqa_benchmark_plus_model(self, monkeypatch: MonkeyPatch) -> None:
        # benchmarks/gpqa/README.md:
        #   ng_run "+config_paths=[benchmarks/gpqa/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
        _, overrides = _dispatch_for(monkeypatch, ["eval", "run", "--benchmark", "gpqa", "--model-type", "vllm_model"])
        paths, others = _split_overrides(overrides)
        assert paths == {
            str(WORKING_DIR / "benchmarks/gpqa/config.yaml"),
            str(WORKING_DIR / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
        }
        assert others == set()

    def test_cli_reference_e2e_rollout_example(self, monkeypatch: MonkeyPatch) -> None:
        # fern .../reference/cli-commands.mdx ng_e2e_collect_rollouts example:
        #   config_paths="responses_api_models/openai_model/configs/openai_model.yaml,
        #                 resources_servers/math_with_judge/configs/math_with_judge.yaml"
        #   ng_e2e_collect_rollouts "+config_paths=[$config_paths]"
        #       ++output_jsonl_fpath=results/test_e2e_rollout_collection/aime24.jsonl ++split=validation
        target, overrides = _dispatch_for(
            monkeypatch,
            [
                "eval",
                "run",
                "--resources-server",
                "math_with_judge",
                "--model-type",
                "openai_model",
                "--output",
                "results/test_e2e_rollout_collection/aime24.jsonl",
                "--split",
                "validation",
            ],
        )
        assert target == "nemo_gym.cli.eval:e2e_rollout_collection"
        paths, others = _split_overrides(overrides)
        assert paths == {
            str(WORKING_DIR / "resources_servers/math_with_judge/configs/math_with_judge.yaml"),
            str(WORKING_DIR / "responses_api_models/openai_model/configs/openai_model.yaml"),
        }
        assert others == {
            "+output_jsonl_fpath=results/test_e2e_rollout_collection/aime24.jsonl",
            "+split=validation",
        }

    def test_cli_reference_prepare_data_example(self, monkeypatch: MonkeyPatch) -> None:
        # fern .../reference/cli-commands.mdx ng_prepare_data example:
        #   config_paths includes resources_servers/example_multi_step/configs/example_multi_step.yaml
        #   ng_prepare_data "+config_paths=[...]" +output_dirpath=data/example_multi_step +mode=example_validation
        target, overrides = _dispatch_for(
            monkeypatch,
            [
                "dataset",
                "collate",
                "--resources-server",
                "example_multi_step",
                "--mode",
                "example_validation",
                "--output-dir",
                "data/example_multi_step",
            ],
        )
        assert target == "nemo_gym.cli.dataset:prepare_data"
        paths, others = _split_overrides(overrides)
        assert paths == {str(WORKING_DIR / "resources_servers/example_multi_step/configs/example_multi_step.yaml")}
        assert others == {
            "+mode=example_validation",
            "+output_dirpath=data/example_multi_step",
        }

    def test_collate_model_type_selector(self, monkeypatch: MonkeyPatch) -> None:
        # `dataset collate` accepts --model-type so model-dependent data workflows (e.g. train_preparation that
        # generates with a model) can select the model by name instead of an internal config path.
        target, overrides = _dispatch_for(
            monkeypatch,
            [
                "dataset",
                "collate",
                "--resources-server",
                "format_verification/freeform_formatting",
                "--model-type",
                "vllm_model",
                "--mode",
                "train_preparation",
            ],
        )
        assert target == "nemo_gym.cli.dataset:prepare_data"
        paths, others = _split_overrides(overrides)
        assert paths == {
            str(WORKING_DIR / "resources_servers/format_verification/configs/freeform_formatting.yaml"),
            str(WORKING_DIR / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
        }
        assert others == {"+mode=train_preparation"}

    def test_resource_server_flavor_syntax(self, monkeypatch: MonkeyPatch) -> None:
        # `<server>/<flavor>` picks a named config inside the server's configs/ dir; math_with_judge ships several
        # flavoured configs (see reference/faq.mdx, which pairs a math_with_judge dataset flavour for profiling).
        _, overrides = _dispatch_for(monkeypatch, ["eval", "run", "--resources-server", "math_with_judge/dapo17k"])
        assert overrides == [
            f"+config_paths=[{WORKING_DIR / 'resources_servers/math_with_judge/configs/dapo17k.yaml'}]"
        ]

    def test_benchmark_flavor_syntax(self, monkeypatch: MonkeyPatch) -> None:
        # Benchmarks are flavoured too: flavor is a sibling `<flavor>.yaml` (no configs/ dir), default `config`.
        # e.g. benchmarks/finance_sec_search ships config_web_search.yaml alongside the default config.yaml.
        _, overrides = _dispatch_for(
            monkeypatch, ["eval", "prepare", "--benchmark", "finance_sec_search/config_web_search"]
        )
        assert overrides == [f"+config_paths=[{WORKING_DIR / 'benchmarks/finance_sec_search/config_web_search.yaml'}]"]

    def test_environment_selector_resolves_to_config(self, monkeypatch: MonkeyPatch) -> None:
        # Environments mirror benchmarks: `--environment <name>` loads `environments/<name>/config.yaml`.
        # Used by `gym env start` / `gym eval run` for environments/ (see environments/*/README.md).
        _, overrides = _dispatch_for(monkeypatch, ["env", "start", "--environment", "circle_count"])
        assert overrides == [f"+config_paths=[{WORKING_DIR / 'environments/circle_count/config.yaml'}]"]

    def test_selectors_merge_into_single_config_paths(self, monkeypatch: MonkeyPatch) -> None:
        # --config and multiple asset selectors all feed one +config_paths list (Hydra rejects duplicates).
        # _split_overrides asserts they coalesce into a single token.
        _, overrides = _dispatch_for(
            monkeypatch,
            ["eval", "run", "--config", "extra.yaml", "--resources-server", "mcqa", "--model-type", "openai_model"],
        )
        paths, others = _split_overrides(overrides)
        assert paths == {
            "extra.yaml",  # raw --config value passes through verbatim; only name selectors get rooted
            str(WORKING_DIR / "resources_servers/mcqa/configs/mcqa.yaml"),
            str(WORKING_DIR / "responses_api_models/openai_model/configs/openai_model.yaml"),
        }
        assert others == set()

    def test_unknown_benchmark_errors_with_available_hint(self, monkeypatch: MonkeyPatch, capsys) -> None:
        monkeypatch.setattr(cli_main, "dispatch", lambda target, overrides: None)
        monkeypatch.setattr(sys, "argv", ["gym", "eval", "prepare", "--benchmark", "does_not_exist"])
        with pytest.raises(SystemExit):
            main()
        err = capsys.readouterr().err
        assert "benchmarks/does_not_exist/config.yaml" in err
        assert "does not exist" in err
        assert "benchmarks/" in err

    def test_unknown_flavor_error_points_at_configs_dir(self, monkeypatch: MonkeyPatch, capsys) -> None:
        # For a known server with an unknown flavor, the hint should point at that server's configs/ dir.
        monkeypatch.setattr(cli_main, "dispatch", lambda target, overrides: None)
        monkeypatch.setattr(sys, "argv", ["gym", "env", "start", "--resources-server", "mcqa/nope"])
        with pytest.raises(SystemExit):
            main()
        err = capsys.readouterr().err
        assert "resources_servers/mcqa/configs/nope.yaml" in err
        assert "resources_servers/mcqa/configs/" in err


class TestDidYouMean:
    """difflib-backed "did you mean?" hints for mistyped commands, flags, and component names (proposal UX 4)."""

    def test_helper_suggests_close_match(self) -> None:
        assert cli_main._did_you_mean("evl", ["list", "eval", "env"]) == " Did you mean `eval`?"

    def test_helper_silent_when_nothing_close(self) -> None:
        assert cli_main._did_you_mean("zzzzzz", ["list", "eval", "env"]) == ""

    def _run_expecting_exit(self, monkeypatch: MonkeyPatch, capsys, argv: list[str]) -> str:
        monkeypatch.setattr(cli_main, "dispatch", lambda target, overrides: None)
        monkeypatch.setattr(sys, "argv", ["gym", *argv])
        with pytest.raises(SystemExit):
            main()
        return capsys.readouterr().err

    def test_mistyped_group(self, monkeypatch: MonkeyPatch, capsys) -> None:
        err = self._run_expecting_exit(monkeypatch, capsys, ["evl"])
        assert "invalid choice: 'evl'" in err
        assert "Did you mean `eval`?" in err

    def test_mistyped_action(self, monkeypatch: MonkeyPatch, capsys) -> None:
        err = self._run_expecting_exit(monkeypatch, capsys, ["eval", "rnu"])
        assert "Did you mean `run`?" in err

    def test_mistyped_flag_choice(self, monkeypatch: MonkeyPatch, capsys) -> None:
        # --storage validates choices, so the parser-level hint kicks in.
        err = self._run_expecting_exit(monkeypatch, capsys, ["dataset", "upload", "--storage", "gitlb"])
        assert "Did you mean `gitlab`?" in err

    def test_misspelled_flag(self, monkeypatch: MonkeyPatch, capsys) -> None:
        err = self._run_expecting_exit(monkeypatch, capsys, ["eval", "run", "--benchmrk", "aalcr"])
        assert "unrecognized arguments: --benchmrk" in err
        assert "Did you mean `--benchmark`?" in err

    def test_misspelled_component_name(self, monkeypatch: MonkeyPatch, capsys) -> None:
        err = self._run_expecting_exit(monkeypatch, capsys, ["eval", "prepare", "--benchmark", "aalcrr"])
        assert "Did you mean `aalcr`?" in err

    def test_misspelled_component_flavor(self, monkeypatch: MonkeyPatch, capsys) -> None:
        err = self._run_expecting_exit(
            monkeypatch, capsys, ["eval", "run", "--resources-server", "math_with_judge/dapo17"]
        )
        assert "Did you mean `dapo17k`?" in err


class TestSearchDir:
    """--search-dir registers extra roots that the name->config selectors also search (REQ 5)."""

    def _make_user_benchmark(self, tmp_path, name: str = "mybench") -> None:
        bench_dir = tmp_path / "benchmarks" / name
        bench_dir.mkdir(parents=True)
        (bench_dir / "config.yaml").write_text("{}\n")

    def test_resolves_component_from_user_dir(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        self._make_user_benchmark(tmp_path)
        _, overrides = _dispatch_for(
            monkeypatch, ["eval", "prepare", "--benchmark", "mybench", "--search-dir", str(tmp_path)]
        )
        # User-dir matches are returned rooted so Hydra can resolve them.
        assert overrides == [f"+config_paths=[{tmp_path / 'benchmarks' / 'mybench' / 'config.yaml'}]"]

    def test_builtin_resolves_when_search_dir_lacks_it(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        # A built-in still resolves under WORKING_DIR when a --search-dir is provided that does not shadow it.
        _, overrides = _dispatch_for(
            monkeypatch, ["eval", "prepare", "--benchmark", "gsm8k", "--search-dir", str(tmp_path)]
        )
        assert overrides == [f"+config_paths=[{WORKING_DIR / 'benchmarks/gsm8k/config.yaml'}]"]

    def test_ambiguous_match_errors(self, monkeypatch: MonkeyPatch, tmp_path, capsys) -> None:
        # A built-in name also present in a --search-dir is ambiguous; the user must disambiguate with --config.
        self._make_user_benchmark(tmp_path, name="gsm8k")  # gsm8k also exists under WORKING_DIR
        monkeypatch.setattr(cli_main, "dispatch", lambda target, overrides: None)
        monkeypatch.setattr(
            sys, "argv", ["gym", "eval", "prepare", "--benchmark", "gsm8k", "--search-dir", str(tmp_path)]
        )
        with pytest.raises(SystemExit):
            main()
        err = capsys.readouterr().err
        assert "ambiguous" in err
        assert str(WORKING_DIR / "benchmarks" / "gsm8k" / "config.yaml") in err
        assert str(tmp_path / "benchmarks" / "gsm8k" / "config.yaml") in err

    def test_search_dir_alone_emits_nothing(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        # --search-dir is consumed by the selectors; on its own it is not a Hydra override.
        _, overrides = _dispatch_for(monkeypatch, ["eval", "prepare", "--search-dir", str(tmp_path)])
        assert overrides == []

    def test_did_you_mean_spans_user_dir(self, monkeypatch: MonkeyPatch, tmp_path, capsys) -> None:
        self._make_user_benchmark(tmp_path)
        monkeypatch.setattr(cli_main, "dispatch", lambda target, overrides: None)
        monkeypatch.setattr(
            sys, "argv", ["gym", "eval", "prepare", "--benchmark", "mybnch", "--search-dir", str(tmp_path)]
        )
        with pytest.raises(SystemExit):
            main()
        assert "Did you mean `mybench`?" in capsys.readouterr().err


class TestInstallRootResolution:
    """Built-in assets resolve from the Gym install root regardless of cwd (REQ C2/C5).

    In a wheel install PARENT_DIR is site-packages (where the asset trees are installed) while the
    user runs from an unrelated project dir, so WORKING_DIR/cwd is not the install root. The selector
    must still find built-ins under PARENT_DIR. Editable-from-repo keeps working because PARENT_DIR,
    WORKING_DIR, and cwd all coincide and dedupe to a single root.
    """

    def _make_resources_server(self, root, name: str = "foo") -> None:
        config_dir = root / "resources_servers" / name / "configs"
        config_dir.mkdir(parents=True)
        (config_dir / f"{name}.yaml").write_text("{}\n")

    def test_builtin_resolves_from_install_root_when_cwd_differs(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        install_root = tmp_path / "site-packages"
        user_cwd = tmp_path / "my-project"
        install_root.mkdir()
        user_cwd.mkdir()
        self._make_resources_server(install_root)  # built-in only under the install root
        monkeypatch.setattr(cli_main, "PARENT_DIR", install_root)
        monkeypatch.setattr(cli_main, "WORKING_DIR", user_cwd)
        monkeypatch.chdir(user_cwd)

        resolved = cli_main._asset_config_path("resources-server", "foo")
        assert resolved == str(install_root / "resources_servers" / "foo" / "configs" / "foo.yaml")

    def test_user_cwd_asset_resolves_when_not_builtin(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        install_root = tmp_path / "site-packages"
        user_cwd = tmp_path / "my-project"
        install_root.mkdir()
        user_cwd.mkdir()
        self._make_resources_server(user_cwd, name="myenv")  # exists only in the user's project
        monkeypatch.setattr(cli_main, "PARENT_DIR", install_root)
        monkeypatch.setattr(cli_main, "WORKING_DIR", user_cwd)
        monkeypatch.chdir(user_cwd)

        resolved = cli_main._asset_config_path("resources-server", "myenv")
        assert resolved == str(user_cwd / "resources_servers" / "myenv" / "configs" / "myenv.yaml")

    def test_same_name_in_install_root_and_cwd_is_ambiguous(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        # A user asset shadowing a built-in of the same name is ambiguous; they must disambiguate with --config.
        install_root = tmp_path / "site-packages"
        user_cwd = tmp_path / "my-project"
        install_root.mkdir()
        user_cwd.mkdir()
        self._make_resources_server(install_root)
        self._make_resources_server(user_cwd)
        monkeypatch.setattr(cli_main, "PARENT_DIR", install_root)
        monkeypatch.setattr(cli_main, "WORKING_DIR", user_cwd)
        monkeypatch.chdir(user_cwd)

        with pytest.raises(ValueError, match="ambiguous"):
            cli_main._asset_config_path("resources-server", "foo")

    def test_editable_layout_single_root_not_self_ambiguous(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        # Editable install: PARENT_DIR == WORKING_DIR == cwd. The same file found via all three roots
        # must dedupe to one match, not raise a spurious ambiguity error.
        repo_root = tmp_path / "Gym"
        repo_root.mkdir()
        self._make_resources_server(repo_root)
        monkeypatch.setattr(cli_main, "PARENT_DIR", repo_root)
        monkeypatch.setattr(cli_main, "WORKING_DIR", repo_root)
        monkeypatch.chdir(repo_root)

        resolved = cli_main._asset_config_path("resources-server", "foo")
        assert resolved == str(repo_root / "resources_servers" / "foo" / "configs" / "foo.yaml")


class TestListEnvironmentsRouting:
    def test_list_environments_dispatches(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(monkeypatch, ["list", "environments"])
        assert target == "nemo_gym.cli.env:list_environments"
        assert overrides == []

    def test_list_environments_json_dispatches(self, monkeypatch: MonkeyPatch) -> None:
        target, overrides = _dispatch_for(monkeypatch, ["list", "environments", "--json"])
        assert target == "nemo_gym.cli.env:list_environments"
        assert overrides == ["+json=true"]
