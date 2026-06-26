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
"""The CLI refactor moved these symbols into `nemo_gym.cli`; the old import paths must keep
working (with a `DeprecationWarning`) for the deprecation period. Remove once they are dropped."""

import importlib

import pytest


# (old_module, old_name, new_module, new_name) for each relocated symbol.
MOVED_SYMBOLS = [
    ("nemo_gym.benchmarks", "list_benchmarks", "nemo_gym.cli.eval", "list_benchmarks"),
    ("nemo_gym.benchmarks", "PrepareBenchmarkConfig", "nemo_gym.cli.eval", "PrepareBenchmarkConfig"),
    ("nemo_gym.benchmarks", "prepare_benchmark", "nemo_gym.cli.eval", "prepare_benchmark"),
    (
        "nemo_gym.dataset_orchestrator",
        "upload_jsonl_dataset_to_hf_cli",
        "nemo_gym.cli.dataset",
        "upload_jsonl_dataset_to_hf_cli",
    ),
    (
        "nemo_gym.dataset_orchestrator",
        "upload_jsonl_dataset_to_hf_and_delete_gitlab_cli",
        "nemo_gym.cli.dataset",
        "upload_jsonl_dataset_to_hf_and_delete_gitlab_cli",
    ),
    (
        "nemo_gym.dataset_orchestrator",
        "download_jsonl_dataset_from_hf_cli",
        "nemo_gym.cli.dataset",
        "download_jsonl_dataset_from_hf_cli",
    ),
    (
        "nemo_gym.dataset_orchestrator",
        "delete_jsonl_dataset_from_gitlab_cli",
        "nemo_gym.cli.dataset",
        "delete_jsonl_dataset_from_gitlab_cli",
    ),
    ("nemo_gym.gitlab_utils", "upload_jsonl_dataset_cli", "nemo_gym.cli.dataset", "upload_jsonl_dataset_cli"),
    ("nemo_gym.gitlab_utils", "download_jsonl_dataset_cli", "nemo_gym.cli.dataset", "download_jsonl_dataset_cli"),
    ("nemo_gym.prompt", "materialize_prompts_cli", "nemo_gym.cli.dataset", "materialize_prompts_cli"),
    ("nemo_gym.reward_profile", "reward_profile", "nemo_gym.cli.eval", "reward_profile"),
    ("nemo_gym.rollout_collection", "collect_rollouts", "nemo_gym.cli.eval", "collect_rollouts"),
    ("nemo_gym.rollout_collection", "aggregate_rollouts", "nemo_gym.cli.eval", "aggregate_rollouts"),
    ("nemo_gym.train_data_utils", "prepare_data", "nemo_gym.cli.dataset", "prepare_data"),
    ("nemo_gym.cli", "RunHelper", "nemo_gym.cli.env", "RunHelper"),
    ("nemo_gym.cli", "RunConfig", "nemo_gym.cli.env", "RunConfig"),
    ("nemo_gym.cli", "TestConfig", "nemo_gym.cli.env", "TestConfig"),
    ("nemo_gym.cli", "TestAllConfig", "nemo_gym.cli.env", "TestAllConfig"),
    ("nemo_gym.cli", "PipListConfig", "nemo_gym.cli.env", "PipListConfig"),
    ("nemo_gym.cli", "run", "nemo_gym.cli.env", "run"),
    ("nemo_gym.cli", "test", "nemo_gym.cli.env", "test"),
    ("nemo_gym.cli", "test_all", "nemo_gym.cli.env", "test_all"),
    ("nemo_gym.cli", "init_resources_server", "nemo_gym.cli.env", "init_resources_server"),
    ("nemo_gym.cli", "dump_config", "nemo_gym.cli.env", "dump_config"),
    ("nemo_gym.cli", "status", "nemo_gym.cli.env", "status"),
    ("nemo_gym.cli", "pip_list", "nemo_gym.cli.env", "pip_list"),
    ("nemo_gym.cli", "e2e_rollout_collection", "nemo_gym.cli.eval", "e2e_rollout_collection"),
    ("nemo_gym.cli", "dev_test", "nemo_gym.cli.dev", "dev_test"),
    ("nemo_gym.cli", "VersionConfig", "nemo_gym.cli.general", "VersionConfig"),
    ("nemo_gym.cli", "version", "nemo_gym.cli.general", "version"),
    ("nemo_gym.cli", "GlobalConfigDictParserConfig", "nemo_gym.global_config", "GlobalConfigDictParserConfig"),
    ("nemo_gym.cli_setup_command", "run_command", "nemo_gym.cli.setup_command", "run_command"),
    ("nemo_gym.cli_setup_command", "setup_env_command", "nemo_gym.cli.setup_command", "setup_env_command"),
]


@pytest.mark.parametrize("old_module, old_name, new_module, new_name", MOVED_SYMBOLS)
def test_deprecated_import_still_resolves_and_warns(old_module, old_name, new_module, new_name) -> None:
    old_mod = importlib.import_module(old_module)
    with pytest.warns(DeprecationWarning):
        resolved = getattr(old_mod, old_name)
    assert resolved is getattr(importlib.import_module(new_module), new_name)
