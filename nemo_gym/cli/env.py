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

import importlib.util
import json
import os
import re
import shlex
import sys
import traceback
from functools import cache
from glob import glob
from pathlib import Path
from shutil import rmtree
from subprocess import Popen
from tempfile import TemporaryDirectory
from time import time
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

import rich
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
from omegaconf.resolvers.oc import env as _oc_env
from pydantic import BaseModel, Field, ValidationError
from rich.markup import escape
from rich.table import Table
from tqdm.auto import tqdm

from nemo_gym import PARENT_DIR, component_search_roots
from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.cli.utils import (
    exit_cleanly_on_config_error,
    exit_unknown_component,
    fuzzy_matches,
    print_no_matches,
    print_rich_table,
    render_component_inspection,
)
from nemo_gym.config.config_types import BaseNeMoGymCLIConfig, ConfigError, EnvironmentServerConfigError
from nemo_gym.config.global_config import (
    COMPONENT_NAME_KEY_NAME,
    ENVIRONMENT_SERVER_TYPE_KEY_NAME,
    JSON_OUTPUT_KEY_NAME,
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
    QUERY_KEY_NAME,
    SERVER_TYPE_KEYS,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_global_config_dict,
)
from nemo_gym.environment.manifest import EnvironmentKind, IntegrationProfile
from nemo_gym.environment.onboarding import (
    EnvironmentOnboardingError,
    VerifierReport,
    prepare_verifier_run,
)
from nemo_gym.environment.publication import finalize_publication
from nemo_gym.environment.registry import (
    EnvironmentCatalogEntry,
    RegistryError,
    discover_environment_catalog,
    manifest_workloads_for_server,
    read_environment_details,
    resolve_catalog_entry,
)
from nemo_gym.environment.scaffold import scaffold_environment, scaffold_resources_server
from nemo_gym.environment.validation import (
    EnvironmentValidationReport,
    resolve_config_servers,
    validate_environment,
)
from nemo_gym.server_runtime.entrypoint_flags import entrypoint_server_class_names
from nemo_gym.server_runtime.server_status import StatusCommand
from nemo_gym.server_runtime.supervisor.server_supervisor import (
    RunHelper,
    _configured_servers,
    _resolve_server_dir,
)
from nemo_gym.server_runtime.supervisor.server_venv import get_venv_path, run_command, setup_env_command


class RunConfig(BaseNeMoGymCLIConfig):
    """
    Start NeMo Gym servers for agents, models, and resources.

    Examples:

    ```bash
    config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\\
    responses_api_models/openai_model/configs/openai_model.yaml"
    gym env start "+config_paths=[${config_paths}]"
    ```
    """

    entrypoint: str = Field(
        description="Entrypoint for this command. This must be a relative path with 2 parts. Should look something like `responses_api_agents/simple_agent`."
    )


class TestConfig(RunConfig):
    """
    Test a specific server module by running its pytest suite and optionally validating example data.

    Examples:

    ```bash
    gym env test +entrypoint=resources_servers/example_single_tool_call
    ```
    """

    should_validate_data: bool = Field(
        default=False,
        description="Whether or not to validate the example data (examples, metrics, rollouts, etc) for this server.",
    )

    _dir_path: Path  # initialized in model_post_init

    def model_post_init(self, context):  # pragma: no cover
        self._dir_path = Path(self.entrypoint)
        assert not self.dir_path.is_absolute()

        return super().model_post_init(context)

    @property
    def dir_path(self) -> Path:
        return self._dir_path

    @property
    def resolved_dir_path(self) -> Path:
        """Absolute server dir resolved against the cwd, then the Gym install root.

        Use this for filesystem access (reading data, running the suite); use ``dir_path`` (the
        relative entrypoint) for display and example commands shown to the user.
        """
        return _resolve_server_dir(self._dir_path)


@exit_cleanly_on_config_error
def prefetch(
    global_config_dict_parser_config: Optional[GlobalConfigDictParserConfig] = None,
):  # pragma: no cover
    """
    Pre-warm per-server venvs without starting any servers.

    Accepts the same config format as 'gym env start'. For each server in the config,
    creates its isolated venv and installs dependencies serially — similar to how
    RL's prefetch_venvs.py installs actor venvs at container build time.

    No Ray is initialised and no server processes are started. Intended for use in
    Dockerfile builds so venvs are ready at runtime with no network access needed.

    Configuration Parameter:
        config_paths (List[str]): Paths to YAML configuration files. Specify via Hydra: `+config_paths="[file1.yaml,file2.yaml]"`

    Examples:

    ```bash
    # Pre-warm venvs for a specific config
    config_paths="responses_api_models/local_vllm_model/configs/local_vllm_model.yaml,\\
    resources_servers/math_with_judge/configs/math_with_judge.yaml"
    gym env prefetch "+config_paths=[${config_paths}]"
    ```
    """
    global_config_dict = get_global_config_dict(global_config_dict_parser_config=global_config_dict_parser_config)
    GlobalConfigDictParser().raise_on_no_server_instances(global_config_dict)

    top_level_paths = [k for k in global_config_dict.keys() if k not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS]

    seen_dirs: set = set()
    for top_level_path in top_level_paths:
        server_config_dict = global_config_dict[top_level_path]
        if not isinstance(server_config_dict, DictConfig):
            continue

        first_key = list(server_config_dict)[0]
        server_config_dict = server_config_dict[first_key]
        if not isinstance(server_config_dict, DictConfig):
            continue
        second_key = list(server_config_dict)[0]
        server_config_dict = server_config_dict[second_key]
        if not isinstance(server_config_dict, DictConfig):
            continue

        if "entrypoint" not in server_config_dict:
            continue

        dir_path = _resolve_server_dir(Path(first_key, second_key))
        if dir_path in seen_dirs:
            print(f"Skipping duplicate: {dir_path}")
            continue
        seen_dirs.add(dir_path)
        print(f"Pre-warming venv for: {dir_path}")

        # Run only the setup (venv creation + dep install) — no server start.
        # Serial execution reduces uv cache size (same reasoning as dry_run in RunHelper.start).
        setup_cmd = setup_env_command(dir_path, global_config_dict, top_level_path)
        process = run_command(setup_cmd, dir_path, server_name=top_level_path)
        returncode = process.wait()
        if returncode != 0:
            print(f"ERROR: prefetch failed for {dir_path} (exit {returncode})")
            raise SystemExit(returncode)

    print("Prefetch complete.")


@exit_cleanly_on_config_error
def run(
    global_config_dict_parser_config: Optional[GlobalConfigDictParserConfig] = None,
):  # pragma: no cover
    """
    Start NeMo Gym servers for agents, models, and resources.

    This command reads configuration from YAML files specified via `+config_paths` and starts all configured servers.
    The configuration files should define server instances with their entrypoints and settings.

    Configuration Parameter:
        config_paths (List[str]): Paths to YAML configuration files. Specify via Hydra: `+config_paths="[file1.yaml,file2.yaml]"`

    Examples:

    ```bash
    # Start servers with specific configs
    config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\\
    responses_api_models/openai_model/configs/openai_model.yaml"
    gym env start "+config_paths=[${config_paths}]"
    ```
    """
    global_config_dict = get_global_config_dict(global_config_dict_parser_config=global_config_dict_parser_config)
    # Just here for help
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    rh = RunHelper()
    rh.start(global_config_dict_parser_config)
    rh.run_forever()


def _validate_data_single(test_config: TestConfig) -> None:  # pragma: no cover
    if not test_config.should_validate_data:
        return

    # We have special data checks for resources servers
    if test_config.dir_path.parts[0] != "resources_servers":
        return

    # A server that only manifest workloads use carries verifier fixtures instead of the legacy five-row rollout
    # artifacts (see manifest_workloads_for_server). Exercise every workload sharing it, offline and without requiring
    # its full dataset to be downloaded on CI workers.
    manifest_entries = manifest_workloads_for_server(
        test_config.dir_path.name, test_config.resolved_dir_path, discover_environment_catalog()
    )
    if manifest_entries:
        conflict_paths = glob(str(test_config.resolved_dir_path / "data/*conflict*"))
        assert not conflict_paths, f"Found conflicting paths: {conflict_paths}"
        for entry in manifest_entries:
            validation = validate_environment(entry.manifest_path, entry.config_path, validate_datasets=False)
            _run_manifest_verifier(entry, update_expected=False, validation=validation)
        print(f"The verifier fixtures for {test_config.dir_path} have been successfully validated!")
        return

    # Check that the required examples and example metrics are present. Read from the resolved dir
    # (built-ins live under the install root) while messages reference the relative entrypoint.
    example_fpath = test_config.resolved_dir_path / "data/example.jsonl"
    assert example_fpath.exists(), (
        f"A jsonl file containing 5 examples is required for the {test_config.dir_path} resources server. The file must be found at {example_fpath}. Usually this example data is just the first 5 examples of your train dataset."
    )
    with open(example_fpath) as f:
        count = sum(1 for _ in f)
    assert count == 5, f"Expected 5 examples at {example_fpath} but got {count}."

    server_type_name = test_config.dir_path.parts[-1]
    example_metrics_fpath = test_config.resolved_dir_path / "data/example_metrics.json"
    assert (
        example_metrics_fpath.exists()
    ), f"""You must run the example data validation for the example data found at {example_fpath}.
Your command should look something like the following (you should update this command with your actual server config path):
```bash
gym dataset collate "+config_paths=[{test_config._dir_path}/configs/{server_type_name}.yaml]" \\
    +output_dirpath=data/{server_type_name} \\
    +mode=example_validation
```
and your config must include an agent server config with an example dataset like:
```yaml
example_multi_step_simple_agent:
  responses_api_agents:
    simple_agent:
      ...
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/example_multi_step/data/example.jsonl
```

See `resources_servers/example_multi_step/configs/example_multi_step.yaml` for a full config example.
"""
    with open(example_metrics_fpath) as f:
        example_metrics = json.load(f)
    assert example_metrics["Number of examples"] == 5, (
        f"Expected 5 examples in the metrics at {example_metrics_fpath}, but got {example_metrics['Number of examples']}"
    )

    conflict_paths = glob(str(test_config.resolved_dir_path / "data/*conflict*"))
    conflict_paths_str = "\n- ".join([""] + conflict_paths)
    assert not conflict_paths, f"Found {len(conflict_paths)} conflicting paths: {conflict_paths_str}"

    example_rollouts_fpath = test_config.resolved_dir_path / "data/example_rollouts.jsonl"
    assert example_rollouts_fpath.exists(), f"""You must run the example data through your agent and provide the example rollouts at `{example_rollouts_fpath}`.

Your commands should look something like:
```bash
# Server spinup
example_multi_step_config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_multi_step/configs/example_multi_step.yaml"
gym env start "+config_paths=[${{example_multi_step_config_paths}}]"

# Collect the rollouts
gym eval run --no-serve +agent_name=example_multi_step_simple_agent \
    +input_jsonl_fpath=resources_servers/example_multi_step/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/example_multi_step/data/example_rollouts.jsonl \
    +limit=null

# View your rollouts
head -1 resources_servers/example_multi_step/data/example_rollouts.jsonl
```
"""
    with open(example_rollouts_fpath) as f:
        count = sum(1 for _ in f)
    assert count == 5, f"Expected 5 example rollouts in {example_rollouts_fpath}, but got {count}"

    print(f"The data for {test_config.dir_path} has been successfully validated!")


def _test_single(test_config: TestConfig, global_config_dict: DictConfig) -> Popen:  # pragma: no cover
    # Eventually we may want more sophisticated testing here, but this is sufficient for now.
    prefix = test_config.entrypoint.replace("/", "\\/")
    resolved_dir = test_config.resolved_dir_path
    pytest_args = ""
    junit_dir_value = os.environ.get("GYM_CI_JUNIT_DIR", "").strip()
    if junit_dir_value:
        junit_dir = Path(junit_dir_value).expanduser().resolve()
        junit_dir.mkdir(parents=True, exist_ok=True)
        entrypoint = test_config.entrypoint.replace("\\", "/").strip("/")
        report_name = f"{entrypoint.replace('/', '__')}.xml"
        junit_path = junit_dir / report_name
        junit_prefix = entrypoint.replace("/", ".")
        pytest_args = f" --junitxml={shlex.quote(str(junit_path))} --junit-prefix={shlex.quote(junit_prefix)}"
    command = f"""{setup_env_command(resolved_dir, global_config_dict, prefix)} && pytest{pytest_args}"""
    # Generated server tests import `resources_servers.<name>...`, so the project root (the dir
    # holding the server-type dirs) must be on PYTHONPATH when running from outside a repo checkout.
    return run_command(command, resolved_dir, project_root=resolved_dir.parent.parent)


def _setup_single(test_config: TestConfig, global_config_dict: DictConfig) -> Popen:
    """Install one server's venv exactly as `_test_single` does, without running its tests."""
    prefix = test_config.entrypoint.replace("/", "\\/")
    resolved_dir = test_config.resolved_dir_path
    command = setup_env_command(resolved_dir, global_config_dict, prefix)
    return run_command(command, resolved_dir, project_root=resolved_dir.parent.parent)


def test():  # pragma: no cover
    global_config_dict = get_global_config_dict()
    test_config = TestConfig.model_validate(global_config_dict)

    proc = _test_single(test_config, global_config_dict)
    return_code = proc.wait()
    if return_code != 0:
        print(f"You can run detailed tests via `cd {test_config.entrypoint} && source .venv/bin/activate && pytest`.")
        exit(return_code)

    try:
        _validate_data_single(test_config)
    except AssertionError as e:
        print(f"""{e}
Data validation failed for {test_config.entrypoint}. You can rerun just the data validation like:
```bash
gym env test +entrypoint={test_config.entrypoint} +should_validate_data=true
```
""")
        exit(1)


def _display_list_of_paths(paths: List[Path]) -> str:  # pragma: no cover
    paths = list(map(str, paths))
    return "".join(f"\n- {p}" for p in paths)


def _format_pct(count: int, total: int) -> str:  # pragma: no cover
    return f"{count} / {total} ({100 * count / total:.2f}%)"


class TestAllConfig(BaseNeMoGymCLIConfig):
    """
    Run tests for all server modules in the project.

    Examples:

    ```bash
    gym env test --all
    ```
    """

    fail_on_total_and_test_mismatch: bool = Field(
        default=False,
        description="Fail if the number of server modules doesn't match the number with tests (default: False).",
    )
    delete_venvs_after_each_test: bool = Field(
        default=False,
        description="Delete each server venv after its tests have been run (default: False).",
    )
    num_shards: int = Field(
        default=1,
        ge=1,
        description="Total number of shards to split the server suite across (default: 1 = no sharding). "
        "Used to parallelize the suite across CI runners.",
    )
    shard_index: int = Field(
        default=0,
        ge=0,
        description="Which shard (0-based) this invocation runs; must be < num_shards (default: 0).",
    )
    setup_only: bool = Field(
        default=False,
        description="Only install each selected server's venv; run no tests and validate no data (default: False). "
        "Seeds `uv_cache_dir` so the server suite can later install the same venvs offline.",
    )


def _select_shard(dir_paths: List[Path], shard_index: int, num_shards: int) -> List[Path]:
    """Deterministically select this shard's subset of modules.

    Round-robin (stride) over a sorted list spreads heavy modules across shards more evenly than
    contiguous chunks, which balances wall-time when the suite is parallelized across CI runners.
    """
    if num_shards <= 1:
        return dir_paths
    assert 0 <= shard_index < num_shards, (
        f"shard_index ({shard_index}) must be in [0, num_shards) for num_shards={num_shards}"
    )
    return sorted(dir_paths, key=str)[shard_index::num_shards]


def _delete_server_venv(dir_path: Path, global_config_dict: DictConfig) -> None:
    venv_path = get_venv_path(dir_path, global_config_dict)
    print(f"Deleting {venv_path} since `delete_venvs_after_each_test=true`")
    rmtree(venv_path, ignore_errors=True)


def _setup_all(dir_paths: List[Path], test_all_config: TestAllConfig, global_config_dict: DictConfig) -> None:
    """Install every selected server's venv the way the suite would, then exit non-zero if any failed.

    The container build runs this to seed its uv cache, so `scripts/ci/server_tests.sh` can create the same
    per-server venvs offline. Every server is attempted before failing, so one run reports every gap.
    """
    setup_failed: List[Path] = []
    for dir_path in tqdm(dir_paths, desc="Setting up server venvs"):
        proc = _setup_single(TestConfig(entrypoint=str(dir_path)), global_config_dict)
        if proc.wait() != 0:
            setup_failed.append(dir_path)
        if test_all_config.delete_venvs_after_each_test:
            _delete_server_venv(_resolve_server_dir(dir_path), global_config_dict)

    set_up = len(dir_paths) - len(setup_failed)
    print(f"""Server venvs set up {_format_pct(set_up, len(dir_paths))}

Server venv setup failed {_format_pct(len(setup_failed), len(dir_paths))}:{_display_list_of_paths(setup_failed)}
""")
    if setup_failed:
        exit(1)


def test_all():  # pragma: no cover
    global_config_dict = get_global_config_dict()
    test_all_config = TestAllConfig.model_validate(global_config_dict)

    # Discover server modules across every component-search root: NEMO_GYM_EXTRA_ROOTS (plugins), the cwd
    # (a user's project), and the Gym install root (built-ins, under PARENT_DIR in editable and wheel
    # installs). Entrypoints are kept relative; earlier roots shadow later ones for same-named modules. This
    # lets `gym env test` discover and run built-in and plugin servers from any cwd, not only a repo checkout.
    seen_rel_paths: set[str] = set()
    candidate_dir_paths: List[str] = []
    for root in component_search_roots():
        for server_type_dir in SERVER_TYPE_KEYS:
            for module_path in sorted((root / server_type_dir).glob("*")):
                if (
                    "pycache" in module_path.name
                    or module_path.name.endswith((".egg-info", ".dist-info"))
                    or not module_path.is_dir()
                ):
                    continue
                rel_path = f"{server_type_dir}/{module_path.name}"
                if rel_path in seen_rel_paths:
                    continue
                seen_rel_paths.add(rel_path)
                candidate_dir_paths.append(rel_path)
    print(f"Found {len(candidate_dir_paths)} total modules:{_display_list_of_paths(candidate_dir_paths)}\n")
    dir_paths: List[Path] = list(map(Path, candidate_dir_paths))
    dir_paths = [p for p in dir_paths if (_resolve_server_dir(p) / "README.md").exists()]
    print(f"Found {len(dir_paths)} modules to test:{_display_list_of_paths(dir_paths)}\n")

    # Keep the full list for the total-vs-tested mismatch check below, then narrow to this shard.
    full_dir_paths = dir_paths
    dir_paths = _select_shard(dir_paths, test_all_config.shard_index, test_all_config.num_shards)
    if test_all_config.num_shards > 1:
        print(
            f"Shard {test_all_config.shard_index + 1}/{test_all_config.num_shards}: "
            f"testing {len(dir_paths)} of {len(full_dir_paths)} modules:{_display_list_of_paths(dir_paths)}\n"
        )

    if test_all_config.setup_only:
        _setup_all(dir_paths, test_all_config, global_config_dict)
        return

    tests_passed: List[Path] = []
    tests_failed: List[Path] = []
    tests_missing: List[Path] = []
    tests_unrecognized: List[Path] = []
    data_validation_failed: List[Path] = []
    times_taken: List[Tuple[float, Path]] = []
    for dir_path in tqdm(dir_paths, desc="Running tests"):
        start_time = time()

        test_config = TestConfig(
            entrypoint=str(dir_path),
            should_validate_data=True,  # Test all always validates data.
        )
        proc = _test_single(test_config, global_config_dict)
        return_code = proc.wait()

        match return_code:
            case 0:
                tests_passed.append(dir_path)
            case 1 | 2:
                tests_failed.append(dir_path)
            case 5:
                tests_missing.append(dir_path)
            case _:
                tests_unrecognized.append(dir_path)

        try:
            _validate_data_single(test_config)
        except AssertionError:
            data_validation_failed.append(dir_path)

        if test_all_config.delete_venvs_after_each_test:
            _delete_server_venv(_resolve_server_dir(dir_path), global_config_dict)

        times_taken.append((time() - start_time, dir_path))

    times_taken.sort(reverse=True)
    table = Table(title="Times taken per test (sorted from highest to lowest)")
    table.add_column("Server path")
    table.add_column("Time taken (s)")
    for time_taken, dir_path in times_taken:
        table.add_row(str(dir_path), f"{time_taken:.2f}")
    print_rich_table(table)

    print(f"""Found {len(candidate_dir_paths)} total modules:{_display_list_of_paths(candidate_dir_paths)}

Found {len(dir_paths)} modules to test:{_display_list_of_paths(dir_paths)}

Tests passed {_format_pct(len(tests_passed), len(dir_paths))}:{_display_list_of_paths(tests_passed)}

Tests failed {_format_pct(len(tests_failed), len(dir_paths))}:{_display_list_of_paths(tests_failed)}

Tests missing {_format_pct(len(tests_missing), len(dir_paths))}:{_display_list_of_paths(tests_missing)}

Tests that returned unrecognized exit codes {_format_pct(len(tests_unrecognized), len(dir_paths))}:{_display_list_of_paths(tests_unrecognized)}

Data validation failed {_format_pct(len(data_validation_failed), len(dir_paths))}:{_display_list_of_paths(data_validation_failed)}
""")

    if tests_failed or tests_missing or tests_unrecognized:
        print(f"""You can rerun just the server with failed, missing or unrecognized tests results like:
```bash
gym env test +entrypoint={(tests_failed + tests_missing + tests_unrecognized)[0]}
```
""")
    if data_validation_failed:
        print(f"""You can rerun just the server with failed data validation like:
```bash
gym env test +entrypoint={data_validation_failed[0]} +should_validate_data=true
```
""")

    if test_all_config.fail_on_total_and_test_mismatch:
        # Compare against the full (unsharded) module list — every module must be testable
        # regardless of how many shards we split the run into.
        extra_candidates = [p for p in candidate_dir_paths if Path(p) not in full_dir_paths]
        assert (
            len(candidate_dir_paths) == len(full_dir_paths)
        ), f"""Mismatch on the number of total modules found ({len(candidate_dir_paths)}) and the number of actual modules tested ({len(full_dir_paths)})!

Extra candidate paths:{_display_list_of_paths(extra_candidates)}"""

    if tests_missing or tests_failed or tests_unrecognized or data_validation_failed:
        exit(1)


class InitEnvironmentConfig(BaseNeMoGymCLIConfig):
    scaffold_kind: EnvironmentKind
    scaffold_name: str
    profile: IntegrationProfile = IntegrationProfile.CUSTOM_GYM_VERIFIER
    reuse_verifier: Optional[str] = None
    reward_range: Optional[Tuple[float, float]] = None
    higher_is_better: Optional[bool] = None


def _command_overrides() -> DictConfig:
    """Parse the already-translated local command flags without resolving Gym runtime config."""
    return OmegaConf.from_dotlist([token.lstrip("+") for token in sys.argv[1:] if "=" in token])


@exit_cleanly_on_config_error
def init_environment() -> None:
    """Create a manifest-backed environment or benchmark skeleton."""
    config = InitEnvironmentConfig.model_validate(_command_overrides())
    result = scaffold_environment(
        kind=config.scaffold_kind,
        name=config.scaffold_name,
        profile=config.profile,
        reuse_verifier=config.reuse_verifier,
        reward_range=config.reward_range,
        higher_is_better=config.higher_is_better,
    )

    if result.created:
        rich.print(f"[green]✓[/green] Created {result.asset_dir}")
        for path in result.created:
            print(f"  {path}")
    else:
        rich.print(f"[green]✓[/green] {result.asset_dir} is already up to date.")


def init_resources_server():  # pragma: no cover
    """
    Initialize a new resources server with template files and directory structure.

    Examples:

    ```bash
    gym env init --resources-server my_server
    ```
    """
    config_dict = get_global_config_dict()
    run_config = RunConfig.model_validate(config_dict)

    dirpath = Path(run_config.entrypoint).resolve()

    if dirpath.exists():
        print(f"Folder already exists: {dirpath}. Exiting init.")
        exit()

    checkout_root = PARENT_DIR if (PARENT_DIR / "pyproject.toml").is_file() else None
    scaffold_resources_server(directory=dirpath, checkout_root=checkout_root)


@cache
def _entrypoint_config_class(entrypoint_fpath: Path, module_name: str) -> type[BaseModel]:
    """The config class of the server an entrypoint runs, the one its launch validates its block against."""
    spec = importlib.util.spec_from_file_location(module_name, entrypoint_fpath)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib internals
        raise ImportError(f"no import spec for {entrypoint_fpath}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec, so annotations that pydantic resolves through sys.modules are found.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    config_classes = {
        server_class.model_fields["config"].annotation
        for class_name in entrypoint_server_class_names(entrypoint_fpath)
        if isinstance(server_class := getattr(module, class_name, None), type)
        and issubclass(server_class, BaseModel)
        and "config" in server_class.model_fields
    }
    if len(config_classes) != 1:
        raise LookupError("it does not call run_webserver() on exactly one server class with a config")
    return config_classes.pop()


def _check_environment_server_configs(global_config_dict: DictConfig) -> List[str]:
    """Validate each Environment Server block against its server's config class without starting it.

    `gym env start` applies that class only when the server launches, after the other servers are up. Environment
    Servers ship with Gym and need few packages of their own, so this process imports their entrypoints. A server
    whose entrypoint does not import here is left to its launch: the returned warnings name each one.

    Raises EnvironmentServerConfigError listing every invalid field, by its path in the merged config.
    """
    errors: List[str] = []
    unchecked: List[str] = []
    for server in _configured_servers(global_config_dict):
        if server.server_type != ENVIRONMENT_SERVER_TYPE_KEY_NAME:
            continue
        source = f"{server.server_type}/{server.name}/{server.entrypoint_fpath}"
        module_name = ".".join(
            ["nemo_gym_environment_server_entrypoint", server.name, *server.entrypoint_fpath.with_suffix("").parts]
        )
        try:
            config_class = _entrypoint_config_class(server.dir_path / server.entrypoint_fpath, module_name)
        except Exception as error:  # Its own venv may have what this one lacks.
            unchecked.append(
                f"Could not check Environment Server '{server.top_level_path}' before launch, so `gym env start` "
                f"checks its config when it launches. Loading {source} here failed: {type(error).__name__}: {error}"
            )
            continue
        server_config = OmegaConf.to_container(server.config, resolve=True) | {"name": server.top_level_path}
        try:
            config_class.model_validate(server_config)
        except ValidationError as error:
            block_path = f"{server.top_level_path}.{server.server_type}.{server.name}"
            errors.append(f"'{server.top_level_path}' ({source}) does not match {config_class.__name__}:")
            errors.extend(
                f"  {'.'.join([block_path, *map(str, detail['loc'])])}: {detail['msg']}" for detail in error.errors()
            )
    if errors:
        raise EnvironmentServerConfigError(
            "Environment Server config is invalid, so `gym env start` would fail when the server launches.\n"
            + "\n".join(errors)
        )
    return unchecked


class ServerConfigProblem(NamedTuple):
    """A server block value that its server's config class rejects, or a block that could not be checked."""

    path: str
    """Dotted path in the merged config: the rejected field, else the block (or the unknown server type)."""
    message: str
    """The config class's message, or why the block's config class could not be loaded."""
    config_class: Optional[str] = None
    """The name of the config class that rejected the value; None when the block could not be checked."""
    value: object = None
    """The rejected input, as the config class received it."""


def validate_server_configs(global_config_dict: DictConfig, server_types: Iterable[str]) -> List[ServerConfigProblem]:
    """Validate each server block of the given server types against its own server's config class, starting nothing.

    Each block is checked against the class its server validates it with at launch. This process imports the
    entrypoint to find that class. A block whose entrypoint does not import here, for example because the server's
    own venv has packages this one lacks, is one problem with ``config_class`` None, and its fields stay unchecked.
    A server type that Gym does not define is such a problem too, at the type's name, so a caller written for
    another Gym version still gets an answer. An empty list means every checked block is valid.
    """
    wanted = frozenset(server_types)
    problems = [
        ServerConfigProblem(name, f"Gym defines no server type '{name}'; it defines {', '.join(SERVER_TYPE_KEYS)}")
        for name in sorted(wanted.difference(SERVER_TYPE_KEYS))
    ]
    for server in _configured_servers(global_config_dict):
        if server.server_type not in wanted:
            continue
        block_path = f"{server.top_level_path}.{server.server_type}.{server.name}"
        module_name = ".".join(
            ["nemo_gym_server_config_check", server.server_type, server.name]
            + list(server.entrypoint_fpath.with_suffix("").parts)
        )
        try:
            config_class = _entrypoint_config_class(server.dir_path / server.entrypoint_fpath, module_name)
        except Exception as error:  # Its own venv may have what this one lacks.
            message = f"could not load {server.entrypoint_fpath}: {type(error).__name__}: {error}"
            problems.append(ServerConfigProblem(block_path, message))
            continue
        try:
            server_config = OmegaConf.to_container(server.config, resolve=True) | {"name": server.top_level_path}
        except OmegaConfBaseException as error:
            problems.append(ServerConfigProblem(block_path, f"could not resolve the block: {error}"))
            continue
        try:
            config_class.model_validate(server_config)
        except ValidationError as error:
            problems.extend(
                ServerConfigProblem(
                    ".".join([block_path, *map(str, detail["loc"])]),
                    detail["msg"],
                    config_class.__name__,
                    detail.get("input"),
                )
                for detail in error.errors()
            )
    return problems


@exit_cleanly_on_config_error
def dump_config():  # pragma: no cover
    """
    Display the resolved Hydra configuration for debugging purposes.

    Examples:

    ```bash
    gym env resolve "+config_paths=[<config1>,<config2>]"
    ```
    """
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            hide_secrets=True,
        ),
    )

    # Just here for help
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    print(OmegaConf.to_yaml(global_config_dict, resolve=True))

    # The config prints either way, since it is what you debug a failing config with.
    try:
        unchecked = _check_environment_server_configs(global_config_dict)
    except EnvironmentServerConfigError as error:
        unchecked = [str(error)]
    for warning in unchecked:
        rich.print(f"[yellow]Warning:[/yellow] {escape(warning)}", file=sys.stderr)


class ManifestCommandConfig(BaseNeMoGymCLIConfig):
    onboarding_name: Optional[str] = None
    catalog_kind: Optional[EnvironmentKind] = None
    manifest_path: Optional[Path] = None
    sync: bool = False
    update_expected: bool = False


_MANIFEST_VALIDATE_KEYS = frozenset({"onboarding_name", "catalog_kind", "manifest_path", "sync", "json", "verbose"})
_MANIFEST_TEST_KEYS = frozenset({"onboarding_name", "catalog_kind", "update_expected", "json", "verbose"})
_MANIFEST_PUBLISH_KEYS = frozenset({"onboarding_name", "catalog_kind", "json", "verbose"})


def _reject_manifest_command_extras(command_dict: DictConfig, allowed: frozenset[str]) -> None:
    extras = sorted(set(command_dict) - allowed)
    if extras:
        raise ConfigError("Manifest-backed commands do not accept runtime config overrides: " + ", ".join(extras))


def _manifest_entry(config: ManifestCommandConfig) -> Optional[EnvironmentCatalogEntry]:
    if config.onboarding_name is not None and config.manifest_path is not None:
        raise ConfigError("select a catalog name or --manifest, not both")
    if config.manifest_path is not None and config.catalog_kind is not None:
        raise ConfigError("--kind is only valid when selecting a catalog name")
    if config.onboarding_name is None:
        return None
    return resolve_catalog_entry(config.onboarding_name, config.catalog_kind)


def _print_validation_report(report: EnvironmentValidationReport, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report.to_dict()))
        return
    rich.print(
        f"[green]✓[/green] Manifest: {report.name} {report.version} "
        f"kind={report.kind} profile={report.declared_profile} inferred={report.inferred_profile}"
    )
    print(f"    Profile evidence: {report.profile_evidence}")
    rich.print("[green]✓[/green] Components:")
    for component in report.components:
        print(f"    {component.role}: {component.name} -> {component.implementation} [{component.boundary}]")
    rich.print("[green]✓[/green] Datasets:")
    for dataset in report.datasets:
        rich.print(f"    {dataset.name}: {dataset.rows} rows ({dataset.type})")
    if report.synchronized_fields:
        rich.print(f"[green]✓[/green] Synchronized: {', '.join(report.synchronized_fields)}")
    for warning in report.warnings:
        rich.print(f"[yellow]Warning:[/yellow] {warning}", file=sys.stderr)


# Stands in for a value only the job supplies, under --allow-job-time-values.
JOB_TIME_VALUE = "gym-job-time-value"
# Each round of composition names at most three unset values, so this covers 48 of them.
_JOB_TIME_VALUE_ROUNDS = 16


_NO_DEFAULT = object()


def _env_or_job_time_value(key: str, default: object = _NO_DEFAULT) -> Optional[str]:
    """``oc.env`` that reads an unset variable without a default as a job-time value."""
    if default is _NO_DEFAULT:
        return os.environ.get(key, JOB_TIME_VALUE)
    return _oc_env(key, default)


def _compose_for_validation(*, allow_job_time_values: bool, supplied: List[str]) -> DictConfig:
    """Compose the config as `gym eval run` does from the same run args; the job adds the policy model itself."""
    for _ in range(_JOB_TIME_VALUE_ROUNDS):
        try:
            try:
                return get_global_config_dict()
            except ConfigError:
                return get_global_config_dict(
                    global_config_dict_parser_config=GlobalConfigDictParserConfig(
                        initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                    ),
                )
        except ConfigError as error:
            missing = [key for key in re.findall(r"\+\+([A-Za-z0-9_.]+)=<value>", str(error)) if key not in supplied]
            if not allow_job_time_values or not missing:
                raise
            supplied.extend(missing)
            sys.argv.extend(f"++{key}={JOB_TIME_VALUE}" for key in missing)
    raise ConfigError(f"composition still needs values after {len(supplied)} job-time stand-ins")


def _composition_failure(error: BaseException) -> Dict[str, str]:
    """Why the config did not compose, and the key whose value did not resolve when the error names one.

    A failing resolver (for example ``oc.decode`` given an int) is reported at its key with the resolver's own
    message. Gym's own explanation, such as the override that supplies a missing key, is kept as it reads.
    """
    resolution = error if isinstance(error, OmegaConfBaseException) else error.__cause__
    full_key = getattr(resolution, "full_key", None)
    if not isinstance(resolution, OmegaConfBaseException) or not isinstance(full_key, str) or not full_key:
        return {"why": f"{type(error).__name__}: {error}"}
    # OmegaConf appends the key and the node type to its own message; the path field carries the key.
    message = resolution.msg.split("\n    full_key:", 1)[0]
    why = message if str(error) == str(resolution) else f"{type(error).__name__}: {error}"
    return {"path": full_key, "why": why}


def _validate_server_types(command_dict: DictConfig) -> int:
    """`gym env validate --server-types`: print one finding per line; return 0, 1, 2, or 3 for a crash."""
    # Importing entrypoints must write nothing into the checkout.
    sys.dont_write_bytecode = True
    allow_job_time_values = bool(command_dict.get("allow_job_time_values", False))
    marks = [str(mark) for mark in command_dict.get("ignore_input_containing") or []]
    marks += [JOB_TIME_VALUE] if allow_job_time_values else []
    json_output = bool(command_dict.get(JSON_OUTPUT_KEY_NAME, False))

    def emit(finding: dict) -> None:
        if json_output:
            print(json.dumps(finding), flush=True)
        else:
            print(" ".join(f"{key}={value}" for key, value in finding.items()), flush=True)

    if allow_job_time_values:
        OmegaConf.register_new_resolver("oc.env", _env_or_job_time_value, replace=True, use_cache=False)
    try:
        supplied: List[str] = []
        # A block that fails Gym's generic server check is reported field by field below, not as a failed composition.
        sys.argv.append("++error_on_almost_servers=false")
        try:
            global_config_dict = _compose_for_validation(
                allow_job_time_values=allow_job_time_values, supplied=supplied
            )
        except (ConfigError, OmegaConfBaseException) as error:
            emit({"kind": "uncomposed", **_composition_failure(error), "supplied": supplied})
            return 2
        invalid = 0
        server_types = command_dict.server_types
        server_types = [server_types] if isinstance(server_types, str) else list(server_types)
        for problem in validate_server_configs(global_config_dict, server_types):
            if problem.config_class is None:
                emit({"kind": "unchecked", "path": problem.path, "why": problem.message})
            elif not any(mark in repr(problem.value) for mark in marks):
                invalid += 1
                emit(
                    {
                        "kind": "invalid",
                        "path": problem.path,
                        "class": problem.config_class,
                        "message": problem.message,
                        "input": repr(problem.value)[:120],
                    }
                )
        return 1 if invalid else 0
    except Exception:  # A crash must not read as a finding: its exit code is neither 1 nor 2.
        traceback.print_exc()
        return 3
    finally:
        if allow_job_time_values:
            OmegaConf.register_new_resolver("oc.env", _oc_env, replace=True, use_cache=False)


@exit_cleanly_on_config_error
def validate() -> None:
    """Validate a manifest-backed workload or a legacy Gym config without starting services."""
    command_dict = _command_overrides()
    if command_dict.get("server_types") is not None:
        sys.exit(_validate_server_types(command_dict))
    command_config = ManifestCommandConfig.model_validate(command_dict)
    entry = _manifest_entry(command_config)
    if command_config.manifest_path is not None or entry is not None:
        _reject_manifest_command_extras(command_dict, _MANIFEST_VALIDATE_KEYS)
        manifest_path = command_config.manifest_path if entry is None else entry.manifest_path
        if manifest_path is None:
            raise ConfigError(
                f"'{entry.name}' has no manifest; validate its config with --environment or --benchmark."
            )
        report = validate_environment(
            manifest_path,
            entry.config_path if entry is not None else None,
            sync=command_config.sync,
        )
        _print_validation_report(report, json_output=command_dict.get(JSON_OUTPUT_KEY_NAME, False))
        return
    unsupported = [
        key for key in ("catalog_kind", "sync", "update_expected", "json") if command_dict.get(key) is not None
    ]
    if unsupported:
        raise ConfigError("The following options require a workload name or --manifest: " + ", ".join(unsupported))

    # Resolve what a run resolves. Falling back to the stand-ins keeps a config that needs no model
    # checkable, but only an error the stand-ins account for is downgraded: any other still propagates.
    try:
        global_config_dict = get_global_config_dict()
        incomplete_config_error = None
    except ConfigError as error:
        if command_dict.get("strict", False):
            raise
        incomplete_config_error = error
        global_config_dict = get_global_config_dict(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
            ),
        )

    BaseNeMoGymCLIConfig.model_validate(global_config_dict)
    for warning in _check_environment_server_configs(global_config_dict):
        rich.print(f"[yellow]Warning:[/yellow] {escape(warning)}", file=sys.stderr)
    if incomplete_config_error is not None:
        rich.print(
            "[yellow]Warning:[/yellow] this config does not define model server configuration "
            "and will not start as-is. Re-run with [bold]--strict[/bold] to fail on this "
            f"instead of warning.\n{incomplete_config_error}",
            file=sys.stderr,
        )
    rich.print("[green]✓[/green] Config is valid.")


@exit_cleanly_on_config_error
def test_environment_manifest() -> None:
    """Exercise a manifest-backed workload's verifier fixture without starting services."""
    command_dict = _command_overrides()
    command_config = ManifestCommandConfig.model_validate(command_dict)
    entry = _manifest_entry(command_config)
    if entry is None:
        raise ConfigError("gym env test requires a catalog name or --resources-server")
    _reject_manifest_command_extras(command_dict, _MANIFEST_TEST_KEYS)
    report = _run_manifest_verifier(entry, update_expected=command_config.update_expected)
    if command_dict.get(JSON_OUTPUT_KEY_NAME, False):
        print(json.dumps(report.to_dict()))
        return
    rich.print(
        f"[green]✓[/green] Verifier: {report.name} ({report.resources_server}) {len(report.cases)} cases passed."
    )


@exit_cleanly_on_config_error
def publish_environment_manifest() -> None:
    """Run local publication checks and confirm a workload is cataloged."""
    command_dict = _command_overrides()
    command_config = ManifestCommandConfig.model_validate(command_dict)
    entry = _manifest_entry(command_config)
    if entry is None:
        raise ConfigError("gym env publish requires a catalog name")
    _reject_manifest_command_extras(command_dict, _MANIFEST_PUBLISH_KEYS)
    if entry.manifest_path is None:
        raise ConfigError(f"'{entry.name}' has no manifest and cannot be published.")

    validation = validate_environment(entry.manifest_path, entry.config_path)
    verifier = _run_manifest_verifier(entry, update_expected=False, validation=validation)
    report = finalize_publication(entry, validation, verifier)
    if command_dict.get(JSON_OUTPUT_KEY_NAME, False):
        print(json.dumps(report.to_dict()))
        return
    annotation = f"catalog status={report.status} " if report.status else ""
    rich.print(
        f"[green]✓[/green] Publication checks passed for {report.kind} {report.name} {report.version}; "
        f"{annotation}({report.verifier_cases} verifier cases)."
    )


def _run_manifest_verifier(
    entry: EnvironmentCatalogEntry,
    *,
    update_expected: bool,
    validation: EnvironmentValidationReport | None = None,
) -> VerifierReport:
    spec = prepare_verifier_run(entry, validation)
    setup_config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    server_dir = Path(spec.server_dir)
    prefix = f"resources_servers/{spec.resources_server}".replace("/", "\\/")

    with TemporaryDirectory(prefix="nemo-gym-verifier-") as temporary_dir:
        request_path = Path(temporary_dir) / "request.json"
        result_path = Path(temporary_dir) / "result.json"
        request_path.write_text(
            json.dumps({"spec": spec.to_dict(), "update_expected": update_expected}),
            encoding="utf-8",
        )
        python = get_venv_path(server_dir, setup_config) / "bin" / "python"
        runner = (
            f"{shlex.quote(str(python))} -m nemo_gym.environment._verifier_runner "
            f"--request {shlex.quote(str(request_path))} --result {shlex.quote(str(result_path))}"
        )
        command = f"{setup_env_command(server_dir, setup_config, prefix)} && {runner}"
        process = run_command(
            command,
            server_dir,
            server_name=spec.resources_server,
            project_root=PARENT_DIR if (PARENT_DIR / "pyproject.toml").is_file() else None,
            global_config_dict=setup_config,
            stdout_target=sys.stderr,
            stderr_target=sys.stderr,
        )
        return_code = process.wait()
        if not result_path.is_file():
            raise EnvironmentOnboardingError(
                f"Verifier runner exited with code {return_code} without producing a result."
            )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise EnvironmentOnboardingError(f"Could not read verifier result: {error}") from error
        if not isinstance(result, dict) or not result.get("ok"):
            message = result.get("error") if isinstance(result, dict) else None
            raise EnvironmentOnboardingError(message or f"Verifier runner failed with code {return_code}.")
        if return_code != 0:
            raise EnvironmentOnboardingError(f"Verifier runner exited with code {return_code}.")
        try:
            return VerifierReport.from_dict(result["report"])
        except (KeyError, TypeError, ValueError) as error:
            raise EnvironmentOnboardingError(f"Verifier runner returned an invalid report: {error}") from error


def _inspect_environment(
    name: str,
    entries: Tuple[EnvironmentCatalogEntry, ...],
    global_config_dict: DictConfig,
) -> None:
    """Render one entry from the unified environment catalog."""
    kind = global_config_dict.get("catalog_kind")
    if kind not in (None, "environment", "benchmark"):  # else a bogus kind renders as the error noun
        raise RegistryError(f"Unknown catalog kind '{kind}'.")
    # Scope the did-you-mean pool to the requested kind so it never suggests the kind the user did not ask for.
    matching_names = {entry.name: entry for entry in entries if kind is None or entry.kind == kind}
    if name not in matching_names:
        # A `kind` is always set here when the name exists at all, since an unset one matches everything.
        known = next((entry for entry in entries if entry.name == name), None)
        if known is not None:
            article = "an" if known.kind.startswith(("a", "e", "i", "o", "u")) else "a"
            rich.print(
                f"[red]'{name}' is {article} {known.kind}, not a {kind}.[/red] Try `gym list {known.kind}s {name}`."
            )
            sys.exit(1)
        exit_unknown_component(name, matching_names, kind or "environment")
        return
    entry = resolve_catalog_entry(name, kind, entries=entries)
    parsed = read_environment_details(entry.config_path)
    details = {"config": str(entry.config_path.resolve())}
    if entry.status is not None:
        details["status"] = entry.status
    if entry.manifest_path is not None:
        details["manifest"] = str(entry.manifest_path.resolve())
    for label, value in (
        ("version", entry.version),
        ("profile", entry.integration_profile),
        ("modality", entry.modality),
        ("licensing", entry.licensing),
        ("lifecycle", entry.lifecycle),
    ):
        if value:
            details[label] = value
    resources_server, agent_server = _entry_servers(entry, parsed)
    if resources_server:
        details["resources servers"] = resources_server
    if agent_server:
        details["agent"] = agent_server
    if parsed["datasets"]:
        details["datasets"] = ", ".join(parsed["datasets"])

    description = entry.description or parsed["description"]
    if parsed["value"]:  # surface `value` as a trailing line of the description
        description = f"{description}\nValue: {parsed['value']}" if description else f"Value: {parsed['value']}"

    render_component_inspection(
        json_output=global_config_dict.get(JSON_OUTPUT_KEY_NAME, False),
        name=name,
        type_noun=entry.kind,
        domain=entry.domain or parsed["domain"],
        description=description,
        details=details,
        usage=(
            f"gym env start --resources-server {entry.resources_server_selector} --model-type vllm_model"
            if entry.resources_server_selector
            else f"gym env start --{entry.kind} {name} --model-type vllm_model"
        ),
    )


def _entry_servers(
    entry: EnvironmentCatalogEntry, details: Optional[Dict[str, object]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """The resources server and agent an entry runs with, as the catalog views show them.

    The manifest says when there is one. Otherwise the config's runtime routes select the servers. A
    config that needs values only a run supplies, such as a judge model server, or that is malformed does not
    resolve; then the names written in the config file are used, each only when the file names exactly one, since a
    file naming several does not say which one runs. ``details`` is the config's ``read_environment_details``
    result when the caller has it.
    """
    if entry.manifest_path is not None:
        return entry.resources_server, entry.agent_server
    resolved = resolve_config_servers(entry.config_path)
    if resolved is not None:
        return resolved
    details = details or read_environment_details(entry.config_path)
    resources_servers, agents = set(details["resources_servers"]), set(details["agents"])
    return (
        next(iter(resources_servers)) if len(resources_servers) == 1 else None,
        next(iter(agents)) if len(agents) == 1 else None,
    )


def _catalog_payload(entry: EnvironmentCatalogEntry) -> Dict[str, object]:
    resources_server, agent_server = _entry_servers(entry)
    payload = {
        "name": entry.name,
        "kind": entry.kind,
        "status": entry.status,
        "domain": entry.domain,
        "description": entry.description,
        "version": entry.version,
        "integration_profile": entry.integration_profile,
        "modality": entry.modality,
        "licensing": entry.licensing,
        "lifecycle": entry.lifecycle,
        # Absolute, like the agent inspect view, so an entry found under any search root can be located.
        "path": str(entry.path.resolve()),
        "config_path": str(entry.config_path.resolve()),
        "manifest_path": str(entry.manifest_path.resolve()) if entry.manifest_path else None,
        "resources_server": resources_server,
        "agent_server": agent_server,
    }
    if entry.status is None:
        payload.pop("status")
    return payload


@exit_cleanly_on_config_error
def list_environments() -> None:
    """List or inspect the manifest and legacy environment/benchmark catalog."""
    global_config_dict = _command_overrides()
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    discovered = discover_environment_catalog()
    entries = list(discovered)

    name = global_config_dict.get(COMPONENT_NAME_KEY_NAME)
    if name:
        _inspect_environment(name, discovered, global_config_dict)
        return

    query = global_config_dict.get(QUERY_KEY_NAME)
    if query:
        entries = [
            entry for entry in entries if fuzzy_matches(query, entry.name, entry.domain or "", entry.description or "")
        ]
    for field in ("domain", "catalog_kind", "modality", "licensing", "status", "lifecycle"):
        expected = global_config_dict.get(field)
        if expected is None:
            continue
        attribute = "kind" if field == "catalog_kind" else field
        missing = sum(getattr(entry, attribute) is None for entry in entries)
        if missing and field != "status":
            noun = "entry" if missing == 1 else "entries"
            print(
                f"Warning: {missing} catalog {noun} {'has' if missing == 1 else 'have'} no "
                f"{attribute} metadata and could not be matched.",
                file=sys.stderr,
            )
        entries = [entry for entry in entries if getattr(entry, attribute) == expected]

    if global_config_dict.get(JSON_OUTPUT_KEY_NAME, False):
        print(json.dumps([_catalog_payload(entry) for entry in entries]))
        return

    if not entries:
        print_no_matches("environments", query)
        return

    coverage = sum(entry.manifest_path is not None for entry in discovered)
    title = (
        f"Catalog entries matching '{query}' ({len(entries)}; manifests {coverage}/{len(discovered)})"
        if query
        else f"Available environments and benchmarks ({len(entries)}; manifests {coverage}/{len(discovered)})"
    )
    table = Table(title=title)
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("Lifecycle")
    table.add_column("Domain")
    table.add_column("Description")
    for entry in entries:
        table.add_row(
            entry.name,
            entry.kind,
            entry.status or "",
            entry.lifecycle or "",
            entry.domain or "",
            entry.description or "",
        )

    print_rich_table(table)


@exit_cleanly_on_config_error
def status():  # pragma: no cover
    global_config_dict = get_global_config_dict()
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    status_cmd = StatusCommand()
    servers = status_cmd.discover_servers()

    if global_config_dict.get(JSON_OUTPUT_KEY_NAME, False):
        print(json.dumps([server.model_dump(mode="json") for server in servers]))
        return

    status_cmd.display_status(servers)


class PipListConfig(RunConfig):
    format: Optional[str] = Field(
        default=None,
        description="Output format for pip list. Options: 'columns' (default), 'freeze', 'json'",
    )
    outdated: bool = Field(
        default=False,
        description="List outdated packages",
    )


@exit_cleanly_on_config_error
def pip_list():  # pragma: no cover
    """List packages installed in a server's virtual environment."""
    global_config_dict = get_global_config_dict()
    config = PipListConfig.model_validate(global_config_dict)

    dir_path = _resolve_server_dir(Path(config.entrypoint))
    venv_path = dir_path / ".venv"

    if not venv_path.exists():
        print(f"  Virtual environment not found at: {venv_path}")
        print("  Run tests or setup the server first using:")
        print(f"  gym env test +entrypoint={config.entrypoint}")
        exit(1)

    pip_list_cmd = "uv pip list"
    if config.format:
        pip_list_cmd += f" --format={config.format}"
    if config.outdated:
        pip_list_cmd += " --outdated"

    command = f"""cd {dir_path} \\
    && source .venv/bin/activate \\
    && {pip_list_cmd}"""

    # Only in the default human view: an explicit --format (json, freeze) is meant to be piped, so stdout
    # must carry nothing but `uv pip list` output.
    if not config.format:
        print(f"  Package list for: {config.entrypoint}")
        print(f"Virtual environment: {venv_path.absolute()}")
        print("-" * 72)

    proc = run_command(command, dir_path)
    return_code = proc.wait()
    exit(return_code)


class SchemaConfig(RunConfig):
    """
    Print a resources server's task-data schema as JSON Schema.

    Examples:

    ```bash
    gym env schema --resources-server example_multi_step
    ```
    """


@exit_cleanly_on_config_error
def show_schema():  # pragma: no cover
    """Print the selected server's ``TaskData`` schema (see ``nemo_gym/data/task_data.py``)."""
    from nemo_gym.data.task_data import TaskDataSchemaError, load_task_data_schema

    global_config_dict = get_global_config_dict()
    config = SchemaConfig.model_validate(global_config_dict)

    dir_path = _resolve_server_dir(Path(config.entrypoint))
    try:
        adapter = load_task_data_schema(dir_path)
    except TaskDataSchemaError as e:
        print(str(e))
        exit(1)
    if adapter is None:
        print(
            f"{config.entrypoint} does not ship a task_data.py schema yet. "
            "Add one exporting `TaskData` (see nemo_gym/data/task_data.py for the protocol)."
        )
        exit(1)
    print(json.dumps(adapter.json_schema(), indent=2))


# Names that moved to other modules stay importable from here until their removal release; see nemo_gym/_moved_paths.py.
__getattr__ = moved_names_getattr(__name__)
