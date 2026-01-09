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
import asyncio
import json
import os
import platform
import shlex
import sys
import tomllib
from glob import glob
from importlib.metadata import version as md_version
from os import environ, makedirs
from os.path import exists
from pathlib import Path
from signal import SIGINT
from subprocess import Popen
from threading import Thread
from time import sleep, time
from typing import Dict, List, Optional, Tuple

import psutil
import rich
import uvicorn
from devtools import pprint
from omegaconf import DictConfig, OmegaConf
from pydantic import Field
from tqdm.auto import tqdm

from nemo_gym import PARENT_DIR, __version__
from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.global_config import (
    HEAD_SERVER_DEPS_KEY_NAME,
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
    PYTHON_VERSION_KEY_NAME,
    UV_PIP_SET_PYTHON_KEY_NAME,
    GlobalConfigDictParserConfig,
    get_global_config_dict,
)
from nemo_gym.server_status import StatusCommand
from nemo_gym.server_utils import (
    HEAD_SERVER_KEY_NAME,
    HeadServer,
    ServerClient,
    ServerInstanceDisplayConfig,
    ServerStatus,
    initialize_ray,
)


def _setup_env_command(dir_path: Path, global_config_dict: DictConfig) -> str:  # pragma: no cover
    head_server_deps = global_config_dict[HEAD_SERVER_DEPS_KEY_NAME]

    uv_venv_cmd = f"uv venv --seed --allow-existing --python {global_config_dict[PYTHON_VERSION_KEY_NAME]} .venv"

    has_pyproject_toml = exists(f"{dir_path / 'pyproject.toml'}")
    has_requirements_txt = exists(f"{dir_path / 'requirements.txt'}")

    # explicitly set python path if specified. In Google colab, ng_run fails due to uv pip install falls back to system python (/usr) without this and errors.
    # not needed for most clusters. should be safe in all scenarios, but only minimally tested outside of colab.
    # see discussion and examples here: https://github.com/NVIDIA-NeMo/Gym/pull/526#issuecomment-3676230383
    uv_pip_set_python = global_config_dict.get(UV_PIP_SET_PYTHON_KEY_NAME, False)
    uv_pip_python_flag = "--python .venv/bin/python" if uv_pip_set_python else ""

    if has_pyproject_toml and has_requirements_txt:
        raise RuntimeError(
            f"Found both pyproject.toml and requirements.txt for uv venv setup in server dir: {dir_path}. Please only use one or the other!"
        )
    elif has_pyproject_toml:
        install_cmd = f"""uv pip install {uv_pip_python_flag} '-e .' {" ".join(head_server_deps)}"""
    elif has_requirements_txt:
        install_cmd = f"""uv pip install {uv_pip_python_flag} -r requirements.txt {" ".join(head_server_deps)}"""
    else:
        raise RuntimeError(f"Missing pyproject.toml or requirements.txt for uv venv setup in server dir: {dir_path}")

    cmd = f"""cd {dir_path} \\
    && {uv_venv_cmd} \\
    && source .venv/bin/activate \\
    && {install_cmd} \\
    """

    return cmd


def _run_command(command: str, working_dir_path: Path) -> Popen:  # pragma: no cover
    work_dir = f"{working_dir_path.absolute()}"
    custom_env = environ.copy()
    py_path = custom_env.get("PYTHONPATH", None)
    if py_path is not None:
        custom_env["PYTHONPATH"] = f"{work_dir}:{py_path}"
    else:
        custom_env["PYTHONPATH"] = work_dir
    return Popen(command, executable="/bin/bash", shell=True, env=custom_env)


class RunConfig(BaseNeMoGymCLIConfig):
    """
    Start NeMo Gym servers for agents, models, and resources.

    Examples:

    ```bash
    config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\\
    responses_api_models/openai_model/configs/openai_model.yaml"
    ng_run "+config_paths=[${config_paths}]"
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
    ng_test +entrypoint=resources_servers/example_single_tool_call
    ```
    """

    should_validate_data: bool = Field(
        default=False,
        description="Whether or not to validate the example data (examples, metrics, rollouts, etc) for this server.",
    )

    _dir_path: Path  # initialized in model_post_init

    def model_post_init(self, context):  # pragma: no cover
        # TODO: This currently only handles relative entrypoints. Later on we can resolve the absolute path.
        self._dir_path = Path(self.entrypoint)
        assert not self.dir_path.is_absolute()
        assert len(self.dir_path.parts) == 2

        return super().model_post_init(context)

    @property
    def dir_path(self) -> Path:
        return self._dir_path


class RunHelper:  # pragma: no cover
    _head_server: uvicorn.Server
    _head_server_thread: Thread
    _head_server_instance: HeadServer

    _processes: Dict[str, Popen]
    _server_instance_display_configs: List[ServerInstanceDisplayConfig]
    _server_client: ServerClient

    def start(self, global_config_dict_parser_config: GlobalConfigDictParserConfig) -> None:
        global_config_dict = get_global_config_dict(global_config_dict_parser_config=global_config_dict_parser_config)

        # Initialize Ray cluster in the main process
        # Note: This function will modify the global config dict - update `ray_head_node_address`
        initialize_ray()

        # Assume Nemo Gym Run is for a single agent.
        escaped_config_dict_yaml_str = shlex.quote(OmegaConf.to_yaml(global_config_dict))

        # We always run the head server in this `run` command.
        self._head_server, self._head_server_thread, self._head_server_instance = HeadServer.run_webserver()

        top_level_paths = [k for k in global_config_dict.keys() if k not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS]

        self._processes: Dict[str, Popen] = dict()
        self._server_instance_display_configs: List[ServerInstanceDisplayConfig] = []

        start_time = time()

        # TODO there is a better way to resolve this that uses nemo_gym/global_config.py::ServerInstanceConfig
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

            # TODO: This currently only handles relative entrypoints. Later on we can resolve the absolute path.
            entrypoint_fpath = Path(server_config_dict.entrypoint)
            assert not entrypoint_fpath.is_absolute()

            dir_path = PARENT_DIR / Path(first_key, second_key)

            command = f"""{_setup_env_command(dir_path, global_config_dict)} \\
    && {NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME}={escaped_config_dict_yaml_str} \\
    {NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME}={shlex.quote(top_level_path)} \\
    python {str(entrypoint_fpath)}"""

            process = _run_command(command, dir_path)
            self._processes[top_level_path] = process

            host = server_config_dict.get("host")
            port = server_config_dict.get("port")

            self._server_instance_display_configs.append(
                ServerInstanceDisplayConfig(
                    process_name=top_level_path,
                    server_type=first_key,
                    name=second_key,
                    dir_path=str(dir_path),
                    entrypoint=str(entrypoint_fpath),
                    host=host,
                    port=port,
                    url=f"http://{host}:{port}" if host and port else None,
                    pid=process.pid,
                    config_path=top_level_path,
                    start_time=start_time,
                )
            )

        self._head_server_instance.set_server_instances(
            [inst.model_dump(mode="json") for inst in self._server_instance_display_configs]
        )

        self._server_client = ServerClient(
            head_server_config=ServerClient.load_head_server_config(),
            global_config_dict=global_config_dict,
        )

        print("Waiting for head server to spin up")
        while True:
            status = self._server_client.poll_for_status(HEAD_SERVER_KEY_NAME)
            if status == "success":
                break

            print(f"Head server is not up yet (status `{status}`). Sleeping 3s")
            sleep(3)

        print("Waiting for servers to spin up")
        self.wait_for_spinup()

    def display_server_instance_info(self) -> None:
        if not self._server_instance_display_configs:
            print("No server instances to display.")
            return

        print(f"""
{"#" * 100}
#
# Server Instances
#
{"#" * 100}
""")

        for i, inst in enumerate(self._server_instance_display_configs, 1):
            print(f"[{i}] {inst.process_name} ({inst.server_type}/{inst.name})")
            pprint(inst.model_dump(mode="json", exclude={"start_time", "status", "uptime_seconds"}))
        print(f"{'#' * 100}\n")

    def poll(self) -> None:
        if not self._head_server_thread.is_alive():
            raise RuntimeError("Head server finished unexpectedly!")

        for process_name, process in self._processes.items():
            if process.poll() is not None:
                proc_out, proc_err = process.communicate()
                print_str = f"Process `{process_name}` finished unexpectedly!"

                if isinstance(proc_out, bytes):
                    proc_out = proc_out.decode("utf-8")
                    print_str = f"""{print_str}
Process `{process_name}` stdout:
{proc_out}
"""
                if isinstance(proc_err, bytes):
                    proc_err = proc_err.decode("utf-8")
                    print_str = f"""{print_str}
Process `{process_name}` stderr:
{proc_err}"""

                raise RuntimeError(print_str)

    def wait_for_spinup(self) -> None:
        sleep_interval = 3

        # Until we spin up or error out.
        while True:
            self.poll()
            statuses = self.check_http_server_statuses()

            num_spun_up = 0
            waiting = []
            for name, status in statuses:
                if status == "success":
                    num_spun_up += 1
                else:
                    waiting.append(name)
            if len(statuses) != num_spun_up:
                print(
                    f"""{num_spun_up} / {len(statuses)} servers ready ({statuses.count("timeout")} timed out, {statuses.count("connection_error")} connection errored, {statuses.count("unknown_error")} had unknown errors).
Waiting for servers to spin up: {waiting}
Sleeping {sleep_interval}s..."""
                )
            else:
                print(f"All {num_spun_up} / {len(statuses)} servers ready! Polling every 60s")
                self.display_server_instance_info()
                return

            sleep(sleep_interval)

    def shutdown(self) -> None:
        print("Sending interrupt signals to servers...")
        for process in self._processes.values():
            process.send_signal(SIGINT)

        print("Waiting for processes to finish...")
        for process in self._processes.values():
            process.wait()

        self._processes = dict()

        self._head_server.should_exit = True
        self._head_server_thread.join()

        self._head_server = None
        self._head_server_thread = None

        print("NeMo Gym finished!")

    def run_forever(self) -> None:
        async def sleep():
            # Indefinitely
            while True:
                self.poll()
                await asyncio.sleep(60)

        try:
            asyncio.run(sleep())
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def check_http_server_statuses(self) -> List[Tuple[str, ServerStatus]]:
        print(
            "Checking for HTTP server statuses (you should see some HTTP requests to `/` that may 404. This is expected.)"
        )
        statuses = []
        for server_instance_display_config in self._server_instance_display_configs:
            name = server_instance_display_config.config_path
            status = self._server_client.poll_for_status(name)
            statuses.append((name, status))

        return statuses


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
    ng_run "+config_paths=[${config_paths}]"
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

    # Check that the required examples and example metrics are present.
    example_fpath = test_config.dir_path / "data/example.jsonl"
    assert example_fpath.exists(), (
        f"A jsonl file containing 5 examples is required for the {test_config.dir_path} resources server. The file must be found at {example_fpath}. Usually this example data is just the first 5 examples of your train dataset."
    )
    with open(example_fpath) as f:
        count = sum(1 for _ in f)
    assert count == 5, f"Expected 5 examples at {example_fpath} but got {count}."

    server_type_name = test_config.dir_path.parts[-1]
    example_metrics_fpath = test_config.dir_path / "data/example_metrics.json"
    assert (
        example_metrics_fpath.exists()
    ), f"""You must run the example data validation for the example data found at {example_fpath}.
Your command should look something like the following (you should update this command with your actual server config path):
```bash
ng_prepare_data "+config_paths=[responses_api_models/openai_model/configs/openai_model.yaml,configs/{server_type_name}.yaml]" \\
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

    conflict_paths = glob(str(test_config.dir_path / "data/*conflict*"))
    conflict_paths_str = "\n- ".join([""] + conflict_paths)
    assert not conflict_paths, f"Found {len(conflict_paths)} conflicting paths: {conflict_paths_str}"

    example_rollouts_fpath = test_config.dir_path / "data/example_rollouts.jsonl"
    assert example_rollouts_fpath.exists(), f"""You must run the example data through your agent and provide the example rollouts at `{example_rollouts_fpath}`.

Your commands should look something like:
```bash
# Server spinup
example_multi_step_config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_multi_step/configs/example_multi_step.yaml"
ng_run "+config_paths=[${{example_multi_step_config_paths}}]"

# Collect the rollouts
ng_collect_rollouts +agent_name=example_multi_step_simple_agent \
    +input_jsonl_fpath=resources_servers/example_multi_step/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/example_multi_step/data/example_rollouts.jsonl \
    +limit=null

# View your rollouts
ng_viewer +jsonl_fpath=resources_servers/example_multi_step/data/example_rollouts.jsonl
```
"""
    with open(example_rollouts_fpath) as f:
        count = sum(1 for _ in f)
    assert count == 5, f"Expected 5 example rollouts in {example_rollouts_fpath}, but got {count}"

    print(f"The data for {test_config.dir_path} has been successfully validated!")


def _test_single(test_config: TestConfig, global_config_dict: DictConfig) -> Popen:  # pragma: no cover
    # Eventually we may want more sophisticated testing here, but this is sufficient for now.
    command = f"""{_setup_env_command(test_config.dir_path, global_config_dict)} && pytest"""
    return _run_command(command, test_config.dir_path)


def test():  # pragma: no cover
    global_config_dict = get_global_config_dict()
    test_config = TestConfig.model_validate(global_config_dict)

    proc = _test_single(test_config, global_config_dict)
    return_code = proc.wait()
    if return_code != 0:
        print(f"You can run detailed tests via `cd {test_config.entrypoint} && source .venv/bin/activate && pytest`.")
        exit(return_code)

    _validate_data_single(test_config)


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
    ng_test_all
    ```
    """

    fail_on_total_and_test_mismatch: bool = Field(
        default=False,
        description="Fail if the number of server modules doesn't match the number with tests (default: False).",
    )


def test_all():  # pragma: no cover
    global_config_dict = get_global_config_dict()
    test_all_config = TestAllConfig.model_validate(global_config_dict)

    candidate_dir_paths = [
        *glob("resources_servers/*"),
        *glob("responses_api_agents/*"),
        *glob("responses_api_models/*"),
    ]
    candidate_dir_paths = [p for p in candidate_dir_paths if "pycache" not in p]
    print(f"Found {len(candidate_dir_paths)} total modules:{_display_list_of_paths(candidate_dir_paths)}\n")
    dir_paths: List[Path] = list(map(Path, candidate_dir_paths))
    dir_paths = [p for p in dir_paths if (p / "README.md").exists()]
    print(f"Found {len(dir_paths)} modules to test:{_display_list_of_paths(dir_paths)}\n")

    tests_passed: List[Path] = []
    tests_failed: List[Path] = []
    tests_missing: List[Path] = []
    data_validation_failed: List[Path] = []
    for dir_path in tqdm(dir_paths, desc="Running tests"):
        test_config = TestConfig(
            entrypoint=str(dir_path),
            should_validate_data=True,  # Test all always validates data.
        )
        proc = _test_single(test_config, global_config_dict)
        return_code = proc.wait()

        match return_code:
            case 0:
                tests_passed.append(dir_path)
            case 1:
                tests_failed.append(dir_path)
            case 5:
                tests_missing.append(dir_path)
            case _:
                raise ValueError(
                    f"""Hit unrecognized exit code {return_code} while running tests for {dir_path}.
You can rerun just these tests using `ng_test +entrypoint={dir_path}` or run detailed tests via `cd {dir_path} && source .venv/bin/activate && pytest`."""
                )

        try:
            _validate_data_single(test_config)
        except AssertionError:
            data_validation_failed.append(dir_path)

    print(f"""Found {len(candidate_dir_paths)} total modules:{_display_list_of_paths(candidate_dir_paths)}

Found {len(dir_paths)} modules to test:{_display_list_of_paths(dir_paths)}

Tests passed {_format_pct(len(tests_passed), len(dir_paths))}:{_display_list_of_paths(tests_passed)}

Tests failed {_format_pct(len(tests_failed), len(dir_paths))}:{_display_list_of_paths(tests_failed)}

Tests missing {_format_pct(len(tests_missing), len(dir_paths))}:{_display_list_of_paths(tests_missing)}

Data validation failed {_format_pct(len(data_validation_failed), len(dir_paths))}:{_display_list_of_paths(data_validation_failed)}
""")

    if tests_failed or tests_missing:
        print(f"""You can rerun just the server with failed or missing tests like:
```bash
ng_test +entrypoint={(tests_failed + tests_missing)[0]}
```
""")
    if data_validation_failed:
        print(f"""You can rerun just the server with failed data validation like:
```bash
ng_test +entrypoint={data_validation_failed[0]} +should_validate_data=true
```
""")

    if test_all_config.fail_on_total_and_test_mismatch:
        extra_candidates = [p for p in candidate_dir_paths if Path(p) not in dir_paths]
        assert (
            len(candidate_dir_paths) == len(dir_paths)
        ), f"""Mismatch on the number of total modules found ({len(candidate_dir_paths)}) and the number of actual modules tested ({len(dir_paths)})!

Extra candidate paths:{_display_list_of_paths(extra_candidates)}"""

    if tests_missing or tests_failed or data_validation_failed:
        exit(1)


def dev_test():  # pragma: no cover
    """
    Run core NeMo Gym tests with coverage reporting (runs pytest with --cov flag).

    Examples:

    ```bash
    ng_dev_test
    ```
    """
    global_config_dict = get_global_config_dict()
    # Just here for help
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    proc = Popen("pytest --cov=. --durations=10", shell=True)
    exit(proc.wait())


def init_resources_server():  # pragma: no cover
    """
    Initialize a new resources server with template files and directory structure.

    Examples:

    ```bash
    ng_init_resources_server +entrypoint=resources_servers/my_server
    ```
    """
    config_dict = get_global_config_dict()
    run_config = RunConfig.model_validate(config_dict)

    if exists(run_config.entrypoint):
        print(f"Folder already exists: {run_config.entrypoint}. Exiting init.")
        exit()

    dirpath = Path(run_config.entrypoint)
    assert len(dirpath.parts) == 2
    makedirs(dirpath)

    server_type = dirpath.parts[0]
    assert server_type == "resources_servers"
    server_type_name = dirpath.parts[-1].lower()
    server_type_title = "".join(x.capitalize() for x in server_type_name.split("_"))

    configs_dirpath = dirpath / "configs"
    makedirs(configs_dirpath)

    config_fpath = configs_dirpath / f"{server_type_name}.yaml"
    with open(config_fpath, "w") as f:
        f.write(f"""{server_type_name}_resources_server:
  {server_type}:
    {server_type_name}:
      entrypoint: app.py
      domain: other
{server_type_name}_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: {server_type_name}_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        jsonl_fpath: resources_servers/{server_type_name}/data/train.jsonl
        num_repeats: 1
        gitlab_identifier:
          dataset_name: {server_type_name}
          version: 0.0.1
          artifact_fpath: train.jsonl
        license: Apache 2.0
      - name: validation
        type: validation
        jsonl_fpath: resources_servers/{server_type_name}/data/validation.jsonl
        num_repeats: 1
        gitlab_identifier:
          dataset_name: {server_type_name}
          version: 0.0.1
          artifact_fpath: validation.jsonl
        license: Apache 2.0
      - name: example
        type: example
        jsonl_fpath: resources_servers/{server_type_name}/data/example.jsonl
        num_repeats: 1
""")

    app_fpath = dirpath / "app.py"
    with open("resources/resources_server_template.py") as f:
        app_template = f.read()
    app_content = app_template.replace("ExampleMultiStep", server_type_title)
    with open(app_fpath, "w") as f:
        f.write(app_content)

    tests_dirpath = dirpath / "tests"
    makedirs(tests_dirpath)

    tests_fpath = tests_dirpath / "test_app.py"
    with open("resources/resources_server_test_template.py") as f:
        tests_template = f.read()
    tests_content = tests_template.replace("ExampleMultiStep", server_type_title)
    tests_content = tests_content.replace("from app", f"from resources_servers.{server_type_name}.app")
    with open(tests_fpath, "w") as f:
        f.write(tests_content)

    requirements_fpath = dirpath / "requirements.txt"
    with open(requirements_fpath, "w") as f:
        f.write("""-e nemo-gym[dev] @ ../../
""")

    readme_fpath = dirpath / "README.md"
    with open(readme_fpath, "w") as f:
        f.write("""# Description

Data links: ?

# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
""")

    data_dirpath = dirpath / "data"
    data_dirpath.mkdir(exist_ok=True)

    data_gitignore_fpath = data_dirpath / ".gitignore"
    with open(data_gitignore_fpath, "w") as f:
        f.write("""*train.jsonl
*validation.jsonl
*train_prepare.jsonl
*validation_prepare.jsonl
*example_prepare.jsonl
""")


def dump_config():  # pragma: no cover
    """
    Display the resolved Hydra configuration for debugging purposes.

    Examples:

    ```bash
    ng_dump_config "+config_paths=[<config1>,<config2>]"
    ```
    """
    global_config_dict = get_global_config_dict()
    # Just here for help
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    print(OmegaConf.to_yaml(global_config_dict, resolve=True))


def display_help():  # pragma: no cover
    """
    Display a list of available NeMo Gym CLI commands.

    Examples:

    ```bash
    ng_help
    ```
    """
    global_config_dict = get_global_config_dict()
    # Just here for help
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    pyproject_path = Path(PARENT_DIR) / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproject_data = tomllib.load(f)

    project_scripts = pyproject_data["project"]["scripts"]
    rich.print("""Run a command with `+h=true` or `+help=true` to see more detailed information!

[bold]Available CLI scripts[/bold]
-----------------""")
    for script in project_scripts:
        if not script.startswith("ng_"):
            continue

        print(script)


def status():  # pragma: no cover
    global_config_dict = get_global_config_dict()
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    status_cmd = StatusCommand()
    servers = status_cmd.discover_servers()
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


def pip_list():  # pragma: no cover
    """List packages installed in a server's virtual environment."""
    global_config_dict = get_global_config_dict()
    config = PipListConfig.model_validate(global_config_dict)

    dir_path = Path(config.entrypoint)
    venv_path = dir_path / ".venv"

    if not venv_path.exists():
        print(f"  Virtual environment not found at: {venv_path}")
        print("  Run tests or setup the server first using:")
        print(f"  ng_test +entrypoint={config.entrypoint}")
        exit(1)

    pip_list_cmd = "uv pip list"
    if config.format:
        pip_list_cmd += f" --format={config.format}"
    if config.outdated:
        pip_list_cmd += " --outdated"

    command = f"""cd {dir_path} \\
    && source .venv/bin/activate \\
    && {pip_list_cmd}"""

    print(f"  Package list for: {config.entrypoint}")
    print(f"Virtual environment: {venv_path.absolute()}")
    print("-" * 72)

    proc = _run_command(command, dir_path)
    return_code = proc.wait()
    exit(return_code)


class VersionConfig(BaseNeMoGymCLIConfig):
    """
    Display gym version and system information.

    Examples:

    ```bash
    # Display version information
    ng_version

    # Output as JSON
    ng_version +json=true
    ```
    """

    json_format: bool = Field(default=False, alias="json", description="Output in JSON format for programmatic use.")


def version():  # pragma: no cover
    """Display gym version and system information."""
    global_config_dict = get_global_config_dict()
    config = VersionConfig.model_validate(global_config_dict)

    json_output = config.json_format

    version_info = {
        "nemo_gym": __version__,
        "python": platform.python_version(),
        "python_path": sys.executable,
        "installation_path": str(PARENT_DIR),
    }

    key_deps = [
        "openai",
        "ray",
    ]

    dependencies = {dep: md_version(dep) for dep in key_deps}

    version_info["dependencies"] = dependencies

    # System info
    version_info["system"] = {
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpus": os.cpu_count(),
    }

    # Memory info
    mem = psutil.virtual_memory()
    version_info["system"]["memory_gb"] = round(mem.total / (1024**3), 2)

    # Output
    if json_output:
        print(json.dumps(version_info))
    else:
        output = f"""\
NeMo Gym v{version_info["nemo_gym"]}
Python {version_info["python"]} ({version_info["python_path"]})
Installation: {version_info["installation_path"]}"""

        if "dependencies" in version_info:
            deps_lines = "\n".join(f"  {dep}: {ver}" for dep, ver in version_info["dependencies"].items())
            sys_info = version_info["system"]
            output += f"""

Key Dependencies:
{deps_lines}

System:
  OS: {sys_info["os"]}
  Platform: {sys_info["platform"]}
  Architecture: {sys_info["architecture"]}
  Processor: {sys_info["processor"]}
  CPUs: {sys_info["cpus"]}
  Memory: {sys_info["memory_gb"]} GB"""

        print(output)
