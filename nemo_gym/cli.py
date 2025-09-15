# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import shlex
from glob import glob
from os import environ, makedirs
from os.path import exists
from pathlib import Path
from subprocess import Popen
from threading import Thread
from time import sleep
from typing import Dict, List, Optional

from devtools import pprint
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from tqdm.auto import tqdm

from nemo_gym import PARENT_DIR
from nemo_gym.global_config import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
    GlobalConfigDictParserConfig,
    get_global_config_dict,
)
from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME, HeadServer, ServerClient, ServerStatus


def _setup_env_command(dir_path: Path) -> str:  # pragma: no cover
    return f"""cd {dir_path} \\
    && uv venv --allow-existing \\
    && source .venv/bin/activate \\
    && uv pip install -r requirements.txt \\
   """


def _run_command(command: str, working_directory: Path) -> Popen:  # pragma: no cover
    custom_env = environ.copy()
    custom_env["PYTHONPATH"] = f"{working_directory.absolute()}:{custom_env.get('PYTHONPATH', '')}"
    print(f"Executing command:\n{command}\n")
    return Popen(command, executable="/bin/bash", shell=True, env=custom_env)


class RunConfig(BaseModel):
    entrypoint: str


class TestConfig(RunConfig):
    should_validate_data: bool = False

    dir_path: Path = None  # initialized in model_post_init

    def model_post_init(self, context):  # pragma: no cover
        # TODO: This currently only handles relative entrypoints. Later on we can resolve the absolute path.
        self.dir_path = Path(self.entrypoint)
        assert not self.dir_path.is_absolute()
        assert len(self.dir_path.parts) == 2

        return super().model_post_init(context)


class ServerInstanceDisplayConfig(BaseModel):
    process_name: str
    server_type: str
    name: str
    dir_path: Path
    entrypoint: str
    host: Optional[str] = None
    port: Optional[int] = None
    pid: Optional[int] = None
    config_path: str
    url: Optional[str] = None


class RunHelper:  # pragma: no cover
    _head_server_thread: Thread
    _processes: Dict[str, Popen]
    _server_instance_display_configs: List[ServerInstanceDisplayConfig]
    _server_client: ServerClient

    def start(self, global_config_dict_parser_config: GlobalConfigDictParserConfig) -> None:
        global_config_dict = get_global_config_dict(global_config_dict_parser_config=global_config_dict_parser_config)

        # Assume Nemo Gym Run is for a single agent.
        escaped_config_dict_yaml_str = shlex.quote(OmegaConf.to_yaml(global_config_dict))

        # We always run the head server in this `run` command.
        self._head_server_thread = HeadServer.run_webserver()

        top_level_paths = [k for k in global_config_dict.keys() if k not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS]

        self._processes: Dict[str, Popen] = dict()
        self._server_instance_display_configs: List[ServerInstanceDisplayConfig] = []

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

            command = f"""{_setup_env_command(dir_path)} \\
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
                )
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
            pprint(inst.model_dump())
        print(f"{'#' * 100}\n")

    def poll(self) -> None:
        if not self._head_server_thread.is_alive():
            raise RuntimeError("Head server finished unexpectedly!")

        for process_name, process in self._processes.items():
            if process.poll() is not None:
                raise RuntimeError(f"Process `{process_name}` finished unexpectedly!")

    def wait_for_spinup(self) -> None:
        sleep_interval = 3

        # Until we spin up or error out.
        while True:
            self.poll()
            statuses = self.check_http_server_statuses()

            num_spun_up = statuses.count("success")
            if len(statuses) != num_spun_up:
                print(
                    f"""{num_spun_up} / {len(statuses)} servers ready ({statuses.count("timeout")} timed out, {statuses.count("connection_error")} connection errored, {statuses.count("unknown_error")} had unknown errors).
Waiting for servers to spin up. Sleeping {sleep_interval}s..."""
                )
            else:
                print(f"All {num_spun_up} / {len(statuses)} servers ready! Polling every 60s")
                self.display_server_instance_info()
                return

            sleep(sleep_interval)

    def run_forever(self) -> None:
        async def sleep():
            # Indefinitely
            while True:
                self.poll()

                statuses = self.check_http_server_statuses()
                assert statuses.count("success") == len(statuses), "Found non-success statuses"

                await asyncio.sleep(60)

        try:
            asyncio.run(sleep())
        except KeyboardInterrupt:
            pass
        finally:
            for process_name, process in self._processes.items():
                print(f"Killing `{process_name}`")
                process.kill()
                process.wait()

            print("NeMo Gym finished!")

    def check_http_server_statuses(self) -> List[ServerStatus]:
        print(
            "Checking for HTTP server statuses (you should see some HTTP requests to `/` that may 404. This is expected.)"
        )
        statuses = []
        for server_instance_display_config in self._server_instance_display_configs:
            name = server_instance_display_config.config_path
            status = self._server_client.poll_for_status(name)
            statuses.append(status)

        return statuses


def run(
    global_config_dict_parser_config: Optional[GlobalConfigDictParserConfig] = None,
):  # pragma: no cover
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

    server_type_name = test_config.dir_path.parts[1]
    example_metrics_fpath = test_config.dir_path / "data/example_metrics.json"
    assert (
        example_metrics_fpath.exists()
    ), f"""You must run the example data validation for the example data found at {example_fpath}.
Your command should look something like:
```bash
ng_prepare_data "+config_paths=[configs/{server_type_name}.yaml]" \\
    +output_dirpath=data/{server_type_name} \\
    +mode=example_validation
```
and your config must include an agent server config with an example dataset like:
```yaml
multineedle_simple_agent:
  responses_api_agents:
    simple_agent:
      ...
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/multineedle/data/example.jsonl
```

See `resources_servers/multineedle/configs/multineedle.yaml` for a full config example.
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
multineedle_config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/multineedle/configs/multineedle.yaml"
ng_run "+config_paths=[${{multineedle_config_paths}}]"

# Collect the rollouts
ng_collect_rollouts +agent_name=multineedle_simple_agent \
    +input_jsonl_fpath=resources_servers/multineedle/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/multineedle/data/example_rollouts.jsonl \
    +limit=null

# View your rollouts
ng_viewer +jsonl_fpath=resources_servers/multineedle/data/example_rollouts.jsonl
```
"""
    with open(example_rollouts_fpath) as f:
        count = sum(1 for _ in f)
    assert count == 5, f"Expected 5 example rollouts in {example_rollouts_fpath}, but got {count}"

    print(f"The data for {test_config.dir_path} has been successfully validated!")


def _test_single(test_config: TestConfig) -> Popen:  # pragma: no cover
    # Eventually we may want more sophisticated testing here, but this is sufficient for now.
    command = f"""{_setup_env_command(test_config.dir_path)} && pytest"""
    return _run_command(command, test_config.dir_path)


def test():  # pragma: no cover
    config_dict = get_global_config_dict()
    test_config = TestConfig.model_validate(config_dict)

    proc = _test_single(test_config)
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


class TestAllConfig(BaseModel):
    fail_on_total_and_test_mismatch: bool = False


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
        proc = _test_single(test_config)
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
    proc = Popen("pytest --cov=. --durations=10", shell=True)
    exit(proc.wait())


def init_resources_server():  # pragma: no cover
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
    server_type_name = dirpath.parts[1].lower()
    server_type_title = "".join(x.capitalize() for x in server_type_name.split("_"))

    configs_dirpath = dirpath / "configs"
    makedirs(configs_dirpath)

    config_fpath = configs_dirpath / f"{server_type_name}.yaml"
    with open(config_fpath, "w") as f:
        f.write(f"""{server_type_name}_resources_server:
  {server_type}:
    {server_type_name}:
      entrypoint: app.py
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
        gitlab_identifier:
          dataset_name: {server_type_name}
          version: 0.0.1
          artifact_fpath: train.jsonl
        license: Apache 2.0
      - name: validation
        type: validation
        jsonl_fpath: resources_servers/{server_type_name}/data/validation.jsonl
        gitlab_identifier:
          dataset_name: {server_type_name}
          version: 0.0.1
          artifact_fpath: validation.jsonl
        license: Apache 2.0
      - name: example
        type: example
        jsonl_fpath: resources_servers/{server_type_name}/data/example.jsonl
""")

    app_fpath = dirpath / "app.py"
    with open("resources/resources_server_template.py") as f:
        app_template = f.read()
    app_content = app_template.replace("MultiNeedle", server_type_title)
    with open(app_fpath, "w") as f:
        f.write(app_content)

    tests_dirpath = dirpath / "tests"
    makedirs(tests_dirpath)

    tests_fpath = tests_dirpath / "test_app.py"
    with open("resources/resources_server_test_template.py") as f:
        tests_template = f.read()
    tests_content = tests_template.replace("MultiNeedle", server_type_title)
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
    global_config_dict = get_global_config_dict()
    print(OmegaConf.to_yaml(global_config_dict, resolve=True))
