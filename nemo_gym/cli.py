from typing import Dict, List

from os import environ, makedirs
from os.path import exists

from pathlib import Path

from glob import glob

from subprocess import Popen

import asyncio

import shlex

from tqdm.auto import tqdm

from omegaconf import OmegaConf, DictConfig

from pydantic import BaseModel

from nemo_gym.server_utils import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
    HeadServer,
    get_global_config_dict,
)


def _setup_env_command(dir_path: Path) -> str:  # pragma: no cover
    return f"""cd {dir_path} \\
    && uv venv --allow-existing \\
    && source .venv/bin/activate \\
    && uv pip install -r requirements.txt \\
   """


def _run_command(command: str, working_directory: Path) -> Popen:  # pragma: no cover
    custom_env = environ.copy()
    custom_env["PYTHONPATH"] = (
        f"{working_directory.absolute()}:{custom_env.get('PYTHONPATH', '')}"
    )
    print(f"Executing command:\n{command}\n")
    return Popen(command, executable="/bin/bash", shell=True, env=custom_env)


class RunConfig(BaseModel):
    entrypoint: str


def run():  # pragma: no cover
    global_config_dict = get_global_config_dict()

    # Assume Nemo Gym Run is for a single agent.
    escaped_config_dict_yaml_str = shlex.quote(OmegaConf.to_yaml(global_config_dict))

    # We always run the head server in this `run` command.
    head_server_thread = HeadServer.run_webserver()

    top_level_paths = [
        k
        for k in global_config_dict.keys()
        if k not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS
    ]

    processes: Dict[str, Popen] = dict()
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

        dir_path = Path(first_key, second_key)

        command = f"""{_setup_env_command(dir_path)} \\
    && {NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME}={escaped_config_dict_yaml_str} \\
    {NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME}={shlex.quote(top_level_path)} \\
    python {str(entrypoint_fpath)}"""

        process = _run_command(command, dir_path)
        processes[top_level_path] = process

    async def sleep():
        # Indefinitely
        while True:
            if not head_server_thread.is_alive():
                raise RuntimeError("Head server finished unexpectedly!")

            for process_name, process in processes.items():
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Process `{process_name}` finished unexpectedly!"
                    )

            await asyncio.sleep(60)  # Check every 60s.

    try:
        asyncio.run(sleep())
    except KeyboardInterrupt:
        pass
    finally:
        for process_name, process in processes.items():
            print(f"Killing `{process_name}`")
            process.kill()
            process.wait()

        print("NeMo Gym finished!")


def _test_single(dir_path: Path) -> Popen:  # pragma: no cover
    # Eventually we may want more sophisticated testing here, but this is sufficient for now.
    command = f"""{_setup_env_command(dir_path)} && pytest"""
    return _run_command(command, dir_path)


def test():  # pragma: no cover
    config_dict = get_global_config_dict()
    run_config = RunConfig.model_validate(config_dict)

    # TODO: This currently only handles relative entrypoints. Later on we can resolve the absolute path.
    dir_path = Path(run_config.entrypoint)
    assert not dir_path.is_absolute()

    proc = _test_single(dir_path)
    return_code = proc.wait()
    exit(return_code)


def _display_list_of_paths(paths: List[Path]) -> str:  # pragma: no cover
    paths = list(map(str, paths))
    return "".join(f"\n- {p}" for p in paths)


def _format_pct(count: int, total: int) -> str:  # pragma: no cover
    return f"{count} / {total} ({100 * count / total:.2f}%)"


def test_all():  # pragma: no cover
    dir_paths = [
        *glob("resources_servers/*"),
        *glob("responses_api_agents/*"),
        *glob("responses_api_models/*"),
    ]
    dir_paths = [p for p in dir_paths if "pycache" not in p]
    print(f"Found {len(dir_paths)} modules to test:{_display_list_of_paths(dir_paths)}")
    dir_paths: List[Path] = list(map(Path, dir_paths))
    dir_paths = [p for p in dir_paths if (p / "README.md").exists()]

    tests_passed: List[Path] = []
    tests_failed: List[Path] = []
    tests_missing: List[Path] = []
    for dir_path in tqdm(dir_paths, desc="Running tests"):
        proc = _test_single(dir_path)
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
                    f"Hit unrecognized exit code {return_code} while running tests for {dir_path}"
                )

    print(f"""Tests passed {_format_pct(len(tests_passed), len(dir_paths))}:{_display_list_of_paths(tests_passed)}

Tests failed {_format_pct(len(tests_failed), len(dir_paths))}:{_display_list_of_paths(tests_failed)}

Tests missing {_format_pct(len(tests_missing), len(dir_paths))}:{_display_list_of_paths(tests_missing)}
""")

    if tests_missing or tests_failed:
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
        f.write(f"""{server_type_name}:
  {server_type}:
    {server_type_name}:
      entrypoint: app.py
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
    with open(tests_fpath, "w") as f:
        f.write(tests_content)

    requirements_fpath = dirpath / "requirements.txt"
    with open(requirements_fpath, "w") as f:
        f.write("""-e nemo-gym @ ../../
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
