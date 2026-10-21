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
import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest
import requests
from omegaconf import DictConfig
from pytest import MonkeyPatch

import nemo_gym.cli.eval as cli_eval
import nemo_gym.global_config
import nemo_gym.server_utils
from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.cli.eval import _validate_prepared_split_file_exists, _validate_split_datasets_declared
from nemo_gym.cli.main import main
from nemo_gym.config_types import ConfigError, ResponsesAPIAgentServerInstanceConfig
from nemo_gym.global_config import NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME


def _make_agent_instance_config(name: str, dataset_specs: list) -> ResponsesAPIAgentServerInstanceConfig:
    server_type_config_dict = {
        "responses_api_agents": {
            "simple_agent": {
                "host": "127.0.0.1",
                "port": 12345,
                "entrypoint": "app.py",
                "datasets": [
                    {
                        "name": d["name"],
                        "type": d["type"],
                        "jsonl_fpath": d.get("jsonl_fpath", f"path/{d['name']}.jsonl"),
                        "license": None if d["type"] == "example" else "Apache 2.0",
                    }
                    for d in dataset_specs
                ],
                "resources_server": {
                    "type": "resources_servers",
                    "name": f"{name}_resources_server",
                },
                "model_server": {
                    "type": "responses_api_models",
                    "name": "policy_model",
                },
            }
        }
    }
    return ResponsesAPIAgentServerInstanceConfig(
        name=name,
        server_type_config_dict=DictConfig(server_type_config_dict),
        responses_api_agents=server_type_config_dict["responses_api_agents"],
    )


class TestValidateSplitDatasetsDeclared:
    def test_passes_when_a_dataset_of_the_split_type_is_declared(self) -> None:
        configs = [_make_agent_instance_config("my_agent", [{"name": "train_data", "type": "train"}])]
        _validate_split_datasets_declared("train", configs)

    def test_fails_fast_when_only_example_data_is_declared(self) -> None:
        configs = [
            _make_agent_instance_config(
                "example_agent",
                [{"name": "example", "type": "example", "jsonl_fpath": "resources_servers/x/data/example.jsonl"}],
            )
        ]
        with pytest.raises(ConfigError) as exc_info:
            _validate_split_datasets_declared("train", configs)
        message = str(exc_info.value)
        # The error must name the requested split, list what is declared, and give the
        # copy-pasteable --no-serve recipe for the example file.
        assert "No dataset of type `train`" in message
        assert "example_agent: example (type: example)" in message
        assert "--no-serve --input resources_servers/x/data/example.jsonl" in message

    def test_fails_when_no_datasets_are_declared_at_all(self) -> None:
        configs = [_make_agent_instance_config("bare_agent", [])]
        with pytest.raises(ConfigError, match=r"- \(none\)"):
            _validate_split_datasets_declared("validation", configs)

    def test_mismatched_split_lists_declared_types(self) -> None:
        configs = [_make_agent_instance_config("val_agent", [{"name": "val_data", "type": "validation"}])]
        with pytest.raises(ConfigError, match=r"val_agent: val_data \(type: validation\)"):
            _validate_split_datasets_declared("train", configs)


class TestValidatePreparedSplitFileExists:
    def test_passes_when_the_file_exists(self, tmp_path: Path) -> None:
        fpath = tmp_path / "train.jsonl"
        fpath.write_text("{}\n")
        _validate_prepared_split_file_exists(fpath, "train", tmp_path)

    def test_fails_with_the_split_and_the_files_actually_prepared(self, tmp_path: Path) -> None:
        (tmp_path / "validation.jsonl").write_text("{}\n")
        with pytest.raises(ConfigError, match=r"split `train`.*\['validation.jsonl'\]"):
            _validate_prepared_split_file_exists(tmp_path / "train.jsonl", "train", tmp_path)

    def test_fails_with_none_when_the_output_dir_is_missing(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "does_not_exist"
        with pytest.raises(ConfigError, match=r"none"):
            _validate_prepared_split_file_exists(missing_dir / "train.jsonl", "train", missing_dir)


def test_aggregate_cli_exits_nonzero_after_saving_failed_agent_entry(tmp_path: Path) -> None:
    """A single-agent HTTP 500 must fail the real CLI process after saving error details."""
    shard = tmp_path / "shard.jsonl"
    shard.write_text(
        json.dumps({"agent_ref": {"name": "agent_a"}, "_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0})
        + "\n"
    )
    output = tmp_path / "rollouts.jsonl"
    script = dedent("""\
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        from aiohttp import ClientResponseError
        from omegaconf import OmegaConf
        from nemo_gym.cli.main import main
        from nemo_gym.rollout_collection import RolloutCollectionHelper

        error = ClientResponseError(
            request_info=SimpleNamespace(real_url="http://agent/aggregate_metrics"),
            history=(), status=500, message="aggregation unavailable",
        )
        client = SimpleNamespace(
            post=AsyncMock(side_effect=error),
            global_config_dict=OmegaConf.create({
                "agent_a": {"responses_api_agents": {"impl": {}}},
                "environment": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "agent_a"}}}},
            }),
        )
        with patch.object(RolloutCollectionHelper, "setup_server_client", return_value=client):
            main()
        """)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "eval",
            "aggregate",
            "--input",
            str(shard),
            "--output",
            str(output),
            "--no-health-check",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert "Aggregation failed for agents: agent_a" in completed.stdout + completed.stderr
    metrics = json.loads((tmp_path / "rollouts_aggregate_metrics.json").read_text())
    assert metrics[0]["aggregation_error"]["http_status"] == 500
    assert json.loads(output.read_text()) == json.loads(shard.read_text())


class TestPrepareDependencies:
    @staticmethod
    def _benchmark(project=None, dependencies=()):
        from nemo_gym.config_types import BenchmarkDatasetConfig

        dataset = BenchmarkDatasetConfig(
            name="b",
            type="benchmark",
            jsonl_fpath=Path("data.jsonl"),
            prepare_script=Path("prepare.py"),
            prepare_project=project,
            prepare_dependencies=list(dependencies),
        )
        return SimpleNamespace(name="b", dataset=dataset)

    def test_declared_project_syncs_without_mutating_parent(self, tmp_path, monkeypatch):
        from nemo_gym.cli import _locked_env

        calls = []
        monkeypatch.setattr(_locked_env, "sync_environment", lambda *a: calls.append(a))
        python = cli_eval._prepare_environment(self._benchmark(tmp_path))
        assert calls == [(tmp_path, tmp_path / ".venv", sys.executable, [], [])]
        assert python == tmp_path / ".venv/bin/python"

    def test_custom_venv_root_does_not_write_to_component(self, tmp_path, monkeypatch):
        from nemo_gym.cli import _locked_env

        project = tmp_path / "installed/benchmarks/example"
        monkeypatch.setattr(cli_eval, "PARENT_DIR", tmp_path / "installed")
        calls = []
        monkeypatch.setattr(_locked_env, "sync_environment", lambda *a: calls.append(a))
        python = cli_eval._prepare_environment(self._benchmark(project), tmp_path / "writable")
        venv = tmp_path / "writable/preparation/benchmarks/example/.venv"
        assert python == venv / "bin/python"
        assert calls == [(project, venv, sys.executable, [], [])]
        assert not project.exists()

    def test_legacy_unlocked_arguments_fail_with_migration(self):
        with pytest.raises(ConfigError, match="commit uv.lock, and set prepare_project"):
            cli_eval._prepare_environment(self._benchmark(dependencies=["-e", "benchmarks/automationbench"]))

    def test_nothing_runs_without_project(self, monkeypatch):
        from nemo_gym.cli import _locked_env

        monkeypatch.setattr(_locked_env, "sync_environment", lambda *a: pytest.fail("unexpected sync"))
        assert cli_eval._prepare_environment(self._benchmark()) is None

    def test_failed_sync_names_dataset(self, tmp_path, monkeypatch):
        from nemo_gym.cli import _locked_env

        def fail(*args):
            raise RuntimeError("stale lock")

        monkeypatch.setattr(_locked_env, "sync_environment", fail)
        with pytest.raises(ConfigError, match="locked prepare_project for dataset 'b': stale lock"):
            cli_eval._prepare_environment(self._benchmark(tmp_path))

    def test_worker_imports_preparer_only_in_child_and_checks_cache(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        output = tmp_path / "data.jsonl"
        (tmp_path / "isolated_preparer.py").write_text(
            "from pathlib import Path\n"
            "def prepare(target):\n    p=Path(target); p.write_text('data'); return p\n"
            "def is_prepared_data_current(path):\n    return path.read_text() == 'data'\n"
        )
        python = Path(sys.executable)
        probe = cli_eval._run_isolated_prepare(python, "isolated_preparer", "inspect", output, {"check_cache": True})
        assert probe == {"has_prepare": True, "cache_current": False}
        result = cli_eval._run_isolated_prepare(
            python, "isolated_preparer", "prepare", output, {"target": str(output)}
        )
        assert result == {"output": str(output)}
        assert output.read_text() == "data"
        assert "isolated_preparer" not in sys.modules
        probe = cli_eval._run_isolated_prepare(python, "isolated_preparer", "inspect", output, {"check_cache": True})
        assert probe["cache_current"] is True
        output.write_text("old")
        probe = cli_eval._run_isolated_prepare(python, "isolated_preparer", "inspect", output, {"check_cache": True})
        assert probe["cache_current"] is False

    def test_absolute_script_executes_in_child_without_importing_in_parent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        script = tmp_path / "external scripts" / "prepare.py"
        script.parent.mkdir()
        output = tmp_path / "data.jsonl"
        script.write_text(
            "from pathlib import Path\ndef prepare(target):\n    p=Path(target); p.write_text('prepared'); return p\n"
        )
        before = set(sys.modules)
        probe = cli_eval._run_isolated_prepare(
            Path(sys.executable), str(script), "inspect", output, {"check_cache": False}
        )
        assert probe == {"has_prepare": True, "cache_current": False}
        result = cli_eval._run_isolated_prepare(
            Path(sys.executable), str(script), "prepare", output, {"target": str(output)}
        )
        assert result == {"output": str(output)}
        assert output.read_text() == "prepared"
        assert not any(name.startswith("_gym_preparer_") for name in set(sys.modules) - before)

    def test_preparer_children_use_the_locked_python_and_environment(self, tmp_path, monkeypatch):
        import os
        import venv

        isolated = tmp_path / "isolated venv"
        venv.EnvBuilder(with_pip=False).create(isolated)
        script = tmp_path / "prepare.py"
        script.write_text(
            "from pathlib import Path\nimport os, subprocess\n"
            "def prepare(target):\n"
            "    child = subprocess.check_output(['python', '-c', 'import sys; print(sys.prefix)'], text=True).strip()\n"
            "    assert child == os.environ['NEMO_GYM_PREPARE_VENV']\n"
            "    assert os.environ['VIRTUAL_ENV'] == child\n"
            "    output = Path(target); output.write_text(child); return output\n"
        )
        output = tmp_path / "prepared.jsonl"
        parent_path = os.environ["PATH"]
        monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "parent venv"))
        result = cli_eval._run_isolated_prepare(
            isolated / "bin/python", str(script), "prepare", output, {"target": str(output)}
        )
        assert result == {"output": str(output)}
        assert output.read_text() == str(isolated)
        assert os.environ["PATH"] == parent_path
        assert os.environ["VIRTUAL_ENV"] == str(tmp_path / "parent venv")

    def test_absolute_script_loads_and_executes_without_a_prepare_project(self, tmp_path):
        from nemo_gym.cli._prepare_worker import load_preparer

        output = tmp_path / "data.jsonl"
        script = tmp_path / "prepare.py"
        script.write_text(
            "from pathlib import Path\n"
            "def prepare():\n"
            f"    p=Path({str(output)!r}); p.write_text('prepared'); return p\n"
        )
        module = load_preparer(str(script))
        try:
            benchmark = self._benchmark()
            benchmark.dataset.jsonl_fpath = output
            cli_eval._multiprocess_benchmark_prepare_fn((benchmark, str(script), {}, None))
            assert output.read_text() == "prepared"
        finally:
            sys.modules.pop(module.__name__, None)

    def test_worker_rejects_wrong_output_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "wrong_preparer.py").write_text(
            "from pathlib import Path\ndef prepare(): return Path('wrong.jsonl')\n"
        )
        with pytest.raises(subprocess.CalledProcessError):
            cli_eval._run_isolated_prepare(
                Path(sys.executable), "wrong_preparer", "prepare", tmp_path / "expected", {}
            )


class TestEvalRunNoServeWithoutHeadServer:
    """`gym eval run --no-serve` collects against servers that are already running, so nothing listening
    on the head server port is the user's most likely mistake (#2687). Driven through the real `main()`
    so the whole path is exercised: the config is parsed, the rows are materialized, and the head server
    fetch is the only thing faked (it refuses the connection, exactly as a closed port does)."""

    def _arrange_no_serve_run_against_a_closed_port(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        input_path = tmp_path / "input.jsonl"
        input_path.write_text(json.dumps({"responses_create_params": {"input": "hi"}}) + "\n")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "gym",
                "eval",
                "run",
                "--no-serve",
                "--agent",
                "simple_agent",
                "-i",
                str(input_path),
                "-o",
                str(tmp_path / "out.jsonl"),
            ],
        )
        # Deterministic config: env.yaml is read from the first of NEMO_GYM_EXTRA_ROOTS, cwd, and the install
        # root that has one, and a set NEMO_GYM_CONFIG_DICT skips parsing (and the head server fetch) entirely.
        # So clear both, shadow any developer env.yaml with an empty one in cwd, and force a fresh parse.
        monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        (tmp_path / "env.yaml").write_text("")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)
        # Rich soft-wraps at 80 columns when stdout is not a TTY, which would split the message mid-sentence.
        monkeypatch.setenv("COLUMNS", "1000")

        # `requests.exceptions.ConnectionError` is what `ServerClient.load_from_global_config` catches; it is
        # unrelated to the builtin `ConnectionError`, which would sail straight through.
        def refuse(*args, **kwargs):
            raise requests.exceptions.ConnectionError("[Errno 61] Connection refused")

        monkeypatch.setattr(nemo_gym.server_utils.requests, "get", refuse)

    def test_exits_one_with_a_single_error_line_and_no_traceback(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, capsys
    ) -> None:
        self._arrange_no_serve_run_against_a_closed_port(tmp_path, monkeypatch)

        with pytest.raises(SystemExit) as exit_info:
            main()

        assert exit_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Could not connect to the head server at http://127.0.0.1:11000." in captured.out
        assert "Start it with: `gym env start`." in captured.out
        assert "Traceback" not in captured.out + captured.err
        assert "ValueError" not in captured.out + captured.err


class TestLockedPrepareSearchRoots:
    def test_project_resolves_from_extra_root(self, tmp_path, monkeypatch):
        from nemo_gym.cli import _locked_env

        project = tmp_path / "plugin/benchmarks/custom"
        project.mkdir(parents=True)
        monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(tmp_path / "plugin"))
        monkeypatch.chdir(tmp_path)
        calls = []
        monkeypatch.setattr(_locked_env, "sync_environment", lambda *args: calls.append(args))
        benchmark = TestPrepareDependencies._benchmark(Path("benchmarks/custom"))
        cli_eval._prepare_environment(benchmark)
        assert calls[0][0] == project

    @pytest.mark.parametrize("absolute", [False, True])
    def test_worker_finds_plugin_script_outside_cwd(self, tmp_path, monkeypatch, absolute):
        plugin = tmp_path / "plugin"
        script = plugin / "custom_preparer.py"
        plugin.mkdir()
        script.write_text("from pathlib import Path\ndef prepare(target):\n    return Path(target)\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(plugin))
        reference = str(script) if absolute else "custom_preparer"
        result = cli_eval._run_isolated_prepare(
            Path(sys.executable), reference, "inspect", tmp_path / "data.jsonl", {"check_cache": False}
        )
        assert result == {"has_prepare": True, "cache_current": False}
