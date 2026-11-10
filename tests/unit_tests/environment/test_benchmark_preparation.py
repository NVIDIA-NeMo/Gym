# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import DictConfig

from nemo_gym.config.config_types import ConfigError
from nemo_gym.environment import benchmark_preparation


class TestPrepareDependencies:
    @staticmethod
    def _benchmark(project=None, dependencies=()):
        from nemo_gym.config.config_types import BenchmarkDatasetConfig

        dataset = BenchmarkDatasetConfig(
            name="b",
            type="benchmark",
            jsonl_fpath=Path("data.jsonl"),
            prepare_script=Path("prepare.py"),
            prepare_project=project,
            prepare_dependencies=list(dependencies),
        )
        return SimpleNamespace(name="b", dataset=dataset)

    def test_declared_dependencies_are_installed_into_the_running_interpreter(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(benchmark_preparation.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
        monkeypatch.setattr(benchmark_preparation.site, "addsitedir", lambda path: None)

        benchmark_preparation._install_prepare_dependencies(
            self._benchmark(dependencies=["-e", "benchmarks/automationbench"])
        )

        assert len(calls) == 1
        # --python sys.executable: installing into whatever interpreter happens to
        # be on PATH would put the package somewhere this process cannot import.
        assert calls[0] == ["uv", "pip", "install", "--python", sys.executable, "-e", "benchmarks/automationbench"]

    def test_an_editable_install_is_made_importable_without_a_restart(self, monkeypatch) -> None:
        """An editable install only drops a .pth file, which `site` reads at startup."""
        added = []
        monkeypatch.setattr(benchmark_preparation.subprocess, "run", lambda cmd, **kw: None)
        monkeypatch.setattr(benchmark_preparation.site, "addsitedir", lambda path: added.append(path))

        benchmark_preparation._install_prepare_dependencies(
            self._benchmark(dependencies=["-e", "benchmarks/automationbench"])
        )

        assert added == [benchmark_preparation.sysconfig.get_paths()["purelib"]]

    def test_nothing_runs_when_no_dependencies_are_declared(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(benchmark_preparation.subprocess, "run", lambda cmd, **kw: calls.append(cmd))

        benchmark_preparation._install_prepare_dependencies(self._benchmark(dependencies=[]))

        assert calls == []

    def test_a_failed_install_is_reported_against_the_dataset(self, monkeypatch) -> None:
        def boom(cmd, **kw):
            raise benchmark_preparation.subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(benchmark_preparation.subprocess, "run", boom)

        with pytest.raises(ConfigError, match="prepare_dependencies for dataset 'b'"):
            benchmark_preparation._install_prepare_dependencies(self._benchmark(dependencies=["nope"]))

    def test_declared_project_syncs_without_mutating_parent(self, tmp_path, monkeypatch):
        from nemo_gym.cli import _locked_env

        calls = []
        monkeypatch.setattr(_locked_env, "sync_environment", lambda *a: calls.append(a))
        python = benchmark_preparation._prepare_environment(self._benchmark(tmp_path))
        assert calls == [(tmp_path, tmp_path / ".venv", sys.executable, [], [])]
        assert python == tmp_path / ".venv/bin/python"

    def test_custom_venv_root_does_not_write_to_component(self, tmp_path, monkeypatch):
        from nemo_gym.cli import _locked_env

        project = tmp_path / "installed/benchmarks/example"
        monkeypatch.setattr(benchmark_preparation, "PARENT_DIR", tmp_path / "installed")
        calls = []
        monkeypatch.setattr(_locked_env, "sync_environment", lambda *a: calls.append(a))
        python = benchmark_preparation._prepare_environment(self._benchmark(project), tmp_path / "writable")
        venv = tmp_path / "writable/preparation/benchmarks/example/.venv"
        assert python == venv / "bin/python"
        assert calls == [(project, venv, sys.executable, [], [])]
        assert not project.exists()

    @pytest.mark.parametrize("dependencies", [["tqdm"], ["-e", "benchmarks/automationbench"], ["-r", "deps.txt"]])
    def test_dependency_arguments_retain_install_and_import_behavior(self, dependencies, monkeypatch):
        calls = []
        monkeypatch.setattr(benchmark_preparation.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
        monkeypatch.setattr(benchmark_preparation.importlib, "invalidate_caches", lambda: calls.append("invalidate"))
        monkeypatch.setattr(benchmark_preparation.site, "addsitedir", lambda path: calls.append(("site", path)))
        assert benchmark_preparation._prepare_environment(self._benchmark(dependencies=dependencies)) is None
        assert calls == [
            ((["uv", "pip", "install", "--python", sys.executable, *dependencies],), {"check": True}),
            "invalidate",
            ("site", benchmark_preparation.sysconfig.get_paths()["purelib"]),
        ]

    def test_non_benchmark_config_preserves_dependency_arguments(self, monkeypatch):
        from nemo_gym.config.config_types import DatasetConfig

        dataset = DatasetConfig(
            name="training",
            type="train",
            jsonl_fpath="data.jsonl",
            license="MIT",
            prepare_script=Path("prepare.py"),
            prepare_dependencies=["tqdm"],
        )
        calls = []
        monkeypatch.setattr(
            benchmark_preparation, "_install_prepare_dependencies", lambda config: calls.append(config)
        )
        benchmark = SimpleNamespace(name="training", dataset=dataset)
        assert benchmark_preparation._prepare_environment(benchmark) is None
        assert calls == [benchmark]
        assert dataset.model_dump()["prepare_dependencies"] == ["tqdm"]

    def test_dependency_arguments_take_precedence_over_project(self, tmp_path, monkeypatch):
        from nemo_gym.cli import _locked_env

        monkeypatch.setattr(_locked_env, "sync_environment", lambda *a: pytest.fail("unexpected sync"))
        calls = []
        benchmark = self._benchmark(tmp_path, ["tqdm"])
        monkeypatch.setattr(
            benchmark_preparation, "_install_prepare_dependencies", lambda config: calls.append(config)
        )
        assert benchmark_preparation._prepare_environment(benchmark) is None
        assert calls == [benchmark]

    @pytest.mark.parametrize("error", [FileNotFoundError("uv"), subprocess.CalledProcessError(1, ["uv"])])
    def test_dependency_install_failure_names_dataset(self, error, monkeypatch):
        def fail(*args, **kwargs):
            raise error

        monkeypatch.setattr(benchmark_preparation.subprocess, "run", fail)
        monkeypatch.setattr(
            benchmark_preparation.site, "addsitedir", lambda *a: pytest.fail("unexpected import update")
        )
        with pytest.raises(ConfigError, match="prepare_dependencies for dataset 'b'") as exc:
            benchmark_preparation._prepare_environment(self._benchmark(dependencies=["tqdm"]))
        assert exc.value.__cause__ is error

    def test_dependency_helper_without_arguments_does_not_install(self, monkeypatch):
        monkeypatch.setattr(
            benchmark_preparation.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected install")
        )
        assert benchmark_preparation._install_prepare_dependencies(self._benchmark()) is None

    def test_nothing_runs_without_project(self, monkeypatch):
        from nemo_gym.cli import _locked_env

        monkeypatch.setattr(_locked_env, "sync_environment", lambda *a: pytest.fail("unexpected sync"))
        assert benchmark_preparation._prepare_environment(self._benchmark()) is None

    def test_failed_sync_names_dataset(self, tmp_path, monkeypatch):
        from nemo_gym.cli import _locked_env

        def fail(*args):
            raise RuntimeError("stale lock")

        monkeypatch.setattr(_locked_env, "sync_environment", fail)
        with pytest.raises(ConfigError, match="locked prepare_project for dataset 'b': stale lock"):
            benchmark_preparation._prepare_environment(self._benchmark(tmp_path))

    def test_worker_imports_preparer_only_in_child_and_checks_cache(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        output = tmp_path / "data.jsonl"
        (tmp_path / "isolated_preparer.py").write_text(
            "from pathlib import Path\n"
            "def prepare(target):\n    p=Path(target); p.write_text('data'); return p\n"
            "def is_prepared_data_current(path):\n    return path.read_text() == 'data'\n"
        )
        python = Path(sys.executable)
        probe = benchmark_preparation._run_isolated_prepare(
            python, "isolated_preparer", "inspect", output, {"check_cache": True}
        )
        assert probe == {"has_prepare": True, "cache_current": False}
        result = benchmark_preparation._run_isolated_prepare(
            python, "isolated_preparer", "prepare", output, {"target": str(output)}
        )
        assert result == {"output": str(output)}
        assert output.read_text() == "data"
        assert "isolated_preparer" not in sys.modules
        probe = benchmark_preparation._run_isolated_prepare(
            python, "isolated_preparer", "inspect", output, {"check_cache": True}
        )
        assert probe["cache_current"] is True
        output.write_text("old")
        probe = benchmark_preparation._run_isolated_prepare(
            python, "isolated_preparer", "inspect", output, {"check_cache": True}
        )
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
        probe = benchmark_preparation._run_isolated_prepare(
            Path(sys.executable), str(script), "inspect", output, {"check_cache": False}
        )
        assert probe == {"has_prepare": True, "cache_current": False}
        result = benchmark_preparation._run_isolated_prepare(
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
        result = benchmark_preparation._run_isolated_prepare(
            isolated / "bin/python", str(script), "prepare", output, {"target": str(output)}
        )
        assert result == {"output": str(output)}
        assert output.read_text() == str(isolated)
        assert os.environ["PATH"] == parent_path
        assert os.environ["VIRTUAL_ENV"] == str(tmp_path / "parent venv")

    def test_absolute_script_loads_and_executes_without_a_prepare_project(self, tmp_path):
        from nemo_gym.server_runtime.supervisor._prepare_worker import load_preparer

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
            benchmark_preparation._multiprocess_benchmark_prepare_fn((benchmark, str(script), {}, None))
            assert output.read_text() == "prepared"
        finally:
            sys.modules.pop(module.__name__, None)

    def test_worker_directory_does_not_shadow_preparer_packages(self, tmp_path):
        # nemo_gym/cli/agents.py sits next to the worker and must not hide the `agents` package of openai-agents.
        package = tmp_path / "site" / "agents"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "function_schema.py").write_text("")
        output = tmp_path / "prepared.jsonl"
        (tmp_path / "shadow_preparer.py").write_text(
            "from pathlib import Path\n"
            "def prepare():\n"
            "    import agents.function_schema\n"
            f"    p = Path({str(output)!r}); p.write_text('ok'); return p\n"
        )
        worker = Path(importlib.util.find_spec("nemo_gym.server_runtime.supervisor._prepare_worker").origin)
        (tmp_path / "args.json").write_text("{}")
        subprocess.run(
            [
                sys.executable,
                str(worker),
                "--root",
                str(tmp_path),
                "--module",
                "shadow_preparer",
                "--action",
                "prepare",
                "--expected",
                str(output),
                "--args-file",
                str(tmp_path / "args.json"),
                "--result",
                str(tmp_path / "r.json"),
            ],
            check=True,
            env=os.environ | {"PYTHONPATH": str(tmp_path / "site")},
        )
        assert output.read_text() == "ok"

    def test_worker_rejects_wrong_output_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "wrong_preparer.py").write_text(
            "from pathlib import Path\ndef prepare(): return Path('wrong.jsonl')\n"
        )
        with pytest.raises(benchmark_preparation.PreparationFailedError) as stopped:
            benchmark_preparation._run_isolated_prepare(
                Path(sys.executable), "wrong_preparer", "prepare", tmp_path / "expected", {}
            )
        assert isinstance(stopped.value.__cause__, subprocess.CalledProcessError)


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
        benchmark_preparation._prepare_environment(benchmark)
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
        result = benchmark_preparation._run_isolated_prepare(
            Path(sys.executable), reference, "inspect", tmp_path / "data.jsonl", {"check_cache": False}
        )
        assert result == {"has_prepare": True, "cache_current": False}


def test_isolated_prepare_receives_hf_config_and_structured_arguments(tmp_path, monkeypatch):
    """Use the real config reader in a child with OmegaConf values and path arguments."""
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "configured_preparer.py"
    script.write_text(
        "from pathlib import Path\n"
        "from nemo_gym.config.global_config import get_hf_token\n"
        "def prepare(target, values, settings):\n"
        "    assert get_hf_token() == 'fixture-token'\n"
        "    assert values == [2, 3]\n"
        "    assert settings == {'size': 2}\n"
        "    path = Path(target)\n"
        "    path.write_text('prepared')\n"
        "    return path\n"
    )
    output = tmp_path / "prepared.jsonl"
    arguments = DictConfig(
        {"target": output, "values": ["${settings.size}", 3], "settings": {"size": 2}}, flags={"allow_objects": True}
    )
    result = benchmark_preparation._run_isolated_prepare(
        Path(sys.executable),
        str(script),
        "prepare",
        output,
        arguments,
        config=DictConfig({"hf_token": "fixture-token"}),
    )
    assert result == {"output": str(output)}
    assert output.read_text() == "prepared"


def test_a_failed_isolated_preparer_is_a_typed_terminal_stop(tmp_path, monkeypatch) -> None:
    import subprocess

    from nemo_gym.environment import benchmark_preparation as preparation

    script = tmp_path / "prepare.py"
    script.write_text("")

    def failing_worker(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(preparation.subprocess, "run", failing_worker)

    with pytest.raises(preparation.PreparationFailedError) as stopped:
        preparation._run_isolated_prepare(
            tmp_path / "python", str(script), "prepare", tmp_path / "out.jsonl", {}, config={}
        )

    assert str(stopped.value).startswith("EVAL STOPPED: data preparation failed for ")
    assert (stopped.value.failure_kind, stopped.value.stop_exit_code) == ("environment_prepare_failed", 76)


def test_the_cli_exits_76_and_leaves_the_kind_for_the_chain_guard(tmp_path, monkeypatch) -> None:
    from nemo_gym.cli.utils import exit_cleanly_on_config_error
    from nemo_gym.environment.benchmark_preparation import PreparationFailedError
    from nemo_gym.orchestration.executors.resume_script import DRIVER_FAILURE_ENV

    failure_file = tmp_path / ".gym_driver_failure.100"
    monkeypatch.setenv(DRIVER_FAILURE_ENV, str(failure_file))

    @exit_cleanly_on_config_error
    def prepare() -> None:
        raise PreparationFailedError("benchmarks.example.prepare", "prepare", "exit status 1")

    with pytest.raises(SystemExit) as exited:
        prepare()

    assert exited.value.code == 76
    assert failure_file.read_text() == "environment_prepare_failed\n"


def test_an_in_process_preparer_that_raises_is_a_typed_terminal_stop(tmp_path) -> None:
    script = tmp_path / "prepare.py"
    script.write_text("def prepare(**kwargs):\n    raise ValueError('bad row')\n")
    benchmark = SimpleNamespace(name="demo", dataset=SimpleNamespace(jsonl_fpath=str(tmp_path / "out.jsonl")))

    with pytest.raises(benchmark_preparation.PreparationFailedError) as stopped:
        benchmark_preparation._multiprocess_benchmark_prepare_fn((benchmark, str(script), {}, None))

    assert "ValueError: bad row" in str(stopped.value)
    assert stopped.value.stop_exit_code == 76


def test_a_preparation_stop_survives_pickling_across_a_worker_boundary() -> None:
    import pickle

    original = benchmark_preparation.PreparationFailedError("benchmarks.example.prepare", "prepare", "KeyError: 'foo'")

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is benchmark_preparation.PreparationFailedError
    assert str(restored) == str(original)
    assert (restored.failure_kind, restored.stop_exit_code) == ("environment_prepare_failed", 76)


def test_a_failing_worker_in_the_parallel_pool_stops_with_76_instead_of_hanging(tmp_path, monkeypatch) -> None:
    import threading

    from nemo_gym.cli.utils import exit_cleanly_on_config_error
    from nemo_gym.config.config_types import BenchmarkDatasetConfig

    benchmarks = []
    for name in ("first", "second"):
        script = tmp_path / f"{name}_prepare.py"
        script.write_text(
            """
def prepare(**kwargs):
    raise ValueError("bad row")
"""
        )
        dataset = BenchmarkDatasetConfig(
            name=name, type="benchmark", jsonl_fpath=tmp_path / f"{name}.jsonl", prepare_script=script
        )
        benchmarks.append(SimpleNamespace(name=name, dataset=dataset))
    monkeypatch.setenv("GYM_DRIVER_FAILURE_FILE", str(tmp_path / "driver_failure"))
    settings = benchmark_preparation.PrepareBenchmarkConfig(uv_venv_dir=tmp_path, num_prepare_benchmark_processes=2)
    outcome: dict = {}

    @exit_cleanly_on_config_error
    def prepare() -> None:
        benchmark_preparation.prepare_datasets(benchmarks, settings)

    def run() -> None:
        try:
            prepare()
        except SystemExit as exited:
            outcome["code"] = exited.code

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=60)

    assert not worker.is_alive(), "the parent waits forever for a result the pool could not deliver"
    assert outcome["code"] == 76
