# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.gdpval.hsg.checkpoint_e2e import rollout_runtime as runtime


def _write(path: Path, text: str = "fixture\n", *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)
    return path


def test_node_local_path_accepts_a_file_and_resolves_internal_symlinks(tmp_path: Path) -> None:
    local = tmp_path / "local"
    target = _write(local / "python" / "bin" / "python")
    link = local / "venv-python"
    link.symlink_to(target)

    assert runtime.assert_node_local(target, local_root=local) == target.resolve()
    assert runtime.assert_node_local(link, local_root=local) == target.resolve()


@pytest.mark.parametrize("escape", ["file", "parent", "prefix-lookalike"])
def test_node_local_path_rejects_resolved_shared_files(tmp_path: Path, escape: str) -> None:
    local = tmp_path / "local"
    local.mkdir()
    shared_file = _write(tmp_path / "local-shared" / "bin" / "python")
    if escape == "file":
        candidate = local / "python"
        candidate.symlink_to(shared_file)
    elif escape == "parent":
        (local / "python-home").symlink_to(shared_file.parent, target_is_directory=True)
        candidate = local / "python-home" / shared_file.name
    else:
        candidate = shared_file

    with pytest.raises(ValueError):
        runtime.assert_node_local(candidate, local_root=local)


def test_node_local_path_rejects_missing_executables(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()

    with pytest.raises((ValueError, FileNotFoundError)):
        runtime.assert_node_local(local / "missing-python", local_root=local)


@pytest.fixture
def local_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    local = tmp_path / "node-local"
    gym = local / "gym"
    venvs = local / "component-venvs"
    gym.mkdir(parents=True)
    venvs.mkdir()
    python = _write(local / "python" / "bin" / "python", executable=True)
    venv_python = gym / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(python)
    stdlib = local / "python" / "lib"
    stdlib.mkdir()
    module = SimpleNamespace(__file__=str(_write(gym / "nemo_gym" / "__init__.py")))
    interpreter = SimpleNamespace(
        executable=str(venv_python),
        _base_executable=str(python),
        prefix=str(gym / ".venv"),
        base_prefix=str(python.parent.parent),
        path=[str(gym), str(stdlib), str(stdlib / "python313.zip")],
        meta_path=[],
        path_hooks=[],
        modules={"nemo_gym": module, "builtins": SimpleNamespace(), "frozen": SimpleNamespace(__file__="<frozen>")},
    )
    monkeypatch.setattr(runtime, "sys", interpreter)
    for name in (
        "TMPDIR",
        "RAY_TMPDIR",
        "UV_CACHE_DIR",
        "UV_PYTHON_INSTALL_DIR",
        "XDG_CACHE_HOME",
        "APPTAINER_TMPDIR",
        "APPTAINER_CACHEDIR",
        "PYTHONPYCACHEPREFIX",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "GDPVAL_REF_FILES_DIR",
    ):
        directory = local / "cache" / name.lower()
        directory.mkdir(parents=True)
        monkeypatch.setenv(name, str(directory))
    uv = _write(local / "bin" / "uv", executable=True)
    apptainer = _write(local / "bin" / "apptainer", executable=True)
    sif = _write(local / "assets" / "agent.sif")
    monkeypatch.setenv("MARS_UV", str(uv))
    monkeypatch.setenv("PATH", str(uv.parent))
    monkeypatch.setenv("APPTAINER_BIN", str(apptainer.parent))
    monkeypatch.setenv("GDPVAL_CONTAINER_PATH", str(sif))
    monkeypatch.delenv("NEMO_GYM_EXTRA_ROOTS", raising=False)
    return SimpleNamespace(root=local, gym=gym, venvs=venvs, interpreter=interpreter, python=python)


def _verify(fixture: SimpleNamespace) -> dict:
    return runtime.verify_runtime(fixture.gym, fixture.venvs, local_root=fixture.root)


def test_runtime_accepts_local_python_modules_caches_and_absent_stdlib_zip(local_runtime: SimpleNamespace) -> None:
    report = _verify(local_runtime)

    assert report["executable"] == str(local_runtime.python.resolve())
    assert report["UV_CACHE_DIR"] == os.environ["UV_CACHE_DIR"]
    assert report["GDPVAL_CONTAINER_PATH"] == os.environ["GDPVAL_CONTAINER_PATH"]


def test_runtime_audit_does_not_trigger_lazy_module_imports(local_runtime: SimpleNamespace) -> None:
    class LazyModule:
        def __getattr__(self, name: str):
            raise AssertionError(f"audit triggered lazy lookup: {name}")

    local_runtime.interpreter.modules["lazy_module"] = LazyModule()
    local_runtime.interpreter.modules["pending_import"] = None

    _verify(local_runtime)


@pytest.fixture
def editable_finder(local_runtime: SimpleNamespace) -> ModuleType:
    name = "__editable___nemo_gym_0_7_0rc0_finder"
    finder = ModuleType(name)
    finder.__file__ = str(_write(local_runtime.gym / ".venv/lib/site-packages" / f"{name}.py"))
    finder.MAPPING = {"nemo_gym": str(local_runtime.gym / "nemo_gym")}
    finder.NAMESPACES = {"fixture_namespace": [str(local_runtime.gym)], "virtual_namespace": []}
    finder.PATH_PLACEHOLDER = "__editable__.nemo_gym-0.7.0rc0.finder.__path_hook__"
    # The registered classmethod and its module-level tables are setuptools'
    # protocol. No placeholder is a real directory or an importable module.
    exec(
        "class _EditableFinder: pass\n"
        "class _EditableNamespaceFinder:\n"
        "    @classmethod\n"
        "    def _path_hook(cls, path):\n"
        "        if path == PATH_PLACEHOLDER: return cls\n"
        "        raise ImportError(path)\n",
        vars(finder),
    )
    interpreter = local_runtime.interpreter
    interpreter.modules[name] = finder
    interpreter.meta_path.append(finder._EditableFinder)
    interpreter.path_hooks.append(finder._EditableNamespaceFinder._path_hook)
    interpreter.path.append(finder.PATH_PLACEHOLDER)
    interpreter.modules["fixture_namespace"] = SimpleNamespace(
        __path__=[str(local_runtime.gym), finder.PATH_PLACEHOLDER]
    )
    return finder


def test_runtime_accepts_registered_editable_namespace_placeholders(
    local_runtime: SimpleNamespace, editable_finder: ModuleType
) -> None:
    assert not Path(editable_finder.PATH_PLACEHOLDER).exists()
    _verify(local_runtime)


@pytest.mark.parametrize("registration", ["module", "meta_path", "path_hooks"])
def test_runtime_rejects_unregistered_editable_lookalikes(
    local_runtime: SimpleNamespace, editable_finder: ModuleType, registration: str
) -> None:
    if registration == "module":
        del local_runtime.interpreter.modules[editable_finder.__name__]
    else:
        getattr(local_runtime.interpreter, registration).clear()

    with pytest.raises(FileNotFoundError):
        _verify(local_runtime)


@pytest.mark.parametrize("location", ["sys.path", "namespace"])
def test_runtime_does_not_ignore_unrecognized_synthetic_paths(
    local_runtime: SimpleNamespace, editable_finder: ModuleType, location: str
) -> None:
    unknown = editable_finder.PATH_PLACEHOLDER.replace("0.7.0rc0", "other-package")
    if location == "sys.path":
        local_runtime.interpreter.path.append(unknown)
    else:
        local_runtime.interpreter.modules["fixture_namespace"].__path__.append(unknown)

    with pytest.raises(FileNotFoundError):
        _verify(local_runtime)


@pytest.mark.parametrize("location", ["finder_file", "mapping", "namespace", "mapping_symlink"])
def test_runtime_rejects_editable_finder_shared_roots_even_before_the_module_is_imported(
    local_runtime: SimpleNamespace, editable_finder: ModuleType, tmp_path: Path, location: str
) -> None:
    shared = _write(tmp_path / "shared" / "unimported" / "__init__.py")
    if location == "finder_file":
        local_finder = Path(editable_finder.__file__)
        local_finder.unlink()
        local_finder.symlink_to(shared)
    elif location == "namespace":
        editable_finder.NAMESPACES["unimported"] = [str(shared.parent)]
    else:
        source = shared.parent
        if location == "mapping_symlink":
            source = local_runtime.gym / "unimported"
            source.symlink_to(shared.parent, target_is_directory=True)
        editable_finder.MAPPING["unimported"] = str(source)

    with pytest.raises(ValueError, match="outside node-local storage"):
        _verify(local_runtime)


def test_runtime_does_not_allow_a_finder_to_hide_a_real_shared_path(
    local_runtime: SimpleNamespace, editable_finder: ModuleType, tmp_path: Path
) -> None:
    shared = _write(tmp_path / "shared" / "module.py")
    old = editable_finder.PATH_PLACEHOLDER
    editable_finder.PATH_PLACEHOLDER = str(shared.parent)
    local_runtime.interpreter.path.remove(old)
    local_runtime.interpreter.path.append(str(shared.parent))

    with pytest.raises(ValueError, match="invalid editable path placeholder"):
        _verify(local_runtime)


@pytest.mark.parametrize("attribute", ["executable", "_base_executable", "prefix", "base_prefix"])
def test_runtime_rejects_shared_python_even_when_the_venv_is_local(
    local_runtime: SimpleNamespace, tmp_path: Path, attribute: str
) -> None:
    _verify(local_runtime)
    shared = _write(tmp_path / "shared" / "python")
    setattr(local_runtime.interpreter, attribute, str(shared))

    with pytest.raises(ValueError, match="outside node-local storage"):
        _verify(local_runtime)


@pytest.mark.parametrize(
    "variable",
    [
        "TMPDIR",
        "RAY_TMPDIR",
        "UV_CACHE_DIR",
        "UV_PYTHON_INSTALL_DIR",
        "XDG_CACHE_HOME",
        "APPTAINER_TMPDIR",
        "APPTAINER_CACHEDIR",
        "PYTHONPYCACHEPREFIX",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "GDPVAL_REF_FILES_DIR",
        "GDPVAL_CONTAINER_PATH",
    ],
)
def test_runtime_rejects_shared_caches_and_agent_container(
    local_runtime: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    _verify(local_runtime)
    shared = _write(tmp_path / "shared" / "asset")
    monkeypatch.setenv(variable, str(shared))

    with pytest.raises(ValueError, match="outside node-local storage"):
        _verify(local_runtime)


def test_runtime_rejects_missing_cache_contract(
    local_runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UV_CACHE_DIR")

    with pytest.raises(ValueError, match="UV_CACHE_DIR"):
        _verify(local_runtime)


@pytest.mark.parametrize("source", ["sys.path", "extra-roots", "loaded-module", "namespace-module"])
def test_runtime_rejects_shared_module_resolution(
    local_runtime: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    _verify(local_runtime)
    shared = _write(tmp_path / "shared" / "module.py")
    if source == "sys.path":
        local_runtime.interpreter.path.append(str(shared.parent))
    elif source == "extra-roots":
        monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", f"{local_runtime.gym}{os.pathsep}{shared.parent}")
    elif source == "loaded-module":
        local_runtime.interpreter.modules["shared_module"] = SimpleNamespace(__file__=str(shared))
    else:
        local_runtime.interpreter.modules["shared_namespace"] = SimpleNamespace(__path__=[str(shared.parent)])

    with pytest.raises(ValueError, match="outside node-local storage"):
        _verify(local_runtime)


def test_runtime_rejects_an_empty_import_path_when_cwd_is_shared(
    local_runtime: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    local_runtime.interpreter.path.append("")

    with pytest.raises(ValueError, match="outside node-local storage"):
        _verify(local_runtime)


@pytest.mark.parametrize("executable", ["uv", "apptainer", "agent.sif"])
def test_runtime_rejects_local_execution_symlinks_back_to_shared_assets(
    local_runtime: SimpleNamespace, tmp_path: Path, executable: str
) -> None:
    _verify(local_runtime)
    target = _write(tmp_path / "shared" / executable, executable=True)
    directory = "assets" if executable == "agent.sif" else "bin"
    local = local_runtime.root / directory / executable
    local.unlink()
    local.symlink_to(target)

    with pytest.raises(ValueError, match="outside node-local storage"):
        _verify(local_runtime)


def test_runtime_requires_path_to_select_the_staged_uv(
    local_runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    alternate = _write(local_runtime.root / "alternate" / "uv", executable=True)
    monkeypatch.setenv("PATH", str(alternate.parent))

    with pytest.raises(ValueError, match="uv on PATH differs"):
        _verify(local_runtime)


@pytest.mark.parametrize("escape", ["venv-directory", "venv-python"])
def test_runtime_rejects_component_venvs_that_resolve_to_shared_storage(
    local_runtime: SimpleNamespace, tmp_path: Path, escape: str
) -> None:
    _verify(local_runtime)
    shared_python = _write(tmp_path / "shared" / ".venv" / "bin" / "python", executable=True)
    component = local_runtime.venvs / "responses_api_agents" / "stirrup_agent"
    component.mkdir(parents=True)
    venv = component / ".venv"
    if escape == "venv-directory":
        venv.symlink_to(shared_python.parent.parent, target_is_directory=True)
    else:
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").symlink_to(shared_python)

    with pytest.raises(ValueError, match="outside node-local storage"):
        _verify(local_runtime)


@pytest.fixture
def component_installer(local_runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    # Keep Gym's real setup command and parser. Only replace external package
    # installation and the child probe process with an executable event recorder.
    _write(local_runtime.gym / "pyproject.toml", "[project]\nname = 'fixture-gym'\n")
    components = (
        "responses_api_models/vllm_model",
        "responses_api_models/openai_model",
        "resources_servers/gdpval",
        "responses_api_agents/stirrup_agent",
    )
    for component in components:
        manifest = "pyproject.toml" if component.endswith("vllm_model") else "requirements.txt"
        _write(local_runtime.gym / component / manifest)
    _write(local_runtime.gym / "responses_api_agents/stirrup_agent/overrides.txt")
    log = local_runtime.root / "events.jsonl"
    monkeypatch.setenv("TEST_INSTALL_EVENTS", str(log))
    monkeypatch.setenv("PATH", f"{local_runtime.root}/bin:/usr/bin:/bin")
    _write(
        local_runtime.root / "bin" / "uv",
        f"#!{sys.executable}\n"
        + r"""import json
import os
import sys
from pathlib import Path

tool = Path(sys.argv[0]).name
event = {"tool": tool, "args": sys.argv, "cwd": os.getcwd(),
         "pythonpath": os.environ.get("PYTHONPATH"), "cache": os.environ.get("UV_CACHE_DIR")}
with open(os.environ["TEST_INSTALL_EVENTS"], "a") as output:
    output.write(json.dumps(event) + "\n")
if tool == "uv" and sys.argv[1] == "venv":
    venv = Path(sys.argv[-1])
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.write_text(Path(__file__).read_text())
    python.chmod(0o755)
    (venv / "bin" / "activate").write_text("export VIRTUAL_ENV=/lustre/stale-activation\n")
elif tool == "uv" and sys.argv[1] == "pip":
    raise SystemExit(29 if os.environ.get("TEST_INSTALL_FAIL") == "pip" else 0)
elif tool == "python":
    raise SystemExit(37 if os.environ.get("TEST_INSTALL_FAIL") == "probe" else 0)
else:
    raise SystemExit("unexpected fixture command")
""",
        executable=True,
    )
    local_runtime.events = log
    local_runtime.components = components
    return local_runtime


def test_component_setup_preserves_local_cache_and_gym_parent_pins(local_runtime: SimpleNamespace) -> None:
    cache = os.environ["UV_CACHE_DIR"]

    config = runtime._component_setup_config(local_runtime.venvs)

    assert os.environ["UV_CACHE_DIR"] == cache
    assert config.uv_cache_dir == cache
    assert config.uv_venv_dir == str(local_runtime.venvs)
    assert config.python_version == local_runtime.interpreter.executable
    assert config.uv_pip_set_python is True
    assert config.skip_venv_if_present is False
    assert any(pin.startswith("ray[") and "==" in pin for pin in config.head_server_deps)


@pytest.mark.parametrize("resolver_available", [True, False])
def test_component_setup_uses_gym_installer_and_probes_each_selected_interpreter(
    component_installer: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, resolver_available: bool
) -> None:
    fixture = component_installer
    if not resolver_available:
        from nemo_gym import cli
        from nemo_gym.cli import setup_command

        # Older Gym exports its installer without a public venv resolver. Keep
        # the real command builder while presenting that older public API.
        monkeypatch.setattr(cli, "setup_command", SimpleNamespace(setup_env_command=setup_command.setup_env_command))

    runtime.prepare_components(fixture.gym, fixture.venvs, local_root=fixture.root)

    events = [json.loads(line) for line in fixture.events.read_text().splitlines()]
    assert len(events) == 12
    for index, component in enumerate(fixture.components):
        create, install, probe = events[index * 3 : (index + 1) * 3]
        python = fixture.venvs / component / ".venv/bin/python"
        assert create["tool"] == install["tool"] == "uv"
        assert create["args"][1] == "venv"
        assert create["args"][-1] == str(python.parent.parent)
        assert install["args"][1:3] == ["pip", "install"]
        assert install["args"][install["args"].index("--python") + 1] == str(python)
        assert any(argument.startswith("ray[") and "==" in argument for argument in install["args"])
        assert probe["tool"] == "python"
        assert probe["args"][0] == str(python)
        assert probe["args"][2] == "verify"
        assert probe["args"][-2:] == ["--module", "app"]
        assert probe["cwd"] == str(fixture.gym / component)
        assert probe["pythonpath"] == f"{fixture.gym / component}:{fixture.gym}"
        assert probe["cache"] == os.environ["UV_CACHE_DIR"]
    assert "--override" in events[10]["args"]
    assert "-e ." in events[1]["args"]


@pytest.mark.parametrize("failure", ["pip", "probe"])
def test_failed_component_install_or_probe_cannot_be_reused_as_success(
    component_installer: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    fixture = component_installer
    monkeypatch.setenv("TEST_INSTALL_FAIL", failure)

    with pytest.raises(subprocess.CalledProcessError) as error:
        runtime.prepare_components(fixture.gym, fixture.venvs, local_root=fixture.root)
    assert error.value.returncode == (29 if failure == "pip" else 37)
    first_attempt = fixture.events.read_bytes()
    events = [json.loads(line) for line in first_attempt.splitlines()]
    assert len(events) == (2 if failure == "pip" else 3)
    assert (fixture.venvs / fixture.components[0] / ".venv/bin/python").exists()
    assert not (fixture.venvs / fixture.components[1] / ".venv").exists()

    monkeypatch.delenv("TEST_INSTALL_FAIL")
    with pytest.raises(ValueError, match="existing component environment"):
        runtime.prepare_components(fixture.gym, fixture.venvs, local_root=fixture.root)
    assert fixture.events.read_bytes() == first_attempt


def test_component_setup_rejects_a_stale_later_component_before_any_install(
    component_installer: SimpleNamespace,
) -> None:
    fixture = component_installer
    stale = fixture.venvs / fixture.components[-1] / ".venv"
    stale.mkdir(parents=True)

    with pytest.raises(ValueError, match="existing component environment"):
        runtime.prepare_components(fixture.gym, fixture.venvs, local_root=fixture.root)

    assert not fixture.events.exists()
