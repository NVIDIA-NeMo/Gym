# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise source/dependency binding with real Git repositories and Python imports."""

import copy
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf
from scripts import swe_source_mount as mount


def commit(root):
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "test snapshot",
        ],
        check=True,
    )
    return mount.git(root, "rev-parse", "HEAD").strip()


def repository(root):
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "pyproject.toml").write_text('[project]\nname = "test"\nversion = "1"\n')
    (root / "uv.lock").write_text("version = 1\n")
    (root / "scripts").mkdir()
    (root / "scripts/example.py").write_text("VALUE = 1\n")
    return commit(root)


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / "rl"
    repository(root)
    gym = root / mount.GYM_RELATIVE
    base_gym = repository(gym)
    base_rl = commit(root)
    return root, gym, base_rl, base_gym


def test_compatible_source_reuses_image_and_checks_nested_source(sources):
    root, gym, base_rl, base_gym = sources
    (gym / "scripts/example.py").write_text("VALUE = 2\n")
    current_gym = commit(gym)
    current_rl = commit(root)
    source = mount.snapshot(root, base_rl_commit=base_rl, base_gym_commit=base_gym)
    assert source["repositories"]["."] == {"base_commit": base_rl, "commit": current_rl}
    assert source["repositories"][mount.GYM_RELATIVE]["commit"] == current_gym
    assert source["changed_files"] == [f"{mount.GYM_RELATIVE}/scripts/example.py"]
    image = {"rl_commit": base_rl, "gym_commit": base_gym, "rl_lock_sha256": "rl-lock", "gym_lock_sha256": "gym-lock"}
    manifest = image | {"rl_commit": current_rl, "gym_commit": current_gym, "source_mount": {"base_image": image}}
    original = copy.deepcopy(image)
    mount.check_image_binding(manifest, image)
    assert image == original
    (gym / "scripts/example.py").write_text("VALUE = 3\n")
    with pytest.raises(ValueError, match="Mounted source changed"):
        mount.check_snapshot(root, source)


@pytest.mark.parametrize("name", ["uv.lock", "pyproject.toml", "scripts/build.sh", "scripts/setup.py", "kernel.cu"])
def test_dependency_and_compiled_changes_need_another_runtime(sources, name):
    root, gym, base_rl, base_gym = sources
    (gym / name).write_text("changed dependency or build input\n")
    commit(gym)
    commit(root)
    with pytest.raises(ValueError, match="Dependency/build input changed"):
        mount.snapshot(root, base_rl_commit=base_rl, base_gym_commit=base_gym)


@pytest.mark.parametrize("kind", ["bytes", "mode", "extra-module", "bytecode", "escape", "index", "symlink-directory"])
def test_snapshot_rejects_tampering_and_shadow_modules(sources, kind):
    root, gym, base_rl, base_gym = sources
    source = mount.snapshot(root, base_rl_commit=base_rl, base_gym_commit=base_gym)
    path = gym / "scripts/example.py"
    if kind == "bytes":
        path.write_text("VALUE = 99\n")
    elif kind == "mode":
        path.chmod(0o755)
    elif kind == "extra-module":
        (gym / "scripts/shadow.py").write_text("VALUE = 99\n")
    elif kind == "bytecode":
        (gym / "scripts/example.pyc").write_bytes(b"unchecked bytecode")
    elif kind == "escape":
        path.unlink()
        path.symlink_to("/etc/passwd")
    elif kind == "index":
        subprocess.run(["git", "-C", str(gym), "update-index", "--force-remove", "scripts/example.py"], check=True)
    else:
        external = root.parent / "external"
        external.mkdir()
        (external / "__init__.py").write_text("VALUE = 99\n")
        (gym / "shadow_package").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="Mounted source changed|Untracked executable|symlink escapes|Git inventory"):
        mount.check_snapshot(root, source)


def test_image_binding_preserves_baked_mode_and_rejects_changed_image_or_lock():
    image = {"rl_commit": "base", "gym_commit": "base", "rl_lock_sha256": "rl", "gym_lock_sha256": "gym"}
    mount.check_image_binding(image, image)
    changed = image | {"gym_commit": "candidate"}
    with pytest.raises(ValueError, match="provenance differs"):
        mount.check_image_binding(changed, image)
    candidate = changed | {"source_mount": {"base_image": image}}
    mount.check_image_binding(candidate, image)
    with pytest.raises(ValueError, match="dependency lock differs"):
        mount.check_image_binding(candidate | {"gym_lock_sha256": "changed"}, image)
    with pytest.raises(ValueError, match="base image provenance changed"):
        mount.check_image_binding(candidate, image | {"squashfs_sha256": "different image"})


def test_import_probe_uses_real_interpreter_and_rejects_wrong_source(tmp_path):
    module = tmp_path / "mount_probe_package.py"
    module.write_text("VALUE = 1\n")
    python = Path(sys.prefix) / "bin/python"
    receipt = mount.check_imports(python=python, modules={"mount_probe_package": module}, cwd=tmp_path)
    assert receipt["modules"] == {"mount_probe_package": str(module)}
    with pytest.raises(ValueError, match="Mounted-source import failed"):
        mount.check_imports(python=python, modules={"mount_probe_package": tmp_path / "other.py"}, cwd=tmp_path)
    with pytest.raises(ValueError, match="environment missing"):
        mount.check_imports(python=tmp_path / "missing-python", modules={}, cwd=tmp_path)


@pytest.mark.parametrize("entrypoint", ["check_swe_training_recipe.py", "prepare_swe_single_agent_turn_recipe.py"])
def test_normal_entrypoint_does_not_create_snapshot_bytecode(tmp_path, entrypoint):
    source = Path(__file__).resolve().parents[3]
    for name in (entrypoint, "swe_source_mount.py", "swe_verification_cache.py"):
        (tmp_path / name).write_bytes((source / "scripts" / name).read_bytes())
    env = os.environ | {"PYTHONPATH": str(source)}
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    result = subprocess.run(
        [sys.executable, str(tmp_path / entrypoint), "--help"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.rglob("*.pyc"))


def test_compiled_submodule_revision_change_requires_runtime_rebuild(sources):
    root, _, _, base_gym = sources
    compiled = root / "3rdparty/compiled"
    repository(compiled)
    base_rl = commit(root)
    (compiled / "scripts/example.py").write_text("VALUE = 2\n")
    commit(compiled)
    commit(root)
    with pytest.raises(ValueError, match="Compiled submodule changed"):
        mount.snapshot(root, base_rl_commit=base_rl, base_gym_commit=base_gym)


def test_runtime_orchestration_accepts_rl_typed_dict_and_probes_component_interpreter(tmp_path, monkeypatch):
    from nemo_gym.config.global_config import GlobalConfigDictParser
    from nemo_gym.orchestration import server_venv
    from nemo_gym.utils.venv_setup import SETUP_COMPLETE_MARKER

    root = tmp_path / "source"
    (root / "nemo_rl/environments").mkdir(parents=True)
    (root / "nemo_rl/environments/nemo_gym.py").touch()
    actor = "nemo_rl.environments.nemo_gym.NemoGym"
    registry = ModuleType("nemo_rl.distributed.actor_environments")
    registry.ACTOR_ENVIRONMENTS = {actor: ["nemo_gym"]}
    rl_gym = ModuleType("nemo_rl.environments.nemo_gym")
    # NemoGymConfig is a TypedDict: the real builder returns a plain dictionary.
    rl_gym.build_nemo_gym_config = lambda *args, **kwargs: {"initial_global_config_dict": {}}
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setitem(sys.modules, rl_gym.__name__, rl_gym)
    monkeypatch.setattr(mount, "IMAGE_RL_ROOT", root)
    monkeypatch.setattr(mount.os, "statvfs", lambda _: SimpleNamespace(f_flag=os.ST_RDONLY))
    monkeypatch.setattr(mount.sys, "prefix", "/opt/nemo_rl_venv")
    for key, value in {
        "UV_PROJECT_ENVIRONMENT": "/opt/nemo_rl_venv",
        "NEMO_RL_VENV_DIR": "/opt/ray_venvs",
        "NEMO_GYM_VENV_DIR": "/opt/gym_venvs",
    }.items():
        monkeypatch.setenv(key, value)
    resolved = OmegaConf.create(
        {
            "skip_venv_if_present": True,
            "uv_venv_dir": "/opt/gym_venvs",
            "agent": {"responses_api_agents": {"opencode_agent": {"entrypoint": "app.py"}}},
        }
    )
    monkeypatch.setattr(GlobalConfigDictParser, "parse", lambda *args, **kwargs: resolved)
    environment = tmp_path / "component-venv"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/activate").touch()
    (environment / SETUP_COMPLETE_MARKER).touch()
    monkeypatch.setattr(server_venv, "get_venv_path", lambda *args: environment)
    calls = []

    def probe(**kwargs):
        calls.append(kwargs)
        return {"probe": len(calls)}

    monkeypatch.setattr(mount, "check_imports", probe)
    result = mount.check_runtime_imports(
        OmegaConf.create({"env": {"nemo_gym": {}}, "policy": {"model_name": "model"}})
    )
    assert set(result) == {"driver", actor, "responses_api_agents/opencode_agent"}
    assert calls[1]["python"] == Path("/opt/ray_venvs") / actor / "bin/python"
    assert calls[2]["python"] == environment / "bin/python"
    assert (
        calls[2]["modules"]["responses_api_agents.opencode_agent.app"]
        == root / mount.GYM_RELATIVE / "responses_api_agents/opencode_agent/app.py"
    )


def test_actor_module_can_be_a_package(tmp_path):
    package = tmp_path / "nemo_rl/algorithms/async_utils"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package exports the Ray actor\n")
    assert mount.module_source(tmp_path, "nemo_rl.algorithms.async_utils") == package / "__init__.py"
    with pytest.raises(ValueError, match="No pinned source"):
        mount.module_source(tmp_path, "nemo_rl.missing")


def test_launcher_mounts_at_editable_paths_and_keeps_external_environments(tmp_path):
    root = tmp_path / "rl"
    gym = root / mount.GYM_RELATIVE
    (gym / "scripts").mkdir(parents=True)
    launcher = Path(__file__).resolve().parents[3] / "scripts/launch_swe_single_agent_turn_training.sh"
    tools = tmp_path / "reviewed-launch-tools"
    tools.mkdir()
    copied = tools / launcher.name
    copied.write_bytes(launcher.read_bytes())
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "uv").write_text("#!/bin/sh\necho unexpected-preflight >&2\nexit 89\n")
    (binary / "uv").chmod(0o755)
    (binary / "sbatch").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$MOUNTS" "$UV_PROJECT_ENVIRONMENT" "$NEMO_RL_VENV_DIR" '
        '"$NEMO_GYM_VENV_DIR" "$UV_NO_SYNC" "$COMMAND" > "$SUBMISSION_RECEIPT"\nprintf "%s\\n" "$@" >> "$SUBMISSION_RECEIPT"\n'
    )
    (binary / "sbatch").chmod(0o755)
    receipt = tmp_path / "receipt"
    env = os.environ | {
        "PATH": f"{binary}:{os.environ['PATH']}",
        "NEMO_RL_ROOT": str(root),
        "GPUS_PER_NODE": "4",
        "CONFIG_PATH": str(tmp_path / "config.yaml"),
        "CONTAINER": str(tmp_path / "image.sqsh"),
        "CONTAINER_PROVENANCE": str(tmp_path / "image.json"),
        "SHARED_ROOT": str(tmp_path.parent),
        "SLURM_ACCOUNT": "test",
        "SLURM_PARTITION": "test",
        "OPENSANDBOX_DOMAIN": "example.invalid",
        "OPENSANDBOX_API_KEY": "test-key",
        "SUBMISSION_RECEIPT": str(receipt),
    }
    result = subprocess.run(
        [
            "bash",
            str(copied),
            "--source-mount",
            "--gym-root",
            str(gym),
            "--exclude",
            "unhealthy-node",
            "--walltime",
            "02:00:00",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = receipt.read_text().splitlines()
    assert lines[:5] == [
        f"{tmp_path.parent}:{tmp_path.parent},{root}:/opt/nemo-rl:ro",
        "/opt/nemo_rl_venv",
        "/opt/ray_venvs",
        "/opt/gym_venvs",
        "1",
    ]
    assert "uv run --no-sync examples/run_grpo_single_controller.py" in lines[5]
    assert "check_swe_training_recipe" not in lines[5] and "--runtime" not in lines[5]
    assert "test-key" not in result.stdout
    assert "--exclude=unhealthy-node" in lines[6:]
    assert "--time=02:00:00" in lines[6:]
    assert f"--chdir={tmp_path}" in lines[6:]
    assert f"--output={tmp_path}/slurm/slurm-%j.out" in lines[6:]
    assert f"--error={tmp_path}/slurm/slurm-%j.err" in lines[6:]
    assert (tmp_path / "slurm").is_dir()
