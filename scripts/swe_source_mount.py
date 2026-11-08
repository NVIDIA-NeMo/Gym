# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind an unchanged compiled image to a separately reviewed, compatible source tree.

The source is mounted at the image's editable-install paths. Dependency metadata,
build inputs, environment selection code and compiled submodules must stay identical.
This is deliberately narrower than a general dependency compatibility resolver.
"""

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


GYM_RELATIVE = "3rdparty/Gym-workspace/Gym"
IMAGE_RL_ROOT = Path("/opt/nemo-rl")
ENVIRONMENT_CODE = {
    "nemo_rl/distributed/actor_environments.py",
    "nemo_rl/distributed/ray_actor_environment_registry.py",
    "nemo_rl/utils/venvs.py",
    "nemo_gym/orchestration/server_venv.py",
    "nemo_gym/orchestration/executors/script_templates.py",
    "nemo_gym/utils/venv_setup.py",
    "nemo_gym/cli/_locked_env.py",
}


def git(root: Path, *args: str) -> str:
    """Read the local repository; source mounting performs no fetch or installation."""
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def git_tree(root: Path, revision: str) -> dict[str, tuple[str, str]]:
    """Return Git modes and object identities, including submodule entries."""
    result = {}
    for row in git(root, "ls-tree", "-r", "-z", revision).split("\0"):
        if row:
            metadata, name = row.split("\t", 1)
            mode, _, oid = metadata.split()
            result[name] = (mode, oid)
    return result


def source_change_allowed(name: str) -> bool:
    """Allow ordinary Python/config edits; require a new image for build/runtime inputs."""
    path = Path(name)
    if (
        name in ENVIRONMENT_CODE
        or path.name in {"setup.py", "_backend.py", "conftest.py"}
        or path.name.startswith(("setup_", "install_", "prefetch_"))
    ):
        return False
    if path.parts[0] in {"tests", "docs", "fern"} or path.suffix in {".md", ".mdx", ".rst"}:
        return True
    if name == "scripts/launch_swe_single_agent_turn_training.sh":
        return True
    return path.parts[0] in {
        "nemo_rl",
        "nemo_gym",
        "scripts",
        "examples",
        "resources_servers",
        "responses_api_agents",
        "responses_api_models",
        "environment_servers",
        "benchmarks",
    } and path.suffix in {".py", ".yaml", ".yml", ".js", ".mjs", ".cjs"}


def snapshot(rl_root: Path, *, base_rl_commit: str, base_gym_commit: str) -> dict:
    """Check clean nested repositories and record all tracked bytes against their Git objects."""
    repositories = {}
    files = {}
    changes = []

    def visit(relative: str, base: str, expected: str | None = None) -> None:
        root = rl_root / relative
        if (
            not (root / ".git").is_dir()
            or (root / ".git").is_symlink()
            or (root / ".git/objects/info/alternates").exists()
            or (root / ".git/commondir").exists()
        ):
            raise ValueError(f"Mount a self-contained Git snapshot, without worktree/alternate pointers: {relative}")
        revision = git(root, "rev-parse", "HEAD").strip()
        if expected is not None and revision != expected:
            raise ValueError(f"Submodule checkout differs from its parent pin: {relative}")
        subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD"], check=True)
        before, after = git_tree(root, base), git_tree(root, revision)
        repositories[relative] = {"base_commit": base, "commit": revision}
        for name in sorted(before.keys() | after.keys()):
            old, new = before.get(name), after.get(name)
            full = (Path(relative) / name).as_posix()
            if (old and old[0] == "160000") or (new and new[0] == "160000"):
                if old is None or new is None or old[0] != new[0]:
                    raise ValueError(f"Source mounting cannot add or remove a compiled submodule: {full}")
                if full != GYM_RELATIVE and old != new:
                    raise ValueError(f"Compiled submodule changed; rebuild its runtime: {full}")
                if full == GYM_RELATIVE and old[1] != base_gym_commit:
                    raise ValueError("Image Gym revision differs from the base RL submodule pin")
                visit(full, old[1], new[1])
                continue
            if old != new:
                if relative not in (".", GYM_RELATIVE) or not source_change_allowed(name):
                    raise ValueError(f"Dependency/build input changed; rebuild the runtime: {full}")
                changes.append(full)
            if new is not None:
                files[full] = {"mode": new[0], "git_blob_sha1": new[1]}

    visit(".", base_rl_commit)
    if GYM_RELATIVE not in repositories:
        raise ValueError("RL source must contain the pinned Gym submodule")
    result = {"repositories": repositories, "files": files, "changed_files": changes}
    check_snapshot(rl_root, result)
    return result


def check_snapshot(root: Path, source: dict) -> None:
    """Check exact file bytes/modes and reject symlinks escaping the mounted snapshot."""
    root = root.resolve()
    for relative, expected in source["repositories"].items():
        repository = root / relative
        if (
            not (repository / ".git").is_dir()
            or (repository / ".git").is_symlink()
            or (repository / ".git/objects/info/alternates").exists()
            or (repository / ".git/commondir").exists()
        ):
            raise ValueError(f"Source requires portable Git metadata: {relative}")
        if git(repository, "rev-parse", "HEAD").strip() != expected["commit"]:
            raise ValueError(f"Mounted source revision changed: {relative}")
        index = {}
        for row in git(repository, "ls-files", "--stage", "-z").split("\0"):
            if row:
                metadata, name = row.split("\t", 1)
                mode, oid, stage = metadata.split()
                if stage != "0":
                    raise ValueError(f"Unmerged mounted source index: {relative}/{name}")
                index[name] = (mode, oid)
        if index != git_tree(repository, expected["commit"]):
            raise ValueError(f"Mounted source Git inventory changed: {relative}")
    for name, expected in source["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Source symlink escapes the mounted tree: {name}")
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            actual_mode, data = "120000", os.fsencode(os.readlink(path))
        elif stat.S_ISREG(mode):
            actual_mode = "100755" if mode & stat.S_IXUSR else "100644"
            data = path.read_bytes()
        else:
            raise ValueError(f"Unexpected source file type: {name}")
        oid = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual_mode != expected["mode"] or oid != expected["git_blob_sha1"]:
            raise ValueError(f"Mounted source changed: {name}")
    # A stale bytecode file or an extra Python module can shadow otherwise correct
    # tracked sources. Host preparation may have a .venv; runtime uses /opt venvs.
    for directory, dirs, names in os.walk(root):
        for name in dirs:
            path = Path(directory) / name
            if name != ".venv" and path.is_symlink() and path.relative_to(root).as_posix() not in source["files"]:
                raise ValueError(
                    f"Untracked executable source directory in mounted snapshot: {path.relative_to(root)}"
                )
        dirs[:] = [name for name in dirs if name not in {".git", ".venv"}]
        for name in names:
            relative = (Path(directory) / name).relative_to(root).as_posix()
            if relative not in source["files"]:
                raise ValueError(f"Untracked executable source in mounted snapshot: {relative}")


def check_image_binding(manifest: dict, image: dict) -> None:
    """Keep the image's baked revisions distinct from the source that will execute."""
    binding = manifest.get("source_mount")
    expected = binding["base_image"] if binding else manifest
    for key in ("gym_commit", "rl_commit", "gym_lock_sha256", "rl_lock_sha256"):
        if image[key] != expected[key]:
            raise ValueError(f"Container build provenance differs from the prepared recipe: {key}")
    if binding:
        if image != binding["base_image"]:
            raise ValueError("Source mount base image provenance changed")
        for key in ("gym_lock_sha256", "rl_lock_sha256"):
            if image[key] != manifest[key]:
                raise ValueError(f"Source mount dependency lock differs from the image: {key}")


def check_imports(*, python: Path, modules: dict[str, Path], cwd: Path) -> dict:
    """Import real modules in an existing interpreter and verify each resolved source path."""
    if not python.is_file():
        raise ValueError(f"Compiled environment missing; source mounts never install it: {python}")
    probe = """
import importlib, json, sys
from pathlib import Path
expected = json.loads(sys.argv[1])
observed = {}
for name, path in expected.items():
    module = importlib.import_module(name)
    actual = Path(module.__file__).resolve()
    if actual != Path(path).resolve():
        raise ValueError(f'{name} imported {actual}, expected {path}')
    observed[name] = str(actual)
print('SOURCE_IMPORTS=' + json.dumps({'executable': sys.executable, 'prefix': sys.prefix, 'modules': observed}))
"""
    result = subprocess.run(
        [str(python), "-c", probe, json.dumps({k: str(v) for k, v in modules.items()})],
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=180,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"Mounted-source import failed in {python}: {result.stderr[-4000:]}")
    rows = [
        line.removeprefix("SOURCE_IMPORTS=")
        for line in result.stdout.splitlines()
        if line.startswith("SOURCE_IMPORTS=")
    ]
    if len(rows) != 1:
        raise ValueError(f"Missing source import receipt from {python}")
    receipt = json.loads(rows[0])
    if Path(receipt["prefix"]).resolve() != python.parent.parent.resolve():
        raise ValueError(f"Interpreter resolved outside the selected environment: {python}")
    return receipt


def module_source(root: Path, module: str) -> Path:
    """Resolve a pinned Python module or package without importing it in the driver."""
    path = root.joinpath(*module.split("."))
    for candidate in (path / "__init__.py", path.with_suffix(".py")):
        if candidate.is_file():
            return candidate
    raise ValueError(f"No pinned source for Python module: {module}")


def check_runtime_imports(config: DictConfig) -> dict:
    """Probe the existing driver, registered actor interpreters and configured Gym components.

    No environment builder, Ray initialization, server startup or package installer
    runs here. The real controller subsequently exercises actor/server startup.
    """
    from nemo_rl.distributed.actor_environments import ACTOR_ENVIRONMENTS
    from nemo_rl.environments.nemo_gym import build_nemo_gym_config

    from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
    from nemo_gym.orchestration.server_venv import get_venv_path
    from nemo_gym.utils.venv_setup import SETUP_COMPLETE_MARKER

    root = IMAGE_RL_ROOT
    gym = root / GYM_RELATIVE
    if not os.statvfs(root).f_flag & os.ST_RDONLY:
        raise ValueError("Executed source snapshot must be mounted read-only at /opt/nemo-rl")
    for key, expected in {
        "UV_PROJECT_ENVIRONMENT": "/opt/nemo_rl_venv",
        "NEMO_RL_VENV_DIR": "/opt/ray_venvs",
        "NEMO_GYM_VENV_DIR": "/opt/gym_venvs",
    }.items():
        if os.environ.get(key) != expected:
            raise ValueError(f"Source mount requires {key}={expected}")
    if Path(sys.prefix) != Path("/opt/nemo_rl_venv"):
        raise ValueError("Normal uv driver did not select the image's driver environment")
    packages = {"nemo_rl": root / "nemo_rl/__init__.py", "nemo_gym": gym / "nemo_gym/__init__.py"}
    results = {"driver": check_imports(python=Path(sys.executable), modules=packages, cwd=root)}
    tiers = {("vllm", "nemo_gym"), ("mcore", "nemo_gym"), ("vllm",), ("nemo_gym",)}
    for actor, extras in ACTOR_ENVIRONMENTS.items():
        # SFT is not part of these GRPO recipes; do not imply it was exercised.
        if extras is None or tuple(extras) not in tiers or actor.endswith(".SFTMegatronPolicyWorker"):
            continue
        module = actor.rsplit(".", 1)[0]
        modules = {"nemo_rl": packages["nemo_rl"], module: module_source(root, module)}
        if "nemo_gym" in extras:
            modules["nemo_gym"] = packages["nemo_gym"]
        results[actor] = check_imports(python=Path("/opt/ray_venvs") / actor / "bin/python", modules=modules, cwd=root)
    actor_config = build_nemo_gym_config(
        OmegaConf.to_container(config.env, resolve=True),
        base_urls=["http://127.0.0.1:1/v1"],
        model_name=config.policy.model_name,
        enable_router_replay=False,
        use_fastokens=False,
        turn_checkpointing_enabled=False,
    )
    initial = OmegaConf.merge(
        actor_config["initial_global_config_dict"],
        {
            "policy_model_name": config.policy.model_name,
            "policy_base_url": "http://127.0.0.1:1/v1",
            "policy_api_key": "source-import-probe",
        },
    )
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    if not resolved.skip_venv_if_present or resolved.uv_venv_dir != "/opt/gym_venvs":
        raise ValueError("Gym must reuse the image's component environments")
    for value in resolved.values():
        if not isinstance(value, DictConfig):
            continue
        for kind in ("environment_servers", "resources_servers", "responses_api_agents", "responses_api_models"):
            for name, settings in value.get(kind, {}).items():
                component = f"{kind}/{name}"
                environment = get_venv_path(gym / component, resolved)
                if not (environment / SETUP_COMPLETE_MARKER).is_file() or not (environment / "bin/activate").is_file():
                    raise ValueError(f"Gym component was not prepared in the image: {component}")
                entrypoint = settings.get("entrypoint", "app.py")
                module = f"{kind}.{name}.{Path(entrypoint).stem}"
                results[component] = check_imports(
                    python=environment / "bin/python",
                    modules={"nemo_gym": packages["nemo_gym"], module: gym / component / entrypoint},
                    cwd=gym,
                )
    return results
