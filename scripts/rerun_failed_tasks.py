# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select original task rows and delegate execution and scoring to their original Gym configs."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _rows(path: str | Path) -> list[dict]:
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row: {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object: {path}:{line_number}")
            result.append(row)
    return result


def _write(path: Path, value: Any, *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        if jsonl:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")


def _atomic_update(path: Path, value: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _path(value: str, base: Path) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _fresh_private_directory(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists():
        raise ValueError(f"Output directory must be fresh: {path}")
    for ancestor in (path, *path.parents):
        if (ancestor / ".git").exists():
            raise ValueError("Private task and result outputs must be outside Git worktrees")
    return path


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or any(character in value for character in "\r\n\x00"):
        raise ValueError(f"Invalid {label}")
    return value


def _index(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid {label}: expected lowercase SHA-256")
    return value


def _check_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"Missing or changed {label}: {path}")


def _tree_manifest(root: Path) -> list[dict]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Missing or unsupported original asset directory: {root}")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Runtime asset snapshots require regular files and directories")
        entry = {
            "path": str(path.relative_to(root)),
            "type": "file" if path.is_file() else "directory",
            "mode": stat.S_IMODE(path.stat().st_mode),
        }
        if path.is_file():
            entry["sha256"] = sha256_file(path)
        entries.append(entry)
    return entries


def _check_assets(assets: list[dict]) -> None:
    for asset in assets:
        if _tree_manifest(Path(asset["source_dir"])) != asset["entries"]:
            raise ValueError("Original runtime asset tree changed")


def _set_identity(row: dict, key: str, value: Any) -> None:
    if key in row and row[key] != value:
        raise ValueError(f"Original task has conflicting {key}")
    row[key] = value


def prepare_dataset(selection_path: str | Path, routes_path: str | Path, output_dir: str | Path) -> dict:
    """Copy whole original dataset rows once per selected failed attempt; perform no model calls."""
    selection_path, routes_path = Path(selection_path).resolve(), Path(routes_path).resolve()
    output = _fresh_private_directory(output_dir)
    selection_digest, routes_digest = sha256_file(selection_path), sha256_file(routes_path)
    selections, routes = _rows(selection_path), _load(routes_path)
    if not selections:
        raise ValueError("Selection is empty")
    if routes.get("schema_version") != "original-task-rerun/routes/v1" or not isinstance(
        routes.get("benchmarks"), dict
    ):
        raise ValueError("Invalid routes schema")
    jobs_by_benchmark: dict[str, dict] = {}
    group_indices: dict[tuple[str, int], int] = {}
    group_hashes: dict[tuple[str, int], str] = {}
    group_rollouts: dict[tuple[str, int], int] = {}
    attempts: set[tuple[str, int, int]] = set()
    identities: set[tuple[str, str]] = set()
    for selection in selections:
        benchmark = _string(selection.get("benchmark"), "benchmark")
        trajectory = _string(selection.get("orig_trajectory_id"), "orig_trajectory_id")
        task_index = _index(selection.get("orig_task_index"), "orig_task_index")
        rollout_index = _index(selection.get("orig_rollout_index"), "orig_rollout_index")
        dataset_index = _index(selection.get("dataset_row_index"), "dataset_row_index")
        expected_hash = _digest(selection.get("expected_task_sha256"), "expected_task_sha256")
        identity, attempt = (benchmark, trajectory), (benchmark, task_index, rollout_index)
        if identity in identities or attempt in attempts:
            raise ValueError("Duplicate original trajectory or task attempt")
        identities.add(identity)
        attempts.add(attempt)
        if benchmark not in jobs_by_benchmark:
            route = routes["benchmarks"].get(benchmark)
            if not isinstance(route, dict):
                raise ValueError(f"No original route for {benchmark}")
            dataset = _path(_string(route.get("dataset_path"), "dataset_path"), routes_path.parent)
            dataset_hash = _digest(route.get("dataset_sha256"), "dataset_sha256")
            _check_file(dataset, dataset_hash, "original dataset")
            configs = route.get("config_paths")
            if not isinstance(configs, list) or not configs:
                raise ValueError("Every original route requires nonempty config_paths")
            config_files = []
            for config in configs:
                path = _path(_string(config, "config_path"), routes_path.parent)
                if not path.is_file():
                    raise ValueError(f"Missing original config: {path}")
                config_files.append({"path": str(path), "sha256": sha256_file(path)})
            revision = route.get("gym_revision", routes.get("gym_revision"))
            if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                raise ValueError("Every route requires an exact 40-character gym_revision")
            jobs_by_benchmark[benchmark] = {
                "benchmark": benchmark,
                "dataset_path": str(dataset),
                "dataset_sha256": dataset_hash,
                "agent_name": _string(route.get("agent_name"), "agent_name"),
                "config_files": config_files,
                "gym_revision": revision,
                "required_files": [],
                "runtime_assets": [],
                "dataset_rows": _rows(dataset),
                "inputs": [],
                "mapping": [],
            }
            required_files = route.get("required_files", [])
            runtime_assets = route.get("runtime_assets", [])
            if not isinstance(required_files, list) or not isinstance(runtime_assets, list):
                raise ValueError("required_files and runtime_assets must be lists")
            for required in required_files:
                if not isinstance(required, dict):
                    raise ValueError("Invalid required_files entry")
                required_path = _path(_string(required.get("path"), "required_file_path"), routes_path.parent)
                required_hash = _digest(required.get("sha256"), "required_file_sha256")
                _check_file(required_path, required_hash, "original required asset")
                jobs_by_benchmark[benchmark]["required_files"].append(
                    {"path": str(required_path), "sha256": required_hash}
                )
            for asset in runtime_assets:
                if not isinstance(asset, dict):
                    raise ValueError("Invalid runtime_assets entry")
                source_dir = _path(_string(asset.get("source_dir"), "runtime_asset_source_dir"), routes_path.parent)
                key = _string(asset.get("config_key"), "runtime_asset_config_key")
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", key):
                    raise ValueError("runtime_assets config_key must be a dotted configuration location")
                jobs_by_benchmark[benchmark]["runtime_assets"].append(
                    {"source_dir": str(source_dir), "config_key": key, "entries": _tree_manifest(source_dir)}
                )
        job = jobs_by_benchmark[benchmark]
        if dataset_index >= len(job["dataset_rows"]):
            raise ValueError("dataset_row_index is outside the original dataset")
        original = job["dataset_rows"][dataset_index]
        if canonical_sha256(original) != expected_hash:
            raise ValueError("Original task row hash mismatch")
        group = (benchmark, task_index)
        if group in group_hashes and group_hashes[group] != expected_hash:
            raise ValueError("One original task index selects conflicting original task rows")
        group_hashes[group] = expected_hash
        group_indices.setdefault(group, len(group_indices))
        next_rollout = group_rollouts.get(group, 0)
        group_rollouts[group] = next_rollout + 1
        row = copy.deepcopy(original)
        reference = row.get("agent_ref")
        if reference is not None and (not isinstance(reference, dict) or reference.get("name") != job["agent_name"]):
            raise ValueError("Original agent_ref conflicts with original route")
        if reference is None:
            row["agent_ref"] = {"name": job["agent_name"]}
        source_identity = {
            "benchmark": benchmark,
            "orig_trajectory_id": trajectory,
            "orig_task_index": task_index,
            "orig_rollout_index": rollout_index,
            "task_group_id": f"{benchmark}:{task_index}",
        }
        for key, value in source_identity.items():
            _set_identity(row, key, value)
        mapping = {
            **source_identity,
            "dataset_row_index": dataset_index,
            "expected_task_sha256": expected_hash,
            "source_bookkeeping": {
                key: copy.deepcopy(original[key]) for key in ("_ng_task_index", "_ng_rollout_index") if key in original
            },
            "_ng_task_index": group_indices[group],
            "_ng_rollout_index": next_rollout,
        }
        row["_ng_task_index"], row["_ng_rollout_index"] = mapping["_ng_task_index"], mapping["_ng_rollout_index"]
        job["inputs"].append(row)
        job["mapping"].append(mapping)
    # Validate all referenced inputs before the first output is created.
    _check_file(selection_path, selection_digest, "selection")
    _check_file(routes_path, routes_digest, "routes")
    for job in jobs_by_benchmark.values():
        _check_file(Path(job["dataset_path"]), job["dataset_sha256"], "original dataset")
        for config in job["config_files"]:
            _check_file(Path(config["path"]), config["sha256"], "original config")
        for required in job["required_files"]:
            _check_file(Path(required["path"]), required["sha256"], "original required asset")
        _check_assets(job["runtime_assets"])
    output.mkdir(parents=True, mode=0o700)
    jobs = []
    for index, job in enumerate(jobs_by_benchmark.values()):
        slug = re.sub(r"[^A-Za-z0-9.-]+", "-", job["benchmark"])[:64] or "taskset"
        stem = f"jobs/{index:03d}-{slug}"
        input_path, mapping_path = output / f"{stem}.input.jsonl", output / f"{stem}.mapping.jsonl"
        _write(input_path, job.pop("inputs"), jsonl=True)
        _write(mapping_path, job.pop("mapping"), jsonl=True)
        job.pop("dataset_rows")
        jobs.append(
            {
                **job,
                "input_path": str(input_path.relative_to(output)),
                "input_sha256": sha256_file(input_path),
                "mapping_path": str(mapping_path.relative_to(output)),
                "mapping_sha256": sha256_file(mapping_path),
                "num_rows": len(_rows(input_path)),
            }
        )
    manifest = {
        "schema_version": "original-task-rerun/prepared/v1",
        "selection_path": str(selection_path),
        "selection_sha256": selection_digest,
        "routes_path": str(routes_path),
        "routes_sha256": routes_digest,
        "num_rows": len(selections),
        "num_task_groups": len(group_indices),
        "num_repeats": 1,
        "policy_server_names": routes.get("policy_server_names", ["policy_model"]),
        "jobs": jobs,
        "semantics": "Original task rows and original agents/configured graders; no replacement scoring or prompt wrapper.",
    }
    _write(output / "prepared_manifest.json", manifest)
    return manifest


def _git_revision(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError("gym_root must be the pinned original Git checkout")
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=root, capture_output=True, check=False)
    if dirty.returncode:
        raise ValueError(
            "Original Gym checkout has tracked modifications; an exact clean runtime revision is required"
        )
    return result.stdout.strip()


def _validate_policy_config(path: Path, allowed_names: Any) -> list[str]:
    if (
        not isinstance(allowed_names, list)
        or not allowed_names
        or any(not isinstance(name, str) or not name for name in allowed_names)
        or len(set(allowed_names)) != len(allowed_names)
    ):
        raise ValueError("policy_server_names must be a nonempty list of distinct server names")
    try:
        policy = _load(path)
    except (ValueError, TypeError) as error:
        raise ValueError("Target policy overlay must be JSON (also valid YAML)") from error
    if not isinstance(policy, dict) or not policy or any(key not in allowed_names for key in policy):
        raise ValueError("Target policy overlay may configure only the allowlisted policy server names")
    forbidden = {
        "resources_servers",
        "responses_api_agents",
        "responses_create_params",
        "tools",
        "instructions",
        "input",
        "prompt",
        "prompt_template",
        "judge",
        "grader",
        "reward_model",
        "agent_ref",
    }

    def check(value: Any) -> None:
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise ValueError("Target policy overlay may not override task prompts, tools, agents, or grading")
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(policy)
    return sorted(policy)


def plan_run(
    manifest_path: str | Path,
    gym_root: str | Path,
    policy_config: str | Path,
    output_dir: str | Path,
    *,
    python_executable: str = sys.executable,
    concurrency: int = 4,
) -> dict:
    """Validate every job and return commands; do not create files or start servers."""
    manifest_path = Path(manifest_path).resolve()
    prepared = _load(manifest_path)
    if prepared.get("schema_version") != "original-task-rerun/prepared/v1" or prepared.get("num_repeats") != 1:
        raise ValueError("Invalid prepared manifest")
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be positive")
    output, gym_root, policy_config = (
        _fresh_private_directory(output_dir),
        Path(gym_root).resolve(),
        Path(policy_config).resolve(),
    )
    if not policy_config.is_file():
        raise ValueError("Missing target policy config")
    policy_server_names = _validate_policy_config(policy_config, prepared.get("policy_server_names", ["policy_model"]))
    _string(python_executable, "python_executable")
    revision = _git_revision(gym_root)
    jobs = []
    seen_indices: set[tuple[int, int]] = set()
    total_rows = 0
    for job_index, job in enumerate(prepared.get("jobs", [])):
        if job.get("gym_revision") != revision:
            raise ValueError("Original Gym revision does not match the execution checkout")
        _check_file(Path(job["dataset_path"]), job["dataset_sha256"], "original dataset")
        original_rows = _rows(job["dataset_path"])
        input_path = _path(job["input_path"], manifest_path.parent)
        mapping_path = _path(job["mapping_path"], manifest_path.parent)
        _check_file(input_path, job["input_sha256"], "prepared input")
        _check_file(mapping_path, job["mapping_sha256"], "attempt mapping")
        inputs, mappings = _rows(input_path), _rows(mapping_path)
        if len(inputs) != len(mappings) or len(inputs) != job["num_rows"]:
            raise ValueError("Prepared input and attempt mapping counts disagree")
        for row, mapping in zip(inputs, mappings, strict=True):
            key = (mapping["_ng_task_index"], mapping["_ng_rollout_index"])
            if key in seen_indices:
                raise ValueError("Duplicate dispatched rollout indices")
            seen_indices.add(key)
            for identity_key in (
                "benchmark",
                "orig_trajectory_id",
                "orig_task_index",
                "orig_rollout_index",
                "task_group_id",
            ):
                if row.get(identity_key) != mapping.get(identity_key):
                    raise ValueError("Prepared identity and mapping disagree")
            if row.get("_ng_task_index") != key[0] or row.get("_ng_rollout_index") != key[1]:
                raise ValueError("Prepared bookkeeping and mapping disagree")
            original = original_rows[mapping["dataset_row_index"]]
            if canonical_sha256(original) != mapping["expected_task_sha256"]:
                raise ValueError("Original task row changed")
            expected = copy.deepcopy(original)
            if expected.get("agent_ref") is None:
                expected["agent_ref"] = {"name": job["agent_name"]}
            for name in (
                "benchmark",
                "orig_trajectory_id",
                "orig_task_index",
                "orig_rollout_index",
                "task_group_id",
                "_ng_task_index",
                "_ng_rollout_index",
            ):
                expected[name] = mapping[name]
            if row != expected:
                raise ValueError(
                    "Prepared task differs from the whole original row beyond documented identity/bookkeeping"
                )
        for config in job["config_files"]:
            _check_file(Path(config["path"]), config["sha256"], "original config")
        for required in job.get("required_files", []):
            _check_file(Path(required["path"]), required["sha256"], "original required asset")
        _check_assets(job.get("runtime_assets", []))
        result_path = output / f"{job_index:03d}.rollouts.jsonl"
        dotenv = gym_root / "env.yaml"
        worker_spec = {
            "gym_root": str(gym_root),
            "gym_revision": revision,
            "config_files": job["config_files"],
            "required_files": job.get("required_files", []),
            "runtime_assets": job.get("runtime_assets", []),
            "policy_config": str(policy_config),
            "policy_config_sha256": sha256_file(policy_config),
            "policy_server_names": policy_server_names,
            "dotenv_path": str(dotenv) if dotenv.is_file() else None,
            "dotenv_sha256": sha256_file(dotenv) if dotenv.is_file() else None,
            "input_path": str(input_path),
            "input_sha256": job["input_sha256"],
            "output_path": str(result_path),
            "agent_name": job["agent_name"],
            "num_rows": job["num_rows"],
            "concurrency": concurrency,
        }
        worker_path = output / f"{job_index:03d}.worker.json"
        worker_spec_sha256 = canonical_sha256(worker_spec)
        command = [
            python_executable,
            str(Path(__file__).resolve()),
            "_worker",
            "--job",
            str(worker_path),
            "--expected-job-sha256",
            worker_spec_sha256,
        ]
        jobs.append(
            {
                "benchmark": job["benchmark"],
                "agent_name": job["agent_name"],
                "num_rows": job["num_rows"],
                "command": command,
                "input_path": str(input_path),
                "mapping_path": str(mapping_path),
                "mapping_sha256": job["mapping_sha256"],
                "output_path": str(result_path),
                "status": "planned",
                "worker_path": str(worker_path),
                "worker_spec": worker_spec,
                "worker_spec_sha256": worker_spec_sha256,
            }
        )
        total_rows += len(inputs)
    if not jobs or total_rows != prepared.get("num_rows"):
        raise ValueError("Empty or inconsistent prepared jobs")
    return {
        "schema_version": "original-task-rerun/run/v1",
        "prepared_manifest": str(manifest_path),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "gym_root": str(gym_root),
        "gym_revision": revision,
        "policy_config": str(policy_config),
        "policy_config_sha256": sha256_file(policy_config),
        "output_dir": str(output),
        "policy_server_names": policy_server_names,
        "num_rows": total_rows,
        "num_task_groups": prepared["num_task_groups"],
        "jobs": jobs,
        "execute": False,
        "python_executable": python_executable,
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(__file__),
    }


def _recheck_plan_inputs(plan: dict) -> None:
    _check_file(Path(plan["runner_path"]), plan["runner_sha256"], "rerun helper")
    _check_file(Path(plan["prepared_manifest"]), plan["prepared_manifest_sha256"], "prepared manifest")
    _check_file(Path(plan["policy_config"]), plan["policy_config_sha256"], "policy config")
    if _git_revision(Path(plan["gym_root"])) != plan["gym_revision"]:
        raise ValueError("Original Gym revision changed during execution")
    prepared = _load(plan["prepared_manifest"])
    base = Path(plan["prepared_manifest"]).parent
    for job in prepared["jobs"]:
        _check_file(Path(job["dataset_path"]), job["dataset_sha256"], "original dataset")
        _check_file(_path(job["input_path"], base), job["input_sha256"], "prepared input")
        _check_file(_path(job["mapping_path"], base), job["mapping_sha256"], "attempt mapping")
        for config in job["config_files"]:
            _check_file(Path(config["path"]), config["sha256"], "original config")
        for required in job.get("required_files", []):
            _check_file(Path(required["path"]), required["sha256"], "original required asset")
        _check_assets(job.get("runtime_assets", []))
    for job in plan["jobs"]:
        worker = job["worker_spec"]
        if worker.get("dotenv_path"):
            _check_file(Path(worker["dotenv_path"]), worker["dotenv_sha256"], "original environment config")
        elif (Path(plan["gym_root"]) / "env.yaml").exists():
            raise ValueError("Original environment config appeared after planning")


def execute_original_job(job_spec: dict, *, preflight_only: bool = False) -> dict:
    """Use the original server and collection APIs without full-dataset collation."""
    root = Path(job_spec["gym_root"]).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError("Original runtime worker must run with cwd=gym_root")
    if _git_revision(root) != job_spec["gym_revision"]:
        raise ValueError("Original worker runtime revision changed")
    if os.environ.get("NEMO_GYM_CONFIG_DICT") or os.environ.get("NEMO_GYM_CONFIG_PATH"):
        raise ValueError("Worker cannot inherit another Gym server's resolved configuration")
    for item in [*job_spec["config_files"], *job_spec.get("required_files", [])]:
        _check_file(Path(item["path"]), item["sha256"], "original config or required asset")
    _check_assets(job_spec.get("runtime_assets", []))
    _check_file(Path(job_spec["policy_config"]), job_spec["policy_config_sha256"], "policy config")
    _check_file(Path(job_spec["input_path"]), job_spec["input_sha256"], "prepared input")
    _validate_policy_config(Path(job_spec["policy_config"]), job_spec["policy_server_names"])
    if job_spec.get("dotenv_path"):
        _check_file(Path(job_spec["dotenv_path"]), job_spec["dotenv_sha256"], "original environment config")
    elif (root / "env.yaml").exists():
        raise ValueError("Original environment config appeared after planning")
    sys.path.insert(0, str(root))
    package = importlib.import_module("nemo_gym")
    package_file = getattr(package, "__file__", None)
    if not package_file or not Path(package_file).resolve().is_relative_to(root):
        raise ValueError("Python imported a different Gym checkout")
    global_config = importlib.import_module("nemo_gym.global_config")
    collection = importlib.import_module("nemo_gym.rollout_collection")
    environment = importlib.import_module("nemo_gym.cli.env")
    omega = importlib.import_module("omegaconf").OmegaConf
    for module in (global_config, collection, environment):
        module_file = getattr(module, "__file__", None)
        if module_file and not Path(module_file).resolve().is_relative_to(root):
            raise ValueError("Python mixed modules from different Gym checkouts")
    initial = {
        "config_paths": [item["path"] for item in job_spec["config_files"]],
        **_load(job_spec["policy_config"]),
        "input_jsonl_fpath": job_spec["input_path"],
        "output_jsonl_fpath": job_spec["output_path"],
        "agent_name": job_spec["agent_name"],
        "num_repeats": 1,
        "num_samples_in_parallel": job_spec["concurrency"],
        "resume_from_cache": False,
        "num_repeats_add_seed": False,
        "upload_rollouts_to_wandb": False,
        "wandb_project": None,
        "wandb_name": None,
        "wandb_api_key": None,
    }
    initial = omega.create(initial)
    for index, asset in enumerate(job_spec.get("runtime_assets", [])):
        location = Path(asset["source_dir"])
        if not preflight_only:
            location = (
                Path(job_spec["output_path"]).parent
                / (Path(job_spec["output_path"]).stem + ".runtime-assets")
                / str(index)
            )
            if location.exists():
                raise ValueError("Runtime asset destination must be fresh")
            shutil.copytree(asset["source_dir"], location)
            if _tree_manifest(location) != asset["entries"]:
                raise ValueError("Runtime asset copy does not match its immutable source")
        omega.update(initial, asset["config_key"], str(location), force_add=True)
    parser_config = global_config.GlobalConfigDictParserConfig(
        skip_load_from_cli=True,
        initial_global_config_dict=initial,
        dotenv_path=Path(job_spec["dotenv_path"]) if job_spec.get("dotenv_path") else None,
        skip_load_from_dotenv=not bool(job_spec.get("dotenv_path")),
    )
    resolved = global_config.get_global_config_dict(global_config_dict_parser_config=parser_config)
    config = collection.RolloutCollectionConfig.model_validate(omega.to_container(resolved, resolve=True))
    if Path(config.input_jsonl_fpath).resolve() != Path(job_spec["input_path"]).resolve() or config.num_repeats != 1:
        raise ValueError("Original runtime changed the explicitly selected inputs or repeat count")
    if getattr(config, "limit", None) is not None:
        raise ValueError("Original run configuration limits the selected input rows")
    if getattr(config, "rollout_collection_driver", None):
        raise ValueError("Custom collection drivers require an explicit original-driver integration")
    if getattr(config, "prompt_config", None) is not None or getattr(config, "skills", None) is not None:
        raise ValueError("Prepared original tasks cannot be rewritten by a run-level prompt or skills configuration")
    overrides = getattr(config, "responses_create_params", None) or {}
    overrides = omega.to_container(omega.create(overrides), resolve=True)
    allowed_sampling = {"temperature", "top_p", "max_output_tokens", "reasoning", "seed"}
    if not isinstance(overrides, dict) or set(overrides) - allowed_sampling:
        raise ValueError("Original task loader may override only explicit sampling fields")
    canonical_sha256(overrides)
    helper = collection.RolloutCollectionHelper()
    prepared = helper._preprocess_rows_from_config(config)
    expected_rows = _rows(job_spec["input_path"])
    if len(prepared) != job_spec["num_rows"] or len(prepared) != len(expected_rows):
        raise ValueError("Original runtime changed the selected attempt count")
    expected_keys = {(row["_ng_task_index"], row["_ng_rollout_index"]) for row in expected_rows}
    actual_keys = {(row["_ng_task_index"], row["_ng_rollout_index"]) for row in prepared}
    if actual_keys != expected_keys:
        raise ValueError("Original runtime changed the selected attempt identities")
    expected_by_key = {(row["_ng_task_index"], row["_ng_rollout_index"]): row for row in expected_rows}
    for row in prepared:
        expected = copy.deepcopy(expected_by_key[(row["_ng_task_index"], row["_ng_rollout_index"])])
        if overrides:
            expected["responses_create_params"] = {**expected["responses_create_params"], **overrides}
        if row != expected:
            raise ValueError("Original task loader changed task content beyond recorded sampling overrides")
    receipt = {
        "status": "preflight_passed",
        "num_rows": len(prepared),
        "input_sha256": job_spec["input_sha256"],
        "gym_revision": job_spec["gym_revision"],
        "gym_package_path": str(Path(package_file).resolve()),
        "agent_name": job_spec["agent_name"],
        "sampling_overrides": overrides,
        "preprocessed_rows_sha256": canonical_sha256(prepared),
    }
    if preflight_only:
        return receipt
    servers = environment.RunHelper()
    started = False
    try:
        servers.start(parser_config)
        started = True
        asyncio.run(helper.run_from_config(config))
    finally:
        original_error = sys.exc_info()[0] is not None
        if started or hasattr(servers, "_processes"):
            try:
                servers.shutdown()
            except Exception:
                if not original_error:
                    raise
    return {**receipt, "status": "completed"}


def run_prepared(
    manifest_path: str | Path,
    gym_root: str | Path,
    policy_config: str | Path,
    output_dir: str | Path,
    *,
    python_executable: str = sys.executable,
    concurrency: int = 4,
    execute: bool = False,
) -> dict:
    """Plan by default. Explicit execution delegates all generation, tools, and grading to Gym."""
    plan = plan_run(
        manifest_path,
        gym_root,
        policy_config,
        output_dir,
        python_executable=python_executable,
        concurrency=concurrency,
    )
    if not execute:
        return plan
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, mode=0o700)
    plan["execute"] = True
    state_path = output / "run_manifest.json"
    _write(state_path, plan)
    for job in plan["jobs"]:
        _write(Path(job["worker_path"]), job["worker_spec"])
    # Resolve all original configurations and preprocess all subsets before starting any server.
    for index, job in enumerate(plan["jobs"]):
        preflight_command = [*job["command"], "--preflight-only"]
        log_path = output / f"{index:03d}.preflight.log"
        with log_path.open("x", encoding="utf-8") as log:
            os.chmod(log_path, 0o600)
            try:
                process = subprocess.run(preflight_command, cwd=plan["gym_root"], stdout=log, stderr=log, check=False)
            except OSError as error:
                job["status"], job["error"] = "preflight_failed", type(error).__name__
                _atomic_update(state_path, plan)
                return plan
        job["preflight_returncode"] = process.returncode
        if process.returncode:
            job["status"] = "preflight_failed"
            _atomic_update(state_path, plan)
            return plan
    _atomic_update(state_path, plan)
    for index, job in enumerate(plan["jobs"]):
        # Revalidate all immutable inputs before every benchmark process starts.
        try:
            _recheck_plan_inputs(plan)
        except ValueError:
            job["status"] = "inputs_changed"
            _atomic_update(state_path, plan)
            raise
        job["status"] = "running"
        _atomic_update(state_path, plan)
        stdout_path, stderr_path = output / f"{index:03d}.stdout.log", output / f"{index:03d}.stderr.log"
        with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
            os.chmod(stdout_path, 0o600)
            os.chmod(stderr_path, 0o600)
            try:
                process = subprocess.run(
                    job["command"], cwd=plan["gym_root"], stdout=stdout, stderr=stderr, check=False
                )
                job["returncode"] = process.returncode
                job["status"] = "completed" if process.returncode == 0 else "process_failed"
            except OSError as error:
                job["status"], job["error"] = "process_failed", type(error).__name__
        if Path(job["output_path"]).is_file():
            job["output_sha256"] = sha256_file(job["output_path"])
        _atomic_update(state_path, plan)
        if job["status"] != "completed":
            break
    return plan


def _original_result_has_error(row: dict) -> bool:
    if any(value for name, value in row.items() if name.endswith("_error") or name in ("error", "errors")):
        return True
    response = row.get("response")
    if isinstance(response, dict) and (
        response.get("error")
        or response.get("incomplete_details")
        or response.get("status") in ("queued", "in_progress", "incomplete", "failed", "cancelled")
    ):
        return True
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("exception_info"):
        return True

    def diagnostic_error(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                (
                    bool(child)
                    if key in ("error", "errors", "exception_info") or key.endswith("_error")
                    else diagnostic_error(child)
                )
                for key, child in value.items()
            )
        if isinstance(value, list):
            return any(diagnostic_error(child) for child in value)
        return False

    return any(
        diagnostic_error(value)
        for name, value in row.items()
        if any(
            marker in name for marker in ("judge_result", "grader_output", "verifier_result", "verification_result")
        )
    )


def summarize_results(run_manifest_path: str | Path, output_dir: str | Path) -> dict:
    """Join source identities to untouched Gym results, without inventing or changing any score."""
    run = _load(run_manifest_path)
    if run.get("schema_version") != "original-task-rerun/run/v1":
        raise ValueError("Invalid run manifest")
    output = _fresh_private_directory(output_dir)
    matched = []
    summaries = []
    for job in run["jobs"]:
        mapping_path = Path(job["mapping_path"])
        _check_file(mapping_path, job["mapping_sha256"], "attempt mapping")
        mappings = _rows(mapping_path)
        lookup = {(row["_ng_task_index"], row["_ng_rollout_index"]): row for row in mappings}
        seen: set[tuple[int, int]] = set()
        counts = {
            "benchmark": job["benchmark"],
            "expected": len(mappings),
            "observed": 0,
            "missing": 0,
            "duplicates": 0,
            "unknown_indices": 0,
            "recorded_reward_zero": 0,
            "recorded_reward_positive": 0,
            "missing_reward": 0,
            "masked": 0,
            "errors": 0,
            "scored_reward_zero": 0,
            "scored_reward_positive": 0,
        }
        result_path = Path(job["output_path"])
        if result_path.exists():
            if job.get("output_sha256"):
                _check_file(result_path, job["output_sha256"], "Gym output")
            for row in _rows(result_path):
                key = (row.get("_ng_task_index"), row.get("_ng_rollout_index"))
                if key not in lookup:
                    counts["unknown_indices"] += 1
                    continue
                if key in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(key)
                counts["observed"] += 1
                reward = row.get("reward")
                valid_reward = (
                    isinstance(reward, (int, float)) and not isinstance(reward, bool) and math.isfinite(reward)
                )
                if valid_reward:
                    if reward == 0:
                        counts["recorded_reward_zero"] += 1
                    elif reward > 0:
                        counts["recorded_reward_positive"] += 1
                else:
                    counts["missing_reward"] += 1
                masked = any(row.get(name) == 0 for name in ("reward_mask", "grade_mask", "scoring_mask")) or bool(
                    row.get("skip") or row.get("skipped") or row.get("mask_sample")
                )
                if masked:
                    counts["masked"] += 1
                has_error = _original_result_has_error(row)
                if has_error:
                    counts["errors"] += 1
                if not masked and not has_error:
                    if valid_reward and reward == 0:
                        counts["scored_reward_zero"] += 1
                    elif valid_reward and reward > 0:
                        counts["scored_reward_positive"] += 1
                matched.append({"source_identity": lookup[key], "original_gym_result": row})
        counts["missing"] = len(mappings) - len(seen)
        counts["comparison_ready"] = (
            not any(
                counts[key]
                for key in ("missing", "duplicates", "unknown_indices", "missing_reward", "masked", "errors")
            )
            and job.get("status") == "completed"
        )
        summaries.append(counts)
    summary = {
        "schema_version": "original-task-rerun/summary/v1",
        "run_manifest": str(Path(run_manifest_path).resolve()),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "jobs": summaries,
        "expected": sum(job["expected"] for job in summaries),
        "observed": sum(job["observed"] for job in summaries),
        "missing": sum(job["missing"] for job in summaries),
        "comparison_ready": all(job["comparison_ready"] for job in summaries),
        "scoring": "Original Gym reward and diagnostics are retained verbatim; missing and duplicate results are not rescored.",
    }
    output.mkdir(parents=True, mode=0o700)
    _write(output / "matched_results.jsonl", matched, jsonl=True)
    _write(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Select original task rows offline")
    prepare.add_argument("--selection", required=True)
    prepare.add_argument("--routes", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--expected-count", type=int)
    run = commands.add_parser("run", help="Print an original-Gym run plan; --execute starts it")
    run.add_argument("--manifest", required=True)
    run.add_argument("--gym-root", required=True)
    run.add_argument("--policy-config", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--python", dest="python_executable", default=sys.executable)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--execute", action="store_true")
    summary = commands.add_parser("summarize", help="Attach source identities to original scores")
    summary.add_argument("--run-manifest", required=True)
    summary.add_argument("--output-dir", required=True)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--job", required=True)
    worker.add_argument("--expected-job-sha256", required=True)
    worker.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "prepare":
            if arguments.expected_count is not None and len(_rows(arguments.selection)) != arguments.expected_count:
                raise ValueError("Selection count does not match --expected-count")
            result = prepare_dataset(arguments.selection, arguments.routes, arguments.output_dir)
        elif arguments.command == "run":
            result = run_prepared(
                arguments.manifest,
                arguments.gym_root,
                arguments.policy_config,
                arguments.output_dir,
                python_executable=arguments.python_executable,
                concurrency=arguments.concurrency,
                execute=arguments.execute,
            )
        elif arguments.command == "summarize":
            result = summarize_results(arguments.run_manifest, arguments.output_dir)
        else:
            job = _load(arguments.job)
            if canonical_sha256(job) != arguments.expected_job_sha256:
                raise ValueError("Original worker specification changed after planning")
            result = execute_original_job(job, preflight_only=arguments.preflight_only)
    except (ValueError, OSError, KeyError, IndexError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if (
        arguments.command == "run"
        and arguments.execute
        and any(job["status"] != "completed" for job in result["jobs"])
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
