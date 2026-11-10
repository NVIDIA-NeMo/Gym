# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download original registered tasks with their native processor, then bind a private failure selection."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerun_failed_tasks as rerun


def _private_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    for ancestor in (root, *root.parents):
        if (ancestor / ".git").exists():
            raise ValueError("Original task caches must remain outside Git worktrees")
    return root


def _identifier(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("backend") not in ("gitlab", "huggingface"):
        raise ValueError("registry_identifier requires an explicit supported backend")
    required = (
        ("dataset_name", "version", "artifact_fpath")
        if value["backend"] == "gitlab"
        else ("repo_id", "artifact_fpath")
    )
    for key in required:
        rerun._string(value.get(key), "registry_identifier." + key)
    return copy.deepcopy(value)


def prepare_download_plan(
    routes_path: str | Path, gym_root: str | Path, output_dir: str | Path, *, python_executable: str = sys.executable
) -> dict:
    routes_path, gym_root = Path(routes_path).resolve(), Path(gym_root).resolve()
    routes, output = rerun._load(routes_path), _private_root(output_dir)
    if routes.get("schema_version") != "original-task-download/routes/v1" or not isinstance(
        routes.get("benchmarks"), dict
    ):
        raise ValueError("Invalid original dataset download routes")
    revision = rerun._git_revision(gym_root)
    if revision != routes.get("gym_revision"):
        raise ValueError("Original Gym revision mismatch")
    if output.exists() and not (output / "download_manifest.json").is_file():
        raise ValueError("Existing cache directory has no matching download manifest")
    jobs = []
    for index, (benchmark, route) in enumerate(routes["benchmarks"].items()):
        rerun._string(benchmark, "benchmark")
        if not isinstance(route, dict):
            raise ValueError("Invalid download route")
        paths = route.get("config_paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError("Original config_paths are required")
        configs = []
        for item in paths:
            path = rerun._path(rerun._string(item, "config_path"), gym_root)
            if not path.is_file():
                raise ValueError("Missing original dataset configuration")
            configs.append({"path": str(path), "sha256": rerun.sha256_file(path)})
        identifier = _identifier(route.get("registry_identifier"))
        slug = re.sub(r"[^A-Za-z0-9.-]", "-", benchmark)[:64] or "dataset"
        dotenv = gym_root / "env.yaml"
        spec = {
            "benchmark": benchmark,
            "gym_root": str(gym_root),
            "gym_revision": revision,
            "config_files": configs,
            "agent_name": rerun._string(route.get("agent_name"), "agent_name"),
            "dataset_name": rerun._string(route.get("dataset_name"), "dataset_name"),
            "dataset_type": route.get("dataset_type", "validation"),
            "registry_identifier": identifier,
            "cache_dir": str(output / f"{index:03d}-{slug}"),
            "dotenv_path": str(dotenv) if dotenv.is_file() else None,
            "dotenv_sha256": rerun.sha256_file(dotenv) if dotenv.is_file() else None,
            "registry_assets": copy.deepcopy(route.get("registry_assets", [])),
        }
        if spec["dataset_type"] not in ("train", "validation"):
            raise ValueError("Native download selection requires a train or validation dataset")
        if not isinstance(spec["registry_assets"], list):
            raise ValueError("registry_assets must be a list")
        for asset in spec["registry_assets"]:
            if not isinstance(asset, dict):
                raise ValueError("Invalid registry asset")
            rerun._string(asset.get("name"), "registry asset name")
            if _identifier(asset.get("registry_identifier"))["backend"] != "gitlab":
                raise ValueError("Binary asset downloads require the original GitLab registry")
            rerun._digest(asset.get("sha256"), "registry asset sha256")
            config_key = rerun._string(asset.get("config_key"), "registry asset config_key")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", config_key):
                raise ValueError("Registry asset config_key must be a dotted location")
            if asset.get("runtime_copy") and not asset.get("archive"):
                raise ValueError("Runtime copies require an explicitly declared directory archive")
            archive = asset.get("archive")
            if archive:
                if not isinstance(archive, dict) or archive.get("format") != "tar.gz":
                    raise ValueError("Unsupported registry asset archive format")
                _safe_archive_path(archive.get("strip_prefix", ""))
                prefixes = archive.get("select_prefixes")
                if not isinstance(prefixes, list) or not prefixes:
                    raise ValueError("Archive selection requires explicit task prefixes")
                for prefix in prefixes:
                    if not isinstance(prefix, str) or not _safe_archive_path(prefix).parts:
                        raise ValueError("Invalid selected archive prefix")
        if route.get("local_dataset_path"):
            local = rerun._path(route["local_dataset_path"], routes_path.parent)
            rerun._check_file(
                local,
                rerun._digest(route.get("local_dataset_sha256"), "local_dataset_sha256"),
                "local original dataset",
            )
            spec.update(local_dataset_path=str(local), local_dataset_sha256=route["local_dataset_sha256"])
        jobs.append({"benchmark": benchmark, "worker_spec": spec, "status": "planned"})
    if not jobs:
        raise ValueError("No original datasets selected")
    plan = {
        "schema_version": "original-task-download/manifest/v1",
        "routes_path": str(routes_path),
        "routes_sha256": rerun.sha256_file(routes_path),
        "gym_root": str(gym_root),
        "gym_revision": revision,
        "output_dir": str(output),
        "python_executable": python_executable,
        "jobs": jobs,
    }
    previous = output / "download_manifest.json"
    if previous.exists():
        old = rerun._load(previous)
        if old.get("routes_sha256") != plan["routes_sha256"] or old.get("gym_revision") != revision:
            raise ValueError("Existing cache belongs to different routes or original runtime")
    return plan


def _validate_cache(spec: dict) -> dict | None:
    root = Path(spec["cache_dir"])
    if not root.exists():
        return None
    path = root / "receipt.json"
    if not path.is_file():
        raise ValueError("Incomplete original dataset cache cannot be treated as validated")
    receipt = rerun._load(path)
    if receipt.get("request_sha256") != rerun.canonical_sha256(spec):
        raise ValueError("Cached dataset request or configuration changed")
    for item in receipt["files"]:
        rerun._check_file(root / item["path"], item["sha256"], "cached original artifact")
    for item in receipt["expanded_config_files"]:
        rerun._check_file(Path(item["path"]), item["sha256"], "original dataset config")
    if "tree" in receipt:
        actual_tree = [entry for entry in rerun._tree_manifest(root) if entry["path"] != "receipt.json"]
        if actual_tree != receipt["tree"]:
            raise ValueError("Cached original artifact tree changed")
    for index, asset in enumerate(spec.get("registry_assets", [])):
        rerun._check_file(root / "assets" / str(index) / "artifact", asset["sha256"], "cached original registry asset")
    return receipt


def _safe_archive_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
        raise ValueError("Unsafe original asset archive path")
    return path


def extract_registry_archive(archive_path: Path, destination: Path, options: dict) -> None:
    """Extract declared regular task files without following links or executing archive contents."""
    if options.get("format") != "tar.gz":
        raise ValueError("Only explicitly declared tar.gz registry assets are supported")
    strip = _safe_archive_path(options.get("strip_prefix", ""))
    prefixes = [_safe_archive_path(value) for value in options.get("select_prefixes", [])]
    if not prefixes or any(not path.parts for path in prefixes):
        raise ValueError("Archive extraction requires explicit nonempty select_prefixes")
    destination.mkdir(parents=True, mode=0o700)
    found = set()
    seen = set()
    with tarfile.open(archive_path, "r|gz") as stream:
        for member in stream:
            path = _safe_archive_path(member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError("Archive links and special files are unsupported")
            if strip.parts:
                if path == strip:
                    continue
                if strip not in path.parents:
                    continue
                path = path.relative_to(strip)
            chosen = [index for index, prefix in enumerate(prefixes) if path == prefix or prefix in path.parents]
            if not chosen:
                continue
            if path in seen:
                raise ValueError("Duplicate selected archive path")
            seen.add(path)
            found.update(chosen)
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
            os.chmod(target, member.mode & 0o777)
    if len(found) != len(prefixes):
        raise ValueError("Original archive does not contain every selected task prefix")


def _native_modules(spec: dict):
    root = Path(spec["gym_root"])
    if Path.cwd().resolve() != root or rerun._git_revision(root) != spec["gym_revision"]:
        raise ValueError("Downloader must use the pinned original Gym checkout")
    if os.environ.get("NEMO_GYM_CONFIG_DICT") or os.environ.get("NEMO_GYM_CONFIG_PATH"):
        raise ValueError("Downloader cannot inherit another Gym server's configuration")
    sys.path.insert(0, str(root))
    package = importlib.import_module("nemo_gym")
    if not getattr(package, "__file__", None) or not Path(package.__file__).resolve().is_relative_to(root):
        raise ValueError("Downloader imported a different Gym checkout")
    modules = [
        importlib.import_module(name) for name in ("nemo_gym.global_config", "nemo_gym.train_data_utils", "omegaconf")
    ]
    for module in modules[:2]:
        if getattr(module, "__file__", None) and not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError("Downloader mixed Gym implementations")
    return modules


def download_original_source(spec: dict, *, preflight_only: bool = False) -> dict:
    """Use the original registry reader and TrainDataProcessor; never start an agent or grader."""
    root = _private_root(spec["cache_dir"])
    for item in spec["config_files"]:
        rerun._check_file(Path(item["path"]), item["sha256"], "original dataset config")
    cached = _validate_cache(spec)
    if cached is not None:
        return {"status": "cached", "receipt": cached}
    global_module, native, omega_module = _native_modules(spec)
    omega = omega_module.OmegaConf
    parser = global_module.GlobalConfigDictParser()
    expanded_paths, configs = parser.load_extra_config_paths([item["path"] for item in spec["config_files"]])
    merged = omega.merge(*configs)
    parser._recursively_swap_keys(merged)
    agent = merged.get(spec["agent_name"])
    implementations = agent.get("responses_api_agents") if agent is not None else None
    if implementations is None or len(implementations) != 1:
        raise ValueError("Original dataset agent is missing or ambiguous")
    implementation = next(iter(implementations))
    inner = implementations[implementation]
    declarations = omega.to_container(inner.get("datasets", []), resolve=False)
    matches = [
        item
        for item in declarations
        if item.get("name") == spec["dataset_name"] and item.get("type") == spec["dataset_type"]
    ]
    if len(matches) != 1:
        raise ValueError("Original registered dataset declaration is missing or ambiguous")
    declaration = copy.deepcopy(matches[0])
    identifier = spec["registry_identifier"]
    configured = declaration.get(identifier["backend"] + "_identifier")
    if configured is None and isinstance(declaration.get("source"), dict):
        source = declaration["source"]
        if source.get("type") == identifier["backend"]:
            configured = {key: value for key, value in source.items() if key != "type"}
    expected = {key: value for key, value in identifier.items() if key != "backend"}
    if configured != expected:
        raise ValueError("Selected dataset registry identifier differs from its original configuration")
    expanded_files = []
    for value in expanded_paths:
        path = rerun._path(value, Path(spec["gym_root"]))
        expanded_files.append({"path": str(path), "sha256": rerun.sha256_file(path)})
    if preflight_only:
        return {
            "status": "preflight_passed",
            "dataset_declaration_sha256": rerun.canonical_sha256(declaration),
            "registry_identifier": identifier,
            "expanded_config_files": expanded_files,
        }
    root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=".download-", dir=root.parent))
    try:
        raw_path = staging / "raw" / "dataset.jsonl"
        raw_path.parent.mkdir(mode=0o700)
        local = spec.get("local_dataset_path")
        if local:
            rerun._check_file(Path(local), spec["local_dataset_sha256"], "local original dataset")
        else:
            candidate = rerun._path(declaration["jsonl_fpath"], Path(spec["gym_root"]))
            local = str(candidate) if candidate.is_file() else None
        if local:
            shutil.copy2(local, raw_path)
        relocated = copy.deepcopy(declaration)
        relocated["jsonl_fpath"] = str(raw_path)
        relocated["num_repeats"] = 1
        processor_config = omega.create(
            {
                spec["agent_name"]: {
                    "responses_api_agents": {
                        implementation: {"entrypoint": inner.get("entrypoint", "app.py"), "datasets": [relocated]}
                    }
                },
                "mode": "train_preparation",
                "should_download": True,
                "data_source": identifier["backend"],
                "output_dirpath": str(staging / "prepared"),
            }
        )
        dotenv = spec.get("dotenv_path")
        if dotenv:
            rerun._check_file(Path(dotenv), spec["dotenv_sha256"], "original registry environment")
        registry_credentials = {}
        if dotenv:
            environment = omega.load(dotenv)
            credential_names = {"hf_token"} if identifier["backend"] == "huggingface" else set()
            if identifier["backend"] == "gitlab" or spec.get("registry_assets"):
                credential_names.update(("mlflow_tracking_uri", "mlflow_tracking_token"))
            for name in credential_names:
                value = omega.select(environment, name)
                if value is not None:
                    registry_credentials[name] = value
        initial = omega.merge(
            global_module.GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
            {"config_paths": [], "wandb_project": None, "wandb_name": None, "wandb_api_key": None},
            registry_credentials,
        )
        global_module.get_global_config_dict(
            global_config_dict_parser_config=global_module.GlobalConfigDictParserConfig(
                skip_load_from_cli=True,
                initial_global_config_dict=initial,
                skip_load_from_dotenv=True,
            )
        )
        native.TrainDataProcessor().run(processor_config)
        prepared = staging / "prepared" / (spec["dataset_type"] + ".jsonl")
        rows = rerun._rows(prepared)
        if not rows:
            raise ValueError("Original registry dataset is empty")
        assets = []
        for index, asset in enumerate(spec.get("registry_assets", [])):
            directory = staging / "assets" / str(index)
            directory.mkdir(parents=True, mode=0o700)
            artifact = directory / "artifact"
            gitlab = importlib.import_module("nemo_gym.gitlab_utils")
            config_types = importlib.import_module("nemo_gym.config_types")
            artifact_identifier = {
                key: value for key, value in asset["registry_identifier"].items() if key != "backend"
            }
            config = config_types.DownloadBinaryAssetGitlabConfig.model_validate(
                {**artifact_identifier, "output_fpath": str(artifact)}
            )
            gitlab.download_binary_asset(config)
            rerun._check_file(artifact, asset["sha256"], "original registry asset")
            target = artifact
            if asset.get("archive"):
                target = directory / "tree"
                extract_registry_archive(artifact, target, asset["archive"])
            assets.append(
                {
                    "name": asset["name"],
                    "config_key": asset["config_key"],
                    "path": str(target.relative_to(staging)),
                    "runtime_copy": bool(asset.get("runtime_copy")),
                    "registry_identifier": asset["registry_identifier"],
                    "artifact_sha256": asset["sha256"],
                }
            )
        files = [
            {"path": str(path.relative_to(staging)), "sha256": rerun.sha256_file(path)}
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        receipt = {
            "schema_version": "original-task-download/cache/v1",
            "request_sha256": rerun.canonical_sha256(spec),
            "registry_identifier": identifier,
            "dataset_declaration_sha256": rerun.canonical_sha256(declaration),
            "prepared_dataset_path": str(prepared.relative_to(staging)),
            "prepared_dataset_sha256": rerun.sha256_file(prepared),
            "num_rows": len(rows),
            "source_dataset_num_repeats": declaration.get("num_repeats", 1),
            "cache_dataset_num_repeats": 1,
            "expanded_config_files": expanded_files,
            "files": files,
            "assets": assets,
            "tree": rerun._tree_manifest(staging),
        }
        rerun._write(staging / "receipt.json", receipt)
        os.rename(staging, root)
        return {"status": "completed", "receipt": receipt}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def download_sources(routes_path, gym_root, output_dir, *, python_executable=sys.executable, execute=False) -> dict:
    plan = prepare_download_plan(routes_path, gym_root, output_dir, python_executable=python_executable)
    if not execute:
        return plan
    output = Path(plan["output_dir"])
    validated = [_validate_cache(job["worker_spec"]) for job in plan["jobs"]]
    if all(receipt is not None for receipt in validated):
        for job, receipt in zip(plan["jobs"], validated, strict=True):
            job.update(status="cached", receipt=receipt)
        return plan
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    state = output / "download_manifest.json"
    rerun._atomic_update(state, plan)
    for index, job in enumerate(plan["jobs"]):
        cached = _validate_cache(job["worker_spec"])
        if cached is not None:
            job.update(status="cached", receipt=cached)
            continue
        spec_path = output / f".{index:03d}.request.json"
        if spec_path.exists():
            if rerun._load(spec_path) != job["worker_spec"]:
                raise ValueError("Native download request changed")
        else:
            rerun._write(spec_path, job["worker_spec"])
        command = [
            python_executable,
            str(Path(__file__).resolve()),
            "_worker",
            "--job",
            str(spec_path),
            "--expected-job-sha256",
            rerun.canonical_sha256(job["worker_spec"]),
        ]
        log_descriptor, log_name = tempfile.mkstemp(prefix=f"{index:03d}.", suffix=".download.log", dir=output)
        with os.fdopen(log_descriptor, "w") as log:
            process = subprocess.run(command, cwd=plan["gym_root"], stdout=log, stderr=log, check=False)
        job["returncode"] = process.returncode
        job["log_path"] = log_name
        job["status"] = "completed" if process.returncode == 0 else "failed"
        if process.returncode == 0:
            job["receipt"] = _validate_cache(job["worker_spec"])
            if job["receipt"] is None:
                raise ValueError("Native downloader reported success without a validated cache")
        rerun._atomic_update(state, plan)
        if process.returncode:
            break
    rerun._atomic_update(state, plan)
    return plan


def bind_selection(selection_path: str | Path, download_manifest_path: str | Path, output_dir: str | Path) -> dict:
    manifest = rerun._load(download_manifest_path)
    if manifest.get("schema_version") != "original-task-download/manifest/v1":
        raise ValueError("Invalid native download manifest")
    original_routes = rerun._load(manifest["routes_path"])
    rerun._check_file(Path(manifest["routes_path"]), manifest["routes_sha256"], "original download routes")
    jobs = {job["benchmark"]: job for job in manifest["jobs"]}
    selections = rerun._rows(selection_path)
    prepared_rows = {}
    rerun_routes = {
        "schema_version": "original-task-rerun/routes/v1",
        "gym_revision": manifest["gym_revision"],
        "benchmarks": {},
    }
    for benchmark, job in jobs.items():
        spec = job["worker_spec"]
        receipt = _validate_cache(spec)
        if receipt is None:
            raise ValueError("Original task source has not been downloaded successfully")
        dataset = Path(spec["cache_dir"]) / receipt["prepared_dataset_path"]
        prepared_rows[benchmark] = rerun._rows(dataset)
        source = original_routes["benchmarks"][benchmark]
        route = {
            "dataset_path": str(dataset),
            "dataset_sha256": rerun.sha256_file(dataset),
            "config_paths": [item["path"] for item in spec["config_files"]],
            "agent_name": spec["agent_name"],
        }
        for key in ("required_files", "runtime_assets"):
            if key in source:
                route[key] = copy.deepcopy(source[key])
        overlay = copy.deepcopy(source.get("config_overrides", {}))
        if not isinstance(overlay, dict):
            raise ValueError("config_overrides must be a JSON object")
        for asset in receipt.get("assets", []):
            path = Path(spec["cache_dir"]) / asset["path"]
            if asset["runtime_copy"]:
                route.setdefault("runtime_assets", []).append(
                    {"source_dir": str(path), "config_key": asset["config_key"]}
                )
            else:
                route.setdefault("required_files", []).append({"path": str(path), "sha256": rerun.sha256_file(path)})
                cursor = overlay
                parts = asset["config_key"].split(".")
                for part in parts[:-1]:
                    cursor = cursor.setdefault(part, {})
                cursor[parts[-1]] = str(path)
        route["_asset_overlay"] = overlay
        rerun_routes["benchmarks"][benchmark] = route
    for selection in selections:
        rows = prepared_rows.get(selection["benchmark"])
        if rows is None:
            raise ValueError("Selected failure has no original registered source")
        expected = selection["expected_task_sha256"]
        index = selection["dataset_row_index"]
        if not (type(index) is int and 0 <= index < len(rows) and rerun.canonical_sha256(rows[index]) == expected):
            matches = [index for index, row in enumerate(rows) if rerun.canonical_sha256(row) == expected]
            if len(matches) != 1:
                raise ValueError("Original failed task cannot be bound uniquely to the downloaded original dataset")
            selection["dataset_row_index"] = matches[0]
    output = rerun._fresh_private_directory(output_dir)
    output.mkdir(parents=True, mode=0o700)
    for index, route in enumerate(rerun_routes["benchmarks"].values()):
        overlay = route.pop("_asset_overlay")
        if overlay:
            overlay_path = output / f"{index:03d}.assets.json"
            rerun._write(overlay_path, overlay)
            route["config_paths"].append(str(overlay_path))
    rerun._write(output / "selection.jsonl", selections, jsonl=True)
    rerun._write(output / "routes.json", rerun_routes)
    receipt = {
        "schema_version": "original-task-download/binding/v1",
        "num_attempts": len(selections),
        "selection_sha256": rerun.sha256_file(selection_path),
        "download_manifest_sha256": rerun.sha256_file(download_manifest_path),
        "bound_selection_path": str(output / "selection.jsonl"),
        "bound_routes_path": str(output / "routes.json"),
    }
    rerun._write(output / "binding_receipt.json", receipt)
    return receipt


def prepare_failed_tasks(
    selection_path: str | Path,
    routes_path: str | Path,
    gym_root: str | Path,
    output_dir: str | Path,
    *,
    python_executable: str = sys.executable,
) -> dict:
    """Download original registered data, bind selected failures, and prepare original-agent reruns."""
    workspace = _private_root(output_dir)
    workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
    completed = workspace / "preparation_receipt.json"
    selection_hash, routes_hash = rerun.sha256_file(selection_path), rerun.sha256_file(routes_path)
    download = download_sources(
        routes_path, gym_root, workspace / "downloads", python_executable=python_executable, execute=True
    )
    if any(job["status"] not in ("completed", "cached") for job in download["jobs"]):
        raise ValueError("Original data download failed; inspect the private download manifest and logs")
    if completed.exists():
        receipt = rerun._load(completed)
        if receipt["selection_sha256"] != selection_hash or receipt["routes_sha256"] != routes_hash:
            raise ValueError("Existing preparation belongs to another selection or route catalog")
        manifest_path = Path(receipt["prepared_manifest"])
        rerun._check_file(manifest_path, receipt["prepared_manifest_sha256"], "prepared rerun manifest")
        manifest = rerun._load(manifest_path)
        for name in ("selection", "routes"):
            rerun._check_file(Path(manifest[name + "_path"]), manifest[name + "_sha256"], "bound " + name)
        for job in manifest["jobs"]:
            rerun._check_file(Path(job["dataset_path"]), job["dataset_sha256"], "original prepared dataset")
            for name in ("input", "mapping"):
                rerun._check_file(
                    manifest_path.parent / job[name + "_path"], job[name + "_sha256"], "prepared " + name
                )
            for item in job["config_files"]:
                rerun._check_file(Path(item["path"]), item["sha256"], "original rerun config")
            rerun._check_assets(job.get("runtime_assets", []))
            for item in job.get("required_files", []):
                rerun._check_file(Path(item["path"]), item["sha256"], "original asset")
        return {**receipt, "status": "cached"}
    bound = bind_selection(selection_path, workspace / "downloads" / "download_manifest.json", workspace / "bound")
    manifest = rerun.prepare_dataset(bound["bound_selection_path"], bound["bound_routes_path"], workspace / "prepared")
    path = workspace / "prepared" / "prepared_manifest.json"
    receipt = {
        "schema_version": "original-task-download/preparation/v1",
        "status": "completed",
        "selection_sha256": selection_hash,
        "routes_sha256": routes_hash,
        "num_rows": manifest["num_rows"],
        "num_task_groups": manifest["num_task_groups"],
        "prepared_manifest": str(path),
        "prepared_manifest_sha256": rerun.sha256_file(path),
    }
    rerun._write(completed, receipt)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    for name in ("routes", "gym-root", "output-dir"):
        download.add_argument("--" + name, required=True)
    download.add_argument("--python", dest="python_executable", default=sys.executable)
    download.add_argument("--execute", action="store_true")
    bind = commands.add_parser("bind")
    bind.add_argument("--selection", required=True)
    bind.add_argument("--download-manifest", required=True)
    bind.add_argument("--output-dir", required=True)
    prepare = commands.add_parser("prepare", help="Fetch original data and prepare selected failures; no model calls")
    for name in ("selection", "routes", "gym-root", "output-dir"):
        prepare.add_argument("--" + name, required=True)
    prepare.add_argument("--python", dest="python_executable", default=sys.executable)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--job", required=True)
    worker.add_argument("--expected-job-sha256", required=True)
    worker.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            result = download_sources(
                args.routes,
                args.gym_root,
                args.output_dir,
                python_executable=args.python_executable,
                execute=args.execute,
            )
        elif args.command == "prepare":
            result = prepare_failed_tasks(
                args.selection, args.routes, args.gym_root, args.output_dir, python_executable=args.python_executable
            )
        elif args.command == "bind":
            result = bind_selection(args.selection, args.download_manifest, args.output_dir)
        else:
            job = rerun._load(args.job)
            if rerun.canonical_sha256(job) != args.expected_job_sha256:
                raise ValueError("Native download worker request changed")
            result = download_original_source(job, preflight_only=args.preflight_only)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(
        args.command == "download"
        and args.execute
        and any(job["status"] not in ("completed", "cached") for job in result["jobs"])
    )


if __name__ == "__main__":
    sys.exit(main())
