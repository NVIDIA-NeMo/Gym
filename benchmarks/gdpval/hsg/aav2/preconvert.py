#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare candidate/reference copies with native Office renders and media proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath

from benchmarks.gdpval.hsg.aav2.completion import read_rows, rollout_complete, task_ids
from benchmarks.gdpval.hsg.aav2.media import build, tree_paths
from benchmarks.gdpval.hsg.aav2.source_copy import copy_tree
from benchmarks.gdpval.hsg.aav2.source_copy import inventory as source_inventory
from resources_servers.gdpval.preconvert import find_convertible_files, preconvert_dir


REFERENCE_KEY = "gdpval_resources_server.resources_servers.gdpval.reference_models"
CACHE_FILE = re.compile(r"repeat_[0-9]+_verify_response(?:_(?:[0-9a-f]{12}|[0-9a-f]{16}))?\.json$")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path, *, prepared: bool = False, top_level_names: set[str] | None = None) -> dict:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"expected a real input directory: {root}")
    files = {}
    for path in tree_paths(root, top_level_names):
        relative = path.relative_to(root)
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"unsupported symlink/special file: {path}")
        if path.is_dir():
            continue
        if prepared and (
            relative.as_posix() == "manifest.json"
            or (
                len(relative.parts) == 3
                and relative.parts[0] == "candidate"
                and relative.parts[1].startswith("task_")
                and CACHE_FILE.fullmatch(path.name)
            )
        ):
            continue
        files[relative.as_posix()] = digest(path)
    return files


def office(root: Path) -> None:
    ok, failed, errors = preconvert_dir(root, max_concurrent=4)
    print(f"Office: converted={ok}, failed={failed}", flush=True)
    if failed or find_convertible_files(root):
        raise ValueError("Office conversion incomplete: " + "; ".join(errors[:5]))


def reference_repeats(task: Path) -> list[Path]:
    if not task.is_dir():
        return []
    repeats = sorted(path for path in task.iterdir() if path.is_dir() and path.name.startswith("repeat_"))
    completed = [path for path in (repeats or [task]) if (path / "finish_params.json").is_file()]
    if len(completed) != 1:
        raise ValueError(f"expected exactly one completed reference repeat: {task}; found {len(completed)}")
    return completed


def require_reference_files(root: Path, dataset: Path) -> None:
    for row in read_rows(dataset):
        names = row.get("reference_files") or []
        if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("dataset reference_files must be a list of relative paths")
        expected = []
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
                raise ValueError(f"unsafe dataset reference path: {name!r}")
            expected.append(path if path.parts[0] == "reference_files" else PurePosixPath("reference_files") / path)
        for repeat in reference_repeats(root / f"task_{row['task_id']}"):
            for path in expected:
                # Earlier persisted runs wrapped the corpus prefix a second time.
                if not (repeat / path).is_file() and not (repeat / "reference_files" / path).is_file():
                    raise ValueError(f"missing dataset reference file: {repeat / path}")


def prepare(run_dir: Path, *, config_json: Path | None = None) -> Path:
    run_dir = run_dir.resolve(strict=True)
    settings = json.loads((run_dir / "run.json").read_text())
    if Path(settings["RUN_DIR"]).resolve() != run_dir:
        raise ValueError("run.json does not belong to this run directory")
    dataset = Path(settings["DATASET"])
    selected_ids = task_ids(dataset)
    task_names = {f"task_{task_id}" for task_id in selected_ids}
    rollout_complete(dataset, run_dir / "deliverables")
    judge_path = Path(settings["JUDGE_CONFIG"])
    if config_json is None:
        from omegaconf import OmegaConf

        config = OmegaConf.to_container(OmegaConf.load(judge_path), resolve=False)
    else:
        config = json.loads(config_json.read_text())
    references = config
    for key in REFERENCE_KEY.split("."):
        references = references.get(key, {})
    if not references:
        raise ValueError("judge config must define reference_models with absolute deliverables_dir paths")
    sources = {"candidate": run_dir / "deliverables"}
    for name, reference in references.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"unsafe reference model ID: {name}")
        root = Path(reference["deliverables_dir"])
        if not root.is_absolute():
            raise ValueError(f"reference root must be an absolute directory: {name}")
        for task_name in sorted(task_names):
            reference_repeats(root / task_name)
        sources[f"references/{name}"] = root
    output = run_dir / "prepared"
    if any(
        output.is_relative_to(root.resolve()) or root.resolve().is_relative_to(output) for root in sources.values()
    ):
        raise ValueError("prepared output and original trees must be disjoint")
    inputs = {
        "task_ids": sorted(selected_ids),
        "dataset_sha256": digest(dataset),
        "judge_config_sha256": digest(judge_path),
        "sources": {name: source_inventory(root, task_names) for name, root in sources.items()},
    }
    if output.exists() or output.is_symlink():
        if output.is_symlink():
            raise ValueError("prepared output cannot be a symlink")
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["inputs"] != inputs or manifest["outputs"] != inventory(output, prepared=True):
            raise ValueError("prepared input/output changed; use a fresh run directory")
        print("Prepared inputs verified; reusing completed preconversion", flush=True)
        return output
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=run_dir) as temporary:
        staged = Path(temporary) / "prepared"
        staged.mkdir()
        reference_copies = {}
        for name, root in sources.items():
            destination = staged / name
            print(f"Preparing {name}", flush=True)
            if name == "candidate":
                build(root, destination, prepare_zip=office, top_level_names=task_names | {"FILTER_MANIFEST.txt"})
            else:
                raw = Path(temporary) / "raw" / name
                reference_copies[name] = copy_tree(root, raw, task_names)
                require_reference_files(raw, dataset)
                build(raw, destination, prepare_zip=office)
            office(destination)
        for name in references:
            references[name]["deliverables_dir"] = str(output / "references" / name)
        agent = config
        for key in ("gdpval_stirrup_agent", "responses_api_agents", "stirrup_agent"):
            agent = agent.setdefault(key, {})
        agent["persist_deliverables_dir"] = str(output / "candidate")
        # JSON is valid YAML; interpolation strings remain unresolved.
        (staged / "judge.yaml").write_text(json.dumps(config, indent=2) + "\n")
        if (
            inputs["dataset_sha256"] != digest(dataset)
            or inputs["judge_config_sha256"] != digest(judge_path)
            or any(source_inventory(root, task_names) != inputs["sources"][name] for name, root in sources.items())
        ):
            raise ValueError("original inputs changed during preconversion")
        manifest = {
            "inputs": inputs,
            "reference_copies": reference_copies,
            "outputs": inventory(staged, prepared=True),
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.rename(staged, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config-json", type=Path, help="Unresolved judge YAML decoded by the host Gym environment")
    args = parser.parse_args()
    print(prepare(args.run_dir, config_json=args.config_json))
