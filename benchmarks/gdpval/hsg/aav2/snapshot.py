#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze source and inputs for the three AA-v2 jobs; submit no jobs."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


if __package__:
    from .completion import rollout_complete
    from .source_copy import copy_tree
else:
    from completion import rollout_complete
    from source_copy import copy_tree


PACKAGE = Path("benchmarks/gdpval/hsg/aav2")
FILES = (
    "run_aav2.sh",
    "aav2_rollout.sbatch",
    "aav2_preconvert.sbatch",
    "aav2_judge.sbatch",
    "snapshot.py",
    "completion.py",
    "node_local.sh",
    "rollout_runtime.py",
    "rollout_serving.py",
    "media.py",
    "preconvert.py",
    "source_copy.py",
    "true3_transport.yaml",
)
BYTE_LIMITS_MIB = {
    "FILE_BYTES": 250,
    "TOTAL_RAW_ATTACHMENT_BYTES": 300,
    "TOTAL_ENCODED_ATTACHMENT_CHARS": 400,
    "SECTION_RAW_ATTACHMENT_BYTES": 96,
    "SECTION_ENCODED_ATTACHMENT_CHARS": 128,
    "TOTAL_SERIALIZED_REQUEST_BYTES": 420,
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(args: argparse.Namespace) -> Path:
    destination = args.run_dir.expanduser().resolve()
    if destination.exists():
        raise ValueError("run directory exists; resume its phase or choose a fresh directory")
    if not re.fullmatch(r"[A-Za-z0-9_./ -]+", str(destination)):
        raise ValueError("run path must use letters, digits, spaces, '.', '_', '-', '/'")
    source = args.source.resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.revision}^{{commit}}"], cwd=source, text=True
    ).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source):
        raise ValueError("commit tracked source changes before preparing a run")
    if args.concurrency < 1 or args.agent_max_turns < 1:
        raise ValueError("concurrency and agent turns must be positive")
    if not args.existing_rollout and not args.profile:
        raise ValueError("rollout requires --profile; imports use --existing-rollout")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".aav2-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir(mode=0o700)
        with (stage / "source.tar").open("xb") as archive:
            subprocess.run(["git", "archive", revision], cwd=source, stdout=archive, check=True)
        (stage / "package").mkdir()
        with tarfile.open(stage / "source.tar") as archive:
            for name in FILES:
                member = archive.getmember(str(PACKAGE / name))
                if not member.isfile():
                    raise ValueError(f"helper must be a tracked regular file: {name}")
                with archive.extractfile(member) as stream, (stage / "package" / name).open("xb") as output:
                    shutil.copyfileobj(stream, output)
        for argument, name in (
            (args.dataset, "dataset.jsonl"),
            (args.judge_config, "judge.yaml"),
            (args.profile, "serving.env"),
        ):
            if argument is not None:
                shutil.copyfile(argument.resolve(strict=True), stage / name)
        rows = [json.loads(line) for line in (stage / "dataset.jsonl").read_text().splitlines() if line.strip()]
        ids = [row.get("task_id") for row in rows]
        if not ids or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in ids):
            raise ValueError("dataset must have nonempty path-safe task IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("dataset task IDs must be unique")
        smoke_rows = rows[:4]
        if args.smoke_dataset:
            smoke_rows = [json.loads(line) for line in args.smoke_dataset.read_text().splitlines() if line.strip()]
        smoke_ids = [row.get("task_id") for row in smoke_rows]
        if not smoke_ids or len(set(smoke_ids)) != len(smoke_ids) or not set(smoke_ids) <= set(ids):
            raise ValueError("smoke dataset must be a nonempty unique subset of the rollout dataset")
        canonical = {row["task_id"]: row for row in rows}
        if any(row != canonical[row["task_id"]] for row in smoke_rows):
            raise ValueError("smoke rows must match the canonical dataset, including prompts and reference URLs")
        (stage / "smoke_dataset.jsonl").write_text("".join(json.dumps(row) + "\n" for row in smoke_rows[:4]))
        if args.existing_rollout:
            original = args.existing_rollout.absolute() / "deliverables"
            receipt = copy_tree(original, stage / "deliverables", {f"task_{task_id}" for task_id in ids})
            rollout_complete(stage / "dataset.jsonl", stage / "deliverables")
            (stage / "import.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        environment = {
            "RUN_DIR": str(destination),
            "DATASET": str(destination / "dataset.jsonl"),
            "SMOKE_DATASET": str(destination / "smoke_dataset.jsonl"),
            "JUDGE_CONFIG": str(destination / "judge.yaml"),
            "PROFILE": str(destination / "serving.env") if args.profile else "",
            "EXISTING_ROLLOUT": str(args.existing_rollout.absolute()) if args.existing_rollout else "",
            "ENV_FILE": str(args.env_file.resolve(strict=True)),
            "GYM_REVISION": revision,
            "UV_SOURCE": str(args.uv_source.expanduser().resolve(strict=True)),
            "AGENT_SIF": str(args.agent_sif.resolve(strict=True)),
            "JUDGE_SIF": str((args.judge_sif or args.agent_sif).resolve(strict=True)),
            "APPTAINER_BIN": str(args.apptainer_bin.resolve(strict=True)),
            "CONCURRENCY": str(args.concurrency),
            "AGENT_MAX_TURNS": str(args.agent_max_turns),
        }
        for limit, default_mib in BYTE_LIMITS_MIB.items():
            name = f"GDPVAL_MAX_{limit}_FOR_JUDGE"
            value = int(os.environ.get(name, default_mib * 1024 * 1024))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            environment[name] = str(value)
        (stage / "run.env").write_text(
            "".join(f"export {key}={shlex.quote(value)}\n" for key, value in environment.items())
        )
        (stage / "run.json").write_text(json.dumps(environment, indent=2) + "\n")
        files = sorted(path for path in stage.rglob("*") if path.is_file())
        (stage / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.relative_to(stage)}\n" for path in files))
        for path in [*files, stage / "SHA256SUMS"]:
            path.chmod(0o400)
        stage.rename(destination)
    return destination


def verify(run_dir: Path) -> None:
    for line in (run_dir / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = run_dir / name
        if path.is_symlink() or not path.resolve().is_relative_to(run_dir.resolve()) or digest(path) != expected:
            raise ValueError(f"prepared input changed: {name}; use a fresh run directory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("run_dir", type=Path)
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--revision", default="HEAD")
    prepare_parser.add_argument("--dataset", type=Path, required=True)
    prepare_parser.add_argument("--smoke-dataset", type=Path)
    prepare_parser.add_argument("--profile", type=Path)
    prepare_parser.add_argument("--existing-rollout", type=Path, help="Copy only deliverables from a finished rollout")
    prepare_parser.add_argument("--judge-config", type=Path, required=True)
    prepare_parser.add_argument("--env-file", type=Path, required=True)
    prepare_parser.add_argument("--uv-source", type=Path, required=True)
    prepare_parser.add_argument("--agent-sif", type=Path, required=True)
    prepare_parser.add_argument("--judge-sif", type=Path, help="Office/media image; defaults to the agent image")
    prepare_parser.add_argument("--apptainer-bin", type=Path, required=True)
    prepare_parser.add_argument("--concurrency", type=int, default=8)
    prepare_parser.add_argument("--agent-max-turns", type=int, default=250)
    commands.add_parser("verify").add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            print(prepare(args))
        else:
            verify(args.run_dir)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
