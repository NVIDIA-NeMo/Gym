# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export ordered CAD draws, warm a bounded range and verify it without rendering."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np

from resources_servers.deepcad.dataset.lazy_inputs import (
    LazyRenderConfig,
    UnconfirmedReferenceFailure,
    materialize_one,
)
from resources_servers.deepcad.dataset.source_operation_sampler import (
    SourceOperationSampler,
    SourceOperationSamplingConfig,
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(manifest: Path, output: Path, sampler_config: Path, overrides: Path | None, count: int, batch_size: int):
    if count < 1:
        raise ValueError("Draw count must be positive")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    settings = json.loads(sampler_config.read_text())
    settings["samples_per_epoch"] = count
    sampling = SourceOperationSamplingConfig.model_validate(settings)
    deployment = json.loads(overrides.read_text()) if overrides else {}
    with manifest.open() as stream:
        sampler = SourceOperationSampler(stream, sampling, batch_size)
    indices = np.fromiter(iter(sampler), dtype="<i8", count=count)
    wanted = set(indices.tolist())
    encoded = {}
    with manifest.open() as stream:
        for index, line in enumerate(stream):
            if index in wanted:
                row = json.loads(line)
                lazy = row["deepcad_lazy"]
                config = {**lazy["config"], **deployment, "warm_start_cache": True}
                lazy["config"] = LazyRenderConfig.model_validate(config).model_dump()
                encoded[index] = json.dumps(row, separators=(",", ":")) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cad-export-", dir=output.parent) as work:
        staged = Path(work) / "draws"
        staged.mkdir()
        with (staged / "draws.jsonl").open("w") as draws:
            for index in indices:
                draws.write(encoded[index])
        indices.tofile(staged / "source-indices.i64")
        info = {
            "draw_count": count,
            "batch_size": batch_size,
            "sampler_config": sampling.model_dump(),
            "sampler_fingerprint": sampler.fingerprint,
            "draws_sha256": sha256(staged / "draws.jsonl"),
            "source_manifest_sha256": sha256(manifest),
        }
        (staged / "draws-manifest.json").write_text(json.dumps(info, indent=2) + "\n")
        staged.rename(output)


def read_range(output: Path, start: int, count: int):
    info = json.loads((output / "draws-manifest.json").read_text())
    if start < 0 or count < 1 or start + count > info["draw_count"]:
        raise ValueError("Requested range is outside the exported draws")
    draws = output / "draws.jsonl"
    if sha256(draws) != info["draws_sha256"]:
        raise ValueError("Ordered draw manifest checksum mismatch")
    with draws.open() as stream:
        for line in islice(stream, start, start + count):
            yield json.loads(line)


def verify(output: Path, start: int, count: int):
    checked = 0
    for example in read_range(output, start, count):
        materialize_one(example, cache_only=True)
        checked += 1
    if checked != count:
        raise ValueError("Incomplete draw range")
    return {
        "start": start,
        "count": checked,
        "draws_sha256": json.loads((output / "draws-manifest.json").read_text())["draws_sha256"],
    }


def warm_one(example):
    config = LazyRenderConfig.model_validate(example["deepcad_lazy"]["config"])
    for attempt in range(config.max_reference_replacements + 2):
        try:
            materialize_one(example)
            return
        except UnconfirmedReferenceFailure:
            if attempt == config.max_reference_replacements + 1:
                raise


def warm(output: Path, start: int, count: int, workers: int):
    if not 1 <= workers <= 64:
        raise ValueError("Worker count must be between 1 and 64")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        draws = read_range(output, start, count)
        while batch := list(islice(draws, workers * 4)):
            for _ in pool.map(warm_one, batch):
                pass
    verified = subprocess.run(
        [
            sys.executable,
            "-m",
            "resources_servers.deepcad.dataset.sampled_lazy",
            "verify",
            "--output",
            str(output),
            "--start",
            str(start),
            "--count",
            str(count),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    receipt = json.loads(verified.stdout)
    with tempfile.NamedTemporaryFile(mode="w", dir=output, delete=False) as temp:
        json.dump(receipt, temp)
    Path(temp.name).replace(output / f"warm-{start}-{count}-ready.json")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("export")
    select.add_argument("--manifest", type=Path, required=True)
    select.add_argument("--sampler-config", type=Path, required=True)
    select.add_argument("--overrides", type=Path)
    select.add_argument("--batch-size", type=int, required=True)
    prepare = commands.add_parser("warm")
    prepare.add_argument("--workers", type=int, default=8)
    check = commands.add_parser("verify")
    for command in (select, prepare, check):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--count", type=int, required=True)
    for command in (prepare, check):
        command.add_argument("--start", type=int, default=0)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = {"export": export, "warm": warm, "verify": verify}[command](**args)
    if result:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
