# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export ordered CAD draws, warm or prepare a range, and verify cache readiness."""

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from resources_servers.deepcad.dataset.lazy_inputs import (
    LazyRenderConfig,
    cache_target,
    materialize_one,
    resolution_target,
)
from resources_servers.deepcad.dataset.source_operation_sampler import (
    SourceOperationSampler,
    SourceOperationSamplingConfig,
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(
    manifest: Path, output: Path, sampler_config: Path, overrides: Path | None, count: int, batch_size: int
) -> None:
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
    catalogs = set()
    with manifest.open() as stream:
        for index, line in enumerate(stream):
            if index in wanted:
                row = json.loads(line)
                if any(key.startswith("_") for key in row):
                    raise ValueError("Expected raw lazy descriptors without transient runtime IDs")
                lazy = row["deepcad_lazy"]
                config = {**lazy["config"], **deployment, "warm_start_cache": True}
                lazy["config"] = LazyRenderConfig.model_validate(config).model_dump()
                catalogs.add(lazy["config"]["reference_catalog"])
                encoded[index] = json.dumps(row, separators=(",", ":")) + "\n"
    if len(encoded) != len(wanted) or len(catalogs) != 1:
        raise ValueError("Incomplete source selection or inconsistent reference catalog")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cad-export-", dir=output.parent) as work:
        staged = Path(work) / "draws"
        staged.mkdir()
        with (staged / "draws.jsonl").open("w") as draws:
            for index in indices:
                draws.write(encoded[index])
        indices.tofile(staged / "source-indices.i64")
        info = {
            "version": 1,
            "ready": True,
            "ordered": True,
            "draw_count": count,
            "batch_size": batch_size,
            "seed": sampling.seed,
            "sampler_config": sampling.model_dump(),
            "sampler_fingerprint": sampler.fingerprint,
            "draws_sha256": sha256(staged / "draws.jsonl"),
            "source_manifest_sha256": sha256(manifest),
            "source_indices_sha256": sha256(staged / "source-indices.i64"),
            "reference_catalog_sha256": sha256(Path(next(iter(catalogs)))) if None not in catalogs else None,
            "overrides_sha256": sha256(overrides) if overrides else None,
            "rendered_during_export": 0,
            "gym_sources": {
                name: sha256(Path(__file__).with_name(name))
                for name in ("lazy_inputs.py", "source_operation_sampler.py", "sampled_lazy.py")
            },
        }
        (staged / "draws-manifest.json").write_text(json.dumps(info, indent=2) + "\n")
        staged.rename(output)


def read_range(output: Path, start: int, count: int) -> Iterator[dict[str, Any]]:
    info = json.loads((output / "draws-manifest.json").read_text())
    if start < 0 or count < 1 or start + count > info["draw_count"]:
        raise ValueError("Requested range is outside the exported draws")
    draws = output / "draws.jsonl"
    if sha256(draws) != info["draws_sha256"]:
        raise ValueError("Ordered draw manifest checksum mismatch")
    for name, digest in info.get("gym_sources", {}).items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"Gym source changed after descriptor export: {name}")
    with draws.open() as stream:
        for line in islice(stream, start, start + count):
            yield json.loads(line)


def verify(output: Path, start: int, count: int) -> dict[str, Any]:
    checked = 0
    for example in read_range(output, start, count):
        materialize_one(example, cache_only=True)
        checked += 1
    if checked != count:
        raise ValueError("Incomplete draw range")
    return {
        "start": start,
        "count": checked,
        "fresh_process_cache_hits": True,
        "draws_sha256": json.loads((output / "draws-manifest.json").read_text())["draws_sha256"],
    }


def prepare(output: Path, destination: Path, *, start: int, count: int, cache_only: bool = False) -> dict[str, Any]:
    """Publish an ordered range as ordinary rows and content-addressed assets."""
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    draws = read_range(output, start, count)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cad-prepare-", dir=destination.parent) as work:
        staged = Path(work) / "prepared"
        assets = staged / "assets"
        assets.mkdir(parents=True)
        hashes = {}

        def save_asset(data: bytes, suffix: str) -> str:
            digest = hashlib.sha256(data).hexdigest()
            name = f"{digest[:2]}/{digest}{suffix}"
            if name not in hashes:
                target = assets / name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(data)
                hashes[name] = digest
            return name

        written = 0
        with (staged / "rows.jsonl").open("w") as stream:
            for example in draws:
                config = LazyRenderConfig.model_validate(example["deepcad_lazy"]["config"])
                if config.harness != "simple" or example.get("target_images"):
                    raise ValueError("Prepared asset paths require the simple harness without target_images")
                row = deepcopy(materialize_one(example, cache_only=cache_only))
                if row.get("target_images"):
                    raise ValueError("Prepared asset paths require the simple harness without target_images")
                lazy = example["deepcad_lazy"]
                choice = resolution_target(lazy)
                record = json.loads(choice.read_text())["record"] if choice.exists() else lazy["record"]
                if (
                    record["task_id"] != row["verifier_metadata"]["task_id"]
                    or Path(record["reference_step"]).name != row["verifier_metadata"]["reference_step"]
                ):
                    raise ValueError("Resolved row does not match its pinned reference")
                root = Path(config.dataset_root).resolve()
                reference = (root / record["reference_step"]).resolve()
                if not reference.is_relative_to(root):
                    raise ValueError("Reference STEP must remain inside dataset_root")
                reference_bytes = reference.read_bytes()
                expected = (
                    json.loads(reference.with_suffix(".json").read_text())["step_sha256"]
                    if record.get("source_kind") == "cad_corpus"
                    else record["reference_step_sha256"]
                )
                if hashlib.sha256(reference_bytes).hexdigest() != expected:
                    raise ValueError(f"Reference STEP checksum mismatch: {reference}")
                row["verifier_metadata"]["reference_step"] = save_asset(reference_bytes, ".step")
                for item in row["responses_create_params"]["input"]:
                    if not isinstance(item.get("content"), list):
                        continue
                    for part in item["content"]:
                        if part.get("type") != "input_image":
                            continue
                        url = part["image_url"]
                        header, encoded = url.split(",", 1)
                        suffix = {
                            "data:image/png;base64": ".png",
                            "data:image/jpeg;base64": ".jpg",
                            "data:image/webp;base64": ".webp",
                        }[header]
                        data = base64.b64decode(encoded, validate=True)
                        if not data:
                            raise ValueError("Prepared images must be nonempty")
                        with Image.open(BytesIO(data)) as image:
                            image.verify()
                        name = save_asset(data, suffix)
                        part["image_url"] = str(destination / "assets" / name)
                if "deepcad_lazy" in row:
                    raise ValueError("Prepared rows must not contain lazy descriptors")
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                written += 1
        if written != count:
            raise ValueError("Incomplete draw range")
        receipt = {
            "version": 1,
            "start": start,
            "count": written,
            "draws_sha256": sha256(output / "draws.jsonl"),
            "draw_manifest_sha256": sha256(output / "draws-manifest.json"),
            "rows_sha256": sha256(staged / "rows.jsonl"),
            "assets_sha256": hashes,
        }
        (assets / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
        staged.rename(destination)
    return receipt


def warm_one(example: dict[str, Any]) -> dict[str, Any]:
    config = LazyRenderConfig.model_validate(example["deepcad_lazy"]["config"])
    requested = cache_target(example["deepcad_lazy"])
    choice = resolution_target(example["deepcad_lazy"])
    lazy = deepcopy(example["deepcad_lazy"])
    if choice.exists():
        lazy["record"] = json.loads(choice.read_text())["record"]
    hit = cache_target(lazy).exists()
    started = time.monotonic()
    row = materialize_one(example)
    if choice.exists():
        lazy["record"] = json.loads(choice.read_text())["record"]
    target = cache_target(lazy)
    reference = Path(config.dataset_root) / lazy["record"]["reference_step"]
    return {
        "requested_task_id": example["deepcad_lazy"]["record"]["task_id"],
        "resolved_task_id": row["verifier_metadata"]["task_id"],
        "cache_key": target.stem,
        "cache_path": str(target),
        "cache_sha256": sha256(target),
        "requested_cache_path": str(requested),
        "resolution_path": str(choice) if choice.exists() else None,
        "resolution_sha256": sha256(choice) if choice.exists() else None,
        "reference_path": str(reference),
        "reference_sha256": sha256(reference),
        "cache_hit": hit,
        "elapsed_seconds": time.monotonic() - started,
    }


def verify_fresh(output: Path, start: int, count: int) -> dict[str, Any]:
    """Verify the requested range in another interpreter without rendering."""
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
    return json.loads(verified.stdout)


def warm(output: Path, start: int, count: int, workers: int) -> dict[str, Any]:
    """Publish first-draw readiness, then verify the complete bounded warm range."""
    if not 1 <= workers <= 64:
        raise ValueError("Worker count must be between 1 and 64")
    draws = read_range(output, start, count)
    first_draw = next(draws)
    config = LazyRenderConfig.model_validate(first_draw["deepcad_lazy"]["config"])
    entries = [{"draw_index": start, **warm_one(first_draw)}]
    info_path = output / "draws-manifest.json"
    info = json.loads(info_path.read_text())
    receipt = {
        "version": 1,
        "ready": True,
        "start_index": start,
        "draws_sha256": info["draws_sha256"],
        "draw_manifest_sha256": sha256(info_path),
        "overrides_sha256": info.get("overrides_sha256"),
        "cache_root": config.cache_root,
        "persistent_reference_root": str(Path(config.dataset_root) / "references"),
    }
    first = output / f"draw{start}-ready.json"
    if not first.exists():
        with tempfile.NamedTemporaryFile(mode="w", dir=output, delete=False) as temp:
            json.dump(
                {**receipt, "count": 1, "entries": entries[:], "cache_verification": verify_fresh(output, start, 1)},
                temp,
            )
        Path(temp.name).replace(first)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while batch := list(islice(draws, workers * 4)):
            keys = [json.dumps(draw["deepcad_lazy"], sort_keys=True) for draw in batch]
            unique = dict(zip(keys, batch))
            warmed = dict(zip(unique, pool.map(warm_one, unique.values()), strict=True))
            for key in keys:
                entries.append({"draw_index": start + len(entries), **warmed[key]})
    verified = verify_fresh(output, start, count)
    receipt.update(verified)
    receipt.update(entries=entries, cache_verification=verified)
    with tempfile.NamedTemporaryFile(mode="w", dir=output, delete=False) as temp:
        json.dump(receipt, temp)
    Path(temp.name).replace(output / f"warm-{start}-{count}-ready.json")
    if start == 0 and count == 1024 and not (output / "warm1024-ready.json").exists():
        with tempfile.NamedTemporaryFile(mode="w", dir=output, delete=False) as temp:
            json.dump(receipt, temp)
        Path(temp.name).replace(output / "warm1024-ready.json")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("export")
    select.add_argument("--manifest", type=Path, required=True)
    select.add_argument("--sampler-config", type=Path, required=True)
    select.add_argument("--overrides", type=Path)
    select.add_argument("--batch-size", type=int, required=True)
    warmer = commands.add_parser("warm")
    warmer.add_argument("--workers", type=int, default=8)
    prepared = commands.add_parser("prepare")
    prepared.add_argument("--destination", type=Path, required=True)
    prepared.add_argument("--cache-only", action="store_true")
    check = commands.add_parser("verify")
    for command in (select, warmer, prepared, check):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--count", type=int, required=True)
    for command in (warmer, prepared, check):
        command.add_argument("--start", type=int, default=0)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = {"export": export, "warm": warm, "prepare": prepare, "verify": verify}[command](**args)
    if result:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
