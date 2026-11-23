# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline sampled CAD task releases, with resumable packed render chunks."""

import argparse
import base64
import hashlib
import json
import shutil
import sqlite3
import tarfile
import tempfile
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from resources_servers.deepcad.dataset.lazy_inputs import (
    LazyRenderConfig,
    materialize_one,
    validate_row,
)
from resources_servers.deepcad.dataset.source_operation_sampler import (
    SourceOperationSampler,
    SourceOperationSamplingConfig,
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def validate_materialized_row(row: dict[str, Any]) -> None:
    """Reject lazy, incomplete or run-specific task inputs before publication."""
    if "deepcad_lazy" in row or any(key.startswith("_") for key in row):
        raise ValueError("A materialized task must not contain lazy or per-run fields")
    metadata = row["verifier_metadata"]
    reference = Path(metadata["reference_step"])
    if reference.is_absolute() or ".." in reference.parts:
        raise ValueError("Reference STEP must be a contained relative path")
    record = {key: metadata[key] for key in ("task_id", "source_sha256", "difficulty")}
    record["reference_step"] = reference.name
    checked = deepcopy(row)
    checked["verifier_metadata"]["reference_step"] = reference.name
    validate_row(checked, record)
    for item in row["responses_create_params"]["input"]:
        if not isinstance(item.get("content"), list):
            continue
        for part in item["content"]:
            if part.get("type") == "input_image":
                header, encoded = part["image_url"].split(",", 1)
                decoded = base64.b64decode(encoded, validate=True)
                if not header.endswith(";base64") or not decoded:
                    raise ValueError("Materialized images require nonempty base64 data")
                try:
                    with Image.open(BytesIO(decoded)) as image:
                        image.verify()
                except (OSError, SyntaxError) as exc:
                    raise ValueError("Materialized image bytes cannot be decoded") from exc


def select(
    manifest: Path,
    output: Path,
    sampler_config: Path,
    num_samples: int,
    batch_size: int,
    seed: int,
    render_overrides: Path | None = None,
) -> None:
    """Freeze actual sampler draws without executing a CAD program."""
    if batch_size <= 0 or num_samples <= 0 or num_samples % batch_size:
        raise ValueError("num_samples must be a positive multiple of batch_size")
    if output.exists():
        raise FileExistsError(output)
    destination = output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as work:
        output = Path(work)
        config = SourceOperationSamplingConfig.model_validate(json.loads(sampler_config.read_text()))
        config = config.model_copy(update={"seed": seed, "samples_per_epoch": num_samples})
        with manifest.open() as raw:
            sampler = SourceOperationSampler(raw, config, batch_size)
        indices = list(iter(sampler))
        wanted = set(indices)
        selected = {index: row for index, row in enumerate(read_jsonl(manifest)) if index in wanted}
        overrides = json.loads(render_overrides.read_text()) if render_overrides else {}
        path = output / "selected.jsonl"
        catalogs = set()
        with path.open("w") as stream:
            for draw_index, source_index in enumerate(indices):
                row = deepcopy(selected[source_index])
                row["deepcad_lazy"]["config"].update(overrides)
                row["deepcad_lazy"]["config"] = LazyRenderConfig.model_validate(
                    row["deepcad_lazy"]["config"]
                ).model_dump()
                catalogs.add(row["deepcad_lazy"]["config"]["reference_catalog"])
                if any(key.startswith("_") for key in row):
                    raise ValueError("Select raw manifest rows, not already dispatched tasks")
                stream.write(
                    json.dumps(
                        {
                            "draw_index": draw_index,
                            "source_index": source_index,
                            "example": row,
                        }
                    )
                    + "\n"
                )
        if len(catalogs) != 1 or None in catalogs:
            raise ValueError("Select requires one pinned exclusion-aware reference catalog")
        catalog = Path(next(iter(catalogs)))
        write_json(
            output / "selection.json",
            {
                "version": 1,
                "num_samples": num_samples,
                "batch_size": batch_size,
                "source_manifest_sha256": sha256(manifest),
                "sampler_fingerprint": sampler.fingerprint,
                "sampler_config": config.model_dump(),
                "selected_sha256": sha256(path),
                "sampler_source_sha256": sha256(Path(__file__).with_name("source_operation_sampler.py")),
                "builder_source_sha256": sha256(Path(__file__)),
                "render_overrides": overrides,
                "reference_catalog_sha256": sha256(catalog),
                "semantics": "finite sampled occurrence pool; duplicate draws retained",
            },
        )
        output.rename(destination)


def load_selection(selection: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((selection / "selection.json").read_text())
    if sha256(selection / "selected.jsonl") != metadata["selected_sha256"]:
        raise ValueError("Frozen selection checksum mismatch")
    rows = list(read_jsonl(selection / "selected.jsonl"))
    if len(rows) != metadata["num_samples"] or [row["draw_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("Frozen selection occurrence order/count mismatch")
    return metadata, rows


def _resolve(example: dict[str, Any]) -> dict[str, Any]:
    config = LazyRenderConfig.model_validate(example["deepcad_lazy"]["config"])
    for attempt in range(config.max_reference_replacements + 2):
        try:
            return materialize_one(example)
        except RuntimeError as exc:
            if (
                not str(exc).startswith("Unconfirmed reference failure; retry in a fresh renderer:")
                or attempt == config.max_reference_replacements + 1
            ):
                raise
    raise AssertionError("Unreachable reference confirmation state")


def render(
    selection: Path,
    output: Path,
    shard_index: int,
    num_shards: int,
    workers: int,
    chunk_size: int,
    render_overrides: Path | None = None,
    max_chunks: int | None = None,
) -> None:
    """Publish complete chunks only; existing valid chunks are reusable."""
    if not 0 <= shard_index < num_shards or workers < 1 or chunk_size < 1:
        raise ValueError("Invalid render shard, worker count or chunk size")
    metadata, selected = load_selection(selection)
    if sha256(Path(__file__)) != metadata["builder_source_sha256"]:
        raise ValueError("Materialization builder changed after frozen selection")
    overrides = json.loads(render_overrides.read_text()) if render_overrides else {}
    first_config = {**selected[0]["example"]["deepcad_lazy"]["config"], **overrides}
    catalog = Path(first_config["reference_catalog"])
    if sha256(catalog) != metadata["reference_catalog_sha256"]:
        raise ValueError("Reference replacement catalog changed after selection")
    output.mkdir(parents=True, exist_ok=True)
    completed = 0
    for chunk_index, start in enumerate(range(0, len(selected), chunk_size)):
        if chunk_index % num_shards != shard_index:
            continue
        if max_chunks is not None and completed >= max_chunks:
            break
        completed += 1
        stem = output / f"chunk-{chunk_index:06d}"
        receipt_path = stem.with_suffix(".receipt.json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt["selection_sha256"] != metadata["selected_sha256"]
                or receipt["start"] != start
                or receipt["chunk_size"] != chunk_size
                or receipt["render_overrides"] != overrides
            ):
                raise ValueError(f"Chunk settings changed: {stem}")
            for name, checksum in receipt["artifacts"].items():
                if sha256(output / name) != checksum:
                    raise ValueError(f"Completed chunk is corrupt: {name}")
            continue
        draws = deepcopy(selected[start : start + chunk_size])
        unique = {}
        for draw in draws:
            example = draw["example"]
            example["deepcad_lazy"]["config"].update(overrides)
            example["deepcad_lazy"]["config"] = LazyRenderConfig.model_validate(
                example["deepcad_lazy"]["config"]
            ).model_dump()
            key = json.dumps(example, sort_keys=True)
            unique.setdefault(key, example)
            draw["work_key"] = key
        with ThreadPoolExecutor(max_workers=workers) as pool:
            prepared = dict(zip(unique, pool.map(_resolve, unique.values()), strict=True))
        rows_path = stem.with_suffix(".jsonl")
        assets_path = stem.with_suffix(".tar")
        with tempfile.TemporaryDirectory(prefix="cad-publish-") as work:
            rows_local, assets_local = (
                Path(work) / "rows.jsonl",
                Path(work) / "references.tar",
            )
            assets = {}
            with (
                rows_local.open("w") as stream,
                tarfile.open(assets_local, "w") as archive,
            ):
                for draw in draws:
                    row = deepcopy(prepared[draw["work_key"]])
                    config = draw["example"]["deepcad_lazy"]["config"]
                    root = Path(config["dataset_root"]).resolve() / "references"
                    reference = (root / row["verifier_metadata"]["reference_step"]).resolve()
                    if not reference.is_relative_to(root.resolve()) or not reference.is_file():
                        raise ValueError(f"Missing or escaping prepared reference: {reference}")
                    digest = sha256(reference)
                    relative = f"{digest[:2]}/{digest}.step"
                    if relative not in assets:
                        archive.add(reference, arcname=relative, recursive=False)
                        assets[relative] = {
                            "sha256": digest,
                            "size": reference.stat().st_size,
                        }
                    row["verifier_metadata"]["reference_step"] = relative
                    validate_materialized_row(row)
                    record = draw["example"]["deepcad_lazy"]["record"]
                    resolved = row["verifier_metadata"]
                    with sqlite3.connect(f"file:{catalog}?mode=ro", uri=True) as database:
                        eligible = database.execute(
                            "SELECT row_json FROM assets WHERE id=? AND status='selected'",
                            (resolved["asset_id"],),
                        ).fetchone()
                    if eligible is None:
                        raise ValueError("Resolved reference is absent from the pinned eligible catalog")
                    retained = json.loads(eligible[0])["deepcad_lazy"]["record"]
                    if (
                        retained["dataset"] != record["dataset"]
                        or resolved["dataset"] != record["dataset"]
                        or retained["source_sha256"] != resolved["source_sha256"]
                    ):
                        raise ValueError("Resolved reference source/checksum does not match the eligible catalog")
                    trace = {
                        "requested_task_id": record["task_id"],
                        "requested_asset_id": record.get("asset_id"),
                        "requested_source": record.get("dataset"),
                        "requested_operations": record.get("reference_operations", []),
                        "resolved_metadata": row["verifier_metadata"],
                    }
                    stream.write(
                        json.dumps(
                            {
                                "draw_index": draw["draw_index"],
                                "source_index": draw["source_index"],
                                "row": row,
                                "trace": trace,
                            }
                        )
                        + "\n"
                    )
            for local, target in ((rows_local, rows_path), (assets_local, assets_path)):
                temporary = target.with_suffix(target.suffix + ".building")
                shutil.copyfile(local, temporary)
                temporary.replace(target)
        write_json(
            receipt_path,
            {
                "selection_sha256": metadata["selected_sha256"],
                "start": start,
                "count": len(draws),
                "chunk_size": chunk_size,
                "render_overrides": overrides,
                "artifacts": {p.name: sha256(p) for p in (rows_path, assets_path)},
                "references": assets,
            },
        )
        print(
            json.dumps({"chunk": chunk_index, "rows": len(draws), "status": "complete"}),
            flush=True,
        )


def assemble(selection: Path, shards: Path, output: Path, prefix_count: int | None = None) -> None:
    """Assemble a complete release, or an explicitly identified smoke prefix."""
    metadata, selected = load_selection(selection)
    count = metadata["num_samples"] if prefix_count is None else prefix_count
    if not 0 < count <= len(selected) or output.exists():
        raise ValueError("Invalid release count or output already exists")
    receipts = sorted(shards.glob("chunk-*.receipt.json"))
    entries, references, archives = {}, {}, []
    render_settings = None
    for path in receipts:
        receipt = json.loads(path.read_text())
        if receipt["selection_sha256"] != metadata["selected_sha256"]:
            raise ValueError(f"Selection mismatch in {path}")
        if receipt["start"] >= count:
            continue
        if render_settings is None:
            render_settings = receipt["render_overrides"]
        elif render_settings != receipt["render_overrides"]:
            raise ValueError("Render settings differ across completed chunks")
        for name, checksum in receipt["artifacts"].items():
            if sha256(shards / name) != checksum:
                raise ValueError(f"Corrupt chunk: {name}")
            if name.endswith(".tar"):
                archives.append(shards / name)
            else:
                for entry in read_jsonl(shards / name):
                    index = entry["draw_index"]
                    if index < count:
                        if index in entries:
                            raise ValueError(f"Duplicate draw: {index}")
                        entries[index] = entry
        references.update(receipt["references"])
    if sorted(entries) != list(range(count)):
        raise ValueError(f"Incomplete release: found {len(entries)} of {count} occurrences")
    needed = {entry["row"]["verifier_metadata"]["reference_step"] for entry in entries.values()}
    destination = output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as work:
        output = Path(work)
        with (
            (output / "train.materialized.jsonl").open("w") as rows,
            (output / "resolutions.jsonl").open("w") as trace,
        ):
            for index in range(count):
                entry = entries[index]
                validate_materialized_row(entry["row"])
                rows.write(json.dumps(entry["row"]) + "\n")
                trace.write(json.dumps({k: v for k, v in entry.items() if k != "row"}) + "\n")
        pending = set(needed)
        with tarfile.open(output / "references.tar", "w") as target:
            for path in archives:
                with tarfile.open(path) as source:
                    for member in source:
                        if member.name in pending:
                            target.addfile(member, source.extractfile(member))
                            pending.remove(member.name)
        if pending:
            raise ValueError(f"Missing packed references: {len(pending)}")
        write_json(
            output / "references.json",
            {name: references[name] for name in sorted(needed)},
        )
        artifacts = {
            name: sha256(output / name)
            for name in (
                "train.materialized.jsonl",
                "resolutions.jsonl",
                "references.tar",
                "references.json",
            )
        }
        release = {
            "version": 1,
            "ready": True,
            "count": count,
            "is_smoke_prefix": count != len(selected),
            "unique_references": len(needed),
            "selection": metadata,
            "artifacts": artifacts,
            "requested_sources": dict(Counter(entries[i]["trace"]["requested_source"] for i in range(count))),
            "render_overrides": render_settings,
        }
        write_json(output / "release.json", release)
        validate(output)
        output.rename(destination)


def validate(release: Path) -> None:
    """Check schemas, counts, hashes and every packed verifier STEP."""
    receipt = json.loads((release / "release.json").read_text())
    if not receipt["ready"]:
        raise ValueError("Release is not ready")
    for name, checksum in receipt["artifacts"].items():
        if sha256(release / name) != checksum:
            raise ValueError(f"Release checksum mismatch: {name}")
    assets = json.loads((release / "references.json").read_text())
    count = 0
    for row in read_jsonl(release / "train.materialized.jsonl"):
        validate_materialized_row(row)
        if row["verifier_metadata"]["reference_step"] not in assets:
            raise ValueError("Task reference is absent from the packed manifest")
        count += 1
    if count != receipt["count"]:
        raise ValueError("Release task count mismatch")
    found = set()
    with tarfile.open(release / "references.tar") as archive:
        for member in archive:
            if member.name not in assets or member.name in found or not member.isfile():
                raise ValueError(f"Unexpected packed reference: {member.name}")
            with archive.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != assets[member.name]["sha256"] or member.size != assets[member.name]["size"]:
                raise ValueError(f"Reference checksum mismatch: {member.name}")
            found.add(member.name)
    if found != set(assets):
        raise ValueError("Missing packed STEP files")
    print(
        json.dumps(
            {
                "release": str(release),
                "count": count,
                "references": len(found),
                "validated": True,
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("select")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--sampler-config", type=Path, required=True)
    command.add_argument("--num-samples", type=int, default=25024)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--render-overrides", type=Path)
    command = commands.add_parser("render")
    command.add_argument("--selection", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--shard-index", type=int, default=0)
    command.add_argument("--num-shards", type=int, default=1)
    command.add_argument("--workers", type=int, default=8)
    command.add_argument("--chunk-size", type=int, default=64)
    command.add_argument("--render-overrides", type=Path)
    command.add_argument("--max-chunks", type=int)
    command = commands.add_parser("assemble")
    command.add_argument("--selection", type=Path, required=True)
    command.add_argument("--shards", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--prefix-count", type=int)
    command = commands.add_parser("validate")
    command.add_argument("release", type=Path)
    args = vars(parser.parse_args())
    globals()[args.pop("command")](**args)


if __name__ == "__main__":
    main()
