# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metadata-only DeepCAD manifests and per-sampled-prompt rendering."""

import argparse
import fcntl
import hashlib
import json
import logging
import os
import signal
import sqlite3
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ReferenceRenderTimeout(RuntimeError):
    """Reference exceeded the render budget; quarantined, not proven invalid."""


class LazyRenderConfig(BaseModel, extra="forbid"):
    harness: Literal["simple", "opencode"] = "simple"
    dataset_root: str
    renderer_python: str
    cache_root: str
    legacy_rows_root: str | None = None
    workers: int = Field(default=4, ge=1, le=16)
    timeout_s: int = Field(default=180, ge=1)
    seed: int = 42
    diagonal_view_probability: float = Field(default=0.75, ge=0, le=1)
    no_annotations_probability: float = Field(default=0.75, ge=0, le=1)
    # Accept disabled legacy flags without importing the full branch's optional prompt features.
    api_cheatsheet: Literal[False] = False
    orientation_axes: Literal[False] = False
    reference_sandbox_image: str | None = None
    reference_sandbox_provider: dict[str, Any] = Field(
        default_factory=lambda: {"docker": {"create": {"network": "none", "cap_drop": ["ALL"]}}}
    )
    reference_execution_timeout_s: int = Field(default=60, ge=1, le=600)
    reference_catalog: str | None = None
    max_reference_replacements: int = Field(default=8, ge=0, le=64)
    operation_hint_probability: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)


def validate_row(row: dict[str, Any], record: dict[str, Any]) -> None:
    # Standalone renderer workers add the checkout root before loading row schemas.
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
    from resources_servers.deepcad.task_data import TaskData

    TaskData.model_validate(row["verifier_metadata"])
    NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    metadata = row["verifier_metadata"]
    if (
        metadata["task_id"] != record["task_id"]
        or metadata["source_sha256"] != record["source_sha256"]
        or metadata["reference_step"] != Path(record["reference_step"]).name
        or metadata["difficulty"] != record["difficulty"]
    ):
        raise ValueError("Cached DeepCAD row does not match the retained representative")
    parts = [
        part
        for item in row["responses_create_params"]["input"]
        if isinstance(item.get("content"), list)
        for part in item["content"]
    ]
    images = [part for part in parts if part.get("type") == "input_image"]
    if len(images) != (1 if metadata["view_mode"] == "diagonal" else 3):
        raise ValueError("Cached DeepCAD row has an incomplete view set")
    if not all(part["image_url"].startswith("data:image/") for part in images):
        raise ValueError("Cached DeepCAD images must be self-contained")


def _confirm_failure(target: Path, detail: dict[str, Any], config: LazyRenderConfig) -> None:
    """Persist one observation; only a matching fresh attempt may quarantine it."""
    pending = target.with_suffix(".failure.json")
    identity = [
        detail["confirmation_key"],
        config.renderer_python,
        config.timeout_s,
        config.reference_execution_timeout_s,
    ]
    if pending.exists() and json.loads(pending.read_text()).get("identity") == identity:
        detail["confirmation_count"] = 2
        return
    temporary = pending.with_suffix(".building")
    temporary.write_text(json.dumps({"identity": identity, "detail": detail}))
    temporary.replace(pending)
    raise RuntimeError(f"Unconfirmed reference failure; retry in a fresh renderer: {detail['reason']}")


def _materialize_one(example: dict[str, Any]) -> dict[str, Any]:
    lazy = example["deepcad_lazy"]
    config = LazyRenderConfig.model_validate(lazy["config"])
    record = lazy["record"]
    # Rendering settings and source identity determine the immutable cache key.
    settings = config.model_dump(
        exclude={
            "renderer_python",
            "harness",
            "orientation_axes",
            "workers",
            "timeout_s",
            "legacy_rows_root",
            "cache_root",
            "reference_catalog",
            "max_reference_replacements",
        }
    )
    if record.get("source_kind") != "cad_corpus":
        settings.pop("reference_sandbox_image")
        settings.pop("reference_sandbox_provider")
        settings.pop("reference_execution_timeout_s")
    if record.get("source_kind") == "cad_corpus":
        from resources_servers.deepcad.operations import OPERATION_NAMES

        settings["reference_adapter_version"] = 3
        settings["reference_operations_schema"] = sorted(OPERATION_NAMES)
    # Preserve existing caches when the new augmentation is disabled.
    if config.operation_hint_probability == 0:
        settings.pop("operation_hint_probability")
    key = hashlib.sha256(
        json.dumps(
            {
                "version": 1,
                "record": record,
                "settings": settings,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    root = Path(config.cache_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    target = root / (key + ".json")
    with (root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rejected = target.with_suffix(".rejected.json")
        if rejected.exists():
            from resources_servers.deepcad.dataset.corpus_inputs import ReferenceRejected

            detail = json.loads(rejected.read_text())
            # Older releases quarantined a single renderer segfault.
            ambiguous = detail.get("confirmation_key") or detail.get("status") == "segfault_quarantined"
            ambiguous = ambiguous or detail["reason"].startswith(
                ("Reference sandbox segfault:", "Reference program exceeded")
            )
            if not ambiguous or detail.get("confirmation_count") == 2:
                raise ReferenceRejected(detail["reason"])
        # Budget-specific quarantine keeps retries and GRPO groups consistent.
        # Separate from invalid CAD; remove this marker to retry the reference.
        timed_out = root / f"{key}.timeout-{config.timeout_s}.json"
        if timed_out.exists() and not target.exists():
            detail = json.loads(timed_out.read_text())
            if detail.get("confirmation_count") == 2:
                raise ReferenceRenderTimeout(detail["reason"])
        if target.exists():
            row = json.loads(target.read_text())
        else:
            legacy = Path(config.legacy_rows_root) / (record["id"] + ".json") if config.legacy_rows_root else None
            if (
                legacy
                and legacy.exists()
                and config.seed == 42
                and config.diagonal_view_probability == 0.75
                and config.no_annotations_probability == 0.75
                and not config.api_cheatsheet
                and config.operation_hint_probability == 0
                and not record.get("reference_operations")
            ):
                row = json.loads(legacy.read_text())
                validate_row(row, record)
            else:
                with tempfile.TemporaryDirectory(prefix="render-", dir=root) as work:
                    request = Path(work) / "request.json"
                    result = Path(work) / "row.json"
                    request.write_text(json.dumps(lazy))
                    env = {
                        **os.environ,
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "VTK_SMP_MAX_THREADS": "1",
                        "LP_NUM_THREADS": "2",
                        "MESA_SHADER_CACHE_DISABLE": "true",
                    }
                    try:
                        process = subprocess.run(
                            [
                                config.renderer_python,
                                str(Path(__file__).resolve()),
                                "render",
                                str(request),
                                str(result),
                            ],
                            capture_output=True,
                            text=True,
                            errors="replace",
                            timeout=config.timeout_s,
                            env=env,
                        )
                    except subprocess.TimeoutExpired as exc:
                        if record.get("source_kind") != "cad_corpus" or not config.reference_catalog:
                            raise
                        detail = {
                            "reason": f"Reference render exceeded {config.timeout_s}s: {record.get('asset_id', record['id'])}",
                            "asset_id": record.get("asset_id"),
                            "source_sha256": record["source_sha256"],
                            "timeout_s": config.timeout_s,
                            "status": "render_timeout_unverified",
                            "confirmation_key": "renderer_timeout",
                        }
                        _confirm_failure(target, detail, config)
                        detail["status"] = "render_timeout_confirmed"
                        temporary = timed_out.with_suffix(".building")
                        temporary.write_text(json.dumps(detail))
                        temporary.replace(timed_out)
                        logger.warning("Quarantined reference render: %s", detail)
                        raise ReferenceRenderTimeout(detail["reason"]) from exc
                    failure = result.with_suffix(".rejected.json")
                    if process.returncode in (-signal.SIGSEGV, 128 + signal.SIGSEGV):
                        if record.get("source_kind") != "cad_corpus" or not config.reference_catalog:
                            raise RuntimeError(
                                f"DeepCAD input render failed for {record['id']}: {process.stderr[-2000:]}"
                            )
                        failure.write_text(
                            json.dumps(
                                {
                                    "reason": f"Reference render segfault for {record['id']} (return_code={process.returncode})",
                                    "return_code": process.returncode,
                                    "stderr": process.stderr[-2000:],
                                    "status": "segfault_quarantined",
                                    "confirmation_key": f"renderer_signal:{process.returncode}",
                                }
                            )
                        )
                    if failure.exists():
                        from resources_servers.deepcad.dataset.corpus_inputs import ReferenceRejected

                        detail = json.loads(failure.read_text())
                        detail.update(asset_id=record.get("asset_id"), source_sha256=record["source_sha256"])
                        if detail.get("confirmation_key"):
                            _confirm_failure(target, detail, config)
                        temporary = rejected.with_suffix(".building")
                        temporary.write_text(json.dumps(detail))
                        temporary.replace(rejected)
                        logger.warning("Quarantined lazy reference: %s", detail)
                        raise ReferenceRejected(detail["reason"])
                    if process.returncode:
                        raise RuntimeError(f"DeepCAD input render failed for {record['id']}: {process.stderr[-2000:]}")
                    row = json.loads(result.read_text())
            validate_row(row, record)
            temporary = target.with_suffix(".building")
            temporary.write_text(json.dumps(row) + "\n")
            temporary.replace(target)
            target.with_suffix(".failure.json").unlink(missing_ok=True)
        validate_row(row, record)
    resolved = deepcopy(example)
    resolved.pop("deepcad_lazy")
    resolved.update(row)
    sampling = deepcopy(example.get("responses_create_params", {}))
    sampling.pop("input", None)
    resolved["responses_create_params"].update(sampling)
    resolved["task_source"] = example["task_source"]
    if config.harness == "opencode":
        from resources_servers.deepcad.prepare_opencode import prepare_row

        resolved = prepare_row(resolved)
    return resolved


def materialize_one(example: dict[str, Any]) -> dict[str, Any]:
    """Replace invalid or render-budget-exceeding references; propagate other outages."""
    from resources_servers.deepcad.dataset.corpus_inputs import ReferenceRejected

    original = example["deepcad_lazy"]
    config = LazyRenderConfig.model_validate(original["config"])
    current = example
    rejected = []
    replacement_reasons = []
    for attempt in range(config.max_reference_replacements + 1):
        try:
            resolved = _materialize_one(current)
            if rejected:
                resolved["verifier_metadata"].update(
                    requested_task_id=original["record"]["task_id"],
                    replaced_reference_asset_ids=rejected,
                    reference_replacement_reasons=replacement_reasons,
                )
            return resolved
        except (ReferenceRejected, ReferenceRenderTimeout) as exc:
            if (
                original["record"].get("source_kind") != "cad_corpus"
                or not config.reference_catalog
                or attempt == config.max_reference_replacements
            ):
                raise
            rejected.append(current["deepcad_lazy"]["record"]["asset_id"])
            replacement_reasons.append({"type": type(exc).__name__, "reason": str(exc)})
            logger.warning(
                "Replacing lazy reference asset=%s requested_task=%s attempt=%d reason=%s",
                rejected[-1],
                original["record"]["task_id"],
                attempt + 1,
                exc,
            )
            root = Path(config.dataset_root).resolve()
            catalog = Path(config.reference_catalog).resolve()
            if not catalog.is_relative_to(root):
                raise ValueError("Reference replacement catalog must remain inside dataset_root")
            key = hashlib.sha256((original["record"]["task_id"] + ":" + str(attempt)).encode()).hexdigest()
            # Replacement must preserve the selected source and operation coverage.
            source = original["record"].get("dataset")
            required_operations = json.dumps(sorted(set(original["record"].get("reference_operations", []))))
            eligible = (
                "status='selected' AND json_extract(row_json, '$.deepcad_lazy.record.dataset') IS ? "
                "AND NOT EXISTS (SELECT 1 FROM json_each(?) AS required WHERE NOT EXISTS "
                "(SELECT 1 FROM json_each(json_extract(row_json, '$.deepcad_lazy.record.reference_operations')) "
                "AS actual WHERE actual.value=required.value)) "
            )
            with sqlite3.connect(f"file:{catalog}?mode=ro", uri=True) as db:
                row = db.execute(
                    "SELECT row_json FROM assets WHERE "
                    + eligible
                    + "AND id>=? AND id NOT IN ("
                    + ",".join("?" for _ in rejected)
                    + ") ORDER BY id LIMIT 1",
                    (source, required_operations, key, *rejected),
                ).fetchone()
                if row is None:
                    row = db.execute(
                        "SELECT row_json FROM assets WHERE "
                        + eligible
                        + "AND id NOT IN ("
                        + ",".join("?" for _ in rejected)
                        + ") ORDER BY id LIMIT 1",
                        (source, required_operations, *rejected),
                    ).fetchone()
            if row is None:
                raise
            current = json.loads(row[0])
            # The catalog supplies an asset, never the caller's agent routing.
            current["task_source"] = example["task_source"]
            current.pop("agent_ref", None)
            if "agent_ref" in example:
                current["agent_ref"] = deepcopy(example["agent_ref"])
            # Keep the caller's view, hint and execution settings for the replacement.
            current["deepcad_lazy"]["config"] = config.model_dump()
    raise AssertionError("Unreachable reference replacement state")


def materialize_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render each unique selected prompt once; leave ordinary Gym rows alone."""
    unique = {}
    for example in examples:
        if "deepcad_lazy" in example:
            key = json.dumps(example["deepcad_lazy"], sort_keys=True)
            unique.setdefault(key, example)
    if not unique:
        return examples
    workers = min(LazyRenderConfig.model_validate(row["deepcad_lazy"]["config"]).workers for row in unique.values())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = dict(zip(unique, pool.map(materialize_one, unique.values()), strict=True))
    # Publish only after the complete selected batch has rendered successfully.
    for example in examples:
        if "deepcad_lazy" in example:
            resolved = rows[json.dumps(example["deepcad_lazy"], sort_keys=True)]
            transport = {
                key: value
                for key, value in example.items()
                if key.startswith("_") or key in ("agent_ref", "task_source")
            }
            sampling = deepcopy(example.get("responses_create_params", {}))
            sampling.pop("input", None)
            example.clear()
            example.update(deepcopy(resolved))
            example["responses_create_params"].update(sampling)
            example.update(transport)
    return examples


def render(request: Path, output: Path) -> None:
    # CAD imports occur only in a bounded worker for a sampled, uncached prompt.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import cadquery as cq
    import numpy as np

    from resources_servers.deepcad.dataset.augment_inputs import sample_annotations
    from resources_servers.deepcad.dataset.prepare_gym import (
        dimension_data,
        prepare_views,
        task_row,
    )

    lazy = json.loads(request.read_text())
    config = LazyRenderConfig.model_validate(lazy["config"])
    record = lazy["record"]
    if record.get("source_kind") == "cad_corpus":
        from resources_servers.deepcad.dataset.corpus_inputs import ReferenceRejected, render_corpus

        if not config.reference_sandbox_image:
            raise ValueError("Corpus rendering requires reference_sandbox_image")
        try:
            render_corpus({"record": record, "config": config.model_dump()}, output)
        except ReferenceRejected as exc:
            output.with_suffix(".rejected.json").write_text(
                json.dumps({"reason": str(exc), "confirmation_key": exc.confirmation_key})
            )
        return
    root = Path(config.dataset_root).resolve()
    paths = [(root / record[key]).resolve() for key in ("cad_json", "reference_step")]
    if not all(path.is_relative_to(root) for path in paths):
        raise ValueError("DeepCAD source paths must remain inside dataset_root")
    source = paths[0].read_bytes()
    if hashlib.sha256(source).hexdigest() != record["source_sha256"]:
        raise ValueError("DeepCAD source checksum mismatch")
    if hashlib.sha256(paths[1].read_bytes()).hexdigest() != record["reference_step_sha256"]:
        raise ValueError("DeepCAD reference checksum mismatch")
    document = json.loads(source)
    bbox = document["properties"]["bounding_box"]
    maximum = max(abs(bbox[name][axis]) for name in ("min_point", "max_point") for axis in "xyz")
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Invalid DeepCAD source bounding box")
    shape = cq.importers.importStep(str(paths[1])).val().scale(maximum / 75)
    images = prepare_views(
        shape,
        output.parent / "views",
        task_id=record["task_id"],
        document=document,
        seed=config.seed,
        diagonal_view_probability=config.diagonal_view_probability,
        no_annotations_probability=config.no_annotations_probability,
    )
    row = task_row(
        record["task_id"],
        paths[1].name,
        images,
        metrics={"difficulty": record["difficulty"], **record["construction_metrics"]},
        dimensions_mm=dimension_data(shape)["size_mm"].tolist(),
        seed=config.seed,
        engineering_annotations=sample_annotations(
            record["task_id"], seed=config.seed, no_annotations_probability=config.no_annotations_probability
        ),
        reference_operations=record.get("reference_operations"),
        operation_hint_probability=config.operation_hint_probability,
    )
    row["verifier_metadata"].update(dedup_representative=record["id"], source_sha256=record["source_sha256"])
    validate_row(row, record)
    output.write_text(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("root", type=Path)
    manifest.add_argument("output", type=Path)
    manifest.add_argument("--runtime-root", required=True)
    manifest.add_argument("--renderer-python", required=True)
    manifest.add_argument("--cache-root", required=True)
    worker = commands.add_parser("render")
    worker.add_argument("request", type=Path)
    worker.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "render":
        render(args.request, args.output)
        return
    assert json.loads((args.root / "validation.json").read_text())["passed"]
    with (args.root / "easy/train.jsonl").open() as stream:
        records = [json.loads(line) for line in stream]
    records.sort(key=lambda row: (row["difficulty"], row["task_id"]))
    needed = {row["id"] for row in records}
    step_hashes = {}
    with (args.root / "_work/features.jsonl").open() as stream:
        for line in stream:
            feature = json.loads(line)
            if feature["id"] in needed and feature["status"] == "ok":
                step_hashes[feature["id"]] = feature["step_sha256"]
    config = LazyRenderConfig(
        dataset_root=args.runtime_root,
        renderer_python=args.renderer_python,
        cache_root=args.cache_root,
        legacy_rows_root=str(Path(args.runtime_root) / "gym/easy/rows"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".building")
    with temporary.open("w") as stream:
        for record in records:
            assert record["split"] == "train" and record["bucket"] == "easy" and record["geometry_status"] == "ok"
            selected = {
                key: record[key]
                for key in (
                    "id",
                    "task_id",
                    "cad_json",
                    "source_sha256",
                    "reference_step",
                    "difficulty",
                    "construction_metrics",
                )
            }
            selected["reference_step_sha256"] = step_hashes[record["id"]]
            row = {
                "task_source": "deepcad_simple_agent",
                "responses_create_params": {"input": []},
                "verifier_metadata": {
                    "task_id": record["task_id"],
                    "reference_step": Path(record["reference_step"]).name,
                    "difficulty": record["difficulty"],
                },
                "deepcad_lazy": {"config": config.model_dump(), "record": selected},
            }
            stream.write(json.dumps(row) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"manifest": str(args.output), "count": len(records), "rendered_during_manifest": 0}))


if __name__ == "__main__":
    main()
