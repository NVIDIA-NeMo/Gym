# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy reference execution and prompt rendering for the combined CAD corpus."""

import asyncio
import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import tempfile
import time
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, resolve_provider_config
from resources_servers.deepcad.operations import OPERATION_NAMES, extract_operation_candidates


class ReferenceRejected(ValueError):
    """Reference failure; ambiguous process exits require a second fresh attempt."""

    def __init__(self, reason: str, *, confirmation_key: str | None = None):
        super().__init__(reason)
        self.confirmation_key = confirmation_key


def load_program(root: Path, record: dict) -> tuple[str, str]:
    database = (root / record["inventory_database"]).resolve()
    if not database.is_relative_to(root):
        raise ValueError("Inventory database must remain inside dataset_root")
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        row = db.execute("SELECT code,code_sha256,dataset FROM assets WHERE id=?", (record["asset_id"],)).fetchone()
    if row is None:
        raise ValueError("Corpus asset is missing from its pinned inventory")
    code, checksum, dataset = row
    if checksum != record["source_sha256"] or hashlib.sha256(code.encode()).hexdigest() != checksum:
        raise ValueError("Corpus source checksum mismatch")
    # GeomCAD publishes its final solid under r; adaptation is explicit source text.
    if dataset == "geomcad":
        code += "\nresult = r\n"
    return code, dataset


async def build_reference(root: Path, record: dict, config: dict, directory: Path) -> tuple[Path, list[str]]:
    """Execute reference programs inside the same isolated CadQuery sandbox as predictions."""
    code, dataset = load_program(root, record)
    prepared_sha = hashlib.sha256(code.encode()).hexdigest()
    trace_schema = hashlib.sha256(json.dumps(sorted(OPERATION_NAMES)).encode()).hexdigest()
    target = (root / record["reference_step"]).resolve()
    if not target.is_relative_to(root / "references"):
        raise ValueError("Corpus references must remain in dataset_root/references")
    metadata_path = target.with_suffix(".json")
    if target.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata["prepared_sha256"] == prepared_sha
            and (
                dataset == "histcad"
                or metadata.get("trace_schema_sha256") == trace_schema
                and metadata.get("adapter_version") == 3
            )
            and metadata["step_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
        ):
            return target, [] if dataset == "histcad" else metadata["source_operations"]
    if dataset == "histcad":
        locator = record["reference_locator"]
        archive = (root / locator["archive"]).resolve()
        if not archive.is_relative_to(root / "histcad") or not 0 < locator["size"] <= 32 * 1024 * 1024:
            raise ValueError("Invalid HistCAD STEP locator")
        with archive.open("rb") as source:
            source.seek(locator["offset"])
            data = source.read(locator["size"])
        if hashlib.sha256(data).hexdigest() != locator["sha256"]:
            raise ValueError("HistCAD reference STEP checksum mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        temporary_path.replace(target)
        metadata_path.write_text(
            json.dumps(
                {
                    "prepared_sha256": prepared_sha,
                    "step_sha256": locator["sha256"],
                    "source_operations": [],
                    "operation_evidence": "native_history_not_cadquery",
                }
            )
        )
        return target, []
    here = Path(__file__).resolve().parents[1]
    provider = resolve_provider_config(config["reference_sandbox_provider"], {})
    spec = SandboxSpec(
        image=config["reference_sandbox_image"],
        ttl_s=config["reference_execution_timeout_s"] + 300,
        workdir="/sandbox",
        env={
            **{key: "" for key in os.environ if key.isidentifier()},
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "ENROOT_MOUNT_HOME": "no",
        },
        resources=SandboxResources(cpu=1, memory_mib=2048),
        files={
            "/sandbox/model.py": code,
            "/sandbox/runner.py": (here / "runner.py").read_text(),
        },
    )
    started = time.monotonic()
    prediction = directory / "reference.step"
    async with AsyncSandbox(provider, spec) as sandbox:
        await sandbox.start()
        result = await sandbox.exec(
            "python /sandbox/runner.py > /sandbox/run.log 2>&1",
            timeout_s=config["reference_execution_timeout_s"],
        )
        if result.error_type == "timeout":
            raise ReferenceRejected(
                f"Reference program exceeded its {config['reference_execution_timeout_s']}s execution budget: {record['asset_id']}",
                confirmation_key="reference_execution_timeout",
            )
        if result.error_type:
            raise RuntimeError(f"Reference sandbox provider failure: {result.error_type}")
        if result.return_code:
            log = await sandbox.exec("tail -c 2000 /sandbox/run.log", timeout_s=10)
            if log.return_code or log.error_type:
                raise RuntimeError("Reference sandbox log collection failed")
            detail = (
                f"Reference program failed for {record['asset_id']} (return_code={result.return_code}): {log.stdout}"
            )
            if result.return_code in (-signal.SIGSEGV, 128 + signal.SIGSEGV, -signal.SIGKILL, 128 + signal.SIGKILL):
                raise ReferenceRejected(
                    "Reference sandbox crash or kill (cause unverified): " + detail,
                    confirmation_key=f"reference_signal:{result.return_code}",
                )
            if not result.error_type and (
                ('File "/sandbox/model.py"' in (log.stdout or "") or 'File "model.py"' in (log.stdout or ""))
                or "Assign valid CadQuery solid geometry" in (log.stdout or "")
                or "ValueError: result must have positive finite volume" in (log.stdout or "")
            ):
                raise ReferenceRejected(detail)
            raise RuntimeError(detail)
        check = await sandbox.exec("wc -c < /sandbox/prediction.step", timeout_s=10)
        if check.error_type:
            raise RuntimeError(f"Reference artifact inspection failed: {check.error_type}")
        if check.return_code:
            raise ReferenceRejected("Reference program did not produce a STEP artifact")
        try:
            size = int((check.stdout or "").strip())
        except ValueError as exc:
            raise RuntimeError("Reference artifact inspection returned invalid output") from exc
        if not 0 < size <= 32 * 1024 * 1024:
            raise ReferenceRejected("Empty or oversized reference STEP")
        operations = sorted(extract_operation_candidates(code))
        await sandbox.download("/sandbox/prediction.step", prediction)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_sha256": record["source_sha256"],
        "prepared_sha256": prepared_sha,
        "step_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
        "source_operations": sorted(set(operations)),
        "execution_seconds": time.monotonic() - started,
        "trace_schema_sha256": trace_schema,
        "operation_evidence": "source_syntax",
        "adapter_version": 3,
    }
    # The renderer holds a per-reference lock, shared across view configurations.
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
        temporary.write(prediction.read_bytes())
        temporary_path = Path(temporary.name)
    temporary_path.replace(target)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return target, metadata["source_operations"]


def render_corpus(lazy: dict, output: Path) -> None:
    """Render only the selected prompt and use its source operation names for hints."""
    import cadquery as cq
    from OCP.Geom import Geom_UndefinedValue

    from resources_servers.deepcad.dataset.augment_inputs import sample_annotations
    from resources_servers.deepcad.dataset.prepare_gym import dimension_data, prepare_views, task_row

    config, record = lazy["config"], lazy["record"]
    root = Path(config["dataset_root"]).resolve()
    reference_target = (root / record["reference_step"]).resolve()
    if not reference_target.is_relative_to(root / "references"):
        raise ValueError("Reference path escapes corpus references")
    reference_target.parent.mkdir(parents=True, exist_ok=True)
    with reference_target.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reference, operations = asyncio.run(build_reference(root, record, config, output.parent))
    try:
        imported = cq.importers.importStep(str(reference))
        shape = cq.Compound.makeCompound([solid for value in imported.vals() for solid in value.Solids()])
        if not shape.Solids() or not shape.isValid() or shape.Volume() <= 0:
            raise ReferenceRejected("Invalid corpus reference geometry")
        # Source code units are arbitrary. Rendering normalizes geometry independently,
        # and annotation dimensions describe this explicit millimetre interpretation.
        shape = shape.scale(0.001)
        images = prepare_views(
            shape,
            output.parent / "views",
            task_id=record["task_id"],
            seed=config["seed"],
            diagonal_view_probability=config["diagonal_view_probability"],
            no_annotations_probability=config["no_annotations_probability"],
        )
        dimensions_mm = dimension_data(shape)["size_mm"].tolist()
    except Geom_UndefinedValue as exc:
        # Valid STEP solids can still contain offset surfaces with undefined normals.
        # Reject only this CAD geometry failure; I/O and renderer outages must propagate.
        raise ReferenceRejected(f"Unrenderable corpus reference {record['asset_id']}: {exc}") from exc
    row = task_row(
        record["task_id"],
        reference.name,
        images,
        metrics={"difficulty": record["difficulty"], **record["construction_metrics"]},
        dimensions_mm=dimensions_mm,
        seed=config["seed"],
        engineering_annotations=sample_annotations(
            record["task_id"], seed=config["seed"], no_annotations_probability=config["no_annotations_probability"]
        ),
        reference_operations=operations,
        operation_hint_probability=config["operation_hint_probability"],
    )
    row["verifier_metadata"].update(
        source_sha256=record["source_sha256"],
        dataset=record["dataset"],
        operation_evidence=("native_history_not_cadquery" if record["dataset"] == "histcad" else "source_syntax"),
        asset_id=record["asset_id"],
    )
    output.write_text(json.dumps(row) + "\n")
