# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Close model-produced GDPval Office-to-PDF gaps without mutating references.

The inventory command reports every Office source without its provenance-safe
PDF. A unique-stem source keeps the historical ``name.pdf`` spelling; sources
that share a stem use injective ``name.ext.pdf`` sidecars so every deliverable
retains its own render. The convert command stages each produced source, imports
the requested GDPval ``preconvert.py``, extends its LibreOffice timeout to 900
seconds, and publishes validated PDFs with an atomic hard link. Missing files
beneath a ``reference_files`` directory are reported as exceptions but are
never changed.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
import tempfile
import types
import uuid
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


OOXML_EXTENSIONS = frozenset({".docx", ".pptx", ".xlsx"})
OFFICE_EXTENSIONS = OOXML_EXTENSIONS | frozenset({".doc", ".ppt", ".xls"})
REFERENCE_DIR_NAME = "reference_files"
CONVERSION_TIMEOUT_SECONDS = 900
DEFAULT_WORKERS = 4
INVENTORY_SCHEMA = "gdpval-preconvert-closure-inventory-v1"
RECEIPT_SCHEMA = "gdpval-preconvert-closure-receipt-v1"
_STAGE_PREFIX = ".gdpval-preconvert-"


class ClosureError(RuntimeError):
    """A fail-closed inventory, normalization, or publication error."""


class DuplicateMemberConflict(ClosureError):
    """An OOXML ZIP repeats one name with non-identical bytes."""

    def __init__(self, message: str, audit: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit = audit


class _TimeoutSubprocessProxy:
    """Delegate to a subprocess module while replacing per-call timeouts."""

    def __init__(self, wrapped: Any, timeout_seconds: int) -> None:
        self._wrapped = wrapped
        self._timeout_seconds = timeout_seconds

    def run(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["timeout"] = self._timeout_seconds
        return self._wrapped.run(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = _json_bytes(document)
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _validate_root(root: Path) -> Path:
    try:
        resolved = root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ClosureError(f"deliverables root is unavailable: {root}: {exc}") from exc
    if not resolved.is_dir():
        raise ClosureError(f"deliverables root is not a directory: {resolved}")
    return resolved


def _iter_office_sources(root: Path) -> list[Path]:
    sources: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith(_STAGE_PREFIX))
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() in OFFICE_EXTENSIONS:
                sources.append(path)
    return sorted(sources)


def _sidecar_pdf(source: Path) -> Path:
    """Return the injective render path used by the pinned GDPVal runtime."""

    return source.with_name(source.name + ".pdf")


def _expected_pdf_paths(sources: Sequence[Path]) -> dict[Path, Path]:
    """Choose the same unambiguous Office provenance mapping as the judge.

    The injective spelling is mandatory when multiple Office files in one
    directory share a stem. It also remains authoritative when it already
    exists for a unique source, matching ``resolve_pdf_provenance`` in the
    pinned GDPVal runtime. Otherwise retain the historical plain sibling.
    """

    groups: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    for source in sources:
        groups[(source.parent, source.stem)].append(source)

    destinations: dict[Path, Path] = {}
    for group in groups.values():
        ambiguous = len(group) > 1
        for source in group:
            injective = _sidecar_pdf(source)
            destinations[source] = (
                injective if ambiguous or injective.exists() or injective.is_symlink() else source.with_suffix(".pdf")
            )
    return destinations


def _source_record(root: Path, source: Path, expected_pdf: Path) -> dict[str, Any]:
    relative = source.relative_to(root)
    record: dict[str, Any] = {
        "source": str(source),
        "relative_source": relative.as_posix(),
        "expected_pdf": str(expected_pdf),
        "reference": REFERENCE_DIR_NAME in relative.parts,
    }
    try:
        source_lstat = source.lstat()
        record.update(
            {
                "source_bytes": source.stat().st_size,
                "source_mode": f"{stat.S_IMODE(source_lstat.st_mode):04o}",
                "source_sha256": _sha256(source),
                "source_symlink": stat.S_ISLNK(source_lstat.st_mode),
            }
        )
    except OSError as exc:
        record.update(
            {
                "source_bytes": None,
                "source_mode": None,
                "source_sha256": None,
                "source_symlink": source.is_symlink(),
                "source_error": str(exc),
            }
        )
    return record


def _validated_pdf_record(
    source_record: dict[str, Any], expected_pdf: Path
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        pdf_lstat = expected_pdf.lstat()
    except FileNotFoundError:
        return None, "missing sibling PDF"
    except OSError as exc:
        return None, f"cannot stat sibling PDF: {exc}"
    if stat.S_ISLNK(pdf_lstat.st_mode):
        return None, "sibling PDF is a symlink"
    if not stat.S_ISREG(pdf_lstat.st_mode):
        return None, "sibling PDF is not a regular file"
    if not _valid_pdf(expected_pdf):
        return None, "sibling PDF does not start with %PDF"
    pair = dict(source_record)
    pair.update(
        {
            "pdf": str(expected_pdf),
            "pdf_bytes": pdf_lstat.st_size,
            "pdf_mode": f"{stat.S_IMODE(pdf_lstat.st_mode):04o}",
            "pdf_sha256": _sha256(expected_pdf),
            "pdf_symlink": False,
        }
    )
    return pair, None


def _closure_fingerprint(produced_pairs: list[dict[str, Any]]) -> str:
    identity = [
        {key: row[key] for key in ("relative_source", "source_bytes", "source_sha256", "pdf_bytes", "pdf_sha256")}
        for row in produced_pairs
    ]
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_inventory(deliverables_root: Path) -> dict[str, Any]:
    root = _validate_root(deliverables_root)
    sources = _iter_office_sources(root)
    expected_pdfs = _expected_pdf_paths(sources)
    produced_missing: list[dict[str, Any]] = []
    reference_exceptions: list[dict[str, Any]] = []
    produced_pairs: list[dict[str, Any]] = []
    reference_pairs: list[dict[str, Any]] = []
    ready = 0
    total = 0
    for source in sources:
        total += 1
        expected_pdf = expected_pdfs[source]
        source_record = _source_record(root, source, expected_pdf)
        pair, pdf_error = _validated_pdf_record(source_record, expected_pdf)
        if pair is not None:
            ready += 1
            if source_record["reference"]:
                reference_pairs.append(pair)
            else:
                produced_pairs.append(pair)
            continue
        source_record["pdf_error"] = pdf_error
        if source_record["reference"]:
            reference_exceptions.append(source_record)
        else:
            produced_missing.append(source_record)

    destination_groups: dict[str, list[str]] = defaultdict(list)
    for record in produced_missing:
        destination_groups[record["expected_pdf"]].append(record["source"])
    collisions = [
        {"expected_pdf": expected_pdf, "sources": sorted(sources)}
        for expected_pdf, sources in sorted(destination_groups.items())
        if len(sources) > 1
    ]

    return {
        "schema": INVENTORY_SCHEMA,
        "status": "CLOSED" if not produced_missing else "OPEN",
        "deliverables_root": str(root),
        "remaining_produced": len(produced_missing),
        "remaining_references": len(reference_exceptions),
        "closure_fingerprint": _closure_fingerprint(produced_pairs),
        "office_extensions": sorted(OFFICE_EXTENSIONS),
        "counts": {
            "office_total": total,
            "ready": ready,
            "produced_missing": len(produced_missing),
            "reference_exceptions": len(reference_exceptions),
            "destination_collisions": len(collisions),
        },
        "produced_missing": produced_missing,
        "produced_pairs": produced_pairs,
        "reference_exceptions": reference_exceptions,
        "reference_pairs": reference_pairs,
        "destination_collisions": collisions,
    }


def _copy_regular_source(source: Path, destination: Path) -> str:
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, source_flags)
    destination_fd: int | None = None
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ClosureError(f"produced Office source is not a regular file: {source}")
        destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fsync(destination_fd)
        final_stat = os.fstat(source_fd)
        if (source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns) != (
            final_stat.st_dev,
            final_stat.st_ino,
            final_stat.st_size,
            final_stat.st_mtime_ns,
        ):
            raise ClosureError(f"produced Office source changed while staging: {source}")
        return digest.hexdigest()
    finally:
        os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)


def _zip_member_sha256(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info) as member:
        for chunk in iter(lambda: member.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_members_equal(archive: zipfile.ZipFile, first: zipfile.ZipInfo, second: zipfile.ZipInfo) -> bool:
    if first.file_size != second.file_size:
        return False
    with archive.open(first) as left, archive.open(second) as right:
        while True:
            left_chunk = left.read(1024 * 1024)
            right_chunk = right.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _copy_zip_member(archive: zipfile.ZipFile, output: zipfile.ZipFile, info: zipfile.ZipInfo) -> None:
    with archive.open(info) as source, output.open(info, "w", force_zip64=True) as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)


def _deduplicate_ooxml(staged_source: Path, normalized_root: Path, source: Path) -> tuple[Path, dict[str, Any]]:
    audit: dict[str, Any] = {
        "source": str(source),
        "staged_source_sha256": _sha256(staged_source),
        "normalized": False,
        "duplicate_members": [],
        "conflicting_members": [],
    }
    try:
        with zipfile.ZipFile(staged_source) as archive:
            rows = archive.infolist()
            grouped: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
            for info in archive.infolist():
                grouped[info.filename].append(info)

            duplicate_names = sorted(name for name, infos in grouped.items() if len(infos) > 1)
            for name in duplicate_names:
                infos = grouped[name]
                row = {
                    "name": name,
                    "occurrences": len(infos),
                    "content_sha256": _zip_member_sha256(archive, infos[0]),
                }
                if any(not _zip_members_equal(archive, infos[0], info) for info in infos[1:]):
                    audit["conflicting_members"].append(row)
                else:
                    audit["duplicate_members"].append(row)

            if audit["conflicting_members"]:
                names = ", ".join(row["name"] for row in audit["conflicting_members"])
                raise DuplicateMemberConflict(
                    f"OOXML ZIP has non-identical duplicate members in {source}: {names}", audit
                )
            if not duplicate_names:
                audit["normalized_source_sha256"] = audit["staged_source_sha256"]
                return staged_source, audit

            normalized_root.mkdir(mode=0o700)
            normalized_source = normalized_root / staged_source.name
            seen: set[str] = set()
            with zipfile.ZipFile(normalized_source, "w") as output:
                for info in rows:
                    if info.filename in seen:
                        continue
                    seen.add(info.filename)
                    _copy_zip_member(archive, output, info)
    except DuplicateMemberConflict:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        audit["zip_error"] = str(exc)
        raise DuplicateMemberConflict(f"invalid OOXML ZIP {source}: {exc}", audit) from exc
    _fsync_file(normalized_source)
    audit["normalized"] = True
    audit["normalized_source_sha256"] = _sha256(normalized_source)
    return normalized_source, audit


def _load_preconvert_module(path: Path) -> tuple[types.ModuleType, str]:
    try:
        module_path = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ClosureError(f"preconvert module is unavailable: {path}: {exc}") from exc
    if not module_path.is_file():
        raise ClosureError(f"preconvert module is not a file: {module_path}")
    module_sha256 = _sha256(module_path)
    module_name = f"_gdpval_preconvert_{module_sha256[:16]}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ClosureError(f"cannot import preconvert module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    converter = getattr(module, "convert_to_pdf", None)
    if not callable(converter):
        raise ClosureError(f"preconvert module has no callable convert_to_pdf: {module_path}")
    module_subprocess = getattr(module, "subprocess", None)
    if module_subprocess is not None:
        module.subprocess = _TimeoutSubprocessProxy(module_subprocess, CONVERSION_TIMEOUT_SECONDS)
    return module, module_sha256


def _valid_pdf(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < 5:
            return False
        with path.open("rb") as stream:
            return stream.read(4) == b"%PDF"
    except OSError:
        return False


def _publish_pdf(staged_pdf: Path, expected_pdf: Path) -> tuple[str, str, tuple[int, int]]:
    if not _valid_pdf(staged_pdf):
        raise ClosureError(f"converter output is not a PDF: {staged_pdf}")
    _fsync_file(staged_pdf)
    staged_sha256 = _sha256(staged_pdf)
    staged_stat = staged_pdf.stat()
    try:
        os.link(staged_pdf, expected_pdf)
        method = "hardlink"
        published_stat = staged_stat
    except FileExistsError:
        if not _valid_pdf(expected_pdf) or _sha256(expected_pdf) != staged_sha256:
            raise ClosureError(f"expected PDF appeared with different bytes: {expected_pdf}")
        method = "existing-identical"
        published_stat = expected_pdf.stat()
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        temporary = expected_pdf.parent / f".{expected_pdf.name}.publish.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with staged_pdf.open("rb") as source, os.fdopen(descriptor, "wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            if not _valid_pdf(temporary) or _sha256(temporary) != staged_sha256:
                raise ClosureError(f"cross-filesystem PDF copy changed bytes: {expected_pdf}")
            try:
                os.link(temporary, expected_pdf)
                method = "copy"
                published_stat = expected_pdf.stat()
            except FileExistsError:
                if not _valid_pdf(expected_pdf) or _sha256(expected_pdf) != staged_sha256:
                    raise ClosureError(f"expected PDF appeared with different bytes: {expected_pdf}")
                method = "existing-identical"
                published_stat = expected_pdf.stat()
        finally:
            temporary.unlink(missing_ok=True)
    _fsync_dir(expected_pdf.parent)
    return method, staged_sha256, (published_stat.st_dev, published_stat.st_ino)


def _remove_just_published_pdf(expected_pdf: Path, published_identity: tuple[int, int]) -> None:
    try:
        current = expected_pdf.lstat()
        if (current.st_dev, current.st_ino) != published_identity or not stat.S_ISREG(current.st_mode):
            return
        expected_pdf.unlink()
        _fsync_dir(expected_pdf.parent)
    except FileNotFoundError:
        return


def _conversion_failure(
    record: dict[str, Any], error: str, normalization: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "ok": False,
        "source": record["source"],
        "expected_pdf": record["expected_pdf"],
        "error": error,
        "normalization": normalization
        or {
            "source": record["source"],
            "normalized": False,
            "duplicate_members": [],
            "conflicting_members": [],
        },
    }


def _convert_one(
    record: dict[str, Any],
    converter: Callable[[Path], tuple[Path, bool, str]],
    scratch_root: Path | None,
) -> dict[str, Any]:
    source = Path(record["source"])
    expected_pdf = Path(record["expected_pdf"])
    if record.get("source_symlink"):
        return _conversion_failure(record, f"produced Office source is a symlink: {source}")
    if record.get("source_error") or not record.get("source_sha256"):
        return _conversion_failure(record, f"produced Office source is unreadable: {source}")

    stage_dir: Path | None = None
    normalization: dict[str, Any] | None = None
    try:
        stage_dir = Path(tempfile.mkdtemp(prefix=_STAGE_PREFIX, dir=scratch_root or source.parent))
        staged_source = stage_dir / source.name
        staged_sha256 = _copy_regular_source(source, staged_source)
        if staged_sha256 != record["source_sha256"]:
            raise ClosureError(f"produced Office source changed after inventory: {source}")
        if source.suffix.lower() in OOXML_EXTENSIONS:
            conversion_source, normalization = _deduplicate_ooxml(staged_source, stage_dir / "normalized", source)
        else:
            conversion_source = staged_source
            normalization = {
                "source": str(source),
                "staged_source_sha256": staged_sha256,
                "normalized": False,
                "legacy_binary_office": True,
                "duplicate_members": [],
                "conflicting_members": [],
            }

        result = converter(conversion_source)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ClosureError(f"convert_to_pdf returned an invalid result for {source}")
        _, ok, message = result
        if not ok:
            raise ClosureError(str(message))
        staged_pdf = conversion_source.with_suffix(".pdf")
        if not _valid_pdf(staged_pdf):
            raise ClosureError(f"convert_to_pdf reported success without a valid PDF for {source}")
        if _sha256(source) != record["source_sha256"]:
            raise ClosureError(f"produced Office source hash changed during conversion: {source}")

        publish_method, pdf_sha256, published_identity = _publish_pdf(staged_pdf, expected_pdf)
        if _sha256(source) != record["source_sha256"]:
            if publish_method in {"hardlink", "copy"}:
                _remove_just_published_pdf(expected_pdf, published_identity)
            raise ClosureError(f"produced Office source hash changed during PDF publication: {source}")
        return {
            "ok": True,
            "source": str(source),
            "expected_pdf": str(expected_pdf),
            "source_sha256": record["source_sha256"],
            "pdf_sha256": pdf_sha256,
            "pdf_bytes": expected_pdf.stat().st_size,
            "publish_method": publish_method,
            "converter_message": str(message),
            "normalization": normalization,
        }
    except DuplicateMemberConflict as exc:
        return _conversion_failure(record, str(exc), exc.audit)
    except Exception as exc:
        return _conversion_failure(record, str(exc), normalization)
    finally:
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)


def convert_closure(
    deliverables_root: Path,
    preconvert_module: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise ClosureError(f"workers must be at least 1, got {workers}")
    initial = build_inventory(deliverables_root)
    root = Path(initial["deliverables_root"])
    if scratch_root is not None:
        scratch_root = scratch_root.expanduser().absolute()
        if scratch_root.is_symlink():
            raise ClosureError(f"scratch root must not be a symlink: {scratch_root}")
        scratch_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not scratch_root.is_dir():
            raise ClosureError(f"scratch root is not a directory: {scratch_root}")
    module_path = preconvert_module.expanduser().absolute()
    module_sha256: str | None = None
    results: list[dict[str, Any]] = []

    collision_sources = {source for collision in initial["destination_collisions"] for source in collision["sources"]}
    for record in initial["produced_missing"]:
        if record["source"] in collision_sources:
            results.append(_conversion_failure(record, f"multiple Office sources target {record['expected_pdf']}"))

    eligible = [record for record in initial["produced_missing"] if record["source"] not in collision_sources]
    if eligible:
        try:
            module, module_sha256 = _load_preconvert_module(preconvert_module)
            converter = module.convert_to_pdf
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_convert_one, record, converter, scratch_root): record for record in eligible
                }
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        results.append(_conversion_failure(futures[future], f"conversion worker failed: {exc}"))
        except Exception as exc:
            results.extend(_conversion_failure(record, f"cannot load preconvert module: {exc}") for record in eligible)
    elif module_path.is_file():
        module_sha256 = _sha256(module_path)

    results.sort(key=lambda row: row["source"])
    final = build_inventory(root)
    converted = [
        {key: value for key, value in row.items() if key not in {"ok", "normalization"}}
        for row in results
        if row["ok"]
    ]
    failures = [
        {key: value for key, value in row.items() if key not in {"ok", "normalization"}}
        for row in results
        if not row["ok"]
    ]
    normalization_audit = [row["normalization"] for row in results]
    remaining_produced = final["produced_missing"]
    passed = not remaining_produced and not failures
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS" if passed else "INCOMPLETE",
        "created_at": _utc_now(),
        "deliverables_root": str(root),
        "preconvert_module": str(module_path),
        "preconvert_module_sha256": module_sha256,
        "timeout_seconds": CONVERSION_TIMEOUT_SECONDS,
        "workers": workers,
        "scratch_root": str(scratch_root) if scratch_root is not None else None,
        "initial_produced_missing": initial["produced_missing"],
        "converted": converted,
        "failures": failures,
        "remaining_produced": remaining_produced,
        "reference_exceptions": final["reference_exceptions"],
        "produced_pairs": final["produced_pairs"],
        "closure_fingerprint": final["closure_fingerprint"],
        "normalization_audit": normalization_audit,
        "counts": {
            "initial_produced_missing": len(initial["produced_missing"]),
            "converted": len(converted),
            "failures": len(failures),
            "remaining_produced": len(remaining_produced),
            "reference_exceptions": len(final["reference_exceptions"]),
        },
    }


def _print_and_optionally_write(document: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        _atomic_write_json(output, document)
    sys.stdout.buffer.write(_json_bytes(document))
    sys.stdout.buffer.flush()


def _add_deliverables_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("deliverables", nargs="?", type=Path)
    parser.add_argument("--root", "--deliverables-root", dest="deliverables_root", type=Path)


def _selected_deliverables(args: argparse.Namespace) -> Path:
    positional = args.deliverables
    option = args.deliverables_root
    if positional is None and option is None:
        raise ClosureError("deliverables root is required (positional or --root)")
    if (
        positional is not None
        and option is not None
        and positional.expanduser().absolute() != option.expanduser().absolute()
    ):
        raise ClosureError(f"conflicting deliverables roots: {positional} != {option}")
    return option or positional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory_parser = commands.add_parser(
        "inventory", help="Print the current produced/reference Office gaps as JSON."
    )
    _add_deliverables_arguments(inventory_parser)
    inventory_parser.add_argument(
        "--json", action="store_true", help="Emit JSON (the default; accepted for wrappers)."
    )
    inventory_parser.add_argument("--output", type=Path, help="Also write the inventory atomically to this path.")

    convert_parser = commands.add_parser("convert", help="Convert produced Office gaps and emit a closure receipt.")
    _add_deliverables_arguments(convert_parser)
    convert_parser.add_argument(
        "--preconvert",
        "--preconvert-module",
        "--preconvert-py",
        dest="preconvert_module",
        type=Path,
        required=True,
        help="Path to resources_servers/gdpval/preconvert.py.",
    )
    convert_parser.add_argument("--workers", "--max-concurrent", type=int, default=DEFAULT_WORKERS)
    convert_parser.add_argument("--scratch", type=Path, help="Parent directory for per-source staging directories.")
    convert_parser.add_argument(
        "--receipt", "--output", type=Path, help="Also write the receipt atomically to this path."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inventory":
            document = build_inventory(_selected_deliverables(args))
            _print_and_optionally_write(document, args.output)
            return 0
        receipt = convert_closure(
            _selected_deliverables(args),
            args.preconvert_module,
            workers=args.workers,
            scratch_root=args.scratch,
        )
        _print_and_optionally_write(receipt, args.receipt)
        return 0 if receipt["status"] == "PASS" else 1
    except Exception as exc:
        error = {
            "schema": RECEIPT_SCHEMA if args.command == "convert" else INVENTORY_SCHEMA,
            "status": "ERROR",
            "error": str(exc),
        }
        _print_and_optionally_write(error, None)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
