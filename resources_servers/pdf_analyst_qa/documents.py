# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate document views and materialize PDFs from an operator-mounted bucket."""

import hashlib
import json
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MountedDocumentsConfig(BaseModel):
    """Bucket mount and prefix for the dataset PDFs."""

    model_config = ConfigDict(extra="forbid")
    mount_root: str = "/mnt/s3-data/data"
    prefix: str = Field(min_length=1)
    node_cache_host_path: str | None = None

    @field_validator("node_cache_host_path")
    @classmethod
    def safe_node_cache_path(cls, value: str | None) -> str | None:
        if value is not None:
            if not value.startswith("/") or ".." in value.split("/") or "\\" in value or "\x00" in value:
                raise ValueError("node_cache_host_path must be an absolute directory without traversal")
            if len(PurePosixPath(value).parts) < 4:
                raise ValueError("node_cache_host_path must identify a dedicated dataset cache")
            return str(PurePosixPath(value))
        return value

    @field_validator("mount_root")
    @classmethod
    def absolute_mount(cls, value: str) -> str:
        if not value.startswith("/") or ".." in value.split("/") or "\x00" in value:
            raise ValueError("mount_root must be an absolute path without traversal")
        return str(PurePosixPath(value))

    @field_validator("prefix")
    @classmethod
    def safe_prefix(cls, value: str) -> str:
        parts = value.rstrip("/").split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("prefix must be a relative path without empty, '.' or '..' components")
        if "\\" in value or "\x00" in value:
            raise ValueError("prefix contains invalid characters")
        return "/".join(parts)


class DocumentRecord(BaseModel):
    """Selected document ID and sandbox path; other dataset metadata is ignored."""

    document_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)

    @field_validator("document_id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if value in {".", ".."} or any(char in value for char in ("/", "\\", "\x00")):
            raise ValueError("invalid document ID")
        return value

    @field_validator("relative_path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in value.split("/") or path.suffix.lower() != ".pdf":
            raise ValueError("document paths must be relative PDF paths")
        if "\\" in value or "\x00" in value:
            raise ValueError("invalid document path")
        return str(path)


def workspace_documents(root: Path, workspace_id: str) -> list[DocumentRecord]:
    """Read a host-side allowlist and validate every source before allocating compute."""
    records = json.loads((root / "workspaces" / workspace_id / "documents.json").read_bytes())
    if not isinstance(records, list) or not records:
        raise ValueError("workspace requires a nonempty permitted document view")
    documents = [DocumentRecord.model_validate(record) for record in records]
    paths = [document.relative_path for document in documents]
    if len(paths) != len(set(paths)):
        raise ValueError("workspace document paths must be unique")
    store = (root / "document_store").resolve()
    for document in documents:
        source = (store / f"{document.document_id}.pdf").resolve()
        if source.parent != store or not source.is_file() or source.stat().st_mode & 0o222:
            raise ValueError("document store must contain read-only PDFs")
    return documents


def mounted_document_manifest(root: Path, workspace_id: str, storage: MountedDocumentsConfig) -> str:
    """Return only selected PDF paths and checksums; never serialize hidden task data."""
    files = []
    for record in workspace_documents(root, workspace_id):
        source = root / "document_store" / f"{record.document_id}.pdf"
        with source.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        files.append(
            {
                "source": str(PurePosixPath(storage.mount_root, storage.prefix, "document_store", source.name)),
                "target": record.relative_path,
                "sha256": digest,
            }
        )
    return json.dumps(files)


def node_document_manifest(root: Path, workspace_id: str, storage: MountedDocumentsConfig) -> str:
    """Describe selected published PDFs by size without rereading their contents."""
    return json.dumps(
        [
            {
                "source": str(
                    PurePosixPath(storage.mount_root, storage.prefix, "document_store", record.document_id + ".pdf")
                ),
                "target": record.relative_path,
                "bytes": (root / "document_store" / (record.document_id + ".pdf")).stat().st_size,
            }
            for record in workspace_documents(root, workspace_id)
        ]
    )


DOCUMENT_MANIFEST_PATH = "/tmp/pdf-analyst-qa-document-manifest.json"

SANDBOX_SCRIPTS_ROOT = "/tmp/pdf-analyst-qa-scripts"
NODE_CACHE_ROOT = "/var/lib/pdf-analyst-qa-cache"


def sandbox_script_files() -> dict[str, str]:
    """Return the standalone scripts to deliver before sandbox setup."""
    scripts = Path(__file__).with_name("sandbox_scripts")
    return {f"{SANDBOX_SCRIPTS_ROOT}/{path.name}": path.read_text() for path in scripts.glob("*.py")}
