# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the sandbox manifest for the selected PDFs in an operator-mounted bucket."""

import json
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator

from resources_servers.pdf_analyst_qa.task_data import DocumentRecord


class MountedDocumentsConfig(BaseModel):
    """Bucket mount and prefix for the dataset PDFs."""

    model_config = ConfigDict(extra="forbid")
    mount_root: str = "/mnt/s3-data/data"
    prefix: str = Field(min_length=1)

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


def mounted_document_manifest(documents: list[DocumentRecord], storage: MountedDocumentsConfig) -> str:
    """Return only selected PDF paths and checksums; never serialize hidden task data."""
    return json.dumps(
        [
            {
                "source": str(PurePosixPath(storage.mount_root, storage.prefix, document.path)),
                "target": document.target,
                "sha256": document.sha256,
            }
            for document in documents
        ]
    )


DOCUMENT_MANIFEST_PATH = "/tmp/pdf-analyst-qa-document-manifest.json"

SANDBOX_SCRIPTS_ROOT = "/tmp/pdf-analyst-qa-scripts"


def sandbox_script_files() -> dict[str, str]:
    """Return the standalone scripts to deliver before sandbox setup."""
    scripts = Path(__file__).with_name("sandbox_scripts")
    return {f"{SANDBOX_SCRIPTS_ROOT}/{path.name}": path.read_text() for path in scripts.glob("*.py")}
