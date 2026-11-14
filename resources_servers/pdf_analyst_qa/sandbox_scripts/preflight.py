# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check document access, credentials, devices, and network access as the agent user."""

import argparse
import os
import pathlib
import socket
import urllib.parse


def check(source_mount: pathlib.Path, model_url: str) -> None:
    """Reject a sandbox that exposes task data or cannot reach the model server."""
    if os.geteuid() != 65534:
        raise RuntimeError("document tools must run without root privileges")
    docs = pathlib.Path("/workspace/docs")
    if not any(docs.rglob("*.pdf")):
        raise RuntimeError("selected PDFs are missing")
    if os.access(str(source_mount), os.R_OK | os.X_OK):
        raise RuntimeError("unselected shared bucket PDFs are accessible")
    if pathlib.Path("/var/lib/pdf-analyst-qa-cache").exists() and os.access(
        "/var/lib/pdf-analyst-qa-cache", os.R_OK | os.X_OK
    ):
        raise RuntimeError("node cache backing paths are accessible")
    for path in [docs, *docs.rglob("*")]:
        if os.access(path, os.W_OK):
            raise RuntimeError("document evidence is writable")
    for name in ("tasks", "document_store", "conflict_dossiers"):
        if pathlib.Path("/workspace", name).exists():
            raise RuntimeError("hidden task data is visible")
    if any(key.lower().endswith(("api_key", "access_token", "secret_access_key")) for key in os.environ):
        raise RuntimeError("control credentials are visible to document tools")
    for pattern in ("nvidia*", "infiniband", "nvidia-caps", "nvidia-imex*"):
        if list(pathlib.Path("/dev").glob(pattern)):
            raise RuntimeError("GPU or fabric devices are exposed")
    try:
        connection = socket.create_connection(("1.1.1.1", 443), timeout=2)
    except OSError:
        pass
    else:
        connection.close()
        raise RuntimeError("sandbox network egress is available")
    model = urllib.parse.urlsplit(model_url)
    if not model.hostname:
        raise RuntimeError("model server address is invalid")
    try:
        connection = socket.create_connection(
            (model.hostname, model.port or (443 if model.scheme == "https" else 80)), timeout=5
        )
    except OSError as error:
        raise RuntimeError("model server is unreachable from the sandbox") from error
    else:
        connection.close()
    probe = pathlib.Path("/workspace/task/.write-probe")
    probe.write_text("scratch")
    probe.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_mount", type=pathlib.Path)
    parser.add_argument("model_url")
    args = parser.parse_args()
    check(args.source_mount, args.model_url)
