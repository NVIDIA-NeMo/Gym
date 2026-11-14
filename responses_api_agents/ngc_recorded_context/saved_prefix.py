# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicitly resume an immutable policy prefix; never resend its API calls."""

import hashlib
import json
from pathlib import Path


def prefix_files(directory: Path):
    order = directory / "api-order.jsonl"
    requests = [directory / json.loads(line)["request"] for line in order.read_text().splitlines()]
    files = [directory / "identity.json", order]
    for path in requests:
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Saved prefix escapes capture")
        files.extend(
            [
                path,
                path.with_name(path.name.replace(".request.json", ".response.json")),
                path.with_name(path.name.replace(".request.json", ".receipt.json")),
            ]
        )
    return requests, files


def prefix_digest(directory: Path) -> str:
    _, files = prefix_files(directory)
    manifest = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def verification_digest(directory: Path) -> str:
    """Bind the grading capture and event receipts to the agent's returned response."""
    manifest = {}
    for name in (
        "identity.json",
        "events.json",
        "drafts.json",
        "terminal_reply.json",
        "result.json",
        "visible-output.json",
        "ordering-verification-input.json",
        "native-step.json",
    ):
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Verification receipt escapes capture")
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


class SavedPolicyPrefix:
    def __init__(self, directory: Path, expected_sha256: str, model: str):
        self.directory = directory.resolve()
        if not expected_sha256 or prefix_digest(self.directory) != expected_sha256:
            raise ValueError("Saved policy prefix checksum mismatch")
        self.requests, _ = prefix_files(self.directory)
        self.identity = json.loads((self.directory / "identity.json").read_text())
        if self.identity["model"] != model:
            raise ValueError("Saved policy prefix model mismatch")
        self.consumed = 0

    def take(self, payload):
        if self.consumed == len(self.requests):
            return None
        path = self.requests[self.consumed]
        if json.loads(path.read_text()) != payload:
            raise ValueError("Reconstructed policy prefix request differs")
        receipt = json.loads(path.with_name(path.name.replace(".request.json", ".receipt.json")).read_text())
        response = json.loads(path.with_name(path.name.replace(".request.json", ".response.json")).read_text())
        if receipt["http_status"] != 200 or response.get("model") != self.identity["model"]:
            raise ValueError("Only successful same-model policy calls may be resumed")
        self.consumed += 1
        return receipt["http_status"], response
