# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wire schema for the sandbox server HTTP protocol.

One source of truth for the request/response bodies, imported by both the
server (which validates incoming requests and serializes responses) and the
remote provider client (which builds requests and parses responses).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from nemo_gym.sandbox.ref import SCOPE_OPERATE


class CreateSandboxRequest(BaseModel):
    """POST /sandboxes — mirrors the createable fields of ``SandboxSpec``."""

    image: Optional[str] = None
    ttl_s: Optional[float] = None
    ready_timeout_s: Optional[float] = None
    workdir: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)
    entrypoint: Optional[list[str]] = None
    provider_options: dict[str, Any] = Field(default_factory=dict)


class ExecRequest(BaseModel):
    """POST /sandboxes/{id}/exec."""

    command: str
    cwd: Optional[str] = None
    env: Optional[dict[str, str]] = None
    timeout_s: Optional[float] = None
    user: Optional[Any] = None


class ExecResponse(BaseModel):
    """Result of exec (mirrors ``SandboxExecResult``)."""

    stdout: Optional[str] = None
    stderr: Optional[str] = None
    return_code: int
    error_type: Optional[str] = None


class UploadRequest(BaseModel):
    """POST /sandboxes/{id}/upload — one base64-encoded file."""

    target_path: str
    contents_b64: str


class DownloadResponse(BaseModel):
    """GET /sandboxes/{id}/download — one base64-encoded file."""

    contents_b64: str


class StatusResponse(BaseModel):
    """GET /sandboxes/{id}/status."""

    status: str


class LeaseRequest(BaseModel):
    """POST /sandboxes/{id}/leases — mint a co-lease on an existing sandbox."""

    scope: str = SCOPE_OPERATE
    ttl_s: Optional[float] = None


class ReleaseResponse(BaseModel):
    """DELETE /sandboxes/{id}/leases/release."""

    released: bool
    remaining_leases: int


class DeleteResponse(BaseModel):
    """DELETE /sandboxes/{id}."""

    deleted: bool
