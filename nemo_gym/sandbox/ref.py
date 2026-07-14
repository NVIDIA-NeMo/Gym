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

"""Serializable capability to operate a sandbox owned by a sandbox server.

A ``SandboxRef`` is the whole reason the sandbox server exists: it names one
physical sandbox by a stable id plus a signed lease token, so a sandbox created
by one Gym server (the agent) can be operated by another (the verifier) over
HTTP. Unlike ``SandboxHandle.raw`` (provider-owned, in-process state that cannot
leave the process that created it), a ``SandboxRef`` is plain JSON and travels
in request bodies.

Scope semantics:
- ``owner``   — may run commands, transfer files, AND end the sandbox lifecycle.
                Held by the server that created the sandbox (the loop owner).
- ``operate`` — may run commands and transfer files, but may NOT end the
                lifecycle. Held by a co-lessee (the verifier that attached).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCOPE_OWNER = "owner"
SCOPE_OPERATE = "operate"


@dataclass(frozen=True)
class SandboxRef:
    """A serializable capability to operate one sandbox on a sandbox server.

    Never carries provider ``raw`` state; ``lease_token`` is the server-signed
    grant that authorizes operations and encodes ``{sandbox_id, rollout_id,
    scope, expires_at}``.
    """

    server_url: str
    sandbox_id: str
    lease_token: str
    provider_name: str = ""
    scope: str = SCOPE_OPERATE
    workdir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_url": self.server_url,
            "sandbox_id": self.sandbox_id,
            "lease_token": self.lease_token,
            "provider_name": self.provider_name,
            "scope": self.scope,
            "workdir": self.workdir,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SandboxRef":
        return cls(
            server_url=str(data["server_url"]),
            sandbox_id=str(data["sandbox_id"]),
            lease_token=str(data["lease_token"]),
            provider_name=str(data.get("provider_name") or ""),
            scope=str(data.get("scope") or SCOPE_OPERATE),
            workdir=data.get("workdir"),
            extra=dict(data.get("extra") or {}),
        )

    @property
    def can_close(self) -> bool:
        return self.scope == SCOPE_OWNER
