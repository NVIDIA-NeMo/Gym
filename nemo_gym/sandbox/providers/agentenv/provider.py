# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentEnv sandbox provider with state-preserving fork support."""

from __future__ import annotations

import ssl
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxEndpoint,
    SandboxExecResult,
    SandboxHandle,
    SandboxProvider,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.utils import coerce_config


@dataclass(frozen=True)
class AgentEnvConnectionConfig:
    endpoint: str | None = None
    api_key: str | None = None
    tls_ca: str | None = None
    request_timeout_s: float = 300.0

    def __post_init__(self) -> None:
        if self.endpoint is not None:
            endpoint = self.endpoint.strip().rstrip("/")
            parsed = urlsplit(endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("connection.endpoint must be an absolute http(s) URL")
            object.__setattr__(self, "endpoint", endpoint)
        if self.request_timeout_s <= 0:
            raise ValueError("connection.request_timeout_s must be > 0")


@dataclass(frozen=True)
class AgentEnvCreateConfig:
    template: str = "osworld-slim"
    timeout_s: int = 7200

    def __post_init__(self) -> None:
        if not self.template.strip():
            raise ValueError("create.template must be non-empty")
        if self.timeout_s <= 0:
            raise ValueError("create.timeout_s must be > 0")


@dataclass(frozen=True)
class AgentEnvProviderOptions:
    template: str | None = None

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any] | None) -> "AgentEnvProviderOptions":
        if options is None:
            return cls()
        if not isinstance(options, Mapping):
            raise TypeError("AgentEnv provider_options must be a mapping")
        unknown = set(options) - {"template"}
        if unknown:
            raise ValueError(f"Unknown AgentEnv provider option(s): {', '.join(sorted(unknown))}. Supported: template")
        template = options.get("template")
        if template is not None and (not isinstance(template, str) or not template.strip()):
            raise ValueError("AgentEnv provider option 'template' must be a non-empty string")
        return cls(template=template)


def _status(value: Any) -> SandboxStatus:
    normalized = str(value or "").lower()
    if normalized in {"running", "ready"}:
        return SandboxStatus.RUNNING
    if normalized in {"creating", "pending", "starting"}:
        return SandboxStatus.STARTING
    if normalized in {"deleted", "ended", "stopped", "terminated"}:
        return SandboxStatus.STOPPED
    if normalized in {"error", "failed"}:
        return SandboxStatus.ERROR
    return SandboxStatus.UNKNOWN


class AgentEnvProvider:
    """Provider for the AgentEnv management API and E2B-style service proxy."""

    name = "agentenv"

    def __init__(
        self,
        *,
        connection: AgentEnvConnectionConfig | Mapping[str, Any] | None = None,
        create: AgentEnvCreateConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = coerce_config(connection, AgentEnvConnectionConfig)
        self._create = coerce_config(create, AgentEnvCreateConfig)
        self._client: httpx.AsyncClient | None = None

    def _require_connection(self) -> tuple[str, str]:
        endpoint = self._connection.endpoint
        api_key = self._connection.api_key
        if not endpoint:
            raise ValueError("AgentEnv connection.endpoint is required")
        if not api_key:
            raise ValueError("AgentEnv connection.api_key is required")
        return endpoint, api_key

    def _build_client(self) -> httpx.AsyncClient:
        endpoint, api_key = self._require_connection()
        verify: ssl.SSLContext | bool = True
        if self._connection.tls_ca:
            ca_path = Path(self._connection.tls_ca)
            if not ca_path.is_file():
                raise FileNotFoundError(f"AgentEnv TLS CA file not found: {ca_path}")
            context = ssl.create_default_context()
            context.load_verify_locations(cafile=str(ca_path))
            verify = context
        return httpx.AsyncClient(
            base_url=endpoint,
            headers={"X-API-Key": api_key},
            timeout=self._connection.request_timeout_s,
            verify=verify,
        )

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = self._build_client()
        return self._client

    async def _request(self, method: str, path: str, *, json: Any = None) -> httpx.Response:
        response = await self._http().request(method, path, json=json)
        response.raise_for_status()
        return response

    @staticmethod
    def _handle(sandbox_id: str, *, template: str | None = None, ttl_s: int | float | None = None) -> SandboxHandle:
        return SandboxHandle(
            sandbox_id=sandbox_id,
            provider_name=AgentEnvProvider.name,
            raw={"template": template, "ttl_s": ttl_s},
        )

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if spec.entrypoint:
            raise SandboxCreateError(
                "AgentEnv templates define their entrypoint; SandboxSpec.entrypoint is unsupported"
            )
        if spec.env or spec.files:
            raise SandboxCreateError("AgentEnv OSWorld templates do not support per-sandbox env or files")
        options = AgentEnvProviderOptions.from_mapping(spec.provider_options)
        template = options.template or (spec.image if spec.image else None) or self._create.template
        ttl_s = spec.ttl_s if spec.ttl_s is not None else self._create.timeout_s
        if ttl_s <= 0:
            raise SandboxCreateError("AgentEnv sandbox ttl_s must be > 0")
        try:
            response = await self._request(
                "POST",
                "/sandboxes",
                json={"templateID": template, "timeout": int(ttl_s)},
            )
            sandbox_id = str(response.json()["sandboxID"])
        except Exception as exc:
            raise SandboxCreateError(f"Failed to create AgentEnv sandbox from template {template!r}: {exc}") from exc
        return self._handle(sandbox_id, template=template, ttl_s=ttl_s)

    async def endpoint(self, handle: SandboxHandle, port: int) -> SandboxEndpoint:
        endpoint, api_key = self._require_connection()
        return SandboxEndpoint(
            endpoint=endpoint,
            headers={
                "X-API-Key": api_key,
                "E2b-Sandbox-Id": handle.sandbox_id,
                "E2b-Sandbox-Port": str(port),
            },
        )

    async def fork(
        self,
        handle: SandboxHandle,
        count: int,
        *,
        ttl_s: int | float | None = None,
    ) -> list[tuple[SandboxProvider, SandboxHandle]]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("AgentEnv fork count must be a positive integer")
        raw = handle.raw if isinstance(handle.raw, Mapping) else {}
        effective_ttl = ttl_s if ttl_s is not None else raw.get("ttl_s") or self._create.timeout_s
        response = await self._request(
            "POST",
            f"/sandboxes/{handle.sandbox_id}/fork",
            json={"count": count, "timeout": int(effective_ttl)},
        )
        payload = response.json()
        if not isinstance(payload, list) or len(payload) != count:
            raise RuntimeError(
                f"AgentEnv fork returned {len(payload) if isinstance(payload, list) else 'invalid'} children"
            )

        children: list[tuple[SandboxProvider, SandboxHandle]] = []
        for item in payload:
            sandbox_id = str(item["sandbox"]["sandboxID"])
            child_provider = AgentEnvProvider(connection=self._connection, create=self._create)
            children.append(
                (
                    child_provider,
                    self._handle(
                        sandbox_id,
                        template=raw.get("template"),
                        ttl_s=effective_ttl,
                    ),
                )
            )
        return children

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        del scope
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        return self._handle(str(descriptor["sandbox_id"]))

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        try:
            response = await self._request("GET", f"/sandboxes/{handle.sandbox_id}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return SandboxStatus.STOPPED
            return SandboxStatus.UNKNOWN
        except Exception:
            return SandboxStatus.UNKNOWN
        return _status(response.json().get("state"))

    async def close(self, handle: SandboxHandle) -> None:
        try:
            await self._request("DELETE", f"/sandboxes/{handle.sandbox_id}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        del handle, command, cwd, env, timeout_s, user
        raise NotImplementedError("AgentEnv OSWorld sandboxes expose services, not a generic exec API")

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        del handle, source_path, target_path
        raise NotImplementedError("AgentEnv OSWorld sandboxes do not support generic file upload")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        del handle, source_path, target_path
        raise NotImplementedError("AgentEnv OSWorld sandboxes do not support generic file download")
