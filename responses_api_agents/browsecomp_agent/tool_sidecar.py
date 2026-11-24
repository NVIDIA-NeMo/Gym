# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BrowseComp runs tools in a sibling OpenSandbox sandbox and grades on the supervisor."""

import asyncio
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal
from urllib.parse import urlsplit

from omegaconf import ListConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PositiveFloat, TypeAdapter, field_validator

from nemo_gym.config import ResourcesServerRef
from nemo_gym.sandbox import AsyncSandbox, AsyncSandboxSidecar, SandboxExecResult, SandboxSidecarSpec, SandboxSpec
from nemo_gym.utils.network_policy import reachable_host, restricted_network_policy


# Only search/tool settings cross into the pod. No judge configuration, answers, model credentials or host URLs.
_TOOL_SETTINGS = (
    "search_provider",
    "tavily_api_key",
    "exa_api_key",
    "max_page_bytes",
    "max_results",
    "search_max_total_length",
    "exa_search_writes_pages",
    "exa_search_type",
    "provider_rate_limit_wait_seconds",
    "provider_request_timeout_seconds",
    "bash_timeout_s",
    "bash_max_concurrency",
)
_SEARCH_HOSTS = {"tavily": "api.tavily.com", "exa": "api.exa.ai"}
# Exit codes from a command that could not run at all: not executable, or not found.
_IMAGE_EXIT_CODES = {126, 127}
_STDERR_TAIL_CHARS = 2000


class ToolSidecarImageError(RuntimeError):
    """The configured tools image cannot run the service."""


def _exec_failure(message: str, result: SandboxExecResult) -> str:
    """Name the exit status and the end of stderr so an operator can see why the tools process failed."""
    stderr = (result.stderr or "").strip()[-_STDERR_TAIL_CHARS:]
    detail = f"exit code {result.return_code}" if result.error_type is None else f"error {result.error_type}"
    return f"{message} ({detail}){': ' + stderr if stderr else ''}"


def _is_image_failure(result: SandboxExecResult) -> bool:
    return result.error_type is None and (
        result.return_code in _IMAGE_EXIT_CODES or "No module named" in (result.stderr or "")
    )


class BrowsecompToolSidecarConfig(BaseModel):
    """A prebuilt tools image and the outside Resources Server whose search settings it uses."""

    model_config = ConfigDict(extra="forbid")

    resources_server: ResourcesServerRef
    image: str = Field(min_length=1)
    python: str = "/opt/gym/.venv/bin/python"
    port: int = Field(default=8080, ge=1024, le=65535)
    workspace: Literal["none", "per_session"] = "none"
    exclude_domains_file_path: str = "/opt/gym/exclude-domains.json"
    resources: dict[str, JsonValue] = Field(default_factory=lambda: {"cpu": 1, "memory_mib": 2048})
    startup_timeout_seconds: PositiveFloat = 120
    model_base_url: str | None = Field(
        default=None,
        description="Full sandbox-facing Responses API base URL, including any deployment prefix and /v1.",
    )

    @field_validator("model_base_url")
    @classmethod
    def _validate_model_base_url(cls, value: str | None) -> str | None:
        if value is not None:
            endpoint_host(value)
            parsed = urlsplit(value)
            if parsed.query or parsed.fragment:
                raise ValueError("model_base_url must not contain a query or fragment")
            return value.rstrip("/")
        return value

    def tool_settings(self, resources: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Select tool configuration without copying verifier or task state."""
        settings = {key: resources[key] for key in _TOOL_SETTINGS if key in resources} | {
            "workspace": self.workspace,
            "exclude_domains_file_path": self.exclude_domains_file_path,
        }
        for key, value in settings.items():
            if isinstance(value, ListConfig):
                settings[key] = OmegaConf.to_container(value, resolve=True)
        provider = str(settings.get("search_provider", "tavily"))
        settings.pop("exa_api_key" if provider == "tavily" else "tavily_api_key", None)
        return settings


def endpoint_host(url: str) -> str:
    """Normalize a granted origin for comparison and the host allowlist."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Tool sidecars require an absolute HTTP(S) model endpoint without embedded credentials")
    return reachable_host(url).lower().rstrip(".")


def sidecar_spec(
    spec: SandboxSpec,
    *,
    config: BrowsecompToolSidecarConfig,
    model_url: str,
    outside_urls: list[str],
    settings: Mapping[str, JsonValue],
    tool_token: str,
) -> SandboxSpec:
    """Allow only inference and the selected public search API through OpenSandbox egress."""
    if spec.ttl_s is None or not math.isfinite(spec.ttl_s) or spec.ttl_s <= 0:
        raise ValueError("BrowseComp tool_sidecar requires a finite, positive sandbox ttl_s")
    if spec.sidecars or "sidecars" in spec.provider_options:
        raise ValueError("BrowseComp tool_sidecar owns the rollout's sole sidecar; remove other sidecar declarations")
    if "network_policy" in spec.provider_options:
        raise ValueError("BrowseComp tool_sidecar supplies its OpenSandbox egress policy; remove custom network rules")
    model_host = endpoint_host(model_url)
    # Host rules do not distinguish services on different ports of the same host.
    outside_hosts = {urlsplit(url).hostname.lower().rstrip(".") for url in outside_urls if urlsplit(url).hostname}
    if model_host in outside_hosts:
        raise ValueError("Use a dedicated model-serving host, separate from the Agent and Resources Servers")
    search_provider = str(settings.get("search_provider", "tavily"))
    if search_provider not in _SEARCH_HOSTS:
        raise ValueError("BrowseComp tool sidecars support the tavily and exa search providers")
    tools = SandboxSidecarSpec(
        name="tools",
        image=config.image,
        resources=config.resources,
        env={
            "BROWSECOMP_TOOL_SETTINGS": json.dumps({**settings, "port": config.port}),
            "BROWSECOMP_TOOL_TOKEN": tool_token,
        },
    )
    policy = restricted_network_policy("opensandbox", [model_url, f"https://{_SEARCH_HOSTS[search_provider]}"])
    return replace(spec, sidecars=(tools,), provider_options={**spec.provider_options, "network_policy": policy})


@dataclass
class BrowsecompToolSidecar:
    """An explicitly started process in an already-owned placement group; it never creates or destroys a sandbox."""

    sandbox: AsyncSandbox
    config: BrowsecompToolSidecarConfig
    service: asyncio.Task[SandboxExecResult] | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.config.port}"

    async def start(self, *, lifetime_seconds: float) -> None:
        """Start the preconfigured tools process and probe through named sidecar exec."""
        sidecar = self.sandbox.sidecar("tools")
        self.service = asyncio.create_task(
            sidecar.exec(
                self.config.python,
                "-m",
                "resources_servers.browsecomp_advanced_harness.tools_sidecar",
                timeout_s=lifetime_seconds,
            )
        )
        try:
            async with asyncio.timeout(self.config.startup_timeout_seconds):
                await self._wait_until_ready(sidecar)
        except TimeoutError as error:
            raise TimeoutError(
                f"BrowseComp tools sidecar was not ready on {self.url}/health within "
                f"{self.config.startup_timeout_seconds:g} s"
            ) from error

    async def _wait_until_ready(self, sidecar: AsyncSandboxSidecar) -> None:
        while True:
            if self.service.done():
                exited = await self.service
                message = _exec_failure("BrowseComp tools sidecar exited before readiness", exited)
                if _is_image_failure(exited):
                    raise ToolSidecarImageError(f"{message}; check tool_sidecar.image")
                raise RuntimeError(message)
            probe = await sidecar.exec(
                self.config.python,
                "-c",
                "import sys,urllib.request; urllib.request.urlopen(sys.argv[1],timeout=3).read()",
                f"{self.url}/health",
                timeout_s=5,
            )
            if probe.return_code == 0 and not probe.error_type:
                return
            await asyncio.sleep(0.25)

    async def metrics(self) -> dict[str, JsonValue]:
        """The host pulls tool metrics before sandbox teardown; the agent never calls the verifier."""
        result = await self.sandbox.sidecar("tools").exec(
            self.config.python,
            "-c",
            "import os,sys,urllib.request as u; req=u.Request(sys.argv[1],"
            "headers={'Authorization':'Bearer '+os.environ['BROWSECOMP_TOOL_TOKEN']}); "
            "print(u.urlopen(req,timeout=10).read().decode())",
            f"{self.url}/_metrics",
            timeout_s=15,
        )
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(_exec_failure("Could not collect BrowseComp tool-sidecar metrics", result))
        return TypeAdapter(dict[str, JsonValue]).validate_json(result.stdout or "")

    async def close(self) -> None:
        """Release the process waiter after the owner has stopped the sandbox group."""
        if self.service is not None:
            self.service.cancel()
            await asyncio.gather(self.service, return_exceptions=True)
