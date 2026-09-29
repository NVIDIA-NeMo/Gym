# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pinned HTTP / strict NIXL router, in its own environment and process group."""

import hashlib
import json
import re
import socket
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from responses_api_models.local_vllm_model.subprocess_launcher import (
    VLLMSubprocessConfig,
    VLLMSubprocessLauncher,
    redacted_argv,
)


class VLLMRouterConfig(VLLMSubprocessConfig):
    executable: str = "vllm-router"
    expected_version: Literal["0.1.15"] = "0.1.15"
    executable_type: Literal["python", "binary"] = "binary"
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    policy: Literal["consistent_hash", "session_balanced", "cache_aware"] = "consistent_hash"
    # Opt in explicitly when comparing with the unpatched external-vLLM script.
    # Strict routing remains the default for existing managed deployments.
    profile: Literal["strict", "external_benchmark"] = "strict"
    host: str = "127.0.0.1"
    request_timeout_seconds: float = Field(default=2, gt=0, allow_inf_nan=False)
    inference_timeout_seconds: int = Field(default=600, ge=1)
    # Liveness probes must tolerate busy HTTP frontends independently of the
    # inference deadline. Never compensate for a failed generation with retries.
    health_check_interval_seconds: int = Field(default=5, ge=1, strict=True)
    health_check_timeout_seconds: int = Field(default=5, ge=1, strict=True)
    health_failure_threshold: int = Field(default=3, ge=1, strict=True)
    health_success_threshold: int = Field(default=2, ge=1, strict=True)
    metrics_port: int | None = Field(default=None, ge=1, le=65535)
    cache_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    balance_abs_threshold: int | None = Field(default=None, ge=0, strict=True)
    balance_rel_threshold: float | None = Field(default=None, ge=1, allow_inf_nan=False)
    eviction_interval: int | None = Field(default=None, ge=1, strict=True)
    max_tree_size: int | None = Field(default=None, ge=1, strict=True)
    log_dir: Path = Path("results/local_vllm_router")

    @model_validator(mode="after")
    def binary_requires_digest(self) -> "VLLMRouterConfig":
        if self.executable_type == "binary" and not self.expected_sha256:
            raise ValueError("Pin expected_sha256 when selecting a standalone router binary")
        return self


class VLLMRouterLauncher(VLLMSubprocessLauncher):
    config: VLLMRouterConfig

    def __init__(
        self, *, config: VLLMRouterConfig, model: str, api_key: str, allocated_worker_hosts: set[str] | None = None
    ):
        # Only the allocation-aware cluster owner supplies remote addresses.
        # Normal managed local serving keeps the loopback-only boundary.
        self.worker_hosts = allocated_worker_hosts or {"127.0.0.1", "localhost"}
        self.remote_workers = allocated_worker_hosts is not None
        super().__init__(config=config, model=model, api_key=api_key, kwargs={}, env={}, cache_dir="")
        if not api_key:
            self.env.pop("VLLM_API_KEY", None)

    async def start(
        self,
        port: int,
        *,
        worker_urls: list[str],
        dp_size: int = 1,
        prefill_urls: list[str] | None = None,
        decode_urls: list[str] | None = None,
        dry_run: bool = False,
    ) -> str:
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Router port must be between 1 and 65535")
        if self.run_dir is not None:
            raise RuntimeError("VLLMRouterLauncher is single-use")
        pd = prefill_urls is not None or decode_urls is not None
        if pd and (not prefill_urls or not decode_urls or worker_urls):
            raise ValueError("PD requires both prefill_urls and decode_urls, without worker_urls")
        all_urls = [*prefill_urls, *decode_urls] if pd else worker_urls
        if not all_urls or len(set(all_urls)) != len(all_urls):
            raise ValueError("Router requires distinct worker URLs")
        for url in all_urls:
            parsed = urlsplit(url)
            if (
                parsed.scheme != "http"
                or parsed.hostname not in self.worker_hosts
                or not parsed.port
                or parsed.path
                or parsed.query
                or parsed.fragment
                or parsed.username
            ):
                raise ValueError("Managed router workers must be approved HTTP origins, without /v1")
        if type(dp_size) is not int or dp_size < 1:
            raise ValueError("dp_size must be a positive integer")
        host = self.config.host
        if host not in self.worker_hosts | {"127.0.0.1", "localhost"}:
            raise ValueError("Router must bind to loopback or an allocated worker address")
        self._prepare_run(port, host=host, prefix="router-", dry_run=dry_run)
        try:
            digest = hashlib.sha256(Path(self.executable).read_bytes()).hexdigest()
            if self.config.expected_sha256 and digest != self.config.expected_sha256:
                raise ValueError("Router executable SHA256 does not match expected_sha256")
            # The pinned Python console entrypoint has no --version. Query only
            # its own interpreter, never import router/vLLM into Gym's Python.
            if self.config.executable_type == "binary":
                output = await self._probe(["--version"], "version")
                versions = re.findall(r"^vllm-router ([0-9.]+)\s*$", output, re.MULTILINE)
                if len(versions) != 1:
                    raise ValueError("Router --version did not contain one unambiguous version line")
                version = versions[0]
            else:
                interpreter = Path(self.executable).with_name("python")
                version = (
                    await self._probe(
                        ["-c", "from importlib.metadata import version; print(version('vllm-router'))"],
                        "version",
                        executable=str(interpreter),
                    )
                ).strip()
            if version != self.config.expected_version:
                raise ValueError(f"Expected vllm-router {self.config.expected_version}, got {version!r}")
            help_text = await self._probe(["--help"], "help")
            if self.config.policy == "session_balanced" and not re.search(r"\bsession_balanced\b", help_text):
                raise ValueError("Selected router does not advertise the session_balanced policy")
            with socket.socket() as metrics:
                metrics.bind(("127.0.0.1", self.config.metrics_port or 0))
                metrics_port = metrics.getsockname()[1]
            if metrics_port == port:
                raise ValueError("Router metrics and HTTP ports must differ")
            strict = self.config.profile == "strict"
            args = {
                "host": host,
                "port": port,
                "intra-node-data-parallel-size": dp_size,
                "request-timeout-secs": self.config.inference_timeout_seconds,
                "worker-startup-timeout-secs": int(self.config.startup_timeout_seconds),
                "prometheus-host": host,
                "prometheus-port": metrics_port,
            }
            for knob in (
                "cache_threshold",
                "balance_abs_threshold",
                "balance_rel_threshold",
                "eviction_interval",
                "max_tree_size",
            ):
                value = getattr(self.config, knob)
                if value is not None:
                    args[knob.replace("_", "-")] = value
            if strict:
                args.update(
                    {
                        "policy": self.config.policy,
                        "api-key": self.api_key,
                        "worker-startup-check-interval": 1,
                        "health-check-interval-secs": self.config.health_check_interval_seconds,
                        "health-check-timeout-secs": self.config.health_check_timeout_seconds,
                        "health-failure-threshold": self.config.health_failure_threshold,
                        "health-success-threshold": self.config.health_success_threshold,
                        "cb-failure-threshold": 1,
                        "cb-success-threshold": 1,
                        "cb-timeout-duration-secs": 30,
                        "cb-window-duration-secs": 60,
                    }
                )
            else:
                # Preserve the external benchmark's upstream retry/health defaults.
                args["log-level"] = "error"
                if self.api_key:
                    args["api-key"] = self.api_key
                if not pd:
                    args["policy"] = self.config.policy
            mode_flags = {"--worker-urls"}
            if pd:
                args.update({"prefill-policy": self.config.policy, "decode-policy": self.config.policy})
                mode_flags = {"--vllm-pd-disaggregation", "--prefill", "--decode"}
                if strict:
                    args["kv-connector"] = "nixl"
                    mode_flags.add("--strict-nixl")
            required = {f"--{key}" for key in args} | mode_flags
            if strict:
                required.add("--disable-retries")
            missing = required - set(re.findall(r"--[a-z][a-z0-9-]*", help_text))
            if missing:
                raise ValueError(f"Selected router does not advertise {sorted(missing)}")
            argv = [self.executable, *(["--disable-retries"] if strict else [])]
            if pd:
                argv.append("--vllm-pd-disaggregation")
                if strict:
                    argv.append("--strict-nixl")
                for flag, urls in (("--prefill", prefill_urls), ("--decode", decode_urls)):
                    for url in urls:
                        argv.extend([flag, url])
            else:
                argv.extend(["--worker-urls", *worker_urls])
            argv.extend(f"--{key}={value}" for key, value in args.items())
            manifest = {
                "router_version": version,
                "profile": self.config.profile,
                "executable": self.executable,
                "entrypoint_sha256": digest,
                "executable_type": self.config.executable_type,
                "argv_redacted": redacted_argv(argv),
                "base_url": self.base_url,
                "metrics_url": f"http://{host}:{metrics_port}/metrics",
                "worker_urls": worker_urls,
                "dp_size_per_worker": dp_size,
                "routing_authority": "vllm_router",
                "session_header": "X-Session-ID",
                "dry_run": dry_run,
                "scope": ("allocated multi-node" if self.remote_workers else "single-node")
                + (" NIXL PD" if pd else " non-PD"),
                "prefill_urls": prefill_urls,
                "decode_urls": decode_urls,
                "combined_fallback": False if pd and strict else None,
            }
            (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if not dry_run:
                await self.owner.start(argv, self.env, self.run_dir / "server.log")
                await self._await_ready()
            return self.base_url
        except BaseException:
            await self.stop()
            raise
