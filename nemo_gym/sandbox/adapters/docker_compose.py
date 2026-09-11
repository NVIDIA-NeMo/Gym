# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Compose services through the sandbox API.

Docker Compose normalizes the input YAML; unsupported runtime semantics are
rejected before provisioning services.
"""

import asyncio
import re
import shlex
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxSpec, create_provider
from nemo_gym.sandbox.providers.base import (
    SupportsSandboxEndpoint,
    SupportsSandboxNetwork,
    SupportsSandboxPortForwarding,
    SupportsSandboxRuntimeRequirements,
    SupportsSandboxSharedStorage,
)


async def _command(argv: Sequence[str]) -> str:
    process = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await process.communicate()
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"{argv[0]} failed ({process.returncode}): {stderr.decode(errors='replace')}")
    return stdout.decode(errors="replace")


def _seconds(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = re.findall(r"(\d+(?:\.\d+)?)(ns|us|ms|s|m|h)", value)
    if not parts or "".join(number + unit for number, unit in parts) != value:
        raise ValueError(f"Invalid Compose duration: {value!r}")
    return sum(
        float(number) * {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1, "m": 60, "h": 3600}[unit]
        for number, unit in parts
    )


class AsyncSandboxCompose:
    """Start services from a Compose YAML file and own their sandbox lifecycle.

    Deployment settings stay in the provider and optional ``service_specs``.
    Compose fields override the corresponding spec fields.
    """

    def __init__(
        self,
        provider,
        compose_file: str | Path | None,
        *,
        compose_command: Sequence[str] = ("docker-compose",),
        image_platform: str = "linux/amd64",
        image_configs: Mapping[str, Mapping[str, Any]] | None = None,
        registry_options: Mapping[str, Any] | None = None,
        service_specs: Mapping[str, SandboxSpec] | None = None,
        timeout_s: float = 1200,
        poll_interval_s: float = 0.5,
        volume_init_image: str = "alpine:3.22",
        volume_sources: Mapping[str, str] | None = None,
    ):
        self.provider = create_provider(provider) if isinstance(provider, Mapping) else provider
        self.compose_file = Path(compose_file).resolve() if compose_file is not None else None
        self.compose_command = tuple(compose_command)
        self.document: dict[str, Any] = {}
        self._image_configs = dict(image_configs or {})
        self.image_platform = image_platform
        if len(image_platform.split("/")) not in (2, 3) or not all(image_platform.split("/")):
            raise ValueError("image_platform must be os/architecture[/variant]")
        self.registry_options = dict(registry_options or {})
        self.service_specs = dict(service_specs or {})
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        if timeout_s <= 0 or poll_interval_s <= 0:
            raise ValueError("Compose timeout_s and poll_interval_s must be positive")
        self.volume_init_image = volume_init_image
        self.volume_sources = dict(volume_sources or {})
        self.project = "compose-" + uuid.uuid4().hex
        self.services: dict[str, AsyncSandbox] = {}
        self._plans: dict[str, dict[str, Any]] = {}
        self._started = False
        self._ready = False
        self._closed = False
        self._volume_helper: AsyncSandbox | None = None
        self._stop_task: asyncio.Task | None = None
        self._processes: dict[str, asyncio.Task] = {}
        self._seeds: list[AsyncSandbox] = []

    def _inspect_image(self, image: str) -> dict:
        from oras.container import Container
        from oras.provider import Registry

        container = Container(image)
        if container.registry in {"docker.io", "index.docker.io"}:
            container.registry = "registry-1.docker.io"
            container.namespace = container.namespace or "library"
        platform = dict(zip(("os", "architecture", "variant"), self.image_platform.split("/")))
        client = Registry(**self.registry_options)
        # ORAS shares this session with its authentication backend.
        client.session.request = partial(client.session.request, timeout=self.timeout_s)
        try:
            manifest = client.get_manifest(container)
            if "manifests" in manifest:
                candidates = [
                    item
                    for item in manifest["manifests"]
                    if all(item.get("platform", {}).get(key) == value for key, value in platform.items())
                ]
                if len(candidates) != 1:
                    raise ValueError(f"Image {image!r} needs one manifest for {self.image_platform!r}")
                container.digest = candidates[0]["digest"]
                manifest = client.get_manifest(container)
            with client.get_blob(container, manifest["config"]["digest"]) as response:
                response.raise_for_status()
                config = response.json()
            if any(config.get(key) != value for key, value in platform.items() if key != "variant"):
                raise ValueError(f"Image {image!r} does not match {self.image_platform!r}")
            return config["config"]
        finally:
            client.session.close()

    async def _normalize(self):
        raw = await _command([*self.compose_command, "-f", str(self.compose_file), "config", "--format", "yaml"])
        # `config` emits escaped dollars for Compose to consume at runtime.
        document = yaml.safe_load(raw.replace("$$", "$"))
        for service in (document.get("services") or {}).values():
            if "shm_size" in service:
                service["shm_size"] = int(service["shm_size"])
        return document

    def _validate(self) -> list[str]:
        unknown = set(self.document) - {"name", "services", "networks", "volumes"}
        if unknown:
            raise NotImplementedError(f"Unsupported Compose fields: {sorted(unknown)}")
        services = self.document.get("services")
        if not isinstance(services, Mapping) or not services:
            raise ValueError("Compose requires a non-empty services mapping")
        networks = self.document.get("networks") or {}
        default_network = networks.get("default") or {}
        if set(networks) - {"default"} or set(default_network) - {"name", "ipam"} or default_network.get("ipam"):
            raise NotImplementedError("Custom Compose networks require provider network isolation support")
        dependencies = {}
        aliases_seen = set(services)
        for name, service in services.items():
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
                raise ValueError(f"Invalid Compose service name: {name!r}")
            unknown = set(service) - {
                "image",
                "entrypoint",
                "command",
                "environment",
                "working_dir",
                "user",
                "depends_on",
                "healthcheck",
                "expose",
                "ports",
                "networks",
                "volumes",
                "cpus",
                "mem_limit",
                "labels",
                "restart",
                "network_mode",
                "cap_add",
                "shm_size",
                "x-sandbox",
            }
            if unknown:
                raise NotImplementedError(f"Service {name!r}: unsupported Compose fields {sorted(unknown)}")
            if service.get("restart", "no") != "no":
                raise NotImplementedError(f"Service {name!r}: restart policies are unsupported")
            if not service.get("image"):
                raise ValueError(f"Service {name!r} requires a prebuilt image in the Compose YAML")
            runtime = {"cap_add": tuple(service.get("cap_add") or ()), "shm_size": service.get("shm_size")}
            if runtime["cap_add"] or runtime["shm_size"] is not None:
                if not isinstance(self.provider, SupportsSandboxRuntimeRequirements):
                    raise NotImplementedError("Sandbox provider does not support runtime requirements")
                self.provider.validate_runtime_requirements(**runtime)
            options = service.get("x-sandbox") or {}
            if set(options) - {"hosts", "resolve_environment"}:
                raise NotImplementedError(f"Service {name!r}: unsupported x-sandbox options")
            selected = options.get("resolve_environment", [])
            if not isinstance(selected, list) or any(not isinstance(key, str) for key in selected):
                raise ValueError("x-sandbox.resolve_environment requires a list of environment variable names")
            if "hosts" in options and options["hosts"] != []:
                raise ValueError("x-sandbox.hosts only supports [] to explicitly disable host injection")
            dependencies[name] = dict(service.get("depends_on") or {})
            for dependency, config in dependencies[name].items():
                if dependency not in services:
                    raise ValueError(f"Service {name!r}: unknown dependency {dependency!r}")
                if (
                    set(config) - {"condition", "required", "restart"}
                    or config.get("restart")
                    or config.get("required") is False
                ):
                    raise NotImplementedError(f"Service {name!r}: unsupported depends_on options")
                if config.get("condition", "service_started") not in {
                    "service_started",
                    "service_healthy",
                    "service_completed_successfully",
                }:
                    raise ValueError(f"Service {name!r}: invalid dependency condition")
            mode = service.get("network_mode")
            if mode:
                target = mode.removeprefix("service:")
                if not mode.startswith("service:") or target not in services or target == name:
                    raise NotImplementedError(f"Unsupported Compose network_mode: {mode!r}")
                if service.get("networks") or service.get("ports"):
                    raise NotImplementedError(
                        "Forwarded service network mode cannot declare networks or published ports"
                    )
                if not isinstance(self.provider, SupportsSandboxPortForwarding):
                    raise NotImplementedError("Sandbox provider does not support service network forwarding")
                self.provider.validate_port_forwarding()
                dependencies[name].setdefault(target, {"condition": "service_started"})
            service_networks = service.get("networks") or {"default": {}}
            if set(service_networks) != {"default"}:
                raise NotImplementedError(f"Service {name!r}: unsupported networks")
            if set(service_networks["default"] or {}) - {"aliases"}:
                raise NotImplementedError(f"Service {name!r}: unsupported network options")
            for alias in (service_networks["default"] or {}).get("aliases", []):
                if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", alias):
                    raise ValueError(f"Invalid network alias: {alias!r}")
                if alias in aliases_seen and alias != name:
                    raise ValueError(f"Ambiguous Compose service alias: {alias!r}")
                aliases_seen.add(alias)
            for port in service.get("ports", []):
                if set(port) - {"target", "published", "host_ip", "protocol", "mode"}:
                    raise NotImplementedError(f"Service {name!r}: unsupported ports options")
                if port.get("published") not in (None, "0", 0) or port.get("host_ip"):
                    raise NotImplementedError("Fixed published ports/host_ip cannot be honored by sandbox endpoints")
                if port.get("protocol", "tcp") != "tcp" or port.get("mode", "ingress") != "ingress":
                    raise NotImplementedError("Only TCP sandbox endpoint ports are supported")
                if not isinstance(self.provider, SupportsSandboxEndpoint):
                    raise NotImplementedError("Sandbox provider does not support service endpoints")
            for volume in service.get("volumes", []):
                if not isinstance(self.provider, SupportsSandboxSharedStorage):
                    raise NotImplementedError("Sandbox provider does not support shared volumes")
                if not isinstance(volume, Mapping) or volume.get("type") not in {"bind", "volume"}:
                    raise NotImplementedError("Only normalized bind and named volumes are supported")
                if set(volume) - {"type", "source", "target", "read_only", "bind", "volume"}:
                    raise NotImplementedError("Unsupported Compose volume options")
                if not volume.get("source") or not str(volume.get("target", "")).startswith("/"):
                    raise ValueError("Volumes require a source and absolute target")
                if set(volume.get("bind") or {}) - {"create_host_path"} or set(volume.get("volume") or {}) - {
                    "nocopy"
                }:
                    raise NotImplementedError("Unsupported bind/volume options")
                if volume["type"] == "bind" and volume["source"] not in self.volume_sources:
                    raise NotImplementedError("Remote bind mounts require a volume_sources mapping to shared storage")
                if volume["type"] == "volume":
                    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", volume["source"]):
                        raise ValueError("Named volumes require a valid Compose volume name")
                    declaration = (self.document.get("volumes") or {}).get(volume["source"])
                    if declaration is None:
                        raise ValueError(f"Undeclared Compose volume: {volume['source']!r}")
                    if set(declaration) - {"name", "external"}:
                        raise NotImplementedError("Unsupported named volume driver/options")
                    if declaration.get("external") and volume["source"] not in self.volume_sources:
                        raise NotImplementedError("External volumes require a volume_sources mapping")
                source = self.volume_sources.get(volume["source"], f"{self.project}/{volume['source']}")
                self.provider.shared_volume_options(source, volume["target"], read_only=volume.get("read_only", False))
        try:
            order = list(TopologicalSorter(dependencies).static_order())
        except CycleError as error:
            raise ValueError("Compose dependency cycle") from error
        if not isinstance(self.provider, SupportsSandboxNetwork):
            raise NotImplementedError("Sandbox provider does not support networking between sandboxes")
        self.provider.validate_networking()
        return order

    async def _prepare(self):
        for name, service in self.document["services"].items():
            image = service["image"]
            if image not in self._image_configs:
                self._image_configs[image] = await asyncio.to_thread(self._inspect_image, image)
            image_config = self._image_configs[image]
            entrypoint = service.get("entrypoint")
            command = service.get("command")
            if entrypoint is None:
                entrypoint = image_config.get("Entrypoint") or []
                if command is None:
                    command = image_config.get("Cmd") or []
            elif command is None:
                command = []
            if isinstance(entrypoint, str):
                entrypoint = shlex.split(entrypoint)
            if isinstance(command, str):
                command = shlex.split(command)
            argv = list(entrypoint) + list(command or [])
            if not argv:
                raise ValueError(f"Service {name!r}: image and Compose specify no command")
            env = dict(item.split("=", 1) for item in image_config.get("Env", []) if "=" in item)
            configured_env = service.get("environment") or {}
            for key, value in configured_env.items():
                if value is None:
                    if key in env:
                        raise NotImplementedError(
                            f"Service {name!r}: unsetting inherited image environment is unsupported"
                        )
                    env.pop(key, None)
                else:
                    env[key] = str(value)
            for key in (service.get("x-sandbox") or {}).get("resolve_environment", []):
                url = urlsplit(env.get(key, ""))
                if not url.scheme or url.hostname not in self.document["services"]:
                    raise ValueError(f"Service {name!r}: environment {key!r} must be a URL naming a Compose service")
                # Validate the port before provisioning any sandboxes.
                _ = url.port
            health = {}
            if image_config.get("Healthcheck"):
                original = image_config["Healthcheck"]
                health = {"test": original["Test"]}
                for src, dest in (
                    ("Interval", "interval"),
                    ("Timeout", "timeout"),
                    ("StartPeriod", "start_period"),
                    ("StartInterval", "start_interval"),
                ):
                    if original.get(src):
                        health[dest] = original[src] / 1e9
                if original.get("Retries"):
                    health["retries"] = original["Retries"]
            health.update(service.get("healthcheck") or {})
            if set(health) - {"test", "disable", "interval", "timeout", "retries", "start_period", "start_interval"}:
                raise NotImplementedError(f"Service {name!r}: unsupported healthcheck options")
            test = health.get("test") or ["NONE"]
            if health.get("disable"):
                test = ["NONE"]
            if not isinstance(test, list) or test[0] not in {"CMD", "CMD-SHELL", "NONE"}:
                raise ValueError(f"Service {name!r}: invalid healthcheck test")
            if (test[0] == "CMD-SHELL" and len(test) != 2) or (test[0] == "CMD" and len(test) < 2):
                raise ValueError(f"Service {name!r}: healthcheck requires a command")
            for field in ("timeout", "interval", "start_interval", "start_period"):
                if field in health and _seconds(health[field]) < (0 if field == "start_period" else 1e-9):
                    raise ValueError(f"Service {name!r}: invalid healthcheck {field}")
            if not isinstance(health.get("retries", 3), int) or health.get("retries", 3) < 1:
                raise ValueError(f"Service {name!r}: healthcheck retries must be positive")
            ports = [int(str(port).removesuffix("/tcp")) for port in service.get("expose", [])]
            ports += [
                int(port.removesuffix("/tcp"))
                for port in image_config.get("ExposedPorts", {})
                if port.endswith("/tcp")
            ]
            ports += [int(port["target"]) for port in service.get("ports", [])]
            spec = self.service_specs.get(name, SandboxSpec(ttl_s=self.timeout_s + 3600))
            resources = dict(vars(spec.resources))
            if "cpus" in service:
                resources["cpu"] = float(service["cpus"])
            if "mem_limit" in service:
                resources["memory_mib"] = (int(service["mem_limit"]) + 1048575) // 1048576
            user = service.get("user") or None
            if isinstance(user, str) and ":" in user:
                raise NotImplementedError(f"Service {name!r}: user:group overrides are unsupported by the sandbox API")
            if isinstance(user, str) and user.isdecimal():
                user = int(user)
            runtime = f"/tmp/{self.project}/{name}"
            self._plans[name] = {
                "command": shlex.join(argv),
                "user": user,
                "health": {**health, "test": test},
                "shell": image_config.get("Shell") or ["/bin/sh", "-c"],
                "runtime": runtime,
                "spec": replace(
                    spec,
                    image=image,
                    env={**spec.env, **env},
                    workdir=service.get("working_dir", image_config.get("WorkingDir") or "/"),
                    resources=resources,
                    ports=tuple(dict.fromkeys(ports)),
                    entrypoint=["/bin/sh", "-c", "while :; do sleep 3600; done"],
                    metadata={**spec.metadata, **(service.get("labels") or {})},
                ),
            }
        for name, service in self.document["services"].items():
            if service.get("network_mode"):
                target = service["network_mode"].removeprefix("service:")
                if not self._plans[target]["spec"].ports:
                    raise ValueError(f"Service {target!r} must expose TCP ports for forwarding")
            for dep, options in (service.get("depends_on") or {}).items():
                if options.get("condition") == "service_healthy" and self._plans[dep]["health"]["test"][0] == "NONE":
                    raise ValueError(f"Service {name!r} requires {dep!r} healthy, but it has no healthcheck")

    async def _prepare_volumes(self):
        root = "/tmp/compose-volumes"
        metadata = self.provider.shared_volume_metadata()
        helper = AsyncSandbox(
            self.provider,
            SandboxSpec(
                image=self.volume_init_image,
                ttl_s=self.timeout_s + 3600,
                entrypoint=["sh", "-c", "while :; do sleep 3600; done"],
                metadata=metadata,
                provider_options=self.provider.shared_volume_options(None, root),
            ),
            owns_provider=False,
        )
        await helper.start()
        self._volume_helper = helper
        for name, service in self.document["services"].items():
            mounts = []
            for volume in service.get("volumes", []):
                source = volume["source"]
                managed = source not in self.volume_sources
                relative = self.volume_sources.get(source, f"{self.project}/{source}")
                if managed:
                    result = await self._volume_helper.exec(
                        f"mkdir -p {shlex.quote(root + '/' + relative)}", user="root"
                    )
                    if result.return_code:
                        raise RuntimeError(f"Shared volume initialization failed: {result.stderr}")
                    if not (volume.get("volume") or {}).get("nocopy"):
                        seed_spec = replace(
                            self._plans[name]["spec"],
                            metadata=metadata,
                            provider_options=self.provider.shared_volume_options(relative, root),
                        )
                        seed = AsyncSandbox(self.provider, seed_spec, owns_provider=False)
                        self._seeds.append(seed)
                        await seed.start()
                        try:
                            target = shlex.quote(volume["target"])
                            result = await seed.exec(
                                f'if test -n "$(ls -A {root})"; then :; '
                                f"elif test -d {target}; then cp -a {target}/. {root}/; "
                                f"elif test -e {target}; then exit 1; fi",
                                user="root",
                                timeout_s=300,
                            )
                            if result.return_code:
                                raise RuntimeError(f"Named volume copy-up failed: {result.stderr}")
                        finally:
                            await seed.stop()
                options = self.provider.shared_volume_options(
                    relative, volume["target"], read_only=volume.get("read_only", False)
                )
                mounts.extend(options["volumes"])
            if mounts:
                spec = self._plans[name]["spec"]
                self._plans[name]["spec"] = replace(
                    spec,
                    metadata={**metadata, **spec.metadata},
                    provider_options={
                        **spec.provider_options,
                        "volumes": [*spec.provider_options.get("volumes", []), *mounts],
                    },
                )

    async def _wait(self, name, condition):
        if condition == "service_started":
            return
        plan = self._plans[name]
        health = plan["health"]
        started = plan.get("started_at", asyncio.get_running_loop().time())
        failures = 0
        while True:
            done = await self.services[name].exec(
                f"if test -f {plan['runtime']}/exit; then cat {plan['runtime']}/exit; fi", user=None
            )
            if done.return_code:
                raise RuntimeError(f"Cannot read service {name!r} exit status: {done.stderr}")
            if done.stdout and done.stdout.strip():
                if condition == "service_completed_successfully" and done.stdout.strip() == "0":
                    return
                log = await self.services[name].exec(f"tail -c 4096 {plan['runtime']}/log")
                raise RuntimeError(f"Compose service {name!r} exited ({done.stdout.strip()}): {log.stdout}")
            if condition == "service_completed_successfully":
                await asyncio.sleep(self.poll_interval_s)
                continue
            test = health["test"]
            command = shlex.join(test[1:]) if test[0] == "CMD" else shlex.join([*plan["shell"], test[1]])
            try:
                result = await self.services[name].exec(
                    command,
                    timeout_s=_seconds(health.get("timeout", "30s")),
                    user=plan["user"],
                    env=plan.get("resolved_env"),
                )
                healthy = result.return_code == 0
                detail = result.stderr or result.stdout
            except TimeoutError:
                healthy, detail = False, "healthcheck timed out"
            if healthy:
                return
            elapsed = asyncio.get_running_loop().time() - started
            in_start_period = elapsed < _seconds(health.get("start_period", 0))
            if not in_start_period:
                failures += 1
                if failures >= health.get("retries", 3):
                    raise RuntimeError(f"Compose service {name!r} unhealthy: {detail}")
            interval = health.get("start_interval", "5s") if in_start_period else health.get("interval", "30s")
            await asyncio.sleep(_seconds(interval))

    async def start(self):
        if self._started or self._closed:
            raise RuntimeError("Compose collection already started or closed")
        if self.compose_file is None:
            raise ValueError("Starting a Compose collection requires a YAML file")
        self._started = True
        try:
            async with asyncio.timeout(self.timeout_s):
                self.document = await self._normalize()
                order = self._validate()
                await self._prepare()
                if any(service.get("volumes") for service in self.document["services"].values()):
                    await self._prepare_volumes()
                for name in order:
                    self.services[name] = AsyncSandbox(self.provider, self._plans[name]["spec"], owns_provider=False)
                    await self.services[name].start()
                    service = self.document["services"][name]
                    if service.get("cap_add") or service.get("shm_size") is not None:
                        await self.provider.configure_runtime(
                            self.services[name]._require_handle(),
                            cap_add=tuple(service.get("cap_add") or ()),
                            shm_size=service.get("shm_size"),
                        )
                if self.services:
                    hosts = {}
                    for name, sandbox in self.services.items():
                        address = await self.provider.network_address(sandbox._require_handle())
                        service = self.document["services"][name]
                        aliases = ((service.get("networks") or {}).get("default") or {}).get("aliases", [])
                        for alias in [name, *aliases]:
                            hosts[alias] = address
                    for name, sandbox in self.services.items():
                        options = self.document["services"][name].get("x-sandbox") or {}
                        plan = self._plans[name]
                        plan["resolved_env"] = {}
                        for key in options.get("resolve_environment", []):
                            url = urlsplit(plan["spec"].env[key])
                            address = hosts[url.hostname]
                            authority = f"[{address}]" if ":" in address else address
                            if url.port is not None:
                                authority += f":{url.port}"
                            if "@" in url.netloc:
                                authority = url.netloc.rsplit("@", 1)[0] + "@" + authority
                            plan["resolved_env"][key] = url._replace(netloc=authority).geturl()
                        if options.get("hosts") != []:
                            await self.provider.set_hosts(sandbox._require_handle(), hosts)
                for name in order:
                    service = self.document["services"][name]
                    for dep, options in (service.get("depends_on") or {}).items():
                        await self._wait(dep, options.get("condition", "service_started"))
                    plan = self._plans[name]
                    runtime = plan["runtime"]
                    result = await self.services[name].exec(f"mkdir -p {runtime} && chmod 777 {runtime}")
                    if result.return_code:
                        raise RuntimeError(f"Cannot initialize service {name!r}: {result.stderr or result.stdout}")
                    if service.get("network_mode"):
                        target = service["network_mode"].removeprefix("service:")
                        ports = self._plans[target]["spec"].ports
                        ready = runtime + "/forwarding-ready"
                        relay = asyncio.create_task(
                            self.provider.forward_ports(
                                self.services[name]._require_handle(), hosts[target], ports, ready_file=ready
                            )
                        )
                        self._processes[name + ":forwarding"] = relay
                        while (await self.services[name].exec(f"test -f {ready}")).return_code:
                            if relay.done():
                                await relay
                                raise RuntimeError(f"TCP forwarding for {name!r} exited before becoming ready")
                            await asyncio.sleep(self.poll_interval_s)
                    script = f"touch {runtime}/started\n{plan['command']}\nstatus=$?\nprintf '%s' \"$status\" > {runtime}/exit\n"
                    with tempfile.TemporaryDirectory() as temp:
                        path = Path(temp) / "run.sh"
                        path.write_text(script)
                        await self.services[name].upload(path, runtime + "/run.sh")
                    # Keep the exec request alive: some providers reap background
                    # children as soon as their launching shell exits.
                    process = asyncio.create_task(
                        self.services[name].exec(
                            f"sh {runtime}/run.sh >{runtime}/log 2>&1 </dev/null",
                            user=plan["user"],
                            env=plan.get("resolved_env"),
                            timeout_s=None,
                        )
                    )
                    self._processes[name] = process
                    while True:
                        finished = process.done()
                        result = await self.services[name].exec(f"test -f {runtime}/started")
                        if result.return_code == 0:
                            break
                        if finished:
                            outcome = process.result()
                            raise RuntimeError(f"Compose service {name!r} did not start: {outcome.stderr}")
                        await asyncio.sleep(self.poll_interval_s)
                    plan["started_at"] = asyncio.get_running_loop().time()
                for name in order:
                    if self._plans[name]["health"]["test"][0] != "NONE":
                        await self._wait(name, "service_healthy")
                for name, process in self._processes.items():
                    if name.endswith(":forwarding") and process.done():
                        await process
                        raise RuntimeError(f"TCP forwarding for {name!r} stopped during startup")
        except BaseException:
            await self.stop()
            raise
        self._ready = True
        return self

    async def serialize(self, *, scope: str | None = None) -> dict[str, Any]:
        """Describe a running collection using provider connection descriptors.

        The creating process must keep the collection alive: it owns service
        and forwarding tasks and managed-volume cleanup. Provider configuration
        and YAML are not included.
        """
        if not self._ready or self._closed:
            raise RuntimeError("Serializing requires a running Compose collection")
        return {
            "services": {name: await sandbox.serialize(scope=scope) for name, sandbox in self.services.items()},
        }

    @classmethod
    async def connect(cls, descriptor: Mapping[str, Any], *, provider) -> "AsyncSandboxCompose":
        """Connect to all members without provisioning or restarting services.

        Like AsyncSandbox.connect, stop() closes the connected sandboxes using
        provider semantics. The creator retains ownership of managed-volume
        cleanup and running service/forwarding tasks and must also call stop().
        """
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"services"}:
            raise ValueError("Invalid Compose connection descriptor")
        services = descriptor["services"]
        if (
            not isinstance(services, Mapping)
            or not services
            or any(not isinstance(name, str) or not isinstance(value, Mapping) for name, value in services.items())
        ):
            raise ValueError("Invalid Compose connection members")
        collection = cls(provider, None)
        try:
            for name, member in services.items():
                collection.services[name] = await AsyncSandbox.connect(
                    member, provider=collection.provider, owns_provider=False
                )
        except BaseException:
            # A partial attach must not destroy a collection owned by another server.
            await collection.provider.aclose()
            raise
        collection._started = collection._ready = True
        return collection

    async def stop(self):
        if self._closed:
            return
        if self._stop_task is None or self._stop_task.done():
            self._stop_task = asyncio.create_task(self._stop())
        try:
            await asyncio.shield(self._stop_task)
        except asyncio.CancelledError:
            await self._stop_task
            raise

    async def _stop(self):
        for process in self._processes.values():
            process.cancel()
        await asyncio.gather(*self._processes.values(), return_exceptions=True)
        errors = []
        for sandbox in [*reversed(list(self.services.values())), *self._seeds]:
            try:
                await sandbox.stop()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Compose cleanup failed", errors)
        if self._volume_helper:
            result = await self._volume_helper.exec(f"rm -rf /tmp/compose-volumes/{self.project}", user="root")
            if result.return_code:
                raise RuntimeError(f"Shared volume cleanup failed: {result.stderr}")
            await self._volume_helper.stop()
            self._volume_helper = None
        await self.provider.aclose()
        self._closed = True

    async def __aenter__(self):
        if self._ready and not self._closed:
            return self
        return await self.start()

    async def __aexit__(self, *exc):
        await self.stop()
