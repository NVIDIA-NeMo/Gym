"""Configuration models for the Gym OSWorld Resources Server."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from nemo_gym.base_resources_server import BaseResourcesServerConfig


class RemoteDockerWorkerConfig(BaseModel):
    """One SSH-controlled Docker/QEMU worker in the execution pool."""

    name: str
    remote_host: str = ""
    data_host: str
    control_url: str = ""
    publish_host: str = "0.0.0.0"
    ssh_key: Optional[str] = None
    ssh_port: int = Field(default=22, ge=1, le=65535)
    capacity: int = Field(default=1, ge=1)
    image: str = "happysixd/osworld-docker:latest"
    assets_dir: str = "~/osworld-assets"
    vm_filename: str = "Ubuntu.qcow2"
    kvm: bool = True
    ram_size: str = "4G"
    cpu_cores: str = "4"
    disk_size: str = "32G"
    network: str = ""
    docker_extra_args: List[str] = Field(default_factory=list)
    transport: Literal["direct_http", "http_control"] = "direct_http"

    @model_validator(mode="after")
    def validate_direct_http(self) -> "RemoteDockerWorkerConfig":
        if self.transport == "direct_http" and not self.remote_host.strip():
            raise ValueError("remote_host must not be empty")
        if self.transport == "http_control" and not self.control_url.startswith(("http://", "https://")):
            raise ValueError("http_control requires an HTTP control_url")
        if not self.data_host.strip():
            raise ValueError("data_host must not be empty for direct_http")
        if not self.publish_host.strip():
            raise ValueError("publish_host must not be empty for direct_http")
        return self

    def provider_options(
        self,
        *,
        session_id: str,
        deployment_id: str,
        control_token: str = "",
    ) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "remote_host": self.remote_host,
            "control_url": self.control_url,
            "control_token": control_token,
            "ssh_port": self.ssh_port,
            "data_host": self.data_host,
            "publish_host": self.publish_host,
            "transport": self.transport,
            "dynamic_ports": True,
            "session_id": session_id,
            "deployment_id": deployment_id,
            "image": self.image,
            "assets_dir": self.assets_dir,
            "vm_filename": self.vm_filename,
            "kvm": self.kvm,
            "ram_size": self.ram_size,
            "cpu_cores": self.cpu_cores,
            "disk_size": self.disk_size,
            "network": self.network,
            "docker_extra_args": self.docker_extra_args,
        }
        if self.ssh_key:
            options["ssh_key"] = os.path.abspath(os.path.expanduser(self.ssh_key))
        return options


class OSWorldResourcesServerConfig(BaseResourcesServerConfig):
    """Gym server config plus remote OSWorld worker-pool settings."""

    deployment_id: str = "osworld-decoupled"
    auth_token_env: str = "OSWORLD_RESOURCES_TOKEN"
    require_auth: bool = True
    cache_dir: str = "cache/remote-osworld"
    state_dir: str = "state/remote-osworld"
    enable_proxy: bool = False
    proxy_config_file: Optional[str] = None
    session_ttl_seconds: int = Field(default=10800, ge=60)
    reaper_interval_seconds: int = Field(default=60, ge=5)
    cleanup_orphans_on_start: bool = True
    discover_workers: bool = True
    registration_token_env: str = "NEMO_GYM_REGISTRATION_TOKEN"
    workers: List[RemoteDockerWorkerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_resources_server(self) -> "OSWorldResourcesServerConfig":
        if not self.workers and not self.discover_workers:
            raise ValueError("at least one remote Docker worker is required")
        names = [worker.name for worker in self.workers]
        if len(names) != len(set(names)):
            raise ValueError("worker names must be unique")
        if self.enable_proxy and not self.proxy_config_file:
            raise ValueError("proxy_config_file is required when enable_proxy=true")
        if self.num_workers not in {None, 1}:
            raise ValueError(
                "OSWorld Resources Server keeps live DesktopEnv objects in process; "
                "num_workers must be 1"
            )
        return self

    @property
    def worker_capacity(self) -> int:
        return sum(worker.capacity for worker in self.workers)

    def auth_token(self) -> str:
        return os.environ.get(self.auth_token_env, "").strip()

    def registration_token(self) -> str:
        return os.environ.get(self.registration_token_env, "").strip()

    def resolved_cache_dir(self) -> str:
        return os.path.abspath(os.path.expanduser(self.cache_dir))

    def resolved_state_dir(self) -> str:
        return os.path.abspath(os.path.expanduser(self.state_dir))
