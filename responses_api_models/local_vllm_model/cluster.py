# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit allocation-owned Slurm/per-rank MP deployment; never allocates GPUs.

Run inside a dedicated allocation. The controller launches only its own srun
steps, renews a node-agent lease, and publishes a private Gym attach config.
The router binds to loopback by default; an allocated address is opt-in.
"""

import argparse
import asyncio
import ipaddress
import json
import os
import re
import signal
import socket
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from aiohttp import ClientError, ClientTimeout
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.server_utils import GlobalAIOHTTPAsyncClientConfig, request, set_global_aiohttp_client
from responses_api_models.local_vllm_model.pd_launcher import PDRoleConfig, reserve_ports
from responses_api_models.local_vllm_model.router_launcher import VLLMRouterConfig, VLLMRouterLauncher
from responses_api_models.local_vllm_model.subprocess_launcher import (
    OwnedProcess,
    kwargs_to_argv,
    normalize_kwargs,
    redacted_argv,
    validate_managed_kwargs,
)


class ClusterGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nodes: int = Field(ge=1, strict=True)
    gpus_per_node: int = Field(ge=1, le=8, strict=True)
    # Stay below this cluster's 9000..65000 ephemeral range. Each node also
    # checks its actual kernel range before starting any runtime processes.
    api_port: int = Field(default=8000, ge=1024, le=65535)
    rpc_port: int = Field(default=7000, ge=1024, le=65535)
    side_channel_port: int = Field(default=5600, ge=1024, le=65535)
    kv_lease_duration: int | None = Field(default=None, ge=1, strict=True)
    serve_kwargs: dict[str, Any] = Field(default_factory=dict)


class ClusterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    image: Path
    executable: str = "vllm"
    expected_vllm_version: str
    # Empty is an explicit opt-out for the legacy external benchmark profile.
    api_key: str = "dummy"  # pragma: allowlist secret
    groups: dict[Literal["combined", "prefill", "decode"], ClusterGroup]
    # Independent node-local TP replicas; never a cross-node DP collective.
    deployment_mode: Literal["native_dp", "replicas"] = "native_dp"
    serve_kwargs: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    router: VLLMRouterConfig
    cpus_per_node: int = Field(default=32, ge=1)
    startup_timeout_seconds: float = Field(default=1800, gt=0, allow_inf_nan=False)
    shutdown_timeout_seconds: float = Field(default=20, gt=0, allow_inf_nan=False)
    lease_seconds: float = Field(default=60, ge=15, allow_inf_nan=False)
    probe_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    container_mounts: list[str] = Field(default_factory=lambda: ["/lustre:/lustre", "/home:/home"])

    @model_validator(mode="after")
    def supported_layout(self) -> "ClusterConfig":
        if not self.api_key and self.router.profile != "external_benchmark":
            raise ValueError("Empty worker API keys require the explicit external_benchmark router profile")
        if set(self.groups) not in ({"combined"}, {"prefill", "decode"}):
            raise ValueError("Cluster groups must be combined, or both prefill and decode")
        if "combined" in self.groups and self.groups["combined"].kv_lease_duration is not None:
            raise ValueError("kv_lease_duration requires prefill/decode groups")
        if not self.image.is_absolute():
            raise ValueError("Use an absolute, shared container image path")
        if (
            "prefill" in self.groups
            and len({group.nodes * group.gpus_per_node for group in self.groups.values()}) != 1
        ):
            raise ValueError("Initial cluster PD requires symmetric P/D rank counts")
        self.serve_kwargs = normalize_kwargs(self.serve_kwargs)
        validate_managed_kwargs(self.serve_kwargs, self.env)
        tp = self.serve_kwargs.get("tensor_parallel_size", 1)
        if type(tp) is not int or tp < 1:
            raise ValueError("tensor_parallel_size must be a positive integer")
        if self.deployment_mode == "native_dp" and tp != 1:
            raise ValueError("Per-rank wide-EP launcher supports TP=1 and PP=1")
        if "data_parallel_size" in self.serve_kwargs:
            raise ValueError("Cluster DP size is derived from each group's allocated GPUs")
        if any(
            key in self.env
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "VLLM_NIXL_SIDE_CHANNEL_HOST",
                "VLLM_NIXL_SIDE_CHANNEL_PORT",
                "VLLM_PORT",
            )
        ):
            raise ValueError("Cluster placement owns GPU visibility and worker network placement")
        for group in self.groups.values():
            group.serve_kwargs = PDRoleConfig(serve_kwargs=group.serve_kwargs).serve_kwargs
            if group.gpus_per_node % tp:
                raise ValueError("Node-local TP replicas must exactly partition each node's GPUs")
            if (
                self.deployment_mode == "native_dp"
                and group.nodes * group.gpus_per_node > 1
                and not self.serve_kwargs.get("enable_expert_parallel")
            ):
                raise ValueError(
                    "Multi-rank cluster mode is restricted to native MoE wide-EP; dense replicas are not DP"
                )
            ports = [
                group.rpc_port,
                *range(group.api_port, group.api_port + group.gpus_per_node),
                *range(group.side_channel_port, group.side_channel_port + group.nodes * group.gpus_per_node),
            ]
            if max(ports) > 65535 or len(set(ports)) != len(ports):
                raise ValueError("Cluster port ranges must be distinct and within 1..65535")
        return self


def deployment_plan(config: ClusterConfig, hosts: list[tuple[str, str]], output: Path, job_id: str) -> dict:
    needed = sum(group.nodes for group in config.groups.values())
    if (
        len(hosts) != needed
        or len({name for name, _ in hosts}) != needed
        or len({address for _, address in hosts}) != needed
    ):
        raise ValueError("Provide exactly the distinct nodes/addresses required by the deployment")
    for name, address in hosts:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Invalid Slurm node name")
        parsed = ipaddress.IPv4Address(address)
        if parsed.is_unspecified or parsed.is_multicast:
            raise ValueError("Worker address must be a unicast IPv4 address")
    run_token, cursor, nodes, urls = uuid4().hex, 0, [], {}
    for role, group in config.groups.items():
        group_hosts = hosts[cursor : cursor + group.nodes]
        cursor += group.nodes
        tp = config.serve_kwargs.get("tensor_parallel_size", 1)
        ranks_per_node = group.gpus_per_node // tp
        replicas = config.deployment_mode == "replicas"
        dp_size = 1 if replicas else group.nodes * group.gpus_per_node
        urls[role] = []
        for node_index, (node, address) in enumerate(group_hosts):
            ranks = []
            for local_rank in range(ranks_per_node):
                rank = node_index * ranks_per_node + local_rank
                kwargs = dict(config.serve_kwargs | group.serve_kwargs)
                checkpoint = kwargs.pop("model", config.model)
                if not isinstance(checkpoint, str) or not checkpoint or checkpoint.startswith("-"):
                    raise ValueError("Model checkpoint must be a nonempty name/path, not a CLI option")
                names = kwargs.setdefault("served_model_name", [config.model])
                if config.model not in ([names] if isinstance(names, str) else names):
                    raise ValueError("served_model_name must include the requested model alias")
                kwargs.update(
                    host=address,
                    port=group.api_port + local_rank,
                    tensor_parallel_size=tp,
                    pipeline_parallel_size=1,
                    data_parallel_size=dp_size,
                    distributed_executor_backend="mp",
                    data_parallel_backend="mp",
                )
                if replicas:
                    kwargs["data_parallel_size_local"] = 1
                    kwargs["api_server_count"] = 1
                if dp_size > 1:
                    kwargs.update(
                        data_parallel_rank=rank,
                        data_parallel_size_local=1,
                        data_parallel_external_lb=True,
                        data_parallel_address=group_hosts[0][1],
                        data_parallel_rpc_port=group.rpc_port,
                    )
                env = {}
                if role != "combined":
                    kwargs["kv_transfer_config"] = {
                        "kv_connector": "NixlConnector",
                        "kv_role": ("kv_producer" if role == "prefill" else "kv_consumer") if replicas else "kv_both",
                        "kv_load_failure_policy": "fail",
                    }
                    if group.kv_lease_duration is not None:
                        kwargs["kv_transfer_config"]["kv_connector_extra_config"] = {
                            "kv_lease_duration": group.kv_lease_duration
                        }
                    env = {
                        "VLLM_NIXL_SIDE_CHANNEL_HOST": address,
                        # Independent engines all have DP rank zero. Give each
                        # node-local TP replica its own side-channel range.
                        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(
                            group.side_channel_port + (local_rank * tp if replicas else 0)
                        ),
                    }
                flags = {"--" + key.replace("_", "-") for key in kwargs} | {
                    "--no-" + key.replace("_", "-") for key in kwargs
                }
                argv = ["serve", checkpoint, *kwargs_to_argv(kwargs, flags)]
                ranks.append(
                    {
                        "rank": rank,
                        "local_rank": local_rank,
                        "gpu_indices": list(range(local_rank * tp, (local_rank + 1) * tp)),
                        "argv": argv,
                        "env": env,
                    }
                )
                urls[role].append(f"http://{address}:{group.api_port + local_rank}")
            ports = list(range(group.api_port, group.api_port + ranks_per_node))
            if role != "combined":
                ports += (
                    list(range(group.side_channel_port, group.side_channel_port + group.gpus_per_node))
                    if replicas
                    else [group.side_channel_port + rank["rank"] for rank in ranks]
                )
            if node_index == 0 and dp_size > 1:
                ports.append(group.rpc_port)
            required = {arg.split("=", 1)[0] for rank in ranks for arg in rank["argv"] if arg.startswith("--")}
            nodes.append(
                {
                    "job_id": job_id,
                    "run_token": run_token,
                    "node": node,
                    "address": address,
                    "role": role,
                    "ranks": ranks,
                    "gpus_per_node": group.gpus_per_node,
                    "reserved_ports": ports,
                    "required_flags": sorted(required),
                    "expected_vllm_version": config.expected_vllm_version,
                    "executable": config.executable,
                    "api_key": config.api_key,
                    # Longer than the pinned router's 50s/90s idle pools;
                    # explicit cluster env remains authoritative.
                    "env": {"VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "120", **config.env},
                    "heartbeat": str(output / "heartbeat.json"),
                    "lease_seconds": config.lease_seconds,
                    "probe_timeout": config.probe_timeout_seconds,
                    "shutdown_timeout": config.shutdown_timeout_seconds,
                }
            )
    return {"job_id": job_id, "run_token": run_token, "nodes": nodes, "urls": urls}


def write_private(path: Path, data: Any) -> None:
    path.touch(mode=0o600, exist_ok=False)
    path.write_text(json.dumps(data, indent=2) + "\n")


async def allocation_hosts() -> tuple[str, list[tuple[str, str]]]:
    job_id, node_list = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_JOB_NODELIST")
    if not job_id or not job_id.isdigit() or not node_list:
        raise ValueError("Run inside a dedicated Slurm allocation; this CLI never submits or attaches jobs")
    proc = await asyncio.create_subprocess_exec(
        "scontrol", "show", "hostnames", node_list, stdout=asyncio.subprocess.PIPE
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), 10)
    except BaseException:
        proc.kill()
        await proc.wait()
        raise
    if proc.returncode:
        raise RuntimeError("Cannot resolve the current allocation's nodes")
    names = stdout.decode().splitlines()
    return job_id, [(name, socket.gethostbyname(name)) for name in names]


async def launch_cluster(
    config: ClusterConfig,
    output: Path,
    *,
    dry_run: bool = False,
    hosts: list[tuple[str, str]] | None = None,
    stop_event: asyncio.Event | None = None,
) -> dict[str, Any]:
    if hosts is not None and not dry_run:
        raise ValueError("Live nodes must come from the current Slurm allocation")
    job_id, hosts = ("dry-run", hosts) if hosts is not None else await allocation_hosts()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = deployment_plan(config, hosts, output, job_id)
    # Node specs contain credentials; the public manifest contains neither env nor keys.
    for node in plan["nodes"]:
        write_private(output / f"{node['node']}.private.json", node)
    manifest = {
        "job_id": job_id,
        "run_token": plan["run_token"],
        "dry_run": dry_run,
        "urls": plan["urls"],
        "expected_vllm_version": config.expected_vllm_version,
        "container_image": str(config.image),
        "deployment_mode": config.deployment_mode,
        "ranks": [
            {
                "node": node["node"],
                "role": node["role"],
                "rank": rank["rank"],
                "gpu_indices": rank["gpu_indices"],
                "argv_redacted": redacted_argv(rank["argv"]),
            }
            for node in plan["nodes"]
            for rank in node["ranks"]
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if dry_run:
        return manifest
    owners, heartbeat_task, router = [], None, None
    completed = False
    (output / "state.json").write_text(json.dumps({"state": "starting", "run_token": plan["run_token"]}))
    stop = stop_event or asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async def heartbeat():
        while True:
            temporary = output / "heartbeat.pending"
            temporary.write_text(json.dumps({"run_token": plan["run_token"], "timestamp": time.time()}))
            temporary.replace(output / "heartbeat.json")
            await asyncio.sleep(2)

    try:
        heartbeat_task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        for node in plan["nodes"]:
            owner = OwnedProcess(config.shutdown_timeout_seconds + 15)
            owners.append(owner)
            argv = [
                "srun",
                "--jobid",
                job_id,
                "--nodes=1",
                "--ntasks=1",
                "--exact",
                "--exclusive",
                "--kill-on-bad-exit=1",
                "--cpu-bind=none",
                f"--nodelist={node['node']}",
                f"--gres=gpu:{node['gpus_per_node']}",
                f"--cpus-per-task={config.cpus_per_node}",
                f"--container-image={config.image}",
                f"--container-mounts={','.join(config.container_mounts)}",
                "--no-container-mount-home",
                "--container-remap-root",
                "--no-container-entrypoint",
                "python3",
                str(Path(__file__).with_name("cluster_node.py")),
                "--spec",
                str(output / f"{node['node']}.private.json"),
                "--output",
                str(output / node["node"]),
            ]
            await owner.start(argv, os.environ.copy(), output / f"{node['node']}.srun.log")
        pending = {url for urls in plan["urls"].values() for url in urls}
        async with asyncio.timeout(config.startup_timeout_seconds):
            while pending:
                if (
                    stop.is_set()
                    or heartbeat_task.done()
                    or any(owner.process.returncode is not None for owner in owners)
                ):
                    raise RuntimeError("Cluster startup interrupted or a node agent failed")
                owned_nodes = set()
                for node in plan["nodes"]:
                    try:
                        started = json.loads((output / node["node"] / "started.json").read_text())
                    except (OSError, ValueError):
                        continue
                    if (
                        started.get("job_id") != job_id
                        or started.get("run_token") != plan["run_token"]
                        or started.get("vllm_version") != config.expected_vllm_version
                    ):
                        raise RuntimeError("Node startup identity does not belong to this deployment")
                    owned_nodes.add(node["address"])
                for url in tuple(pending):
                    if url.split(":")[1].removeprefix("//") not in owned_nodes:
                        continue
                    try:
                        response = await request(
                            "GET",
                            url + "/v1/models",
                            _retry=False,
                            headers={"Authorization": f"Bearer {config.api_key}"},
                            timeout=ClientTimeout(total=1),
                        )
                        async with response:
                            if response.status == 200 and any(
                                item.get("id") == config.model for item in (await response.json()).get("data", [])
                            ):
                                pending.remove(url)
                    except (ClientError, TimeoutError, ValueError):
                        pass
                await asyncio.sleep(0.2)
        router = VLLMRouterLauncher(
            config=config.router.model_copy(update={"log_dir": output / "router"}),
            model=config.model,
            api_key=config.api_key,
            allocated_worker_hosts={address for _, address in hosts},
        )
        with ExitStack() as reservation:
            port = reserve_ports(reservation, first=config.router.port)
        options = {"worker_urls": plan["urls"].get("combined", []), "dp_size": 1}
        if "prefill" in plan["urls"]:
            options.update(prefill_urls=plan["urls"]["prefill"], decode_urls=plan["urls"]["decode"])
        base_url = await router.start(port, **options)
        write_private(
            output / "gym-connection.private.json",
            {
                "base_url": base_url,
                "api_key": config.api_key,
                "model": config.model,
                "routing_authority": "vllm_router",
                "routing_timeout_seconds": config.router.inference_timeout_seconds,
            },
        )
        (output / "state.json").write_text(
            json.dumps({"state": "ready", "run_token": plan["run_token"], "job_id": job_id})
        )
        print(f"Cluster ready: {output / 'gym-connection.private.json'}", flush=True)
        while not stop.is_set():
            if (
                any(owner.process.returncode is not None for owner in owners)
                or router.owner.process.returncode is not None
                or heartbeat_task.done()
            ):
                raise RuntimeError("Owned cluster component exited; stopping the entire deployment")
            await asyncio.sleep(0.2)
        completed = True
        return manifest
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        try:
            if router:
                await router.stop()
        finally:
            results = await asyncio.gather(*(owner.stop() for owner in owners), return_exceptions=True)
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)
            if any(isinstance(result, BaseException) for result in results):
                (output / "state.json").write_text(
                    json.dumps({"state": "cleanup_failed", "run_token": plan["run_token"]})
                )
                raise RuntimeError("Cluster cleanup failed; inspect only the owned Slurm steps")
            (output / "state.json").write_text(
                json.dumps({"state": "stopped" if completed else "failed", "run_token": plan["run_token"]})
            )


async def main_async(args: argparse.Namespace) -> None:
    resolved = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.override))
    config = ClusterConfig.model_validate(OmegaConf.to_container(resolved, resolve=True))
    hosts = [tuple(item.split("=", 1)) for item in args.node] if args.node else None
    client = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    try:
        await launch_cluster(config, args.output, dry_run=args.dry_run, hosts=hosts)
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New shared-filesystem run directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--node", action="append", help="Dry-run only: explicit hostname=IPv4")
    parser.add_argument("--override", action="append", default=[], help="Typed dotted config override: key=value")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
