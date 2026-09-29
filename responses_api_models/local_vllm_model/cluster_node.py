# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only Slurm node agent. Never imports Gym, torch, NIXL or vLLM.

The allocation owner creates a private spec and renews its heartbeat. Each rank
has a lifetime-pipe supervisor, including when this agent dies unexpectedly.
"""

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path


def validate_listener_ports(ports: list[int], ephemeral_range: tuple[int, int]) -> None:
    low, high = ephemeral_range
    if not 1 <= low <= high <= 65535:
        raise ValueError("Invalid kernel ephemeral TCP port range")
    conflicts = sorted(port for port in ports if low <= port <= high)
    if conflicts:
        raise ValueError(
            f"Configured listener ports {conflicts} overlap kernel ephemeral range {low}..{high}; "
            "choose API/RPC/NIXL ports outside that range. An initial free-port check cannot "
            "prevent vLLM's later internal connections from claiming ephemeral ports."
        )


def owned_command(
    argv: list[str], env: dict[str, str], log_path: Path, shutdown_timeout: float
) -> subprocess.Popen[bytes]:
    with log_path.open("wb") as log:
        return subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("process_supervisor.py")),
                "--shutdown-timeout",
                str(shutdown_timeout),
                "--",
                *argv,
            ],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            close_fds=True,
        )


def run_node(spec: dict, output: Path) -> int:
    if os.environ.get("SLURM_JOB_ID") != str(spec["job_id"]):
        raise ValueError("Node agent is outside the owning Slurm allocation")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != spec["gpus_per_node"] or not all(devices) or len(set(devices)) != len(devices):
        raise ValueError("Node GPU visibility does not match the exact scheduler-granted rank count")
    placements = [rank["gpu_indices"] for rank in spec["ranks"]]
    indices = [index for placement in placements for index in placement]
    if (
        not all(placements)
        or any(type(index) is not int for index in indices)
        or sorted(indices) != list(range(len(devices)))
        or any("CUDA_VISIBLE_DEVICES" in rank["env"] for rank in spec["ranks"])
    ):
        raise ValueError("Rank GPU placements must exactly partition scheduler-granted devices")
    ephemeral_range = tuple(map(int, Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()))
    validate_listener_ports(spec["reserved_ports"], ephemeral_range)
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    output.mkdir(parents=True, exist_ok=False)
    (output / "network-ports.json").write_text(
        json.dumps({"ephemeral_range": ephemeral_range, "listener_ports": spec["reserved_ports"]}) + "\n"
    )
    environment = os.environ | spec["env"]
    if spec["api_key"]:
        environment["VLLM_API_KEY"] = spec["api_key"]
    else:
        # An explicit empty key matches external serving on a trusted allocation.
        environment.pop("VLLM_API_KEY", None)
    if "VLLM_PORT" in environment:
        raise ValueError("Unset VLLM_PORT: its internal allocator can collide with managed listener ports")
    executable = shutil.which(spec["executable"], path=environment.get("PATH"))
    if not executable:
        raise ValueError("Selected vLLM executable is not installed in the container")
    for label, args in (("version", ["--version"]), ("help", ["serve", "--help=all"])):
        log_path = output / f"{label}.log"
        probe = owned_command([executable, *args], environment, log_path, spec["shutdown_timeout"])
        try:
            probe.wait(timeout=spec["probe_timeout"])
        finally:
            probe.stdin.close()
            probe.wait(timeout=spec["shutdown_timeout"] + 10)
        text = log_path.read_text(errors="replace")
        if probe.returncode:
            raise RuntimeError(f"vLLM {label} probe failed; inspect node log")
        if label == "version":
            versions = re.findall(r"^(?:vllm )?(\d+\.\d+\.\d+[\w.+-]*)\s*$", text, re.MULTILINE)
            if versions != [spec["expected_vllm_version"]]:
                raise ValueError("Node runtime version does not match the pinned cluster runtime")
        else:
            flags = set(re.findall(r"--[a-z][a-z0-9-]*", text))
            missing = set(spec["required_flags"]) - flags
            if missing:
                raise ValueError(f"Node vLLM does not advertise required flags: {sorted(missing)}")
    # Fail on occupied API/side-channel ports, never attach to a prior deployment.
    with ExitStack() as reservations:
        for port in spec["reserved_ports"]:
            sock = reservations.enter_context(socket.socket())
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((spec["address"], port))
            sock.listen()
    owners = []
    status = 0
    try:
        for rank in spec["ranks"]:
            env = (
                environment
                | rank["env"]
                | {"CUDA_VISIBLE_DEVICES": ",".join(devices[index] for index in rank["gpu_indices"])}
            )
            log_path = output / f"rank-{rank['rank']}.log"
            owner = owned_command([executable, *rank["argv"]], env, log_path, spec["shutdown_timeout"])
            owners.append(owner)
        (output / "started.json").write_text(
            json.dumps(
                {
                    "job_id": spec["job_id"],
                    "run_token": spec["run_token"],
                    "node": spec["node"],
                    "node_agent_pid": os.getpid(),
                    "vllm_version": spec["expected_vllm_version"],
                    "http_timeout_keep_alive": environment.get("VLLM_HTTP_TIMEOUT_KEEP_ALIVE"),
                    "supervisor_pids": [owner.pid for owner in owners],
                    "cuda_visible_devices": devices,
                },
                indent=2,
            )
            + "\n"
        )
        while not stopping:
            if any(owner.poll() is not None for owner in owners):
                status = 1
                break
            try:
                heartbeat = json.loads(Path(spec["heartbeat"]).read_text())
                if (
                    heartbeat["run_token"] != spec["run_token"]
                    or time.time() - heartbeat["timestamp"] > spec["lease_seconds"]
                ):
                    status = 1
                    break
            except (OSError, ValueError, KeyError):
                status = 1
                break
            time.sleep(0.2)
    finally:
        for owner in owners:
            owner.stdin.close()
        deadline = time.monotonic() + spec["shutdown_timeout"] + 10
        for owner in owners:
            owner.wait(timeout=max(0.1, deadline - time.monotonic()))
        (output / "stopped.json").write_text(
            json.dumps({"status": status, "returncodes": [owner.returncode for owner in owners]}) + "\n"
        )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run_node(json.loads(args.spec.read_text()), args.output)


if __name__ == "__main__":
    raise SystemExit(main())
