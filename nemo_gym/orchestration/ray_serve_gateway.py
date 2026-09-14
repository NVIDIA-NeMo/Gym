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

"""Ray Serve gateway that launches multiple vLLM instances and routes requests across them.

Selected automatically (see `api.effective_ray_serve`) whenever an instance's TP/PP footprint
would need to span multiple Slurm nodes, or via `use_ray_serve: true`. Joins the Ray cluster
already bootstrapped by the sbatch script and defines one Ray Serve deployment with
`number_of_instances` replicas; each replica launches its own `vllm serve
--distributed-executor-backend ray` subprocess and proxies requests to it. Ray Serve's own
`max_replicas_per_node` and HTTP proxy handle node placement and routing, replacing what used to
be hand-rolled here. Deliberately not using `ray.serve.llm`: it conflicts with vLLM's own
RayDistributedExecutor over nested placement groups (ray-project/ray#59064).
"""

import argparse
import logging
import os
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request

import aiohttp
import ray
from fastapi import FastAPI, Request, Response
from ray import serve


logger = logging.getLogger(__name__)

HEALTH_PATH = "/health"
HEALTH_POLL_INTERVAL_S = 5.0
HEALTH_TIMEOUT_S = 900.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True, help="Port the gateway itself listens on.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--number-of-instances", type=int, default=1)
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        help="Used to compute max_replicas_per_node; without it Serve packs replicas freely.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--extra-args", default="", help="Raw extra flags appended verbatim to `vllm serve`.")
    return parser.parse_args(argv)


def free_local_port() -> int:
    """An OS-assigned free TCP port on this node, since colocated replicas can't share a fixed one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def max_replicas_per_node(
    tensor_parallel_size: int, pipeline_parallel_size: int, gpus_per_node: int | None
) -> int | None:
    """How many instance drivers may share one physical node's GPU capacity; None if unknown."""
    if not gpus_per_node:
        return None
    tp_pp = tensor_parallel_size * pipeline_parallel_size
    return max(1, gpus_per_node // tp_pp)


def build_instance_command(
    model: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    trust_remote_code: bool,
    port: int,
    served_model_name: str | None = None,
    extra_args: str = "",
) -> list[str]:
    """The `vllm serve` command each replica runs for its own instance."""
    cmd = [
        "vllm",
        "serve",
        model,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--distributed-executor-backend",
        "ray",
    ]
    if served_model_name:
        cmd += ["--served-model-name", served_model_name]
    if pipeline_parallel_size > 1:
        cmd += ["--pipeline-parallel-size", str(pipeline_parallel_size)]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if extra_args:
        cmd += shlex.split(extra_args)
    return cmd


app = FastAPI()


@serve.deployment
@serve.ingress(app)
class VLLMInstance:
    """One Ray Serve replica = one vLLM instance, proxying every request to its own subprocess."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        trust_remote_code: bool,
        served_model_name: str | None = None,
        extra_args: str = "",
    ) -> None:
        port = free_local_port()
        self._base_url = f"http://localhost:{port}"
        cmd = build_instance_command(
            model, tensor_parallel_size, pipeline_parallel_size, trust_remote_code, port, served_model_name, extra_args
        )
        # RAY_ADDRESS makes vLLM's own Ray executor join this cluster instead of starting its own.
        env = {**os.environ, "RAY_ADDRESS": ray.get_runtime_context().gcs_address}
        self._proc = subprocess.Popen(cmd, env=env)
        self._session = aiohttp.ClientSession()
        self._wait_until_healthy()

    def _wait_until_healthy(self) -> None:
        # Blocking is deliberate: Serve won't route traffic to a replica until __init__ returns.
        deadline = time.monotonic() + HEALTH_TIMEOUT_S
        while True:
            if self._proc.poll() is not None:
                raise RuntimeError(f"vLLM instance exited early with code {self._proc.returncode}")
            try:
                with urllib.request.urlopen(f"{self._base_url}{HEALTH_PATH}", timeout=5) as resp:
                    if resp.status == 200:
                        logger.info("vLLM instance (%s) is healthy.", self._base_url)
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"vLLM instance ({self._base_url}) did not become healthy in time")
            time.sleep(HEALTH_POLL_INTERVAL_S)

    def check_health(self) -> None:
        # Raising here marks this replica unhealthy if the vLLM subprocess has died.
        if self._proc.poll() is not None:
            raise RuntimeError(f"vLLM instance ({self._base_url}) exited with code {self._proc.returncode}")

    @app.get(HEALTH_PATH)
    async def health(self) -> Response:
        return Response(status_code=200)

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(self, request: Request, path: str) -> Response:
        body = await request.body()
        forward_headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        async with self._session.request(
            request.method,
            f"{self._base_url}/{path}",
            params=request.query_params,
            data=body,
            headers=forward_headers,
        ) as resp:
            content = await resp.read()
            response_headers = {k: v for k, v in resp.headers.items() if k.lower() != "content-length"}
            return Response(content=content, status_code=resp.status, headers=response_headers)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Covers the single-node path; the multi-node path exports this before `ray start` runs.
    os.environ.setdefault("RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S", "1.0")
    try:
        ray.init(address="auto")
    except ConnectionError:
        # No existing cluster to join - start a local one.
        ray.init()

    deployment = VLLMInstance.options(
        num_replicas=args.number_of_instances,
        max_replicas_per_node=max_replicas_per_node(
            args.tensor_parallel_size, args.pipeline_parallel_size, args.gpus_per_node
        ),
    ).bind(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=args.trust_remote_code,
        served_model_name=args.served_model_name,
        extra_args=args.extra_args,
    )
    serve.start(http_options={"host": "0.0.0.0", "port": args.port})
    serve.run(deployment)

    logger.info(
        "Ray Serve gateway ready on port %d, routing across %d instance(s).", args.port, args.number_of_instances
    )
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
