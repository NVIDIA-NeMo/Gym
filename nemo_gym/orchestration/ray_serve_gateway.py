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

Selected automatically by `nemo_gym.orchestration` (see `api.effective_ray_serve` and
`executors/slurm_script._build_vllm_ray_serve_command`) whenever a vLLM service's
tensor/pipeline-parallel footprint would require an individual data-parallel instance to itself
span multiple Slurm nodes - something vLLM's own multi-node data-parallel mechanism can't express -
or whenever a user opts in via `use_ray_serve: true`.

This process joins the (possibly multi-node) Ray cluster already bootstrapped by the sbatch script,
launches `number_of_instances` independent `vllm serve` subprocesses (each using vLLM's own Ray
core executor, `--distributed-executor-backend ray` - the same proven mechanism used for a single
instance spanning nodes), waits for each to become healthy, and then runs a Ray Serve HTTP ingress
that round-robins incoming requests across them. Ray's own placement-group scheduler decides where
each instance's tensor/pipeline-parallel GPU footprint lands, spanning nodes automatically when an
instance's own footprint requires it - this module never touches vLLM engine internals directly.
"""

import argparse
import asyncio
import itertools
import logging
import os
import time

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
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def instance_port(gateway_port: int, instance_index: int) -> int:
    """Each backing vLLM instance listens on its own port, offset above the gateway's own port so
    it never collides with the gateway's listening socket."""
    return gateway_port + 1 + instance_index


def build_instance_command(args: argparse.Namespace, instance_index: int) -> list[str]:
    """Same flags as a single-instance-multi-node `vllm serve` invocation
    (see `_build_vllm_single_instance_multi_node_command`), just run once per instance."""
    cmd = [
        "vllm",
        "serve",
        args.model,
        "--port",
        str(instance_port(args.port, instance_index)),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--distributed-executor-backend",
        "ray",
    ]
    if args.pipeline_parallel_size > 1:
        cmd += ["--pipeline-parallel-size", str(args.pipeline_parallel_size)]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


class RoundRobinRouter:
    """Cycles through backend instance URLs in order. Not load-aware - just spreads requests
    evenly, mirroring what vLLM's own --data-parallel-size router would have done."""

    def __init__(self, urls: list[str]):
        if not urls:
            raise ValueError("RoundRobinRouter requires at least one backend URL")
        self._urls = list(urls)
        self._cycle = itertools.cycle(self._urls)

    def next_url(self) -> str:
        return next(self._cycle)


async def _wait_until_healthy(
    session: aiohttp.ClientSession, url: str, proc: asyncio.subprocess.Process, instance_index: int, deadline: float
) -> None:
    while True:
        if proc.returncode is not None:
            raise RuntimeError(f"vLLM instance {instance_index} ({url}) exited early with code {proc.returncode}")
        try:
            async with session.get(f"{url}{HEALTH_PATH}", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    logger.info("vLLM instance %d (%s) is healthy.", instance_index, url)
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"vLLM instance {instance_index} ({url}) did not become healthy in time")
        await asyncio.sleep(HEALTH_POLL_INTERVAL_S)


async def launch_instances_and_wait(args: argparse.Namespace, gcs_address: str) -> list[str]:
    """Launch every vLLM instance subprocess concurrently and block until all are healthy.

    Each `vllm serve` subprocess does its own internal `ray.init()` (inside its Ray distributed
    executor) - without RAY_ADDRESS in its environment it can't discover the cluster this gateway
    process already joined, and silently starts a separate, single-machine local Ray cluster of its
    own instead. That defeats the whole point (Ray's placement-group scheduler no longer sees the
    other instances, so multiple instances contend for the same GPUs). Passing RAY_ADDRESS
    explicitly is what makes every instance join the one shared cluster.

    Returns the list of ready instance base URLs, in instance order.
    """
    env = {**os.environ, "RAY_ADDRESS": gcs_address}
    commands = [build_instance_command(args, i) for i in range(args.number_of_instances)]
    procs = [await asyncio.create_subprocess_exec(*cmd, env=env) for cmd in commands]
    urls = [f"http://localhost:{instance_port(args.port, i)}" for i in range(args.number_of_instances)]

    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            *(_wait_until_healthy(session, url, proc, i, deadline) for i, (url, proc) in enumerate(zip(urls, procs)))
        )
    return urls


app = FastAPI()


@serve.deployment
@serve.ingress(app)
class VLLMGateway:
    """Thin Ray Serve HTTP ingress that forwards every request to one of the ready vLLM instances.

    Ray Serve owns both instance creation (launch_instances_and_wait, called before serve.run) and
    request routing (this class) - vLLM's own data-parallel mechanism is not used at all.
    """

    def __init__(self, instance_urls: list[str]):
        self._router = RoundRobinRouter(instance_urls)
        self._session = aiohttp.ClientSession()

    @app.get(HEALTH_PATH)
    async def health(self) -> Response:
        # By the time this deployment is serving traffic, every backing instance already passed
        # its own health check in launch_instances_and_wait - nothing further to aggregate.
        return Response(status_code=200)

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(self, request: Request, path: str) -> Response:
        target = self._router.next_url()
        body = await request.body()
        forward_headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        async with self._session.request(
            request.method,
            f"{target}/{path}",
            params=request.query_params,
            data=body,
            headers=forward_headers,
        ) as resp:
            content = await resp.read()
            response_headers = {k: v for k, v in resp.headers.items() if k.lower() != "content-length"}
            return Response(content=content, status_code=resp.status, headers=response_headers)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        ray.init(address="auto")
    except ConnectionError:
        # No existing cluster to join (e.g. the single-node opt-in case, where the sbatch script
        # skips the multi-node Ray bootstrap entirely) - start a local one.
        ray.init()
    gcs_address = ray.get_runtime_context().gcs_address

    instance_urls = asyncio.run(launch_instances_and_wait(args, gcs_address))

    serve.start(http_options={"host": "0.0.0.0", "port": args.port})
    serve.run(VLLMGateway.bind(instance_urls))

    logger.info("Ray Serve gateway ready on port %d, routing across %d instance(s).", args.port, len(instance_urls))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
