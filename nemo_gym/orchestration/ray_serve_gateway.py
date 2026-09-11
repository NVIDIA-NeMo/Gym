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
then defines ONE Ray Serve deployment with `number_of_instances` replicas: each replica IS one
vLLM instance (it launches its own `vllm serve --distributed-executor-backend ray` subprocess in
`__init__` and proxies incoming HTTP requests to it locally). Ray Serve itself owns both concerns
that used to be hand-rolled here:
  - Spreading instances across nodes: `max_replicas_per_node` (computed from `--gpus-per-node` and
    each instance's own TP*PP footprint, see `max_replicas_per_node()`) caps how many instance
    drivers may share one node's GPU capacity - otherwise every instance's driver could land on
    this process's own node, and vLLM refuses to start once that node's local GPU share is
    exhausted by an earlier instance, even though other nodes in the cluster are completely free.
    Ray's own placement-group scheduler (triggered inside vLLM's own Ray executor, not by us) then
    decides where each instance's *worker* ranks land, spanning nodes automatically when an
    instance's own footprint requires it - Serve's job is only to place the N driver processes.
  - Routing requests across instances: Ray Serve's built-in HTTP proxy load-balances across a
    deployment's replicas natively once `num_replicas > 1` - no custom round-robin code needed.

Deliberately NOT using `ray.serve.llm` (Ray's own higher-level declarative vLLM integration):
its `placement_group_config` pre-reserves GPUs in an outer placement group, which conflicts with
vLLM v1's own `RayDistributedExecutor` trying to create a nested placement group for the same GPUs
(see https://github.com/ray-project/ray/issues/59064, closed as not planned - the documented
workaround is exactly what this module does: run vLLM as an independent process, not through
ray.serve.llm's engine wrapper). Each replica here claims num_gpus=0 for itself and lets vLLM's own
Ray executor claim GPUs, avoiding that conflict entirely.
"""

import argparse
import logging
import os
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
        help="Used only to decide how many instance drivers may share one physical node (see "
        "max_replicas_per_node) - not required, but without it Serve is left free to pack "
        "replicas onto nodes without regard for GPU capacity.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def free_local_port() -> int:
    """An OS-assigned free TCP port on this node. Replicas can't share a fixed local port for
    their own vLLM subprocess (e.g. gateway_port + 1) - `max_replicas_per_node` deliberately allows
    multiple replicas to colocate on one node whenever their combined GPU footprint fits, so each
    replica must pick its own port at startup instead."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def max_replicas_per_node(
    tensor_parallel_size: int, pipeline_parallel_size: int, gpus_per_node: int | None
) -> int | None:
    """How many instance replicas (drivers) may share one physical node, passed straight through
    to `@serve.deployment`'s own `max_replicas_per_node` option - the one piece of scheduling
    Ray Serve can't infer on its own without knowing GPU capacity, since each replica claims
    num_gpus=0 for itself (see VLLMInstance's docstring).

    None (Serve's own unconstrained default) when there's no gpus_per_node info to reason with.
    Otherwise: if one instance's own TP*PP footprint fits within a node, more than one instance may
    share that node - up to as many as actually fit (e.g. TP2 x 4 instances comfortably share one
    8-GPU node). If an instance's own footprint exceeds a single node's GPU count (it must itself
    span multiple nodes) - or exactly fills one node, leaving no room for a second instance's driver
    there - no other instance's driver may share any of those nodes, or the original bug this
    design fixes reappears: vLLM refuses to start once a node's local GPU share is exhausted by an
    earlier instance, even though other nodes are completely free.
    """
    if not gpus_per_node:
        return None
    tp_pp = tensor_parallel_size * pipeline_parallel_size
    return max(1, gpus_per_node // tp_pp)


def build_instance_command(
    model: str, tensor_parallel_size: int, pipeline_parallel_size: int, trust_remote_code: bool, port: int
) -> list[str]:
    """Same flags as a single-instance-multi-node `vllm serve` invocation
    (see `_build_vllm_single_instance_multi_node_command`) - every replica runs one of these."""
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
    if pipeline_parallel_size > 1:
        cmd += ["--pipeline-parallel-size", str(pipeline_parallel_size)]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


app = FastAPI()


@serve.deployment
@serve.ingress(app)
class VLLMInstance:
    """One Ray Serve replica = one vLLM instance. Launches its own `vllm serve` subprocess in
    `__init__` (blocking until it's actually healthy, so Serve doesn't route traffic to a replica
    that isn't ready yet) and proxies every request to it locally. `max_replicas_per_node` (set at
    bind time in main(), via max_replicas_per_node()) is what keeps instance drivers from
    over-packing a node's GPU capacity; Ray Serve's own HTTP proxy is what load-balances requests
    across replicas - this class has no routing logic of its own.
    """

    def __init__(
        self, model: str, tensor_parallel_size: int, pipeline_parallel_size: int, trust_remote_code: bool
    ) -> None:
        port = free_local_port()
        self._base_url = f"http://localhost:{port}"
        cmd = build_instance_command(model, tensor_parallel_size, pipeline_parallel_size, trust_remote_code, port)
        # This subprocess's own internal `ray.init()` (inside vLLM's Ray distributed executor)
        # needs RAY_ADDRESS to discover the cluster this replica actor already joined - without it,
        # it silently starts a separate, single-machine local Ray cluster of its own instead,
        # defeating the whole point (Ray's placement-group scheduler wouldn't see the other
        # instances, so multiple instances could contend for the same GPUs).
        env = {**os.environ, "RAY_ADDRESS": ray.get_runtime_context().gcs_address}
        self._proc = subprocess.Popen(cmd, env=env)
        self._session = aiohttp.ClientSession()
        self._wait_until_healthy()

    def _wait_until_healthy(self) -> None:
        # Synchronous/blocking is deliberate: Serve doesn't consider a replica "started" (and
        # won't route traffic to it) until __init__ returns, so blocking here is what gates
        # startup - matching (and simplifying) the previous design's separate pre-serve.run() wait.
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
        # Called periodically by Ray Serve after startup - raising here marks this replica
        # unhealthy (and Serve stops routing new requests to it) if the vLLM subprocess has died.
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

    # Serve's own replica-routing RPC (asking a replica for its current queue length, to pick the
    # less-loaded of two candidates - see module docstring) defaults to a 0.1s deadline, tight
    # enough that cross-node hops on a Slurm cluster - especially while a replica's node is busy
    # doing GPU inference - routinely miss it and fall back to less-informed routing. Passed as a
    # job-level runtime_env so it reaches Serve's own controller/proxy/replica actors, which are
    # only created once serve.start()/serve.run() runs below, as part of this same job.
    runtime_env = {"env_vars": {"RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S": "1.0"}}
    try:
        ray.init(address="auto", runtime_env=runtime_env)
    except ConnectionError:
        # No existing cluster to join (e.g. the single-node opt-in case, where the sbatch script
        # skips the multi-node Ray bootstrap entirely) - start a local one.
        ray.init(runtime_env=runtime_env)

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
