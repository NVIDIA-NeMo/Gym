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
"""Compare issue #3011's old/new payload construction and a real resources HTTP hop.

Run: python scripts/benchmark_simple_agent_verify.py --output work/verify-benchmark.json
The hop includes construction, ServerClient transport, typed resources ingress,
echoed verification response, and final agent response validation. Model inference
and /seed_session are intentionally outside this focused benchmark.
"""

import argparse
import asyncio
import gc
import json
import multiprocessing
import socket
import tracemalloc
from contextlib import asynccontextmanager
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time

import orjson
import uvicorn
from fastapi import Request
from omegaconf import OmegaConf

from nemo_gym.base_resources_server import BaseResourcesServerConfig, SimpleResourcesServer
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import (
    BaseServerConfig,
    GlobalAIOHTTPAsyncClientConfig,
    ServerClient,
    get_response_json,
    raise_for_status,
    set_global_aiohttp_client,
)
from responses_api_agents.simple_agent.app import (
    SimpleAgentRunRequest,
    SimpleAgentVerifyRequest,
    SimpleAgentVerifyResponse,
)


def build_payload(body, response, legacy):
    if legacy:
        return SimpleAgentVerifyRequest.model_validate(body.model_dump() | {"response": response}).model_dump()
    payload = body.model_dump()
    payload["response"] = response
    return payload


def fingerprint(payload):
    return sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()


def make_case(name):
    token_count = 65536 if name == "training_64k" else 8
    response = NeMoGymResponse.model_validate(
        {
            "id": "terminal-response",
            "created_at": 1.0,
            "model": "benchmark",
            "object": "response",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "message-1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "answer " * token_count, "annotations": []}],
                    "prompt_token_ids": list(range(token_count)),
                    "generation_token_ids": list(range(token_count)),
                    "generation_log_probs": [-0.125] * token_count,
                }
            ],
        }
    ).model_dump(mode="json")
    metadata = {"expected": "answer", "labels": ["a", None, {"terminal": "call-1"}]}
    if name == "large_metadata":
        metadata["records"] = [{"index": i, "text": "metadata " * 128} for i in range(2048)]
    body = SimpleAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "question"}]},
            "task_id": "task-1",
            "_ng_task_index": 3,
            "_ng_rollout_index": 2,
            "verifier_metadata": metadata,
            "terminal_call_id": "call-1",
        }
    )
    return body, response


class EchoResourcesServer(SimpleResourcesServer):
    async def verify(self, request: Request, body: SimpleAgentVerifyRequest) -> SimpleAgentVerifyResponse:
        return SimpleAgentVerifyResponse.model_validate(
            body.model_dump() | {"reward": 1.0, "wire_fingerprint": fingerprint(await request.json())}
        )


def make_client(port):
    return ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=port),
        global_config_dict=OmegaConf.create(
            {"resources": {"resources_servers": {"echo": {"host": "127.0.0.1", "port": port}}}}
        ),
    )


def serve(sock, ready):
    resources = EchoResourcesServer(
        config=BaseResourcesServerConfig(name="resources", entrypoint="", host="127.0.0.1", port=0),
        server_client=make_client(sock.getsockname()[1]),
    )
    app = resources.setup_webserver()

    @asynccontextmanager
    async def lifespan(app):
        ready.set()
        yield

    app.router.lifespan_context = lifespan
    uvicorn.Server(uvicorn.Config(app, log_level="error", http="httptools", loop="uvloop")).run(sockets=[sock])


def distribution(samples, elapsed, cpu):
    ordered = sorted(samples)
    return {
        "requests": len(samples),
        "requests_per_second": len(samples) / elapsed,
        "client_cpu_us_per_request": cpu * 1e6 / len(samples),
        "p50_ms": ordered[(len(ordered) - 1) // 2] * 1000,
        "p99_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))] * 1000,
    }


def construction(body, response, legacy, iterations):
    for _ in range(10):
        build_payload(body, response, legacy)
    samples = []
    cpu_start, start = process_time(), perf_counter()
    for _ in range(iterations):
        before = perf_counter()
        build_payload(body, response, legacy)
        samples.append(perf_counter() - before)
    metrics = distribution(samples, perf_counter() - start, process_time() - cpu_start)
    # Separate allocation sampling from timing. These are retained blocks and peak
    # traced memory for one result, not a count of every transient malloc/free.
    gc.collect()
    tracemalloc.start()
    payload = build_payload(body, response, legacy)
    retained, peak = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    metrics.update(
        retained_bytes=retained,
        peak_traced_bytes=peak,
        retained_allocation_blocks=sum(item.count for item in snapshot.statistics("filename")),
    )
    assert payload["response"]["id"] == response["id"]
    return metrics


async def hop(client, body, response, legacy, concurrency, iterations):
    expected = fingerprint(build_payload(body, response, legacy))

    async def one():
        started = perf_counter()
        reply = await client.post("resources", "/verify", json=build_payload(body, response, legacy), cookies={})
        await raise_for_status(reply)
        result = SimpleAgentVerifyResponse.model_validate(await get_response_json(reply))
        assert result.reward == 1.0 and result.model_extra["wire_fingerprint"] == expected
        return perf_counter() - started

    await one()
    samples = []

    async def worker(count):
        for _ in range(count):
            samples.append(await one())

    cpu_start, start = process_time(), perf_counter()
    await asyncio.gather(
        *(worker(iterations // concurrency + (index < iterations % concurrency)) for index in range(concurrency))
    )
    return distribution(samples, perf_counter() - start, process_time() - cpu_start)


async def benchmark(port, iterations):
    session = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    client = make_client(port)
    report = {
        "note": "Client CPU only; resources run in a separate process. Allocation sampling covers construction only.",
        "cases": {},
    }
    try:
        for name in ("small", "training_64k", "large_metadata"):
            body, response = make_case(name)
            before = deepcopy((body, response))
            old_wire = orjson.loads(orjson.dumps(build_payload(body, response, True)))
            new_wire = orjson.loads(orjson.dumps(build_payload(body, response, False)))
            assert old_wire == new_wire
            case = {"wire_equal": True, "wire_bytes": len(orjson.dumps(new_wire)), "variants": {}}
            for legacy in (True, False):
                label = "before" if legacy else "after"
                metrics = {"construction": construction(body, response, legacy, iterations)}
                for concurrency in (1, 32):
                    metrics[f"http_concurrency_{concurrency}"] = await hop(
                        client, body, response, legacy, concurrency, iterations
                    )
                case["variants"][label] = metrics
                print(json.dumps({"case": name, "variant": label, **metrics}), flush=True)
            assert (body, response) == before
            report["cases"][name] = case

        malformed = build_payload(*make_case("small"), False)
        malformed["response"]["output"][0].pop("generation_log_probs")
        rejected = await client.post("resources", "/verify", json=malformed)
        assert rejected.status == 422
        report["malformed_tokens_http_status"] = rejected.status
        await rejected.read()
        return report
    finally:
        await session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 32:
        parser.error("--iterations must be at least 32")
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        process = context.Process(target=serve, args=(sock, ready))
        process.start()
        try:
            if not ready.wait(timeout=30):
                raise RuntimeError("resources server did not start")
            report = asyncio.run(benchmark(sock.getsockname()[1], args.iterations))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()


if __name__ == "__main__":
    main()
