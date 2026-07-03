#!/usr/bin/env python3
"""Start one topology-pinned vLLM server from a labeled Slurm task."""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from threading import Thread

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


def parse_launcher_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--startup-timeout", type=int, required=True)
    parser.add_argument("vllm_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.vllm_args[:1] == ["--"]:
        args.vllm_args = args.vllm_args[1:]
    if not args.vllm_args:
        parser.error("vLLM serve arguments must follow --")
    return args


@ray.remote(num_cpus=1)
class VLLMServerActor:
    def __init__(self, vllm_args: list[str]) -> None:
        logging.getLogger("ray").setLevel(logging.WARNING)

        from vllm.entrypoints.openai.api_server import (
            FlexibleArgumentParser,
            cli_env_setup,
            make_arg_parser,
            validate_parsed_serve_args,
        )

        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        cli_env_setup()
        parser = FlexibleArgumentParser(
            description="vLLM OpenAI-Compatible RESTful API server."
        )
        server_args = make_arg_parser(parser).parse_args(vllm_args)
        if server_args.model_tag is not None:
            server_args.model = server_args.model_tag
        validate_parsed_serve_args(server_args)

        self._patch_signal_handlers()
        self._error: BaseException | None = None

        def run() -> None:
            try:
                from vllm.entrypoints.openai.api_server import run_server

                asyncio.run(run_server(server_args))
            except BaseException as error:
                self._error = error
                raise

        self._thread = Thread(target=run, daemon=True)
        self._thread.start()

    @staticmethod
    def _patch_signal_handlers() -> None:
        from asyncio import get_running_loop

        import vllm.entrypoints.openai.api_server as api_server

        original_serve_http = api_server.serve_http

        def serve_http_without_signal_handlers(*args, **kwargs):
            loop = get_running_loop()
            loop.add_signal_handler = lambda *args, **kwargs: None
            return original_serve_http(*args, **kwargs)

        api_server.serve_http = serve_http_without_signal_handlers
        signal.signal = lambda *args, **kwargs: None

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError("vLLM server thread failed") from self._error
        if not self._thread.is_alive():
            raise RuntimeError("vLLM server thread exited")


def main() -> None:
    args = parse_launcher_args()
    engine_index = int(os.environ["SLURM_PROCID"])
    groups = [group.split(",") for group in os.environ["VLLM_ENGINE_NODE_GROUPS"].split(";")]
    gpus_per_node = int(os.environ["GPUS_PER_NODE"])

    if engine_index >= len(groups):
        raise RuntimeError(
            f"engine rank {engine_index} has no node group in VLLM_ENGINE_NODE_GROUPS"
        )
    node_ips = groups[engine_index]
    expected_nodes = int(os.environ["NODES_PER_ENGINE"])
    if len(node_ips) != expected_nodes:
        raise RuntimeError(
            f"engine {engine_index} received {len(node_ips)} nodes; expected {expected_nodes}"
        )

    ray.init(address=os.environ["RAY_ADDRESS"], logging_level=logging.INFO)
    logging.getLogger("ray").setLevel(logging.WARNING)
    gpu_bundles = [
        {"GPU": 1.0, f"node:{node_ip}": 0.001}
        for node_ip in node_ips
        for _ in range(gpus_per_node)
    ]
    bundles = [*gpu_bundles, {"CPU": 1.0, f"node:{node_ips[0]}": 0.001}]
    placement_group = ray.util.placement_group(
        bundles,
        strategy="PACK",
        name=f"vllm-{os.environ['SLURM_JOB_ID']}-{engine_index}",
    )
    actor = None
    try:
        ray.get(placement_group.ready(), timeout=args.startup_timeout)
        print(
            f"Engine {engine_index} placement group ready on {','.join(node_ips)}",
            flush=True,
        )
        actor = VLLMServerActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=len(gpu_bundles),
                placement_group_capture_child_tasks=True,
            ),
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "VLLM_RAY_PER_WORKER_GPUS": "1",
                }
            },
        ).remote(args.vllm_args)

        while True:
            ray.get(actor.check.remote())
            time.sleep(5)
    finally:
        if actor is not None:
            ray.kill(actor, no_restart=True)
        ray.util.remove_placement_group(placement_group)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
