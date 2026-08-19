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

"""Ray Serve application wrapping vLLM's OpenAI-compatible server.

Not imported by the gym package itself - `vllm` and `ray[serve]` are only present inside the
vLLM service container (e.g. vllm/vllm-openai) that the `distributed_backend: {type: ray_serve}`
Slurm command installs nemo_gym into and runs against, via:

    serve run nemo_gym.orchestration.ray_serve_vllm_app:build_app \\
        model=... port=... tensor_parallel_size=... pipeline_parallel_size=... \\
        number_of_instances=... trust_remote_code=...

`build_app` takes a single dict parameter (Ray Serve's builder-function convention for CLI args),
not individual keyword parameters.

Each replica runs its own vLLM engine sized to tensor_parallel_size x pipeline_parallel_size GPUs
(vLLM's default in-process executor, not Ray) and exposes the same routes as `vllm serve`
(/v1/chat/completions, /v1/models, /health, ...); Ray Serve schedules and bin-packs
number_of_instances replicas across whatever GPUs are free across the cluster.
"""

from typing import Any


def _to_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in ("1", "true", "yes")


def build_app(args: dict) -> Any:
    """Build the Ray Serve application.

    `serve run module:build_app key=val ...` requires the builder to take exactly one parameter: a
    dict of the passed args (always strings, since they arrive from the CLI).
    """
    from ray import serve
    from ray.serve.exceptions import RayServeException
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.openai.api_server import build_app as build_vllm_fastapi_app
    from vllm.entrypoints.openai.api_server import init_app_state
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    try:
        # Newer vLLM releases moved FlexibleArgumentParser out of vllm.utils.
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        from vllm.utils import FlexibleArgumentParser

    model = args["model"]
    port = int(args.get("port", 8000))
    tensor_parallel_size = int(args.get("tensor_parallel_size", 1))
    pipeline_parallel_size = int(args.get("pipeline_parallel_size", 1))
    number_of_instances = int(args.get("number_of_instances", 1))
    trust_remote_code = _to_bool(args.get("trust_remote_code", False))

    # `serve run`'s own CLI flags for the HTTP proxy's host/port aren't consistently available
    # across Ray versions for import-path apps, so set them here instead. `serve run` may have
    # already started Serve with its own defaults before calling this builder; in that case
    # serve.start raises (Serve is already running), which is fine as long as it's already bound to
    # the port we need.
    try:
        serve.start(http_options={"host": "0.0.0.0", "port": port})
    except RayServeException:
        pass

    cli_args = [
        "--model",
        model,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--pipeline-parallel-size",
        str(pipeline_parallel_size),
    ]
    if trust_remote_code:
        cli_args.append("--trust-remote-code")

    vllm_args = make_arg_parser(FlexibleArgumentParser()).parse_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(vllm_args)

    fastapi_app = build_vllm_fastapi_app(vllm_args)
    gpus_per_replica = tensor_parallel_size * pipeline_parallel_size

    @serve.deployment(num_replicas=number_of_instances, ray_actor_options={"num_gpus": gpus_per_replica})
    @serve.ingress(fastapi_app)
    class VLLMReplica:
        def __init__(self) -> None:
            self.engine_client = AsyncLLMEngine.from_engine_args(engine_args)
            # vLLM's V1 engine (AsyncLLM, what from_engine_args returns by default) exposes
            # model_config directly; the legacy V0 AsyncLLMEngine nests it under .engine instead.
            model_config = getattr(self.engine_client, "model_config", None)
            if model_config is None:
                model_config = self.engine_client.engine.get_model_config()
            init_app_state(self.engine_client, model_config, fastapi_app.state, vllm_args)

    return VLLMReplica.bind()
