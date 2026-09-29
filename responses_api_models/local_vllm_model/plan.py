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
"""Probe executables and emit manifests without starting serving/GPU workers.

Pass a resolved Gym YAML from `gym env resolve`, or one LocalVLLMModel config.
This runs bounded --version/--help probes, not model loading or inference.
"""

import argparse
import asyncio
from pathlib import Path

import yaml

from responses_api_models.local_vllm_model.app import LocalVLLMModelConfig
from responses_api_models.local_vllm_model.pd_launcher import VLLMPDLauncher
from responses_api_models.local_vllm_model.router_launcher import VLLMRouterLauncher
from responses_api_models.local_vllm_model.subprocess_launcher import VLLMSubprocessLauncher


async def plan(config: LocalVLLMModelConfig, vllm_port: int, router_port: int) -> list[Path]:
    if config.launcher != "subprocess" or config.base_url:
        raise ValueError("Dry-run manifests require managed launcher=subprocess, without base_url")
    if config.router and router_port == vllm_port:
        raise ValueError("Router and vLLM ports must differ")
    launcher_class = VLLMPDLauncher if config.pd is not None else VLLMSubprocessLauncher
    pd_options = {"pd": config.pd, "router": config.router} if config.pd is not None else {}
    launcher = launcher_class(
        config=config.subprocess,
        model=config.model,
        kwargs=config.vllm_serve_kwargs,
        env={"HF_HOME": config.hf_home, **config.vllm_serve_env_vars},
        api_key=config.api_key,
        cache_dir=str(Path(config.hf_home) / "hub"),
        show_stats=config.show_vllm_engine_stats,
        **pd_options,
    )
    if config.pd is not None:
        await launcher.start(vllm_port, router_port=router_port, dry_run=True)
        return [
            launcher.run_dir / "manifest.json",
            *(worker.run_dir / "manifest.json" for worker in launcher.workers.values()),
            launcher.router.run_dir / "manifest.json",
        ]
    base_url = await launcher.start(vllm_port, dry_run=True)
    manifests = [launcher.run_dir / "manifest.json"]
    if config.router:
        router = VLLMRouterLauncher(config=config.router, model=config.model, api_key=config.api_key)
        await router.start(
            router_port,
            worker_urls=[base_url.removesuffix("/v1")],
            dp_size=launcher.topology["data_parallel_size"],
            dry_run=True,
        )
        manifests.append(router.run_dir / "manifest.json")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--name", default="policy_model")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--router-port", type=int, default=8001)
    args = parser.parse_args()
    data = yaml.safe_load(args.config.read_text())
    if args.name in data:
        data = data[args.name]["responses_api_models"]["local_vllm_model"]
    config = LocalVLLMModelConfig.model_validate(data)
    for manifest in asyncio.run(
        plan(
            config,
            config.subprocess.port or args.vllm_port,
            config.router.port or args.router_port if config.router else args.router_port,
        )
    ):
        print(manifest)


if __name__ == "__main__":
    main()
