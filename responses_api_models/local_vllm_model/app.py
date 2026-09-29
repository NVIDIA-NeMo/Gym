# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import sys
from argparse import Namespace
from contextlib import asynccontextmanager
from pathlib import Path
from time import sleep
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

import ray
import requests
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from ray import available_resources, cluster_resources
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from requests.exceptions import ConnectionError

from nemo_gym.global_config import (
    DISALLOWED_PORTS_KEY_NAME,
    find_open_port,
    get_global_config_dict,
    get_hf_token,
)
from responses_api_models.local_vllm_model.pd_launcher import VLLMPDConfig, VLLMPDLauncher
from responses_api_models.local_vllm_model.router_launcher import VLLMRouterConfig, VLLMRouterLauncher
from responses_api_models.local_vllm_model.subprocess_launcher import (
    VLLMSubprocessConfig,
    VLLMSubprocessLauncher,
    normalize_kwargs,
    validate_managed_kwargs,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class LocalVLLMModelConfig(VLLMModelConfig):
    # We inherit these configs from VLLMModelConfig, but they are set to optional since they will be set later on after we spin up a model endpoint.
    base_url: Union[str, List[str]] = Field(default_factory=list)
    # Not used on local deployments
    api_key: str = "dummy"  # pragma: allowlist secret

    hf_home: Optional[str] = None
    vllm_serve_kwargs: Dict[str, Any]
    vllm_serve_env_vars: Dict[str, str]

    # Preserve legacy defaults until the subprocess GPU acceptance gate has passed.
    launcher: Literal["ray", "subprocess"] = "ray"
    subprocess: VLLMSubprocessConfig = Field(default_factory=VLLMSubprocessConfig)
    router: VLLMRouterConfig | None = None
    pd: VLLMPDConfig | None = None

    ray_worker_py_executable: str = sys.executable

    show_vllm_engine_stats: bool = False
    debug: bool = False

    @model_validator(mode="after")
    def validate_launcher(self) -> "LocalVLLMModelConfig":
        if self.pd is not None and (self.launcher != "subprocess" or self.base_url or not self.router):
            raise ValueError("Managed PD requires launcher=subprocess and router, without base_url")
        if self.router is not None and (self.launcher != "subprocess" or self.base_url):
            raise ValueError("Managed router requires launcher=subprocess without base_url")
        if self.launcher == "subprocess" and not self.base_url:
            if (self.num_workers or 1) != 1:
                raise ValueError(
                    "Managed subprocess serving requires num_workers=1; use base_url for external workers"
                )
            validate_managed_kwargs(normalize_kwargs(self.vllm_serve_kwargs), self.vllm_serve_env_vars)
            self.routing_authority = "vllm_router" if self.router else "gym"
            self.native_dp_size = (
                1 if self.router else normalize_kwargs(self.vllm_serve_kwargs).get("data_parallel_size", 1)
            )
            if self.router:
                self.routing_timeout_seconds = self.router.inference_timeout_seconds
            self.validate_routing()
        return self

    def model_post_init(self, context):
        # Default to the .cache/huggingface in this directory.
        if not self.hf_home:
            current_directory = Path.cwd()
            self.hf_home = str(current_directory / ".cache" / "huggingface")

        return super().model_post_init(context)


class GetInnerVLLMConfigResponse(BaseModel):
    base_url: List[str]
    api_key: str
    model: str
    routing_authority: Literal["gym", "vllm_router"] = "gym"
    native_dp_size: int = 1
    routing_timeout_seconds: float = 600


def _legacy_actor_class():
    # Keep vLLM and its private API patches out of subprocess/external deployments.
    from responses_api_models.local_vllm_model.local_vllm_model_actor import LocalVLLMModelActor

    return LocalVLLMModelActor


class LocalVLLMModel(VLLMModel):
    non_generating_model_routes: ClassVar[frozenset[tuple[str, str]]] = frozenset({("GET", "/get_inner_vllm_config")})
    config: LocalVLLMModelConfig

    _local_vllm_model_actor: Any = PrivateAttr(default=None)
    _subprocess_launcher: VLLMSubprocessLauncher | VLLMPDLauncher | None = PrivateAttr(default=None)
    _router_launcher: Optional[VLLMRouterLauncher] = PrivateAttr(default=None)

    def setup_webserver(self):
        managed_subprocess = self.config.launcher == "subprocess" and not self.config.base_url
        if not managed_subprocess:
            self.start_vllm_server()

        app = super().setup_webserver()

        # This route is only used to support LocalVLLMModelProxy
        app.get("/get_inner_vllm_config")(self.get_inner_vllm_config)

        if managed_subprocess:
            original_lifespan = app.router.lifespan_context

            @asynccontextmanager
            async def lifespan(application):
                env = {"HF_HOME": self.config.hf_home, **self.config.vllm_serve_env_vars}
                if token := get_hf_token():
                    env.setdefault("HF_TOKEN", token)
                launcher_class = VLLMPDLauncher if self.config.pd is not None else VLLMSubprocessLauncher
                pd_options = {"pd": self.config.pd, "router": self.config.router} if self.config.pd is not None else {}
                launcher = launcher_class(
                    config=self.config.subprocess,
                    model=self.config.model,
                    kwargs=self.config.vllm_serve_kwargs,
                    env=env,
                    api_key=self.config.api_key,
                    cache_dir=self.get_cache_dir(),
                    show_stats=self.config.show_vllm_engine_stats,
                    **pd_options,
                )
                self._subprocess_launcher = launcher
                try:
                    port = (
                        self.config.subprocess.port
                        or (self.config.pd.prefill.port if self.config.pd else None)
                        or find_open_port(disallowed_ports=get_global_config_dict().get(DISALLOWED_PORTS_KEY_NAME, []))
                    )
                    base_url = await launcher.start(port)
                    if self.config.router and self.config.pd is None:
                        router = VLLMRouterLauncher(
                            config=self.config.router, model=self.config.model, api_key=self.config.api_key
                        )
                        self._router_launcher = router
                        router_port = self.config.router.port or find_open_port(
                            disallowed_ports=[port, *get_global_config_dict().get(DISALLOWED_PORTS_KEY_NAME, [])]
                        )
                        base_url = await router.start(
                            router_port,
                            worker_urls=[base_url.removesuffix("/v1")],
                            dp_size=launcher.topology["data_parallel_size"],
                        )
                    self.config.base_url = [base_url]
                    self._post_init()
                    async with original_lifespan(application) as state:
                        yield state
                finally:
                    try:
                        if self._router_launcher:
                            await self._router_launcher.stop()
                    finally:
                        await launcher.stop()
                        self.config.base_url = []
                        self._clients = []

            app.router.lifespan_context = lifespan

        return app

    async def get_inner_vllm_config(self) -> GetInnerVLLMConfigResponse:
        return GetInnerVLLMConfigResponse(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.config.model,
            routing_authority=self.config.routing_authority,
            native_dp_size=self.config.native_dp_size,
            routing_timeout_seconds=self.config.routing_timeout_seconds,
        )

    def get_cache_dir(self) -> str:
        # We need to reconstruct the cache dir as HF does it given HF_HOME. See https://github.com/huggingface/huggingface_hub/blob/b2723cad81f530e197d6e826f194c110bf92248e/src/huggingface_hub/constants.py#L146
        return str(Path(self.config.hf_home) / "hub")

    def _configure_vllm_serve(self) -> Tuple[Namespace, Dict[str, str]]:
        try:
            from vllm.entrypoints.openai.api_server import (
                FlexibleArgumentParser,
                cli_env_setup,
                make_arg_parser,
                validate_parsed_serve_args,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "vllm":
                raise RuntimeError(
                    "launcher=ray requires local-vllm-model[legacy]; alternatively select launcher=subprocess"
                ) from exc
            raise

        server_args = self.config.vllm_serve_kwargs

        port = find_open_port(disallowed_ports=get_global_config_dict()[DISALLOWED_PORTS_KEY_NAME])
        cache_dir = self.get_cache_dir()
        server_args = server_args | {
            "model": self.config.model,
            "host": "0.0.0.0",  # Must be 0.0.0.0 for cross-node communication.
            "port": port,
            "distributed_executor_backend": "ray",
            "data_parallel_backend": "ray",
            "download_dir": cache_dir,
        }

        env_vars = {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
        # vLLM accepts a `hf_token` parameter but it's not used everywhere. We need to set HF_TOKEN environment variable here.
        maybe_hf_token = get_hf_token()
        if maybe_hf_token:
            env_vars["HF_TOKEN"] = maybe_hf_token

        env_vars.update(self.config.vllm_serve_env_vars)

        assert "VLLM_RAY_DP_PACK_STRATEGY" in env_vars, (
            f"Please provide a value for `VLLM_RAY_DP_PACK_STRATEGY` for `{self.config.name}`"
        )
        assert server_args.get("data_parallel_size")
        assert server_args.get("tensor_parallel_size")
        assert server_args.get("pipeline_parallel_size")

        # With our vLLM patches, this assert is no longer necessary
        # Ray backend only works if dp_size > 1
        # assert server_args.get("data_parallel_size") is None or server_args.get("data_parallel_size") > 1, (
        #     "Ray backend only works with data parallel size > 1!"
        # )

        # With our vLLM patches, this is no longer necessary for people to set.
        server_args["data_parallel_size_local"] = 1

        # TODO multi-node model instances still need to be properly supported
        # We get a vLLM error: Exception: Error setting CUDA_VISIBLE_DEVICES: local range: [0, 16) base value: "0,1,2,3,4,5,6,7"
        if env_vars.get("VLLM_RAY_DP_PACK_STRATEGY") == "span":
            # Unset this flag since it's set by default using span
            server_args.pop("data_parallel_size_local", None)

        cli_env_setup()
        parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)
        final_args = parser.parse_args(namespace=Namespace(**server_args))
        validate_parsed_serve_args(final_args)

        # @bxyu-nvidia: TODO remove, specific to Nemotron 3 Ultra vLLM version.
        # Upstream vLLM only exposes `enable_return_routed_experts`, so alias it across.
        final_args.return_routed_experts = final_args.enable_return_routed_experts

        if self.config.debug:
            env_vars_to_print = env_vars.copy()
            if "HF_TOKEN" in env_vars_to_print:
                env_vars_to_print["HF_TOKEN"] = "****"
            print(f"""Final vLLM serve arguments: {final_args}
Environment variables: {env_vars_to_print}""")

        return final_args, env_vars

    def _ray_actor_path(self) -> str:
        """PATH for the Ray actor running vLLM.

        runtime_env.py_executable gives the actor this server's venv interpreter, but does not
        put that venv's bin directory on its PATH, so console scripts installed next to the
        interpreter are not resolvable in the actor or its child processes. vLLM declares ninja
        as a runtime dependency and shells out to it when compiling kernels, which fails with
        "No such file or directory: 'ninja'" even though ninja is installed in the venv.

        Prepend the interpreter's directory, keeping the inherited PATH as a fallback.
        """
        venv_bin_dir = str(Path(self.config.ray_worker_py_executable).resolve().parent)
        return os.pathsep.join(filter(None, [venv_bin_dir, os.environ.get("PATH", "")]))

    def _select_vllm_server_head_node(self, server_args: Namespace, env_vars: Dict[str, str]) -> PlacementGroup:
        """
        Our LocalVLLMModelActor Ray actor scheduling strategy is as follows:
        1. We estimate the size of a single placement group vLLM will make using TP * PP
        2. We pre-maturely create one placement group of this size which will server as the master node for the vLLM instance
        3. This placement group is also provided on input to the LocalVLLMModelActor, which will schedule (DP - 1) additional placement groups of size TP * PP
        """
        # This mirrors the placement group logic above
        pack_strategy = env_vars["VLLM_RAY_DP_PACK_STRATEGY"]
        if pack_strategy in ("strict", "fill"):
            placement_strategy = "STRICT_PACK"
        else:
            placement_strategy = "PACK"

        device_str = "GPU"
        device_bundle = [{device_str: 1.0}]
        world_size = server_args.pipeline_parallel_size * server_args.tensor_parallel_size
        bundles = device_bundle * world_size + [{"CPU": 1.0}]
        head_node_placement_group = ray.util.placement_group(
            name=f"{self.config.name}_dp_rank_0",
            strategy=placement_strategy,
            bundles=bundles,
        )
        ray.get(head_node_placement_group.ready())

        return head_node_placement_group

    def start_vllm_server(self) -> None:
        # If base_url is already set, skip local launch — connect to external server.
        if self.config.base_url:
            print(f"External base_url configured: {self.config.base_url}. Skipping local vLLM launch.")
            self._post_init()
            return

        if self.config.launcher == "subprocess":
            raise RuntimeError("Subprocess startup is asynchronous; enter the app's FastAPI lifespan instead")

        if self.config.debug:
            print(f"""Currently available Ray cluster resources: {available_resources()}
Total Ray cluster resources: {cluster_resources()}""")

        server_args, env_vars = self._configure_vllm_serve()
        head_node_placement_group = self._select_vllm_server_head_node(server_args, env_vars)

        pythonpath = str(Path(__file__).parent.parent.parent)
        if self.config.debug:
            print(f"Using PYTHONPATH={pythonpath}")

        self._local_vllm_model_actor = (
            _legacy_actor_class()
            .options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=head_node_placement_group,
                ),
                runtime_env=dict(
                    py_executable=self.config.ray_worker_py_executable,
                    env_vars={
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        "PYTHONPATH": pythonpath,
                        # Listed before `env_vars` so a server config can still override PATH.
                        "PATH": self._ray_actor_path(),
                        **env_vars,
                    },
                ),
            )
            .remote(
                head_node_placement_group=head_node_placement_group,
                server_args=server_args,
                env_vars=env_vars,
                server_name=self.config.name,
                debug=self.config.debug,
                show_vllm_engine_stats=self.config.show_vllm_engine_stats,
            )
        )

        self.config.base_url = [ray.get(self._local_vllm_model_actor.base_url.remote())]

        # Reset clients after base_url config
        self._post_init()

        self.await_server_ready()

    def await_server_ready(self) -> None:
        poll_count = 0
        while True:
            is_alive = ray.get(self._local_vllm_model_actor.is_alive.remote())
            assert is_alive, f"{self.config.name} LocalVLLMModel server spinup failed, see the error logs above!"

            try:
                requests.get(url=f"{self.config.base_url[0]}/models")
                return
            except ConnectionError:
                if poll_count % 10 == 0:  # Print every 30s
                    print(f"Waiting for {self.config.name} LocalVLLMModel server to spinup...")

                poll_count += 1
                sleep(3)


if __name__ == "__main__":
    LocalVLLMModel.run_webserver()
