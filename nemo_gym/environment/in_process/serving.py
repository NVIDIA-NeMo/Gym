# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start a benchmark's servers from Python."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Optional

from omegaconf import OmegaConf

from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig, ConfigError, ignoring_command_line
from nemo_gym.config.global_config import (
    HEAD_SERVER_KEY_NAME,
    GlobalConfigDictParserConfig,
)
from nemo_gym.data.tasks.harbor.prepare import HarborBenchmark
from nemo_gym.environment.assets import asset_config_path
from nemo_gym.environment.in_process.harbor import require_run_files, resolve_harbor_benchmark
from nemo_gym.environment.in_process.overrides import _checked_overrides
from nemo_gym.environment.in_process.planning import PlannedBenchmark
from nemo_gym.server_runtime.supervisor.server_supervisor import (
    ServerSpinupTimeoutError,
    ServerStartFailedError,
    ServerSupervisor,
)
from nemo_gym.utils.decorators import experimental


@experimental(once=True)
@contextmanager
def serve_benchmark(
    benchmark: str,
    *,
    model_type: str = "openai_model",
    overrides: Optional[Mapping[str, Any]] = None,
    registry_port: Optional[int] = None,
    planned: Optional[PlannedBenchmark] = None,
) -> Iterator[BaseServerConfig]:
    """Start the benchmark's servers, and stop them on exit. Yields the registry server a `TaskRunner` talks to.

    The benchmark `NAME[/FLAVOR]` and the model type are named as in
    `gym env start --benchmark NAME --model-type TYPE`. `overrides` are config keys merged on top, as in
    `plan_benchmark`, so planning and serving read the same dict: nested dicts, and a dotted key is refused
    (`ConfigError`). The model's
    `policy_base_url`, `policy_api_key` and `policy_model_name` are among them. `registry_port` fixes the registry
    server's port, so a caller that serves from another process can know the address before the servers start;
    Gym chooses the port when it is None.

    A Harbor-backed benchmark (its config sets `harbor_target`) is served from the configuration `plan_benchmark`
    wrote for it, so `planned` is required and must have been planned in this process's file system: the Harbor
    environment server and the agent start from it, as under `gym eval run --benchmark NAME`.

    Returns once every server is ready; raises `ConfigError` if a server or a model endpoint is not, or if
    `registry_port` is in use. Like `gym env start`, it sets this process's Gym configuration, so serve one
    benchmark per process.
    """
    harbor = resolve_harbor_benchmark(benchmark)
    if harbor is None:
        benchmark_paths = [asset_config_path("benchmark", benchmark)]
    else:
        benchmark_paths = _harbor_config_paths(benchmark, harbor, planned)
    config_paths = [*benchmark_paths, asset_config_path("model-type", model_type)]
    initial = OmegaConf.merge({"config_paths": config_paths}, _checked_overrides(overrides))
    if registry_port is not None:
        if not 0 < registry_port < 65536:
            raise ConfigError(f"registry_port must be a port number from 1 to 65535, got {registry_port!r}")
        initial = OmegaConf.merge(initial, {HEAD_SERVER_KEY_NAME: {"port": registry_port}})
    supervisor = ServerSupervisor()
    try:
        with ignoring_command_line():
            supervisor.start(
                GlobalConfigDictParserConfig(
                    initial_global_config_dict=initial, skip_load_from_cli=True, skip_exporters=True
                )
            )
    except (ServerSpinupTimeoutError, ServerStartFailedError) as error:
        raise ConfigError(f"Servers of benchmark {benchmark!r} did not start: {error}") from error
    try:
        registry_server = ServerClient.load_head_server_config()
        if registry_port is not None and registry_server.port != registry_port:
            raise ConfigError(
                f"registry_port {registry_port} is in use: the registry server started on port {registry_server.port}"
            )
        yield registry_server
    finally:
        supervisor.shutdown()


def _harbor_config_paths(benchmark: str, harbor: HarborBenchmark, planned: Optional[PlannedBenchmark]) -> list[str]:
    if planned is None or planned.harbor is None:
        raise ConfigError(
            f"Benchmark {benchmark!r} runs the Harbor target {harbor.target!r}: serve it with the "
            "PlannedBenchmark that plan_benchmark returned for it"
        )
    if planned.harbor.target != harbor.target:
        raise ConfigError(
            f"The plan is for Harbor target {planned.harbor.target!r}, but {benchmark!r} runs {harbor.target!r}"
        )
    require_run_files(planned.harbor)
    return list(planned.harbor.config_paths)
