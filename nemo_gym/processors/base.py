# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server surface for episode processors."""

from abc import abstractmethod
from functools import wraps
from typing import Any

from fastapi import Body, FastAPI

from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, BaseRunServerInstanceConfig
from nemo_gym.processors.contracts import EpisodeRequest, EpisodeResponse
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_correlation import rollout_context
from nemo_gym.server_utils import SimpleServer, rollout_path_prefix
from nemo_gym.telemetry.endpoints import traced_rollout_endpoint


class BaseProcessorConfig(BaseRunServerInstanceConfig):
    token_id_capture: bool = False


class BaseProcessor(AggregateMetricsMixin, SimpleServer):
    """Validate and execute one native episode request."""

    config: BaseProcessorConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        attributes = {"nemo.gym.server.name": self.config.name}
        run = traced_rollout_endpoint(self.run, attributes)

        @wraps(run)
        async def run_with_rollout_context(*args: Any, **kwargs: Any) -> EpisodeResponse:
            body = kwargs.get("body")
            if body is None:
                body = next((arg for arg in args if isinstance(arg, EpisodeRequest)), None)
            rollout_id = body.episode_id.rollout_id if isinstance(body, EpisodeRequest) else None
            with rollout_context(rollout_id):
                return await run(*args, **kwargs)

        app.post("/run")(run_with_rollout_context)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    def url_path_for_run(self, url_path: str, body: EpisodeRequest) -> str:
        rollout_id = body.episode_id.rollout_id
        if body.episode_id.attempt:
            rollout_id = f"{rollout_id}-a{body.episode_id.attempt}"
        return f"{rollout_path_prefix(rollout_id, token_capture=self.config.token_id_capture)}{url_path}"

    @abstractmethod
    async def run(self, body: EpisodeRequest = Body()) -> EpisodeResponse:
        pass

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )
