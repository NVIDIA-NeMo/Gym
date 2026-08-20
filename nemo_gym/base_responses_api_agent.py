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
from abc import abstractmethod
from collections.abc import Mapping
from functools import wraps
from typing import Any, Optional
from warnings import warn

from fastapi import Body, FastAPI, Request

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.config_types import ROLLOUT_PATH_PREFIX, TOKEN_CAPTURE_PATH_SEGMENT
from nemo_gym.global_config import (
    OBSERVABILITY_ENABLED_KEY_NAME,
    TOKEN_ID_CAPTURE_BLOCK,
    get_first_server_config_dict,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body, rollout_context
from nemo_gym.server_utils import (
    BaseRunServerInstanceConfig,
    BaseServer,
    SimpleServer,
    apply_rollout_prefix,
    rollout_path_prefix,
)


class BaseResponsesAPIAgentConfig(BaseRunServerInstanceConfig):
    skip_verification: bool = False
    skip_verification_reward: float = 0.0
    # Whether this agent's rollouts participate in training token capture.
    # Native agents already receive token ids inline and normally leave this disabled.
    # Opaque external harnesses enable it because their returned output has no token ids.
    # The run-level ``token_id_capture.enabled`` setting gates the capture infrastructure.
    # The run-level ``token_id_capture.all_agents`` setting overrides this agent-level choice.
    token_id_capture: bool = False


class BaseResponsesAPIAgent(BaseServer):
    config: BaseResponsesAPIAgentConfig


class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, AggregateMetricsMixin, SimpleServer):
    config: BaseResponsesAPIAgentConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        app.post("/v1/responses")(self.responses)
        # A self-call made with ``url_path_for_run`` lands on a prefixed twin.
        # ``responses`` recovers the rollout id from the path.
        # The same handler serves prefixed and unprefixed calls.
        app.post(f"/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/v1/responses")(self.responses)
        app.post(f"/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/{TOKEN_CAPTURE_PATH_SEGMENT}/v1/responses")(self.responses)

        run = self.run

        @wraps(run)
        async def run_with_rollout_context(*args: Any, **kwargs: Any) -> BaseVerifyResponse:
            body = kwargs.get("body")
            if body is None:
                body = next((arg for arg in args if isinstance(arg, BaseRunRequest)), None)
            with rollout_context(self.rollout_id_from_run(body)):
                return await run(*args, **kwargs)

        app.post("/run")(run_with_rollout_context)
        app.post("/aggregate_metrics")(self.aggregate_metrics)

        return app

    def _capture_correlation_enabled(self) -> bool:
        """Return whether this agent needs rollout correlation.

        Evaluation uses ``/ng-rollout/<id>/...`` for every agent.
        Training capture uses ``/ng-rollout/<id>/training-token-capture/...``.
        Training capture requires ``token_id_capture.enabled``.
        It also requires the static agent flag or run-level ``all_agents``.
        Missing global configuration disables correlation.
        """
        return self._model_call_capture_enabled() or self._token_id_capture_enabled()

    def _model_call_capture_enabled(self) -> bool:
        """Whether evaluation model-call observability is enabled."""
        global_config = getattr(self.server_client, "global_config_dict", None)
        if not isinstance(global_config, Mapping):
            return False
        return bool(global_config.get(OBSERVABILITY_ENABLED_KEY_NAME, False))

    def _token_id_capture_enabled(self) -> bool:
        """Whether this agent explicitly opted into training-token capture."""
        global_config = getattr(self.server_client, "global_config_dict", None)
        if not isinstance(global_config, Mapping):
            return False
        block = global_config.get(TOKEN_ID_CAPTURE_BLOCK) or {}
        if not isinstance(block, Mapping) or not block.get("enabled", False):
            return False
        return bool(block.get("all_agents", False)) or bool(
            getattr(getattr(self, "config", None), "token_id_capture", False)
        )

    def rollout_id_from_run(self, body: Any) -> Optional[str]:
        """Return the capture id for a run request.

        Return ``None`` when capture is disabled.
        Return ``None`` when the body has no usable identity.
        """
        if not self._capture_correlation_enabled():
            return None
        return maybe_rollout_id_from_run_body(body)

    def url_path_for_run(self, url_path: str, body: Any) -> str:
        """Apply this run's capture path to a downstream URL path.

        Evaluation uses ``/ng-rollout/<id>/...``.
        Training capture uses ``/ng-rollout/<id>/training-token-capture/...``.
        Calls without a rollout id remain unchanged.
        """
        return (
            f"{rollout_path_prefix(self.rollout_id_from_run(body), token_capture=self._token_id_capture_enabled())}"
            f"{url_path}"
        )

    def base_url_for_run(self, base_url: str, body: Any) -> str:
        """Apply this run's capture path to a model-server root URL.

        Append the API-version suffix after this method returns.
        """
        return apply_rollout_prefix(
            base_url,
            self.rollout_id_from_run(body),
            token_capture=self._token_id_capture_enabled(),
        )

    def url_path_for_request(self, url_path: str, request: Optional[Request]) -> str:
        """Carry an inbound capture path onto a downstream URL path.

        Prefixed self-calls expose the rollout id as a path parameter.
        Training-capture requests preserve their dedicated path segment.
        Unprefixed requests remain unchanged.
        """
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        request_path = getattr(getattr(request, "url", None), "path", "")
        token_capture = f"/{TOKEN_CAPTURE_PATH_SEGMENT}/" in request_path
        return f"{rollout_path_prefix(rollout_id, token_capture=token_capture)}{url_path}"

    def resolve_model_base_url(self, model_server_name: str, rollout_id: Optional[str] = None) -> str:
        """Resolve a model-server URL with an optional rollout prefix."""
        server_config = get_first_server_config_dict(self.server_client.global_config_dict, model_server_name)
        base_url = self.server_client._build_server_base_url(server_config)
        return f"{apply_rollout_prefix(base_url, rollout_id, token_capture=self._token_id_capture_enabled())}/v1"

    # TODO: right now there is no validation on the TypedDict NeMoGymResponseCreateParamsNonStreaming
    # We should explicitly add validation at this server level or we should explicitly not validate so that there is flexibility in this API.
    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass

    @abstractmethod
    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        pass

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Default: same RewardProfiler aggregation as resources server. Override to proxy."""
        if self.config.skip_verification:
            warn(
                "Skipping aggregate metrics because skip_verification=True; "
                "use disable_aggregation=True to avoid writing aggregate metric files.",
                RuntimeWarning,
                stacklevel=2,
            )
            return AggregateMetrics()

        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )
