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
from typing import Any, Optional

from fastapi import Body, FastAPI, Request

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_model import maybe_rollout_id_from_run_body
from nemo_gym.config_types import ROLLOUT_PATH_PREFIX
from nemo_gym.global_config import OBSERVABILITY_ENABLED_KEY_NAME, get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.server_utils import (
    BaseRunServerInstanceConfig,
    BaseServer,
    SimpleServer,
    apply_rollout_prefix,
    rollout_path_prefix,
)


class BaseResponsesAPIAgentConfig(BaseRunServerInstanceConfig):
    pass


class BaseResponsesAPIAgent(BaseServer):
    config: BaseResponsesAPIAgentConfig


class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, AggregateMetricsMixin, SimpleServer):
    config: BaseResponsesAPIAgentConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        app.post("/v1/responses")(self.responses)
        # Prefixed twin of /v1/responses: a self-call made with url_path_for_run() lands here, and
        # responses() recovers the rollout id from the path (see url_path_for_request) to correlate
        # its model calls. Same handler, so unprefixed calls are unaffected.
        app.post(f"/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/v1/responses")(self.responses)
        app.post("/run")(self.run)
        app.post("/aggregate_metrics")(self.aggregate_metrics)

        return app

    def _model_call_capture_enabled(self) -> bool:
        # Fail closed: an agent whose client carries no usable global config runs uncorrelated
        # rather than erroring on every model call.
        global_config = getattr(self.server_client, "global_config_dict", None)
        try:
            import sys as _sys
            _keys = list(global_config.keys()) if isinstance(global_config, Mapping) else None
            print(
                f"NGDBG gate: server_client_type={type(self.server_client).__name__} "
                f"has_gcd_attr={hasattr(self.server_client, 'global_config_dict')} "
                f"gcd_type={type(global_config).__name__} is_mapping={isinstance(global_config, Mapping)} "
                f"key={OBSERVABILITY_ENABLED_KEY_NAME!r} present={(_keys is not None and OBSERVABILITY_ENABLED_KEY_NAME in _keys)} "
                f"value={global_config.get(OBSERVABILITY_ENABLED_KEY_NAME) if isinstance(global_config, Mapping) else 'NA'!r} "
                f"top_level_keys={_keys}",
                file=_sys.stderr, flush=True,
            )
        except Exception as _e:
            print(f"NGDBG gate: EXC {_e!r}", file=__import__('sys').stderr, flush=True)
        if not isinstance(global_config, Mapping):
            return False
        return bool(global_config.get(OBSERVABILITY_ENABLED_KEY_NAME, False))

    def rollout_id_from_run(self, body: Any) -> Optional[str]:
        """Per-rollout capture id for a run-request (its task/rollout indices).

        None when model-call capture (observability) is disabled or the body carries no indices,
        so callers apply no correlation prefix in either case.
        """
        enabled = self._model_call_capture_enabled()
        rid = maybe_rollout_id_from_run_body(body) if enabled else None
        import sys as _sys
        _bt = body if isinstance(body, Mapping) else getattr(body, "__dict__", body)
        _idx = None
        try:
            _dump = body.model_dump() if hasattr(body, "model_dump") else (dict(body) if isinstance(body, Mapping) else {})
            _idx = {k: _dump.get(k) for k in ("_ng_task_index", "_ng_rollout_index", "_ng_attempt_index")}
        except Exception as _e:
            _idx = f"EXC {_e!r}"
        print(f"NGDBG rollout_id_from_run: enabled={enabled} rollout_id={rid!r} body_indices={_idx}", file=_sys.stderr, flush=True)
        return rid

    def url_path_for_run(self, url_path: str, body: Any) -> str:
        """A downstream url_path with the per-rollout capture-correlation prefix applied.

        Returns ``/ng-rollout/<id><url_path>`` when observability is enabled and the run body
        carries task/rollout indices; otherwise ``url_path`` unchanged. Use for calls made while
        handling ``/run`` — both direct model-server calls and self-calls to ``/v1/responses``
        (the prefixed self-call route carries the id into ``responses()``).
        """
        return f"{rollout_path_prefix(self.rollout_id_from_run(body))}{url_path}"

    def base_url_for_run(self, base_url: str, body: Any) -> str:
        """A model-server base URL with the per-rollout capture-correlation prefix applied.

        ``base_url_for_run`` is the base-URL counterpart of ``url_path_for_run`` for SDK-style
        harnesses that configure a client once instead of prefixing each call: same gating, applied
        to a server root URL (append the API-version suffix afterwards).
        """
        _rid = self.rollout_id_from_run(body)
        _out = apply_rollout_prefix(base_url, _rid)
        import sys as _sys
        print(f"NGDBG base_url_for_run: in={base_url!r} rollout_id={_rid!r} out={_out!r}", file=_sys.stderr, flush=True)
        return _out

    def url_path_for_request(self, url_path: str, request: Optional[Request]) -> str:
        """Carry an inbound ``/ng-rollout/<id>`` self-call prefix onto a downstream url_path.

        Agents whose model calls happen inside ``responses()`` receive the correlation id as the
        ``rollout_id`` path parameter of the prefixed self-call route; this re-applies it to the
        outgoing model call. Unprefixed requests pass through unchanged.
        """
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        return f"{rollout_path_prefix(rollout_id)}{url_path}"

    def resolve_model_base_url(self, model_server_name: str, rollout_id: Optional[str] = None) -> str:
        """Resolve a model-server URL with an optional rollout prefix."""
        server_config = get_first_server_config_dict(self.server_client.global_config_dict, model_server_name)
        base_url = self.server_client._build_server_base_url(server_config)
        return f"{apply_rollout_prefix(base_url, rollout_id)}/v1"

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
        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )
