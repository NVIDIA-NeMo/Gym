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
import asyncio
import logging
import threading
from abc import abstractmethod
from collections.abc import Mapping
from functools import wraps
from typing import Any, Optional

from fastapi import Body, FastAPI, Request

from nemo_gym.agent_execution_capture import AgentExecutionCaptureStore, AgentExecutionRecorder
from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_model import ModelCallCaptureConfig, maybe_rollout_id_from_run_body
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


logger = logging.getLogger(__name__)


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
        run_handler = self.run
        if self.captures_agent_execution():
            run_handler = self._run_with_agent_execution_capture(run_handler)
        app.post("/run")(run_handler)
        app.post("/aggregate_metrics")(self.aggregate_metrics)

        return app

    def captures_agent_execution(self) -> bool:
        """Whether this Agent Server emits agent-side execution evidence."""
        return False

    def agent_execution_capture_requires_single_worker(self) -> bool:
        """Whether capture crosses a process-local self-call boundary."""
        return False

    def _agent_execution_state(self) -> tuple[threading.Lock, dict[str, AgentExecutionRecorder]]:
        lock = getattr(self, "_agent_execution_lock", None)
        recorders = getattr(self, "_agent_execution_recorders", None)
        if lock is None or recorders is None:
            lock = threading.Lock()
            recorders = {}
            self._agent_execution_lock = lock
            self._agent_execution_recorders = recorders
        return lock, recorders

    def _agent_execution_store(self) -> Optional[AgentExecutionCaptureStore]:
        if "_agent_execution_store_instance" in self.__dict__:
            return self.__dict__["_agent_execution_store_instance"]
        global_config = getattr(self.server_client, "global_config_dict", None)
        if not isinstance(global_config, Mapping):
            return None
        try:
            config = ModelCallCaptureConfig.model_validate(global_config)
            store = (
                AgentExecutionCaptureStore(config.model_call_capture_dir)
                if config.observability_enabled and config.model_call_capture_dir is not None
                else None
            )
        except (OSError, ValueError):
            logger.warning("Could not initialize agent execution capture; disabling it.", exc_info=True)
            store = None
        self._agent_execution_store_instance = store
        return store

    def _start_agent_execution_capture(self, body: Any) -> Optional[AgentExecutionRecorder]:
        rollout_id = self.rollout_id_from_run(body)
        if rollout_id is None or self._agent_execution_store() is None:
            return None
        if self.agent_execution_capture_requires_single_worker() and (self.config.num_workers or 1) > 1:
            logger.warning(
                "Agent execution capture is disabled for %s with num_workers=%s because its self-call may land "
                "in another process.",
                self.config.name,
                self.config.num_workers,
            )
            return None

        recorder = AgentExecutionRecorder(rollout_id, self.config.name)
        lock, recorders = self._agent_execution_state()
        with lock:
            if rollout_id in recorders:
                logger.warning(
                    "Agent execution capture already active for rollout %s; skipping duplicate.", rollout_id
                )
                return None
            recorders[rollout_id] = recorder
        return recorder

    async def _finish_agent_execution_capture(self, recorder: Optional[AgentExecutionRecorder]) -> None:
        if recorder is None:
            return
        try:
            store = self._agent_execution_store()
            if store is not None:
                write_task = asyncio.create_task(asyncio.to_thread(store.write, recorder.capture()))
                try:
                    await asyncio.shield(write_task)
                except asyncio.CancelledError:
                    await asyncio.gather(write_task, return_exceptions=True)
                    raise
        except Exception:
            logger.warning("Agent execution capture finalization failed.", exc_info=True)
        finally:
            lock, recorders = self._agent_execution_state()
            with lock:
                if recorders.get(recorder.rollout_id) is recorder:
                    del recorders[recorder.rollout_id]

    def _run_with_agent_execution_capture(self, run_handler):
        @wraps(run_handler)
        async def wrapped(*args, **kwargs):
            recorder = self._start_agent_execution_capture(kwargs.get("body"))
            try:
                result = await run_handler(*args, **kwargs)
            except asyncio.CancelledError:
                if recorder is not None:
                    recorder.set_invocation_status(recorder.ROOT_INVOCATION_ID, "cancelled")
                raise
            except BaseException:
                if recorder is not None:
                    recorder.set_invocation_status(recorder.ROOT_INVOCATION_ID, "failed")
                raise
            else:
                if recorder is not None:
                    recorder.set_invocation_status(recorder.ROOT_INVOCATION_ID, "completed")
                return result
            finally:
                await self._finish_agent_execution_capture(recorder)

        return wrapped

    def agent_execution_recorder_for_request(self, request: Optional[Request]) -> Optional[AgentExecutionRecorder]:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        if rollout_id is None:
            return None
        lock, recorders = self._agent_execution_state()
        with lock:
            return recorders.get(rollout_id)

    def agent_execution_recorder_for_run(self, body: Any) -> Optional[AgentExecutionRecorder]:
        rollout_id = self.rollout_id_from_run(body)
        if rollout_id is None:
            return None
        lock, recorders = self._agent_execution_state()
        with lock:
            return recorders.get(rollout_id)

    def _model_call_capture_enabled(self) -> bool:
        # Fail closed: an agent whose client carries no usable global config runs uncorrelated
        # rather than erroring on every model call.
        global_config = getattr(self.server_client, "global_config_dict", None)
        if not isinstance(global_config, Mapping):
            return False
        return bool(global_config.get(OBSERVABILITY_ENABLED_KEY_NAME, False))

    def rollout_id_from_run(self, body: Any) -> Optional[str]:
        """Per-rollout capture id for a run-request (its task/rollout indices).

        None when model-call capture (observability) is disabled or the body carries no indices,
        so callers apply no correlation prefix in either case.
        """
        if not self._model_call_capture_enabled():
            return None
        return maybe_rollout_id_from_run_body(body)

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
        return apply_rollout_prefix(base_url, self.rollout_id_from_run(body))

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
