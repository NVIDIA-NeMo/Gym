# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import hashlib
import importlib
import inspect
import logging
import math
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, BeforeValidator, Field, PrivateAttr, ValidationError

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import SESSION_EXPIRED, SESSION_LOST, EpisodeId, TaskId
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ClosedResourcesSessionRecords,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFence,
    ResourcesSessionLostError,
    SimpleResourcesServer,
    lost_session_verification,
)
from resources_servers.newton_bench.newton_bench_utils.sandbox import (
    SessionHandle,
    validate_python_code,
)
from resources_servers.newton_bench.newton_bench_utils.schemas import MODULE_REQUEST_CLASSES_MAPPING
from resources_servers.newton_bench.setup_newton_bench import NEWTON_BENCH_PATH, ensure_newton_bench


if str(NEWTON_BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(NEWTON_BENCH_PATH))

_loaded_modules: dict = {}


def _json_safe_result(x: Any) -> Any:
    """Replace nan/inf with None so response is JSON-serializable."""
    if x is None:
        return None
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, dict):
        return {k: _json_safe_result(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_json_safe_result(v) for v in x)
    return x


def _load_module(module_name: str):
    if not module_name:
        raise ImportError("No module_name provided to _load_module")
    if module_name in _loaded_modules:
        return _loaded_modules[module_name]
    try:
        core = importlib.import_module(f"modules.{module_name}.core")
    except Exception as e:
        raise ImportError(f"Unable to import core for NewtonBench module '{module_name}': {e}") from e
    param_description = None
    try:
        prompts_mod = importlib.import_module(f"modules.{module_name}.prompts")
        param_description = getattr(prompts_mod, "PARAM_DESCRIPTION", None)
    except Exception:
        param_description = None
    _loaded_modules[module_name] = {"core": core, "param_description": param_description}
    return _loaded_modules[module_name]


def _reject_typed_seed(value: object) -> object:
    # A body that names a resources_session_id, or carries typed episode_id and task_id objects, is a malformed typed
    # seed and must not create a /run session. A dataset row's own scalar task_id does not make a body typed.
    if isinstance(value, dict) and (
        "resources_session_id" in value
        or (isinstance(value.get("episode_id"), dict) and isinstance(value.get("task_id"), dict))
    ):
        raise ValueError("A typed seed must supply a valid resources session, episode, task and task_data")
    return value


class NewtonBenchResourcesServerConfig(BaseResourcesServerConfig):
    max_execution_time: int = 60  # 1 minute
    session_ttl: int = 1800  # 30 minutes


class RunExperimentResponse(BaseModel):
    result: Optional[Union[float, dict]]  # float for vanilla_equation, dict for systems; None if nan


class ExecutePythonRequest(BaseModel):
    code: str


class ExecutePythonResponse(BaseModel):
    success: bool
    stdout: str
    stderr: str
    error_message: Optional[str] = None
    result: Optional[str] = None


class NewtonBenchRunRequest(BaseRunRequest):
    difficulty: str
    system: str
    noise_level: float


class NewtonBenchSeedSessionRequest(BaseSeedSessionRequest):
    module_name: str
    difficulty: str
    system: str
    law_version: str
    noise_level: float


class NewtonBenchVerifyRequest(BaseVerifyRequest):
    pass


class NewtonBenchVerifyResponse(BaseVerifyResponse):
    difficulty: Optional[str] = None
    system: Optional[str] = None
    noise_level: Optional[float] = None
    law_version: Optional[str] = None
    extracted_law: Optional[str] = None
    rmsle: Optional[float] = None
    exact_accuracy: Optional[float] = None
    symbolic_equivalent: Optional[bool] = None
    evaluation_error: Optional[str] = None


class NewtonBenchEndSessionRequest(BaseModel):
    pass


class NewtonBenchEndSessionResponse(BaseModel):
    pass


class NewtonBenchResourcesServer(SimpleResourcesServer):
    config: NewtonBenchResourcesServerConfig
    # Keyed by the session cookie's SESSION_ID_KEY, which a typed seed binds to its resources_session_id.
    session_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    _sessions: Dict[str, SessionHandle] = PrivateAttr(default_factory=dict)
    # The episode and task each Environment Server session was seeded for, keyed by resources_session_id.
    _session_episodes: Dict[str, tuple[EpisodeId, TaskId]] = PrivateAttr(default_factory=dict)
    # Sessions the idle sweep closed after session_ttl, so a late call reports them expired rather than lost.
    _expired_sessions: ClosedResourcesSessionRecords[None] = PrivateAttr()
    _verified_sessions: ClosedResourcesSessionRecords[
        tuple[Optional[tuple[EpisodeId, TaskId]], str, dict[str, Any]]
    ] = PrivateAttr()
    _fence: ResourcesSessionFence = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._verified_sessions = ClosedResourcesSessionRecords(self.config.session_close_retry_window_seconds)
        self._fence = ResourcesSessionFence(self.config.session_close_retry_window_seconds)
        # A client idle long enough to be swept can stay away about as long again, often longer than a close retry.
        self._expired_sessions = ClosedResourcesSessionRecords(
            max(self.config.session_close_retry_window_seconds, self.config.session_ttl)
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            cleanup_task = asyncio.create_task(self._background_cleanup_task())

            async with parent_lifespan(app):
                yield

            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass

            for sid in list(self._sessions.keys()):
                handle = self._sessions.pop(sid, None)
                if handle:
                    handle.close()

        app.router.lifespan_context = lifespan

        ensure_newton_bench()
        key_found = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key_found:
            logging.warning(
                "No API key found for evaluation. Please set OPENROUTER_API_KEY or OPENAI_API_KEY in the environment variables and restart the server."
            )

        modules_dir = NEWTON_BENCH_PATH / "modules"
        try:
            if modules_dir.exists() and modules_dir.is_dir():
                for child in sorted(modules_dir.iterdir()):
                    if not child.is_dir():
                        continue
                    module_name = child.name
                    if module_name == "common":
                        continue

                    route_path = f"/run_experiment_{module_name}"
                    app.add_api_route(route_path, self._create_module_handler(module_name), methods=["POST"])
        except Exception:
            logging.exception("Failed to dynamically register module endpoints")

        app.post("/execute_python")(self.execute_python)
        app.post("/end_session")(self.end_session)

        return app

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[NewtonBenchSeedSessionRequest, BeforeValidator(_reject_typed_seed)],
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        """Store the task's metadata for the session; the Python worker starts on the first execute_python call.

        An Agent's /run seeds with the row's fields under the session its cookie names. An Environment Server seeds
        with a ResourcesSeedSessionRequest whose task_data holds those fields: the session is keyed by
        resources_session_id and the cookie is bound to that ID, so the agent's tool calls reach it.
        """
        if not isinstance(body, ResourcesSeedSessionRequest):
            if body.module_name not in MODULE_REQUEST_CLASSES_MAPPING:
                raise HTTPException(status_code=400, detail=f"Invalid module_name '{body.module_name}'.")
            session_id = request.session[SESSION_ID_KEY]
            async with self._fence.locked(session_id):
                self._fence.raise_if_closed(session_id)
                verified = self._verified_sessions.get(session_id)
                if session_id in self._session_episodes or (verified is not None and verified[0] is not None):
                    raise ResourcesSessionConflictError("A legacy seed cannot replace a typed session")
                # A new /run may reuse its caller's cookie: it keeps the ID and replaces the previous run's state,
                # including that run's verify retry record and expiry label.
                metadata = self._session_metadata(body)
                self._close_session(session_id)
                self.session_metadata[session_id] = metadata
                self._verified_sessions.forget(session_id)
                self._expired_sessions.forget(session_id)
            return BaseSeedSessionResponse()

        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            self._fence.raise_if_closed(session_id)
            if session_id in self._verified_sessions:
                raise ResourcesSessionConflictError("resources session has already been verified")
            identity = self._session_episodes.get(session_id)
            if identity is None:
                if session_id in self.session_metadata:
                    raise ResourcesSessionConflictError("A typed seed cannot replace a legacy session")
                # Input no retry can fix gets a final 422 rather than a 500 that would dispatch the episode again.
                try:
                    task = NewtonBenchSeedSessionRequest.model_validate(body.task_data)
                except ValidationError as error:
                    raise ResourcesSeedRejectedError(f"task_data is not a NewtonBench task: {error}") from error
                if task.module_name not in MODULE_REQUEST_CLASSES_MAPPING:
                    raise ResourcesSeedRejectedError(f"Invalid module_name '{task.module_name}'.")
                self.session_metadata[session_id] = self._session_metadata(task)
                self._session_episodes[session_id] = (body.episode_id, body.task_id)
            elif identity != (body.episode_id, body.task_id):
                raise ResourcesSessionConflictError("resources_session_id is already bound to another episode or task")
            else:
                # An idle sweep or /end_session may already have released the state. Never acknowledge a retry
                # as successfully seeded when its first tool call can only report a lost session.
                self._require_metadata(session_id)
            # A repeated seed for the same episode keeps the session, including its Python worker's state.
            request.session[SESSION_ID_KEY] = session_id
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Release the session's Python worker and metadata, if verify or the idle sweep has not already done so.

        Keyed by resources_session_id, so a close without the cookie (the seed's reply was lost) still releases it.
        Closing an unknown or already-closed session succeeds, and the fence refuses a later seed of it.
        """
        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            closed_episode = self._fence.closed_episode(session_id)
            if closed_episode is not None and closed_episode != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
            verified = self._verified_sessions.get(session_id)
            identity = self._session_episodes.get(session_id) or (verified[0] if verified is not None else None)
            if identity is not None and identity[0] != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
            self._close_session(session_id)
            self._session_episodes.pop(session_id, None)
            self._fence.mark_closed(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def retire_session_state(self, session_id: str) -> None:
        """Release the Python worker and metadata of a retired checkpoint attempt."""
        async with self._fence.locked(session_id):
            self._close_session(session_id)
            identity = self._session_episodes.pop(session_id, None)
            if identity is not None:
                self._fence.mark_closed(session_id, identity[0])

    @staticmethod
    def _session_metadata(task: NewtonBenchSeedSessionRequest) -> Dict[str, Any]:
        return {
            "module_name": task.module_name,
            "difficulty": task.difficulty,
            "system": task.system,
            "noise_level": task.noise_level,
            "law_version": task.law_version,
            "last_used": time.time(),
        }

    def _lost_failure_kind(self, session_id: Optional[str]) -> str:
        return SESSION_EXPIRED if session_id in self._expired_sessions else SESSION_LOST

    def _require_metadata(self, session_id: Optional[str]) -> Dict[str, Any]:
        """The metadata of the session a tool call names, or a 404 if it was never seeded or has already ended."""
        metadata = self.session_metadata.get(session_id)
        if not metadata:
            raise ResourcesSessionLostError(
                "Session not initialized or already ended. Please call seed_session first.",
                failure_kind=self._lost_failure_kind(session_id),
            )
        return metadata

    async def execute_python(self, request: Request, body: ExecutePythonRequest) -> ExecutePythonResponse:
        """Execute Python code in a session-based environment."""
        sid = request.session[SESSION_ID_KEY]
        metadata = self._require_metadata(sid)

        metadata["last_used"] = time.time()
        loop = asyncio.get_running_loop()
        try:
            is_valid, error_message = validate_python_code(body.code)
            if not is_valid:
                return ExecutePythonResponse(
                    success=False,
                    stdout="",
                    stderr="",
                    error_message=error_message,
                )

            if sid in self._sessions and self._sessions[sid].is_closed:
                self._sessions.pop(sid, None)

            if sid not in self._sessions:
                self._sessions[sid] = SessionHandle(self.config.max_execution_time)
            handle = self._sessions[sid]

            try:
                stdout, stderr, result = await loop.run_in_executor(
                    None,
                    handle.exec,
                    body.code,
                )
                return ExecutePythonResponse(
                    success=True,
                    stdout=stdout,
                    stderr=stderr,
                    result=result,
                )
            except Exception as e:
                if sid in self._sessions and self._sessions[sid].is_closed:
                    self._sessions.pop(sid, None)
                raise e

        except Exception as e:
            return ExecutePythonResponse(
                success=False,
                stdout="",
                stderr="",
                error_message=str(e),
            )

    async def verify(self, request: Request, body: NewtonBenchVerifyRequest) -> NewtonBenchVerifyResponse:
        """Release live state after evaluation and replay only the same completed verification payload."""
        session_id = request.session[SESSION_ID_KEY]
        digest = hashlib.sha256(body.model_dump_json().encode()).hexdigest()
        # Nothing below yields to the event loop (evaluate_law is synchronous), so a seed, close or resent verify for
        # this session cannot interleave between the checks and the release. Hold self._fence.locked(session_id)
        # across them if evaluation ever awaits.
        verified = self._verified_sessions.get(session_id)
        if self._fence.closed_episode(session_id) is not None or (verified is not None and verified[1] != digest):
            return NewtonBenchVerifyResponse(
                **(
                    body.model_dump()
                    | lost_session_verification("No active NewtonBench session for this verification")
                )
            )
        if verified is not None:
            return NewtonBenchVerifyResponse(**body.model_dump(), **verified[2])
        if session_id not in self.session_metadata:
            return await self._verify_session(request, body)
        try:
            result = await self._verify_session(request, body)
            # Retain only measured fields, never the prompt, trajectory or live Python worker.
            outcome = result.model_dump(exclude=set(body.model_dump()))
            self._verified_sessions.remember(
                session_id, (self._session_episodes.pop(session_id, None), digest, outcome)
            )
            return result
        finally:
            self._close_session(session_id)

    async def _verify_session(self, request: Request, body: NewtonBenchVerifyRequest) -> NewtonBenchVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        metadata = self.session_metadata.get(session_id)
        if not metadata:
            # Never seeded, already verified or closed, or reaped by the idle sweep: the task's metadata is gone, so
            # there is nothing to score against. Mask the sample rather than score the policy 0.
            failure_kind = self._lost_failure_kind(session_id)
            reason = (
                "The NewtonBench session outlived session_ttl"
                if failure_kind == SESSION_EXPIRED
                else "No active NewtonBench session to verify"
            )
            return NewtonBenchVerifyResponse(
                **(body.model_dump() | lost_session_verification(reason, failure_kind=failure_kind))
            )

        metadata["last_used"] = time.time()
        difficulty = metadata.get("difficulty")
        system = metadata.get("system")
        noise_level = metadata.get("noise_level")
        law_version = metadata.get("law_version")

        try:
            key_found = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not key_found:
                logging.warning(
                    "No API key found for evaluation. Please set OPENROUTER_API_KEY or OPENAI_API_KEY in the environment variables and restart the server."
                )

            extracted_law = self._extract_law_from_response(body.response)

            if extracted_law is None:
                return NewtonBenchVerifyResponse(
                    **body.model_dump(),
                    difficulty=difficulty,
                    system=system,
                    noise_level=noise_level,
                    law_version=law_version,
                    reward=0.0,
                    extracted_law=None,
                    evaluation_error="No law found in response. Expected <final_law>...</final_law> tags.",
                )

            # Use NewtonBench eval func
            module_name = metadata.get("module_name")
            if not module_name:
                return NewtonBenchVerifyResponse(
                    **body.model_dump(),
                    difficulty=difficulty,
                    system=system,
                    noise_level=noise_level,
                    law_version=law_version,
                    reward=0.0,
                    extracted_law=extracted_law,
                    evaluation_error="Missing module_name in configuration.",
                )

            _mod = _load_module(module_name)
            core = _mod["core"]
            param_description = _mod.get("param_description", None)

            eval_result = core.evaluate_law(
                llm_function_str=extracted_law,
                param_description=param_description,
                difficulty=difficulty,
                law_version=law_version,
                judge_model_name="gpt41",
            )

            # Symbolic equivalence uses LLM judge
            symbolic_equivalent_raw = eval_result.get("symbolic_equivalent")
            is_symbolically_correct = symbolic_equivalent_raw is True
            symbolic_missing = symbolic_equivalent_raw is None

            rmsle_raw = eval_result.get("rmsle")
            rmsle_is_missing = rmsle_raw is None or (
                isinstance(rmsle_raw, float) and (math.isnan(rmsle_raw) or math.isinf(rmsle_raw))
            )
            rmsle = rmsle_raw if not rmsle_is_missing else None

            # Reward: weight 0.3 symbolic, 0.7 RMSLE. Edge cases when RMSLE is missing.
            if rmsle_is_missing:
                if symbolic_missing:
                    reward = 0.0
                elif is_symbolically_correct:
                    reward = 1.0
                else:
                    reward = -0.3
            else:
                # reward = 0.3 * R_symbolic + 0.7 * R_RMSLE
                # R_symbolic: +1 if equivalent, -1 otherwise
                R_symbolic = 1.0 if is_symbolically_correct else -1.0
                # R_RMSLE = 1 - 2*x/(x+3), x = RMSLE; range (-1, 1]
                x = float(rmsle)
                R_rmsle = 1.0 - (2.0 * x / (x + 3.0))
                reward = 0.3 * R_symbolic + 0.7 * R_rmsle

            # JSON does not allow nan; coerce to None or a finite float for response
            def _json_safe_float(x: Optional[float], default: Optional[float] = None) -> Optional[float]:
                if x is None:
                    return default
                if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                    return default
                return x

            return NewtonBenchVerifyResponse(
                **body.model_dump(),
                difficulty=difficulty,
                system=system,
                noise_level=noise_level,
                law_version=law_version,
                reward=reward if math.isfinite(reward) else 0.0,
                extracted_law=extracted_law,
                rmsle=_json_safe_float(eval_result.get("rmsle")),
                exact_accuracy=_json_safe_float(eval_result.get("exact_accuracy")),
                symbolic_equivalent=eval_result.get("symbolic_equivalent"),
                evaluation_error=eval_result.get("error"),
            )

        except Exception as e:
            return NewtonBenchVerifyResponse(
                **body.model_dump(),
                difficulty=difficulty,
                system=system,
                noise_level=noise_level,
                law_version=law_version,
                reward=0.0,
                extracted_law=extracted_law,
                evaluation_error=f"Evaluation failed: {str(e)}",
            )

    async def end_session(self, request: Request, body: NewtonBenchEndSessionRequest) -> NewtonBenchEndSessionResponse:
        """Clean up session handle for Python execution and metadata."""
        sid = request.session[SESSION_ID_KEY]
        self._close_session(sid)
        return NewtonBenchEndSessionResponse()

    def _close_session(self, sid: str):
        """Internal helper to clean up session handle and metadata.

        SessionHandle.close() escalates to kill and logs rather than raises, so releasing a session cannot fail.
        """
        handle = self._sessions.pop(sid, None)
        if handle:
            handle.close()
        self.session_metadata.pop(sid, None)

    def _create_module_handler(self, module_name: str):
        model_cls = MODULE_REQUEST_CLASSES_MAPPING.get(module_name)
        if model_cls is None:
            raise RuntimeError(f"Missing request class for NewtonBench module '{module_name}'")

        async def handler(request: Request, body: Any):
            session_id = request.session.get(SESSION_ID_KEY)
            metadata = self._require_metadata(session_id)

            session_module_name = metadata.get("module_name")
            if session_module_name != module_name:
                raise HTTPException(
                    status_code=400,
                    detail=f"Session configured for '{session_module_name}', but received run_experiment call for '{module_name}'.",
                )

            metadata["last_used"] = time.time()
            difficulty = metadata.get("difficulty")
            system = metadata.get("system")
            noise_level = metadata.get("noise_level")
            law_version = metadata.get("law_version")

            try:
                body_dict = body.model_dump()
            except Exception:
                body_dict = dict(body)

            effective_kwargs = dict(body_dict or {})
            effective_kwargs.setdefault("difficulty", difficulty)
            effective_kwargs.setdefault("system", system)
            effective_kwargs.setdefault("noise_level", noise_level)
            effective_kwargs.setdefault("law_version", law_version)

            try:
                _mod = _load_module(module_name)
                core = _mod["core"]
            except ImportError as ie:
                raise HTTPException(status_code=500, detail=str(ie))

            try:
                # Use NewtonBench experiment runner
                result = core.run_experiment_for_module(**effective_kwargs)
                # JSON does not allow nan; replace nan/inf with None recursively
                return RunExperimentResponse(result=_json_safe_result(result))

            except Exception as e:
                return RunExperimentResponse(result={"error": str(e)})

        handler.__name__ = f"run_experiment_{module_name}"

        try:
            params = [
                inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Request),
                inspect.Parameter("body", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=model_cls),
            ]
            handler.__signature__ = inspect.Signature(parameters=params)
        except Exception:
            logging.debug("Failed to set dynamic signature for handler %s", handler.__name__)

        return handler

    async def _background_cleanup_task(self):
        """Periodically check and remove expired sessions."""
        while True:
            try:
                self._cleanup_sessions()
            except Exception:
                logging.exception("Error in background cleanup task")
            await asyncio.sleep(600)  # Check every 10 minutes

    def _cleanup_sessions(self):
        """Remove sessions that have been inactive for longer than session_ttl."""
        now = time.time()

        for sid in list(self.session_metadata.keys()):
            last_activity = self.session_metadata[sid].get("last_used", 0)

            if now - last_activity > self.config.session_ttl:
                self._close_session(sid)
                self._expired_sessions.remember(sid, None)

    def _extract_law_from_response(self, response: Any) -> Optional[str]:
        for output in reversed(response.output):
            if output.type == "message" and output.role == "assistant":
                text_content = ""
                for content in output.content:
                    if content.type == "output_text":
                        text_content += content.text

                start_tag = "<final_law>"
                end_tag = "</final_law>"
                start_index = text_content.rfind(start_tag)
                if start_index == -1:
                    continue
                end_index = text_content.find(end_tag, start_index)
                if end_index == -1:
                    continue

                return text_content[start_index + len(start_tag) : end_index].strip()

        return None


if __name__ == "__main__":
    NewtonBenchResourcesServer.run_webserver()
