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
import functools
import hashlib
import importlib
import inspect
import logging
import math
import sys
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from time import monotonic
from typing import Annotated, Any, ClassVar, Dict, Optional, Union, cast

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, BeforeValidator, Field, PrivateAttr

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import SESSION_EXPIRED, SESSION_LOST
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    MCPTool,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionLostError,
    SimpleResourcesServer,
    lost_session_verification,
    reject_partial_typed_seed,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import JudgeError
from resources_servers.newton_bench.newton_bench_utils.sandbox import (
    SessionHandle,
    validate_python_code,
)
from resources_servers.newton_bench.newton_bench_utils.schemas import MODULE_REQUEST_CLASSES_MAPPING
from resources_servers.newton_bench.setup_newton_bench import NEWTON_BENCH_PATH, ensure_newton_bench


if str(NEWTON_BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(NEWTON_BENCH_PATH))

_loaded_modules: dict = {}

# NewtonBench's name for the model that judges symbolic equivalence in every module's evaluate_law.
JUDGE_MODEL_NAME = "gpt41"

# Each judge API attempt of the verification in progress: None for a reply, else the error the attempt raised.
# evaluate_law must run in the verification's context (directly or through asyncio.to_thread) to record them.
_judge_attempts: ContextVar[Optional[list[Optional[Exception]]]] = ContextVar(
    "newton_bench_judge_attempts", default=None
)


def _require_judge_key() -> None:
    """Refuse to serve when NewtonBench has no API key for its symbolic judge.

    Every module's evaluate_law runs the judge, so without a key no verification can be scored. NewtonBench reads
    the keys once, from the environment and its ``.env`` file, when ``utils.call_llm_api`` is first imported.
    """
    llm_api = importlib.import_module("utils.call_llm_api")
    try:
        llm_api.resolve_model_and_source(JUDGE_MODEL_NAME, llm_api.keys)
    except ValueError as error:
        raise ValueError(
            f"NewtonBench's symbolic judge ({JUDGE_MODEL_NAME}) has no API key. Set OPENAI_API_KEY or "
            f"OPENROUTER_API_KEY in the environment or in {NEWTON_BENCH_PATH / '.env'}, then restart the server."
        ) from error


def _record_judge_attempts() -> None:
    """Report each API attempt of NewtonBench's symbolic judge to the verification in progress.

    The judge catches the error of every failed API call and, out of attempts, returns False: the same value as a
    "not equivalent" verdict. Wrapping the call it makes is how verify tells the two apart.
    """
    evaluation = importlib.import_module("modules.common.evaluation")
    call_llm_api = evaluation.call_llm_api
    if getattr(call_llm_api, "records_judge_attempts", False):
        return

    @functools.wraps(call_llm_api)
    def recorded_call_llm_api(*args, **kwargs):
        attempts = _judge_attempts.get()
        try:
            reply = call_llm_api(*args, **kwargs)
        except Exception as error:
            if attempts is not None:
                attempts.append(error)
            raise
        if attempts is not None:
            attempts.append(None)
        return reply

    recorded_call_llm_api.records_judge_attempts = True
    evaluation.call_llm_api = recorded_call_llm_api


def _evaluate_law(core: Any, **kwargs: Any) -> dict:
    """Run a module's evaluate_law, raising a JudgeError when its symbolic judge gave no verdict for an API failure."""
    attempts: list[Optional[Exception]] = []
    token = _judge_attempts.set(attempts)
    try:
        eval_result = core.evaluate_law(**kwargs)
    finally:
        _judge_attempts.reset(token)
    # The judge returns as soon as a reply carries a verdict, so when its last attempt raised, the False it returned
    # means it ran out of attempts, not that the laws differ.
    if attempts and attempts[-1] is not None:
        error = attempts[-1]
        failed = sum(attempt is not None for attempt in attempts)
        raise JudgeError(
            f"NewtonBench symbolic judge ({JUDGE_MODEL_NAME}) gave no verdict: {failed} of {len(attempts)} API "
            f"attempts failed, the last with {type(error).__name__}: {error}"
        ) from error
    return eval_result


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
    _record_judge_attempts()
    param_description = None
    try:
        prompts_mod = importlib.import_module(f"modules.{module_name}.prompts")
        param_description = getattr(prompts_mod, "PARAM_DESCRIPTION", None)
    except Exception:
        param_description = None
    _loaded_modules[module_name] = {"core": core, "param_description": param_description}
    return _loaded_modules[module_name]


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
    seed_request_model: ClassVar[type[BaseSeedSessionRequest]] = NewtonBenchSeedSessionRequest
    # Keyed by the session cookie's SESSION_ID_KEY, which a typed seed binds to its resources_session_id.
    session_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    _sessions: Dict[str, SessionHandle] = PrivateAttr(default_factory=dict)
    # Sessions the idle sweep closed after session_ttl, so a late call reports them expired rather than lost.
    _expired_sessions: dict[str, float] = PrivateAttr(default_factory=dict)
    _starting_sessions: dict[str, asyncio.Task[SessionHandle]] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            cleanup_task = asyncio.create_task(self._background_cleanup_task())
            try:
                async with parent_lifespan(app):
                    yield
            finally:
                cleanup_task.cancel()
                try:
                    await cleanup_task
                except asyncio.CancelledError:
                    pass

                failures = []
                for sid in set(self.session_metadata) | set(self._sessions) | set(self._starting_sessions):
                    try:
                        async with self.session_lock(sid):
                            await self.release_session_state(sid)
                    except Exception as error:
                        failures.append(error)
                if failures:
                    raise ExceptionGroup("NewtonBench shutdown could not release all sessions", failures)

        app.router.lifespan_context = lifespan

        ensure_newton_bench()
        _require_judge_key()

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

    def mcp_tools(self, harvested: list[MCPTool], catchall: Optional[Any]) -> Optional[list[MCPTool]]:
        """Expose the experiment and Python tools; /end_session is a client cleanup route, not a model tool."""
        return [tool for tool in harvested if tool.name != "end_session"]

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self.session_metadata or session_id in self._starting_sessions

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        """Build task metadata after the base has released any preceding cookie run."""
        task = NewtonBenchSeedSessionRequest.model_validate(body)
        if task.module_name not in MODULE_REQUEST_CLASSES_MAPPING:
            raise HTTPException(status_code=400, detail=f"Invalid module_name '{task.module_name}'.")
        metadata = self._session_metadata(task)
        self.session_metadata[session_id] = metadata
        self._expired_sessions.pop(session_id, None)

    async def release_session_state(self, session_id: str) -> None:
        """Release the Python worker and metadata for close or checkpoint retirement."""
        await self._release_worker(session_id)
        self.session_metadata.pop(session_id, None)

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[NewtonBenchSeedSessionRequest, BeforeValidator(reject_partial_typed_seed)],
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        """Use the shared seed policy, checking that a repeated typed seed still has live metadata.

        The idle sweep can release metadata before the episode closes. The shared lifecycle does not revalidate an
        existing binding, so this adapter preserves the expired/lost response for that retry.
        Cookie seeds use the base lifecycle and keep their session ID when replacing a completed run.
        """
        try:
            response = await super().seed_session(body, request=request)
        except HTTPException as error:
            # Cookie callers historically receive 400 for an unknown module; typed task rejection is 422.
            if isinstance(body, ResourcesSeedSessionRequest) and error.status_code == 400:
                raise ResourcesSeedRejectedError(str(error.detail)) from error
            raise
        if isinstance(body, ResourcesSeedSessionRequest):
            self._require_metadata(body.resources_session_id)
        return response

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
        now = monotonic()
        self._expired_sessions = {sid: expiry for sid, expiry in self._expired_sessions.items() if expiry > now}
        return SESSION_EXPIRED if session_id in self._expired_sessions else SESSION_LOST

    def _require_metadata(self, session_id: Optional[str]) -> Dict[str, Any]:
        """The metadata of the session a tool call names, or a 404 if it was never seeded or has already ended."""
        metadata = self.session_metadata.get(session_id)
        if (
            session_id is None
            or not metadata
            or self.session_is_closed(session_id)
            or self.session_is_verified(session_id)
        ):
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

            # One start per session at a time, so concurrent first calls share a worker.
            async with self.session_lock(sid):
                if self._require_metadata(sid) is not metadata:
                    raise ResourcesSessionLostError("The session ended before its Python worker could start.")
                if sid not in self._sessions:
                    # Keep the thread's result reachable even if this request is cancelled before publication.
                    starting = self._starting_sessions.get(sid)
                    if starting is None:
                        starting = asyncio.create_task(
                            asyncio.to_thread(SessionHandle, self.config.max_execution_time)
                        )
                        self._starting_sessions[sid] = starting
                    try:
                        handle = await asyncio.shield(starting)
                    except Exception:
                        self._starting_sessions.pop(sid, None)
                        raise
                    self._sessions[sid] = handle
                    self._starting_sessions.pop(sid, None)
                    if self.session_metadata.get(sid) is not metadata:
                        await self._release_worker(sid)
                        return ExecutePythonResponse(
                            success=False,
                            stdout="",
                            stderr="",
                            error_message="The session ended while its Python worker started.",
                        )
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

        except ResourcesSessionLostError:
            raise
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
        async with self.verifying(session_id) as verification:
            retained = cast(tuple[str, dict[str, Any]] | None, verification.retained)
            if verification.is_lost or (retained is not None and retained[0] != digest):
                return NewtonBenchVerifyResponse(
                    **(
                        body.model_dump()
                        | lost_session_verification(
                            "No active NewtonBench session for this verification",
                            failure_kind=self._lost_failure_kind(session_id),
                        )
                    )
                )
            if retained is not None:
                return NewtonBenchVerifyResponse(**body.model_dump(), **retained[1])
            result = await self._verify_session(request, body)
            # Cache measured fields so retrying the same answer never reruns a stochastic evaluation.
            verification.retain((digest, result.model_dump(exclude=set(body.model_dump()))))
            return result

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

            eval_result = _evaluate_law(
                core,
                llm_function_str=extracted_law,
                param_description=param_description,
                difficulty=difficulty,
                law_version=law_version,
                judge_model_name=JUDGE_MODEL_NAME,
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

        except JudgeError:
            # An unreachable judge is not a wrong answer: judge_failsafe masks the sample instead of scoring it.
            raise
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
        """Stop the session's Python worker; a later execute_python starts a fresh one.

        The task metadata stays until verify, close, checkpoint retirement or the idle sweep: a model can call this
        route as a tool, and releasing the metadata would mask its own verification.
        """
        sid = request.session[SESSION_ID_KEY]
        async with self.session_lock(sid):
            await self._release_worker(sid)
        return NewtonBenchEndSessionResponse()

    async def _release_worker(self, sid: str) -> None:
        """Wait for a cancelled caller's startup and retain the worker until close succeeds."""
        starting = self._starting_sessions.get(sid)
        if starting is not None:
            try:
                handle = await asyncio.shield(starting)
            except Exception:
                # The constructor did not produce a handle; a later tool call may try construction again.
                self._starting_sessions.pop(sid, None)
            else:
                self._sessions[sid] = handle
                self._starting_sessions.pop(sid, None)
        handle = self._sessions.get(sid)
        if handle is not None:
            # close waits for the process and may contend with an executor thread already closing it.
            # Keep the handle owned if this await is cancelled; a retry's close serializes on the handle's lock.
            await handle.aclose()
            self._sessions.pop(sid, None)

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
                await self._cleanup_sessions()
            except Exception:
                logging.exception("Error in background cleanup task")
            await asyncio.sleep(600)  # Check every 10 minutes

    async def _cleanup_sessions(self) -> None:
        """Release idle sessions under the same lock as seed, verification, and close."""
        now = time.time()
        failures = []
        for sid in list(self.session_metadata):
            try:
                async with self.session_lock(sid):
                    metadata = self.session_metadata.get(sid)
                    if metadata is not None and now - metadata.get("last_used", 0) > self.config.session_ttl:
                        await self.release_session_state(sid)
                        self._expired_sessions[sid] = monotonic() + max(
                            self.config.session_close_retry_window_seconds, self.config.session_ttl
                        )
            except Exception as error:
                failures.append(error)
        self._lost_failure_kind(None)  # Expire bounded diagnostic records even if no late request arrives.
        if failures:
            raise ExceptionGroup("NewtonBench idle sweep could not release all expired sessions", failures)

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
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = NewtonBenchResourcesServer.run_webserver()  # noqa: F401
