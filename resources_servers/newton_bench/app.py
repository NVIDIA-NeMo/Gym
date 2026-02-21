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
import importlib
import inspect
import logging
import math
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.newton_bench.newton_bench_utils.sandbox import (
    SessionHandle,
    validate_python_code,
)
from resources_servers.newton_bench.newton_bench_utils.schemas import MODULE_REQUEST_CLASSES_MAPPING
from resources_servers.newton_bench.setup_newton_bench import ensure_newton_bench


REPO_ROOT = Path(__file__).parent.parent.parent
NEWTON_BENCH_PATH = REPO_ROOT / "NewtonBench"
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
    session_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    _sessions: Dict[str, SessionHandle] = PrivateAttr(default_factory=dict)

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

    async def seed_session(self, request: Request, body: NewtonBenchSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        module_name = body.module_name

        if module_name not in MODULE_REQUEST_CLASSES_MAPPING:
            raise HTTPException(status_code=400, detail=f"Invalid module_name '{module_name}'.")

        self.session_metadata[session_id] = {
            "module_name": module_name,
            "difficulty": body.difficulty,
            "system": body.system,
            "noise_level": body.noise_level,
            "law_version": body.law_version,
            "last_used": time.time(),
        }
        return BaseSeedSessionResponse()

    async def execute_python(self, request: Request, body: ExecutePythonRequest) -> ExecutePythonResponse:
        """Execute Python code in a session-based environment."""
        sid = request.session[SESSION_ID_KEY]
        metadata = self.session_metadata.get(sid)
        if not metadata:
            raise HTTPException(status_code=400, detail="Session not initialized. Please call seed_session first.")

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
        session_id = request.session[SESSION_ID_KEY]
        metadata = self.session_metadata.get(session_id)
        if not metadata:
            raise HTTPException(status_code=400, detail="Session not initialized. Please call seed_session first.")

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

            eval_result = core.evaluate_law(
                llm_function_str=extracted_law,
                param_description=param_description,
                difficulty=difficulty,
                law_version=law_version,
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
        finally:
            self._close_session(session_id)

    async def end_session(self, request: Request, body: NewtonBenchEndSessionRequest) -> NewtonBenchEndSessionResponse:
        """Clean up session handle for Python execution and metadata."""
        sid = request.session[SESSION_ID_KEY]
        self._close_session(sid)
        return NewtonBenchEndSessionResponse()

    def _close_session(self, sid: str):
        """Internal helper to clean up session handle and metadata."""
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
            metadata = self.session_metadata.get(session_id)
            if not metadata:
                raise HTTPException(status_code=400, detail="Session not initialized. Please call seed_session first.")

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
