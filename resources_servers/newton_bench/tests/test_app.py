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
import json
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from copy import deepcopy
from unittest.mock import MagicMock, patch

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pytest import MonkeyPatch, fixture, mark, raises

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSessionLostError,
    check_resources_session_contract,
    maybe_auto_expose,
)
from nemo_gym.verification import judge_failsafe
from resources_servers.newton_bench.app import (
    BaseSeedSessionResponse,
    ExecutePythonRequest,
    ExecutePythonResponse,
    NewtonBenchEndSessionRequest,
    NewtonBenchEndSessionResponse,
    NewtonBenchResourcesServer,
    NewtonBenchResourcesServerConfig,
    NewtonBenchSeedSessionRequest,
    NewtonBenchVerifyRequest,
    NewtonBenchVerifyResponse,
    RunExperimentResponse,
)
from resources_servers.newton_bench.newton_bench_utils.sandbox import SessionHandle


class TestApp:
    SERVER_NAME = "newton_bench"

    # ============================================================================
    # 1. Fixtures & Configuration
    # ============================================================================
    @fixture
    def config(self) -> NewtonBenchResourcesServerConfig:
        return NewtonBenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )

    @fixture
    def server(self, config: NewtonBenchResourcesServerConfig) -> NewtonBenchResourcesServer:
        server_mock = MagicMock(spec=ServerClient)
        return NewtonBenchResourcesServer(config=config, server_client=server_mock)

    @fixture
    def mock_request(self) -> Request:
        request = MagicMock(spec=Request)
        request.session = {SESSION_ID_KEY: "test_session_123"}
        return request

    def _make_response_with_law(self, text: str) -> NeMoGymResponse:
        """Build minimal NeMoGymResponse with message + output_text containing given text."""
        return NeMoGymResponse.model_validate(
            {
                "id": "resp_test",
                "object": "response",
                "created_at": 1756531804.0,
                "model": "test",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
            }
        )

    def test_config_defaults(self) -> None:
        """Test that config defaults are set correctly."""
        config = NewtonBenchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        assert config.max_execution_time == 60
        assert config.session_ttl == 1800

    # ============================================================================
    # 2. Session Lifecycle Tests
    # ============================================================================
    @mark.asyncio
    async def test_seed_session_success(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test successful seeding of a session."""
        seed_request = NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation", noise_level=0.0, law_version="v0"
        )

        response = await server.seed_session(mock_request, seed_request)
        assert isinstance(response, BaseSeedSessionResponse)

        session_id = mock_request.session[SESSION_ID_KEY]
        assert session_id in server.session_metadata
        metadata = server.session_metadata[session_id]
        assert metadata["module_name"] == "m0_gravity"
        assert metadata["difficulty"] == "easy"

    @mark.asyncio
    async def test_seed_session_invalid_module_name(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test seed_session with invalid module_name."""
        seed_request = NewtonBenchSeedSessionRequest(
            module_name="invalid_module_xyz",
            difficulty="easy",
            system="vanilla_equation",
            noise_level=0.0,
            law_version="v0",
        )
        with raises(HTTPException) as exc_info:
            await server.seed_session(mock_request, seed_request)
        assert exc_info.value.status_code == 400
        assert "Invalid module_name" in exc_info.value.detail

    @mark.asyncio
    async def test_end_session(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test that end_session stops the Python worker and keeps the task metadata for verify."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        request_body = ExecutePythonRequest(code="x = 5")
        response = await server.execute_python(mock_request, request_body)
        assert response.success is True

        session_id = mock_request.session[SESSION_ID_KEY]
        assert session_id in server._sessions

        end_response = await server.end_session(mock_request, NewtonBenchEndSessionRequest())

        assert isinstance(end_response, NewtonBenchEndSessionResponse)
        assert session_id not in server._sessions
        assert session_id in server.session_metadata

    @mark.asyncio
    async def test_end_session_nonexistent(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test that end_session handles non-existent sessions gracefully."""
        end_response = await server.end_session(mock_request, NewtonBenchEndSessionRequest())

        assert isinstance(end_response, NewtonBenchEndSessionResponse)

    @mark.asyncio
    async def test_end_session_idempotent(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test that end_session is idempotent and the next call runs in a fresh worker."""
        seed_request = NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation", noise_level=0.0, law_version="v0"
        )
        await server.seed_session(mock_request, seed_request)

        from resources_servers.newton_bench.app import ExecutePythonRequest

        exec_request = ExecutePythonRequest(code="a = 1\na")
        exec_resp = await server.execute_python(mock_request, exec_request)
        assert exec_resp.success is True

        end_req = NewtonBenchEndSessionRequest()
        res1 = await server.end_session(mock_request, end_req)
        assert isinstance(res1, NewtonBenchEndSessionResponse)

        res2 = await server.end_session(mock_request, end_req)
        assert isinstance(res2, NewtonBenchEndSessionResponse)

        fresh = await server.execute_python(mock_request, ExecutePythonRequest(code="a"))
        assert fresh.success is False
        assert "name 'a' is not defined" in fresh.error_message

    @mark.asyncio
    async def test_concurrent_end_session_race(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Simulate concurrent client teardown + janitor race: no exceptions, the worker stops, the task stays."""
        seed_request = NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation", noise_level=0.0, law_version="v0"
        )
        await server.seed_session(mock_request, seed_request)

        from resources_servers.newton_bench.app import ExecutePythonRequest

        exec_request = ExecutePythonRequest(code="x = 42\nx")
        exec_resp = await server.execute_python(mock_request, exec_request)
        assert exec_resp.success is True

        end_req = NewtonBenchEndSessionRequest()

        async def call_end():
            await server.end_session(mock_request, end_req)

        tasks = [asyncio.create_task(call_end()) for _ in range(20)]
        tasks.append(asyncio.create_task(server._cleanup_sessions()))

        await asyncio.gather(*tasks)

        sid = mock_request.session[SESSION_ID_KEY]
        assert sid in server.session_metadata
        assert sid not in server._sessions

    @mark.asyncio
    async def test_cleanup_sessions_expired(self, server: NewtonBenchResourcesServer) -> None:
        """Test _cleanup_sessions() removes sessions that exceed TTL."""
        req1 = MagicMock(spec=Request)
        req1.session = {SESSION_ID_KEY: "session_1"}
        req2 = MagicMock(spec=Request)
        req2.session = {SESSION_ID_KEY: "session_2"}
        await server.seed_session(
            req1,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        await server.seed_session(
            req2,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        server.session_metadata["session_1"]["last_used"] = time.time() - 2000
        server.session_metadata["session_2"]["last_used"] = time.time() - 100
        server._sessions["session_1"] = MagicMock(spec=SessionHandle)
        server._sessions["session_2"] = MagicMock(spec=SessionHandle)
        await server._cleanup_sessions()
        assert "session_1" not in server.session_metadata
        assert "session_1" not in server._sessions
        assert "session_2" in server.session_metadata
        assert "session_2" in server._sessions

    @mark.asyncio
    async def test_cleanup_sessions_closes_handles(self, server: NewtonBenchResourcesServer) -> None:
        """Test _cleanup_sessions() calls close() on SessionHandle objects."""
        req = MagicMock(spec=Request)
        req.session = {SESSION_ID_KEY: "session_1"}
        await server.seed_session(
            req,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        mock_handle = MagicMock(spec=SessionHandle)
        server._sessions["session_1"] = mock_handle
        server.session_metadata["session_1"]["last_used"] = time.time() - 2000
        await server._cleanup_sessions()
        mock_handle.aclose.assert_called_once()

    @mark.asyncio
    async def test_last_used_updated_on_activity(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test that last_used timestamp is updated on each endpoint call."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        sid = mock_request.session[SESSION_ID_KEY]
        initial_time = server.session_metadata[sid]["last_used"]
        await asyncio.sleep(0.05)
        await server.execute_python(mock_request, ExecutePythonRequest(code="x = 1"))
        updated_time = server.session_metadata[sid]["last_used"]
        assert updated_time > initial_time

    # ============================================================================
    # 3. Experiment Execution Tests
    # ============================================================================
    def test_create_module_handler_missing_request_class(self, server: NewtonBenchResourcesServer) -> None:
        """Test _create_module_handler() when module not in MODULE_REQUEST_CLASSES_MAPPING."""
        with patch("resources_servers.newton_bench.app.MODULE_REQUEST_CLASSES_MAPPING", {}):
            with raises(RuntimeError) as exc_info:
                server._create_module_handler("nonexistent_module")
            assert "Missing request class" in str(exc_info.value)

    @mark.asyncio
    async def test_create_module_handler_import_error(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test dynamic handler when module import fails."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        handler = server._create_module_handler("m0_gravity")
        with patch("resources_servers.newton_bench.app._load_module") as mock_load:
            mock_load.side_effect = ImportError("Module not found")
            with raises(HTTPException) as exc_info:
                await handler(mock_request, {"mass1": 10.0, "mass2": 20.0, "distance": 5.0})
            assert exc_info.value.status_code == 500
            assert "Module not found" in exc_info.value.detail

    @mark.asyncio
    async def test_run_experiment_success(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test that the dynamic handler correctly merges session metadata when it calls the core module."""
        seed_request = NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="hard", system="complex_system", noise_level=0.5, law_version="v1"
        )
        await server.seed_session(mock_request, seed_request)

        mock_core = MagicMock()
        mock_core.run_experiment_for_module.return_value = {"mocked_key": "mocked_value"}

        with patch("resources_servers.newton_bench.app._load_module") as mock_load:
            mock_load.return_value = {"core": mock_core, "param_description": None}

            handler = server._create_module_handler("m0_gravity")
            run_body = {"mass1": 10.0, "mass2": 20.0, "distance": 5.0}

            response = await handler(mock_request, run_body)

            assert isinstance(response, RunExperimentResponse)
            assert response.result == {"mocked_key": "mocked_value"}

            mock_core.run_experiment_for_module.assert_called_once()
            called_kwargs = mock_core.run_experiment_for_module.call_args.kwargs

            # Check user-provided values
            assert called_kwargs["mass1"] == 10.0
            assert called_kwargs["mass2"] == 20.0
            assert called_kwargs["distance"] == 5.0
            # Check session-injected metadata
            assert called_kwargs["difficulty"] == "hard"
            assert called_kwargs["system"] == "complex_system"
            assert called_kwargs["noise_level"] == 0.5
            assert called_kwargs["law_version"] == "v1"

    @mark.asyncio
    async def test_run_experiment_wrong_module(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test that calling a different module than the one seeded."""
        seed_request = NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation", noise_level=0.0, law_version="v0"
        )
        await server.seed_session(mock_request, seed_request)

        handler = server._create_module_handler("m1_coulomb_force")

        with raises(HTTPException) as exc_info:
            await handler(mock_request, {})

        assert exc_info.value.status_code == 400
        assert "Session configured for 'm0_gravity'" in exc_info.value.detail

    @mark.asyncio
    async def test_run_experiment_uninitialized_session(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test that calling run_experiment without seeding session first."""
        handler = server._create_module_handler("m0_gravity")
        with raises(ResourcesSessionLostError) as exc_info:
            await handler(mock_request, {})

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail["failure_kind"] == "session_lost"
        assert "Session not initialized" in exc_info.value.detail["message"]

    @mark.asyncio
    async def test_run_experiment_missing_parameters(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test that missing required physical parameters are handled by returning an error result."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )

        handler = server._create_module_handler("m0_gravity")
        invalid_body = {"mass1": 10.0, "distance": 5.0}  # mass2 missing

        response = await handler(mock_request, invalid_body)

        assert isinstance(response, RunExperimentResponse)
        assert isinstance(response.result, dict)
        assert "error" in response.result
        assert "mass2" in response.result["error"]

    @mark.asyncio
    @mark.parametrize(
        "module_name, input_data",
        [
            ("m0_gravity", {"mass1": 10.0, "mass2": 10.0, "distance": 1.0}),
            ("m1_coulomb_force", {"q1": 1.0, "q2": 1.0, "distance": 1.0}),
            ("m2_magnetic_force", {"current1": 1.0, "current2": 1.0, "distance": 1.0}),
            ("m3_fourier_law", {"k": 1.0, "A": 1.0, "delta_T": 10.0, "d": 1.0}),
            ("m4_snell_law", {"refractive_index_1": 1.1, "refractive_index_2": 1.2, "incidence_angle": 30.0}),
            ("m5_radioactive_decay", {"N0": 100.0, "lambda_constant": 0.01, "t": 10.0}),
            ("m6_underdamped_harmonic", {"k_constant": 1.0, "mass": 1.0, "b_constant": 0.1}),
            ("m7_malus_law", {"I_0": 1.0, "theta": 0.5}),
            ("m8_sound_speed", {"adiabatic_index": 1.4, "temperature": 300.0, "molar_mass": 0.029}),
            ("m9_hooke_law", {"x": 0.5}),
            ("m10_be_distribution", {"omega": 1e8, "temperature": 30.0}),
            ("m11_heat_transfer", {"m": 1.0, "c": 4186.0, "delta_T": 10.0}),
        ],
    )
    async def test_run_experiment_all_modules_vanilla(
        self, server: NewtonBenchResourcesServer, mock_request: Request, module_name: str, input_data: dict
    ) -> None:
        """Test connectivity for all 12 modules in 'vanilla_equation' mode."""
        seed_request = NewtonBenchSeedSessionRequest(
            module_name=module_name, difficulty="easy", system="vanilla_equation", noise_level=0.0, law_version="v0"
        )
        await server.seed_session(mock_request, seed_request)
        handler = server._create_module_handler(module_name)

        try:
            response = await handler(mock_request, input_data)
        except Exception as e:
            raise e

        assert isinstance(response, RunExperimentResponse)
        assert isinstance(response.result, (float, int))
        assert response.result > 0

    @mark.asyncio
    @mark.parametrize(
        "system_mode, extra_inputs",
        [
            ("vanilla_equation", {}),
            ("simple_system", {"initial_velocity": 0.0, "duration": 1.0, "time_step": 0.1}),
            ("complex_system", {"initial_velocity": 1.0, "duration": 1.0, "time_step": 0.1}),
        ],
    )
    async def test_run_experiment_all_systems(
        self, server: NewtonBenchResourcesServer, mock_request: Request, system_mode: str, extra_inputs: dict
    ) -> None:
        """Test running experiment with all 3 system modes (vanilla_equation, simple_system, complex_system)."""
        module_name = "m0_gravity"

        seed_request = NewtonBenchSeedSessionRequest(
            module_name=module_name, difficulty="easy", system=system_mode, noise_level=0.0, law_version="v0"
        )
        await server.seed_session(mock_request, seed_request)

        handler = server._create_module_handler(module_name)

        base_input = {"mass1": 10.0, "mass2": 10.0, "distance": 2.0}
        input_data = {**base_input, **extra_inputs}

        response = await handler(mock_request, input_data)
        assert isinstance(response, RunExperimentResponse)
        if system_mode == "vanilla_equation":
            assert isinstance(response.result, (float, int))
            assert response.result > 0
        else:
            assert isinstance(response.result, dict)
            assert "time" in response.result
            assert "position" in response.result
            assert "velocity" in response.result
            assert len(response.result["time"]) > 0
            assert len(response.result["position"]) == len(response.result["time"]) and len(
                response.result["velocity"]
            ) == len(response.result["time"])

    @mark.asyncio
    async def test_run_experiment_concurrent_sessions(self, server: NewtonBenchResourcesServer) -> None:
        """Test concurrent independent sessions with different modules (m0 and m1)."""
        req_a = MagicMock(spec=Request)
        req_a.session = {SESSION_ID_KEY: "session_A"}
        await server.seed_session(
            req_a,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )

        req_b = MagicMock(spec=Request)
        req_b.session = {SESSION_ID_KEY: "session_B"}
        await server.seed_session(
            req_b,
            NewtonBenchSeedSessionRequest(
                module_name="m1_coulomb_force",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )

        handler_m0 = server._create_module_handler("m0_gravity")
        res_a = await handler_m0(req_a, {"mass1": 10.0, "mass2": 10.0, "distance": 2.0})
        assert isinstance(res_a, RunExperimentResponse)
        assert isinstance(res_a.result, (float, int))

        handler_m1 = server._create_module_handler("m1_coulomb_force")
        res_b = await handler_m1(req_b, {"q1": 1.0, "q2": -1.0, "distance": 1.0})
        assert isinstance(res_b, RunExperimentResponse)
        assert isinstance(res_b.result, (float, int))

        with raises(HTTPException) as exc:
            await handler_m0(req_b, {"mass1": 10.0, "mass2": 10.0, "distance": 2.0})
        assert "Session configured for 'm1_coulomb_force'" in exc.value.detail

    # ============================================================================
    # 4. Sandbox & Python Execution Tests
    # ============================================================================
    @mark.asyncio
    async def test_execute_python_basic(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test basic Python code execution."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        request_body = ExecutePythonRequest(code="x = 5\ny = 10\nx + y")

        response = await server.execute_python(mock_request, request_body)

        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert response.result == "15"
        assert response.stdout == ""
        assert response.stderr == ""
        assert response.error_message is None

    @mark.asyncio
    async def test_execute_python_with_numpy(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test Python code execution with numpy."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        request_body = ExecutePythonRequest(code="import numpy as np\narr = np.array([1, 2, 3])\narr.sum()")

        response = await server.execute_python(mock_request, request_body)

        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert response.result == "6"
        assert response.error_message is None

    @mark.asyncio
    async def test_execute_python_session_persistence(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test that variables persist across multiple execute_python calls in the same session."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        # First call: define variable
        request_body1 = ExecutePythonRequest(code="x = 42")
        response1 = await server.execute_python(mock_request, request_body1)
        assert response1.success is True

        # Second call: use variable from first call
        request_body2 = ExecutePythonRequest(code="x * 2")
        response2 = await server.execute_python(mock_request, request_body2)

        assert isinstance(response2, ExecutePythonResponse)
        assert response2.success is True
        assert response2.result == "84"

    @mark.asyncio
    async def test_execute_python_with_print(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test Python code execution with print statements."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        request_body = ExecutePythonRequest(code="print('Hello, World!')\nprint('Test output')")

        response = await server.execute_python(mock_request, request_body)

        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert "Hello, World!" in response.stdout
        assert "Test output" in response.stdout
        assert response.error_message is None

    @mark.asyncio
    async def test_execute_python_expression_result(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test that the last expression value is captured correctly."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        request_body = ExecutePythonRequest(code="import math\nmath.sqrt(16)")

        response = await server.execute_python(mock_request, request_body)

        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert response.result == "4.0"

    @mark.asyncio
    async def test_execute_python_no_expression_result(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test code that doesn't have a bare expression at the end."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        request_body = ExecutePythonRequest(code="x = 5\ny = 10\nz = x + y")

        response = await server.execute_python(mock_request, request_body)

        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert response.result is None

    @mark.asyncio
    async def test_execute_python_syntax_error(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test validation of syntax errors."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        request_body = ExecutePythonRequest(code="x = 5 +\n  # Invalid syntax")

        response = await server.execute_python(mock_request, request_body)

        assert isinstance(response, ExecutePythonResponse)
        assert response.success is False
        assert response.error_message is not None
        assert "Syntax error" in response.error_message

    @mark.asyncio
    async def test_execute_python_dangerous_import_os(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test that dangerous imports like 'os' are blocked."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        request_body = ExecutePythonRequest(code="import os\nos.system('ls')")

        response = await server.execute_python(mock_request, request_body)

        assert isinstance(response, ExecutePythonResponse)
        assert response.success is False
        assert response.error_message is not None
        assert "dangerous operation" in response.error_message.lower()

    @mark.asyncio
    async def test_execute_python_dangerous_eval(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test that dangerous operations like eval() are blocked."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        request_body = ExecutePythonRequest(code="eval('1+1')")

        response = await server.execute_python(mock_request, request_body)

        assert isinstance(response, ExecutePythonResponse)
        assert response.success is False
        assert response.error_message is not None
        assert "dangerous operation" in response.error_message.lower()

    @mark.asyncio
    async def test_execute_python_dangerous_open(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test that file operations like open() are blocked."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        request_body = ExecutePythonRequest(code="f = open('test.txt', 'w')")

        response = await server.execute_python(mock_request, request_body)

        assert isinstance(response, ExecutePythonResponse)
        assert response.success is False
        assert response.error_message is not None
        assert "dangerous operation" in response.error_message.lower()

    @mark.asyncio
    async def test_execute_python_runtime_error(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test handling of runtime errors during execution."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        request_body = ExecutePythonRequest(code="x = 5 / 0")  # Division by zero

        response = await server.execute_python(mock_request, request_body)

        assert isinstance(response, ExecutePythonResponse)
        assert response.success is False
        assert response.error_message is not None
        assert "ZeroDivisionError" in response.error_message or "division by zero" in response.error_message.lower()

    @mark.asyncio
    async def test_execute_python_stderr_field(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test that stderr field is properly returned in response."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        request_body = ExecutePythonRequest(code="x = 42")

        response = await server.execute_python(mock_request, request_body)

        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert hasattr(response, "stderr")
        assert isinstance(response.stderr, str)

    @mark.asyncio
    async def test_execute_python_with_seed_session(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test execute_python after seed_session to verify module access works."""
        seed_request = NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation", noise_level=0.0, law_version="v0"
        )
        await server.seed_session(mock_request, seed_request)

        request_body = ExecutePythonRequest(code="x = 100\ny = 200\nx + y")
        response = await server.execute_python(mock_request, request_body)

        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert response.result == "300"

    @mark.asyncio
    async def test_execute_python_without_seed_session(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test execute_python without seed_session raises 404 for a lost session."""
        request_body = ExecutePythonRequest(code="import math\nmath.pi * 2")

        with raises(ResourcesSessionLostError) as exc_info:
            await server.execute_python(mock_request, request_body)

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail["failure_kind"] == "session_lost"

    @mark.asyncio
    async def test_execute_python_multiple_sessions(self, server: NewtonBenchResourcesServer) -> None:
        """Test that different sessions maintain separate execution environments."""
        request1 = MagicMock(spec=Request)
        request1.session = {SESSION_ID_KEY: "session_1"}
        request2 = MagicMock(spec=Request)
        request2.session = {SESSION_ID_KEY: "session_2"}
        seed = NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation", noise_level=0.0, law_version="v0"
        )
        await server.seed_session(request1, seed)
        await server.seed_session(request2, seed)

        body1 = ExecutePythonRequest(code="x = 10")
        response1 = await server.execute_python(request1, body1)
        assert response1.success is True

        body2 = ExecutePythonRequest(code="x = 20")
        response2 = await server.execute_python(request2, body2)
        assert response2.success is True

        body1_check = ExecutePythonRequest(code="x")
        response1_check = await server.execute_python(request1, body1_check)
        assert response1_check.success is True
        assert response1_check.result == "10"

        body2_check = ExecutePythonRequest(code="x")
        response2_check = await server.execute_python(request2, body2_check)
        assert response2_check.success is True
        assert response2_check.result == "20"

    @mark.asyncio
    async def test_execute_python_complex_calculation(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test complex mathematical calculations."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        request_body = ExecutePythonRequest(
            code="""
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
mean = arr.mean()
std = arr.std()
result = mean + std * 2
result
"""
        )

        response = await server.execute_python(mock_request, request_body)

        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert response.result is not None
        result_value = float(response.result)
        assert math.isclose(result_value, 5.82842712474619)

    @mark.asyncio
    async def test_execute_python_function_definition(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test that function definitions work and persist across calls."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )

        body1 = ExecutePythonRequest(code="def add(a, b):\n    return a + b")
        response1 = await server.execute_python(mock_request, body1)
        assert response1.success is True

        body2 = ExecutePythonRequest(code="add(15, 25)")
        response2 = await server.execute_python(mock_request, body2)

        assert isinstance(response2, ExecutePythonResponse)
        assert response2.success is True
        assert response2.result == "40"

    @mark.asyncio
    async def test_execute_python_timeout(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test that long-running code is terminated by the timeout."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )

        server.config.max_execution_time = 2
        request_body = ExecutePythonRequest(code="while True: pass")
        response = await server.execute_python(mock_request, request_body)

        assert response.success is False
        assert response.error_message is not None
        assert "timeout" in response.error_message.lower()

    @staticmethod
    def _blocking_worker_start(
        worker: MagicMock,
    ) -> tuple[threading.Event, threading.Event, Callable[[int], MagicMock]]:
        """A SessionHandle stand-in that returns ``worker`` once the test sets ``release``, which only the loop does."""
        started, release = threading.Event(), threading.Event()

        def start(max_execution_time: int) -> MagicMock:
            started.set()
            assert release.wait(5), "the worker start blocked the event loop"
            return worker

        return started, release, start

    @mark.asyncio
    async def test_worker_starts_off_the_event_loop_once_per_session(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        worker = MagicMock(spec=SessionHandle, is_closed=False)
        worker.exec.return_value = ("", "", "1")
        started, release, start = self._blocking_worker_start(worker)

        with patch("resources_servers.newton_bench.app.SessionHandle", side_effect=start) as session_handle:
            calls = [
                asyncio.create_task(server.execute_python(mock_request, ExecutePythonRequest(code="1")))
                for _ in range(2)
            ]
            assert await asyncio.to_thread(started.wait, 5)
            release.set()
            responses = await asyncio.gather(*calls)

        assert [response.result for response in responses] == ["1", "1"]
        session_handle.assert_called_once()
        assert server._sessions[mock_request.session[SESSION_ID_KEY]] is worker

    @mark.asyncio
    async def test_worker_started_for_a_released_session_is_stopped(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        worker = MagicMock(spec=SessionHandle, is_closed=False)
        started, release, start = self._blocking_worker_start(worker)

        with patch("resources_servers.newton_bench.app.SessionHandle", side_effect=start):
            call = asyncio.create_task(server.execute_python(mock_request, ExecutePythonRequest(code="1")))
            assert await asyncio.to_thread(started.wait, 5)
            # External state loss during startup must not orphan the completed worker.
            server.session_metadata.pop(mock_request.session[SESSION_ID_KEY])
            release.set()
            response = await call

        assert response.success is False
        worker.aclose.assert_called_once()
        assert server._sessions == {}

    # ============================================================================
    # 5. Verification Logic Tests
    # ============================================================================
    @mark.asyncio
    async def test_verify_success_symbolic_match(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test verify() when law is symbolically equivalent."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        law_text = "def discovered_law(mass1, mass2, distance):\n    C = 8e-8\n    return (C * mass1 * mass2) / (distance ** 2)"
        mock_response = self._make_response_with_law(f"My discovery:\n<final_law>\n{law_text}\n</final_law>")
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=mock_response,
        )
        with patch("resources_servers.newton_bench.app._load_module") as mock_load:
            mock_mod = {"core": MagicMock(), "param_description": None}
            mock_mod["core"].evaluate_law.return_value = {
                "symbolic_equivalent": True,
                "rmsle": 0.1,
                "exact_accuracy": 1.0,
            }
            mock_load.return_value = mock_mod
            response = await server.verify(mock_request, verify_request)
        assert isinstance(response, NewtonBenchVerifyResponse)
        assert math.isclose(response.reward, 0.9548387096774194, rel_tol=1e-9, abs_tol=1e-9)
        assert response.symbolic_equivalent is True
        assert response.extracted_law is not None

    @mark.asyncio
    async def test_verify_success_numerical_match(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test verify() when law is numerically correct (RMSLE < 1e-5) but not symbolically equivalent."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        law_text = "def discovered_law(mass1, mass2, distance):\n    return 8e-8 * mass1 * mass2 / (distance ** 2)"
        mock_response = self._make_response_with_law(f"<final_law>\n{law_text}\n</final_law>")
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=mock_response,
        )
        with patch("resources_servers.newton_bench.app._load_module") as mock_load:
            mock_mod = {"core": MagicMock(), "param_description": None}
            mock_mod["core"].evaluate_law.return_value = {
                "symbolic_equivalent": False,
                "rmsle": 1e-6,
                "exact_accuracy": 0.0,
            }
            mock_load.return_value = mock_mod
            response = await server.verify(mock_request, verify_request)
        assert isinstance(response, NewtonBenchVerifyResponse)
        assert math.isclose(response.reward, 0.4, rel_tol=1e-6, abs_tol=1e-6)
        assert response.rmsle == 1e-6
        assert response.symbolic_equivalent is False

    @mark.asyncio
    async def test_verify_failure_no_law_tags(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test verify() when response has no <final_law> tags."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        mock_response = self._make_response_with_law("I think the law is F = G * m1 * m2 / r^2")
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=mock_response,
        )
        response = await server.verify(mock_request, verify_request)
        assert isinstance(response, NewtonBenchVerifyResponse)
        assert response.reward == 0.0
        assert response.extracted_law is None
        assert "No law found" in (response.evaluation_error or "")

    @mark.asyncio
    async def test_verify_failure_uninitialized_session(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test verify() without seed_session masks the sample: there is no task metadata to score against."""
        mock_response = self._make_response_with_law("<final_law>def f(): pass</final_law>")
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=mock_response,
        )
        result = await server.verify(mock_request, verify_request)
        assert result.reward == 0.0
        assert result.mask_sample is True
        assert result.failure_kind == "session_lost"

    @mark.asyncio
    async def test_verify_failure_missing_module_name(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test verify() when session metadata is missing module_name."""
        sid = mock_request.session[SESSION_ID_KEY]
        server.session_metadata[sid] = {
            "difficulty": "easy",
            "system": "vanilla_equation",
            "noise_level": 0.0,
            "law_version": "v0",
            "last_used": time.time(),
        }
        mock_response = self._make_response_with_law(
            "<final_law>def discovered_law(mass1, mass2, distance): return 1.0</final_law>"
        )
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=mock_response,
        )
        response = await server.verify(mock_request, verify_request)
        assert isinstance(response, NewtonBenchVerifyResponse)
        assert response.reward == 0.0
        assert "Missing module_name" in (response.evaluation_error or "")

    @mark.asyncio
    async def test_verify_evaluation_exception(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test verify() when core.evaluate_law() raises an exception."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        mock_response = self._make_response_with_law(
            "<final_law>def discovered_law(mass1, mass2, distance): return 1.0</final_law>"
        )
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=mock_response,
        )
        with patch("resources_servers.newton_bench.app._load_module") as mock_load:
            mock_mod = {"core": MagicMock(), "param_description": None}
            mock_mod["core"].evaluate_law.side_effect = ValueError("Invalid function syntax")
            mock_load.return_value = mock_mod
            response = await server.verify(mock_request, verify_request)
        assert isinstance(response, NewtonBenchVerifyResponse)
        assert response.reward == 0.0
        assert response.extracted_law is not None
        assert "Evaluation failed" in (response.evaluation_error or "")
        assert "Invalid function syntax" in (response.evaluation_error or "")

    @mark.asyncio
    async def test_verify_failure_both_wrong(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test verify() when law is both symbolically and numerically incorrect."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        mock_response = self._make_response_with_law(
            "<final_law>def discovered_law(mass1, mass2, distance): return mass1 + mass2</final_law>"
        )
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=mock_response,
        )
        with patch("resources_servers.newton_bench.app._load_module") as mock_load:
            mock_mod = {"core": MagicMock(), "param_description": None}
            mock_mod["core"].evaluate_law.return_value = {
                "symbolic_equivalent": False,
                "rmsle": 100.0,
                "exact_accuracy": 0.0,
            }
            mock_load.return_value = mock_mod
            response = await server.verify(mock_request, verify_request)
        assert isinstance(response, NewtonBenchVerifyResponse)
        assert math.isclose(response.reward, -0.9592233009708737, rel_tol=1e-9, abs_tol=1e-9)
        assert response.symbolic_equivalent is False
        assert response.rmsle == 100.0

    def test_extract_law_multiple_tags_uses_last(self, server: NewtonBenchResourcesServer) -> None:
        """Test _extract_law_from_response() when multiple <final_law> tags exist (uses last)."""
        text = (
            "First:\n<final_law>def wrong(): pass</final_law>\n\n"
            "Corrected:\n<final_law>def discovered_law(mass1, mass2, distance): return 1.0</final_law>"
        )
        mock_response = self._make_response_with_law(text)
        extracted = server._extract_law_from_response(mock_response)
        assert extracted == "def discovered_law(mass1, mass2, distance): return 1.0"
        assert "wrong" not in (extracted or "")

    def test_extract_law_malformed_missing_end_tag(self, server: NewtonBenchResourcesServer) -> None:
        """Test _extract_law_from_response() with malformed tags (missing closing tag)."""
        text = "<final_law>def discovered_law(mass1, mass2, distance): return 1.0\n"
        mock_response = self._make_response_with_law(text)
        extracted = server._extract_law_from_response(mock_response)
        assert extracted is None

    def test_extract_law_multiple_messages(self, server: NewtonBenchResourcesServer) -> None:
        """Test _extract_law_from_response() when tags are in second message."""
        resp = NeMoGymResponse.model_validate(
            {
                "id": "resp_test",
                "object": "response",
                "created_at": 1756531804.0,
                "model": "test",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "id": "msg_0",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "First message", "annotations": []}],
                    },
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "<final_law>def discovered_law(mass1, mass2, distance): return 1.0</final_law>",
                                "annotations": [],
                            }
                        ],
                    },
                ],
            }
        )
        extracted = server._extract_law_from_response(resp)
        assert extracted == "def discovered_law(mass1, mass2, distance): return 1.0"

    def test_extract_law_empty_tags(self, server: NewtonBenchResourcesServer) -> None:
        """Test _extract_law_from_response() with empty <final_law></final_law> tags."""
        mock_response = self._make_response_with_law("<final_law></final_law>")
        extracted = server._extract_law_from_response(mock_response)
        assert extracted == ""

    # The m0_gravity easy v0 ground truth itself, so its RMSLE is 0 and only the judge can lower the reward.
    EXACT_GRAVITY_LAW = (
        "def discovered_law(mass1, mass2, distance):\n    return 6.674e-5 * mass1 * mass2 / distance ** 1.5"
    )

    @fixture
    def judge_keys(self, monkeypatch: MonkeyPatch) -> dict:
        """NewtonBench's judge API keys, cleared; the judge resolves its provider from this dict on every call."""
        keys = importlib.import_module("utils.call_llm_api").keys
        monkeypatch.setitem(keys, "oa", None)
        monkeypatch.setitem(keys, "or", None)
        return keys

    def _judge_reply(self, status_code: int, content: str = "") -> requests.Response:
        reply = requests.Response()
        reply.status_code = status_code
        reply.reason = "Too Many Requests" if status_code == 429 else "OK"
        reply.url = "https://openrouter.ai/api/v1/chat/completions"
        reply._content = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        return reply

    async def _verify_exact_law(self, server: NewtonBenchResourcesServer, mock_request: Request):
        """Verify the exact law with NewtonBench's real evaluate_law, wrapped in judge_failsafe as /verify is."""
        await server.seed_session(
            mock_request,
            NewtonBenchSeedSessionRequest(
                module_name="m0_gravity",
                difficulty="easy",
                system="vanilla_equation",
                noise_level=0.0,
                law_version="v0",
            ),
        )
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=self._make_response_with_law(f"<final_law>\n{self.EXACT_GRAVITY_LAW}\n</final_law>"),
        )
        return await judge_failsafe(server.verify)(mock_request, verify_request)

    def test_setup_webserver_refuses_to_start_without_judge_key(
        self, server: NewtonBenchResourcesServer, judge_keys: dict
    ) -> None:
        with raises(ValueError, match="OPENAI_API_KEY or OPENROUTER_API_KEY"):
            server.setup_webserver()

    def test_setup_webserver_starts_with_judge_key(
        self, server: NewtonBenchResourcesServer, judge_keys: dict, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setitem(judge_keys, "oa", "test-key")
        app = server.setup_webserver()
        assert "/run_experiment_m0_gravity" in {route.path for route in app.routes}

    @mark.asyncio
    async def test_verify_masks_judge_without_key(
        self, server: NewtonBenchResourcesServer, mock_request: Request, judge_keys: dict
    ) -> None:
        """Every judge attempt fails without a key; the exact law is masked, not scored 0.4 as "not equivalent"."""
        response = await self._verify_exact_law(server, mock_request)
        assert isinstance(response, JSONResponse)
        row = json.loads(response.body)
        assert row["mask_sample"] is True
        assert row["failure_kind"] == "judge_failed"
        assert row["reward"] == 0.0
        assert "No available API key" in row["failure_reason"]

    @mark.asyncio
    async def test_verify_masks_judge_rate_limited_on_every_attempt(
        self, server: NewtonBenchResourcesServer, mock_request: Request, judge_keys: dict, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setitem(judge_keys, "or", "test-key")
        post = MagicMock(return_value=self._judge_reply(429))
        monkeypatch.setattr("requests.post", post)
        response = await self._verify_exact_law(server, mock_request)
        assert post.call_count == 3
        assert isinstance(response, JSONResponse)
        row = json.loads(response.body)
        assert row["mask_sample"] is True
        assert row["failure_kind"] == "judge_failed"
        assert "429" in row["failure_reason"]

    @mark.asyncio
    async def test_verify_scores_judge_verdict_after_rate_limit(
        self, server: NewtonBenchResourcesServer, mock_request: Request, judge_keys: dict, monkeypatch: MonkeyPatch
    ) -> None:
        """A failed attempt the judge recovers from still yields its verdict and the full reward."""
        monkeypatch.setitem(judge_keys, "or", "test-key")
        post = MagicMock(side_effect=[self._judge_reply(429), self._judge_reply(200, "Answer: YES")])
        monkeypatch.setattr("requests.post", post)
        response = await self._verify_exact_law(server, mock_request)
        assert isinstance(response, NewtonBenchVerifyResponse)
        assert response.symbolic_equivalent is True
        assert response.rmsle == 0.0
        assert math.isclose(response.reward, 1.0)


TASK_DATA = {
    "id": 0,
    "module_name": "m0_gravity",
    "difficulty": "easy",
    "system": "vanilla_equation",
    "noise_level": 0.0,
    "law_version": "v0",
}
VERIFY_BODY = {
    "responses_create_params": {"input": [{"role": "user", "content": "test"}]},
    "response": {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "model": "test",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [],
    },
}


def _session_server() -> NewtonBenchResourcesServer:
    config = NewtonBenchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return NewtonBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _typed_seed(resources_session_id: str = "resources-session", task_data: dict | None = None) -> dict:
    # The body single_agent_turn sends: task_data holds the row's own fields.
    return ResourcesSeedSessionRequest(
        resources_session_id=resources_session_id,
        episode_id=EpisodeId(rollout_id="rollout", attempt=0),
        task_id=TaskId(taskset="newton_bench", task_id="0"),
        task_data=TASK_DATA if task_data is None else task_data,
    ).model_dump(mode="json")


def _typed_close(resources_session_id: str = "resources-session") -> dict:
    return ResourcesCloseSessionRequest(
        resources_session_id=resources_session_id, episode_id=EpisodeId(rollout_id="rollout", attempt=0)
    ).model_dump(mode="json")


class TestResourcesSessions:
    def test_resources_session_contract(self) -> None:
        server = _session_server()

        check_resources_session_contract(
            server.setup_webserver(),
            ResourcesSeedSessionRequest.model_validate(_typed_seed("contract-session")),
            keeps_state=True,
            holds_state=lambda resources_session_id: resources_session_id in server.session_metadata,
            verify_body=VERIFY_BODY,
        )

    @mark.parametrize(
        "task_data, message",
        [
            ({**TASK_DATA, "difficulty": None}, "task_data is not a valid NewtonBenchSeedSessionRequest"),
            (
                {k: v for k, v in TASK_DATA.items() if k != "law_version"},
                "task_data is not a valid NewtonBenchSeedSessionRequest",
            ),
            ({**TASK_DATA, "module_name": "m99_unknown"}, "Invalid module_name 'm99_unknown'"),
        ],
    )
    def test_typed_seed_rejects_task_data_no_retry_can_fix(self, task_data: dict, message: str) -> None:
        server = _session_server()

        with TestClient(server.setup_webserver()) as client:
            response = client.post("/seed_session", json=_typed_seed(task_data=task_data))

        assert response.status_code == 422
        assert message in response.text
        assert server.session_metadata == {}

    def test_typed_session_runs_tools_and_close_releases_the_worker(self) -> None:
        server = _session_server()

        with TestClient(server.setup_webserver()) as client:
            assert client.post("/seed_session", json=_typed_seed()).status_code == 200
            assert server.session_metadata["resources-session"]["module_name"] == "m0_gravity"

            experiment = client.post("/run_experiment_m0_gravity", json={"mass1": 1.0, "mass2": 1.0, "distance": 1.0})
            assert experiment.status_code == 200
            assert isinstance(experiment.json()["result"], float)
            python = client.post("/execute_python", json={"code": "x = 41\nx + 1"})
            assert python.json()["result"] == "42"
            handle = server._sessions["resources-session"]

            # A seed retried after its reply was lost arrives without the cookie, and keeps the worker's state.
            client.cookies.clear()
            assert client.post("/seed_session", json=_typed_seed()).status_code == 200
            assert client.post("/execute_python", json={"code": "x"}).json()["result"] == "41"

            client.cookies.clear()
            assert client.post("/close_session", json=_typed_close()).status_code == 200

        assert handle.is_closed
        assert server._sessions == {}
        assert server.session_metadata == {}
        assert server._session_episodes == {}

    def test_close_after_verify_succeeds(self) -> None:
        server = _session_server()

        with TestClient(server.setup_webserver()) as client:
            client.post("/seed_session", json=_typed_seed())
            verify = client.post("/verify", json=VERIFY_BODY)
            # Verification releases live state while a compact record serves retries.
            assert verify.json()["mask_sample"] is False
            assert server.session_metadata == {}
            assert client.post("/close_session", json=_typed_close()).status_code == 200

        assert server.session_metadata == {}
        assert server._session_episodes == {}

    def test_tool_calls_after_close_are_lost(self) -> None:
        server = _session_server()

        with TestClient(server.setup_webserver()) as client:
            client.post("/seed_session", json=_typed_seed())
            client.post("/close_session", json=_typed_close())
            experiment = client.post("/run_experiment_m0_gravity", json={"mass1": 1.0, "mass2": 1.0, "distance": 1.0})
            python = client.post("/execute_python", json={"code": "1"})

        for response in (experiment, python):
            assert response.status_code == 404
            assert response.json()["detail"]["failure_kind"] == "session_lost"
        assert server._sessions == {}

    def test_typed_verify_retry_succeeds_after_metadata_cleanup(self) -> None:
        server = _session_server()

        with TestClient(server.setup_webserver()) as client:
            assert client.post("/seed_session", json=_typed_seed()).status_code == 200
            for _ in range(2):
                response = client.post("/verify", json=VERIFY_BODY)
                assert response.status_code == 200
                assert response.json()["mask_sample"] is False
                assert response.json()["evaluation_error"] == (
                    "No law found in response. Expected <final_law>...</final_law> tags."
                )
            assert server.session_metadata == {}
            assert client.post("/close_session", json=_typed_close()).status_code == 200

        assert server.session_metadata == {}

    def test_repeated_seed_does_not_confirm_a_swept_session(self) -> None:
        server = _session_server()

        with TestClient(server.setup_webserver()) as client:
            assert client.post("/seed_session", json=_typed_seed()).status_code == 200
            server.session_metadata["resources-session"]["last_used"] = time.time() - server.config.session_ttl - 1
            client.portal.call(server._cleanup_sessions)
            other_episode = _typed_seed()
            other_episode["episode_id"]["rollout_id"] = "other"
            assert client.post("/seed_session", json=other_episode).status_code == 409
            response = client.post("/seed_session", json=_typed_seed())

        assert response.status_code == 404
        assert response.json()["detail"]["failure_kind"] == "session_expired"
        assert server.session_metadata == {}

    def test_end_session_stops_the_worker_but_verify_still_scores_the_law(self) -> None:
        # A model can call /end_session as a tool; that must not drop its own sample from the score.
        server = _session_server()
        evaluator = MagicMock()
        evaluator.evaluate_law.return_value = {"symbolic_equivalent": True, "rmsle": 0.0}

        with (
            TestClient(server.setup_webserver()) as client,
            patch("resources_servers.newton_bench.app._load_module", return_value={"core": evaluator}),
        ):
            assert client.post("/seed_session", json=_typed_seed()).status_code == 200
            assert client.post("/execute_python", json={"code": "x = 41\nx + 1"}).json()["result"] == "42"
            handle = server._sessions["resources-session"]
            assert client.post("/end_session", json={}).status_code == 200
            assert handle.is_closed and server._sessions == {}

            # The repeated seed still confirms the session, whose next call runs in a fresh worker.
            client.cookies.clear()
            assert client.post("/seed_session", json=_typed_seed()).status_code == 200
            fresh = client.post("/execute_python", json={"code": "x"}).json()
            assert "name 'x' is not defined" in fresh["error_message"]
            verify = client.post("/verify", json=_final_law_verify_body()).json()
            assert client.post("/close_session", json=_typed_close()).status_code == 200

        assert verify["mask_sample"] is False
        assert verify["reward"] == 1.0
        evaluator.evaluate_law.assert_called_once()
        assert server._sessions == {} and server.session_metadata == {}

    def test_session_reaped_by_idle_sweep_reports_expired(self) -> None:
        server = _session_server()

        with TestClient(server.setup_webserver()) as client:
            client.post("/seed_session", json=_typed_seed())
            server.session_metadata["resources-session"]["last_used"] = time.time() - server.config.session_ttl - 1
            client.portal.call(server._cleanup_sessions)

            python = client.post("/execute_python", json={"code": "1"})
            verify = client.post("/verify", json=VERIFY_BODY)
            # The Environment Server still closes the session it seeded.
            close = client.post("/close_session", json=_typed_close())
            late_verify = client.post("/verify", json=VERIFY_BODY)

        assert python.status_code == 404
        assert python.json()["detail"]["failure_kind"] == "session_expired"
        assert verify.json()["mask_sample"] is True
        assert verify.json()["failure_kind"] == "session_expired"
        assert close.status_code == 200
        assert late_verify.json()["mask_sample"] is True
        assert late_verify.json()["failure_kind"] == "session_expired"

    def test_a_swept_session_still_reports_expired_after_the_close_retry_window(self) -> None:
        server = _session_server()
        now = [1000.0]

        with (
            patch("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: now[0]),
            patch("resources_servers.newton_bench.app.monotonic", lambda: now[0]),
            TestClient(server.setup_webserver()) as client,
        ):
            client.post("/seed_session", json=_typed_seed())
            server.session_metadata["resources-session"]["last_used"] = time.time() - server.config.session_ttl - 1
            client.portal.call(server._cleanup_sessions)
            # The agent's model stalled: its verify arrives well after a close retry would have.
            now[0] += server.config.session_close_retry_window_seconds + 60
            verify = client.post("/verify", json=VERIFY_BODY)

        assert verify.json()["mask_sample"] is True
        assert verify.json()["failure_kind"] == "session_expired"


@mark.parametrize("stage", ["active", "verified", "closed"])
def test_cookie_seed_cannot_overwrite_typed_or_closed_cookie(stage: str) -> None:
    server = _session_server()
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=_typed_seed()).status_code == 200
        if stage == "verified":
            assert client.post("/verify", json=VERIFY_BODY).status_code == 200
        elif stage == "closed":
            assert client.post("/close_session", json=_typed_close()).status_code == 200
        before = dict(server.session_metadata)
        assert client.post("/seed_session", json=TASK_DATA).status_code == 409
        assert server.session_metadata == before


@mark.parametrize("expose_tools_over_mcp", [False, True])
def test_malformed_typed_seed_does_not_fall_back_to_cookie_seed(expose_tools_over_mcp: bool) -> None:
    server = _session_server()
    server.config.expose_tools_over_mcp = expose_tools_over_mcp
    malformed = dict(TASK_DATA) | {"resources_session_id": "incomplete-typed"}
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app) as client:
        assert client.post("/seed_session", json=malformed).status_code == 422
    assert server.session_metadata == {}
    assert server._session_episodes == {}


def test_end_session_is_not_an_mcp_tool() -> None:
    server = _session_server()
    server.config.expose_tools_over_mcp = True

    tools = maybe_auto_expose(server, server.setup_webserver())

    assert "end_session" not in tools
    assert {"execute_python", "run_experiment_m0_gravity"} <= set(tools)


def test_run_seed_keeps_a_row_with_its_own_task_id() -> None:
    # A dataset row's scalar task_id is a row field, not a typed seed's TaskId.
    server = _session_server()
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=dict(TASK_DATA) | {"task_id": "gravity-0"}).status_code == 200
        assert len(server.session_metadata) == 1
        assert server._session_episodes == {}


@mark.parametrize("typed", [False, True])
def test_checkpoint_verify_releases_state_but_retry_survives_until_expiry(typed: bool, monkeypatch) -> None:
    from omegaconf import DictConfig

    clock = [100.0]
    monkeypatch.setattr("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: clock[0])
    server = _session_server()
    server.server_client.global_config_dict = DictConfig(
        {"checkpoint": {"enabled": True, "control_auth_token": "test"}}
    )
    prefix = "/ng-rollout/rollout"
    with TestClient(server.setup_webserver()) as client:
        assert client.post(prefix + "/seed_session", json=_typed_seed() if typed else TASK_DATA).status_code == 200
        sid = next(iter(server.session_metadata))
        assert server._checkpoint.is_open(sid)
        handle = MagicMock(spec=SessionHandle)
        server._sessions[sid] = handle
        first = client.post(prefix + "/verify", json=VERIFY_BODY).json()
        assert not first["mask_sample"]
        assert not server._checkpoint.is_open(sid)
        assert server.session_metadata == {}
        assert server._session_episodes == {}
        assert server._sessions == {}
        handle.aclose.assert_called_once()
        retry = client.post(prefix + "/verify", json=VERIFY_BODY).json()
        assert retry["reward"] == first["reward"] and retry["mask_sample"] is False
        if typed:
            wrong_owner = _typed_close()
            wrong_owner["episode_id"]["rollout_id"] = "other"
            assert client.post(prefix + "/close_session", json=wrong_owner).status_code == 409
        clock[0] += server.config.session_close_retry_window_seconds + 1
        assert client.post(prefix + "/verify", json=VERIFY_BODY).json()["mask_sample"] is True
        assert not server.session_is_verified(sid)


def test_next_cookie_run_keeps_its_cookie_and_replaces_the_finished_run() -> None:
    server = _session_server()
    body = _final_law_verify_body()
    evaluator = MagicMock()
    evaluator.evaluate_law.side_effect = [
        {"symbolic_equivalent": True, "rmsle": 0.0},
        {"symbolic_equivalent": False, "rmsle": 10.0},
    ]
    with (
        TestClient(server.setup_webserver()) as client,
        patch("resources_servers.newton_bench.app._load_module", return_value={"core": evaluator}),
    ):
        assert client.post("/seed_session", json=TASK_DATA).status_code == 200
        [session_id] = server.session_metadata
        assert client.post("/verify", json=body).json()["reward"] == 1.0
        # The next /run on the same cookie keeps the ID and replaces the finished run, retry record included.
        assert client.post("/seed_session", json=TASK_DATA).status_code == 200
        assert list(server.session_metadata) == [session_id]
        assert not server.session_is_verified(session_id)
        # The same answer is evaluated again for the new run, not replayed from the finished one.
        assert client.post("/verify", json=body).json()["reward"] < 1.0
        assert evaluator.evaluate_law.call_count == 2
        assert server.session_metadata == {}


def _final_law_verify_body() -> dict:
    body = deepcopy(VERIFY_BODY)
    body["response"]["output"] = [
        {
            "id": "answer",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "<final_law>def law(): return 1</final_law>", "annotations": []}
            ],
        }
    ]
    return body


def test_verify_retry_reuses_evaluation_and_refuses_a_changed_answer() -> None:
    server = _session_server()
    body = _final_law_verify_body()
    evaluator = MagicMock()
    evaluator.evaluate_law.side_effect = [
        {"symbolic_equivalent": True, "rmsle": 0.0},
        {"symbolic_equivalent": False, "rmsle": 10.0},
    ]
    with (
        TestClient(server.setup_webserver()) as client,
        patch("resources_servers.newton_bench.app._load_module", return_value={"core": evaluator}),
    ):
        client.post("/seed_session", json=_typed_seed())
        first = client.post("/verify", json=body).json()
        retry = client.post("/verify", json=body).json()
        assert first["reward"] == retry["reward"] == 1.0
        evaluator.evaluate_law.assert_called_once()
        changed = deepcopy(body)
        changed["response"]["output"][0]["content"][0]["text"] = "<final_law>def law(): return 2</final_law>"
        assert client.post("/verify", json=changed).json()["mask_sample"] is True
        assert client.post("/close_session", json=_typed_close()).status_code == 200
        assert client.post("/verify", json=body).json()["mask_sample"] is True


def test_failed_verify_cleanup_preserves_grade_and_cached_evaluation_until_close() -> None:
    server = _session_server()
    body = _final_law_verify_body()
    evaluator = MagicMock()
    evaluator.evaluate_law.return_value = {"symbolic_equivalent": True, "rmsle": 0.0}
    handle = MagicMock(spec=SessionHandle)
    handle.is_closed = False
    handle.exec.return_value = ("", "", "1")
    handle.aclose.side_effect = RuntimeError("worker still alive")
    with (
        TestClient(server.setup_webserver(), raise_server_exceptions=False) as client,
        patch("resources_servers.newton_bench.app._load_module", return_value={"core": evaluator}),
    ):
        assert client.post("/seed_session", json=_typed_seed()).status_code == 200
        server._sessions["resources-session"] = handle
        for _ in range(2):
            verified = client.post("/verify", json=body)
            assert verified.status_code == 200
            assert verified.json()["reward"] == 1.0 and not verified.json()["mask_sample"]
        evaluator.evaluate_law.assert_called_once()
        assert handle.aclose.call_count == 2
        assert server._sessions["resources-session"] is handle
        assert server.session_is_verified("resources-session")
        python = client.post("/execute_python", json={"code": "1"})
        experiment = client.post("/run_experiment_m0_gravity", json={"mass1": 1.0, "mass2": 1.0, "distance": 1.0})
        for response in (python, experiment):
            assert response.status_code == 404
            assert response.json()["detail"]["failure_kind"] == "session_lost"
        handle.exec.assert_not_called()
        evaluator.run_experiment_for_module.assert_not_called()
        assert server._sessions["resources-session"] is handle
        assert client.post("/close_session", json=_typed_close()).status_code == 500
        assert server._sessions["resources-session"] is handle
        handle.aclose.side_effect = None
        assert client.post("/close_session", json=_typed_close()).status_code == 200
        assert handle.aclose.call_count == 4
        assert server._sessions == {} and server.session_metadata == {}
        assert server._session_episodes == {} and server.session_is_closed("resources-session")


@mark.parametrize("typed", [False, True])
def test_checkpoint_retire_releases_live_session(typed: bool) -> None:
    from omegaconf import DictConfig

    server = _session_server()
    server.server_client.global_config_dict = DictConfig(
        {"checkpoint": {"enabled": True, "control_auth_token": "test"}}
    )
    with TestClient(server.setup_webserver()) as client:
        assert (
            client.post("/ng-rollout/rollout/seed_session", json=_typed_seed() if typed else TASK_DATA).status_code
            == 200
        )
        sid = next(iter(server.session_metadata))
        handle = MagicMock(spec=SessionHandle)
        server._sessions[sid] = handle
        client.portal.call(server._checkpoint.retire, EpisodeId(rollout_id="rollout", attempt=0))
        assert server.session_metadata == {}
        assert server._session_episodes == {}
        handle.aclose.assert_called_once()
        assert server._sessions == {}


def test_typed_seed_cannot_replace_a_live_cookie_session() -> None:
    server = _session_server()
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=TASK_DATA).status_code == 200
        sid = next(iter(server.session_metadata))
        original = server.session_metadata[sid]
        assert client.post("/seed_session", json=_typed_seed(sid)).status_code == 409
        assert server.session_metadata[sid] is original
        assert server._session_episodes == {}


def test_failed_cookie_reseed_preserves_the_completed_evaluation() -> None:
    server = _session_server()
    body = _final_law_verify_body()
    evaluator = MagicMock()
    evaluator.evaluate_law.side_effect = [
        {"symbolic_equivalent": True, "rmsle": 0.0},
        {"symbolic_equivalent": False, "rmsle": 10.0},
    ]
    with (
        TestClient(server.setup_webserver()) as client,
        patch("resources_servers.newton_bench.app._load_module", return_value={"core": evaluator}),
    ):
        assert client.post("/seed_session", json=TASK_DATA).status_code == 200
        [session_id] = server.session_metadata
        assert client.post("/verify", json=body).json()["reward"] == 1.0
        assert server.session_is_verified(session_id)
        with patch.object(NewtonBenchResourcesServer, "_session_metadata", side_effect=RuntimeError("setup failed")):
            with raises(RuntimeError, match="setup failed"):
                client.post("/seed_session", json=TASK_DATA)
        assert server.session_is_verified(session_id)
        assert server.session_metadata == {} and server._sessions == {}
        assert client.post("/verify", json=body).json()["reward"] == 1.0
        assert evaluator.evaluate_law.call_count == 1

        assert client.post("/seed_session", json=TASK_DATA).status_code == 200
        assert list(server.session_metadata) == [session_id]
        assert not server.session_is_verified(session_id)
        assert client.post("/verify", json=body).json()["reward"] < 1.0
        assert evaluator.evaluate_law.call_count == 2
        assert server.session_metadata == {}


@mark.asyncio
async def test_cancelled_worker_start_remains_owned_until_close() -> None:
    server = _session_server()
    sid = "cancelled-start"
    request = MagicMock(spec=Request)
    request.session = {SESSION_ID_KEY: sid}
    seed = ResourcesSeedSessionRequest.model_validate(_typed_seed(sid))
    await server.seed_session(request, seed)
    started, finish = threading.Event(), threading.Event()
    handles = []

    def delayed_start(max_execution_time: int) -> SessionHandle:
        handle = SessionHandle(max_execution_time)
        handles.append(handle)
        started.set()
        if not finish.wait(15):
            handle.close()
            raise TimeoutError("Test did not release worker startup")
        return handle

    call = closing = None
    try:
        with patch("resources_servers.newton_bench.app.SessionHandle", side_effect=delayed_start):
            call = asyncio.create_task(server.execute_python(request, ExecutePythonRequest(code="1")))
            assert await asyncio.to_thread(started.wait, 10)
            assert handles[0]._proc.is_alive()
            call.cancel()
            with raises(asyncio.CancelledError):
                await call
            closing = asyncio.create_task(
                server.close_resources_session(
                    ResourcesCloseSessionRequest(resources_session_id=sid, episode_id=seed.episode_id)
                )
            )
            await asyncio.sleep(0)
            assert not closing.done()
            finish.set()
            await asyncio.wait_for(closing, 10)
        assert handles[0].is_closed
        assert handles[0]._proc._closed
        assert server._starting_sessions == {} and server._sessions == {} and server.session_metadata == {}
    finally:
        finish.set()
        for task in (call, closing):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (call, closing) if task is not None), return_exceptions=True)
        for handle in handles:
            handle.close()


@mark.asyncio
async def test_close_queued_before_tool_prevents_worker_allocation() -> None:
    server = _session_server()
    sid = "close-before-start"
    request = MagicMock(spec=Request)
    request.session = {SESSION_ID_KEY: sid}
    seed = ResourcesSeedSessionRequest.model_validate(_typed_seed(sid))
    await server.seed_session(request, seed)
    with patch("resources_servers.newton_bench.app.SessionHandle") as factory:
        async with server.session_lock(sid):
            closing = asyncio.create_task(
                server.close_resources_session(
                    ResourcesCloseSessionRequest(resources_session_id=sid, episode_id=seed.episode_id)
                )
            )
            await asyncio.sleep(0)
            executing = asyncio.create_task(server.execute_python(request, ExecutePythonRequest(code="1")))
            await asyncio.sleep(0)
        await closing
        with raises(ResourcesSessionLostError):
            await executing
    factory.assert_not_called()
    assert server._sessions == {} and server._starting_sessions == {}


@mark.asyncio
async def test_verify_queued_before_tool_refuses_execution_when_worker_cleanup_fails() -> None:
    server = _session_server()
    sid = "verify-before-tool"
    request = MagicMock(spec=Request)
    request.session = {SESSION_ID_KEY: sid}
    seed = ResourcesSeedSessionRequest.model_validate(_typed_seed(sid))
    await server.seed_session(request, seed)
    handle = MagicMock(spec=SessionHandle)
    handle.is_closed = False
    handle.exec.return_value = ("", "", "1")
    handle.aclose.side_effect = RuntimeError("worker still alive")
    server._sessions[sid] = handle
    try:
        async with server.session_lock(sid):
            verifying = asyncio.create_task(
                server.verify(request, NewtonBenchVerifyRequest.model_validate(VERIFY_BODY))
            )
            await asyncio.sleep(0)
            executing = asyncio.create_task(server.execute_python(request, ExecutePythonRequest(code="1")))
            await asyncio.sleep(0)
        verified = await verifying
        with raises(ResourcesSessionLostError):
            await executing
        assert not verified.mask_sample
        handle.exec.assert_not_called()
        assert server._sessions[sid] is handle
        assert server.session_is_verified(sid)
    finally:
        handle.aclose.side_effect = None
        await server.close_resources_session(
            ResourcesCloseSessionRequest(resources_session_id=sid, episode_id=seed.episode_id)
        )
    assert server._sessions == {} and server.session_metadata == {}


@mark.asyncio
async def test_idle_sweep_retries_failed_worker_and_still_releases_other_sessions() -> None:
    server = _session_server()
    for sid in ("failed", "other"):
        request = MagicMock(spec=Request)
        request.session = {SESSION_ID_KEY: sid}
        await server.seed_session(request, ResourcesSeedSessionRequest.model_validate(_typed_seed(sid)))
        server.session_metadata[sid]["last_used"] = time.time() - server.config.session_ttl - 1
    failed, other = MagicMock(spec=SessionHandle), MagicMock(spec=SessionHandle)
    failed.aclose.side_effect = RuntimeError("worker still alive")
    server._sessions.update(failed=failed, other=other)
    with raises(ExceptionGroup, match="idle sweep"):
        await server._cleanup_sessions()
    other.aclose.assert_called_once()
    assert set(server._sessions) == set(server.session_metadata) == {"failed"}
    failed.aclose.side_effect = None
    await server._cleanup_sessions()
    assert server._sessions == {} and server.session_metadata == {}


@mark.asyncio
async def test_shutdown_releases_the_other_sessions_when_one_worker_will_not_stop() -> None:
    server = _session_server()
    app = server.setup_webserver()
    for sid in ("failed", "other"):
        request = MagicMock(spec=Request)
        request.session = {SESSION_ID_KEY: sid}
        await server.seed_session(request, ResourcesSeedSessionRequest.model_validate(_typed_seed(sid)))
    failed, other = MagicMock(spec=SessionHandle), MagicMock(spec=SessionHandle)
    failed.aclose.side_effect = RuntimeError("worker still alive")
    server._sessions.update(failed=failed, other=other)

    with raises(ExceptionGroup, match="shutdown could not release"):
        async with app.router.lifespan_context(app):
            pass

    other.aclose.assert_called_once()
    assert set(server._sessions) == set(server.session_metadata) == {"failed"}


@mark.asyncio
@mark.parametrize("typed", [False, True])
async def test_cancelled_close_keeps_worker_owned_and_retry_waits_without_blocking_loop(typed: bool) -> None:
    server = _session_server()
    sid = "cancelled-close"
    request = MagicMock(spec=Request)
    request.session = {SESSION_ID_KEY: sid}
    seed = ResourcesSeedSessionRequest.model_validate(_typed_seed(sid))
    await server.seed_session(request, seed if typed else NewtonBenchSeedSessionRequest(**TASK_DATA))
    original_metadata = server.session_metadata[sid]
    # Exercise the real async close lock/idempotence while controlling process exit.
    handle = SessionHandle.__new__(SessionHandle)
    handle._close_lock = threading.Lock()
    handle._exec_lock = threading.Lock()
    handle._conn = MagicMock()
    handle._proc = MagicMock()
    handle.is_closed = False
    started, finish = asyncio.Event(), asyncio.Event()
    handle._conn.send.side_effect = lambda message: started.set()
    handle._proc.is_alive.side_effect = lambda: not finish.is_set()
    server._sessions[sid] = handle
    body = ResourcesCloseSessionRequest(resources_session_id=sid, episode_id=seed.episode_id)
    closing = retry = None
    try:
        closing = asyncio.create_task(server.close_resources_session(body))
        await asyncio.wait_for(started.wait(), 2)
        # Reaching here while the process join is blocked proves another session can use the event loop.
        assert not closing.done()
        closing.cancel()
        with raises(asyncio.CancelledError):
            await closing
        assert server._sessions[sid] is handle
        assert server.session_metadata[sid] is original_metadata
        assert not server.session_is_closed(sid)
        retry = asyncio.create_task(
            server.close_resources_session(body)
            if typed
            else server.seed_session(request, NewtonBenchSeedSessionRequest(**TASK_DATA))
        )
        await asyncio.sleep(0)
        assert not retry.done()
        assert server.session_metadata[sid] is original_metadata
        finish.set()
        await asyncio.wait_for(retry, 5)
        handle._proc.join.assert_called_once()
        handle._proc.close.assert_called_once()
        assert handle.is_closed and server._sessions == {}
        if typed:
            assert server.session_metadata == {} and server.session_is_closed(sid)
        else:
            assert server.session_metadata[sid] is not original_metadata
    finally:
        finish.set()
        await asyncio.gather(*(task for task in (closing, retry) if task is not None), return_exceptions=True)
        await handle.aclose()


@mark.asyncio
@mark.parametrize("stage", ["startup", "shutdown"])
async def test_parent_lifespan_failure_still_releases_workers(stage: str) -> None:
    server = _session_server()
    request = MagicMock(spec=Request)
    request.session = {SESSION_ID_KEY: "parent-failure"}
    await server.seed_session(request, ResourcesSeedSessionRequest.model_validate(_typed_seed("parent-failure")))
    handle = MagicMock(spec=SessionHandle)
    server._sessions["parent-failure"] = handle

    @asynccontextmanager
    async def failing_parent(app: FastAPI):
        if stage == "startup":
            raise RuntimeError("parent failed")
        yield
        raise RuntimeError("parent failed")

    with patch(
        "resources_servers.newton_bench.app.SimpleResourcesServer.setup_webserver",
        return_value=FastAPI(lifespan=failing_parent),
    ):
        app = server.setup_webserver()
    with raises(RuntimeError, match="parent failed"):
        async with app.router.lifespan_context(app):
            pass
    handle.aclose.assert_called_once()
    assert server._sessions == {} and server.session_metadata == {}


def test_close_can_stop_worker_with_a_saturated_default_executor() -> None:
    handle = SessionHandle(max_execution_time=60)
    release_executor = threading.Event()

    async def close_with_busy_executor() -> None:
        server = _session_server()
        sid = "saturated-executor"
        request = MagicMock(spec=Request)
        request.session = {SESSION_ID_KEY: sid}
        seed = ResourcesSeedSessionRequest.model_validate(_typed_seed(sid))
        await server.seed_session(request, seed)
        server._sessions[sid] = handle
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        started = asyncio.Event()

        def occupy_executor() -> None:
            loop.call_soon_threadsafe(started.set)
            release_executor.wait(10)

        busy = loop.run_in_executor(None, occupy_executor)
        await started.wait()
        try:
            await asyncio.wait_for(
                server.close_resources_session(
                    ResourcesCloseSessionRequest(resources_session_id=sid, episode_id=seed.episode_id)
                ),
                timeout=4,
            )
            assert not busy.done()
            assert handle.is_closed
            assert server._sessions == {} and server.session_metadata == {}
        finally:
            release_executor.set()
            await busy

    try:
        assert handle.exec("1")[2] == "1"
        asyncio.run(close_with_busy_executor())
    finally:
        release_executor.set()
        handle.close()
