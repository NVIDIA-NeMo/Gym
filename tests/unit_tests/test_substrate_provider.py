# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any

import pytest

from nemo_gym.sandbox.providers.base import SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.registry import get_provider_class


pytestmark = pytest.mark.sandbox

pytest.importorskip("grpc", reason="grpcio optional sandbox dependency is not installed")
pytest.importorskip("httpx", reason="httpx optional sandbox dependency is not installed")

import grpc  # noqa: E402
import httpx  # noqa: E402

from nemo_gym.sandbox.providers.substrate import SubstrateProvider  # noqa: E402
from nemo_gym.sandbox.providers.substrate import provider as substrate_provider  # noqa: E402
from nemo_gym.sandbox.providers.substrate._ateapi import ateapi_pb2  # noqa: E402


IMAGE = "docker://registry.example/swebench:latest"
TEMPLATE_NAMESPACE = "ate-swebench"


class _Recorder:
    """Stand-in for the ateapi control plane and the atenet router.

    Replacing ``_control`` and ``_http`` on the instance keeps the test off the
    network: neither a gRPC channel nor the kubeconfig/token path is exercised.
    """

    def __init__(
        self,
        *,
        get_actor_state: int | None = None,
        exec_body: dict[str, Any] | None = None,
        exec_status: int = 200,
        probe_status: int = 200,
        create_error: Exception | None = None,
        exec_delay_s: float = 0.0,
        pause_delay_s: float = 0.0,
    ) -> None:
        self.control_calls: list[tuple[str, Any]] = []
        self.http_paths: list[str] = []
        self._get_actor_state = get_actor_state
        self._exec_body = exec_body if exec_body is not None else {"stdout": "ok", "stderr": "", "exit_code": 0}
        self._exec_status = exec_status
        self._probe_status = probe_status
        self._create_error = create_error
        self._exec_delay_s = exec_delay_s
        self._pause_delay_s = pause_delay_s
        # Interleaved control and HTTP activity, in completion order.
        self.events: list[tuple[str, str]] = []

    async def control(self, rpc: str, request: Any) -> Any:
        self.control_calls.append((rpc, request))
        if rpc == "PauseActor" and self._pause_delay_s:
            await asyncio.sleep(self._pause_delay_s)
        self.events.append(("control", rpc))
        if rpc == "CreateActor" and self._create_error is not None:
            raise self._create_error
        if rpc == "GetActor":
            state = self._get_actor_state
            if state is None:
                state = ateapi_pb2.ActorState.ACTOR_STATE_RUNNING
            return ateapi_pb2.Actor(status=ateapi_pb2.ActorStatus(state=state))
        return None

    async def http(
        self,
        method: str,
        actor: Any,
        path: str,
        *,
        timeout_s: Any = None,
        json: Any = None,
        content: Any = None,
        params: Any = None,
        limited: bool = True,
    ) -> httpx.Response:
        self.http_paths.append(path)
        self.events.append(("http", path))
        if path == "/exec":
            if self._exec_delay_s:
                await asyncio.sleep(self._exec_delay_s)
            return httpx.Response(self._exec_status, json=self._exec_body)
        if path == "/upload":
            return httpx.Response(204)
        # readiness probe and everything else
        return httpx.Response(self._probe_status)

    def rpcs(self) -> list[str]:
        return [rpc for rpc, _ in self.control_calls]


def _make_provider(recorder: _Recorder, *, suspend_policy: str = "never", idle_before_suspend_s: float = 0.0) -> Any:
    provider = substrate_provider.SubstrateProvider(
        connection={"atespace": "swebench", "template_namespace": TEMPLATE_NAMESPACE},
        create={
            "suspend_policy": suspend_policy,
            "idle_before_suspend_s": idle_before_suspend_s,
            # A short, delay-free probe keeps the test fast.
            "ready_timeout_s": 5.0,
        },
        exec={"capacity_retry_delay_s": 0.0},
        probe={"stable_delay_s": 0.0},
    )
    provider._control = recorder.control  # type: ignore[method-assign]
    provider._http = recorder.http  # type: ignore[method-assign]
    return provider


def _grpc_error(code: grpc.StatusCode, details: str) -> grpc.aio.AioRpcError:
    return grpc.aio.AioRpcError(code, grpc.aio.Metadata(), grpc.aio.Metadata(), details=details)


def test_template_name_is_deterministic_and_scheme_stripped() -> None:
    name = substrate_provider.template_name_for_image("registry.example/swebench:latest")
    assert name.startswith(substrate_provider.TEMPLATE_NAME_PREFIX)
    assert name == substrate_provider.template_name_for_image(substrate_provider.normalize_image(IMAGE))
    assert substrate_provider.normalize_image(IMAGE) == "registry.example/swebench:latest"
    assert (
        substrate_provider.normalize_image("docker://docker.io/library/python:3.12-slim") == "library/python:3.12-slim"
    )


def test_template_name_pins_the_swebench_derivation() -> None:
    # The string mini_swe_agent_2 asks for (docker.io/, lowercased) and the one
    # an operator passes to Substrate's demos/sandbox/gym-templates.sh must map
    # to the same template. The generator strips docker.io/ the same way; if
    # either derivation changes, this constant changes with it.
    gym_image = "docker.io/swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-14141:latest"
    operator_image = "swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-14141:latest"
    expected = "swb-2616df737f5e10cb"
    assert substrate_provider.template_name_for_image(substrate_provider.normalize_image(gym_image)) == expected
    assert substrate_provider.template_name_for_image(substrate_provider.normalize_image(operator_image)) == expected


def test_registry_resolves_substrate() -> None:
    assert get_provider_class("substrate") is SubstrateProvider


async def test_create_references_template_by_objectref_then_probes_and_uploads() -> None:
    recorder = _Recorder()
    provider = _make_provider(recorder)

    handle = await provider.create(SandboxSpec(image=IMAGE, files={"/work/seed.txt": "hi"}))

    assert handle.provider_name == "substrate"
    assert handle.sandbox_id.startswith("gym-")

    (rpc, request) = recorder.control_calls[0]
    assert rpc == "CreateActor"
    template = substrate_provider.template_name_for_image(substrate_provider.normalize_image(IMAGE))
    assert request.actor.actor_template.atespace == TEMPLATE_NAMESPACE
    assert request.actor.actor_template.name == template
    assert request.actor.metadata.atespace == "swebench"

    # Readiness is verified before any file lands, and the seed file is uploaded.
    assert recorder.http_paths[0] == "/readyz"
    assert "/upload" in recorder.http_paths


async def test_create_raises_when_template_missing() -> None:
    recorder = _Recorder(create_error=_grpc_error(grpc.StatusCode.NOT_FOUND, "no template"))
    provider = _make_provider(recorder)

    with pytest.raises(substrate_provider.SubstrateCreateError):
        await provider.create(SandboxSpec(image=IMAGE))


async def test_exec_parses_result_and_suspends_after_exec() -> None:
    recorder = _Recorder(exec_body={"stdout": "done", "stderr": "e", "exit_code": 3})
    provider = _make_provider(recorder, suspend_policy="after_exec")
    handle = await provider.create(SandboxSpec(image=IMAGE))

    result = await provider.exec(handle, "echo done")

    assert (result.stdout, result.stderr, result.return_code) == ("done", "e", 3)
    # The defining behavior: the actor is suspended once the exec returns.
    assert recorder.rpcs().count("SuspendActor") == 1


async def test_exec_never_policy_leaves_actor_running() -> None:
    recorder = _Recorder()
    provider = _make_provider(recorder, suspend_policy="never")
    handle = await provider.create(SandboxSpec(image=IMAGE))

    await provider.exec(handle, "echo hi")

    assert "SuspendActor" not in recorder.rpcs()


async def test_exec_router_5xx_is_reported_as_sandbox_failure() -> None:
    recorder = _Recorder(exec_status=503, exec_body={})
    provider = _make_provider(recorder, suspend_policy="after_exec")
    handle = await provider.create(SandboxSpec(image=IMAGE))

    result = await provider.exec(handle, "true")

    assert result.error_type == "sandbox"
    assert result.return_code == substrate_provider.SANDBOX_RUNTIME_RETURN_CODE
    # A sandbox failure is not a suspend point; the actor is left as-is.
    assert "SuspendActor" not in recorder.rpcs()


async def test_exec_pause_policy_pauses_after_exec() -> None:
    recorder = _Recorder()
    provider = _make_provider(recorder, suspend_policy="pause_after_exec")
    handle = await provider.create(SandboxSpec(image=IMAGE))

    await provider.exec(handle, "echo hi")

    assert recorder.rpcs().count("PauseActor") == 1
    assert "SuspendActor" not in recorder.rpcs()


async def test_exec_never_pauses_while_another_exec_is_in_flight() -> None:
    recorder = _Recorder(exec_delay_s=0.01)
    provider = _make_provider(recorder, suspend_policy="pause_after_exec")
    handle = await provider.create(SandboxSpec(image=IMAGE))

    await asyncio.gather(provider.exec(handle, "sleep 1"), provider.exec(handle, "sleep 1"))

    # Only the exec that found the actor idle pauses it.
    assert recorder.rpcs().count("PauseActor") == 1


async def test_idle_delay_defers_the_pause_and_a_new_exec_cancels_it() -> None:
    recorder = _Recorder()
    provider = _make_provider(recorder, suspend_policy="pause_after_exec", idle_before_suspend_s=0.05)
    handle = await provider.create(SandboxSpec(image=IMAGE))

    await provider.exec(handle, "step 1")
    assert "PauseActor" not in recorder.rpcs()
    await provider.exec(handle, "step 2")
    await asyncio.sleep(0.1)

    # Two back-to-back steps, one pause, after the idle delay.
    assert recorder.rpcs().count("PauseActor") == 1


async def test_exec_waits_for_a_pause_already_in_flight() -> None:
    recorder = _Recorder(pause_delay_s=0.05)
    provider = _make_provider(recorder, suspend_policy="pause_after_exec", idle_before_suspend_s=0.01)
    handle = await provider.create(SandboxSpec(image=IMAGE))

    await provider.exec(handle, "step 1")
    await asyncio.sleep(0.02)  # the idle delay has elapsed; PauseActor is now in flight
    await provider.exec(handle, "step 2")

    # The second exec went out only after the pause completed, so the router
    # saw a PAUSED actor to resume, never a PAUSING one.
    execs = [i for i, event in enumerate(recorder.events) if event == ("http", "/exec")]
    pause = recorder.events.index(("control", "PauseActor"))
    assert execs[0] < pause < execs[1]


async def test_exec_retries_while_the_actor_is_mid_transition(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(substrate_provider, "_TRANSITION_RETRY_DELAY_S", 0.0)
    recorder = _Recorder()
    provider = _make_provider(recorder)
    execs = 0

    def router(request: httpx.Request) -> httpx.Response:
        nonlocal execs
        if request.url.path != "/exec":
            return httpx.Response(200)
        execs += 1
        if execs == 1:
            return httpx.Response(
                503,
                text="actor nemo-gym/gym-x unavailable: AssignWorker prerequisite not met for Actor: "
                "nemo-gym/gym-x (got: ACTOR_STATE_PAUSING, want ACTOR_STATE_SUSPENDED or ACTOR_STATE_PAUSED)",
            )
        return httpx.Response(200, json={"stdout": "ok", "stderr": "", "exit_code": 0})

    provider._http = substrate_provider.SubstrateProvider._http.__get__(provider)
    provider._http_client = httpx.AsyncClient(transport=httpx.MockTransport(router), base_url="http://router")
    handle = await provider.create(SandboxSpec(image=IMAGE))

    result = await provider.exec(handle, "true", timeout_s=5)

    assert (result.return_code, result.error_type, execs) == (0, None, 2)


async def test_exec_retries_a_capacity_503_until_the_router_has_room() -> None:
    recorder = _Recorder()
    provider = _make_provider(recorder)
    seen: list[httpx.Request] = []

    def router(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/exec" and sum(r.url.path == "/exec" for r in seen) <= 2:
            return httpx.Response(503, text='actor "swebench/gym-x" unavailable: no worker has room for the actor')
        if request.url.path == "/exec":
            return httpx.Response(200, json={"stdout": "ok", "stderr": "", "exit_code": 0})
        return httpx.Response(200)

    provider._http = substrate_provider.SubstrateProvider._http.__get__(provider)  # real path, fake transport
    provider._http_client = httpx.AsyncClient(transport=httpx.MockTransport(router), base_url="http://router")
    handle = await provider.create(SandboxSpec(image=IMAGE))

    result = await provider.exec(handle, "true", timeout_s=5)

    assert result.return_code == 0 and result.error_type is None
    assert sum(r.url.path == "/exec" for r in seen) == 3
    assert seen[0].headers[substrate_provider.TARGET_ACTOR_HEADER] == f"swebench/{handle.sandbox_id}"


async def test_pause_waits_for_parked_state_and_resume_probes_through_the_router() -> None:
    recorder = _Recorder(get_actor_state=ateapi_pb2.ActorState.ACTOR_STATE_PAUSED)
    provider = _make_provider(recorder)
    handle = await provider.create(SandboxSpec(image=IMAGE))
    probes_before = recorder.http_paths.count("/readyz")

    await provider.pause(handle)
    await provider.resume(handle)

    assert recorder.rpcs().count("PauseActor") == 1
    assert recorder.http_paths.count("/readyz") == probes_before + 1


async def test_control_retries_aborted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(substrate_provider, "_CONTROL_BACKOFF_S", 0.0)
    provider = substrate_provider.SubstrateProvider(connection={"atespace": "swebench", "token": "t"})
    attempts = 0

    class _Stub:
        async def PauseActor(self, request: Any, *, metadata: Any, timeout: Any) -> Any:  # noqa: N802
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise _grpc_error(grpc.StatusCode.ABORTED, "concurrent update conflict, please retry")
            return ateapi_pb2.PauseActorResponse()

    provider._stub = _Stub()
    await provider._control("PauseActor", ateapi_pb2.PauseActorRequest())

    assert attempts == 3


async def test_close_deletes_in_any_state() -> None:
    recorder = _Recorder()
    provider = _make_provider(recorder)
    handle = await provider.create(SandboxSpec(image=IMAGE))

    await provider.close(handle)

    deletes = [request for rpc, request in recorder.control_calls if rpc == "DeleteActor"]
    assert len(deletes) == 1 and deletes[0].any_state
    assert "SuspendActor" not in recorder.rpcs()


def test_router_target_is_atespace_slash_actor() -> None:
    actor = substrate_provider._SubstrateActor(atespace="swebench", name="gym-abc", template="t", image="i")
    assert substrate_provider.TARGET_ACTOR_HEADER == "ate-target-actor"
    assert actor.target == "swebench/gym-abc"


@pytest.mark.parametrize(
    "state, expected",
    [
        (ateapi_pb2.ActorState.ACTOR_STATE_RUNNING, SandboxStatus.RUNNING),
        (ateapi_pb2.ActorState.ACTOR_STATE_RESUMING, SandboxStatus.STARTING),
        (ateapi_pb2.ActorState.ACTOR_STATE_SUSPENDED, SandboxStatus.PAUSED),
        (ateapi_pb2.ActorState.ACTOR_STATE_PAUSED, SandboxStatus.PAUSED),
        (ateapi_pb2.ActorState.ACTOR_STATE_CRASHED, SandboxStatus.ERROR),
        (ateapi_pb2.ActorState.ACTOR_STATE_DELETING, SandboxStatus.STOPPED),
        (ateapi_pb2.ActorState.ACTOR_STATE_UNSPECIFIED, SandboxStatus.UNKNOWN),
    ],
)
async def test_status_maps_actor_state(state: int, expected: SandboxStatus) -> None:
    recorder = _Recorder(get_actor_state=state)
    provider = _make_provider(recorder)
    handle = await provider.create(SandboxSpec(image=IMAGE))

    assert await provider.status(handle) == expected


async def test_status_not_found_is_stopped() -> None:
    recorder = _Recorder()
    provider = _make_provider(recorder)
    handle = await provider.create(SandboxSpec(image=IMAGE))

    async def not_found(rpc: str, request: Any) -> Any:
        raise _grpc_error(grpc.StatusCode.NOT_FOUND, "gone")

    provider._control = not_found  # type: ignore[method-assign]
    assert await provider.status(handle) == SandboxStatus.STOPPED
