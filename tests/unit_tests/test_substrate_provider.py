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

from typing import Any

import pytest

from nemo_gym.sandbox.providers.base import SandboxSpec, SandboxStatus


pytestmark = pytest.mark.sandbox

pytest.importorskip("grpc", reason="grpcio optional sandbox dependency is not installed")
pytest.importorskip("httpx", reason="httpx optional sandbox dependency is not installed")

import grpc  # noqa: E402
import httpx  # noqa: E402

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
    ) -> None:
        self.control_calls: list[tuple[str, Any]] = []
        self.http_paths: list[str] = []
        self._get_actor_state = get_actor_state
        self._exec_body = exec_body if exec_body is not None else {"stdout": "ok", "stderr": "", "exit_code": 0}
        self._exec_status = exec_status
        self._probe_status = probe_status
        self._create_error = create_error

    async def control(self, rpc: str, request: Any) -> Any:
        self.control_calls.append((rpc, request))
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
    ) -> httpx.Response:
        self.http_paths.append(path)
        if path == "/exec":
            return httpx.Response(self._exec_status, json=self._exec_body)
        if path == "/upload":
            return httpx.Response(204)
        # readiness probe and everything else
        return httpx.Response(self._probe_status)

    def rpcs(self) -> list[str]:
        return [rpc for rpc, _ in self.control_calls]


def _make_provider(recorder: _Recorder, *, suspend_policy: str = "never") -> Any:
    provider = substrate_provider.SubstrateProvider(
        connection={"atespace": "swebench", "template_namespace": TEMPLATE_NAMESPACE},
        create={"suspend_policy": suspend_policy},
        # A short, delay-free probe keeps the test fast.
        probe={"deadline_s": 5.0, "stable_delay_s": 0.0},
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


async def test_close_suspends_then_deletes() -> None:
    recorder = _Recorder()
    provider = _make_provider(recorder)
    handle = await provider.create(SandboxSpec(image=IMAGE))

    await provider.close(handle)

    rpcs = recorder.rpcs()
    assert rpcs.index("SuspendActor") < rpcs.index("DeleteActor")


@pytest.mark.parametrize(
    "state, expected",
    [
        (ateapi_pb2.ActorState.ACTOR_STATE_RUNNING, SandboxStatus.RUNNING),
        (ateapi_pb2.ActorState.ACTOR_STATE_RESUMING, SandboxStatus.STARTING),
        (ateapi_pb2.ActorState.ACTOR_STATE_SUSPENDED, SandboxStatus.RUNNING),
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
