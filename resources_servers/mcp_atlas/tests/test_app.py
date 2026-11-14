# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from aiohttp import web
from fastapi import HTTPException

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import ResourcesCloseSessionRequest, ResourcesParticipant
from nemo_gym.verification import JudgeError, judge_failsafe
from resources_servers.mcp_atlas.app import (
    SERVICE_ENV_NAMES,
    MCPAtlasResourcesServer,
    MCPAtlasResourcesServerConfig,
    MCPAtlasSeedSessionRequest,
    MCPAtlasVerifyRequest,
    tool_output_text,
)
from resources_servers.mcp_atlas.sandbox import HTTP_HELPER_PATH, AtlasSandbox, SandboxError
from resources_servers.mcp_atlas.tests.fake_agent_environment import FakeAgentEnvironment
from resources_servers.mcp_atlas.tests.test_sandbox import ExecSandbox, FakeSandbox, pool
from resources_servers.mcp_atlas.tests.test_verifier import message, response
from tests.helpers.reverification import build_reverification_payload


APPTAINER_PROVIDER = {
    "apptainer": {
        "create": {"extra_start_args": ["--cleanenv"]},
        "exec": {"extra_exec_args": ["--cleanenv"]},
    }
}


TOOLS = ["calculator_calculate", "github_get_repository", "met-museum_get-museum-object"]
META = {
    "task_id": "t1",
    "enabled_tools": TOOLS,
    "servers": ["calculator", "github", "met-museum"],
    "claims": ["The repo was created in 2013.", "The domain was registered in 2006."],
}


def judge_client(*outcomes: str) -> MagicMock:
    """A ServerClient whose judge answers the given outcomes in order; any other string is sent as is."""
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    replies = []
    for outcome in outcomes:
        if outcome in ("fulfilled", "partially_fulfilled", "not_fulfilled"):
            outcome = json.dumps({"coverage_outcome": outcome, "justification": "j"})
        completion = {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "judge",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": outcome},
                }
            ],
        }
        reply = AsyncMock()
        reply.ok = True
        reply.read = AsyncMock(return_value=orjson.dumps(completion))
        replies.append(reply)
    client.post = AsyncMock(side_effect=replies)
    return client


def make_server(base_url: str, client: MagicMock, **overrides) -> MCPAtlasResourcesServer:
    overrides.setdefault("sandbox_development_mode", True)
    config = MCPAtlasResourcesServerConfig(
        host="0.0.0.0",
        port=0,
        entrypoint="app.py",
        name="mcp_atlas",
        sandbox_base_url=base_url,
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        **overrides,
    )
    return MCPAtlasResourcesServer(config=config, server_client=client)


def http_request(session_id: str = "s1") -> SimpleNamespace:
    return SimpleNamespace(session={SESSION_ID_KEY: session_id}, cookies={})


def verify_body(text: str) -> MCPAtlasVerifyRequest:
    return MCPAtlasVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=response(message(text)),
        verifier_metadata=META,
    )


@pytest.fixture
async def environment():
    env = FakeAgentEnvironment(
        TOOLS,
        {
            "github_get_repository": (500, {"detail": "Failed to call tool: Not Found: repository does not exist"}),
            "calculator_calculate": (200, [{"type": "text", "text": "14"}]),
        },
    )
    url = await env.start()
    yield env, url
    await env.stop()


async def seed(server: MCPAtlasResourcesServer, session_id: str = "s1", *, metadata: dict | None = None) -> None:
    body = MCPAtlasSeedSessionRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        verifier_metadata=META if metadata is None else metadata,
    )
    await server.seed_session(http_request(session_id), body)


def test_exactly_one_sandbox_source_is_required() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        make_server("", judge_client())


def test_shared_environment_requires_explicit_development_mode() -> None:
    with pytest.raises(ValueError, match="sandbox_base_url requires sandbox_development_mode=true"):
        make_server("http://unused", judge_client(), sandbox_development_mode=False)


def test_apptainer_requires_explicit_development_mode(tmp_path) -> None:
    image = tmp_path / "atlas.sif"
    image.touch()
    with pytest.raises(ValueError, match="Apptainer requires sandbox_development_mode=true"):
        make_server(
            "",
            judge_client(),
            sandbox_image=str(image),
            sandbox_provider=APPTAINER_PROVIDER,
            sandbox_development_mode=False,
        )


@pytest.mark.parametrize("options", [{}, {"network_policy": {}}, {"network_policy": {"defaultAction": "allow"}}])
def test_opensandbox_requires_an_explicit_default_deny_policy(options: dict) -> None:
    with pytest.raises(ValueError, match="network_policy with defaultAction=deny"):
        make_server(
            "",
            judge_client(),
            sandbox_image="ghcr.io/scaleapi/mcp-atlas:1.2.7",
            sandbox_provider={"opensandbox": {}},
            sandbox_options=options,
            sandbox_development_mode=False,
        )


def test_opensandbox_rejects_a_local_sif(tmp_path) -> None:
    image = tmp_path / "atlas.sif"
    image.touch()
    with pytest.raises(ValueError, match="OCI image reference"):
        make_server(
            "",
            judge_client(),
            sandbox_image=str(image),
            sandbox_provider={"opensandbox": {}},
            sandbox_options={"network_policy": {"defaultAction": "deny", "egress": []}},
        )


def test_opensandbox_rejects_domain_port_restrictions_before_pool_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    pool_factory = MagicMock()
    monkeypatch.setattr("resources_servers.mcp_atlas.app.SandboxPool", pool_factory)
    rule = {"action": "allow", "target": "service.example"}
    options = {"network_policy": {"defaultAction": "deny", "egress": [rule]}}
    server = make_server(
        "",
        judge_client(),
        sandbox_image="ghcr.io/scaleapi/mcp-atlas:1.2.7",
        sandbox_provider={"opensandbox": {}},
        sandbox_options=options,
        sandbox_development_mode=False,
    )
    pool_factory.assert_called_once()
    assert server._pool is pool_factory.return_value
    assert pool_factory.call_args.kwargs["sandbox_options"] == options
    pool_factory.reset_mock()

    # SDK versions either discard domain ports or reject them during validation.
    with pytest.raises(ValueError):
        make_server(
            "",
            judge_client(),
            sandbox_image="ghcr.io/scaleapi/mcp-atlas:1.2.7",
            sandbox_provider={"opensandbox": {}},
            sandbox_options={
                "network_policy": {
                    "defaultAction": "deny",
                    "egress": [{**rule, "ports": [443]}],
                }
            },
            sandbox_development_mode=False,
        )
    pool_factory.assert_not_called()


@pytest.mark.parametrize(
    "policy",
    [
        {"defaultAction": "deny", "egress": [], "unexpectedPolicyField": "reject-me"},
        {
            "defaultAction": "deny",
            "egress": [{"action": "allow", "target": "service.example", "unexpectedRuleField": "reject-me"}],
        },
    ],
)
def test_opensandbox_rejects_policy_fields_the_sdk_silently_discards(
    monkeypatch: pytest.MonkeyPatch, policy: dict[str, object]
) -> None:
    pool_factory = MagicMock()
    monkeypatch.setattr("resources_servers.mcp_atlas.app.SandboxPool", pool_factory)
    with pytest.raises(ValueError, match="SDK drops or changes"):
        make_server(
            "",
            judge_client(),
            sandbox_image="ghcr.io/scaleapi/mcp-atlas:1.2.7",
            sandbox_provider={"opensandbox": {}},
            sandbox_options={"network_policy": policy},
            sandbox_development_mode=False,
        )
    pool_factory.assert_not_called()


def test_opensandbox_resolves_provider_and_keeps_control_credentials_host_side(monkeypatch) -> None:
    for name in SERVICE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EXA_API_KEY", "service-only")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "provider-only")
    monkeypatch.setenv("INFERENCE_API_KEY", "model-only")
    monkeypatch.setenv("NVAUTH_TOKEN", "host-only")
    client = judge_client()
    block = {"opensandbox": {"connection": {"api_key": "provider-only"}}, "default_metadata": {"team": "atlas"}}
    client.global_config_dict = {"sandbox": block}
    options = {"network_policy": {"defaultAction": "deny", "egress": []}}
    server = make_server(
        "",
        client,
        sandbox_image="ghcr.io/scaleapi/mcp-atlas:1.2.7",
        sandbox_options=options,
        sandbox_env={"PYTHONUNBUFFERED": "1"},
        sandbox_development_mode=False,
    )
    assert server._pool.sandbox_provider == {"opensandbox": block["opensandbox"]}
    assert server._pool.sandbox_metadata == {"team": "atlas"}
    assert server._pool.sandbox_env == {"EXA_API_KEY": "service-only", "PYTHONUNBUFFERED": "1"}
    assert server._pool.sandbox_options == options


@pytest.mark.parametrize("provider_name", ["opensandbox", "apptainer"])
@pytest.mark.parametrize("files_mode", ["omitted", "none", "custom", "empty"])
async def test_resolved_provider_stages_cache_defaults_and_preserves_file_overrides(
    provider_name: str, files_mode: str, tmp_path, monkeypatch
) -> None:
    custom_files = {
        "/root/.config/uv/uv.toml": 'cache-dir = "/tmp/custom-uv-cache"\n',
        "/root/.npmrc": "cache=/tmp/custom-npm-cache\nupdate-notifier=true\n",
        "/root/custom.conf": "caller configuration\n",
    }
    requested_files = None
    if files_mode == "custom":
        requested_files = custom_files
    elif files_mode == "empty":
        requested_files = {}
    before = dict(requested_files) if requested_files is not None else None
    overrides = {} if files_mode == "omitted" else {"sandbox_files": requested_files}
    client = judge_client()
    provider = APPTAINER_PROVIDER if provider_name == "apptainer" else {provider_name: {}}
    client.global_config_dict = {"runtime": provider}
    image = "ghcr.io/scaleapi/mcp-atlas:1.2.7"
    options = {"network_policy": {"defaultAction": "deny", "egress": []}}
    if provider_name == "apptainer":
        sif = tmp_path / "atlas.sif"
        sif.touch()
        image, options = str(sif), {}
    server = make_server(
        "",
        client,
        sandbox_image=image,
        sandbox_provider="runtime",
        sandbox_options=options,
        sandbox_development_mode=provider_name == "apptainer",
        sandbox_scratch_root=str(tmp_path),
        service_env_names=[],
        **overrides,
    )
    assert server._pool.sandbox_provider == provider
    specs = []

    def runtime(provider, spec):
        specs.append(spec)
        if provider_name == "opensandbox":
            return ExecSandbox(spec)
        return FakeSandbox(provider, spec, tools=["calculate"])

    # Keep config resolution, the real pool, and SandboxSpec construction; fake only the external runtime.
    monkeypatch.setattr(server._pool, "_factory", runtime)
    try:
        await server._pool.start("cache-defaults", servers=["calculator"], tools=["calculator_calculate"])
        assert len(specs) == 1
        staged = {path: content for path, content in specs[0].files.items() if path != HTTP_HELPER_PATH}
        if files_mode in {"omitted", "none"}:
            if provider_name == "opensandbox":
                expected = {
                    "/root/.config/uv/uv.toml": 'exclude-newer = "2026-07-17T00:00:00Z"\n',
                    "/root/.npmrc": "before=2026-07-17\nupdate-notifier=false\n",
                }
            else:
                expected = {
                    "/root/.config/uv/uv.toml": 'exclude-newer = "2026-07-17T00:00:00Z"\ncache-dir = "/tmp/uv-cache"\n',
                    "/root/.npmrc": "before=2026-07-17\ncache=/tmp/npm-cache\nupdate-notifier=false\n",
                }
        else:
            expected = before
        assert staged == expected
        assert server.config.sandbox_files == before
        assert requested_files == before
        assert (HTTP_HELPER_PATH in specs[0].files) is (provider_name == "opensandbox")
    finally:
        await server._shutdown()


def test_tool_output_text() -> None:
    assert tool_output_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert tool_output_text([]) == "success"
    assert tool_output_text([{"type": "text", "text": "  "}]) == "success"


async def test_tool_calls_are_proxied_and_bounded(environment) -> None:
    env, url = environment
    server = make_server(url, judge_client(), max_tool_calls=3)
    await seed(server)
    request = http_request()
    assert await server.call_tool(request, "calculator_calculate", {"expression": "2+3*4"}) == "14"
    # A model error (unknown repository) is a tool output, not an environment failure.
    assert "does not exist" in await server.call_tool(request, "github_get_repository", {"owner": "x"})
    await server.call_tool(request, "met-museum_get-museum-object", {"objectId": 1})
    assert env.calls[-1]["tool_args"] == {"objectId": 1, "returnImage": False}
    assert "not available" in await server.call_tool(request, "fetch_fetch", {"url": "x"})
    assert "limit of 3 tool calls" in await server.call_tool(request, "calculator_calculate", {"expression": "1"})
    assert server._rollouts["s1"].failures == []


async def test_verify_passes_at_threshold_and_reports_coverage(environment) -> None:
    _, url = environment
    server = make_server(url, judge_client("fulfilled", "partially_fulfilled"))
    await seed(server)
    result = await server.verify(http_request(), verify_body("Created 2013; domain from about 2006."))
    assert result.coverage == 0.75
    assert result.reward == 1.0
    assert result.passed_at_0_50 is True
    assert result.n_claims == 2
    assert result.mask_sample is False and result.failure_kind is None
    assert "s1" not in server._rollouts


async def test_verify_below_threshold_scores_zero(environment) -> None:
    _, url = environment
    server = make_server(url, judge_client("fulfilled", "not_fulfilled"))
    await seed(server)
    result = await server.verify(http_request(), verify_body("Created 2013."))
    assert result.coverage == 0.5
    assert result.reward == 0.0
    assert result.passed_at_0_50 is True


async def test_empty_answer_scores_zero_without_the_judge(environment) -> None:
    _, url = environment
    client = judge_client()
    server = make_server(url, client)
    await seed(server)
    result = await server.verify(http_request(), verify_body("   "))
    assert result.reward == 0.0 and result.coverage == 0.0
    client.post.assert_not_called()


async def test_service_error_masks_the_rollout(environment) -> None:
    env, url = environment
    env.results["github_get_repository"] = (500, {"detail": "Failed to call tool: 429 Too Many Requests"})
    server = make_server(url, judge_client("fulfilled", "fulfilled"))
    await seed(server)
    await server.call_tool(http_request(), "github_get_repository", {"owner": "x"})
    result = await server.verify(http_request(), verify_body("Created 2013, registered 2006."))
    assert result.mask_sample is True
    assert result.failure_kind == "mcp_atlas:service_error"
    assert "429" in result.failure_reason
    # The measurement is still recorded for the audit.
    assert result.coverage == 1.0


async def test_service_errors_are_kept_when_masking_is_off(environment) -> None:
    env, url = environment
    env.results["github_get_repository"] = (500, {"detail": "Failed to call tool: rate limit exceeded"})
    server = make_server(url, judge_client("fulfilled", "fulfilled"), mask_service_errors=False)
    await seed(server)
    await server.call_tool(http_request(), "github_get_repository", {"owner": "x"})
    result = await server.verify(http_request(), verify_body("ok"))
    assert result.mask_sample is False
    assert result.failure_kind == "mcp_atlas:service_error"


@pytest.mark.parametrize("status", [200, 404, 422, 500])
@pytest.mark.parametrize(
    "text",
    [
        "Failed to call tool: Object 502 not found",
        "no results for query 'credits'",
        "Invalid expression: quota",
        "Failed to call tool: Invalid api key argument",
        "Failed to call tool 'github_get_repository': Object 502 not found",
        "Failed to call tool 'github_get_repository': 500: Tool 'github_get_repository' execution failed: "
        "Invalid expression: quota",
        "Failed to call tool 'github_get_repository': HTTP 404 Not Found",
        "Failed to call tool 'github_get_repository': HTTP 422 Unprocessable Entity",
    ],
)
async def test_model_tool_errors_are_scored_instead_of_masked(environment, status: int, text: str) -> None:
    env, url = environment
    payload = [{"type": "text", "text": text}] if status == 200 else {"detail": text}
    env.results["github_get_repository"] = (status, payload)
    server = make_server(url, judge_client("not_fulfilled", "not_fulfilled"))
    await seed(server)
    output = await server.call_tool(http_request(), "github_get_repository", {"objectId": 502})
    assert text in output
    result = await server.verify(http_request(), verify_body("wrong answer"))
    assert result.reward == 0.0
    assert result.mask_sample is False
    assert result.failure_kind is None
    assert result.service_errors == []
    assert result.tool_calls == 1


@pytest.mark.parametrize(
    ("status", "text"),
    [
        (429, "Too Many Requests"),
        (503, "Service unavailable"),
        (500, "Failed to call tool: ConnectionError: connection refused"),
        (500, "Failed to call tool 'github_get_repository': 429 Too Many Requests"),
        (500, "Failed to call tool 'github_get_repository': HTTP 500 Internal Server Error"),
        (
            500,
            "Failed to call tool 'github_get_repository': 500: Tool 'github_get_repository' execution failed: "
            "HTTP 429 Too Many Requests",
        ),
    ],
)
async def test_explicit_upstream_failures_still_mask(environment, status: int, text: str) -> None:
    env, url = environment
    payload = [{"type": "text", "text": text}] if status == 200 else {"detail": text}
    env.results["github_get_repository"] = (status, payload)
    server = make_server(url, judge_client("fulfilled", "fulfilled"))
    await seed(server)
    await server.call_tool(http_request(), "github_get_repository", {})
    result = await server.verify(http_request(), verify_body("answer"))
    assert result.mask_sample is True
    assert result.failure_kind == "mcp_atlas:service_error"
    assert len(result.service_errors) == 1
    assert text in result.service_errors[0]


@pytest.mark.parametrize(
    ("status", "text"),
    [
        (401, "authentication rejected"),
        (401, "Failed to call tool: HTTP 403 Forbidden"),
        (401, "Failed to call tool: HTTP 403: API rate limit exceeded for synthetic account"),
        (403, "insufficient scope"),
        (500, "Failed to call tool: HTTP 403 Forbidden: insufficient scope"),
        (500, "Failed to call tool 'github_get_repository': 500: Tool 'github_get_repository' execution failed: 401"),
    ],
    ids=["401", "outer401-conflicting403", "outer401-conflicting-rate", "403", "wrapped403", "nested401"],
)
async def test_auth_failure_rejects_resources_until_close_without_calling_tools_or_judge(
    environment, status, text, caplog
) -> None:
    env, url = environment
    env.results["github_get_repository"] = (
        status,
        [{"type": "text", "text": text}] if status == 200 else {"detail": text},
    )
    client = judge_client()
    server = make_server(url, client)
    await seed(server)
    for _ in range(2):
        with pytest.raises(HTTPException) as rejected:
            await server.call_tool(http_request(), "github_get_repository", {})
        assert rejected.value.status_code == 422
        assert (
            rejected.value.detail
            == "MCP-Atlas authentication failed for github_get_repository; check credentials/scopes. "
            "The environment is unavailable; give your final answer."
        )
        assert text not in rejected.value.detail
    assert len(env.calls) == 1
    diagnostics = [
        record.getMessage() for record in caplog.records if record.name == MCPAtlasResourcesServer.__module__
    ]
    assert len(diagnostics) == 1 and "give your final answer" in diagnostics[0]
    assert text not in diagnostics[0]
    for _ in range(2):
        with pytest.raises(HTTPException) as rejected:
            await judge_failsafe(server.verify)(http_request(), verify_body("answer"))
        assert rejected.value.status_code == 422
    client.post.assert_not_awaited()
    await server.close_resources_session(
        http_request(),
        ResourcesCloseSessionRequest(resources_session_id="s1", episode_id=EpisodeId(rollout_id="episode")),
    )
    with pytest.raises(HTTPException) as closed:
        await server.verify(http_request(), verify_body("answer"))
    assert closed.value.status_code == 422
    client.post.assert_not_awaited()


@pytest.mark.parametrize("field", ["servers", "enabled_tools"])
@pytest.mark.parametrize("value", [None, 7, "calculator", [7], {}])
async def test_bad_seed_metadata_is_rejected_before_provider_allocation(field, value) -> None:
    server = make_server("http://unused", judge_client())
    provider = AsyncMock()
    server._pool = provider
    request = MCPAtlasSeedSessionRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        verifier_metadata={**META, field: value},
    )
    with pytest.raises(HTTPException) as rejected:
        await server.seed_session(http_request(), request)
    assert rejected.value.status_code == 422
    provider.start.assert_not_awaited()


@pytest.mark.parametrize(
    "text",
    [
        "Failed to call tool: Timed out running the program",
        "Failed to call tool 'mcp-code-executor_execute_code': 500: Tool 'mcp-code-executor_execute_code' execution failed: Execution timed out",
    ],
)
async def test_returned_program_deadline_is_scored_at_the_tool_response_boundary(text) -> None:
    # These are controlled error envelopes, not claims about the installed executor's deadline behavior.
    code_tool = "mcp-code-executor_execute_code"
    env = FakeAgentEnvironment([code_tool], {code_tool: (500, {"detail": text})})
    url = await env.start()
    try:
        server = make_server(url, judge_client("not_fulfilled", "not_fulfilled"))
        await seed(server, metadata={**META, "enabled_tools": [code_tool], "servers": ["mcp-code-executor"]})
        assert text in await server.call_tool(http_request(), code_tool, {"code": "while True: pass"})
        result = await server.verify(http_request(), verify_body("wrong answer"))
        assert len(env.calls) == 1
        assert result.reward == 0 and result.mask_sample is False and result.service_errors == []
    finally:
        await env.stop()


async def test_default_bridge_budget_allows_a_delayed_program_timeout_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omegaconf import OmegaConf

    import resources_servers.mcp_atlas.app as app_module
    from resources_servers.mcp_atlas.sandbox import call_sandbox_tool

    class DelayedEnvironment(FakeAgentEnvironment):
        async def _call_tool(self, request: web.Request) -> web.Response:
            # Scale the real HTTP deadline below, so this exceeds the former 60 s budget without
            # spending minutes in a unit test. Only the external tool environment is synthetic.
            await asyncio.sleep(0.12)
            return await super()._call_tool(request)

    code_tool = "mcp-code-executor_execute_code"
    text = "Failed to call tool: Timed out running the program"
    env = DelayedEnvironment([code_tool], {code_tool: (500, {"detail": text})})
    url = await env.start()
    budgets: list[float] = []

    async def scaled_transport(
        box: AtlasSandbox, name: str, arguments: dict[str, object], *, timeout_s: float
    ) -> tuple[int, object]:
        budgets.append(timeout_s)
        return await call_sandbox_tool(box, name, arguments, timeout_s=timeout_s / 1000)

    monkeypatch.setattr(app_module, "call_sandbox_tool", scaled_transport)
    try:
        server = make_server(url, judge_client("not_fulfilled", "not_fulfilled"))
        await seed(server, metadata={**META, "enabled_tools": [code_tool], "servers": ["mcp-code-executor"]})
        output = await server.call_tool(http_request(), code_tool, {"code": "while True: pass"})
        result = await server.verify(http_request(), verify_body("wrong answer"))
        assert text in output and len(env.calls) == 1
        assert result.reward == 0 and result.mask_sample is False and result.service_errors == []
        assert budgets and all(budget > 1800 for budget in budgets)
        official = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/mcp_atlas.yaml")
        assert budgets == [official.mcp_atlas.resources_servers.mcp_atlas.tool_call_timeout_s]
    finally:
        await env.stop()


@pytest.mark.parametrize("mask_service_errors", [False, True])
async def test_bridge_timeout_remains_an_infrastructure_failure_even_for_a_code_tool(
    environment: tuple[FakeAgentEnvironment, str], monkeypatch: pytest.MonkeyPatch, mask_service_errors: bool
) -> None:
    import resources_servers.mcp_atlas.app as app_module

    _, url = environment
    code_tool = "mcp-code-executor_execute_code"
    monkeypatch.setattr(app_module, "call_sandbox_tool", AsyncMock(side_effect=asyncio.TimeoutError))
    server = make_server(url, judge_client("not_fulfilled", "not_fulfilled"), mask_service_errors=mask_service_errors)
    await seed(server, metadata={**META, "enabled_tools": [code_tool], "servers": ["mcp-code-executor"]})
    await server.call_tool(http_request(), code_tool, {"code": "print(1)"})
    result = await server.verify(http_request(), verify_body("answer"))
    assert result.mask_sample is True
    assert result.failure_kind == "mcp_atlas:sandbox_unreachable"
    assert result.service_errors == ["sandbox_unreachable: mcp-code-executor_execute_code: transport timeout"]


async def test_legacy_saved_tool_timeout_retains_its_original_environment_attribution() -> None:
    server = make_server("http://unused", judge_client("not_fulfilled", "not_fulfilled"))
    body = verify_body("wrong answer")
    body.__pydantic_extra__ = {"service_errors": ["tool_timeout: legacy bridge deadline"], "tool_calls": 1}
    result = await server.verify(http_request("reverify"), body)
    assert result.reward == 0 and result.mask_sample is True
    assert result.failure_kind == "mcp_atlas:tool_timeout"


async def test_closed_auth_session_cannot_bypass_rejection_with_saved_fields_but_fresh_reverify_works(
    environment,
) -> None:
    env, url = environment
    env.results["github_get_repository"] = (401, {"detail": "Unauthorized"})
    client = judge_client("not_fulfilled", "not_fulfilled")
    server = make_server(url, client)
    await seed(server)
    with pytest.raises(HTTPException):
        await server.call_tool(http_request(), "github_get_repository", {})
    await server.close_resources_session(
        http_request(),
        ResourcesCloseSessionRequest(resources_session_id="s1", episode_id=EpisodeId(rollout_id="episode")),
    )
    body = verify_body("wrong answer")
    body.__pydantic_extra__ = {"service_errors": [], "tool_calls": 0}
    with pytest.raises(HTTPException) as closed:
        await server.verify(http_request(), body)
    assert closed.value.status_code == 422
    client.post.assert_not_awaited()
    result = await server.verify(http_request("fresh-artifact-reverify"), body)
    assert result.reward == 0 and result.mask_sample is False and result.service_errors == []
    assert client.post.await_count == 2


@pytest.mark.parametrize("service_failure", [False, True])
async def test_single_server_error_wrappers_use_the_served_tool_name(service_failure: bool) -> None:
    server = make_server("http://unused", judge_client("not_fulfilled", "not_fulfilled"))
    server._pool = pool(["get_repository"])
    try:
        await server.seed_session(
            http_request(),
            MCPAtlasSeedSessionRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                verifier_metadata={**META, "enabled_tools": ["github_get_repository"], "servers": ["github"]},
            ),
        )
        env = FakeSandbox.instances[-1].environment
        diagnostic = "HTTP 429 Too Many Requests" if service_failure else "Object 502 not found"
        text = "Failed to call tool 'get_repository': 500: Tool 'get_repository' execution failed: " + diagnostic
        env.results["get_repository"] = (500, {"detail": text})
        assert text in await server.call_tool(http_request(), "github_get_repository", {})
        assert env.calls[-1]["tool_name"] == "get_repository"
        result = await server.verify(http_request(), verify_body("wrong answer"))
        assert result.reward == 0.0
        assert result.mask_sample is service_failure
        assert result.failure_kind == ("mcp_atlas:service_error" if service_failure else None)
        assert len(result.service_errors) == int(service_failure)
    finally:
        await server._shutdown()


async def test_unreachable_sandbox_masks_the_rollout() -> None:
    server = make_server("http://127.0.0.1:9", judge_client("fulfilled", "fulfilled"))
    await seed(server)
    output = await server.call_tool(http_request(), "calculator_calculate", {"expression": "1"})
    assert output.startswith("Error: tool call failed")
    result = await server.verify(http_request(), verify_body("Created 2013, registered 2006."))
    assert result.mask_sample is True
    assert result.failure_kind == "mcp_atlas:sandbox_unreachable"


async def test_calls_after_the_sandbox_stopped_are_refused_and_masked(environment) -> None:
    env, url = environment
    server = make_server(url, judge_client("fulfilled", "fulfilled"))
    await seed(server)
    server._rollouts["s1"].box.stopped = True  # as the reaper leaves it
    output = await server.call_tool(http_request(), "calculator_calculate", {"expression": "1"})
    assert output == "Error: tool call failed: SandboxError"
    assert env.calls == []
    result = await server.verify(http_request(), verify_body("Created 2013, registered 2006."))
    assert result.mask_sample is True
    assert result.failure_kind == "mcp_atlas:sandbox_unreachable"


def close_body(resources_session_id: str = "run:r1") -> ResourcesCloseSessionRequest:
    return ResourcesCloseSessionRequest(
        resources_session_id=resources_session_id, episode_id=EpisodeId(rollout_id="r1")
    )


async def test_close_stops_the_sandbox_and_refuses_a_later_seed() -> None:
    server = make_server("http://unused", judge_client())
    server._pool = pool(TOOLS)
    await seed(server)
    box = server._rollouts["s1"].box
    # A /run close names its run in the body; the cookie names the session this server seeded.
    await server.close_resources_session(http_request(), close_body())
    assert box.stopped
    assert "s1" not in server._rollouts and server._pool.get("s1") is None
    await server.close_resources_session(http_request(), close_body())
    await server.close_resources_session(http_request("never-seeded"), close_body("run:never-seeded"))
    with pytest.raises(ValueError, match="already closed"):
        await seed(server)


async def test_a_close_whose_sandbox_does_not_stop_can_be_retried() -> None:
    server = make_server("http://unused", judge_client())
    server._pool = pool(TOOLS, stop_errors=1)
    await seed(server)
    with pytest.raises(SandboxError, match="sandbox stop failed"):
        await server.close_resources_session(http_request(), close_body())
    assert "s1" in server._rollouts and "s1" not in server._closed_sessions
    await server.close_resources_session(http_request(), close_body())
    assert server._pool.get("s1") is None and "s1" in server._closed_sessions


async def test_reverify_judges_a_saved_row_again(environment) -> None:
    env, url = environment
    env.results["github_get_repository"] = (500, {"detail": "Failed to call tool: 429 Too Many Requests"})
    server = make_server(url, judge_client("fulfilled", "partially_fulfilled", "fulfilled", "fulfilled"))
    await seed(server)
    await server.call_tool(http_request(), "github_get_repository", {"owner": "x"})
    first = await server.verify(http_request(), verify_body("Created 2013; domain from about 2006."))
    assert first.coverage == 0.75
    # gym eval reverify posts the saved row, with its computed fields, and no session.
    saved = MCPAtlasVerifyRequest.model_validate(first.model_dump(mode="json"))
    again = await server.verify(http_request("reverify"), saved)
    assert again.coverage == 1.0
    assert again.tool_calls == 1
    assert again.service_errors == first.service_errors
    assert again.mask_sample is True and again.failure_kind == "mcp_atlas:service_error"


@pytest.mark.parametrize("errors", [[], ["service_error: HTTP 429"]])
async def test_mcp_reverification_rejudges_answer_and_retains_environment(errors: list[str]) -> None:
    client = judge_client("fulfilled")
    server = make_server("http://unused", client)
    payload = build_reverification_payload(
        input_row={
            "responses_create_params": {"input": []},
            "verifier_metadata": {"task_id": "a", "claims": ["A claim"]},
        },
        saved_row={
            "response": {
                "id": "r",
                "created_at": 0,
                "model": "policy",
                "object": "response",
                "output": [
                    {
                        "id": "message",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "The answer.", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
            "service_errors": errors,
            "tool_calls": 1,
            "reward": 0.0,
            "_ng_failure_class": "judge_failed",
        },
    )
    result = await server.verify(
        SimpleNamespace(session={}, cookies={}), MCPAtlasVerifyRequest.model_validate(payload)
    )

    assert result.reward == 1.0 and result.coverage == 1.0
    assert result.mask_sample is bool(errors)
    assert result.failure_kind == ("mcp_atlas:service_error" if errors else None)
    assert result.service_errors == errors and result.tool_calls == 1
    client.post.assert_awaited_once()
    assert "The answer." in client.post.await_args.kwargs["json"]["messages"][0]["content"]


async def test_verify_without_a_session_is_masked(environment) -> None:
    _, url = environment
    server = make_server(url, judge_client())
    result = await server.verify(http_request("never-seeded"), verify_body("text"))
    assert result.mask_sample is True
    assert result.failure_kind == "mcp_atlas:session_lost"
    assert result.reward == 0.0


async def test_unreadable_verdicts_are_retried(environment) -> None:
    _, url = environment
    client = judge_client("not json", "fulfilled", "partially_fulfilled")
    server = make_server(url, client, judge_max_attempts=2)
    await seed(server)
    result = await server.verify(http_request(), verify_body("Created 2013; domain from about 2006."))
    assert result.coverage == 0.75
    assert client.post.await_count == 3


async def test_judge_error_after_the_last_unreadable_verdict(environment) -> None:
    _, url = environment
    client = judge_client("not json", '{"coverage_outcome": "maybe"}', "", "[]")
    server = make_server(url, client, judge_max_attempts=2)
    await seed(server)
    # The measurement fails instead of scoring the claims 0.
    with pytest.raises(JudgeError, match="no readable judge verdict after 2 attempts"):
        await server.verify(http_request(), verify_body("answer"))


async def test_failed_claim_cancels_and_drains_other_judge_requests(environment) -> None:
    _, url = environment
    started = asyncio.Event()
    cancelled = asyncio.Event()
    finished = asyncio.Event()
    release = asyncio.Event()
    cleanup_release = asyncio.Event()
    unreadable = next(judge_client("not json").post.side_effect)
    client = MagicMock(spec=ServerClient)

    async def post(**kwargs):
        prompt = kwargs["json"]["messages"][0]["content"]
        if META["claims"][0] in prompt:
            await started.wait()
            return unreadable
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await cleanup_release.wait()
            raise
        finally:
            finished.set()
        return unreadable

    client.post = AsyncMock(side_effect=post)
    server = make_server(url, client, judge_max_attempts=1, judge_max_concurrency=2)
    await seed(server)
    verification = asyncio.create_task(judge_failsafe(server.verify)(http_request(), verify_body("answer")))
    try:
        await asyncio.wait_for(cancelled.wait(), 2)
        assert not verification.done(), "verify must wait for the pending request's asynchronous cleanup"
        cleanup_release.set()
        failed = await asyncio.wait_for(verification, 2)
        saved = json.loads(failed.body)
        assert saved["failure_kind"] == "judge_failed"
        assert saved["mask_sample"] is True
        assert cancelled.is_set(), "verify must drain the pending judge request before returning"
        assert finished.is_set(), "verify must await the judge request's cancellation finalizer"
        assert client.post.await_count == 2
        assert "s1" not in server._rollouts
    finally:
        release.set()
        cleanup_release.set()
        if not verification.done():
            verification.cancel()
        await asyncio.gather(verification, return_exceptions=True)


async def test_cancelled_verification_drains_judge_requests_and_retires_session(environment) -> None:
    _, url = environment
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    cleanup_release = asyncio.Event()
    requests_started = 0
    requests_cancelled = 0
    requests_finished = 0
    client = MagicMock(spec=ServerClient)

    async def post(**kwargs):
        nonlocal requests_started, requests_cancelled, requests_finished
        requests_started += 1
        if requests_started == 2:
            started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            requests_cancelled += 1
            if requests_cancelled == 2:
                cancelled.set()
            await cleanup_release.wait()
            raise
        finally:
            requests_finished += 1

    client.post = AsyncMock(side_effect=post)
    server = make_server(url, client, judge_max_concurrency=2)
    await seed(server)
    verification = asyncio.create_task(server.verify(http_request(), verify_body("answer")))
    try:
        await asyncio.wait_for(started.wait(), 2)
        verification.cancel()
        await asyncio.wait_for(cancelled.wait(), 2)
        assert not verification.done()
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(verification, 2)
        assert requests_finished == 2
        assert client.post.await_count == 2
        assert "s1" not in server._rollouts
    finally:
        release.set()
        cleanup_release.set()
        if not verification.done():
            verification.cancel()
        await asyncio.gather(verification, return_exceptions=True)


async def test_failed_judge_call_is_routed_to_judge_failures(environment) -> None:
    _, url = environment
    client = MagicMock(spec=ServerClient)
    failed = AsyncMock()
    failed.ok = False
    failed.raise_for_status = MagicMock(side_effect=RuntimeError("HTTP 503"))
    failed.content.read = AsyncMock(return_value=b"unavailable")
    client.post = AsyncMock(return_value=failed)
    server = make_server(url, client)
    await seed(server)
    failed_response = await judge_failsafe(server.verify)(http_request(), verify_body("answer"))
    saved = json.loads(failed_response.body)
    assert saved["failure_kind"] == "judge_failed"
    assert saved["_ng_failure_class"] == "judge_failed"
    assert saved["mask_sample"] is True
    assert "HTTP 503" in saved["failure_reason"]


async def test_close_waits_for_provisioning_and_prevents_session_resurrection() -> None:
    ready_gate = asyncio.Event()
    server = make_server("http://unused", judge_client())
    server._pool = pool(TOOLS, ready_gate=ready_gate)
    seeding = asyncio.create_task(seed(server))
    await asyncio.sleep(0)
    sandbox = FakeSandbox.instances[-1]
    await asyncio.wait_for(sandbox.exec_started.wait(), 2)
    closing = asyncio.create_task(server.close_resources_session(http_request(), close_body()))
    await asyncio.sleep(0)
    assert not closing.done()
    # An unrelated session can close while s1 is blocked.
    await asyncio.wait_for(server.close_resources_session(http_request("other"), close_body("other")), 2)
    ready_gate.set()
    await asyncio.wait_for(seeding, 2)
    await asyncio.wait_for(closing, 2)
    assert sandbox.stopped
    assert server._pool.get("s1") is None
    assert "s1" not in server._rollouts
    with pytest.raises(ValueError, match="already closed"):
        await seed(server)


async def test_concurrent_seeds_replace_only_after_the_first_sandbox_is_owned() -> None:
    ready_gate = asyncio.Event()
    server = make_server("http://unused", judge_client())
    server._pool = pool(TOOLS, ready_gate=ready_gate)
    first = asyncio.create_task(seed(server))
    await asyncio.sleep(0)
    first_sandbox = FakeSandbox.instances[-1]
    await asyncio.wait_for(first_sandbox.exec_started.wait(), 2)
    second = asyncio.create_task(seed(server))
    await asyncio.sleep(0)
    assert not second.done()
    ready_gate.set()
    await asyncio.wait_for(asyncio.gather(first, second), 2)
    assert first_sandbox.stopped
    assert server._pool.get("s1") is server._rollouts["s1"].box
    assert server._rollouts["s1"].box.sandbox is not first_sandbox
    await server.close_resources_session(http_request(), close_body())
    assert server._pool.get("s1") is None


async def test_checkpoint_retirement_retries_failed_teardown_and_releases_capacity() -> None:
    server = make_server("http://unused", judge_client())
    server._pool = pool(TOOLS, stop_errors=1)
    await seed(server)
    box = server._rollouts["s1"].box
    episode = EpisodeId(rollout_id="r1")
    participant = ResourcesParticipant(server, "restart_only")
    participant.seeded("s1", episode)
    with pytest.raises(SandboxError, match="sandbox stop failed"):
        await participant.retire(episode)
    assert not box.stopped
    assert server._pool.get("s1") is box
    assert "s1" in server._rollouts
    await participant.retire(episode)
    assert box.stopped
    assert server._pool.get("s1") is None
    assert "s1" not in server._rollouts
    await participant.retire(episode)
    # Retirement releases ownership without fencing later checkpoint restoration of the same ID.
    assert box.port not in server._pool._ports._leased
    await asyncio.wait_for(seed(server), 2)
    assert not server._rollouts["s1"].box.stopped
    with pytest.raises(SandboxError, match="1 sandboxes did not stop"):
        await server._shutdown()
    await server._shutdown()


async def test_shutdown_warns_about_manual_cleanup_and_preserves_retryable_ownership(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = make_server("http://unused", judge_client())
    server._pool = pool(TOOLS, stop_errors=1)
    await seed(server)
    box = server._rollouts["s1"].box
    try:
        with caplog.at_level("ERROR", logger="resources_servers.mcp_atlas.app"):
            with pytest.raises(SandboxError, match="1 sandboxes did not stop"):
                await server._shutdown()
        assert server._pool.get("s1") is box
        assert not box.stopped
        messages = [
            record.getMessage() for record in caplog.records if record.name == "resources_servers.mcp_atlas.app"
        ]
        assert any(
            "manual cleanup may be required" in text and "Apptainer does not enforce ttl_s" in text
            for text in messages
        )
    finally:
        await server._shutdown()
    assert box.stopped
    assert server._pool.get("s1") is None


async def test_retirement_drops_only_local_state_for_a_shared_development_environment(environment) -> None:
    env, url = environment
    server = make_server(url, judge_client())
    await seed(server)
    await server.retire_session_state("s1")
    assert "s1" not in server._rollouts
    await seed(server, "s2")
    assert await server.call_tool(http_request("s2"), "calculator_calculate", {}) == "14"
    assert env.calls[-1]["tool_name"] == "calculator_calculate"


@pytest.mark.parametrize("named", [False, True])
def test_provider_configuration_resolves_defaults_and_metadata(tmp_path, named: bool) -> None:
    image = tmp_path / "atlas.sif"
    image.touch()
    provider = {
        "apptainer": {
            "create": {"apply_resource_limits": False, "extra_start_args": ["--cleanenv"]},
            "exec": {"extra_exec_args": ["--cleanenv"]},
        }
    }
    block = {**provider, "default_metadata": {"team": "atlas"}}
    client = judge_client()
    client.global_config_dict = {"sandbox": block}
    config = MCPAtlasResourcesServerConfig(
        host="0.0.0.0",
        port=0,
        entrypoint="app.py",
        name="mcp_atlas",
        sandbox_image=str(image),
        sandbox_provider="sandbox" if named else block,
        sandbox_development_mode=True,
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
    )
    server = MCPAtlasResourcesServer(config=config, server_client=client)
    assert server._pool.sandbox_provider == provider
    assert server._pool.sandbox_metadata == {"team": "atlas"}


@pytest.mark.parametrize("service_failure", [False, True])
async def test_judge_failsafe_preserves_environment_snapshot(environment, service_failure: bool) -> None:
    env, url = environment
    if service_failure:
        env.results["github_get_repository"] = (500, {"detail": "Failed to call tool: 429 Too Many Requests"})
    client = judge_client("unreadable", "unreadable")
    server = make_server(url, client, judge_max_attempts=1)
    await seed(server)
    tool = "github_get_repository" if service_failure else "calculator_calculate"
    await server.call_tool(http_request(), tool, {})
    failed = await judge_failsafe(server.verify)(http_request(), verify_body("answer"))
    saved = json.loads(failed.body)
    assert saved["_ng_failure_class"] == "judge_failed"
    assert saved["tool_calls"] == 1
    if service_failure:
        assert len(saved["service_errors"]) == 1
        assert "429" in saved["service_errors"][0]
    else:
        assert saved["service_errors"] == []
    client.post.side_effect = judge_client("fulfilled", "fulfilled").post.side_effect
    again = await server.verify(http_request("reverify"), MCPAtlasVerifyRequest.model_validate(saved))
    assert again.reward == 1.0
    assert again.coverage == 1.0
    assert again.tool_calls == 1
    assert again.mask_sample is service_failure
    assert again.service_errors == saved["service_errors"]


@pytest.mark.parametrize(
    "tool,servers",
    [
        ("filesystem_read_file", ["filesystem"]),
        ("mcp-code-executor_execute_code", ["mcp-code-executor"]),
        ("github_get_repository", ["github"]),
    ],
)
@pytest.mark.parametrize(
    "text",
    [
        "Error: HTTP 401 Unauthorized",
        "Error: Forbidden",
        "Error: HTTP 502 Bad Gateway",
        "Error: Unauthorized " + "diagnostic " * 300,
    ],
)
async def test_successful_tool_content_is_not_interpreted_as_a_failure(tool, servers, text) -> None:
    env = FakeAgentEnvironment([tool], {tool: (200, [{"type": "text", "text": text}])})
    url = await env.start()
    try:
        client = judge_client("not_fulfilled", "not_fulfilled")
        server = make_server(url, client)
        await seed(server, metadata={**META, "servers": servers, "enabled_tools": [tool]})
        assert await server.call_tool(http_request(), tool, {}) == text
        result = await server.verify(http_request(), verify_body("wrong answer"))
        assert not result.mask_sample and result.reward == 0 and result.service_errors == []
        assert len(env.calls) == 1 and client.post.await_count == 2
    finally:
        await env.stop()


@pytest.mark.parametrize(
    "tool,server_name",
    [
        ("mcp-code-executor_execute_code", "mcp-code-executor"),
        ("filesystem_read_file", "filesystem"),
    ],
)
async def test_program_and_file_errors_do_not_claim_service_authentication(tool, server_name) -> None:
    text = f"Failed to call tool '{tool}': 500: Tool '{tool}' execution failed: HTTP 403 Forbidden"
    env = FakeAgentEnvironment([tool], {tool: (500, {"detail": text})})
    url = await env.start()
    try:
        server = make_server(url, judge_client("not_fulfilled", "not_fulfilled"))
        await seed(server, metadata={**META, "servers": [server_name], "enabled_tools": [tool]})
        assert text in await server.call_tool(http_request(), tool, {})
        result = await server.verify(http_request(), verify_body("wrong answer"))
        assert result.reward == 0 and not result.mask_sample and result.service_errors == []
    finally:
        await env.stop()


@pytest.mark.parametrize(
    "status,prefix",
    [
        (403, ""),
        (500, "Failed to call tool: "),
        (500, "Failed to call tool: HTTP 403: "),
        (
            500,
            "Failed to call tool 'github_get_repository': 500: Tool 'github_get_repository' execution failed: HTTP 403 ",
        ),
    ],
)
@pytest.mark.parametrize(
    "message",
    [
        "API rate limit exceeded for synthetic account",
        "You have exceeded a secondary rate limit. Please wait a few minutes before you try again.",
    ],
)
async def test_github_rate_limits_remain_service_failures_not_sticky_credentials(
    environment, status, prefix, message
) -> None:
    env, url = environment
    env.results["github_get_repository"] = (status, {"detail": prefix + message})
    client = judge_client("not_fulfilled", "not_fulfilled")
    server = make_server(url, client)
    await seed(server)
    await server.call_tool(http_request(), "github_get_repository", {})
    await server.call_tool(http_request(), "github_get_repository", {})
    result = await server.verify(http_request(), verify_body("wrong answer"))
    assert result.mask_sample and result.failure_kind == "mcp_atlas:service_error"
    assert result.tool_calls == len(env.calls) == 2
    assert client.post.await_count == 2


@pytest.mark.parametrize("code", ["invalid_auth", "not_authed", "token_expired", "token_revoked", "missing_scope"])
async def test_slack_auth_codes_in_trusted_tool_errors_reject_before_judging(code) -> None:
    tool = "slack_conversations_history"
    text = f"Failed to call tool '{tool}': 500: Tool '{tool}' execution failed: {code}"
    env = FakeAgentEnvironment([tool], {tool: (500, {"detail": text})})
    url = await env.start()
    try:
        client = judge_client()
        server = make_server(url, client)
        await seed(server, metadata={**META, "servers": ["slack"], "enabled_tools": [tool]})
        with pytest.raises(HTTPException) as rejected:
            await server.call_tool(http_request(), tool, {})
        assert rejected.value.status_code == 422 and code not in rejected.value.detail
        with pytest.raises(HTTPException) as rejected:
            await server.verify(http_request(), verify_body("answer"))
        assert rejected.value.status_code == 422
        client.post.assert_not_awaited()
    finally:
        await env.stop()


async def test_ordinary_closed_verification_is_masked_and_cannot_replay_saved_fields(environment) -> None:
    _, url = environment
    client = judge_client()
    server = make_server(url, client)
    await seed(server)
    await server.close_resources_session(
        http_request(),
        ResourcesCloseSessionRequest(resources_session_id="s1", episode_id=EpisodeId(rollout_id="episode")),
    )
    body = verify_body("answer")
    body.__pydantic_extra__ = {"service_errors": [], "tool_calls": 7}
    result = await server.verify(http_request(), body)
    assert result.mask_sample and result.failure_kind == "mcp_atlas:session_lost"
    assert result.tool_calls == 0 and result.reward == 0
    client.post.assert_not_awaited()


@pytest.mark.parametrize(
    "status,text",
    [
        (403, "Forbidden"),
        (500, "Failed to call tool: Forbidden"),
        (500, "Failed to call tool: HTTP 403 Forbidden"),
        (
            500,
            "Failed to call tool 'github_get_repository': 500: Tool 'github_get_repository' execution failed: HTTP 403 Forbidden",
        ),
    ],
)
async def test_unqualified_forbidden_errors_are_not_invalid_credential_proof(environment, status, text) -> None:
    env, url = environment
    env.results["github_get_repository"] = (status, {"detail": text})
    client = judge_client("not_fulfilled", "not_fulfilled")
    server = make_server(url, client)
    await seed(server)
    await server.call_tool(http_request(), "github_get_repository", {})
    await server.call_tool(http_request(), "github_get_repository", {})
    result = await server.verify(http_request(), verify_body("wrong answer"))
    assert result.mask_sample and result.failure_kind == "mcp_atlas:service_error"
    assert result.tool_calls == len(env.calls) == 2
    assert client.post.await_count == 2


def test_startup_rejects_host_credentials_even_with_no_dataset_inventory(tmp_path) -> None:
    image = tmp_path / "development.sif"
    image.touch()
    forbidden_value = "synthetic-host-only-value"
    client = judge_client()
    assert client.global_config_dict == {}
    with pytest.raises(ValueError) as rejected:
        make_server(
            "",
            client,
            sandbox_image=str(image),
            sandbox_provider=APPTAINER_PROVIDER,
            sandbox_env={"CUSTOM_JUDGE_TOKEN": forbidden_value},
            service_env_names=[],
        )
    assert "CUSTOM_JUDGE_TOKEN" in str(rejected.value)
    assert forbidden_value not in str(rejected.value)


async def test_empty_task_server_list_is_rejected_before_provider_allocation() -> None:
    server = make_server("http://unused", judge_client())
    provider = AsyncMock()
    server._pool = provider
    body = MCPAtlasSeedSessionRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        verifier_metadata={**META, "servers": []},
    )
    with pytest.raises(HTTPException) as rejected:
        await server.seed_session(http_request(), body)
    assert rejected.value.status_code == 422
    provider.start.assert_not_awaited()
