# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import cadquery as cq
import pytest

from nemo_gym.core import PROVIDER_UNAVAILABLE, VERIFIER_ERROR
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_runtime.base_server import ServerClient
from resources_servers.deepcad.app import (
    DeepCADResourcesServer,
    DeepCADResourcesServerConfig,
    DeepCADVerifyRequest,
    assistant_text,
    extract_code,
)
from resources_servers.deepcad.geometry import load_solid, score


def request(text: str) -> DeepCADVerifyRequest:
    return DeepCADVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "three views"}]},
            "verifier_metadata": {"task_id": "box", "reference_step": "box.step"},
            "response": {
                "id": "resp_test",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "output": [
                    {
                        "id": "msg",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        }
    )


@pytest.fixture
def server(tmp_path: Path) -> DeepCADResourcesServer:
    cq.exporters.export(cq.Workplane("XY").box(1, 2, 3), str(tmp_path / "box.step"))
    config = DeepCADResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="deepcad",
        reference_root=str(tmp_path),
        sandbox_provider={"docker": {}},
        voxel_resolution=64,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return DeepCADResourcesServer(config=config, server_client=client)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("result = cq.Workplane('XY').box(1,2,3)", "result = cq.Workplane('XY').box(1,2,3)"),
        ("prose\n```python\nresult = 1\n```", "result = 1"),
        ("```py\nresult = 1\n```\n```python\nresult = 2\n```", "result = 2"),
        ("<think>```python\nbad\n```</think>\n```python\ngood\n```", "good"),
        ("hidden reasoning</thinking>\n```\ngood\n```", "good"),
        ("<thinking>secret</thinking>good", "good"),
        ("", ""),
    ],
)
def test_extract_code(text: str, expected: str) -> None:
    assert extract_code(text) == expected


def test_last_assistant_message() -> None:
    body = request("first")
    second = body.response.output[0].model_copy(deep=True)
    second.content[0].text = "last"
    body.response.output.append(second)
    assert assistant_text(body) == "last"
    body.response.output.clear()
    assert assistant_text(body) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, kind", [("", "code"), ("if:", "syntax"), ("x" * 65537, "code")], ids=["empty", "syntax", "oversized"]
)
async def test_invalid_code_zero(server: DeepCADResourcesServer, text: str, kind: str) -> None:
    response = await server.verify(request(text))
    assert response.reward == 0 and not response.mask_sample
    assert response.failure_kind == f"deepcad:{kind}"
    assert set(response.reward_components) == {"voxel_iou"}
    assert sum(response.reward_components.values()) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RecursionError, MemoryError])
async def test_compile_resource_errors_are_unmasked_zero(server, error):
    server._evaluate = AsyncMock()
    with patch("resources_servers.deepcad.app.compile", side_effect=error("compile failed"), create=True):
        result = await server.verify(request("result = 1"))
    assert result.reward == 0 and not result.mask_sample
    assert result.failure_kind == "deepcad:syntax"
    assert result.failure_reason == "compile failed"
    server._evaluate.assert_not_awaited()


@pytest.mark.parametrize("mixed_iou", [None, {}])
def test_geometry_requires_positive_weight_even_with_unused_mixed_config(mixed_iou):
    with pytest.raises(ValueError, match="Geometry scoring requires at least one positive reward weight"):
        DeepCADResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="deepcad",
            mixed_iou=mixed_iou,
            reward_weights={"voxel_iou": 0},
        )


@pytest.mark.asyncio
async def test_continuous_reward(server: DeepCADResourcesServer) -> None:
    with patch.object(DeepCADResourcesServer, "_evaluate", AsyncMock(return_value={"reward": 0.37})):
        assert (await server.verify(request("result = 1"))).reward == 0.37


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["missing.step", "../outside.step"])
async def test_reference_mask(server: DeepCADResourcesServer, reference: str) -> None:
    result = await server._evaluate("result=1", reference)
    assert result["mask_sample"] and result["reward"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("bonus_enabled", [False, True])
@pytest.mark.parametrize(
    "probe, execution, check, kind, mask",
    [
        (SandboxExecResult("", "", 1), None, None, "infrastructure", True),
        (SandboxExecResult("", "", 0), SandboxExecResult("", "", 1), None, "execution", False),
        (SandboxExecResult("", "", 0), SandboxExecResult("", "", 124, "timeout"), None, "execution", False),
        (SandboxExecResult("", "", 0), SandboxExecResult("", "", 125, "runtime"), None, "infrastructure", True),
        (SandboxExecResult("", "", 0), SandboxExecResult("", "", 0), SandboxExecResult("0", "", 0), "artifact", False),
        (
            SandboxExecResult("", "", 0),
            SandboxExecResult("", "", 0),
            SandboxExecResult("huge", "", 0),
            "artifact",
            False,
        ),
        (
            SandboxExecResult("", "", 0),
            SandboxExecResult("", "", 0),
            SandboxExecResult("999999999", "", 0),
            "artifact",
            False,
        ),
        (
            SandboxExecResult("", "", 0),
            SandboxExecResult("", "", 0),
            SandboxExecResult("100", "", 1),
            "artifact",
            False,
        ),
        (
            SandboxExecResult("", "", 0),
            SandboxExecResult("", "", 0),
            SandboxExecResult("100", "", 0, "timeout"),
            "infrastructure",
            True,
        ),
        (SandboxExecResult("", "", 0), SandboxExecResult("", "", 0), SandboxExecResult("100", "", 0), None, False),
    ],
)
async def test_sandbox_outcomes(
    server: DeepCADResourcesServer, probe, execution, check, kind, mask, bonus_enabled, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.config.inconclusive_verification_retries = 0
    server.config.operation_bonus.enabled = bonus_enabled
    monkeypatch.setenv("BASH_FUNC_exit-dramatically%%", "() { exit 1; }")
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [
        probe,
        execution,
        check,
        SandboxExecResult('{"path":"/tmp/deepcad-sealed-test/prediction.step"}', "", 0),
    ]
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=sandbox)
    context.__aexit__ = AsyncMock(return_value=False)
    with patch("resources_servers.deepcad.app.AsyncSandbox", return_value=context) as factory:
        with patch.object(DeepCADResourcesServer, "_score", AsyncMock(return_value={"reward": 0.82})):
            result = await server._evaluate("result=1", "box.step")
    assert result.get("mask_sample", False) is mask
    if kind:
        assert result["reward"] == 0 and result["failure_kind"] == (
            PROVIDER_UNAVAILABLE if kind == "infrastructure" else f"deepcad:{kind}"
        )
    else:
        assert result["reward"] == 0.82
        sandbox.download.assert_awaited_once()
        assert sandbox.exec.await_count == 4
        seal = sandbox.exec.call_args_list[-1]
        assert " -I -c " in seal.args[0]
        assert seal.kwargs["user"] == "root" and seal.kwargs["env"] == {"HOME": "/root"}
        assert 0 < seal.kwargs["timeout_s"] <= server.config.execution_timeout_s
        assert sandbox.download.await_args.args[0] == "/tmp/deepcad-sealed-test/prediction.step"
    assert "--trace-operations" not in str(sandbox.exec.call_args_list)
    spec = factory.call_args.args[1]
    assert set(spec.files) == {"/sandbox/model.py", "/sandbox/runner.py"}
    assert "box.step" not in str(spec.files)
    assert spec.env.get("NEMO_GYM_CONFIG_DICT", "") == ""
    assert "BASH_FUNC_exit-dramatically%%" not in spec.env
    assert all(
        not spec.env.get(key, "")
        for key in os.environ
        if key not in {"PATH", "HOME", "LANG", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ENROOT_MOUNT_HOME"}
    )


def test_identical_translated_scaled_and_wrong_orientation() -> None:
    reference = cq.Workplane("XY").box(1, 2, 3).val()
    result = score(reference, reference)
    assert result["reward"] == pytest.approx(1)
    assert result["scoring_method"] == "brep_iou" and result["iou_backend"] == "brep"
    assert result["volume_iou"] == result["reward"]
    moved = reference.scale(2.5).translate((9, -4, 3))
    assert score(moved, reference)["reward"] == pytest.approx(1)
    rotated = reference.rotate((0, 0, 0), (0, 0, 1), 90)
    assert 0.25 < score(rotated, reference)["reward"] < 0.45
    wrong = cq.Workplane("XY").sphere(1).val()
    assert 0 < score(wrong, reference)["reward"] < 0.85


def test_load_solid_rejects_shell(tmp_path: Path) -> None:
    path = tmp_path / "face.step"
    cq.exporters.export(cq.Workplane("XY").rect(1, 1).wire().val(), str(path))
    with pytest.raises(ValueError, match="solids"):
        load_solid(path)


@pytest.mark.asyncio
async def test_real_worker(server: DeepCADResourcesServer) -> None:
    path = Path(server.config.reference_root) / "box.step"
    assert (await server._score(path, path))["reward"] == pytest.approx(1)
    assert (await server._score(path, path.parent / "missing.step"))["mask_sample"]
    result = await server._score(path.parent / "missing.step", path)
    assert result["reward"] == 0 and not result.get("mask_sample", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "crash", "cancel", "bad_json"])
async def test_worker_cleanup(server: DeepCADResourcesServer, outcome: str) -> None:
    process = MagicMock()
    process.returncode = 1 if outcome == "crash" else 0
    process.communicate = AsyncMock()
    if outcome == "timeout":
        process.communicate.side_effect = [TimeoutError(), (b"", b"")]
        expected = TimeoutError
    elif outcome == "cancel":
        process.communicate.side_effect = [asyncio.CancelledError(), (b"", b"")]
        expected = asyncio.CancelledError
    else:
        process.communicate.return_value = (b"not-json", b"failed\xff")
        expected = RuntimeError if outcome == "crash" else json.JSONDecodeError
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
        with pytest.raises(expected):
            await server._score(Path("a"), Path("b"))
    if outcome in ("timeout", "cancel"):
        process.kill.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "deepcad:execution", "deepcad:infrastructure"])
@pytest.mark.parametrize("reference, expected_bonus", [(["fillet", "loft"], 0.02), (["fillet"], 0.04), ([], 0.0)])
@pytest.mark.parametrize("metric", ["geometry"])
@pytest.mark.parametrize("masked", [False, True])
async def test_operation_bonus_composition_and_failure_gates(
    server, failure, reference, expected_bonus, metric, masked
):
    from resources_servers.deepcad.app import OperationBonusConfig

    server.config.verification_metric = metric
    server.config.operation_bonus = OperationBonusConfig(enabled=True)
    scored = {
        "reward": 0.4,
        "volume_iou": 0.3,
    }
    if failure:
        scored["failure_kind"] = failure
    scored["mask_sample"] = masked or failure == "deepcad:infrastructure"
    server._evaluate = AsyncMock(return_value=scored)
    body = request("import cadquery as cq\nresult = cq.Workplane('XY').box(4,4,4).edges().fillet(.2)")
    body.verifier_metadata.reference_operations = reference
    result = await server.verify(body)
    bonus = 0 if failure or masked else expected_bonus
    assert result.source_operations == ([] if failure or masked else ["box", "fillet"])
    assert result.operation_bonus == pytest.approx(bonus)
    assert result.reward == pytest.approx(0.4 + bonus)
    assert result.reward_components["voxel_iou"] == 0.4
    assert result.reward_components["operation_bonus"] == pytest.approx(bonus)
    assert result.raw_metrics["volume_iou"] == 0.3
    assert result.raw_metrics["operation_matched_count"] == (0 if failure or masked or not reference else 1)
    assert result.raw_metrics["operation_reference_count"] == len(set(reference))
    assert result.raw_metrics["operation_recall"] == (
        0 if failure or masked or not reference else 1 / len(set(reference))
    )


@pytest.mark.asyncio
async def test_operation_bonus_ignores_forged_trace_and_uses_extracted_source(server):
    server.config.operation_bonus.enabled = True
    server._evaluate = AsyncMock(return_value={"reward": 0.4})
    code = """import cadquery as cq
from pathlib import Path
Path("/sandbox/operations.json").write_text('["loft", "fillet"]')
result = cq.Workplane().box(1, 2, 3)
"""
    body = request(f"```python\n{code}```")
    body.verifier_metadata.reference_operations = ["box", "loft", "fillet"]
    result = await server.verify(body)
    assert server._evaluate.call_args.args[0] == code.strip()
    assert result.source_operations == ["box"]
    assert result.matched_operations == ["box"]
    assert result.operation_bonus == pytest.approx(0.4 * 0.1 / 3)
    assert result.reward == pytest.approx(0.4 + result.operation_bonus)
    assert sum(result.reward_components.values()) == pytest.approx(result.reward)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [SyntaxError, RecursionError, MemoryError])
async def test_operation_source_parse_failure_keeps_geometry_reward(server, error):
    server.config.operation_bonus.enabled = True
    server._evaluate = AsyncMock(return_value={"reward": 0.4})
    body = request("result = 1")
    body.verifier_metadata.reference_operations = ["box"]
    with patch("resources_servers.deepcad.app.extract_operation_candidates", side_effect=error):
        result = await server.verify(body)
    assert result.reward == 0.4 and not result.mask_sample
    assert result.operation_bonus == 0
    assert result.source_operations == result.matched_operations == []


def test_operation_bonus_config_rejects_removed_calibration_fields():
    from resources_servers.deepcad.app import OperationBonusConfig

    assert OperationBonusConfig(enabled=True).max_fraction == 0.10
    for obsolete in ({"mean_reference_operations": 5}, {"target_fraction": 0.02}):
        with pytest.raises(ValueError):
            OperationBonusConfig(enabled=True, **obsolete)


@pytest.mark.asyncio
@pytest.mark.parametrize("preload", [False, True])
async def test_preload_cq_reaches_sandbox_runner(server, preload):
    assert server.config.preload_cq is False
    server.config.preload_cq = preload
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [SandboxExecResult("", "", 0), SandboxExecResult("", "", 1)]
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=sandbox)
    context.__aexit__ = AsyncMock(return_value=False)
    with patch("resources_servers.deepcad.app.AsyncSandbox", return_value=context):
        result = await server._evaluate("result=cq.Workplane().box(1,2,3)", "box.step")
    command = sandbox.exec.call_args_list[1].args[0]
    assert ("--preload-cq" in command) is preload
    assert result["failure_kind"] == "deepcad:execution"


def test_brep_operation_bonus_config_is_supported():
    cfg = DeepCADResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="deepcad",
        verification_metric="geometry",
        iou_backend="brep",
        operation_bonus={"enabled": True, "max_fraction": 0.1},
    )
    assert cfg.operation_bonus.enabled
    assert not cfg.preload_cq


@pytest.mark.asyncio
async def test_geometry_library_dirs_reach_worker(server, monkeypatch):
    server.config.geometry_library_dirs = ["/cad/native", "/cad/extra"]
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/cuda")
    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (b'{"volume_iou": 1.0}', b"")
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)) as spawn:
        scored = await server._score_geometry(Path("pred.step"), Path("ref.step"))
    assert scored["volume_iou"] == 1.0
    env = spawn.call_args.kwargs["env"]
    assert env["LD_LIBRARY_PATH"] == "/cad/native:/cad/extra:/existing/cuda"
    assert os.environ["LD_LIBRARY_PATH"] == "/existing/cuda"


@pytest.mark.parametrize("external_libraries", [False, True])
@pytest.mark.parametrize("metric", ["geometry", "mixed_iou"])
def test_geometry_preflight_rejects_broken_runtime(server, tmp_path, external_libraries, metric):
    server.config.geometry_library_dirs = [str(tmp_path)] if external_libraries else []
    server.config.verification_metric = metric
    process = MagicMock(returncode=1, stderr="ImportError: libGL.so.1")
    with patch("resources_servers.deepcad.app.subprocess.run", return_value=process) as probe:
        with pytest.raises(RuntimeError, match="CAD geometry runtime preflight failed:.*libGL"):
            server.setup_webserver()
    assert probe.call_args.args[0][-1] == f"import resources_servers.deepcad.{metric}"
    if external_libraries:
        assert probe.call_args.kwargs["env"]["LD_LIBRARY_PATH"].startswith(str(tmp_path))


def test_geometry_preflight_rejects_missing_library_directory(server, tmp_path):
    server.config.geometry_library_dirs = [str(tmp_path / "missing")]
    with patch("resources_servers.deepcad.app.subprocess.run") as probe:
        with pytest.raises(RuntimeError, match="CAD native library directory is missing"):
            server.setup_webserver()
    probe.assert_not_called()


def test_geometry_preflight_accepts_valid_runtime(server, tmp_path):
    server.config.geometry_library_dirs = [str(tmp_path)]
    sentinel = object()
    with (
        patch("resources_servers.deepcad.app.subprocess.run", return_value=MagicMock(returncode=0)),
        patch("resources_servers.deepcad.app.SimpleResourcesServer.setup_webserver", return_value=sentinel),
    ):
        assert server.setup_webserver() is sentinel


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [True, False])
async def test_verifier_retries_same_source_in_fresh_sandbox(server, recover):
    server.config.inconclusive_verification_retries = 1
    sandboxes = [AsyncMock(), AsyncMock()]
    sandboxes[0].exec.return_value = SandboxExecResult("", "", 0, "runtime")
    sandboxes[1].exec.side_effect = (
        [
            SandboxExecResult("", "", 0),
            SandboxExecResult("", "", 0),
            SandboxExecResult("100", "", 0),
            SandboxExecResult('{"path":"/tmp/deepcad-sealed-test/prediction.step"}', "", 0),
        ]
        if recover
        else [SandboxExecResult("", "", 0, "runtime")]
    )
    contexts = []
    for sandbox in sandboxes:
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=sandbox)
        context.__aexit__ = AsyncMock(return_value=False)
        contexts.append(context)
    with (
        patch("resources_servers.deepcad.app.AsyncSandbox", side_effect=contexts) as factory,
        patch("resources_servers.deepcad.app.asyncio.sleep", new_callable=AsyncMock),
        patch.object(DeepCADResourcesServer, "_score", AsyncMock(return_value={"reward": 0.82})),
    ):
        result = await server.verify(request("result=1"))
    assert factory.call_count == 2
    assert [call.args[1].files["/sandbox/model.py"] for call in factory.call_args_list] == ["result=1"] * 2
    assert all(call.args[1].ttl_s == server.config.execution_timeout_s + 300 for call in factory.call_args_list)
    for context in contexts:
        context.__aexit__.assert_awaited_once()
    assert result.evaluation_completed is recover
    assert result.mask_sample is (not recover)
    assert result.reward == (0.82 if recover else 0)
    assert result.failure_kind == (None if recover else PROVIDER_UNAVAILABLE)


@pytest.mark.asyncio
async def test_scoring_exception_exhausts_retry_with_verifier_kind(server):
    server.config.inconclusive_verification_retries = 0
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [
        SandboxExecResult("", "", 0),
        SandboxExecResult("", "", 0),
        SandboxExecResult("100", "", 0),
        SandboxExecResult('{"path":"/tmp/deepcad-sealed-test/prediction.step"}', "", 0),
    ]
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=sandbox)
    context.__aexit__ = AsyncMock(return_value=False)
    with (
        patch("resources_servers.deepcad.app.AsyncSandbox", return_value=context),
        patch.object(DeepCADResourcesServer, "_score", AsyncMock(side_effect=TimeoutError("scorer timeout"))),
    ):
        result = await server.verify(request("result=1"))
    assert result.mask_sample and not result.evaluation_completed
    assert result.failure_kind == VERIFIER_ERROR


@pytest.mark.asyncio
async def test_model_execution_failure_does_not_retry(server):
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [SandboxExecResult("", "", 0), SandboxExecResult("", "", 124, "timeout")]
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=sandbox)
    context.__aexit__ = AsyncMock(return_value=False)
    with patch("resources_servers.deepcad.app.AsyncSandbox", return_value=context) as factory:
        result = await server.verify(request("result=1"))
    factory.assert_called_once()
    assert result.reward == 0 and not result.mask_sample and result.evaluation_completed
    assert result.failure_kind == "deepcad:execution"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sealed,kind,mask",
    [
        (SandboxExecResult("", "", 1), "deepcad:geometry", False),
        (SandboxExecResult("", "", 124, "timeout"), "deepcad:geometry", False),
        (SandboxExecResult("", "", 125, "runtime"), PROVIDER_UNAVAILABLE, True),
        (SandboxExecResult('{"path":"/etc/passwd"}', "", 0), PROVIDER_UNAVAILABLE, True),
    ],
)
async def test_sealing_failures_preserve_model_and_infrastructure_distinction(server, sealed, kind, mask):
    server.config.inconclusive_verification_retries = 0
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [
        SandboxExecResult("", "", 0),
        SandboxExecResult("", "", 0),
        SandboxExecResult("100", "", 0),
        sealed,
    ]
    context = MagicMock(__aenter__=AsyncMock(return_value=sandbox), __aexit__=AsyncMock(return_value=False))
    with patch("resources_servers.deepcad.app.AsyncSandbox", return_value=context):
        result = await server.verify(request("result=1"))
    assert result.reward == 0 and result.mask_sample is mask
    assert result.failure_kind == kind and result.evaluation_completed is (not mask)
    sandbox.download.assert_not_awaited()


@pytest.mark.asyncio
async def test_fusion_cannot_extend_the_model_execution_budget(server):
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [
        SandboxExecResult("", "", 0),
        SandboxExecResult("", "", 0),
        SandboxExecResult("100", "", 0),
    ]
    context = MagicMock(__aenter__=AsyncMock(return_value=sandbox), __aexit__=AsyncMock(return_value=False))
    with (
        patch("resources_servers.deepcad.app.AsyncSandbox", return_value=context),
        patch("resources_servers.deepcad.app.time") as clock,
    ):
        clock.monotonic.side_effect = [0, server.config.execution_timeout_s + 1]
        result = await server.verify(request("result=1"))
    assert result.reward == 0 and not result.mask_sample
    assert result.failure_kind == "deepcad:execution"
    assert sandbox.exec.await_count == 3
    sandbox.download.assert_not_awaited()
