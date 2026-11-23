# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.sandbox.providers.base import SandboxExecResult
from resources_servers.deepcad.dataset import corpus_inputs


@pytest.fixture
def reference(tmp_path):
    code = 'import cadquery as cq\nresult=cq.Workplane("XY").box(1,1,1)'
    sha = hashlib.sha256(code.encode()).hexdigest()
    with sqlite3.connect(tmp_path / "inventory.sqlite") as db:
        db.execute("CREATE TABLE assets (id,code,code_sha256,dataset)")
        db.execute("INSERT INTO assets VALUES (?,?,?,?)", ("a", code, sha, "zero-to-cad"))
    record = {
        "inventory_database": "inventory.sqlite",
        "asset_id": "a",
        "source_sha256": sha,
        "reference_step": "references/a.step",
    }
    config = {
        "reference_sandbox_image": "image",
        "reference_sandbox_provider": {"docker": {}},
        "reference_execution_timeout_s": 1,
    }
    return tmp_path, record, config


def test_corpus_integrity_and_explicit_geomcad_source_adapter(reference):
    root, record, _ = reference
    with sqlite3.connect(root / "inventory.sqlite") as db:
        db.execute("UPDATE assets SET dataset='geomcad'")
    program, dataset = corpus_inputs.load_program(root, record)
    assert dataset == "geomcad" and program.endswith("\nresult = r\n")
    for changes in ({"source_sha256": "wrong"}, {"inventory_database": "../outside.sqlite"}, {"asset_id": "missing"}):
        with pytest.raises(ValueError):
            corpus_inputs.load_program(root, record | changes)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["execution", "artifact"])
async def test_reference_provider_failures_do_not_trigger_reference_replacement(reference, phase):
    root, record, config = reference
    sandbox = AsyncMock()
    sandbox.exec.side_effect = ([SandboxExecResult("", "", 0)] if phase == "artifact" else []) + [
        SandboxExecResult("100", "", 0, "runtime")
    ]
    context = MagicMock(__aenter__=AsyncMock(return_value=sandbox), __aexit__=AsyncMock(return_value=False))
    with patch.object(corpus_inputs, "AsyncSandbox", return_value=context):
        with pytest.raises(RuntimeError, match="provider failure|artifact inspection failed"):
            await corpus_inputs.build_reference(root, record, config, root)


@pytest.mark.asyncio
async def test_reference_source_operation_evidence_and_cache(reference):
    root, record, config = reference
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [SandboxExecResult("", "", 0), SandboxExecResult("10", "", 0)]

    async def download(source, destination):
        destination.write_text("valid STEP")

    sandbox.download.side_effect = download
    context = MagicMock(__aenter__=AsyncMock(return_value=sandbox), __aexit__=AsyncMock(return_value=False))
    with patch.object(corpus_inputs, "AsyncSandbox", return_value=context) as factory:
        target, operations = await corpus_inputs.build_reference(root, record, config, root)
        assert operations == ["box"] and target.read_text() == "valid STEP"
        assert factory.call_args.args[1].ttl_s == 301
        assert set(factory.call_args.args[1].files) == {"/sandbox/model.py", "/sandbox/runner.py"}
        assert await corpus_inputs.build_reference(root, record, config, root) == (target, ["box"])
        factory.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [-9, 137, -11, 139])
async def test_reference_process_kill_requires_confirmation(reference, exit_code):
    root, record, config = reference
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [SandboxExecResult("", "", exit_code), SandboxExecResult("Killed", "", 0)]
    context = MagicMock(__aenter__=AsyncMock(return_value=sandbox), __aexit__=AsyncMock(return_value=False))
    with patch.object(corpus_inputs, "AsyncSandbox", return_value=context):
        with pytest.raises(corpus_inputs.ReferenceRejected, match="cause unverified") as error:
            await corpus_inputs.build_reference(root, record, config, root)
    assert error.value.confirmation_key == f"reference_signal:{exit_code}"
    assert not (root / record["reference_step"]).exists()


@pytest.mark.asyncio
async def test_provider_failure_with_kill_exit_is_still_an_outage(reference):
    root, record, config = reference
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult("", "", 137, "runtime")
    context = MagicMock(__aenter__=AsyncMock(return_value=sandbox), __aexit__=AsyncMock(return_value=False))
    with patch.object(corpus_inputs, "AsyncSandbox", return_value=context):
        with pytest.raises(RuntimeError, match="provider failure"):
            await corpus_inputs.build_reference(root, record, config, root)


def test_renderer_preserves_process_failure_confirmation_key(tmp_path):
    import json

    from resources_servers.deepcad.dataset import lazy_inputs

    request, output = tmp_path / "request.json", tmp_path / "row.json"
    config = lazy_inputs.LazyRenderConfig(
        dataset_root=str(tmp_path), renderer_python="python", cache_root=str(tmp_path), reference_sandbox_image="image"
    )
    request.write_text(json.dumps({"config": config.model_dump(), "record": {"source_kind": "cad_corpus"}}))
    failure = corpus_inputs.ReferenceRejected("process killed", confirmation_key="reference_signal:137")
    with patch.object(corpus_inputs, "render_corpus", side_effect=failure):
        lazy_inputs.render(request, output)
    assert json.loads(output.with_suffix(".rejected.json").read_text()) == {
        "reason": "process killed",
        "confirmation_key": "reference_signal:137",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "font_error",
    [
        "Fontconfig error: Cannot load default config file",
        "Font_FontMgr, error: unable to find any font!",
    ],
)
async def test_missing_font_infrastructure_does_not_reject_reference(reference, font_error):
    root, record, config = reference
    sandbox = AsyncMock()
    trace = font_error + '\nTraceback:\n  File "model.py", line 73\nAttributeError: FontName'
    sandbox.exec.side_effect = [SandboxExecResult("", "", 1), SandboxExecResult(trace, "", 0)]
    context = MagicMock(__aenter__=AsyncMock(return_value=sandbox), __aexit__=AsyncMock(return_value=False))
    with patch.object(corpus_inputs, "AsyncSandbox", return_value=context):
        with pytest.raises(RuntimeError, match="font infrastructure unavailable"):
            await corpus_inputs.build_reference(root, record, config, root)
    assert not (root / record["reference_step"]).exists()


@pytest.mark.asyncio
async def test_model_error_with_fonts_available_still_rejects_reference(reference):
    root, record, config = reference
    sandbox = AsyncMock()
    trace = (
        'Traceback:\n  File "model.py", line 73\nTypeError: Workplane.text() got an unexpected keyword argument cut'
    )
    sandbox.exec.side_effect = [SandboxExecResult("", "", 1), SandboxExecResult(trace, "", 0)]
    context = MagicMock(__aenter__=AsyncMock(return_value=sandbox), __aexit__=AsyncMock(return_value=False))
    with patch.object(corpus_inputs, "AsyncSandbox", return_value=context):
        with pytest.raises(corpus_inputs.ReferenceRejected, match="unexpected keyword argument"):
            await corpus_inputs.build_reference(root, record, config, root)
