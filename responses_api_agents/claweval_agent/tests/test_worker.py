# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the process boundary without requiring the private native checkout in CI."""

import hashlib
import io
import json
import signal
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field

from responses_api_agents.claweval_agent import worker


class ModelConfig(BaseModel):
    model_id: str = "native-model"
    api_key: str = ""
    extra_body: dict = Field(default_factory=dict)


class NativeConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    judge: ModelConfig = Field(default_factory=ModelConfig)
    user_agent_model: ModelConfig = Field(default_factory=ModelConfig)


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    for name in ("claw_eval", "claw_eval.models", "evaluation"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    for path in ("src/claw_eval/config.py", "evaluation/run_multimodal.py", "evaluation/task_catalog.py"):
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# native runtime fixture\n")
    paths = []
    tasks = {}
    for task_id in ("T001_test", "T002_test"):
        path = tmp_path / "tasks" / task_id / "task.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(f"task_id: {task_id}\nprompt:\n  text: Solve {task_id}\n")
        paths.append(path)
        tasks[path] = SimpleNamespace(
            task_id=task_id, prompt=SimpleNamespace(text=f"Solve {task_id}"), user_agent=SimpleNamespace(enabled=False)
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: {}\n")
    provider = SimpleNamespace(client=Mock())
    native = ModuleType("evaluation.run_multimodal")
    native.build_judge = Mock(return_value=object())
    native.build_provider = Mock(return_value=provider)
    native.run_one = Mock(return_value={"task_id": "T001_test", "task_score": 0.9, "status": "completed"})
    native.validate_assets = Mock(return_value=[])
    native.task_contract_fingerprint = Mock(return_value={"task": "fingerprint"})
    modules = {
        "evaluation.run_multimodal": native,
        "evaluation.task_catalog": SimpleNamespace(discover_tasks=Mock(return_value=paths)),
        "claw_eval.models.task": SimpleNamespace(
            TaskDefinition=SimpleNamespace(from_yaml=Mock(side_effect=tasks.get))
        ),
        "claw_eval.config": SimpleNamespace(
            Config=NativeConfig,
            load_config=Mock(return_value=NativeConfig(model=ModelConfig(extra_body={"max_tokens": 123}))),
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    output = tmp_path / "out"
    output.mkdir()
    payload = {
        "operation": "run",
        "claweval_root": str(tmp_path),
        "claweval_config": "config.yaml",
        "verifier_metadata": {
            "task_id": "T001_test",
            "task_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        },
        "prompt": "Solve T001_test",
        "seed": 1002,
        "no_judge": False,
        "output_dir": str(output),
        "sandbox_image": "native.sqsh",
        "sandbox_dependencies": str(tmp_path / "dependencies"),
        "sandbox_port": 18080,
        "sandbox_ready_timeout": 60,
        "model_overrides": {"model_id": "selected-policy", "extra_body": {"top_p": 0.9}},
    }
    return SimpleNamespace(root=tmp_path, paths=paths, tasks=tasks, native=native, payload=payload, provider=provider)


def test_trial_preserves_native_settings_and_closes_provider_on_failure(boundary):
    b = boundary
    result = worker.run_trial(b.payload, b.root)
    call = b.native.run_one.call_args.kwargs
    assert call["task_yaml"] == b.paths[0]
    assert call["max_turns"] == -1
    assert call["fixture_root"] == b.root / "tasks"
    assert call["cfg"].model.model_id == "selected-policy"
    assert call["cfg"].model.extra_body == {"max_tokens": 123, "top_p": 0.9, "seed": 1002}
    assert call["sandbox_image"] == "native.sqsh"
    assert result["task_score"] == 0.9
    assert result["seed"] == 1002
    assert result["judge_enabled"] is True
    assert result["task_contract"] == {"task": "fingerprint"}
    before = result["provenance"]["runtime_sha256"]
    (b.root / "src/claw_eval/config.py").write_text("# changed runtime\n")
    assert worker.runtime_provenance(b.root, b.root / "config.yaml")["runtime_sha256"] != before
    b.provider.client.close.assert_called_once()
    b.native.run_one.side_effect = RuntimeError("sandbox failed")
    with pytest.raises(RuntimeError, match="sandbox failed"):
        worker.run_trial(b.payload, b.root)
    assert b.provider.client.close.call_count == 2


@pytest.mark.parametrize("problem", ["prompt", "config", "assets", "user_agent"])
def test_invalid_trial_never_constructs_a_model(boundary, problem):
    b = boundary
    if problem == "prompt":
        b.payload["prompt"] = "changed"
    elif problem == "config":
        b.payload["claweval_config"] = "absent.yaml"
    elif problem == "assets":
        b.native.validate_assets.return_value = ["missing fixture"]
    else:
        b.tasks[b.paths[0]].user_agent.enabled = True
    with pytest.raises((ValueError, FileNotFoundError)):
        worker.run_trial(b.payload, b.root)
    b.native.build_provider.assert_not_called()
    b.native.build_judge.assert_not_called()


def test_export_is_public_selected_and_atomic(boundary):
    b = boundary
    output = b.root / "selected.jsonl"
    payload = {"output": str(output), "split": "general", "agent_name": "native", "task_ids": ["T002_test"]}
    worker.prepare(payload, b.root)
    original = output.read_bytes()
    row = json.loads(original)
    assert row == {
        "agent_ref": {"type": "responses_api_agents", "name": "native"},
        "responses_create_params": {"input": [{"role": "user", "content": "Solve T002_test"}]},
        "verifier_metadata": {
            "task_id": "T002_test",
            "task_sha256": hashlib.sha256(b.paths[1].read_bytes()).hexdigest(),
            "split": "general",
        },
    }
    with pytest.raises(ValueError, match="Unknown task IDs"):
        worker.prepare({**payload, "task_ids": ["missing"]}, b.root)
    assert output.read_bytes() == original
    definition = sys.modules["claw_eval.models.task"].TaskDefinition
    definition.from_yaml.side_effect = ValueError("invalid task")
    with pytest.raises(ValueError, match="invalid task"):
        worker.prepare(payload, b.root)
    assert output.read_bytes() == original
    assert not list(b.root.glob("selected.*.tmp"))


def test_worker_protocol_writes_results_and_handles_shutdown(boundary, monkeypatch):
    b = boundary
    monkeypatch.chdir(b.root)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    register = Mock()
    monkeypatch.setattr(signal, "signal", register)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(b.payload)))
    worker.main()
    result = json.loads((b.root / "out/result.json").read_text())
    assert result["task_score"] == 0.9
    assert result["seed"] == 1002
    with pytest.raises(KeyboardInterrupt, match="terminated"):
        register.call_args.args[1](signal.SIGTERM, None)
    prepare = {
        **b.payload,
        "operation": "prepare",
        "split": "general",
        "agent_name": "native",
        "output": str(b.root / "all.jsonl"),
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(prepare)))
    worker.main()
    assert len((b.root / "all.jsonl").read_text().splitlines()) == 2
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({**b.payload, "operation": "unknown"})))
    with pytest.raises(ValueError, match="Unknown operation"):
        worker.main()
