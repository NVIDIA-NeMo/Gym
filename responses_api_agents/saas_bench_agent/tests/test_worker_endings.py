# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process endings must reach live-state grading before application cleanup."""

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from responses_api_agents.saas_bench_agent.runtime import BENCHMARK_ROOT


@pytest.mark.parametrize(
    "ending", ["context_limit", "output_token_limit", "timeout", "model_protocol_error", "setup_failure"]
)
def test_opencode_model_endings_grade_live_state(tmp_path, monkeypatch, ending):
    monkeypatch.syspath_prepend(str(BENCHMARK_ROOT))
    from saas_bench.opencode import worker

    root = tmp_path / "bundle"
    (root / "saas_bench").mkdir(parents=True)
    (root / "saas_bench/apps.yaml").write_text("apps:\n  fixture: {}\n")
    task = {
        "task_id": "fixture",
        "source_sha256": "sha",
        "input_paths": [],
        "meta": {"meta_data": {"sites": ["fixture"]}},
    }
    monkeypatch.setattr(worker, "inventory", lambda *args, **kwargs: {"fixture": task})
    monkeypatch.setattr(worker, "opencode_config", lambda **kwargs: {})
    loader = types.ModuleType("saas_bench.loader")
    loader.build_prompt = lambda *args, **kwargs: ("Fixture", "todo", [])
    slots = types.ModuleType("saas_bench.slot")
    slot = MagicMock()
    slot.get_port_map.return_value = {"fixture": 8000}
    if ending == "setup_failure":
        slot.start_apps.side_effect = RuntimeError("backend unavailable")
    slots.SlotManager = lambda *args: slot
    verifier = types.ModuleType("saas_bench.verify_runner")

    def grade(*args, **kwargs):
        slot.stop_apps.assert_not_called()
        return {"status": "FAIL", "score": 0.5, "checks": []}

    verifier.run_verify = MagicMock(side_effect=grade)
    for name, module in [("loader", loader), ("slot", slots), ("verify_runner", verifier)]:
        monkeypatch.setitem(sys.modules, "saas_bench." + name, module)
    event = {"type": "error", "error": {"name": "ContextOverflowError"}}
    if ending == "output_token_limit":
        event = {"type": "step_finish", "part": {"reason": "length"}}
    if ending == "model_protocol_error":
        event = {"type": "error", "error": {"message": "Model violated the native tool protocol"}}
    executable = tmp_path / "opencode"
    executable.write_text(
        f"#!{sys.executable}\nimport sys, time\n"
        "if '--version' in sys.argv:\n print('1.17.11');sys.exit(0)\n"
        + ("time.sleep(30)\n" if ending == "timeout" else f"print({json.dumps(event)!r}, flush=True)\nsys.exit(1)\n")
    )
    executable.chmod(0o700)
    monkeypatch.delenv("SAAS_NRT_ENROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    result = worker.execute(
        {
            "source_root": str(root),
            "output": str(tmp_path / "episode"),
            "task_id": "fixture",
            "source_sha256": "sha",
            "opencode_version": "1.17.11",
            "command": str(executable),
            "endpoint": "http://localhost:8000/v1",
            "model": "policy",
            "slot_capacity": 1,
            "slot_lock_root": str(tmp_path / "locks"),
            "max_steps": 5,
            "context_tokens": 1000,
            "output_tokens": 100,
            "agent_timeout": 0.1 if ending == "timeout" else 10,
        }
    )
    slot.stop_apps.assert_called_once_with(["fixture"])
    if ending == "setup_failure":
        assert result["mask_sample"] is True and result["failure_kind"] == "saas_bench:infrastructure"
        verifier.run_verify.assert_not_called()
    else:
        verifier.run_verify.assert_called_once()
        assert result["status"] == "completed" and result["mask_sample"] is False, result
        assert result["verify"]["score"] == 0.5  # Measured state is retained separately from policy zero.
        assert result["reward"] == 0
        assert result["execution_health"]["termination"] == ending
