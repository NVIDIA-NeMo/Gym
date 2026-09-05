# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


PACKAGE = Path(__file__).parents[1] / "hsg" / "checkpoint_e2e"


def _load_helper() -> ModuleType:
    path = PACKAGE / "judge_session.py"
    spec = importlib.util.spec_from_file_location("checkpoint_e2e_judge_session", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_service_gate_requires_the_exact_expected_live_set(monkeypatch) -> None:
    helper = _load_helper()
    expected = ["policy_model", "gdpval_resources_server"]

    monkeypatch.setattr(
        helper,
        "_url_json",
        lambda _url: [
            {"config_path": "policy_model", "url": "http://127.0.0.1:12001"},
            {"config_path": "gdpval_resources_server", "url": "http://127.0.0.1:12002"},
        ],
    )

    class Response:
        def close(self) -> None:
            pass

    monkeypatch.setattr(helper.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    assert helper.services_ready("http://127.0.0.1:12000", expected) == (True, "all 2 services ready")

    ready, detail = helper.services_ready("http://127.0.0.1:12000", [*expected, "gdpval_stirrup_agent"])
    assert not ready
    assert "gdpval_stirrup_agent" in detail


def test_persistent_command_contract_is_opt_in_bounded_strict_and_reaped() -> None:
    judge = (PACKAGE / "judge.sbatch").read_text(encoding="utf-8")
    controller = (PACKAGE / "existing_judge_controller.sbatch").read_text(encoding="utf-8")

    assert "PERSISTENT_JUDGE_SESSION=${CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION:-false}" in judge
    legacy = judge.split("if [[ $PERSISTENT_JUDGE_SESSION == false ]]; then", 1)[1].split("\nfi", 1)[0]
    persistent = judge.split("# Opt-in persistent path:", 1)[1]
    assert '"$GYM_PYTHON" "$GYM_ENTRYPOINT" eval run' in legacy
    assert '"$GYM_PYTHON" "$GYM_ENTRYPOINT" env start' in persistent
    assert '"$SESSION_PY" run-pass' in persistent
    assert "CONCURRENCY_LADDER=${CONCURRENCY_LADDER:-16,8,4,1}" in judge
    assert "[[ $CONCURRENCY_LADDER == 16,8,4,1 ]]" in judge
    assert "strict_result && exit 0" in persistent
    assert persistent.index("strict_result && exit 0") < persistent.index("exit 76")

    # EXIT runs for both strict success and ladder exhaustion, and the process-
    # group helper both signals and reaps the env/service owner.
    assert "trap cleanup_session EXIT" in judge
    assert "stop_judge_process_group || true" in judge
    lifecycle = (PACKAGE / "judge_process_group.sh").read_text(encoding="utf-8")
    assert 'kill -TERM -- "-$GYM_PID"' in lifecycle
    assert 'wait "$GYM_PID"' in lifecycle

    assert 'CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION="$PERSISTENT_JUDGE_SESSION"' in controller
    assert '$(job_exit_code "$job" || true) == 76' in controller
    assert "strict result failed after the persistent 16->8->4->1 ladder" in controller
