# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise complete worker admission and cleanup with independent native-module doubles."""

import json
import signal
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from responses_api_agents.wildclawbench_agent import data, worker
from responses_api_agents.wildclawbench_agent.tests.test_app import fixture_source  # noqa: F401


@pytest.fixture
def episode(fixture_source, monkeypatch):
    root = fixture_source
    output = root / "episode"
    output.mkdir()
    state = dict(
        messages=[dict(role="assistant", content=[dict(type="text", text="answer")])],
        score={"overall_score": 0.25},
        exit=0,
        elapsed=1,
        log="",
        error=None,
        calls=[],
        env="",
    )

    class Backend:
        transcript_container_path = "/transcript"

        def __init__(self, **kwargs):
            state["calls"].append("init")

        def run_task(self, spec):
            state["calls"].append("run")
            state["spec"] = spec
            (output / "chat.jsonl").write_text("\n".join(json.dumps(dict(message=m)) for m in state["messages"]))
            (output / "agent.log").write_text(state["log"])
            return SimpleNamespace(
                agent_proc=SimpleNamespace(returncode=state["exit"]),
                gateway_proc=None,
                error=state["error"],
                elapsed_time=state["elapsed"],
            )

        def collect_usage(self, *args):
            return {"input_tokens": 1}

    def grade(**kwargs):
        state["calls"].append("grade")
        if state.get("raise_grade"):
            raise RuntimeError("grading transport failed")
        return state["score"]

    exports = {
        "src.agents.base": dict(AgentTaskSpec=SimpleNamespace),
        "src.agents.openclaw": dict(OpenClawAgent=Backend),
        "src.utils.docker_utils": dict(
            collect_output_from_container=lambda *args: state["calls"].append("collect"),
            remove_container=lambda *args: state["calls"].append("remove"),
            stop_background_process=lambda *args: state["calls"].append("stop"),
        ),
        "src.utils.grading": dict(run_grading=grade),
        "src.utils.openclaw_runtime": dict(available_gateway_port=lambda: 12345, context_defaults=lambda *args: {}),
        "src.utils.task_parser": dict(
            parse_task_md=lambda path: dict(
                timeout_seconds=300,
                workspace_path=str(root / "workspace/category/task"),
                automated_checks="original rubric",
                env=state["env"],
            )
        ),
        "src.utils.trace_artifacts": dict(write_trace_artifact=lambda *args: None),
    }
    for name, attributes in exports.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("BRAVE_API_KEY", "test-search")
    monkeypatch.setattr(worker.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    row = data.prepare_rows(root, ["task"])[0]
    config = dict(
        source_root=str(root),
        output=str(output),
        metadata=row["verifier_metadata"],
        input_prompt=row["responses_create_params"]["input"][0]["content"],
        model_name="policy",
        policy_base_url="http://proxy/v1",
        context_window=131072,
        max_tokens=8192,
        thinking="high",
        evaluation_protocol={"judge_model": "judge", "sandbox_image": "sha256:abc"},
    )
    # execute inserts this path itself. Restore sys.path after each independent test.
    monkeypatch.syspath_prepend(str(root))
    return state, config


@pytest.mark.parametrize("reward", [0, 0.25, 1])
def test_execute_preserves_original_rewards_and_cleans_up(episode, reward):
    state, config = episode
    state["score"] = {"overall_score": reward}
    result = worker.execute(config)
    assert result["reward"] == result["native_reward"] == reward
    assert not result["mask_sample"]
    assert "failure_kind" not in result
    assert state["calls"][-2:] == ["stop", "remove"]
    assert state["calls"].index("stop") < state["calls"].index("grade")
    assert state["spec"].prompt.endswith(config["input_prompt"])
    assert "SECRET" not in state["spec"].prompt
    provider = state["spec"].models_config["providers"]["gym-policy"]
    assert provider["baseUrl"] == config["policy_base_url"]
    assert provider["models"][0]["reasoning"] is True
    assert provider["models"][0]["input"] == ["text", "image"]


@pytest.mark.parametrize(
    "message,valid",
    [
        ("400: context length exceeded", True),
        ("HTTP 400 bad request", True),
        ("413: request too large", True),
        ("HTTP 413 payload", True),
        ("422: invalid request", True),
        ("HTTP 422 invalid", True),
        ("400", True),
        ("429: quota", False),
        ("HTTP 503 unavailable", False),
        ("500: backend", False),
        ("connection refused", False),
        ("4000: other", False),
        ("unclassified error", False),
    ],
)
def test_execute_classifies_terminal_policy_errors(episode, message, valid):
    state, config = episode
    state.update(messages=[dict(role="assistant", stopReason="error", errorMessage=message)], exit=1)
    result = worker.execute(config)
    assert result["mask_sample"] is not valid
    assert result["reward"] == (0.25 if valid else 0)
    assert ("failure_kind" in result) is not valid
    assert state["calls"][-1] == "remove"


@pytest.mark.parametrize(
    "method,masked",
    [
        ("failed", True),
        ("rule_fallback", True),
        ("regex_fallback", True),
        ("keyword_fallback", True),
        ("llm", False),
        ("vlm", False),
        ("not_run_no_assistant_text", False),
    ],
)
def test_execute_masks_only_unreliable_judge_methods(episode, method, masked):
    state, config = episode
    state["score"]["rubric"] = {"answer_judge_method": method}
    result = worker.execute(config)
    assert result["mask_sample"] is masked
    assert result["native_reward"] == 0.25
    assert worker.grader_failures({"answer_method": method}) == []


@pytest.mark.parametrize("reward", [True, None, "0.5", -0.1, 1.1, float("nan"), float("inf")])
def test_execute_rejects_invalid_grader_reward(episode, reward):
    state, config = episode
    state["score"]["overall_score"] = reward
    result = worker.execute(config)
    assert result["mask_sample"] and result["native_reward"] is None
    assert result["failure_kind"].startswith("wildclawbench:")


@pytest.mark.parametrize("mode", ["missing", "gateway", "replay", "exit", "execution", "judge"])
def test_execute_masks_infrastructure_and_transcript_failures(episode, mode):
    state, config = episode
    if mode == "missing":
        state["messages"] = [dict(role="user", content="task")]
    elif mode == "gateway":
        state["log"] = "gateway failed, falling back to embedded"
    elif mode == "replay":
        state["messages"][0]["content"] = [dict(type="thinking", thinking="reason", thinkingSignature="opaque")]
    elif mode == "exit":
        state["exit"] = 1
    elif mode == "execution":
        state["error"] = "container startup failed"
    else:
        state["score"]["nested"] = [{"status": "llm_judge_failed"}]
    result = worker.execute(config)
    assert result["mask_sample"] and result["reward"] == 0
    assert result["failure_kind"].startswith("wildclawbench:")
    assert state["calls"][-1] == "remove"


@pytest.mark.parametrize("returncode", [-signal.SIGTERM, -signal.SIGKILL, 137, 143, 1])
def test_native_deadline_is_scored(episode, returncode):
    state, config = episode
    state.update(exit=returncode, elapsed=300)
    result = worker.execute(config)
    assert result["task_deadline_reached"] and not result["mask_sample"]


def test_cleanup_runs_when_grader_raises(episode):
    state, config = episode
    state["raise_grade"] = True
    with pytest.raises(RuntimeError, match="grading transport"):
        worker.execute(config)
    assert state["calls"][-1] == "remove"


def test_missing_credentials_prevent_policy_execution(episode, monkeypatch):
    state, config = episode
    state["env"] = "# task judge\nJUDGE_KEY\n"
    monkeypatch.delenv("JUDGE_KEY", raising=False)
    with pytest.raises(ValueError, match="JUDGE_KEY"):
        worker.execute(config)
    assert state["calls"] == []


def test_changed_prompt_prevents_native_execution(episode):
    state, config = episode
    config["input_prompt"] += " changed"
    with pytest.raises(ValueError, match="prompt changed"):
        worker.execute(config)
    assert state["calls"] == []


@pytest.mark.parametrize("elapsed,error", [(299, None), (300, "container startup failed")])
def test_nonzero_exit_without_native_deadline_remains_masked(episode, elapsed, error):
    state, config = episode
    state.update(exit=1, elapsed=elapsed, error=error)
    result = worker.execute(config)
    assert result["mask_sample"]
    assert not result["task_deadline_reached"]


def test_agent_stage_leaves_grading_and_container_cleanup_to_resources(episode):
    state, config = episode
    result = worker.execute(config, stage="agent")
    assert result["execution_started"]
    assert "grade" not in state["calls"]
    assert "remove" not in state["calls"]
    assert result["events"]
    assert json.loads((Path(config["output"]) / "harness_stopped.json").read_text())["cleanup_confirmed"]


def test_verifier_requires_confirmed_agent_close_and_preserves_native_score(episode):
    state, config = episode
    output = Path(config["output"])
    result = worker.execute(config, stage="agent")
    (output / "agent_result.json").write_text(json.dumps(result))
    (output / "lease.json").write_text(json.dumps({"episode_id": {"rollout_id": "3-2", "attempt": 0}}))
    with pytest.raises(FileNotFoundError):
        worker.verify_execution(config)
    assert "grade" not in state["calls"]
    (output / "agent_closed.json").write_text(
        json.dumps({"cleanup_confirmed": True, "episode_id": {"rollout_id": "other", "attempt": 0}})
    )
    with pytest.raises(ValueError, match="confirmed agent close"):
        worker.verify_execution(config)
    (output / "agent_closed.json").write_text(
        json.dumps({"cleanup_confirmed": True, "episode_id": {"rollout_id": "3-2", "attempt": 0}})
    )
    result = worker.verify_execution(config)
    assert result["reward"] == result["native_reward"] == 0.25
    assert not result["mask_sample"]
    assert state["calls"].count("grade") == 1
    assert "remove" not in state["calls"]
