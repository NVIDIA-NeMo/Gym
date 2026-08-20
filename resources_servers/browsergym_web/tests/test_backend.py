# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from nemo_gym.web.models import WebAction, WebBenchmark, WebTask
from resources_servers.browsergym_web import backend as backend_module
from resources_servers.browsergym_web.artifacts import WebArtifactStore
from resources_servers.browsergym_web.backend import (
    BrowserGymBackend,
    EvaluatorConfigurationError,
    EvaluatorInfrastructureError,
)
from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig


class FakeEnv:
    def __init__(self):
        self.actions = []
        self.closed = False

    @staticmethod
    def _observation(last_action="", last_action_error=""):
        return {
            # BrowserGym 0.14.3 emits goal_object as a tuple, including the
            # image_url blocks used by VisualWebArena.
            "goal_object": (
                {"type": "text", "text": "Do the task"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ),
            "open_pages_urls": ["https://example.test"],
            "open_pages_titles": ["Example"],
            "active_page_index": [0],
            "url": "https://example.test",
            "screenshot": b"not-a-real-png",
            "axtree_object": {},
            "extra_element_properties": {},
            "last_action": last_action,
            "last_action_error": last_action_error,
            "elapsed_time": [1.25],
        }

    def reset(self, seed):
        assert seed == 7
        return self._observation(), {"task_info": {}}

    def step(self, action):
        self.actions.append(action)
        terminal = action.startswith("send_msg_to_user")
        return self._observation(last_action=action), float(terminal), terminal, False, {}

    def close(self):
        self.closed = True


def _config(tmp_path):
    return BrowserGymWebResourcesServerConfig(
        name="browsergym_web",
        host="localhost",
        port=8000,
        entrypoint="app.py",
        domain="agent",
        artifact_dir=str(tmp_path),
    )


def test_backend_keeps_execution_and_benchmark_scores_separate(tmp_path, monkeypatch):
    env = FakeEnv()
    backend = BrowserGymBackend(
        _config(tmp_path),
        "session-a",
        WebArtifactStore(tmp_path),
    )
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: env)
    monkeypatch.setattr(backend, "_prepare_evaluator", lambda _task: None)
    task = WebTask(benchmark=WebBenchmark.WEBARENA, task_id="0", seed=7)

    observation, info = backend.reset(task)
    assert observation.screenshot is not None
    assert observation.active_tab_index == 0
    assert observation.elapsed_time == 1.25
    assert [block["type"] for block in observation.goal] == ["text", "image_url"]
    assert info["env_id"] == "browsergym/webarena.0"

    step = backend.step(WebAction(name="noop", script="noop()"))
    assert step.execution_ok is True
    assert step.benchmark_reward == 0.0

    evaluation = backend.evaluate("final answer")
    assert evaluation.valid_sample is True
    assert evaluation.raw_score == 1.0
    assert env.actions[-1] == "send_msg_to_user('final answer')"

    backend.close()
    assert env.closed is True


def test_invalid_high_level_action_is_returned_to_agent(tmp_path, monkeypatch):
    class RejectingEnv(FakeEnv):
        def step(self, action):
            error = f"ValueError: invalid high-level action: {action}"
            return self._observation(last_action=action, last_action_error=error), 0.0, False, False, {}

    backend = BrowserGymBackend(
        _config(tmp_path),
        "session-invalid-action",
        WebArtifactStore(tmp_path),
    )
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: RejectingEnv())
    monkeypatch.setattr(backend, "_prepare_evaluator", lambda _task: None)
    backend.reset(WebTask(benchmark=WebBenchmark.WEBARENA, task_id="0", seed=7))

    result = backend.step(WebAction(name="click", script="click('missing')"))

    assert result.execution_ok is False
    assert result.terminated is False
    assert result.truncated is False
    assert result.observation.last_action == "click('missing')"
    assert "invalid high-level action" in result.observation.last_action_error
    assert result.info["action_error"] == result.observation.last_action_error
    backend.close()


def test_evaluator_value_error_is_not_misclassified_as_an_action_error(tmp_path, monkeypatch):
    class BrokenEvaluatorEnv(FakeEnv):
        def step(self, action):
            del action
            raise ValueError("OPENAI_API_KEY environment variable must be set")

    backend = BrowserGymBackend(
        _config(tmp_path),
        "session-evaluator-error",
        WebArtifactStore(tmp_path),
    )
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: BrokenEvaluatorEnv())
    monkeypatch.setattr(backend, "_prepare_evaluator", lambda _task: None)
    backend.reset(WebTask(benchmark=WebBenchmark.WEBARENA, task_id="0", seed=7))

    with pytest.raises(EvaluatorInfrastructureError, match="OPENAI_API_KEY"):
        backend.step(WebAction(name="noop", script="noop()"))

    backend.close()


@pytest.mark.parametrize(
    "reference_answers",
    [
        {"fuzzy_match": ["answer"]},
        {"fuzzy_match": "N/A"},
    ],
)
def test_model_backed_evaluator_requires_explicit_configuration(tmp_path, reference_answers):
    backend = BrowserGymBackend(
        _config(tmp_path),
        "session-missing-evaluator",
        WebArtifactStore(tmp_path),
    )
    task = WebTask(
        benchmark=WebBenchmark.WEBARENA,
        task_id="8",
        original_metadata={"eval": {"reference_answers": reference_answers}},
    )

    with pytest.raises(EvaluatorConfigurationError, match="webarena_evaluator_model"):
        backend.reset(task)


def test_rule_only_visualwebarena_satisfies_upstream_import_without_real_credential(
    tmp_path,
    monkeypatch,
):
    seen_keys = []

    class ImportSensitiveEnv(FakeEnv):
        def reset(self, seed):
            seen_keys.append(os.environ["OPENAI_API_KEY"])
            return super().reset(seed)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = BrowserGymBackend(
        _config(tmp_path),
        "session-vwa-rule-only",
        WebArtifactStore(tmp_path),
    )
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: ImportSensitiveEnv())
    task = WebTask(
        benchmark=WebBenchmark.VISUALWEBARENA,
        task_id="0",
        seed=7,
        original_metadata={"eval": {"reference_answers": {"exact_match": "answer"}}},
    )

    backend.reset(task)

    assert seen_keys == ["unused-for-rule-only-evaluator"]
    assert "OPENAI_API_KEY" not in os.environ
    backend.close()


def test_evaluate_uses_terminal_native_score_and_retains_best_reward(tmp_path, monkeypatch):
    class PartialRewardEnv(FakeEnv):
        def step(self, action):
            self.actions.append(action)
            terminal = action.startswith("send_msg_to_user")
            reward = 0.0 if terminal else 0.5
            return self._observation(last_action=action), reward, terminal, False, {}

    env = PartialRewardEnv()
    backend = BrowserGymBackend(
        _config(tmp_path),
        "session-score-semantics",
        WebArtifactStore(tmp_path),
    )
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: env)
    monkeypatch.setattr(backend, "_prepare_evaluator", lambda _task: None)
    backend.reset(WebTask(benchmark=WebBenchmark.WEBARENA, task_id="0", seed=7))

    step = backend.step(WebAction(name="noop", script="noop()"))
    evaluation = backend.evaluate("done")

    assert step.benchmark_reward == 0.5
    assert evaluation.reward == 0.0
    assert evaluation.metadata["best_observed_reward"] == 0.5
    assert evaluation.metadata["score_semantics"] == "terminal_native_evaluator_reward"
    backend.close()


def test_visualwebarena_evaluator_hook_runs_after_upstream_reset(tmp_path, monkeypatch):
    events = []

    class OrderedEnv(FakeEnv):
        def reset(self, seed):
            events.append("reset")
            return super().reset(seed)

    config = _config(tmp_path).model_copy(update={"visualwebarena_evaluator_model": "azure/anthropic/claude-opus-4-7"})
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    backend = BrowserGymBackend(config, "session-vwa", WebArtifactStore(tmp_path))
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: OrderedEnv())
    monkeypatch.setattr(
        backend_module,
        "configure_evaluator_model",
        lambda model: events.append(f"configure:{model}"),
    )

    backend.reset(WebTask(benchmark=WebBenchmark.VISUALWEBARENA, task_id="0", seed=7))

    assert events == ["reset", "configure:azure/anthropic/claude-opus-4-7"]
    backend.close()


def test_webarena_evaluator_hook_runs_after_upstream_reset(tmp_path, monkeypatch):
    events = []

    class OrderedEnv(FakeEnv):
        def reset(self, seed):
            events.append("reset")
            return super().reset(seed)

    config = _config(tmp_path).model_copy(update={"webarena_evaluator_model": "local-judge"})
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    backend = BrowserGymBackend(config, "session-wa", WebArtifactStore(tmp_path))
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: OrderedEnv())
    monkeypatch.setattr(
        backend_module,
        "configure_evaluator_environment",
        lambda **kwargs: events.append(f"environment:{kwargs['api_key']}"),
    )
    monkeypatch.setattr(
        backend_module,
        "configure_webarena_evaluator_model",
        lambda model: events.append(f"configure:{model}"),
    )

    _observation, info = backend.reset(WebTask(benchmark=WebBenchmark.WEBARENA, task_id="0", seed=7))

    assert events == ["environment:test-only", "reset", "configure:local-judge"]
    assert info["evaluator_model"] == "local-judge"
    assert info["verifier_version"].endswith(":judge=local-judge")
    backend.close()
