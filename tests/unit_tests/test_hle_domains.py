# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from nemo_gym.reward_profile import compute_aggregate_metrics
from resources_servers.equivalence_llm_judge.app import LLMJudgeResourcesServer
from resources_servers.equivalence_llm_judge.domain_metrics import compute_domain_metrics, question_bootstrap_interval


def row(task, repeat, reward, tokens=100, category="Math"):
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": repeat,
        "category": category,
        "raw_subject": "Algebra/Geometry",
        "reward": reward,
        "response": {"usage": {"output_tokens": tokens}},
    }


def test_question_bootstrap_keeps_repeated_generations_together():
    interval = question_bootstrap_interval([(1, 2), (1, 2)])
    assert interval["lower"] == interval["upper"] == 0.5
    assert question_bootstrap_interval([(1, 2)]) is None
    varied = question_bootstrap_interval([(0, 2), (2, 2)])
    assert (varied["lower"], varied["upper"]) == (0, 1)
    assert varied == question_bootstrap_interval([(0, 2), (2, 2)])


def test_domain_means_intervals_and_encoded_labels():
    result = compute_domain_metrics([[row(0, 0, 1, 100), row(0, 1, 1, 200)], [row(1, 0, 0, 300), row(1, 1, 0, 400)]])
    prefix = "domains/category/Math/"
    assert result[prefix + "mean/reward"] == 0.5
    assert result[prefix + "mean/output_tokens"] == 250
    assert result[prefix + "question_count"] == 2
    assert result[prefix + "rollout_count"] == 4
    assert result[prefix + "ci_low_95_question_bootstrap/output_tokens"] == 150
    assert result[prefix + "ci_high_95_question_bootstrap/output_tokens"] == 350
    assert result["domains/raw_subject/Algebra%2FGeometry/mean/reward"] == 0.5


def test_missing_labels_tokens_and_single_question():
    result = compute_domain_metrics([[row(0, 0, 1, None)], [row(1, 0, 0, category=None)]])
    prefix = "domains/category/Math/"
    assert result["domains/category/missing_label_rollouts"] == 1
    assert result[prefix + "missing/output_tokens"] == 1
    assert prefix + "mean/output_tokens" not in result
    assert result[prefix + "ci_available/reward"] is False
    with pytest.raises(ValueError, match="Inconsistent"):
        compute_domain_metrics([[row(0, 0, 1), row(0, 1, 1, category="Physics")]])


def test_aggregation_hook_excludes_masked_and_preserves_headline_score():
    server = SimpleNamespace(config=SimpleNamespace(report_hle_domains=True))
    rows = [row(0, 0, 1), row(1, 0, 0), {**row(2, 0, 1), "mask_sample": True}]
    result = compute_aggregate_metrics(
        rows, compute_metrics_fn=lambda tasks: LLMJudgeResourcesServer.compute_metrics(server, tasks)
    )
    assert result.agent_metrics["domains/category/Math/mean/reward"] == 0.5
    assert result.agent_metrics["domains/category/Math/question_count"] == 2
    assert result.key_metrics["mean/reward"] == 0.5
    server.config.report_hle_domains = False
    assert LLMJudgeResourcesServer.compute_metrics(server, [[row(0, 0, 1)]]) == {}


@pytest.mark.parametrize("include_vision", [False, True])
def test_hle_preparation_preserves_domain_labels(monkeypatch, tmp_path, include_vision):
    import json
    import sys

    import benchmarks.hle.prepare as prepare
    import nemo_gym.global_config as global_config
    import nemo_gym.prompt as prompt

    data = {
        "image": "",
        "question": "Q",
        "answer": "A",
        "answer_type": "exactMatch",
        "id": "q1",
        "category": "Math",
        "raw_subject": "Algebra",
    }
    batch = SimpleNamespace(
        num_rows=1, column_names=list(data), column=lambda name: [SimpleNamespace(as_py=lambda: data[name])]
    )
    dataset = SimpleNamespace(with_format=lambda _: SimpleNamespace(iter=lambda **_: [batch]))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *args, **kwargs: dataset))
    monkeypatch.setattr(global_config, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(prompt, "load_prompt_config", lambda _: {})
    monkeypatch.setattr(prepare, "build_image_input", lambda *args: [])
    monkeypatch.setattr(prepare, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "text.jsonl")
    monkeypatch.setattr(prepare, "OUTPUT_VISION_FPATH", tmp_path / "vision.jsonl")
    record = json.loads(prepare.prepare(include_vision=include_vision).read_text())
    assert {key: record[key] for key in ("uuid", "category", "raw_subject")} == {
        "uuid": "q1",
        "category": "Math",
        "raw_subject": "Algebra",
    }


@pytest.mark.parametrize("vision", [False, True])
def test_hle_verified_preserves_domain_labels(vision, monkeypatch):
    import benchmarks.hle_verified.prepare as prepare

    monkeypatch.setattr(prepare, "build_image_input", lambda *args: [])
    record = prepare.format_entry(
        {"id": "q1", "question": "Q", "answer": "A", "category": "Math", "raw_subject": "Algebra"},
        prompt_config={} if vision else None,
    )
    assert record["category"] == "Math"
    assert record["raw_subject"] == "Algebra"


def test_repeat_ci_is_default_and_question_ci_remains_separate():
    prefix = "domains/category/Math/"
    tasks = [[row(q, 0, 0, 100), row(q, 1, 1, 300)] for q in (0, 1)]
    result = compute_domain_metrics(tasks)
    assert result[prefix + "repeat_count/reward"] == 2
    assert result[prefix + "ci_available/reward"] is True
    assert result[prefix + "ci_low_95_across_repeats/reward"] == pytest.approx(-5.853102368)
    assert result[prefix + "ci_high_95_across_repeats/reward"] == pytest.approx(6.853102368)
    assert result[prefix + "ci_low_95_question_bootstrap/reward"] == 0.5
    assert result[prefix + "ci_high_95_question_bootstrap/reward"] == 0.5
    assert result[prefix + "ci_low_95_question_bootstrap/output_tokens"] == 200
    assert result[prefix + "ci_low_95_across_repeats/output_tokens"] < 100
    single = compute_domain_metrics([[row(q, 0, q)] for q in (0, 1)])
    assert single[prefix + "ci_available/reward"] is False
    assert prefix + "ci_low_95_across_repeats/reward" not in single
    assert single[prefix + "question_ci_available/reward"] is True
    incomplete = compute_domain_metrics([tasks[0], tasks[1][:1]])
    assert incomplete[prefix + "ci_available/reward"] is False
    assert "coverage" in incomplete[prefix + "ci_unavailable_reason/reward"]
