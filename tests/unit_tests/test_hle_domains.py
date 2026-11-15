# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from nemo_gym.metrics.reward_profile import compute_aggregate_metrics
from resources_servers.equivalence_llm_judge.slice_metrics import compute_slice_metrics


def row(task, repeat, reward, tokens=100, **labels):
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": repeat,
        "reward": reward,
        "response": {"usage": {"output_tokens": tokens}},
        **labels,
    }


def test_generic_slices_and_cross_slices_match_shared_statistics():
    tasks = [
        [
            row(q, r, (q + r) % 2, 100 + 20 * q + 10 * r, sector="Finance/Tax", language="en" if q < 2 else "fr")
            for r in (0, 1)
        ]
        for q in range(3)
    ]
    result = compute_slice_metrics(tasks, ["sector", "language"], [("language", "sector")])
    for prefix, selected in [
        ("slices/sector/Finance%2FTax/", tasks),
        ("slices/language/sector/en/Finance%2FTax/", tasks[:2]),
        ("slices/language/fr/", tasks[2:]),
    ]:
        native = compute_aggregate_metrics([r for task in selected for r in task]).agent_metrics
        assert result[prefix + "question_count"] == len(selected)
        assert result[prefix + "rollout_count"] == 2 * len(selected)
        for metric in ("reward", "output_tokens"):
            assert result[prefix + f"mean/{metric}"] == native[f"mean/{metric}"]
            assert result[prefix + f"ci_available/{metric}"] is True
            assert result[prefix + f"repeat_count/{metric}"] == 2
            for bound in ("low", "high"):
                assert (
                    result[prefix + f"ci_{bound}_95_across_repeats/{metric}"]
                    == native[f"ci_{bound}_95_across_repeats/mean/{metric}"]
                )
    assert not any(k.startswith("domains/") for k in result)
    assert compute_slice_metrics(tasks, []) == {}


def test_missing_labels_and_values_do_not_silently_change_denominators():
    tasks = [[row(0, 0, 1, None, sector="Finance")], [row(1, 0, 0, sector="")]]
    result = compute_slice_metrics(tasks, ["sector"], [("sector", "language")])
    prefix = "slices/sector/Finance/"
    assert result["slices/sector/missing_label_rollouts"] == 1
    assert result["slices/sector/language/missing_label_rollouts"] == 2
    assert result[prefix + "missing/output_tokens"] == 1
    assert prefix + "mean/output_tokens" not in result
    assert result[prefix + "ci_available/reward"] is False
    assert prefix + "ci_low_95_across_repeats/reward" not in result


@pytest.mark.parametrize("repeats,reason", [([0, 0], "Duplicate"), ([0, None], "invalid"), ([0, -1], "invalid")])
def test_invalid_repeat_identity_suppresses_interval(repeats, reason):
    result = compute_slice_metrics([[row(0, r, 1, label="x") for r in repeats]], ["label"])
    assert result["slices/label/x/ci_available/reward"] is False
    assert reason in result["slices/label/x/ci_unavailable_reason/reward"]


def test_varying_labels_keep_generic_row_slicing_without_invalid_ci():
    tasks = [[row(0, 0, 1, label="a"), row(0, 1, 0, label="b")], [row(1, 0, 0, label="a"), row(1, 1, 1, label="a")]]
    result = compute_slice_metrics(tasks, ["label"])
    assert result["slices/label/a/question_count"] == 2
    assert result["slices/label/a/rollout_count"] == 3
    assert result["slices/label/a/mean/reward"] == pytest.approx(2 / 3)
    assert result["slices/label/a/ci_available/reward"] is False
    assert "coverage" in result["slices/label/a/ci_unavailable_reason/reward"]
    assert result["slices/label/b/mean/reward"] == 0


def test_field_and_label_paths_are_encoded_without_cross_slice_collisions():
    tasks = [[row(0, 0, 1, **{"a/b": "c/d", "a": "b/c", "b": "d"})]]
    result = compute_slice_metrics(tasks, ["a/b"], [("a", "b")])
    assert result["slices/a%2Fb/c%2Fd/mean/reward"] == 1
    assert result["slices/a/b/b%2Fc/d/mean/reward"] == 1


def test_numeric_labels_are_still_supported():
    result = compute_slice_metrics([[row(0, 0, 1, partition=2)]], ["partition"])
    assert result["slices/partition/2/mean/reward"] == 1


def test_configured_negative_rewards_are_not_treated_as_missing():
    # The shared grader allows reward_if_swap_fails=-1.
    result = compute_slice_metrics([[row(0, 0, -1, label="x"), row(0, 1, 1, label="x")]], ["label"])
    assert result["slices/label/x/missing/reward"] == 0
    assert result["slices/label/x/mean/reward"] == 0
    assert result["slices/label/x/ci_available/reward"] is True


@pytest.mark.parametrize("include_vision", [False, True])
def test_hle_preparation_preserves_domain_labels(monkeypatch, tmp_path, include_vision):
    import json
    import sys

    import benchmarks.hle.prepare as prepare
    import nemo_gym.config.global_config as global_config
    import nemo_gym.data.prompt as prompt

    data = {
        "image": "image-data" if include_vision else "",
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
    assert record["modality"] == ("vision" if include_vision else "text")
    assert "has_image" not in record
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
        {
            "id": "q1",
            "question": "Q",
            "answer": "A",
            "category": "Math",
            "raw_subject": "Algebra",
            "image": "image-data" if vision else "",
        },
        prompt_config={} if vision else None,
    )
    assert record["modality"] == ("vision" if vision else "text")
    assert "has_image" not in record
    assert record["category"] == "Math"
    assert record["raw_subject"] == "Algebra"
