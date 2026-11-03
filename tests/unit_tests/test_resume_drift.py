# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A resume across a changed configuration is decided on what the change is, not on the digest alone.

This is the repin smoke: a manifest saved under one configuration is resumed under another, as a run moved to a new
Gym commit is. Operational drift (a new timeout, a new canary or tracking setting, a removed null default) resumes and
is recorded; semantic drift (model, sampling, repeats, grading) refuses with the keys named. A manifest saved with only
its digest keeps the earlier digest-and-alias rule.
"""

import json
import warnings
from pathlib import Path

import pytest

from nemo_gym.config.config_types import ConfigError
from nemo_gym.rollouts.rollout_recovery import (
    RunManifest,
    _configuration_identity,
    _digest,
    _is_semantic,
    config_identity_path_for,
    resume_drift_path_for,
    validate_resume,
)


BASE = {
    "policy_model": {
        "responses_api_models": {
            "vllm_model": {
                "model": "demo-model",
                "base_url": "http://127.0.0.1:8000/v1",
                "chat_template_kwargs": None,
                "sampling_overrides": None,
                "extra_body": None,
                "request_timeout_s": 600,
            }
        }
    },
    "demo_judge": {
        "resources_servers": {
            "demo_grader": {
                "judge_model_server": {"type": "responses_api_models", "name": "policy_model"},
                "judge_prompt": "Grade the answer.",
                "max_concurrency": 32,
            }
        }
    },
}


def merged(base: dict, change: dict) -> dict:
    out = json.loads(json.dumps(base))
    for dotted, value in change.items():
        *parents, leaf = dotted.split(".")
        node = out
        for key in parents:
            node = node.setdefault(key, {})
        if value is _DROP:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return out


_DROP = object()
VLLM = "policy_model.responses_api_models.vllm_model"
GRADER = "demo_judge.resources_servers.demo_grader"


def saved_run(tmp_path: Path, global_config: dict, collection: dict | None = None) -> tuple[Path, Path]:
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n")
    materialized = tmp_path / "materialized.jsonl"
    materialized.write_text("")
    path = tmp_path / "rollouts_manifest.json"
    RunManifest.create(source, [], collection or {}, global_config).write(path)
    return path, materialized


def current_manifest(tmp_path: Path, global_config: dict, collection: dict | None = None) -> RunManifest:
    return RunManifest.create(tmp_path / "input.jsonl", [], collection or {}, global_config)


@pytest.mark.parametrize(
    "drift",
    [
        {f"{VLLM}.connect_timeout_s": 30},  # a new timeout key
        {f"{VLLM}.request_timeout_s": 1200},
        {f"{GRADER}.max_concurrency": 64},
        {f"{VLLM}.max_retries": 5},
        {f"{VLLM}.canary_cells": 4},  # a new canary setting
        {f"{VLLM}.mlflow_tracking_required": False},  # a tracking flag
        {f"{VLLM}.extra_body": _DROP, f"{VLLM}.sampling_overrides": _DROP},  # removed null defaults
    ],
)
def test_operational_drift_resumes_lists_the_keys_and_records_them(tmp_path: Path, drift: dict) -> None:
    path, materialized = saved_run(tmp_path, BASE)
    current = current_manifest(tmp_path, merged(BASE, drift))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resumed = validate_resume(path, current, materialized)

    assert resumed is not None and not resumed.identity_overridden
    journal = resume_drift_path_for(path)
    if all(value is _DROP for value in drift.values()):
        # Null and absent are the same setting: no drift at all, nothing recorded.
        assert not journal.exists() and not caught
        return
    (entry,) = [json.loads(line) for line in journal.read_text().splitlines()]
    named = {key.removeprefix("servers.") for key in entry["operational_keys"]}
    assert named == set(drift)
    assert any(all(key in str(w.message) for key in named) for w in caught)


@pytest.mark.parametrize(
    "drift",
    [
        {f"{VLLM}.model": "other-model"},
        {f"{VLLM}.extra_body": {"max_thinking_tokens": 2048}},
        {f"{VLLM}.chat_template_kwargs": {"enable_thinking": True}},
        {f"{VLLM}.temperature": 0.6},
        {f"{GRADER}.judge_prompt": "Grade strictly."},
        {f"{GRADER}.judge_model_server.name": "other_judge"},
    ],
)
def test_semantic_drift_refuses_and_names_the_keys(tmp_path: Path, drift: dict) -> None:
    global_config = merged(BASE, {"other_judge": {}} if "other_judge" in str(drift) else {})
    if "other_judge" in str(drift):
        global_config["other_judge"] = {"responses_api_models": {"vllm_model": {"model": "judge"}}}
    path, materialized = saved_run(tmp_path, global_config)
    current = current_manifest(tmp_path, merged(global_config, drift))

    with pytest.raises(ConfigError, match="incompatible resolved configuration") as raised:
        validate_resume(path, current, materialized)

    assert "semantic settings differ" in str(raised.value)
    assert all(key in str(raised.value) for key in drift)
    assert not resume_drift_path_for(path).exists()


def test_repeats_and_task_limits_are_semantic_in_the_collection_identity(tmp_path: Path) -> None:
    path, materialized = saved_run(tmp_path, BASE, {"num_repeats": 4, "limit": 10})
    current = current_manifest(tmp_path, BASE, {"num_repeats": 8, "limit": 10})

    with pytest.raises(ConfigError, match=r"semantic settings differ: collection\.num_repeats"):
        validate_resume(path, current, materialized)


def test_a_digest_only_manifest_keeps_the_digest_rule_and_says_why(tmp_path: Path) -> None:
    path, materialized = saved_run(tmp_path, BASE)
    config_identity_path_for(path).unlink()  # as a manifest from before identities were saved
    current = current_manifest(tmp_path, merged(BASE, {f"{VLLM}.request_timeout_s": 1200}))

    with pytest.raises(ConfigError, match="saved only a digest of its configuration"):
        validate_resume(path, current, materialized)
    # The same configuration still resumes, and its identity is saved for the next resume.
    assert validate_resume(path, current_manifest(tmp_path, BASE), materialized) is not None
    assert config_identity_path_for(path).exists()


def test_an_identity_that_is_not_the_digested_one_is_not_trusted(tmp_path: Path) -> None:
    path, materialized = saved_run(tmp_path, BASE)
    edited = merged(BASE, {f"{VLLM}.model": "other-model"})
    # Someone rewrote the identity to match a semantic change; its digest is no longer the manifest's.
    tampered = _configuration_identity({}, edited, [])
    config_identity_path_for(path).write_text(json.dumps(tampered))
    assert _digest(tampered) != json.loads(path.read_text())["config_digest"]

    with pytest.raises(ConfigError, match="saved only a digest"):
        validate_resume(path, current_manifest(tmp_path, edited), materialized)


@pytest.mark.parametrize(
    ("key", "semantic"),
    [
        ("servers.p.responses_api_models.vllm_model.model", True),
        ("servers.p.responses_api_models.vllm_model.extra_body.timeout", True),
        ("servers.j.resources_servers.g.judge_model_server.name", True),
        ("servers.j.resources_servers.g.reward_threshold", True),
        ("collection.limit", True),
        ("collection.task_parameters.request_timeout_s", True),
        ("servers.j.resources_servers.g.rubric_model.name", True),
        ("servers.j.resources_servers.g.grader.mode", True),
        ("servers.j.resources_servers.g.eval_mode", True),
        ("servers.j.resources_servers.g.answer_format", True),
        ("servers.j.resources_servers.g.reference_key", True),
        ("servers.j.resources_servers.g.grader.request_timeout_s", False),
        ("servers.j.resources_servers.g.judge.max_retries", False),
        ("servers.j.resources_servers.g.judge.max_concurrency", False),
        ("servers.j.resources_servers.g.rubric_model.base_url", False),
        ("servers.j.resources_servers.g.reference_dir", False),
        ("servers.j.resources_servers.g.judge_timeout_s", False),
        ("servers.a.responses_api_agents.osworld_agent.exact_trace_transport_version", False),
        ("servers.p.responses_api_models.vllm_model.port", False),
        ("servers.p.responses_api_models.vllm_model.canary_cells", False),
    ],
)
def test_the_classification_of_a_key(key: str, semantic: bool) -> None:
    assert _is_semantic(key) is semantic


def two_policies(a: dict, b: dict) -> dict:
    return {
        "policy_a": {"responses_api_models": {"vllm_model": {"model": "model-a", **a}}},
        "policy_b": {"responses_api_models": {"vllm_model": {"model": "model-b", **b}}},
    }


def saved_by_an_earlier_gym(tmp_path: Path, global_config: dict) -> tuple[Path, Path]:
    """A manifest whose digest was taken from the identity as resolved, with no identity beside it."""
    path, materialized = saved_run(tmp_path, global_config)
    manifest = json.loads(path.read_text())
    manifest["config_digest"] = _digest(_configuration_identity({}, global_config, []))
    path.write_text(json.dumps(manifest))
    config_identity_path_for(path).unlink()
    return path, materialized


THINKING = {"chat_template_kwargs": {"thinking": True}}


def test_a_default_removed_on_one_server_resumes_beside_another_that_sets_it(tmp_path: Path) -> None:
    # Saved: A had the null default, B its own value. Current: A without the key, B unchanged.
    path, materialized = saved_by_an_earlier_gym(tmp_path, two_policies({"chat_template_kwargs": None}, THINKING))
    current = current_manifest(tmp_path, two_policies({}, THINKING))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert validate_resume(path, current, materialized) is not None

    (entry,) = [json.loads(line) for line in resume_drift_path_for(path).read_text().splitlines()]
    assert entry["operational_keys"] == ["servers.policy_a.responses_api_models.vllm_model.chat_template_kwargs"]
    assert any("policy_a.responses_api_models.vllm_model.chat_template_kwargs" in str(w.message) for w in caught)


def test_a_changed_value_on_the_other_server_is_still_refused(tmp_path: Path) -> None:
    path, materialized = saved_by_an_earlier_gym(tmp_path, two_policies({"chat_template_kwargs": None}, THINKING))
    current = current_manifest(tmp_path, two_policies({}, {"chat_template_kwargs": {"thinking": False}}))

    with pytest.raises(ConfigError, match="incompatible resolved configuration"):
        validate_resume(path, current, materialized)
    assert not resume_drift_path_for(path).exists()


def test_a_saved_identity_names_a_changed_value_on_the_other_server(tmp_path: Path) -> None:
    path, materialized = saved_run(tmp_path, two_policies({"chat_template_kwargs": None}, THINKING))
    current = current_manifest(tmp_path, two_policies({}, {"chat_template_kwargs": {"thinking": False}}))

    with pytest.raises(ConfigError, match=r"semantic settings differ: servers\.policy_b\.") as raised:
        validate_resume(path, current, materialized)
    assert "policy_a" not in str(raised.value)


@pytest.mark.parametrize(
    "key",
    [
        "servers.agent.responses_api_agents.simple_agent.max_steps",
        "servers.agent.responses_api_agents.simple_agent.execute_tools",
        "servers.agent.responses_api_agents.simple_agent.some_new_setting",
    ],
)
def test_an_agent_behaviour_or_unknown_setting_is_semantic(key):
    from nemo_gym.rollouts.rollout_recovery import _is_semantic

    assert _is_semantic(key)


@pytest.mark.parametrize(
    "key",
    [
        "servers.agent.responses_api_agents.simple_agent.concurrency",
        "servers.judge.responses_api_models.openai_model.max_concurrent_requests",
        "servers.judge.responses_api_models.openai_model.request_timeout_s",
        "servers.agent.responses_api_agents.simple_agent.max_retries",
    ],
)
def test_an_operational_setting_stays_operational(key):
    from nemo_gym.rollouts.rollout_recovery import _is_semantic

    assert not _is_semantic(key)


def judge_with_template(template: str) -> dict:
    return {
        "policy_model": BASE["policy_model"],
        "equivalence_llm_judge": {
            "resources_servers": {
                "equivalence_llm_judge": {
                    "judge_model_server": {"type": "responses_api_models", "name": "policy_model"},
                    "judge_prompt_template_fpath": template,
                    "judge_endpoint_max_concurrency": 32,
                }
            }
        },
    }


def test_a_different_grading_prompt_template_refuses_the_resume(tmp_path: Path) -> None:
    # Both are shipped templates: the verifier reads different grading instructions, not a moved copy.
    path, materialized = saved_run(tmp_path, judge_with_template("prompt_templates/equivalence_llm_judge.txt"))
    current = current_manifest(tmp_path, judge_with_template("prompt_templates/lc_judge.txt"))

    with pytest.raises(ConfigError, match=r"semantic settings differ: .*judge_prompt_template_fpath"):
        validate_resume(path, current, materialized)
    assert not resume_drift_path_for(path).exists()


@pytest.mark.parametrize(
    ("key", "semantic"),
    [
        ("servers.j.resources_servers.g.judge_prompt_template_fpath", True),
        ("servers.j.resources_servers.g.rubric_judge_prompt_template_fpath", True),
        ("servers.a.responses_api_agents.r.retrieval_system_prompt_fpath", True),
        ("servers.j.resources_servers.g.policy_verifier_templates_path", True),
        ("servers.j.resources_servers.g.prompt_dir", True),
        ("servers.j.resources_servers.g.judge_prompt_timeout_s", False),
        ("servers.j.resources_servers.g.judge_telemetry_output_dir", False),
        ("servers.j.resources_servers.g.cache_dir", False),
    ],
)
def test_a_prompt_or_template_path_is_semantic_and_other_paths_stay_operational(key: str, semantic: bool) -> None:
    assert _is_semantic(key) is semantic


def claude_code_agent(**settings) -> dict:
    agent = {"model_server": {"type": "responses_api_models", "name": "policy_model"}, **settings}
    return {
        "policy_model": BASE["policy_model"],
        "claude_code": {"responses_api_agents": {"claude_code_agent": agent}},
    }


@pytest.mark.parametrize(
    ("saved", "current"),
    [({}, {"max_turns": None}), ({"max_turns": None}, {})],
    ids=["default-to-unlimited", "unlimited-to-default"],
)
def test_an_explicit_null_turn_budget_is_not_the_default_one(tmp_path: Path, saved: dict, current: dict) -> None:
    # claude_code_agent: max_turns omitted is 30 turns, max_turns null is unlimited.
    path, materialized = saved_run(tmp_path, claude_code_agent(**saved))
    resumed = current_manifest(tmp_path, claude_code_agent(**current))

    assert resumed.config_digest != json.loads(path.read_text())["config_digest"]
    with pytest.raises(ConfigError, match=r"semantic settings differ: .*claude_code_agent\.max_turns"):
        validate_resume(path, resumed, materialized)


def test_a_null_that_is_the_setting_default_is_still_the_same_as_absent(tmp_path: Path) -> None:
    path, materialized = saved_run(tmp_path, BASE)  # vllm_model extra_body etc. default to null
    current = current_manifest(tmp_path, merged(BASE, {f"{VLLM}.extra_body": _DROP}))

    assert current.config_digest == json.loads(path.read_text())["config_digest"]
    assert validate_resume(path, current, materialized) is not None
