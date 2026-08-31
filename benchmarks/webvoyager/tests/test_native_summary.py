# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest
import yaml

from benchmarks.webvoyager import prepare as webvoyager_prepare
from benchmarks.webvoyager.prepare import REPO_ROOT, write_env
from benchmarks.webvoyager.prepare import main as prepare_main
from benchmarks.webvoyager.summarize_native_v3 import (
    load_dataset,
    load_rows,
    summarize,
    write_missing_rows,
)


class _DownloadResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _legacy_source_rows(count: int = 643) -> bytes:
    rows = (
        json.dumps(
            {
                "web_name": "ArXiv",
                "id": f"ArXiv--{index}",
                "ques": f"Find paper {index}",
                "web": "https://arxiv.org/",
            }
        )
        for index in range(count)
    )
    return ("\n".join(rows) + "\n").encode()


def _native_source_rows(count: int = 552) -> bytes:
    rows = (
        json.dumps(
            {
                "web_name": "Allrecipes",
                "id": f"Allrecipes--{index}",
                "ques": f"Find recipe {index}",
                "web": "https://www.allrecipes.com/",
            }
        )
        for index in range(count)
    )
    return ("\n".join(rows) + "\n").encode()


def test_prepare_downloads_and_reuses_pinned_official_legacy_source(monkeypatch, tmp_path) -> None:
    payload = _legacy_source_rows()
    destination = tmp_path / "WebVoyager_data.jsonl"
    calls = []
    monkeypatch.setattr(webvoyager_prepare, "LEGACY_SOURCE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        webvoyager_prepare.urllib.request,
        "urlopen",
        lambda url, timeout: calls.append((url, timeout)) or _DownloadResponse(payload),
    )

    assert webvoyager_prepare._download_legacy_source(destination) == destination
    assert destination.read_bytes() == payload
    assert calls == [(webvoyager_prepare.LEGACY_SOURCE_URL, 60)]

    monkeypatch.setattr(
        webvoyager_prepare.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("a valid cached source must not be downloaded again"),
    )
    assert webvoyager_prepare._download_legacy_source(destination) == destination


def test_prepare_legacy_profile_is_self_contained_and_enforces_population(monkeypatch, tmp_path) -> None:
    source = tmp_path / "WebVoyager_data.jsonl"
    source.write_bytes(_legacy_source_rows())
    output = tmp_path / "prepared.jsonl"
    monkeypatch.delenv("WEBVOYAGER_SOURCE_JSONL", raising=False)
    monkeypatch.setattr(webvoyager_prepare, "_download_legacy_source", lambda: source)

    assert webvoyager_prepare.prepare(output=output) == output
    assert len(output.read_text(encoding="utf-8").splitlines()) == 643

    source.write_bytes(_legacy_source_rows(642))
    with pytest.raises(ValueError, match="exactly 643 tasks"):
        webvoyager_prepare.prepare(output=output)


def test_prepare_downloads_and_reuses_pinned_native_source(monkeypatch, tmp_path) -> None:
    payload = _native_source_rows()
    destination = tmp_path / "webvoyager_native_v3_source.jsonl"
    calls = []
    monkeypatch.setattr(webvoyager_prepare, "NATIVE_V3_SOURCE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        webvoyager_prepare.urllib.request,
        "urlopen",
        lambda url, timeout: calls.append((url, timeout)) or _DownloadResponse(payload),
    )

    assert webvoyager_prepare._download_native_v3_source(destination) == destination
    assert destination.read_bytes() == payload
    assert calls == [(webvoyager_prepare.NATIVE_V3_SOURCE_URL, 60)]

    monkeypatch.setattr(
        webvoyager_prepare.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("a valid cached source must not be downloaded again"),
    )
    assert webvoyager_prepare._download_native_v3_source(destination) == destination


def test_prepare_native_profile_is_self_contained_and_enforces_population(monkeypatch, tmp_path) -> None:
    source = tmp_path / "webvoyager.jsonl"
    payload = _native_source_rows()
    source.write_bytes(payload)
    output = tmp_path / "prepared.jsonl"
    monkeypatch.delenv("WEBVOYAGER_SOURCE_JSONL", raising=False)
    monkeypatch.setattr(webvoyager_prepare, "NATIVE_V3_SOURCE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(webvoyager_prepare, "_download_native_v3_source", lambda: source)

    assert webvoyager_prepare.prepare_native(output=output) == output
    assert len(output.read_text(encoding="utf-8").splitlines()) == 552

    payload = _native_source_rows(551)
    source.write_bytes(payload)
    monkeypatch.setattr(webvoyager_prepare, "NATIVE_V3_SOURCE_SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="exactly 552 tasks"):
        webvoyager_prepare.prepare_native(output=output)


def test_native_v3_source_lock_matches_the_automatic_download() -> None:
    lock_path = Path(__file__).parents[1] / "native_v3_source_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    assert lock == {
        "repository": "https://github.com/jayl940712/webarena_benchmarks",
        "commit": webvoyager_prepare.NATIVE_V3_SOURCE_COMMIT,
        "path": "webvoyager.jsonl",
        "raw_url": webvoyager_prepare.NATIVE_V3_SOURCE_URL,
        "sha256": webvoyager_prepare.NATIVE_V3_SOURCE_SHA256,
        "task_count": 552,
    }


def test_native_v3_policy_preserves_history_thinking() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "native_v3_policy.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    kwargs = config["policy_model"]["responses_api_models"]["vllm_model"]["chat_template_kwargs"]
    assert kwargs == {"truncate_history_thinking": False}

    recipe_lock_path = Path(__file__).parents[1] / "native_v3_recipe_lock.json"
    recipe_lock = json.loads(recipe_lock_path.read_text(encoding="utf-8"))
    assert recipe_lock["policy_transport_endpoint"] == "/v1/chat/completions"
    assert recipe_lock["policy_chat_template_kwargs"] == kwargs
    assert recipe_lock["policy_chat_template_sha256"] == (
        "41428e0c65e312c359df2495ef5284769a9520b15a693deda4c34a1538208faa"  # pragma: allowlist secret
    )


def test_native_v3_robust_evaluation_is_scoped_to_the_benchmark_profile() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "native_v3.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).parents[3]
    base_config = yaml.safe_load(
        (repo_root / "resources_servers/webvoyager_browser/configs/webvoyager_browser.yaml").read_text(
            encoding="utf-8"
        )
    )["webvoyager_browser"]["resources_servers"]["webvoyager_browser"]
    base_agent = yaml.safe_load(
        (repo_root / "responses_api_agents/web_agent/configs/web_agent.yaml").read_text(encoding="utf-8")
    )["web_agent"]["responses_api_agents"]["web_agent"]

    resources = config["native_webvoyager_resources"]["resources_servers"]["webvoyager_browser"]
    agent = config["native_webvoyager_agent"]["responses_api_agents"]["web_agent"]

    assert base_config["terminate_on_action_error"] is True
    assert base_config["max_computer_actions"] == 20
    assert base_agent["native_action_recovery"] == "strict"
    assert base_agent["native_tool_alias_recovery"] == "strict"
    assert base_agent["native_parse_retry_feedback"] is False
    assert base_agent["native_parse_retry_temperature"] is None
    assert base_agent["native_parse_retry_delay_secs"] == 0.0
    assert base_agent["repeated_action_warning_threshold"] == 0
    assert resources["terminate_on_action_error"] is False
    assert resources["max_computer_actions"] == 20
    assert agent["native_action_recovery"] == "repair_single_closing_bracket"
    assert agent["native_tool_alias_recovery"] == "webvoyager_v3"
    assert agent["native_parse_retry_feedback"] is False
    assert agent["native_parse_retry_temperature"] is None
    assert agent["native_parse_retry_delay_secs"] == 1.0
    assert agent["max_consecutive_execution_failures"] == 3
    assert agent["resources_request_timeout_secs"] == 420
    assert agent["judge_request_timeout_secs"] == 540
    assert agent["environment_server"]["name"] == "native_webvoyager_resources"
    assert agent["resources_server"]["name"] == "native_webvoyager_judge"
    # Every opt-in above repairs how an already-chosen action is decoded or
    # executed. The repeat warning instead writes strategy advice into the
    # policy's context, which the pinned reference never sends, so this profile
    # leaves it off and keeps the trajectory comparable.
    assert agent["repeated_action_warning_threshold"] == 0


def test_prepare_writes_private_native_profile_with_standard_resource_roles(tmp_path) -> None:
    input_jsonl = tmp_path / "input.jsonl"
    input_jsonl.write_text("{}\n", encoding="utf-8")
    env_path = tmp_path / "env.yaml"

    assert write_env(
        env_path,
        profile="native_v3",
        input_jsonl=input_jsonl,
        output_jsonl=tmp_path / "rollouts.jsonl",
    )

    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    config = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    assert config["agent_name"] == "native_webvoyager_agent"
    assert config["num_samples_in_parallel"] == 1
    assert config["responses_create_params"] == {
        "max_output_tokens": 16384,
        "temperature": 0.1,
        "top_p": 0.95,
    }
    assert config["policy_api_key"] == "${oc.env:POLICY_API_KEY,local-vllm}"
    assert config["webvoyager_judge_api_key"] == "${oc.env:WEBARENA_JUDGE_API_KEY,unset}"


def test_prepare_rejects_parallel_sessions_on_one_native_display(tmp_path) -> None:
    with pytest.raises(ValueError, match="isolated Gym processes"):
        write_env(
            tmp_path / "env.yaml",
            profile="native_v3",
            input_jsonl=tmp_path / "input.jsonl",
            output_jsonl=tmp_path / "rollouts.jsonl",
            concurrency=2,
        )


def test_prepare_prints_copyable_locked_cli_commands(monkeypatch, capsys, tmp_path) -> None:
    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("benchmarks.webvoyager.prepare.prepare", lambda source, output: prepared)
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--no-env"])

    prepare_main()

    output = capsys.readouterr().out
    gym_cli = str(REPO_ROOT / ".venv" / "bin" / "gym")
    assert f"{gym_cli} env prefetch" in output
    assert f"{gym_cli} env start" in output
    assert f"{gym_cli} eval run --no-serve" in output
    assert "/path/to/Gym" not in output


def test_native_summary_keeps_fixed_denominator_and_exposes_masked_failures() -> None:
    report = summarize(
        [
            {"task_id": "a", "task_success": True, "mask_sample": False},
            {
                "task_id": "b",
                "task_success": False,
                "mask_sample": True,
                "failure_kind": "judge_unparseable",
            },
        ]
    )

    assert report["success"] == 1
    assert report["strict_sr"] == 1 / 552
    assert report["missing"] == 550
    assert report["failure_kinds"] == {"judge_unparseable": 1}
    assert report["comparable"] is False


def test_native_summary_merges_worker_outputs_and_builds_exact_cleanup_input(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset_rows = [
        {"responses_create_params": {"metadata": {"task_id": task_id}}, "payload": task_id}
        for task_id in ("a", "b", "c")
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in dataset_rows), encoding="utf-8")
    worker_root = tmp_path / "workers"
    for worker, rows in {
        "worker-00": [{"task_id": "a", "task_success": True, "mask_sample": False}],
        "worker-01": [{"task_id": "b", "task_success": False, "mask_sample": False}],
    }.items():
        output = worker_root / worker / "rollouts.jsonl"
        output.parent.mkdir(parents=True)
        output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    loaded_dataset, expected = load_dataset(dataset)
    report = summarize(load_rows([worker_root]), expected_task_ids=expected)
    cleanup = tmp_path / "cleanup.jsonl"
    write_missing_rows(loaded_dataset, set(report["missing_task_ids"]), cleanup)

    assert report["expected"] == 3
    assert report["completed_unique"] == 2
    assert report["missing_task_ids"] == ["c"]
    assert report["success"] == 1
    assert [json.loads(line)["payload"] for line in cleanup.read_text().splitlines()] == ["c"]


def test_native_summary_discards_large_trajectory_payloads_while_loading(tmp_path) -> None:
    output = tmp_path / "worker-00" / "rollouts.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text(
        json.dumps(
            {
                "task_id": "a",
                "task_success": False,
                "mask_sample": True,
                "failure_kind": "judge_unparseable",
                "responses": [{"screenshots": ["large-payload"]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_rows([output]) == [
        {
            "task_id": "a",
            "task_success": False,
            "mask_sample": True,
            "failure_kind": "judge_unparseable",
        }
    ]


def test_native_summary_marks_duplicate_worker_results_non_comparable() -> None:
    report = summarize(
        [
            {"task_id": "a", "task_success": True, "mask_sample": False},
            {"task_id": "a", "task_success": True, "mask_sample": False},
        ],
        expected_task_ids={"a"},
    )

    assert report["duplicate_task_ids"] == ["a"]
    assert report["comparable"] is False


def test_native_summary_retries_masked_rows_as_well_as_missing_rows(tmp_path) -> None:
    dataset_rows = [
        {"responses_create_params": {"metadata": {"task_id": task_id}}, "payload": task_id}
        for task_id in ("a", "b", "c")
    ]
    report = summarize(
        [
            {"task_id": "a", "task_success": False, "mask_sample": True},
            {"task_id": "b", "task_success": False, "mask_sample": False},
        ],
        expected_task_ids={"a", "b", "c"},
    )
    cleanup = tmp_path / "cleanup.jsonl"
    write_missing_rows(dataset_rows, set(report["retry_task_ids"]), cleanup)

    assert report["invalid_task_ids"] == ["a"]
    assert report["missing_task_ids"] == ["c"]
    assert report["retry_task_ids"] == ["a", "c"]
    assert [json.loads(line)["payload"] for line in cleanup.read_text().splitlines()] == ["a", "c"]


def test_native_summary_accepts_declared_cleanup_supersession() -> None:
    report = summarize(
        [
            {"task_id": "a", "task_success": False, "mask_sample": True},
            {"task_id": "a", "task_success": True, "mask_sample": False},
        ],
        expected_task_ids={"a"},
        superseded_task_ids={"a"},
    )

    assert report["duplicate_task_ids"] == []
    assert report["superseded_task_ids"] == ["a"]
    assert report["success"] == 1
    assert report["invalid_or_infrastructure"] == 0
    assert report["comparable"] is True
