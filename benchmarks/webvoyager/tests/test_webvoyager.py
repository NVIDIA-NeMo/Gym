# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

from benchmarks.webvoyager import prepare as webvoyager_prepare
from benchmarks.webvoyager.prepare import REPO_ROOT, write_env
from benchmarks.webvoyager.prepare import main as prepare_main
from benchmarks.webvoyager.summarize import load_dataset, load_rows, summarize, write_missing_rows
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


class _DownloadResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _source_rows(count: int = 552) -> bytes:
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


def test_prepare_downloads_and_reuses_the_hash_pinned_source(monkeypatch, tmp_path) -> None:
    payload = _source_rows()
    destination = tmp_path / "webvoyager_source.jsonl"
    calls = []
    monkeypatch.setattr(webvoyager_prepare, "SOURCE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        webvoyager_prepare.urllib.request,
        "urlopen",
        lambda url, timeout: calls.append((url, timeout)) or _DownloadResponse(payload),
    )

    assert webvoyager_prepare._download_source(destination) == destination
    assert destination.read_bytes() == payload
    assert calls == [(webvoyager_prepare.SOURCE_URL, 60)]

    monkeypatch.setattr(
        webvoyager_prepare.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("a valid cached source must not be downloaded again"),
    )
    assert webvoyager_prepare._download_source(destination) == destination


def test_prepare_enforces_the_maintained_552_task_population(monkeypatch, tmp_path) -> None:
    source = tmp_path / "webvoyager.jsonl"
    payload = _source_rows()
    source.write_bytes(payload)
    output = tmp_path / "prepared.jsonl"
    monkeypatch.setattr(webvoyager_prepare, "SOURCE_SHA256", hashlib.sha256(payload).hexdigest())

    assert webvoyager_prepare.prepare(source=source, output=output) == output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 552
    assert rows[0]["responses_create_params"]["input"] == []
    assert rows[0]["web_task"]["runtime_profile"] == "visual_browser"

    payload = _source_rows(551)
    source.write_bytes(payload)
    monkeypatch.setattr(webvoyager_prepare, "SOURCE_SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="exactly 552 tasks"):
        webvoyager_prepare.prepare(source=source, output=output)


def test_source_lock_matches_the_automatic_download() -> None:
    lock_path = Path(__file__).parents[1] / "source_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    assert lock == {
        "repository": "https://github.com/jayl940712/webarena_benchmarks",
        "commit": webvoyager_prepare.SOURCE_COMMIT,
        "path": "webvoyager.jsonl",
        "sha256": webvoyager_prepare.SOURCE_SHA256,
        "raw_url": webvoyager_prepare.SOURCE_URL,
        "task_count": 552,
    }


def test_nano_omni_policy_preserves_history_thinking() -> None:
    benchmark_dir = Path(__file__).parents[1]
    config = yaml.safe_load((benchmark_dir / "configs/nano_omni_policy.yaml").read_text(encoding="utf-8"))
    kwargs = config["policy_model"]["responses_api_models"]["vllm_model"]["chat_template_kwargs"]
    assert kwargs == {"truncate_history_thinking": False}

    recipe_lock = json.loads((benchmark_dir / "nano_omni_recipe_lock.json").read_text(encoding="utf-8"))
    assert recipe_lock["policy_transport_endpoint"] == "/v1/chat/completions"
    assert recipe_lock["policy_chat_template_kwargs"] == kwargs


def test_nano_omni_and_qwen_share_runtime_and_dataset_but_not_policy_protocol() -> None:
    benchmark_dir = Path(__file__).parents[1]
    nano = yaml.safe_load((benchmark_dir / "configs/nano_omni.yaml").read_text(encoding="utf-8"))
    qwen = yaml.safe_load((benchmark_dir / "configs/qwen35_122b_a10b.yaml").read_text(encoding="utf-8"))
    judge = yaml.safe_load(
        (REPO_ROOT / "resources_servers/webvoyager_judge/configs/gemini.yaml").read_text(encoding="utf-8")
    )
    nano_agent = nano["nano_omni_webvoyager_agent"]["responses_api_agents"]["web_agent"]
    qwen_agent = qwen["qwen35_webvoyager_agent"]["responses_api_agents"]["web_agent"]
    judge_timeout = judge["webvoyager_gemini_judge"]["resources_servers"]["webvoyager_judge"][
        "judge_call_timeout_secs"
    ]

    assert nano_agent["environment_server"]["name"] == "webvoyager_environment"
    assert qwen_agent["environment_server"]["name"] == "webvoyager_environment"
    assert nano_agent["datasets"] == qwen_agent["datasets"]
    assert nano_agent["policy_protocol"] == "nano_omni_toolcall"
    assert qwen_agent["policy_protocol"] == "qwen_xml_computer_use"
    assert qwen_agent["max_image_history"] == 20
    assert qwen_agent["qwen_fold_size"] == 10
    assert qwen_agent["qwen_history_n"] == 100
    assert nano_agent["judge_request_timeout_secs"] > judge_timeout
    assert qwen_agent["judge_request_timeout_secs"] > judge_timeout
    model = qwen["qwen35_policy_model"]["responses_api_models"]["vllm_model"]
    assert model["base_url"] == "${policy_base_url}"
    assert model["model"] == "${policy_model_name}"
    assert model["chat_template_kwargs"] == {"enable_thinking": True}
    assert model["sampling_overrides"] == {"temperature": 0.1, "top_p": 0.9}
    assert model["replace_developer_role_with_system"] is True


def test_qwen_profile_composes_with_the_runtime_config_schema() -> None:
    benchmark_dir = Path(__file__).parents[1]
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                DictConfig({"config_paths": [str(benchmark_dir / "configs/qwen35_122b_a10b.yaml")]}),
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )

    model = resolved.qwen35_policy_model.responses_api_models.vllm_model
    assert model.replace_developer_role_with_system is True
    assert model.chat_template_kwargs == {"enable_thinking": True}
    assert model.sampling_overrides == {"temperature": 0.1, "top_p": 0.9}


@pytest.mark.parametrize(
    ("profile", "agent_name", "sampling"),
    [
        ("nano_omni", "nano_omni_webvoyager_agent", {"max_output_tokens": 16384, "temperature": 0.1, "top_p": 0.95}),
        (
            "qwen35_122b_a10b",
            "qwen35_webvoyager_agent",
            {"max_output_tokens": 32768, "temperature": 0.1, "top_p": 0.9},
        ),
    ],
)
def test_prepare_writes_private_single_display_profiles(tmp_path, profile, agent_name, sampling) -> None:
    input_jsonl = tmp_path / "input.jsonl"
    input_jsonl.write_text("{}\n", encoding="utf-8")
    env_path = tmp_path / "env.yaml"

    assert write_env(
        env_path,
        profile=profile,
        input_jsonl=input_jsonl,
        output_jsonl=tmp_path / "rollouts.jsonl",
    )

    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    config = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    assert config["agent_name"] == agent_name
    assert config["num_samples_in_parallel"] == 1
    assert config["responses_create_params"] == sampling


def test_prepare_rejects_parallel_sessions_on_one_display(tmp_path) -> None:
    with pytest.raises(ValueError, match="isolated Gym processes"):
        write_env(
            tmp_path / "env.yaml",
            profile="qwen35_122b_a10b",
            input_jsonl=tmp_path / "input.jsonl",
            output_jsonl=tmp_path / "rollouts.jsonl",
            concurrency=2,
        )


def test_prepare_prints_copyable_cli_commands(monkeypatch, capsys, tmp_path) -> None:
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


def test_summary_keeps_fixed_denominator_and_exposes_masked_failures() -> None:
    report = summarize(
        [
            {"task_id": "a", "task_success": True, "mask_sample": False},
            {"task_id": "b", "task_success": False, "mask_sample": True, "failure_kind": "judge_unparseable"},
        ]
    )

    assert report["success"] == 1
    assert report["strict_sr"] == 1 / 552
    assert report["missing"] == 550
    assert report["failure_kinds"] == {"judge_unparseable": 1}
    assert report["comparable"] is False


def test_summary_merges_worker_outputs_and_builds_exact_cleanup_input(tmp_path) -> None:
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


def test_summary_discards_large_trajectory_payloads_while_loading(tmp_path) -> None:
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


def test_summary_marks_duplicate_worker_results_non_comparable() -> None:
    report = summarize(
        [
            {"task_id": "a", "task_success": True, "mask_sample": False},
            {"task_id": "a", "task_success": True, "mask_sample": False},
        ],
        expected_task_ids={"a"},
    )

    assert report["duplicate_task_ids"] == ["a"]
    assert report["comparable"] is False


def test_summary_retries_masked_rows_as_well_as_missing_rows(tmp_path) -> None:
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


def test_summary_accepts_declared_cleanup_supersession() -> None:
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
