# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import stat
import sys
from datetime import date
from pathlib import Path

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

from benchmarks.webvoyager import prepare as webvoyager_prepare
from benchmarks.webvoyager.prepare import REPO_ROOT, write_env
from benchmarks.webvoyager.prepare import main as prepare_main
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


BOOKING_TEMPLATE = {
    "web_name": "Booking",
    "id": "Booking--5",
    "ques_template": "Search a hotel with free WiFi and air conditioning in Bali from {dates}.",
    "date_policy": {"type": "hotel_stay", "min_days_from_today": 30, "duration_days": 3},
    "render": {"dates": "month_day_to_day_year"},
    "web": "https://www.booking.com/",
}
# The source's question for 2026-07-26: entry 0 starts 30 days later.
BOOKING_SOURCE_QUESTION = "Search a hotel with free WiFi and air conditioning in Bali from August 25 to 28, 2026."


def _source_rows(count: int = webvoyager_prepare.EXPECTED_TASKS) -> bytes:
    rows = [
        {
            "web_name": "Allrecipes",
            "id": f"Allrecipes--{index}",
            "ques": f"Find recipe {index}",
            "web": "https://www.allrecipes.com/",
        }
        for index in range(count - 1)
    ]
    rows.append(
        {"web_name": "Booking", "id": "Booking--5", "ques": BOOKING_SOURCE_QUESTION, "web": "https://www.booking.com/"}
    )
    return ("\n".join(json.dumps(row) for row in rows) + "\n").encode()


def _pin_inputs(monkeypatch, tmp_path, payload: bytes, templates=(BOOKING_TEMPLATE,)) -> tuple[Path, Path]:
    source = tmp_path / "webvoyager.jsonl"
    source.write_bytes(payload)
    template = tmp_path / "webvoyager.template.jsonl"
    template.write_text("".join(json.dumps(entry) + "\n" for entry in templates), encoding="utf-8")
    monkeypatch.setattr(webvoyager_prepare, "SOURCE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(webvoyager_prepare, "DATE_TEMPLATE_SHA256", hashlib.sha256(template.read_bytes()).hexdigest())
    monkeypatch.setattr(webvoyager_prepare, "_today", lambda: date(2026, 10, 12))
    return source, template


def test_prepare_downloads_and_reuses_the_hash_pinned_source(monkeypatch, tmp_path) -> None:
    payload = _source_rows()
    destination = tmp_path / "webvoyager_source.jsonl"
    calls = []
    digest = hashlib.sha256(payload).hexdigest()
    url = webvoyager_prepare.SOURCE_URL
    monkeypatch.setattr(
        webvoyager_prepare.urllib.request,
        "urlopen",
        lambda url, timeout: calls.append((url, timeout)) or _DownloadResponse(payload),
    )

    assert webvoyager_prepare._download("WebVoyager source", url, digest, destination) == destination
    assert destination.read_bytes() == payload
    assert calls == [(url, 60)]

    monkeypatch.setattr(
        webvoyager_prepare.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("a valid cached source must not be downloaded again"),
    )
    assert webvoyager_prepare._download("WebVoyager source", url, digest, destination) == destination

    monkeypatch.setattr(
        webvoyager_prepare.urllib.request, "urlopen", lambda *_args, **_kwargs: _DownloadResponse(b"x")
    )
    with pytest.raises(ValueError, match="WebVoyager source hash mismatch"):
        webvoyager_prepare._download("WebVoyager source", url, digest, tmp_path / "other.jsonl")
    assert not (tmp_path / "other.jsonl").exists()


def test_prepare_enforces_the_maintained_552_task_population(monkeypatch, tmp_path) -> None:
    source, template = _pin_inputs(monkeypatch, tmp_path, _source_rows())
    output = tmp_path / "prepared.jsonl"

    assert webvoyager_prepare.prepare(source=source, output=output, date_template=template) == output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == webvoyager_prepare.EXPECTED_TASKS
    assert rows[0]["responses_create_params"]["input"] == []
    assert rows[0]["web_task"]["runtime_profile"] == "visual_browser"

    source, template = _pin_inputs(monkeypatch, tmp_path, _source_rows(webvoyager_prepare.EXPECTED_TASKS - 1))
    with pytest.raises(ValueError, match=f"exactly {webvoyager_prepare.EXPECTED_TASKS} tasks"):
        webvoyager_prepare.prepare(source=source, output=output, date_template=template)


def test_prepare_renders_dated_questions_for_the_reference_date(monkeypatch, tmp_path) -> None:
    source, template = _pin_inputs(monkeypatch, tmp_path, _source_rows())
    output = tmp_path / "prepared.jsonl"

    webvoyager_prepare.prepare(source=source, output=output, date_template=template, reference_date="2026-10-11")

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    booking = rows[-1]["web_task"]
    # 11 weeks after the source's 2026-07-26: the same weekdays, 30 days ahead.
    assert (
        booking["intent"] == "Search a hotel with free WiFi and air conditioning in Bali from November 10 to 13, 2026."
    )
    assert booking["original_metadata"]["ques"] == BOOKING_SOURCE_QUESTION
    assert rows[0]["web_task"]["intent"] == "Find recipe 0"


def test_prepare_refuses_reference_dates_that_change_weekdays_or_have_passed(monkeypatch, tmp_path) -> None:
    source, template = _pin_inputs(monkeypatch, tmp_path, _source_rows())
    output = tmp_path / "prepared.jsonl"

    with pytest.raises(ValueError, match="whole number of weeks"):
        webvoyager_prepare.prepare(source=source, output=output, date_template=template, reference_date="2026-10-12")
    monkeypatch.setattr(webvoyager_prepare, "_today", lambda: date(2026, 11, 10))
    with pytest.raises(ValueError, match=r"Booking--5 \(2026-11-10\)"):
        webvoyager_prepare.prepare(source=source, output=output, date_template=template, reference_date="2026-10-11")
    assert not output.exists()


def test_date_template_must_reproduce_the_source_question(monkeypatch, tmp_path) -> None:
    edited = {**BOOKING_TEMPLATE, "ques_template": "Search a hotel in Bali from {dates}."}
    source, template = _pin_inputs(monkeypatch, tmp_path, _source_rows(), templates=(edited,))

    with pytest.raises(ValueError, match="does not reproduce Booking--5 for 2026-07-26"):
        webvoyager_prepare.prepare(source=source, output=tmp_path / "prepared.jsonl", date_template=template)


# Expected strings are the source repository's template/render_webvoyager.py output.
@pytest.mark.parametrize(
    ("style", "start", "end", "expected"),
    [
        ("month_day", date(2026, 9, 22), None, "September 22"),
        ("month_day_ordinal", date(2026, 9, 12), None, "September 12th"),
        ("month_day_year", date(2026, 9, 1), None, "September 1, 2026"),
        ("month_day_ordinal_year", date(2026, 9, 22), None, "September 22nd, 2026"),
        ("abbr_month_day", date(2026, 9, 1), None, "Sep. 1"),
        ("day_month_year", date(2026, 9, 12), None, "12 September 2026"),
        ("end_month_day_year", date(2026, 9, 9), date(2026, 9, 12), "September 12, 2026"),
        ("month_day_range", date(2026, 9, 9), date(2026, 9, 12), "September 9-12"),
        ("month_day_range", date(2026, 12, 30), date(2027, 1, 2), "December 30-January 2"),
        ("month_day_range_year", date(2026, 9, 9), date(2026, 9, 12), "September 9-12, 2026"),
        ("month_day_range_year", date(2026, 12, 30), date(2027, 1, 2), "December 30, 2026 - January 2, 2027"),
        ("ordinal_month_day_range", date(2026, 9, 29), date(2026, 10, 2), "September 29th - October 2nd"),
        ("month_day_to_day", date(2026, 9, 9), date(2026, 9, 12), "September 9 to 12"),
        ("month_day_to_day", date(2026, 9, 29), date(2026, 10, 2), "September 29 to October 2"),
        ("month_day_to_day_year", date(2026, 9, 9), date(2026, 9, 12), "September 9 to 12, 2026"),
        ("month_day_to_day_year", date(2026, 9, 29), date(2026, 10, 2), "September 29, 2026 to October 2, 2026"),
        ("day_month_range_year", date(2026, 9, 9), date(2026, 9, 12), "9 September to 12 September 2026"),
        ("day_month_range_year", date(2026, 12, 30), date(2027, 1, 2), "30 December 2026 to 2 January 2027"),
        ("numeric_dmy_range", date(2026, 12, 30), date(2027, 1, 2), "30/12/2026 - 02/01/2027"),
        ("sentence_range_year", date(2026, 9, 29), date(2026, 10, 2), "September 29, 2026, to October 2, 2026"),
        (
            "between_ordinal_range_year",
            date(2026, 12, 30),
            date(2027, 1, 2),
            "December 30th, 2026, and January 2nd, 2027",
        ),
    ],
)
def test_date_styles_match_the_source_renderer(style, start, end, expected) -> None:
    assert webvoyager_prepare._render_date_value(style, start, end) == expected


def test_provenance_matches_the_automatic_download_and_profiles() -> None:
    provenance_path = Path(__file__).parents[1] / "provenance.yaml"
    provenance = yaml.safe_load(provenance_path.read_text(encoding="utf-8"))

    assert provenance["dataset"] == {
        "repository": "https://github.com/jayl940712/webarena_benchmarks",
        "commit": webvoyager_prepare.SOURCE_COMMIT,
        "path": "webvoyager.jsonl",
        "sha256": webvoyager_prepare.SOURCE_SHA256,
        "raw_url": webvoyager_prepare.SOURCE_URL,
        "task_count": webvoyager_prepare.EXPECTED_TASKS,
        "date_template": {
            "path": "template/webvoyager.template.jsonl",
            "sha256": webvoyager_prepare.DATE_TEMPLATE_SHA256,
            "raw_url": webvoyager_prepare.DATE_TEMPLATE_URL,
            "source_reference_date": "2026-07-26",
        },
        "reference_date": webvoyager_prepare.REFERENCE_DATE.isoformat(),
    }
    commit_root = (
        f"https://raw.githubusercontent.com/jayl940712/webarena_benchmarks/{webvoyager_prepare.SOURCE_COMMIT}/"
    )
    assert webvoyager_prepare.DATE_TEMPLATE_URL == commit_root + "template/webvoyager.template.jsonl"
    assert (webvoyager_prepare.REFERENCE_DATE - webvoyager_prepare.SOURCE_REFERENCE_DATE).days % 7 == 0
    assert set(provenance["policy_profiles"]) == set(webvoyager_prepare.PROFILE_CONFIGS)
    assert {
        profile: config["sampling"] for profile, config in provenance["policy_profiles"].items()
    } == webvoyager_prepare.PROFILE_SAMPLING


def test_nano_omni_policy_preserves_history_thinking() -> None:
    benchmark_dir = Path(__file__).parents[1]
    config = yaml.safe_load((benchmark_dir / "configs/nano_omni.yaml").read_text(encoding="utf-8"))
    kwargs = config["policy_model"]["responses_api_models"]["vllm_model"]["chat_template_kwargs"]
    assert kwargs == {"truncate_history_thinking": False}

    provenance = yaml.safe_load((benchmark_dir / "provenance.yaml").read_text(encoding="utf-8"))
    profile = provenance["policy_profiles"]["nano_omni"]
    assert profile["transport_endpoint"] == "/v1/chat/completions"
    assert profile["chat_template"]["kwargs"] == kwargs


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
    assert "nano_omni_action_recovery" not in nano_agent
    assert "nano_omni_tool_alias_recovery" not in nano_agent
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

    provenance = yaml.safe_load((benchmark_dir / "provenance.yaml").read_text(encoding="utf-8"))
    qwen_profile = provenance["policy_profiles"]["qwen35_122b_a10b"]
    assert qwen_profile["chat_template_kwargs"] == model["chat_template_kwargs"]
    assert {
        "temperature": qwen_profile["sampling"]["temperature"],
        "top_p": qwen_profile["sampling"]["top_p"],
    } == model["sampling_overrides"]


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


def test_nano_omni_profile_composes_transport_override_with_runtime_schema() -> None:
    benchmark_dir = Path(__file__).parents[1]
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                DictConfig(
                    {
                        "config_paths": [str(benchmark_dir / "configs/nano_omni.yaml")],
                        "policy_base_url": "http://127.0.0.1:8000/v1",
                    }
                ),
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )

    model = resolved.policy_model.responses_api_models.vllm_model
    assert model.base_url == "http://127.0.0.1:8000/v1"
    assert model.chat_template_kwargs == {"truncate_history_thinking": False}


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
    monkeypatch.setattr("benchmarks.webvoyager.prepare.prepare", lambda source, output, **_dates: prepared)
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--no-env"])

    prepare_main()

    output = capsys.readouterr().out
    gym_cli = str(REPO_ROOT / ".venv" / "bin" / "gym")
    assert f"{gym_cli} env prefetch" in output
    assert f"{gym_cli} env start" in output
    assert f"{gym_cli} eval run --no-serve" in output
