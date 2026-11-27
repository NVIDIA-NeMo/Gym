# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-turn video benchmarks (videomme_v1, videomme_v2, lvbench) and lvbench_agentic: prepare.py, configs and
manifests on invented annotations (no dataset download)."""

import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from benchmarks.lvbench import prepare as lvbench_st
from benchmarks.lvbench_agentic import prepare as lvbench_ag
from benchmarks.videomme_v1 import prepare as v1_st
from benchmarks.videomme_v2 import prepare as v2_st
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.agents.base_responses_api_agent import BaseRunRequest
from nemo_gym.data import TaskDataValidator, load_task_data_schema
from nemo_gym.environment import BenchmarkConfig, load_manifest
from nemo_gym.environment.validation import validate_environment
from resources_servers.video_mcqa.app import AccuracyCohort, VideoMCQAResourcesServerConfig
from resources_servers.video_mcqa.tests.test_videomme_agentic_prepare import v1_record, v2_record
from resources_servers.videomme_v2.app import CohortManifest, VideoMMEV2ResourcesServerConfig
from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig
from responses_api_agents.simple_agent.app import SimpleAgentConfig


REPO_ROOT = Path(__file__).resolve().parents[3]


def assert_rows_valid(owner: str, rows: list[dict]) -> None:
    validator = TaskDataValidator(owner, load_task_data_schema(REPO_ROOT / "resources_servers" / owner), "rows.jsonl")
    for index, row in enumerate(rows):
        BaseRunRequest.model_validate(row)
        validator.validate_row(index, row)
    assert validator.report.clean, validator.report.summary()


def lvbench_video_record(key: str, qa: list[dict], *, video_type: str = "documentary") -> dict:
    return {
        "key": key,
        "video_path": f"{key}.mp4",
        "type": video_type,
        "qa": qa,
    }


def lvbench_qa(
    uid: str | int,
    question: str = "Which invented shape moves first?\n(A) Circle\n(B) Square\n(C) Triangle\n(D) Star",
    answer: str = "B",
    question_type: list[str] | None = None,
) -> dict:
    return {
        "uid": uid,
        "question": question,
        "answer": answer,
        "question_type": question_type or ["entity recognition", "reasoning"],
        "time_reference": "00:10-00:20",
    }


def stage_parquet(tmp_path, monkeypatch, module, records: list[dict], videos: list[str], env: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    parquet = tmp_path / "annotation.parquet"
    pq.write_table(pa.Table.from_pylist(records), parquet)
    monkeypatch.setenv(env, str(parquet))
    monkeypatch.setattr(module, "PARQUET_SHA256", hashlib.sha256(parquet.read_bytes()).hexdigest())
    monkeypatch.setattr(module, "NUM_QUESTIONS", len(records))
    monkeypatch.setattr(module, "OUTPUT_FPATH", tmp_path / "out" / "rows.jsonl")
    monkeypatch.setattr(module, "COHORT_FPATH", tmp_path / "out" / "cohort.json")
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (source / "FILES.sha256").write_text(
        "".join(f"{hashlib.sha256(video.encode()).hexdigest()}  ./video/{video}.mp4\n" for video in videos),
        encoding="utf-8",
    )
    return source


def stage_lvbench(tmp_path, monkeypatch, target_module, records: list[dict], videos: list[str]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    annotations = tmp_path / "video_info.meta.jsonl"
    annotations.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    monkeypatch.setenv("LVBENCH_ANNOTATIONS", str(annotations))
    monkeypatch.setattr(lvbench_st, "ANNOTATIONS_SHA256", hashlib.sha256(annotations.read_bytes()).hexdigest())
    monkeypatch.setattr(lvbench_st, "NUM_VIDEOS", len(records))
    monkeypatch.setattr(lvbench_st, "NUM_QUESTIONS", sum(len(record["qa"]) for record in records))
    monkeypatch.setattr(target_module, "OUTPUT_FPATH", tmp_path / "out" / "rows.jsonl")
    monkeypatch.setattr(target_module, "COHORT_FPATH", tmp_path / "out" / "cohort.json")
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (source / "FILES.sha256").write_text(
        "".join(f"{hashlib.sha256(video.encode()).hexdigest()}  ./video/{video}.mp4\n" for video in videos),
        encoding="utf-8",
    )
    return source


def read_rows(module) -> list[dict]:
    return [json.loads(line) for line in module.OUTPUT_FPATH.read_text(encoding="utf-8").splitlines()]


def test_videomme_v1_single_turn_prepare_and_errors(tmp_path, monkeypatch) -> None:
    records = [v1_record("clipA", 1), v1_record("clipA", 2), v1_record("clipB", 3)]
    source = stage_parquet(tmp_path, monkeypatch, v1_st, records, ["clipA", "clipB"], "VIDEOMME_V1_PARQUET")

    assert v1_st.prepare(source_dir=str(source)) == v1_st.OUTPUT_FPATH
    rows = read_rows(v1_st)
    assert [row["task_id"] for row in rows] == ["videomme_v1:001-1", "videomme_v1:001-2", "videomme_v1:001-3"]
    assert_rows_valid("videomme_v1", rows)
    for row in rows:
        assert "video_binding" not in row
        assert "expected_answer" not in row["responses_create_params"]

    video_part, text_part = rows[2]["responses_create_params"]["input"][0]["content"]
    assert video_part == {"type": "input_video", "video_url": f"file://{source.absolute()}/video/clipB.mp4"}
    assert text_part["text"] == (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\nHow many invented dots appear?\n"
        "A. One.\nB. Two.\nC. Three.\nD. Four.\nThe best answer is:"
    )
    assert rows[2]["verifier_metadata"]["video_sha256"] == hashlib.sha256(b"clipB").hexdigest()

    cohort = AccuracyCohort.model_validate_json(v1_st.COHORT_FPATH.read_text(encoding="utf-8"))
    assert (cohort.benchmark, cohort.setting, cohort.answer_policy, cohort.dataset_revision) == (
        "videomme_v1",
        "single_turn",
        "llm_extraction_v1",
        v1_st.DATASET_REVISION,
    )
    assert cohort.task_subsets["videomme_v1:001-3"] == {
        "duration": ("short",),
        "domain": ("Knowledge",),
        "sub_category": ("Invented",),
        "task_type": ("Counting Problem",),
    }

    # Custom video_url_dir rewrites the file:// URL; relative source_dir still yields an absolute file:// URL.
    v1_st.prepare(source_dir=str(source), video_url_dir="/mnt/vllm/video/")
    custom_video_part = read_rows(v1_st)[2]["responses_create_params"]["input"][0]["content"][0]
    assert custom_video_part == {"type": "input_video", "video_url": "file:///mnt/vllm/video/clipB.mp4"}

    monkeypatch.chdir(tmp_path)
    v1_st.prepare(source_dir="source")
    rel_video_part = read_rows(v1_st)[0]["responses_create_params"]["input"][0]["content"][0]
    assert rel_video_part == {"type": "input_video", "video_url": f"file://{source.absolute()}/video/clipA.mp4"}

    with pytest.raises(ValueError, match="videomme_v1_source_dir"):
        v1_st.prepare()
    with pytest.raises(ValueError, match="absolute path"):
        v1_st.prepare(source_dir=str(source), video_url_dir="relative/video")
    (source / "FILES.sha256").write_text(f"{'0' * 64}  video/clipA.mp4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no SHA-256 for video/clipB.mp4"):
        v1_st.prepare(source_dir=str(source))
    monkeypatch.setattr(v1_st, "PARQUET_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        v1_st.prepare(source_dir=str(source))


def test_videomme_v2_single_turn_prepare_and_errors(tmp_path, monkeypatch) -> None:
    records = [v2_record("001", position, level) for position, level in enumerate((1, 2, 3, 3), start=1)]
    records += [v2_record("002", position, 2, group_type="relevance") for position in range(1, 5)]
    records[5]["options"] = "A. Circle\nB. Circle\nC. Square"
    source = stage_parquet(tmp_path, monkeypatch, v2_st, records, ["001", "002"], "VIDEOMME_V2_PARQUET")

    assert v2_st.prepare(source_dir=str(source), video_url_dir="/mnt/v2/video") == v2_st.OUTPUT_FPATH
    rows = read_rows(v2_st)
    assert [row["task_id"] for row in rows[:4]] == [
        "videommev2-L1/001-1",
        "videommev2-L2/001-2",
        "videommev2-L3/001-3",
        "videommev2-L3/001-4",
    ]
    assert_rows_valid("videomme_v2", rows)
    for row in rows:
        assert "video_binding" not in row
        assert "expected_answer" not in row["responses_create_params"]

    video_part, text_part = rows[0]["responses_create_params"]["input"][0]["content"]
    assert video_part == {"type": "input_video", "video_url": "file:///mnt/v2/video/001.mp4"}
    assert text_part["text"] == (
        "Which invented shape moves in clip 001, part 1?\nA. Circle\nB. Square\nC. Triangle\n"
        "Respond with only the letter (A, B, or C) of the correct option.\nAnswer: "
    )
    sha256 = hashlib.sha256(b"001").hexdigest()
    metadata = rows[0]["verifier_metadata"]
    assert (metadata["expected_answer"], metadata["video_sha256"], metadata["level"], metadata["group_id"]) == (
        "B",
        sha256,
        1,
        "001",
    )
    assert rows[5]["annotation_warnings"] == ["duplicate_option_text"]

    cohort = CohortManifest.model_validate_json(v2_st.COHORT_FPATH.read_text(encoding="utf-8"))
    assert (cohort.schema_version, cohort.setting, cohort.answer_policy, cohort.dataset_revision) == (
        2,
        "single_turn",
        "llm_extraction_v1",
        v2_st.DATASET_REVISION,
    )
    logic, relevance = cohort.groups
    assert (logic.level, logic.group_type, logic.group_structure) == (3, "logic", "[1,2,3,4]")
    assert (relevance.level, relevance.group_type, relevance.group_structure) == (2, "relevance", None)
    assert [cohort.task_levels[task_id] for task_id in logic.task_ids] == [1, 2, 3, 3]

    with pytest.raises(ValueError, match="videomme_v2_source_dir"):
        v2_st.prepare()
    with pytest.raises(ValueError, match="absolute path"):
        v2_st.prepare(source_dir=str(source), video_url_dir="relative/video")
    with pytest.raises(ValueError, match="labelled in order"):
        v2_st.parse_options("B. Circle\nA. Square")
    (source / "FILES.sha256").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no SHA-256 for video/001.mp4"):
        v2_st.prepare(source_dir=str(source))
    monkeypatch.setattr(v2_st, "PARQUET_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        v2_st.prepare(source_dir=str(source))


def test_lvbench_single_turn_prepare_and_errors(tmp_path, monkeypatch) -> None:
    records = [
        lvbench_video_record(
            "clipA",
            [
                lvbench_qa("101", answer="B", question_type=["entity recognition", "reasoning"]),
                lvbench_qa(
                    102,
                    question="Which invented shape repeats?\n(A) Circle\n(B) Circle\n(C) Triangle\n(D) Star",
                    answer="A",
                    question_type=["summarization"],
                ),
            ],
            video_type="documentary",
        ),
        lvbench_video_record(
            "clipB",
            [lvbench_qa("201", answer="D", question_type=["temporal grounding"])],
            video_type="sports",
        ),
    ]
    source = stage_lvbench(tmp_path, monkeypatch, lvbench_st, records, ["clipA", "clipB"])

    assert lvbench_st.prepare(source_dir=str(source)) == lvbench_st.OUTPUT_FPATH
    rows = read_rows(lvbench_st)
    assert [row["task_id"] for row in rows] == ["lvbench:101", "lvbench:102", "lvbench:201"]
    assert_rows_valid("lvbench", rows)
    for row in rows:
        assert "video_binding" not in row
        assert "expected_answer" not in row["responses_create_params"]
        assert "00:10-00:20" not in json.dumps(row["responses_create_params"])

    video_part, text_part = rows[0]["responses_create_params"]["input"][0]["content"]
    assert video_part == {"type": "input_video", "video_url": f"file://{source.absolute()}/video/clipA.mp4"}
    assert text_part["text"] == (
        "Which invented shape moves first?\n(A) Circle\n(B) Square\n(C) Triangle\n(D) Star\n"
        "Answer with the option's letter from the given choices directly."
    )
    assert rows[1]["annotation_warnings"] == ["duplicate_option_text"]
    assert rows[2]["verifier_metadata"]["video_sha256"] == hashlib.sha256(b"clipB").hexdigest()

    cohort = AccuracyCohort.model_validate_json(lvbench_st.COHORT_FPATH.read_text(encoding="utf-8"))
    assert (cohort.benchmark, cohort.setting, cohort.answer_policy, cohort.dataset_revision) == (
        "lvbench",
        "single_turn",
        "llm_extraction_v1",
        lvbench_st.DATASET_REVISION,
    )
    assert cohort.task_subsets["lvbench:101"] == {
        "domain": ("documentary",),
        "task_type": ("entity recognition", "reasoning"),
    }

    lvbench_st.prepare(source_dir=str(source), video_url_dir="/mnt/lvbench/video")
    custom_video_part = read_rows(lvbench_st)[2]["responses_create_params"]["input"][0]["content"][0]
    assert custom_video_part == {"type": "input_video", "video_url": "file:///mnt/lvbench/video/clipB.mp4"}

    with pytest.raises(ValueError, match="lvbench_source_dir"):
        lvbench_st.prepare()
    with pytest.raises(ValueError, match="absolute path"):
        lvbench_st.prepare(source_dir=str(source), video_url_dir="relative/video")
    (source / "FILES.sha256").write_text(f"{'0' * 64}  video/clipA.mp4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no SHA-256 for video/clipB.mp4"):
        lvbench_st.prepare(source_dir=str(source))
    monkeypatch.setattr(lvbench_st, "NUM_QUESTIONS", 99)
    with pytest.raises(ValueError, match="expected 99"):
        lvbench_st.prepare(source_dir=str(source))
    monkeypatch.setattr(lvbench_st, "ANNOTATIONS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        lvbench_st.prepare(source_dir=str(source))


def test_lvbench_agentic_prepare_and_errors(tmp_path, monkeypatch) -> None:
    records = [
        lvbench_video_record("clipA", [lvbench_qa("101", answer="C")]),
        lvbench_video_record("clipB", [lvbench_qa("201", answer="D", question_type=["event understanding"])]),
    ]
    source = stage_lvbench(tmp_path, monkeypatch, lvbench_ag, records, ["clipA", "clipB"])

    assert lvbench_ag.prepare(source_dir=str(source), video_dir="/data/lvbench/video") == lvbench_ag.OUTPUT_FPATH
    rows = read_rows(lvbench_ag)
    assert [row["task_id"] for row in rows] == ["lvbench:101", "lvbench:201"]
    assert_rows_valid("lvbench", rows)
    for row in rows:
        assert "expected_answer" not in row["responses_create_params"]

    (part,) = rows[1]["responses_create_params"]["input"][0]["content"]
    assert part["text"] == (
        "Video: /testbed/videos/clipB.mp4\nWhich invented shape moves first?\n"
        "(A) Circle\n(B) Square\n(C) Triangle\n(D) Star\n"
        "Answer with the option's letter from the given choices directly."
    )
    assert rows[1]["video_binding"] == {
        "source": "/data/lvbench/video/clipB.mp4",
        "sandbox_path": "videos/clipB.mp4",
        "sha256": hashlib.sha256(b"clipB").hexdigest(),
    }

    cohort = AccuracyCohort.model_validate_json(lvbench_ag.COHORT_FPATH.read_text(encoding="utf-8"))
    assert (cohort.benchmark, cohort.setting, cohort.answer_policy, cohort.dataset_revision) == (
        "lvbench",
        "agentic",
        "llm_extraction_v1",
        lvbench_st.DATASET_REVISION,
    )

    with pytest.raises(ValueError, match="lvbench_agentic_source_dir"):
        lvbench_ag.prepare()
    monkeypatch.chdir(tmp_path)
    lvbench_ag.prepare(source_dir="source")
    rel_rows = read_rows(lvbench_ag)
    assert rel_rows[1]["video_binding"]["source"] == str((source / "video" / "clipB.mp4").resolve())
    (source / "FILES.sha256").write_text(f"{'0' * 64}  video/clipA.mp4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no SHA-256 for video/clipB.mp4"):
        lvbench_ag.prepare(source_dir=str(source))


@pytest.mark.parametrize(
    ("special_dir", "encoded_segment"),
    [
        ("video#archive", "video%23archive"),
        ("video?archive", "video%3Farchive"),
        ("video%2Farchive", "video%252Farchive"),
    ],
)
def test_single_turn_file_urls_encode_special_path_characters(
    tmp_path, monkeypatch, special_dir: str, encoded_segment: str
) -> None:
    url_dir = tmp_path / special_dir
    url_dir.mkdir(parents=True, exist_ok=True)

    v1_records = [v1_record("clipA", 1)]
    v1_source = stage_parquet(tmp_path / "v1", monkeypatch, v1_st, v1_records, ["clipA"], "VIDEOMME_V1_PARQUET")
    v1_st.prepare(source_dir=str(v1_source), video_url_dir=str(url_dir))
    v1_url = read_rows(v1_st)[0]["responses_create_params"]["input"][0]["content"][0]["video_url"]
    assert encoded_segment in v1_url
    assert unquote(urlparse(v1_url).path) == str(url_dir / "clipA.mp4")

    lv_records = [lvbench_video_record("clipA", [lvbench_qa("101", answer="B")])]
    lv_source = stage_lvbench(tmp_path / "lv", monkeypatch, lvbench_st, lv_records, ["clipA"])
    lvbench_st.prepare(source_dir=str(lv_source), video_url_dir=str(url_dir))
    lv_url = read_rows(lvbench_st)[0]["responses_create_params"]["input"][0]["content"][0]["video_url"]
    assert encoded_segment in lv_url
    assert unquote(urlparse(lv_url).path) == str(url_dir / "clipA.mp4")

    v2_records = [v2_record("001", position, 1) for position in range(1, 5)]
    v2_source = stage_parquet(tmp_path / "v2", monkeypatch, v2_st, v2_records, ["001"], "VIDEOMME_V2_PARQUET")
    v2_st.prepare(source_dir=str(v2_source), video_url_dir=str(url_dir))
    v2_url = read_rows(v2_st)[0]["responses_create_params"]["input"][0]["content"][0]["video_url"]
    assert encoded_segment in v2_url
    assert unquote(urlparse(v2_url).path) == str(url_dir / "001.mp4")


@pytest.mark.parametrize(
    (
        "benchmark_name",
        "expected_profile",
        "expected_agent",
        "expected_agent_server",
        "expected_resource",
        "expected_metric",
    ),
    [
        (
            "videomme_v1",
            "custom-gym-verifier",
            "videomme_v1_single_turn",
            "simple_agent",
            "videomme_v1",
            "mean/Video_MME",
        ),
        (
            "videomme_v2",
            "custom-gym-verifier",
            "videomme_v2_single_turn",
            "simple_agent",
            "videomme_v2",
            "Video_MME_v2/nonlinear_score",
        ),
        ("lvbench", "custom-gym-verifier", "lvbench_single_turn", "simple_agent", "lvbench", "mean/LVBench"),
        (
            "lvbench_agentic",
            "custom-gym-agent-loop",
            "lvbench_agentic_opencode",
            "opencode_agent",
            "lvbench",
            "mean/LVBench_agentic",
        ),
    ],
)
def test_video_benchmark_manifests_validate_cleanly(
    benchmark_name: str,
    expected_profile: str,
    expected_agent: str,
    expected_agent_server: str,
    expected_resource: str,
    expected_metric: str,
) -> None:
    manifest_path = REPO_ROOT / "benchmarks" / benchmark_name / "manifest.yaml"
    config_path = REPO_ROOT / "benchmarks" / benchmark_name / "config.yaml"
    manifest = load_manifest(manifest_path)
    assert manifest.resources_server == expected_resource
    assert manifest.agent_server == expected_agent_server
    assert manifest.maintainers == ["cg1177"]
    assert manifest.capabilities
    assert manifest.domains == ["general"]
    assert manifest.primary_metric == expected_metric
    assert manifest.requirements is not None
    assert manifest.requirements.gpu is False
    assert manifest.requirements.network == "allowlist"

    report = validate_environment(manifest_path, validate_datasets=False)
    assert report.warnings == ()
    assert report.declared_profile == expected_profile
    assert report.inferred_profile == expected_profile

    bc = BenchmarkConfig.from_config_path(config_path, strict=False)
    assert bc is not None
    assert bc.name == benchmark_name
    assert bc.agent_name == expected_agent
    assert bc.dataset.name == benchmark_name
    assert bc.dataset.prepare_script == Path(f"benchmarks/{benchmark_name}/prepare.py")


@pytest.mark.parametrize("benchmark_name", ["videomme_v1", "videomme_v2", "lvbench"])
def test_single_turn_benchmark_configs_resolve_and_validate(tmp_path, monkeypatch, benchmark_name: str) -> None:
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)

    bench_cfg = OmegaConf.load(REPO_ROOT / "benchmarks" / benchmark_name / "config.yaml")
    included = [OmegaConf.load(REPO_ROOT / path) for path in bench_cfg.config_paths]
    overrides = OmegaConf.create(
        {
            f"{benchmark_name}_source_dir": str(tmp_path / "src"),
            f"{benchmark_name}_cohort_manifest": str(tmp_path / "cohort.json"),
        }
    )
    merged = OmegaConf.to_container(OmegaConf.merge(*included, bench_cfg, overrides), resolve=True)

    config_type = (
        VideoMMEV2ResourcesServerConfig if benchmark_name == "videomme_v2" else VideoMCQAResourcesServerConfig
    )
    resource = config_type(
        name=benchmark_name, host="127.0.0.1", port=8001, **merged[benchmark_name]["resources_servers"][benchmark_name]
    )
    assert resource.judge_model_server.name == "video_mcqa_answer_judge"
    assert resource.judge_chat_completion_create_params.reasoning_effort == "max"
    resource.check_answer_judge("llm_extraction_v1")

    judge = merged["video_mcqa_answer_judge"]["responses_api_models"]["openai_model"]
    assert judge["openai_base_url"] == "https://inference-api.nvidia.com/v1"
    assert judge["openai_model"] == "openai/openai/gpt-6.1-sol"

    agent_name = f"{benchmark_name}_single_turn"
    agent = SimpleAgentConfig(
        name=agent_name, host="127.0.0.1", port=8002, **merged[agent_name]["responses_api_agents"]["simple_agent"]
    )
    assert agent.resources_server.name == benchmark_name and agent.max_steps == 1
    assert merged["prepare_script_args"] == {"source_dir": str(tmp_path / "src"), "video_url_dir": None}

    environment = SingleAgentTurnLegacyEnvironmentServerConfig(
        name=f"{agent_name}_environment",
        host="127.0.0.1",
        port=8003,
        **merged[f"{agent_name}_environment"]["environment_servers"]["single_agent_turn_legacy"],
    )
    assert environment.agent_server.name == agent_name and environment.resources_server.name == benchmark_name


def test_lvbench_agentic_benchmark_config_resolves_and_validates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VIDEO_MCQA_SANDBOX_IMAGE", "registry.example/video-mcqa-sandbox:test")
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)

    bench_cfg = OmegaConf.load(REPO_ROOT / "benchmarks" / "lvbench_agentic" / "config.yaml")
    included = [OmegaConf.load(REPO_ROOT / path) for path in bench_cfg.config_paths]
    source_dir = tmp_path / "lvbench_src"
    overrides = OmegaConf.create(
        {
            "lvbench_agentic_source_dir": str(source_dir),
            "lvbench_agentic_cohort_manifest": str(tmp_path / "cohort.json"),
            "policy_model_name": "policy",
        }
    )
    merged = OmegaConf.to_container(OmegaConf.merge(*included, bench_cfg, overrides), resolve=True)

    resource = VideoMCQAResourcesServerConfig(
        name="lvbench_agentic", port=8001, **merged["lvbench_agentic"]["resources_servers"]["lvbench"]
    )
    assert resource.sandbox_provider == "sandbox" and resource.sandbox_network_access == "model_only"
    assert resource.sandbox_config.image == "registry.example/video-mcqa-sandbox:test"
    assert resource.video_roots == [source_dir / "video"]
    assert resource.judge_model_server.name == "video_mcqa_answer_judge"
    assert resource.judge_chat_completion_create_params.reasoning_effort == "max"
    resource.check_answer_judge("llm_extraction_v1")

    judge = merged["video_mcqa_answer_judge"]["responses_api_models"]["openai_model"]
    assert judge["host"] == "127.0.0.1"
    assert judge["openai_base_url"] == "https://inference-api.nvidia.com/v1"
    assert judge["openai_model"] == "openai/openai/gpt-6.1-sol"

    agent = OpenCodeAgentConfig(
        name="lvbench_agentic_opencode",
        port=8002,
        **merged["lvbench_agentic_opencode"]["responses_api_agents"]["opencode_agent"],
    )
    assert agent.resources_server.name == "lvbench_agentic"
    assert merged["prepare_script_args"] == {
        "source_dir": str(source_dir),
        "video_dir": str(source_dir / "video"),
    }

    environment = SingleAgentTurnLegacyEnvironmentServerConfig(
        name="lvbench_agentic_environment",
        port=8003,
        **merged["lvbench_agentic_environment"]["environment_servers"]["single_agent_turn_legacy"],
    )
    assert environment.agent_server.name == "lvbench_agentic_opencode"
    assert environment.resources_server.name == "lvbench_agentic"
