# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from nemo_gym.client import ServerClient
from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, AggregateMetricsRequest
from nemo_gym.data import TaskDataValidator, load_task_data_schema
from nemo_gym.environment import load_manifest
from resources_servers.cgbench import dataset as cg_dataset
from resources_servers.cgbench import prepare as cg_prepare
from resources_servers.cgbench.app import CGBenchResourcesServer
from resources_servers.cgbench.dataset import (
    CATEGORIES,
    DATASET_REVISION,
    DOMAINS,
    DURATION_BUCKETS,
    SLICES,
    SUB_CATEGORIES,
    CGBenchCohort,
    CGBenchSlice,
    duration_bucket,
    load_cgbench,
)
from resources_servers.cgbench.prepare import (
    load_annotations_file,
    prepare_cgbench_benchmark,
    prepare_cgbench_dataset,
)
from resources_servers.video_mcqa.app import VideoMCQAResourcesServerConfig, VideoMCQAVerifyRequest
from resources_servers.video_mcqa.dataset import VideoSource
from resources_servers.video_mcqa.tests.helpers import response


REPO_ROOT = Path(__file__).resolve().parents[3]


def _validate_prepared_rows(rows: list[dict[str, object]]) -> None:
    schema = load_task_data_schema(REPO_ROOT / "resources_servers" / "cgbench")
    validator = TaskDataValidator("cgbench", schema, "rows.jsonl")
    for index, row in enumerate(rows):
        validator.validate_row(index, row)
    assert validator.report.error_rows == 0, validator.report.error_samples


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (1, "0-10m"),
        (600, "0-10m"),
        (601, "10-20m"),
        (1200, "10-20m"),
        (1201, "20-30m"),
        (1800, "20-30m"),
        (2400, "30-40m"),
        (3000, "40-50m"),
        (3600, "50-60m"),
        (3601, "60m+"),
        (6325, "60m+"),
    ],
)
def test_duration_bucket_boundaries(seconds: int, expected: str) -> None:
    assert duration_bucket(seconds) == expected


@pytest.mark.parametrize("bad", [0, -10, True, "600", float("nan"), float("inf")])
def test_duration_bucket_rejects_invalid_values(bad: object) -> None:
    with pytest.raises(ValueError, match="duration"):
        duration_bucket(bad)


def test_load_cgbench_preserves_literal_options_allows_duplicates_and_hides_clue_intervals() -> None:
    records = [
        {
            "qid": 3550,
            "video_uid": "vid_literal",
            "question": "Which brand is visible on the storefront?",
            "choices": ["J. CREW", "S. R. Nathan", "Target", "Gap", "Target"],
            "answer": "J. CREW",
            "right_answer": "A",
            "clue_intervals": [[50, 90]],
            "domain": "Life Record",
            "sub_category": "Text Perception",
            "duration": 900,
        }
    ]
    media = {
        "vid_literal": VideoSource(
            source_url="cg_videos_720p/vid_literal.mp4",
            sha256="b" * 64,
            video_url="https://example.invalid/vid_literal.mp4",
        )
    }
    items = load_cgbench(records, benchmark="cgbench", dataset_revision=DATASET_REVISION, media=media)
    assert len(items) == 1
    item = items[0]
    assert item.question.options[0] == "J. CREW"
    assert item.question.options[1] == "S. R. Nathan"
    assert item.annotation_warnings == ("duplicate_option_text",)
    assert item.subsets == {
        "category": ("Perception",),
        "sub_category": ("Text Perception",),
        "domain": ("Life Record",),
        "duration": ("10-20m",),
    }

    rows, _ = prepare_cgbench_dataset(
        records, benchmark="cgbench", dataset_revision=DATASET_REVISION, media=media, setting="single_turn"
    )
    dumped = json.dumps(rows[0])
    assert "clue_intervals" not in dumped
    assert "J. CREW" in dumped
    assert "S. R. Nathan" in dumped
    _validate_prepared_rows(rows)


def test_load_cgbench_rejects_inconsistent_or_invalid_records() -> None:
    media = {"v1": VideoSource(source_url="cg_videos_720p/v1.mp4", sha256="c" * 64, video_url="file:///v1.mp4")}
    base = {
        "qid": 1,
        "video_uid": "v1",
        "question": "What happens?",
        "choices": ["One", "Two", "Three", "Four", "Five"],
        "answer": "Five",
        "right_answer": "E",
        "domain": "Driving",
        "sub_category": "Entity Perception",
        "duration": 1000,
    }
    with pytest.raises(ValueError, match="disagrees"):
        load_cgbench([{**base, "answer": "One"}], benchmark="cgbench", dataset_revision=DATASET_REVISION, media=media)
    with pytest.raises(ValueError, match="outside A-E"):
        load_cgbench(
            [{**base, "right_answer": "F"}], benchmark="cgbench", dataset_revision=DATASET_REVISION, media=media
        )
    with pytest.raises(ValueError, match="at most 8"):
        load_cgbench(
            [{**base, "choices": [f"C{i}" for i in range(9)], "answer": "C4"}],
            benchmark="cgbench",
            dataset_revision=DATASET_REVISION,
            media=media,
        )
    with pytest.raises(ValueError, match="sub_category"):
        load_cgbench(
            [{**base, "sub_category": "Unknown"}], benchmark="cgbench", dataset_revision=DATASET_REVISION, media=media
        )
    with pytest.raises(ValueError, match="domain"):
        load_cgbench(
            [{**base, "domain": "Unknown"}], benchmark="cgbench", dataset_revision=DATASET_REVISION, media=media
        )
    with pytest.raises(ValueError, match="Invalid CG-Bench video_uid"):
        load_cgbench(
            [{**base, "video_uid": "../escape"}], benchmark="cgbench", dataset_revision=DATASET_REVISION, media=media
        )


def test_prepare_cgbench_benchmark_offline_with_checksums_and_local_mp4(tmp_path, monkeypatch) -> None:
    records = [
        {
            "qid": 7,
            "video_uid": "demo_vid",
            "question": "Which action takes place?",
            "choices": ["A1", "B2", "C3", "D4", "E5", "F6"],
            "answer": "F6",
            "right_answer": "F",
            "domain": "GUI",
            "sub_category": "Entity Cognition",
            "duration": 1500,
        }
    ]
    source_dir = tmp_path / "src"
    video_dir = source_dir / "cg_videos_720p"
    video_dir.mkdir(parents=True)
    mp4_bytes = b"synthetic-mp4-bytes"
    mp4_path = video_dir / "demo_vid.mp4"
    mp4_path.write_bytes(mp4_bytes)
    mp4_sha = hashlib.sha256(mp4_bytes).hexdigest()

    json_bytes = json.dumps(records).encode("utf-8")
    json_path = source_dir / "cgbench_hard.json"
    json_path.write_bytes(json_bytes)
    json_sha = hashlib.sha256(json_bytes).hexdigest()

    patched_slice = CGBenchSlice(
        benchmark="cgbench_hard",
        json_filename="cgbench_hard.json",
        json_sha256=json_sha,
        num_questions=1,
        num_videos=1,
        metric_name="CG_Bench_hard",
        json_env_var="CGBENCH_HARD_JSON",
    )
    patched_slices = {**SLICES, "cgbench_hard": patched_slice}
    monkeypatch.setattr(cg_dataset, "SLICES", patched_slices)
    monkeypatch.setattr(cg_prepare, "SLICES", patched_slices)

    # 1. Without FILES.sha256 -> hashes local mp4; single_turn
    out_st = tmp_path / "out_st.jsonl"
    cohort_st = tmp_path / "cohort_st.json"
    prepare_cgbench_benchmark(
        benchmark="cgbench_hard",
        setting="single_turn",
        output_fpath=out_st,
        cohort_fpath=cohort_st,
        source_dir=str(source_dir),
    )
    st_row = json.loads(out_st.read_text(encoding="utf-8").strip())
    assert st_row["verifier_metadata"]["video_sha256"] == mp4_sha
    assert st_row["setting"] == "single_turn"
    _validate_prepared_rows([st_row])
    cohort = CGBenchCohort.model_validate_json(cohort_st.read_text(encoding="utf-8"))
    assert cohort.benchmark == "cgbench_hard" and cohort.task_ids == ("cgbench_hard:7",)

    # 2. With FILES.sha256 -> avoids mp4 read; agentic + boxed prompt
    (source_dir / "FILES.sha256").write_text(f"{mp4_sha}  cg_videos_720p/demo_vid.mp4\n", encoding="utf-8")
    mp4_path.unlink()
    out_ag = tmp_path / "out_ag.jsonl"
    cohort_ag = tmp_path / "cohort_ag.json"
    prepare_cgbench_benchmark(
        benchmark="cgbench_hard",
        setting="agentic",
        output_fpath=out_ag,
        cohort_fpath=cohort_ag,
        source_dir=str(source_dir),
        answer_policy="boxed_only_v1",
        prompt="boxed",
    )
    ag_row = json.loads(out_ag.read_text(encoding="utf-8").strip())
    assert ag_row["setting"] == "agentic"
    assert ag_row["video_binding"]["sandbox_path"] == "videos/demo_vid.mp4"
    assert ag_row["video_binding"]["sha256"] == mp4_sha
    assert r"\boxed{...}" in ag_row["responses_create_params"]["input"][0]["content"][0]["text"]
    _validate_prepared_rows([ag_row])

    # 3. Corrupted JSON SHA-256 fails loudly
    json_path.write_bytes(json_bytes + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_cgbench_benchmark(
            benchmark="cgbench_hard",
            setting="single_turn",
            output_fpath=out_st,
            cohort_fpath=cohort_st,
            source_dir=str(source_dir),
        )


def test_server_configs_and_six_benchmark_entries() -> None:
    configs_dir = REPO_ROOT / "resources_servers" / "cgbench" / "configs"
    expected_configs = {
        "cgbench_single_turn.yaml",
        "cgbench_single_turn_llm_extraction.yaml",
        "cgbench_agentic.yaml",
        "cgbench_agentic_llm_extraction.yaml",
    }
    assert {path.name for path in configs_dir.glob("*.yaml")} == expected_configs
    for filename in expected_configs:
        data = yaml.safe_load((configs_dir / filename).read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        if "llm_extraction" not in filename:
            server_key = "cgbench" if "single_turn" in filename else "cgbench_agentic"
            assert data[server_key]["resources_servers"]["cgbench"]["verified"] is False
            if "agentic" in filename:
                assert (
                    data[server_key]["resources_servers"]["cgbench"]["sandbox_config"]["image"]
                    == "${oc.env:VIDEO_MCQA_SANDBOX_IMAGE,null}"
                )
        else:
            server_key = "cgbench" if "single_turn" in filename else "cgbench_agentic"
            assert data[server_key]["resources_servers"]["cgbench"]["verified"] is False
            judge = data["video_mcqa_answer_judge"]["responses_api_models"]["openai_model"]
            assert judge["openai_base_url"] == "${judge_base_url}"
            if "agentic" in filename:
                assert judge["host"] == "127.0.0.1"

    expected_benchmarks = [
        ("cgbench", "simple_agent", "mean/CG_Bench", "no-network"),
        ("cgbench_agentic", "opencode_agent", "mean/CG_Bench_agentic", "allowlist"),
        ("cgbench_mini", "simple_agent", "mean/CG_Bench_mini", "no-network"),
        ("cgbench_mini_agentic", "opencode_agent", "mean/CG_Bench_mini_agentic", "allowlist"),
        ("cgbench_hard", "simple_agent", "mean/CG_Bench_hard", "no-network"),
        ("cgbench_hard_agentic", "opencode_agent", "mean/CG_Bench_hard_agentic", "allowlist"),
    ]
    for name, agent, primary_metric, network in expected_benchmarks:
        bench_dir = REPO_ROOT / "benchmarks" / name
        manifest = yaml.safe_load((bench_dir / "manifest.yaml").read_text(encoding="utf-8"))
        assert manifest["name"] == name
        assert manifest["licensing"] == "MIT"
        assert manifest["resources_server"] == "cgbench"
        assert manifest["agent_server"] == agent
        assert manifest["modality"] == "video"
        parsed = load_manifest(bench_dir / "manifest.yaml")
        assert parsed.maintainers == ["cg1177"]
        assert parsed.primary_metric == primary_metric
        assert parsed.domains == ["general"]
        assert parsed.requirements is not None
        assert parsed.requirements.gpu is False
        assert parsed.requirements.network == network
        assert "multimodal_input_understanding/temporal_media_understanding" in parsed.capabilities
        if agent == "opencode_agent":
            assert "tool_and_computer_use/multi_step_tool_use" in parsed.capabilities
        assert (bench_dir / "config.yaml").is_file()
        assert (bench_dir / "README.md").is_file()
        assert (bench_dir / "data" / ".gitignore").is_file()
        mod = importlib.import_module(f"benchmarks.{name}.prepare")
        assert callable(mod.prepare)
        assert not list((bench_dir / "data").glob("*.jsonl"))


def test_format_video_url_percent_encodes_special_local_paths(tmp_path) -> None:
    special_dir = tmp_path / "video#archive" / "q?part" / "100% ready"
    url = cg_prepare._format_video_url(str(special_dir), "clip.mp4")
    assert url == (special_dir / "clip.mp4").as_uri()
    assert "%23" in url and "%3F" in url and "%2520" not in url and "%20" in url


def test_relative_agentic_prepare_from_non_repo_cwd_binds_and_validates_video_root(tmp_path, monkeypatch) -> None:
    records = [
        {
            "qid": 11,
            "video_uid": "rel_vid",
            "question": "Which synthetic sign is shown?",
            "choices": ["A1", "B2", "C3", "D4", "E5"],
            "answer": "E5",
            "right_answer": "E",
            "domain": "GUI",
            "sub_category": "Entity Cognition",
            "duration": 900,
        }
    ]
    caller_dir = tmp_path / "caller_workdir"
    source_dir = caller_dir / "rel_src"
    video_dir = source_dir / "cg_videos_720p"
    video_dir.mkdir(parents=True)
    mp4_bytes = b"relative-agentic-video"
    mp4_path = video_dir / "rel_vid.mp4"
    mp4_path.write_bytes(mp4_bytes)
    mp4_sha = hashlib.sha256(mp4_bytes).hexdigest()

    json_bytes = json.dumps(records).encode("utf-8")
    (source_dir / "cgbench_hard.json").write_bytes(json_bytes)
    json_sha = hashlib.sha256(json_bytes).hexdigest()
    patched_slice = CGBenchSlice(
        benchmark="cgbench_hard",
        json_filename="cgbench_hard.json",
        json_sha256=json_sha,
        num_questions=1,
        num_videos=1,
        metric_name="CG_Bench_hard",
        json_env_var="CGBENCH_HARD_JSON",
    )
    patched_slices = {**SLICES, "cgbench_hard": patched_slice}
    monkeypatch.setattr(cg_dataset, "SLICES", patched_slices)
    monkeypatch.setattr(cg_prepare, "SLICES", patched_slices)

    monkeypatch.chdir(caller_dir)
    out_rel = tmp_path / "rel.jsonl"
    cohort_rel = tmp_path / "rel_cohort.json"
    prepare_cgbench_benchmark(
        benchmark="cgbench_hard",
        setting="agentic",
        output_fpath=out_rel,
        cohort_fpath=cohort_rel,
        source_dir="rel_src",
        video_dir="rel_src/cg_videos_720p",
        num_repeats=2,
    )

    out_abs = tmp_path / "abs.jsonl"
    cohort_abs = tmp_path / "abs_cohort.json"
    prepare_cgbench_benchmark(
        benchmark="cgbench_hard",
        setting="agentic",
        output_fpath=out_abs,
        cohort_fpath=cohort_abs,
        source_dir=str(source_dir),
        video_dir=str(video_dir),
        num_repeats=2,
    )

    rel_row = json.loads(out_rel.read_text(encoding="utf-8").strip())
    abs_row = json.loads(out_abs.read_text(encoding="utf-8").strip())
    assert rel_row["video_binding"] == abs_row["video_binding"]
    assert Path(rel_row["video_binding"]["source"]).is_absolute()
    assert Path(rel_row["video_binding"]["source"]) == mp4_path.resolve()
    assert CGBenchCohort.model_validate_json(cohort_rel.read_text(encoding="utf-8")).num_repeats == 2

    # Switch back to REPO_ROOT (different cwd) to verify seed-time binding resolution succeeds against video_roots.
    monkeypatch.chdir(REPO_ROOT)
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"num_repeats": 2}
    server = CGBenchResourcesServer(
        config=VideoMCQAResourcesServerConfig(
            name="cgbench",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            cohort_manifest=cohort_rel,
            sandbox_provider="opensandbox",
            sandbox_config={"image": "registry.example/video-mcqa-sandbox:test"},
            video_roots=[video_dir],
        ),
        server_client=client,
    )
    binding, resolved_source = server._binding(rel_row)
    assert binding.sandbox_path == "videos/rel_vid.mp4"
    assert resolved_source == mp4_path.resolve()
    assert hashlib.sha256(resolved_source.read_bytes()).hexdigest() == mp4_sha


@pytest.mark.parametrize(
    "module_name,slice_name",
    [
        ("benchmarks.cgbench.prepare", "cgbench"),
        ("benchmarks.cgbench_agentic.prepare", "cgbench"),
        ("benchmarks.cgbench_mini.prepare", "cgbench_mini"),
        ("benchmarks.cgbench_mini_agentic.prepare", "cgbench_mini"),
        ("benchmarks.cgbench_hard.prepare", "cgbench_hard"),
        ("benchmarks.cgbench_hard_agentic.prepare", "cgbench_hard"),
    ],
)
def test_public_wrappers_propagate_num_repeats(tmp_path, monkeypatch, module_name: str, slice_name: str) -> None:
    records = [
        {
            "qid": 21,
            "video_uid": "wrap_vid",
            "question": "Which synthetic frame appears?",
            "choices": ["A1", "B2", "C3", "D4", "E5"],
            "answer": "C3",
            "right_answer": "C",
            "domain": "News",
            "sub_category": "Entity Perception",
            "duration": 600,
        }
    ]
    source_dir = tmp_path / "src"
    video_dir = source_dir / "cg_videos_720p"
    video_dir.mkdir(parents=True)
    mp4_path = video_dir / "wrap_vid.mp4"
    mp4_path.write_bytes(b"wrapper-video")
    spec = SLICES[slice_name]
    json_bytes = json.dumps(records).encode("utf-8")
    (source_dir / spec.json_filename).write_bytes(json_bytes)
    patched_slice = CGBenchSlice(
        benchmark=slice_name,  # type: ignore[arg-type]
        json_filename=spec.json_filename,
        json_sha256=hashlib.sha256(json_bytes).hexdigest(),
        num_questions=1,
        num_videos=1,
        metric_name=spec.metric_name,
        json_env_var=spec.json_env_var,
    )
    patched_slices = {**SLICES, slice_name: patched_slice}
    monkeypatch.setattr(cg_dataset, "SLICES", patched_slices)
    monkeypatch.setattr(cg_prepare, "SLICES", patched_slices)

    mod = importlib.import_module(module_name)
    out_path = tmp_path / f"{module_name}.jsonl"
    cohort_path = tmp_path / f"{module_name}_cohort.json"
    monkeypatch.setattr(mod, "OUTPUT_FPATH", out_path)
    monkeypatch.setattr(mod, "COHORT_FPATH", cohort_path)

    mod.prepare(source_dir=str(source_dir), num_repeats=2)
    cohort = CGBenchCohort.model_validate_json(cohort_path.read_text(encoding="utf-8"))
    assert cohort.num_repeats == 2


@pytest.mark.skipif(
    not os.environ.get("CGBENCH_REAL_JSON_DIR"),
    reason="Set CGBENCH_REAL_JSON_DIR to run full verification against real CG-Bench JSON slices",
)
async def test_real_cgbench_json_slices_and_sampled_prepare(tmp_path) -> None:
    json_dir = Path(os.environ["CGBENCH_REAL_JSON_DIR"])
    for benchmark, slice_spec in SLICES.items():
        records = load_annotations_file(json_dir / slice_spec.json_filename, slice_spec=slice_spec)
        unique_uids = list(dict.fromkeys(str(record["video_uid"]) for record in records))
        assert len(unique_uids) == slice_spec.num_videos

        # Full dataset validation for every row in the real slice.
        full_media = {
            uid: VideoSource(
                source_url=f"cg_videos_720p/{uid}.mp4",
                sha256=hashlib.sha256(uid.encode("utf-8")).hexdigest(),
                video_url=f"https://example.invalid/cg_videos_720p/{uid}.mp4",
                source=f"/data/cg_videos_720p/{uid}.mp4",
                sandbox_path=f"videos/{uid}.mp4",
            )
            for uid in unique_uids
        }
        items = load_cgbench(records, benchmark=benchmark, dataset_revision=DATASET_REVISION, media=full_media)
        assert len(items) == slice_spec.num_questions
        assert {len(item.question.options) for item in items} == {5, 6, 7, 8}
        assert {item.subsets["category"][0] for item in items} == set(CATEGORIES)
        assert {item.subsets["sub_category"][0] for item in items} == set(SUB_CATEGORIES)
        assert {item.subsets["domain"][0] for item in items} == set(DOMAINS)
        assert {item.subsets["duration"][0] for item in items} == set(DURATION_BUCKETS)

        # Stratified sample covering every option count (5..8), category, and duration bucket.
        seen_keys: set[tuple[str, str]] = set()
        sampled: list[dict[str, object]] = []
        for record in records:
            keys = [
                ("opt", str(len(record["choices"]))),  # type: ignore[arg-type]
                ("sub", str(record["sub_category"])),
                ("dom", str(record["domain"])),
                ("dur", duration_bucket(record["duration"])),
            ]
            if any(key not in seen_keys for key in keys):
                seen_keys.update(keys)
                sampled.append(record)
        assert len(sampled) >= 14

        sample_media = {str(r["video_uid"]): full_media[str(r["video_uid"])] for r in sampled}
        for setting in ("single_turn", "agentic"):
            rows, cohort = prepare_cgbench_dataset(
                sampled,
                benchmark=benchmark,
                dataset_revision=DATASET_REVISION,
                media=sample_media,
                setting=setting,  # type: ignore[arg-type]
            )
            assert len(rows) == len(sampled)
            _validate_prepared_rows(rows)
            for row in rows:
                assert "clue_intervals" not in json.dumps(row)

            manifest = tmp_path / f"real-{benchmark}-{setting}.json"
            manifest.write_text(cohort.model_dump_json(), encoding="utf-8")
            client = MagicMock(spec=ServerClient)
            client.global_config_dict = {}
            server = CGBenchResourcesServer(
                config=VideoMCQAResourcesServerConfig(
                    name="cgbench", host="127.0.0.1", port=8000, entrypoint="app.py", cohort_manifest=manifest
                ),
                server_client=client,
            )
            verify_responses = []
            for idx, row in enumerate(rows):
                expected = row["verifier_metadata"]["expected_answer"]
                reply = f'```json\n{{"result": "{expected}"}}\n```'
                req = VideoMCQAVerifyRequest.model_validate({**row, "response": response(reply)})
                verified = (await server.verify(req)).model_dump()
                assert verified["outcome"] == "correct"
                verified[TASK_INDEX_KEY_NAME] = idx
                verified[ROLLOUT_INDEX_KEY_NAME] = 0
                verify_responses.append(verified)
            agg = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=verify_responses))
            base = slice_spec.metric_name + ("_agentic" if setting == "agentic" else "")
            assert agg.agent_metrics[f"{base}/raw_acc"] == pytest.approx(100.0)
            assert agg.agent_metrics[f"{base}/is_complete"] is True

        # Verify full-slice cohort preparation and fast linear-time subset aggregation on all 12,129 rows.
        if benchmark == "cgbench":
            full_rows, full_cohort = prepare_cgbench_dataset(
                records,
                benchmark=benchmark,
                dataset_revision=DATASET_REVISION,
                media=full_media,
                setting="single_turn",
            )
            assert len(full_rows) == 12129
            assert len(full_cohort.task_ids) == 12129
            # Verify all 36 official breakdown slices sum to 12,129 per axis.
            for axis, expected_labels in (
                ("category", set(CATEGORIES)),
                ("sub_category", set(SUB_CATEGORIES)),
                ("domain", set(DOMAINS)),
                ("duration", set(DURATION_BUCKETS)),
            ):
                axis_counts: dict[str, int] = {}
                for subsets in full_cohort.task_subsets.values():
                    label = subsets[axis][0]
                    axis_counts[label] = axis_counts.get(label, 0) + 1
                assert set(axis_counts) == expected_labels
                assert sum(axis_counts.values()) == 12129
