# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Video-Index dataset loading, benchmark prepare scripts, task_data schema, configs, and manifests."""

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.config import ModelServerRef
from nemo_gym.data import TaskDataValidator, load_task_data_schema
from nemo_gym.environment import load_manifest
from nemo_gym.environment.validation import validate_environment
from resources_servers.video_mcqa.app import VideoMCQAResourcesServerConfig
from resources_servers.video_mcqa.dataset import VideoSource
from resources_servers.video_mcqa.sandbox import PLUGIN_PATH, SANDBOX_IMAGE_DIR
from resources_servers.videoindex import prepare as vi_prepare
from resources_servers.videoindex.dataset import CAPABILITY_GROUPS, RELEASES, load_videoindex
from resources_servers.videoindex.prepare import prepare_videoindex_benchmark, prepare_videoindex_dataset
from resources_servers.videoindex.scoring import LETTERS, VIDEOINDEX_RULE_POLICY
from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig
from responses_api_agents.simple_agent.app import SimpleAgentConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_DIR = REPO_ROOT / "resources_servers" / "videoindex"
BENCHMARKS = (
    "videoindex_v1_0",
    "videoindex_v1_0_agentic",
    "videoindex_v1_1",
    "videoindex_v1_1_agentic",
)


def make_item(index: int, *, group: str | None = None, option_count: int = 4, answer_idx: int = 1, **extra) -> dict:
    group = group or CAPABILITY_GROUPS[index % len(CAPABILITY_GROUPS)]
    return {
        "item_id": f"synth_{index}",
        "benchmark": "SyntheticBench",
        "capability_group": group,
        "question": f"What synthetic event happens in clip {index}?",
        "options": [f"Choice {i}" for i in range(option_count)],
        "answer": LETTERS[answer_idx],
        "answer_idx": answer_idx,
        "video": f"videos/clip_{index}.mp4",
        "video_id": f"clip_{index}",
        **extra,
    }


def make_media(items: list[dict], *, setting: str) -> dict[str, VideoSource]:
    media = {}
    for index, item in enumerate(items):
        rel = item["video"]
        digest = f"{index + 1:064x}"
        if setting == "single_turn":
            media[rel] = VideoSource(source_url=rel, sha256=digest, video_url=f"file:///media/{rel}")
        else:
            media[rel] = VideoSource(source_url=rel, sha256=digest, source=f"/media/{rel}", sandbox_path=rel)
    return media


def test_load_videoindex_validates_two_to_ten_options_and_errors() -> None:
    items = [make_item(0, option_count=2, answer_idx=1), make_item(1, option_count=10, answer_idx=9)]
    loaded = load_videoindex(
        items,
        benchmark="videoindex_v1_0",
        dataset_revision=RELEASES["videoindex_v1_0"].dataset_revision,
        media=make_media(items, setting="single_turn"),
    )
    assert [item.question.task_id for item in loaded] == ["videoindex_v1_0:synth_0", "videoindex_v1_0:synth_1"]
    assert [len(item.question.options) for item in loaded] == [2, 10]
    assert [item.expected_answer for item in loaded] == ["B", "J"]

    # Duplicate options are preserved with a warning instead of dropped.
    dup = make_item(0, options=["Same", "Same", "Other"], answer="B", answer_idx=1)
    dup_loaded = load_videoindex(
        [dup],
        benchmark="videoindex_v1_0",
        dataset_revision=RELEASES["videoindex_v1_0"].dataset_revision,
        media=make_media([dup], setting="single_turn"),
    )
    assert dup_loaded[0].annotation_warnings == ("duplicate_option_text",)

    # Invalid cases must raise ValueError.
    bad_cases = [
        make_item(0, options=["Only one"], answer="A", answer_idx=0),
        make_item(0, options=[f"Opt {i}" for i in range(11)], answer="A", answer_idx=0),
        make_item(0, answer="C", answer_idx=1),
        make_item(0, capability_group="unknown_group"),
        make_item(0, video="../escape.mp4"),
        make_item(0, video="other/clip_0.mp4"),
    ]
    for bad in bad_cases:
        with pytest.raises(ValueError):
            load_videoindex(
                [bad],
                benchmark="videoindex_v1_0",
                dataset_revision=RELEASES["videoindex_v1_0"].dataset_revision,
                media=make_media([make_item(0)], setting="single_turn"),
            )


@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
def test_prepare_videoindex_dataset_privacy_and_task_data_validation(setting: str) -> None:
    items = [make_item(i, option_count=2 + i * 2, answer_idx=1) for i in range(4)]
    rows, cohort = prepare_videoindex_dataset(
        items,
        benchmark="videoindex_v1_1",
        dataset_revision=RELEASES["videoindex_v1_1"].dataset_revision,
        media=make_media(items, setting=setting),
        setting=setting,
    )
    assert cohort.benchmark == "videoindex_v1_1"
    assert cohort.answer_policy == VIDEOINDEX_RULE_POLICY
    assert len(rows) == 4
    validator = TaskDataValidator("videoindex", load_task_data_schema(SERVER_DIR), "rows.jsonl")
    for idx, row in enumerate(rows):
        validator.validate_row(idx, row)
        visible = json.dumps(row["responses_create_params"])
        assert "expected_answer" not in visible
        assert "SyntheticBench" not in visible
        assert "You are given a video. Answer the question based on this video." in visible
        if setting == "single_turn":
            assert row["responses_create_params"]["tools"] == []
            assert row["responses_create_params"]["tool_choice"] == "none"
            assert row["responses_create_params"]["input"][0]["content"][0]["type"] == "input_video"
            assert "video_binding" not in row
        else:
            assert "Video: /testbed/videos/" in visible
            assert row["video_binding"]["sandbox_path"].startswith("videos/")
            assert "tools" not in row["responses_create_params"]
    assert validator.report.clean, validator.report.summary()


@pytest.mark.parametrize("use_checksums_file", [True, False], ids=["files_sha256", "local_hash"])
@pytest.mark.parametrize("benchmark,setting", [("videoindex_v1_0", "single_turn"), ("videoindex_v1_1", "agentic")])
def test_prepare_videoindex_benchmark_offline_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_checksums_file: bool, benchmark: str, setting: str
) -> None:
    source_dir = tmp_path / "src"
    (source_dir / "items").mkdir(parents=True)
    (source_dir / "videos").mkdir(parents=True)
    items = [make_item(i, option_count=3 + i, answer_idx=1) for i in range(4)]
    items_bytes = "".join(json.dumps(item) + "\n" for item in items).encode("utf-8")
    (source_dir / "items" / "test.jsonl").write_bytes(items_bytes)
    items_sha = hashlib.sha256(items_bytes).hexdigest()

    checksum_lines = []
    for index, item in enumerate(items):
        rel = item["video"]
        video_bytes = f"synthetic video bytes {index}".encode("utf-8")
        digest = hashlib.sha256(video_bytes).hexdigest()
        checksum_lines.append(f"{digest}  {rel}\n")
        if not use_checksums_file:
            (source_dir / rel).write_bytes(video_bytes)
    if use_checksums_file:
        (source_dir / "FILES.sha256").write_text("".join(checksum_lines), encoding="utf-8")

    patched = dict(vi_prepare.RELEASES)
    patched[benchmark] = replace(patched[benchmark], items_sha256=items_sha, num_questions=len(items))
    monkeypatch.setattr(vi_prepare, "RELEASES", patched)

    out_jsonl = tmp_path / "out" / "rows.jsonl"
    out_cohort = tmp_path / "out" / "cohort.json"
    result = prepare_videoindex_benchmark(
        benchmark=benchmark,
        setting=setting,
        output_fpath=out_jsonl,
        cohort_fpath=out_cohort,
        source_dir=str(source_dir),
    )
    assert result == out_jsonl
    rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    cohort = json.loads(out_cohort.read_text(encoding="utf-8"))
    assert cohort["benchmark"] == benchmark and cohort["setting"] == setting
    assert len(cohort["task_ids"]) == 4

    validator = TaskDataValidator("videoindex", load_task_data_schema(SERVER_DIR), str(out_jsonl))
    for idx, row in enumerate(rows):
        validator.validate_row(idx, row)
    assert validator.report.clean, validator.report.summary()


def test_prepare_videoindex_benchmark_download_and_sha_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "dl"
    (source_dir / "items").mkdir(parents=True)
    items = [make_item(0, option_count=5, answer_idx=4)]
    items_bytes = (json.dumps(items[0]) + "\n").encode("utf-8")
    (source_dir / "items" / "test.jsonl").write_bytes(items_bytes)
    items_sha = hashlib.sha256(items_bytes).hexdigest()
    video_bytes = b"downloaded video bytes"
    video_sha = hashlib.sha256(video_bytes).hexdigest()

    patched = dict(vi_prepare.RELEASES)
    patched["videoindex_v1_1"] = replace(patched["videoindex_v1_1"], items_sha256=items_sha, num_questions=1)
    monkeypatch.setattr(vi_prepare, "RELEASES", patched)
    monkeypatch.setattr(
        vi_prepare, "_hub_lfs_metadata", lambda release, paths: {paths[0]: (len(video_bytes), video_sha)}
    )

    def fake_download(release, path, *, root):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(video_bytes)
        return target

    monkeypatch.setattr(vi_prepare, "_download_video", fake_download)

    out_jsonl = tmp_path / "out.jsonl"
    out_cohort = tmp_path / "cohort.json"
    prepare_videoindex_benchmark(
        benchmark="videoindex_v1_1",
        setting="single_turn",
        output_fpath=out_jsonl,
        cohort_fpath=out_cohort,
        source_dir=str(source_dir),
        download=True,
    )
    row = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    assert row["verifier_metadata"]["video_sha256"] == video_sha

    # Corrupt items file must fail SHA-256 verification.
    (source_dir / "items" / "test.jsonl").write_text('{"corrupt": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_videoindex_benchmark(
            benchmark="videoindex_v1_1",
            setting="single_turn",
            output_fpath=out_jsonl,
            cohort_fpath=out_cohort,
            source_dir=str(source_dir),
        )


def test_release_pins_match_official_v1_0_and_cleaned_v1_1() -> None:
    v1_0 = RELEASES["videoindex_v1_0"]
    v1_1 = RELEASES["videoindex_v1_1"]
    assert (v1_0.repo_id, v1_0.num_questions) == ("Video-Index/Video-Index", 840)
    assert v1_0.dataset_revision.startswith("fb63b6c9")
    assert (v1_1.repo_id, v1_1.num_questions) == ("Video-Index/Video-Index-v1-1", 144)
    assert v1_1.dataset_revision.startswith("e7404c46")


def test_server_configs_resolve_and_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIDEO_MCQA_SANDBOX_IMAGE", "registry.example/video-mcqa-sandbox:test")
    configs_dir = SERVER_DIR / "configs"
    cohort = tmp_path / "cohort.json"

    # Single-turn rule config
    st_plain = OmegaConf.load(configs_dir / "videoindex_single_turn.yaml")
    st_resolved = OmegaConf.to_container(
        OmegaConf.merge(st_plain, OmegaConf.create({"videoindex_cohort_manifest": str(cohort)})), resolve=True
    )
    st_res = VideoMCQAResourcesServerConfig(
        name="videoindex", host="127.0.0.1", port=8001, **st_resolved["videoindex"]["resources_servers"]["videoindex"]
    )
    assert st_res.cohort_manifest == cohort and st_res.judge_model_server is None
    st_agent = SimpleAgentConfig(
        name="videoindex_single_turn",
        host="127.0.0.1",
        port=8002,
        **st_resolved["videoindex_single_turn"]["responses_api_agents"]["simple_agent"],
    )
    assert st_agent.resources_server.name == "videoindex" and st_agent.max_steps == 1
    st_env = SingleAgentTurnLegacyEnvironmentServerConfig(
        name="videoindex_single_turn_environment",
        host="127.0.0.1",
        port=8003,
        **st_resolved["videoindex_single_turn_environment"]["environment_servers"]["single_agent_turn_legacy"],
    )
    assert st_env.agent_server.name == "videoindex_single_turn"

    # Single-turn LLM extraction config
    st_llm = OmegaConf.load(configs_dir / "videoindex_single_turn_llm_extraction.yaml")
    st_llm_resolved = OmegaConf.to_container(
        OmegaConf.merge(
            st_llm,
            OmegaConf.create(
                {
                    "videoindex_cohort_manifest": str(cohort),
                    "judge_base_url": "https://judge.example.invalid/v1",
                    "judge_api_key": "placeholder",
                    "judge_model_name": "judge-model",
                }
            ),
        ),
        resolve=True,
    )
    st_llm_res = VideoMCQAResourcesServerConfig(
        name="videoindex",
        host="127.0.0.1",
        port=8001,
        **st_llm_resolved["videoindex"]["resources_servers"]["videoindex"],
    )
    assert st_llm_res.judge_model_server == ModelServerRef(type="responses_api_models", name="video_mcqa_answer_judge")

    # Agentic rule config + LLM extraction loopback judge overlay
    ag_plain = OmegaConf.load(configs_dir / "videoindex_agentic.yaml")
    assert list(ag_plain.config_paths) == ["nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"]
    ag_values = {
        "videoindex_agentic_cohort_manifest": str(cohort),
        "videoindex_agentic_video_roots": [str(tmp_path)],
        "policy_model_name": "policy",
        "judge_base_url": "https://judge.example.invalid/v1",
        "judge_api_key": "placeholder",
        "judge_model_name": "judge-model",
    }
    ag_resolved = OmegaConf.to_container(OmegaConf.merge(ag_plain, OmegaConf.create(ag_values)), resolve=True)
    ag_res = VideoMCQAResourcesServerConfig(
        name="videoindex_agentic",
        port=8001,
        **ag_resolved["videoindex_agentic"]["resources_servers"]["videoindex"],
    )
    assert ag_res.sandbox_provider == "sandbox" and ag_res.sandbox_network_access == "model_only"
    assert ag_res.sandbox_config.image == "registry.example/video-mcqa-sandbox:test"
    ag_agent = OpenCodeAgentConfig(
        name="videoindex_agentic_opencode",
        port=8002,
        **ag_resolved["videoindex_agentic_opencode"]["responses_api_agents"]["opencode_agent"],
    )
    assert ag_res.host == ag_agent.host == "127.0.0.1"
    assert ag_agent.plugins == [str(PLUGIN_PATH)]
    dockerfile = (SANDBOX_IMAGE_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert f"{ag_agent.sandbox_executable_sha256}  {ag_agent.sandbox_executable_path}" in dockerfile
    assert ag_agent.sandbox_plugin_dependencies_sha256 in dockerfile

    ag_overlay = OmegaConf.load(configs_dir / "videoindex_agentic_llm_extraction.yaml")
    ag_merged = OmegaConf.to_container(
        OmegaConf.merge(ag_plain, ag_overlay, OmegaConf.create(ag_values)), resolve=True
    )
    ag_llm_res = VideoMCQAResourcesServerConfig(
        name="videoindex_agentic",
        port=8001,
        **ag_merged["videoindex_agentic"]["resources_servers"]["videoindex"],
    )
    assert ag_llm_res.judge_model_server == ModelServerRef(type="responses_api_models", name="video_mcqa_answer_judge")
    assert ag_merged["video_mcqa_answer_judge"]["responses_api_models"]["openai_model"]["host"] == "127.0.0.1"


@pytest.mark.parametrize("benchmark_name", BENCHMARKS)
def test_benchmark_manifests_and_prepare_entrypoints_validate_cleanly(benchmark_name: str) -> None:
    benchmark_dir = REPO_ROOT / "benchmarks" / benchmark_name
    manifest_path = benchmark_dir / "manifest.yaml"
    manifest = load_manifest(manifest_path)
    result = validate_environment(manifest_path, validate_datasets=False)
    assert result.name == benchmark_name
    assert result.declared_profile == result.inferred_profile
    assert result.warnings == ()
    assert manifest.name == benchmark_name
    assert manifest.licensing == "unknown"
    assert manifest.resources_server == "videoindex"
    assert manifest.maintainers == ["cg1177"]
    assert manifest.capabilities
    assert manifest.domains == ["general"]
    is_agentic = benchmark_name.endswith("_agentic")
    base_name = benchmark_name.removesuffix("_agentic")
    expected_metric = f"mean/{RELEASES[base_name].metric_name}{'_agentic' if is_agentic else ''}"
    assert manifest.primary_metric == expected_metric
    assert manifest.requirements is not None
    assert manifest.requirements.gpu is False
    assert manifest.requirements.network == ("allowlist" if is_agentic else "no-network")
    mod = importlib.import_module(f"benchmarks.{benchmark_name}.prepare")
    assert callable(mod.prepare)
    with pytest.raises(ValueError, match="Set source_dir"):
        mod.prepare(source_dir=None)


@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
@pytest.mark.parametrize("local_state", ["missing", "valid", "corrupt"])
def test_download_mode_materializes_and_repairs_even_when_files_sha256_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str, local_state: str
) -> None:
    source_dir = tmp_path / "src"
    (source_dir / "items").mkdir(parents=True)
    items = [make_item(0, option_count=5, answer_idx=4)]
    items_bytes = (json.dumps(items[0]) + "\n").encode("utf-8")
    (source_dir / "items" / "test.jsonl").write_bytes(items_bytes)
    items_sha = hashlib.sha256(items_bytes).hexdigest()

    rel = str(items[0]["video"])
    good_bytes = b"valid video bytes"
    good_sha = hashlib.sha256(good_bytes).hexdigest()
    (source_dir / "FILES.sha256").write_text(f"{good_sha}  {rel}\n", encoding="utf-8")

    local_video = source_dir / rel
    if local_state == "valid":
        local_video.parent.mkdir(parents=True, exist_ok=True)
        local_video.write_bytes(good_bytes)
    elif local_state == "corrupt":
        local_video.parent.mkdir(parents=True, exist_ok=True)
        local_video.write_bytes(b"corrupt")

    patched = dict(vi_prepare.RELEASES)
    patched["videoindex_v1_1"] = replace(patched["videoindex_v1_1"], items_sha256=items_sha, num_questions=1)
    monkeypatch.setattr(vi_prepare, "RELEASES", patched)
    monkeypatch.setattr(
        vi_prepare, "_hub_lfs_metadata", lambda release, paths: {paths[0]: (len(good_bytes), good_sha)}
    )

    download_calls: list[str] = []

    def fake_download(release, path, *, root):
        download_calls.append(path)
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(good_bytes)
        return target

    monkeypatch.setattr(vi_prepare, "_download_video", fake_download)

    out_jsonl = tmp_path / "out.jsonl"
    out_cohort = tmp_path / "cohort.json"
    prepare_videoindex_benchmark(
        benchmark="videoindex_v1_1",
        setting=setting,
        output_fpath=out_jsonl,
        cohort_fpath=out_cohort,
        source_dir=str(source_dir),
        download=True,
    )
    row = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    assert row["verifier_metadata"]["video_sha256"] == good_sha
    assert local_video.is_file()
    assert hashlib.sha256(local_video.read_bytes()).hexdigest() == good_sha
    if local_state == "valid":
        assert download_calls == []
    else:
        assert download_calls == [rel]


@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
@pytest.mark.parametrize("use_custom_video_dir", [False, True], ids=["default_dir", "custom_dir"])
@pytest.mark.parametrize("stale", [False, True], ids=["missing", "stale"])
def test_download_mode_populates_exact_referenced_video_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str, use_custom_video_dir: bool, stale: bool
) -> None:
    source_dir = tmp_path / "src"
    (source_dir / "items").mkdir(parents=True)
    items = [make_item(0, option_count=4, answer_idx=2)]
    items_bytes = (json.dumps(items[0]) + "\n").encode("utf-8")
    (source_dir / "items" / "test.jsonl").write_bytes(items_bytes)
    items_sha = hashlib.sha256(items_bytes).hexdigest()

    rel = str(items[0]["video"])
    good_bytes = b"fresh verified media"
    good_sha = hashlib.sha256(good_bytes).hexdigest()

    custom_dir = tmp_path / "custom_videos" if use_custom_video_dir else None
    effective_dest = (custom_dir or (source_dir / "videos")) / Path(rel).name
    if stale:
        effective_dest.parent.mkdir(parents=True, exist_ok=True)
        effective_dest.write_bytes(b"stale bytes")

    patched = dict(vi_prepare.RELEASES)
    patched["videoindex_v1_0"] = replace(patched["videoindex_v1_0"], items_sha256=items_sha, num_questions=1)
    monkeypatch.setattr(vi_prepare, "RELEASES", patched)
    monkeypatch.setattr(
        vi_prepare, "_hub_lfs_metadata", lambda release, paths: {paths[0]: (len(good_bytes), good_sha)}
    )

    def fake_download(release, path, *, root):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(good_bytes)
        return target

    monkeypatch.setattr(vi_prepare, "_download_video", fake_download)

    out_jsonl = tmp_path / "out.jsonl"
    out_cohort = tmp_path / "cohort.json"
    prepare_videoindex_benchmark(
        benchmark="videoindex_v1_0",
        setting=setting,
        output_fpath=out_jsonl,
        cohort_fpath=out_cohort,
        source_dir=str(source_dir),
        video_dir=str(custom_dir) if custom_dir is not None else None,
        download=True,
    )
    row = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    if setting == "single_turn":
        emitted_url = row["responses_create_params"]["input"][0]["content"][0]["video_url"]
        emitted_path = Path(unquote(urlparse(emitted_url).path))
    else:
        emitted_path = Path(row["video_binding"]["source"])
    assert emitted_path == effective_dest.resolve()
    assert emitted_path.is_file()
    assert (
        hashlib.sha256(emitted_path.read_bytes()).hexdigest() == row["verifier_metadata"]["video_sha256"] == good_sha
    )


@pytest.mark.parametrize("benchmark_name", BENCHMARKS)
def test_public_benchmark_wrappers_propagate_num_repeats_and_normalize_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, benchmark_name: str
) -> None:
    is_agentic = benchmark_name.endswith("_agentic")
    base_name = benchmark_name.removesuffix("_agentic")
    source_dir = tmp_path / "rel_src"
    (source_dir / "items").mkdir(parents=True)
    items = [make_item(i, option_count=3 + i, answer_idx=1) for i in range(4)]
    items_bytes = "".join(json.dumps(item) + "\n" for item in items).encode("utf-8")
    (source_dir / "items" / "test.jsonl").write_bytes(items_bytes)
    items_sha = hashlib.sha256(items_bytes).hexdigest()

    checksum_lines = []
    for index, item in enumerate(items):
        rel = str(item["video"])
        digest = hashlib.sha256(f"vid-{index}".encode()).hexdigest()
        checksum_lines.append(f"{digest}  {rel}\n")
    (source_dir / "FILES.sha256").write_text("".join(checksum_lines), encoding="utf-8")

    patched = dict(vi_prepare.RELEASES)
    patched[base_name] = replace(patched[base_name], items_sha256=items_sha, num_questions=len(items))
    monkeypatch.setattr(vi_prepare, "RELEASES", patched)

    mod = importlib.import_module(f"benchmarks.{benchmark_name}.prepare")
    out_jsonl = tmp_path / "out" / f"{benchmark_name}.jsonl"
    out_cohort = tmp_path / "out" / f"{benchmark_name}_cohort.json"
    monkeypatch.setattr(mod, "OUTPUT_FPATH", out_jsonl)
    monkeypatch.setattr(mod, "COHORT_FPATH", out_cohort)
    monkeypatch.chdir(tmp_path)

    kwargs: dict[str, object] = {"source_dir": "rel_src", "num_repeats": 2}
    if not is_agentic:
        special_url_dir = tmp_path / "video#archive?part=1"
        kwargs["video_url_dir"] = str(special_url_dir)
    mod.prepare(**kwargs)

    cohort = json.loads(out_cohort.read_text(encoding="utf-8"))
    assert cohort["num_repeats"] == 2
    rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    if is_agentic:
        assert Path(rows[0]["video_binding"]["source"]).is_absolute()
        assert Path(rows[0]["video_binding"]["source"]) == (source_dir / "videos" / "clip_0.mp4").resolve()
    else:
        url = rows[0]["responses_create_params"]["input"][0]["content"][0]["video_url"]
        assert "video%23archive%3Fpart%3D1" in url
        assert unquote(urlparse(url).path) == str(special_url_dir / "clip_0.mp4")
