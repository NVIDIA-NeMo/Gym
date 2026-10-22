# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from resources_servers.video_mcqa import prepare
from resources_servers.video_mcqa.prepare import prepare_files, read_annotations, sha256_file
from resources_servers.video_mcqa.tests.helpers import REVISION, lv_record, media, v1_record


def inputs(tmp_path, *, benchmark="videomme_v1", source=True):
    records = [v1_record(), v1_record("two")] if benchmark == "videomme_v1" else [lv_record(), lv_record("two")]
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(records))
    video = tmp_path / "invented-video.bin"
    video.write_bytes(b"synthetic fixture, not a real benchmark video")
    values = {"source": str(video), "sandbox_path": "video/shapes.mp4", "sha256": sha256_file(video)} if source else {}
    bindings = media(**values)
    manifest = tmp_path / "media.json"
    manifest.write_text(json.dumps({identity: value.model_dump() for identity, value in bindings.items()}))
    return {
        "annotations": annotations,
        "media_manifest": manifest,
        "output_dir": tmp_path / "prepared",
        "benchmark": benchmark,
        "dataset_revision": REVISION,
        "setting": "single_turn",
    }, video


@pytest.mark.parametrize("benchmark", ["videomme_v1", "lvbench"])
@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
def test_preparation_writes_complete_non_overwriting_artifacts(tmp_path, benchmark, setting) -> None:
    kwargs, video = inputs(tmp_path, benchmark=benchmark)
    kwargs["setting"] = setting
    original_annotations = kwargs["annotations"].read_bytes()
    with patch.object(prepare, "sha256_file", wraps=sha256_file) as hashing:
        receipt = prepare_files(**kwargs)
    assert sum(call.args == (video.resolve(),) for call in hashing.call_args_list) == 1
    assert receipt["questions"] == 2 and receipt["videos"] == 1 and receipt["original_videos_hashed"] == 1
    assert receipt["audio"] is False and receipt["subtitles"] is False and receipt["evaluation_only"] is True
    directory = kwargs["output_dir"]
    assert json.loads((directory / "receipt.json").read_text()) == receipt
    assert receipt["tasks_sha256"] == sha256_file(directory / "tasks.jsonl")
    assert receipt["cohort_sha256"] == sha256_file(directory / "cohort.json")
    cohort = json.loads((directory / "cohort.json").read_text())
    assert len(cohort["task_ids"]) == 2 and cohort["benchmark"] == benchmark
    tasks = [json.loads(line) for line in (directory / "tasks.jsonl").read_text().splitlines()]
    assert [row["task_id"] for row in tasks] == cohort["task_ids"]
    if setting == "agentic":
        assert tasks[0]["video_binding"]["source"] == str(video.resolve())
    assert kwargs["annotations"].read_bytes() == original_annotations
    assert video.read_bytes() == b"synthetic fixture, not a real benchmark video"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_files(**kwargs)
    assert json.loads((directory / "receipt.json").read_text()) == receipt


def test_single_turn_inventory_may_be_preverified_by_external_preprocessor(tmp_path) -> None:
    kwargs, _ = inputs(tmp_path, source=False)
    assert prepare_files(**kwargs)["original_videos_hashed"] == 0


def test_changed_source_fails_before_any_output(tmp_path) -> None:
    kwargs, video = inputs(tmp_path)
    video.write_bytes(b"changed fixture bytes")
    with pytest.raises(ValueError, match="SHA256 differs"):
        prepare_files(**kwargs)
    assert not kwargs["output_dir"].exists()


def test_unavailable_media_is_a_visible_failure_not_a_smaller_cohort(tmp_path) -> None:
    kwargs, video = inputs(tmp_path)
    # Point to a nonexistent source without deleting the original fixture.
    manifest = json.loads(kwargs["media_manifest"].read_text())
    manifest["shapes"]["source"] = str(video.with_name("unavailable.mp4"))
    kwargs["media_manifest"].write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError):
        prepare_files(**kwargs)
    assert not kwargs["output_dir"].exists()


@pytest.mark.parametrize("payload", [[], [1], "bad"])
def test_media_manifest_must_be_a_mapping(tmp_path, payload) -> None:
    kwargs, _ = inputs(tmp_path)
    kwargs["media_manifest"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="must map"):
        prepare_files(**kwargs)


def test_jsonl_and_invalid_input_layouts(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(v1_record()) + "\n\n" + json.dumps(v1_record("two")) + "\n")
    assert len(read_annotations(path)) == 2
    for suffix, payload in (("json", "{}"), ("json", "[1]"), ("jsonl", "1\n"), ("txt", "[]")):
        bad = tmp_path / f"bad.{suffix}"
        bad.write_text(payload)
        with pytest.raises(ValueError):
            read_annotations(bad)


def test_optional_parquet_loader_and_missing_dependency(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rows.parquet"
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "pyarrow.parquet", None)
        with pytest.raises(ValueError, match="optional pyarrow"):
            read_annotations(path)
    parquet = pytest.importorskip("pyarrow.parquet")
    arrow = pytest.importorskip("pyarrow")
    parquet.write_table(arrow.Table.from_pylist([v1_record()]), path)
    assert read_annotations(path) == [v1_record()]


def test_cli_uses_same_data_contract(tmp_path, monkeypatch, capsys) -> None:
    kwargs, _ = inputs(tmp_path)
    arguments = ["prepare"]
    for name, value in kwargs.items():
        arguments.extend(["--" + name.replace("_", "-"), str(value)])
    monkeypatch.setattr(sys, "argv", arguments)
    prepare.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["questions"] == 2 and receipt["dataset_revision"] == REVISION


def test_invalid_annotations_do_not_leave_a_partial_output(tmp_path) -> None:
    kwargs, _ = inputs(tmp_path)
    kwargs["annotations"].write_text(json.dumps([v1_record(), v1_record()]))
    with pytest.raises(ValueError, match="unique question IDs"):
        prepare_files(**kwargs)
    assert not kwargs["output_dir"].exists()


def test_v2_exports_preserve_the_exact_shared_types() -> None:
    from resources_servers.video_mcqa import scoring, task
    from resources_servers.videomme_v2 import scoring as v2_scoring
    from resources_servers.videomme_v2 import task as v2_task

    assert v2_task.VideoQuestion is task.VideoQuestion
    assert v2_task.VerifierMetadata is task.VerifierMetadata
    assert v2_task.prepare_single_turn is task.prepare_single_turn
    assert v2_scoring.grade_answer is scoring.grade_answer
    assert v2_scoring.aggregate_scores is scoring.aggregate_scores


def test_all_four_configs_keep_the_existing_harness_and_are_unverified() -> None:
    import yaml

    root = Path(__file__).parents[2]
    for benchmark in ("videomme_v1", "lvbench"):
        for setting in ("single_turn", "agentic"):
            config = yaml.safe_load((root / benchmark / "configs" / f"{benchmark}_{setting}.yaml").read_text())
            resource_name = benchmark if setting == "single_turn" else benchmark + "_agentic_resources"
            resource = config[resource_name]["resources_servers"][benchmark]
            assert resource["verified"] is False and resource["entrypoint"] == "app.py"
            agent = config[f"{benchmark}_{setting}"]["responses_api_agents"]
            assert list(agent) == (["simple_agent"] if setting == "single_turn" else ["opencode_video_agent"])
            if setting == "single_turn":
                assert agent["simple_agent"]["max_steps"] == 1
            else:
                assert agent["opencode_video_agent"]["execution_failure_reward_zero"] is True
                assert agent["opencode_video_agent"]["opencode_version"] == "1.17.11"
