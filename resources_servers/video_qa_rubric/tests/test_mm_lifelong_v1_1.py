# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""benchmarks/mm_lifelong_v1_1: its grading config and prepare.py on an invented release (no download).

The invented release has the real layout (SHA256SUMS, manifest.json, v1.1/annotations, an LMDB frame cache of JPEG
frames under eight-digit keys) with a few frames and questions per split. Building it and reading the MP4s back needs
lmdb, PyAV and Pillow, the dataset's prepare_dependencies; without them those tests are skipped.
"""

import copy
import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.mm_lifelong_v1_1 import prepare
from resources_servers.video_qa_rubric.app import ReferenceLevelsMetadata, VideoQARubricResourcesServerConfig
from resources_servers.video_qa_rubric.reference_levels import ReferenceLevelsSettings, judge_messages
from resources_servers.video_qa_rubric.task_data import TaskData


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "mm_lifelong_v1_1"
# SHA-256 of v1.1/grading/prompts.json at the pinned revision (the release's SHA256SUMS lists it).
PROMPTS_SHA256 = "70b29a3e45bd883f0984aedb24a4d4d0d83add564797a22e9aed3ac1ec5c5a06"  # pragma: allowlist secret
# The splits of the invented release: (domain, frame size, frames).
DOMAINS = {"month_val": ("month", (64, 36), 9), "week_test": ("week", (48, 48), 7), "day_test": ("day", (34, 20), 5)}


def grading() -> dict:
    return OmegaConf.to_container(OmegaConf.load(BENCHMARK / "grading.yaml"), resolve=False)


def test_the_grading_config_is_the_dataset_grader_prompt_with_the_servers_field_names() -> None:
    settings = ReferenceLevelsSettings.model_validate(grading()["mm_lifelong_v1_1_grading"])
    official = {
        "system": settings.system_prompt,
        "user_template": settings.user_template.replace("{rubric_json}", "{frozen_rubric_json}").replace(
            "{answer}", "{model_prediction}"
        ),
    }
    # prompts.json is this object as json.dumps(indent=2) writes it, with a final newline.
    text = json.dumps(official, indent=2, ensure_ascii=False) + "\n"
    assert hashlib.sha256(text.encode()).hexdigest() == PROMPTS_SHA256
    assert settings.max_level == 5
    rubric = {"score_rules": {"5": ["Says 5."], "0": ["Anything else."]}, "aliases": [{"expression": "黑风山"}]}
    system, user = judge_messages(settings, question="How many?", reference_answer="5", rubric=rubric, answer="Five.")
    # The grader's request: the rubric as json.dumps(rubric, ensure_ascii=False), the prediction last.
    assert user["content"] == (
        "QUESTION:\nHow many?\nREFERENCE ANSWER:\n5\nFROZEN RUBRIC:\n"
        + json.dumps(rubric, ensure_ascii=False)
        + "\nMODEL PREDICTION:\nFive.\n\nEvaluate this prediction according to the supplied rubric."
    )
    assert system == {"role": "system", "content": settings.system_prompt}


def test_the_config_grades_with_the_dataset_grader_and_one_completion() -> None:
    values = {"mm_lifelong_v1_1_data_dir": "/data/mml", "judge_api_key": "invented"}  # pragma: allowlist secret
    loaded = OmegaConf.merge(
        OmegaConf.load(ROOT / "resources_servers" / "video_qa_rubric" / "configs" / "video_qa_rubric.yaml"),
        OmegaConf.load(BENCHMARK / "grading.yaml"),
        OmegaConf.load(BENCHMARK / "config.yaml"),
        OmegaConf.create(values),
    )
    config = OmegaConf.to_container(loaded, resolve=True)
    server = VideoQARubricResourcesServerConfig(
        name="video_qa_rubric",
        host="127.0.0.1",
        port=8001,
        **config["video_qa_rubric"]["resources_servers"]["video_qa_rubric"],
    )
    assert server.reference_levels.max_level == 5 and server.empty_answer_reward is None
    assert server.judge_chat_completion_create_params.model_dump(exclude_unset=True) == {"messages": [], "n": 1}
    assert config["judge_model_name"] == "openai/openai/gpt-5.6-luna"
    assert config["prepare_script_args"] == {
        "data_dir": "/data/mml",
        "frames_dir": "/data/mml/frames_256",
        "frames_url_dir": "/data/mml/frames_256",
        "num_frames": 256,
    }
    (dataset,) = config["video_qa_rubric_single_turn"]["responses_api_agents"]["simple_agent"]["datasets"]
    assert dataset["jsonl_fpath"] == str(prepare.OUTPUT_FPATH.relative_to(ROOT))
    assert dataset["prepare_script"] == "benchmarks/mm_lifelong_v1_1/prepare.py"


def test_frames_are_the_centres_of_equal_bins() -> None:
    assert prepare.sample_frame_ids(10, 5) == [1, 3, 5, 7, 9]
    assert prepare.sample_frame_ids(7, 7) == list(range(7))
    ids = prepare.sample_frame_ids(169_825, 256)
    assert (ids[0], ids[-1], len(set(ids))) == (331, 169_493, 256)
    with pytest.raises(ValueError, match="num_frames"):
        prepare.sample_frame_ids(5, 6)


def test_splits_alternate_so_any_prefix_covers_each() -> None:
    rows = prepare.interleave([["m1", "m2", "m3"], ["w1"], ["d1", "d2"]])
    assert rows == ["m1", "w1", "d1", "m2", "d2", "m3"]


def question(split: str, index: int) -> dict:
    return {
        "id": f"{split}:{index}",
        "split": split,
        "index": index,
        "question": f"How many invented squares appear in part {index} of the {split} recording?",
        "answer": str(index + 2),
        "total_intervals": [[1, 2]],
        "rubric": {
            "answer_type": "exact_count",
            "score_rules": {"5": [f"Says {index + 2}."], "0": ["Any other count."]},
            "rejection_rules": [{"answer_or_pattern": "Any other count", "reason": "Exact count."}],
        },
    }


@pytest.fixture
def release(tmp_path, monkeypatch) -> Path:
    """An invented release under tmp_path/hf and the module's outputs under tmp_path/out."""
    lmdb = pytest.importorskip("lmdb")
    image = pytest.importorskip("PIL.Image")
    root = tmp_path / "hf"
    files: dict[str, bytes] = {}
    subsets = {}
    for split, (domain, (width, height), total) in DOMAINS.items():
        shards = []
        for first in range(0, total, 4):
            relpath = f"frames/{domain}/shard_{first // 4:05d}.lmdb/data.mdb"
            (root / relpath).parent.mkdir(parents=True)
            environment = lmdb.open(str((root / relpath).parent), map_size=1 << 24)
            with environment.begin(write=True) as transaction:
                for frame in range(first, min(first + 4, total)):
                    buffer = BytesIO()
                    image.new("RGB", (width, height), (frame * 20, 80, 160)).save(buffer, format="JPEG")
                    transaction.put(f"{frame:08d}".encode(), buffer.getvalue())
            environment.close()
            frames = min(4, total - first)
            shards.append({"path": relpath, "frames": frames, "first_frame_id": first})
            files[relpath] = (root / relpath).read_bytes()
        subsets[domain] = {
            "domain": domain,
            "width": width,
            "height": height,
            "fps_cached": 2.0,
            "total_frames": total,
            "duration_seconds": total / 2,
            "shards": shards,
        }
        files[f"v1.1/annotations/{split}.jsonl"] = "".join(
            json.dumps(question(split, index), ensure_ascii=False) + "\n"
            for index in range(3 if domain == "month" else 2)
        ).encode()
    files["manifest.json"] = json.dumps({"subsets": subsets}).encode()
    for relpath, content in files.items():
        (root / relpath).parent.mkdir(parents=True, exist_ok=True)
        (root / relpath).write_bytes(content)
    sums = "".join(f"{hashlib.sha256(content).hexdigest()}  {relpath}\n" for relpath, content in files.items())
    (root / "SHA256SUMS").write_text(sums)
    monkeypatch.setattr(prepare, "CHECKSUMS_SHA256", hashlib.sha256(sums.encode()).hexdigest())
    monkeypatch.setattr(
        prepare, "SPLITS", {"month_val": ("month", 3), "week_test": ("week", 2), "day_test": ("day", 2)}
    )
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "out" / "rows.jsonl")
    return root


def test_prepare_writes_one_frame_video_per_split_and_rows_for_the_judge_only(release, tmp_path) -> None:
    av = pytest.importorskip("av")
    output = prepare.prepare(
        data_dir=str(release), frames_dir=str(tmp_path / "frames"), frames_url_dir="/srv/frames", num_frames=3
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["task_id"] for row in rows] == [
        "mm_lifelong_v1_1/month_val:0",
        "mm_lifelong_v1_1/week_test:0",
        "mm_lifelong_v1_1/day_test:0",
        "mm_lifelong_v1_1/month_val:1",
        "mm_lifelong_v1_1/week_test:1",
        "mm_lifelong_v1_1/day_test:1",
        "mm_lifelong_v1_1/month_val:2",
    ]
    day = rows[2]
    item = question("day_test", 0)
    # The model sees the split's frames and the question as it is; never the answer or the rubric.
    assert day["responses_create_params"]["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_video", "video_url": "file:///srv/frames/day_test.mp4"},
                {"type": "input_text", "text": item["question"]},
            ],
        }
    ]
    assert day["verifier_metadata"] == {
        "grading": "reference_levels",
        "question": item["question"],
        "reference_answer": "2",
        "rubric": item["rubric"],
        "subset": "day_test",
    }
    for row in rows:
        TaskData.model_validate({**row, **row["verifier_metadata"]})
        ReferenceLevelsMetadata.model_validate(row["verifier_metadata"])
    for split, (domain, (width, height), total) in DOMAINS.items():
        record = json.loads((tmp_path / "frames" / f"{split}.json").read_text())
        frame_ids = prepare.sample_frame_ids(total, 3)
        assert record["domain"] == domain and record["dataset_revision"] == prepare.DATASET_REVISION
        assert record["frames"] == [{"frame_id": frame, "timeline_seconds": frame / 2} for frame in frame_ids]
        with av.open(str(tmp_path / "frames" / f"{split}.mp4")) as container:
            frames = list(container.decode(video=0))
        # Three frames at 1 fps, the manifest's size made even.
        assert len(frames) == 3 and (frames[0].width, frames[0].height) == (width // 2 * 2, height // 2 * 2)
        assert [round(float(frame.time)) for frame in frames] == [0, 1, 2]
        # Each frame is its cache frame: the invented frames differ in their red channel.
        reds = [int(frame.to_ndarray(format="rgb24")[..., 0].mean()) for frame in frames]
        assert all(abs(red - frame * 20) <= 6 for red, frame in zip(reds, frame_ids)), reds


@pytest.mark.parametrize("damaged", ["v1.1/annotations/week_test.jsonl", "frames/day/shard_00001.lmdb/data.mdb"])
def test_a_file_that_differs_from_the_release_checksums_is_refused(release, tmp_path, damaged) -> None:
    with (release / damaged).open("ab") as stream:
        stream.write(b"\n")
    # Three frames of the day split's five are 0, 2 and 4: frame 4 is in its second shard.
    with pytest.raises(ValueError, match="SHA-256|differs from SHA256SUMS"):
        prepare.prepare(data_dir=str(release), frames_dir=str(tmp_path / "frames"), num_frames=3)


def test_only_the_shards_that_hold_the_frames_are_read(release, tmp_path) -> None:
    pytest.importorskip("av")
    # Two frames of the day split's five are 1 and 3, both in its first shard.
    with (release / "frames" / "day" / "shard_00001.lmdb" / "data.mdb").open("ab") as stream:
        stream.write(b"\n")
    prepare.prepare(data_dir=str(release), frames_dir=str(tmp_path / "frames"), num_frames=2)


def test_checksums_other_than_the_pinned_ones_are_refused(release, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(prepare, "CHECKSUMS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256SUMS has SHA-256"):
        prepare.prepare(data_dir=str(release), frames_dir=str(tmp_path / "frames"))


def test_questions_the_release_does_not_have_are_refused(release, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(prepare, "SPLITS", {**prepare.SPLITS, "day_test": ("day", 200)})
    with pytest.raises(ValueError, match="day_test: expected 200 questions"):
        prepare.prepare(data_dir=str(release), frames_dir=str(tmp_path / "frames"))


def test_evidence_past_the_timeline_is_refused(release, tmp_path, monkeypatch) -> None:
    path = release / "v1.1" / "annotations" / "day_test.jsonl"
    items = [json.loads(line) for line in path.read_text().splitlines()]
    late = copy.deepcopy(items)
    late[1]["total_intervals"] = [[2, 99]]
    path.write_text("".join(json.dumps(item) + "\n" for item in late))
    sums = (release / "SHA256SUMS").read_text().splitlines()
    sums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  v1.1/annotations/day_test.jsonl"
        if line.endswith("day_test.jsonl")
        else line
        for line in sums
    ]
    (release / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    monkeypatch.setattr(prepare, "CHECKSUMS_SHA256", hashlib.sha256((release / "SHA256SUMS").read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="day_test:1: evidence"):
        prepare.prepare(data_dir=str(release), frames_dir=str(tmp_path / "frames"))


def test_prepare_needs_the_local_frame_cache() -> None:
    with pytest.raises(ValueError, match="mm_lifelong_v1_1_data_dir"):
        prepare.prepare()
