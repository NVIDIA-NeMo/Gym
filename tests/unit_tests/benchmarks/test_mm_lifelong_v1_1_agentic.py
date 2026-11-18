# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""benchmarks/mm_lifelong_v1_1_agentic: prepare.py on an invented release and image catalogs, the image build script
without Docker, the Hub or FFmpeg, and the config (no download, no sandbox)."""

import hashlib
import json
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.mm_lifelong_v1_1 import prepare as single_turn
from benchmarks.mm_lifelong_v1_1_agentic import prepare
from benchmarks.mm_lifelong_v1_1_agentic.sandbox_image import build
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from resources_servers.video_mcqa.tests.helpers import response
from resources_servers.video_qa_rubric.app import VideoQARubricResourcesServerConfig, VideoQARubricVerifyRequest
from resources_servers.video_qa_rubric.task_data import TaskData


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "mm_lifelong_v1_1_agentic"
TIMELINE = {
    "month": [
        {"video_id": str(n), "physical_start": f"03-0{n}T10:00:00Z", "physical_end": f"03-0{n}T11:00:00Z"}
        for n in (1, 2, 10)
    ],
    "week": [{"day": d, "physical_start": f"Day {d} 09:00:00", "physical_end": f"Day {d} 21:00:00"} for d in (1, 2)],
    "day": {"physical_start": "Day 1 00:00:00", "physical_end": "Day 1 23:00:00"},
}
WEEK_CLIPS = [f"day{d}/DAY{d}_CLIP_{n:04d}.mp4" for d in (1, 2) for n in range(60 + d)]
FILES = {"day": ["0.mp4"], "week": WEEK_CLIPS, "month": ["1.mp4", "2.mp4", "10.mp4"]}


def question(split: str, index: int) -> dict:
    return {
        "id": f"{split}:{index}",
        "split": split,
        "question": f"What invented thing happens in part {index} of the {split} recording?",
        "answer": f"Thing {index}.",
        "total_intervals": [],
        "rubric": {"score_rules": {"5": [f"Says thing {index}."], "0": ["Anything else."]}},
    }


@pytest.fixture
def release(tmp_path, monkeypatch) -> Path:
    """The invented release's annotation files, as the single-turn benchmark checks them."""
    root = tmp_path / "hf"
    files = {
        "manifest.json": json.dumps({"subsets": {d: {"duration_seconds": 100.0} for d in ("day", "week", "month")}})
    }
    for split, (_, count) in {"month_val": (0, 3), "week_test": (0, 2), "day_test": (0, 1)}.items():
        files[f"v1.1/annotations/{split}.jsonl"] = "".join(json.dumps(question(split, i)) + "\n" for i in range(count))
    for relpath, text in files.items():
        (root / relpath).parent.mkdir(parents=True, exist_ok=True)
        (root / relpath).write_text(text)
    sums = "".join(f"{hashlib.sha256(text.encode()).hexdigest()}  {relpath}\n" for relpath, text in files.items())
    (root / "SHA256SUMS").write_text(sums)
    monkeypatch.setattr(single_turn, "CHECKSUMS_SHA256", hashlib.sha256(sums.encode()).hexdigest())
    monkeypatch.setattr(
        single_turn, "SPLITS", {"month_val": ("month", 3), "week_test": ("week", 2), "day_test": ("day", 1)}
    )
    monkeypatch.setattr(prepare, "VIDEO_FILES", {domain: len(names) for domain, names in FILES.items()})
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "out" / "rows.jsonl")
    return root


@pytest.fixture
def media(tmp_path) -> Path:
    """build.py's output for the invented videos: the image catalogs and the model's catalogs."""
    out = tmp_path / "media"
    for domain, names in FILES.items():
        expected = {name: (100 + i, f"{i:064x}") for i, name in enumerate(names)}
        catalog = {
            "videos": [
                {"path": f"/testbed/videos/{domain}/{name}", "bytes": size, "sha256": sha}
                for name, (size, sha) in expected.items()
            ]
        }
        records = build.model_records(domain, {name: 10.0 + i for i, name in enumerate(names)}, TIMELINE)
        (out / domain).mkdir(parents=True)
        (out / domain / "catalog.json").write_text(json.dumps(catalog, indent=1) + "\n")
        (out / domain / "video_catalog.json").write_text(json.dumps(records, indent=2) + "\n")
    return out


def test_rows_pin_their_domains_image_catalog_and_ask_with_the_runners_inputs(release, media) -> None:
    output = prepare.prepare(data_dir=str(release), media_dir=str(media))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["task_id"] for row in rows] == [
        "mm_lifelong_v1_1/month_val:0",
        "mm_lifelong_v1_1/week_test:0",
        "mm_lifelong_v1_1/day_test:0",
        "mm_lifelong_v1_1/month_val:1",
        "mm_lifelong_v1_1/week_test:1",
        "mm_lifelong_v1_1/month_val:2",
    ]
    month, week, day = rows[:3]
    catalog = (media / "day" / "catalog.json").read_bytes()
    model_catalog = media / "day" / "video_catalog.json"
    assert day["preloaded_media"] == {
        "image": "mm_lifelong_v1_1_day",
        "catalog": "/testbed/videos/catalog.json",
        "catalog_sha256": hashlib.sha256(catalog).hexdigest(),
        "files": [
            {
                "source": str(model_catalog),
                "sandbox_path": "video_catalog.json",
                "sha256": hashlib.sha256(model_catalog.read_bytes()).hexdigest(),
            }
        ],
    }
    assert "video_binding" not in day and day["setting"] == "agentic"
    # Up to 100 files: one JSON record each, in the catalog's order.
    record = json.loads(model_catalog.read_text())[0]
    assert day["responses_create_params"]["input"][0]["content"][0]["text"] == (
        "VIDEO INPUTS\n"
        + json.dumps(record)
        + "\n\nTIME COORDINATES\n"
        + prepare.TIME_COORDINATES
        + "\n\nQUESTION\n"
        + question("day_test", 0)["question"]
    )
    assert [
        json.loads(line)
        for line in month["responses_create_params"]["input"][0]["content"][0]["text"].split("\n")[1:4]
    ] == [
        {"duration_seconds": 10.0 + n, **TIMELINE["month"][n], "path": f"/testbed/videos/month/{name}"}
        for n, name in enumerate(("1.mp4", "2.mp4", "10.mp4"))
    ]
    # More: the clips per day folder, and where the full catalog is.
    assert week["responses_create_params"]["input"][0]["content"][0]["text"].startswith(
        "VIDEO INPUTS\nDay 1: 61 clips under /testbed/videos/week/day1/\nDay 2: 62 clips under /testbed/videos/week/day2/\n"
        "Read /testbed/video_catalog.json for each individual path and exact recorded-time range.\n\nTIME COORDINATES\n"
    )
    assert day["verifier_metadata"] == {
        "grading": "reference_levels",
        "question": question("day_test", 0)["question"],
        "reference_answer": "Thing 0.",
        "rubric": question("day_test", 0)["rubric"],
        "subset": "day_test",
    }
    for row in rows:
        TaskData.model_validate({**row, **row["verifier_metadata"]})
        VideoQARubricVerifyRequest.model_validate(row | {"response": response("Thing.")})
        assert "Thing " not in row["responses_create_params"]["input"][0]["content"][0]["text"]


@pytest.mark.parametrize("damage", ["count", "paths", "outside"])
def test_catalogs_that_do_not_name_the_domains_files_are_refused(release, media, damage) -> None:
    path = media / "month" / ("video_catalog.json" if damage != "outside" else "catalog.json")
    data = json.loads(path.read_text())
    if damage == "count":
        data = data[:2]
    elif damage == "paths":
        data[0]["path"] = "/testbed/videos/month/99.mp4"
    else:
        data["videos"][0]["path"] = "/testbed/videos/day/1.mp4"
        records = json.loads((media / "month" / "video_catalog.json").read_text())
        records[0]["path"] = "/testbed/videos/day/1.mp4"
        (media / "month" / "video_catalog.json").write_text(json.dumps(records))
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="must name the 3 month files|must be under"):
        prepare.prepare(data_dir=str(release), media_dir=str(media))
    with pytest.raises(ValueError, match="media_dir"):
        prepare.prepare(data_dir=str(release))


def test_the_runners_records_order_and_place_each_file() -> None:
    assert sorted(["10.mp4", "2.mp4", "1.mp4"], key=build.natural_key) == ["1.mp4", "2.mp4", "10.mp4"]
    (day,) = build.model_records("day", {"0.mp4": 84912.702667}, TIMELINE)
    assert list(day.items()) == [
        ("duration_seconds", 84912.702667),
        ("session_physical_start", "Day 1 00:00:00"),
        ("session_physical_end", "Day 1 23:00:00"),
        ("path", "/testbed/videos/day/0.mp4"),
    ]
    week = build.model_records("week", {"day2/b.mp4": 2.0, "day1/a.mp4": 1.0}, TIMELINE)
    assert [list(record) for record in week] == [
        ["duration_seconds", "day", "day_physical_start", "day_physical_end", "path"]
    ] * 2
    assert [(record["day"], record["path"]) for record in week] == [
        (1, "/testbed/videos/week/day1/a.mp4"),
        (2, "/testbed/videos/week/day2/b.mp4"),
    ]
    (month,) = build.model_records("month", {"10.mp4": 3.0}, TIMELINE)
    assert list(month) == ["duration_seconds", "video_id", "physical_start", "physical_end", "path"]
    with pytest.raises(ValueError, match="no day folder"):
        build.model_records("week", {"clip.mp4": 1.0}, TIMELINE)


def test_the_dockerfile_adds_one_read_only_layer_per_file_or_day_folder() -> None:
    assert build.dockerfile("week", ["day1/a.mp4", "day10/c.mp4", "day2/b.mp4"]).splitlines() == [
        "# syntax=docker/dockerfile:1.7",
        "ARG BASE_IMAGE",
        "FROM ${BASE_IMAGE}",
        "RUN mkdir -p -m 0555 /testbed/videos/week",
        "COPY --from=videos --chmod=0555 day1 /testbed/videos/week/day1",
        "COPY --from=videos --chmod=0555 day2 /testbed/videos/week/day2",
        "COPY --from=videos --chmod=0555 day10 /testbed/videos/week/day10",
        "COPY --chmod=0444 catalog.json /testbed/videos/catalog.json",
    ]
    assert "COPY --from=videos --chmod=0444 0.mp4 /testbed/videos/day/0.mp4" in build.dockerfile("day", ["0.mp4"])


def test_the_build_checks_every_video_and_writes_both_catalogs(tmp_path, monkeypatch) -> None:
    videos = tmp_path / "data" / "videos" / "month"
    videos.mkdir(parents=True)
    expected = {}
    for name in ("2.mp4", "10.mp4", "1.mp4"):
        content = name.encode() * 10
        (videos / name).write_bytes(content)
        expected[name] = (len(content), hashlib.sha256(content).hexdigest())
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps(TIMELINE))
    monkeypatch.setattr(build, "release_files", lambda domain: expected)
    monkeypatch.setattr(build, "TIMELINE_SHA256", hashlib.sha256(timeline.read_bytes()).hexdigest())
    monkeypatch.setattr(build, "duration", lambda path, ffprobe: float(path.stem))
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(timeline))
    args = types.SimpleNamespace(
        domain="month",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
        workers=2,
        ffprobe="ffprobe",
        no_build=True,
    )
    out = build.build(args)
    catalog = json.loads((out / "catalog.json").read_text())
    assert [video["path"] for video in catalog["videos"]] == [f"/testbed/videos/month/{n}.mp4" for n in (1, 2, 10)]
    assert catalog["videos"][0] == {
        "path": "/testbed/videos/month/1.mp4",
        "bytes": expected["1.mp4"][0],
        "sha256": expected["1.mp4"][1],
    }
    records = json.loads((out / "video_catalog.json").read_text())
    assert [record["duration_seconds"] for record in records] == [1.0, 2.0, 10.0]
    assert (out / "Dockerfile").read_text() == build.dockerfile("month", ["1.mp4", "2.mp4", "10.mp4"])
    # A video that is not the release's is refused.
    (videos / "2.mp4").write_bytes(b"x" * expected["2.mp4"][0])
    with pytest.raises(ValueError, match="differs from"):
        build.build(args)
    (videos / "2.mp4").write_bytes(b"short")
    with pytest.raises(ValueError, match="missing or not"):
        build.build(args)


def test_the_config_lends_the_rows_images_and_grades_with_the_dataset_grader(monkeypatch) -> None:
    monkeypatch.setenv("MM_LIFELONG_V1_1_WEEK_IMAGE", "registry.invalid/mml/week:1")
    values = {
        "mm_lifelong_v1_1_agentic_data_dir": "/data/mml",
        "mm_lifelong_v1_1_agentic_media_dir": "/data/media",
        "judge_api_key": "invented",  # pragma: allowlist secret
        "policy_model_name": "policy",
    }
    loaded = OmegaConf.merge(
        OmegaConf.load(ROOT / "resources_servers" / "video_qa_rubric" / "configs" / "video_qa_rubric_agentic.yaml"),
        OmegaConf.load(ROOT / "benchmarks" / "mm_lifelong_v1_1" / "grading.yaml"),
        OmegaConf.load(BENCHMARK / "config.yaml"),
        OmegaConf.create(values),
    )
    config = OmegaConf.to_container(loaded, resolve=True)
    server = VideoQARubricResourcesServerConfig(
        name="video_qa_rubric_agentic",
        port=8001,
        **config["video_qa_rubric_agentic"]["resources_servers"]["video_qa_rubric"],
    )
    assert server.sandbox_preloaded_images == {
        "mm_lifelong_v1_1_day": None,
        "mm_lifelong_v1_1_week": "registry.invalid/mml/week:1",
        "mm_lifelong_v1_1_month": None,
    }
    assert server.sandbox_config.image is None and server.video_roots == [Path("/data/media")]
    assert server.video_tools.profile.decode_timeout_seconds == 900 and server.reference_levels.max_level == 5
    assert server.judge_chat_completion_create_params.model_dump(exclude_unset=True) == {"messages": [], "n": 1}
    agent = config["video_qa_rubric_agentic_opencode"]["responses_api_agents"]["opencode_agent"]
    # The rubric's agentic OpenCode settings are kept: sh as the shell, OpenCode's own tools but webfetch, doom_loop
    # asked.
    assert server.sandbox_config.env == {"SHELL": "/bin/sh"}
    assert agent["opencode_config"]["tools"] == {"webfetch": False}
    assert agent["opencode_config"]["permission"]["doom_loop"] == "ask"
    (dataset,) = agent["datasets"]
    assert dataset["jsonl_fpath"] == str(prepare.OUTPUT_FPATH.relative_to(ROOT)) and agent["timeout"] == 7200
    environment = SingleAgentTurnLegacyEnvironmentServerConfig(
        name="video_qa_rubric_agentic_environment",
        port=8003,
        **config["video_qa_rubric_agentic_environment"]["environment_servers"]["single_agent_turn_legacy"],
    )
    # The episode ends before the sandbox's time to live, after OpenCode's own limit and the sandbox start.
    assert agent["timeout"] + server.sandbox_config.ready_timeout_s < environment.default_episode_timeout_seconds
    assert environment.default_episode_timeout_seconds < server.sandbox_config.ttl_s
    assert config["prepare_script_args"] == {"data_dir": "/data/mml", "media_dir": "/data/media"}
