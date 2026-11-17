# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import runpy
import sys
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset

from benchmarks.vials.harbor.adapter import HF_CONFIG, HF_REPO, HF_REVISION, HF_SPLIT, TEMPLATE_DIR, prepare


def source_rows() -> list[dict]:
    return [
        {
            "id": task_id,
            "question": "How many cells in A &amp; B? ${literal}",
            "answer": "REFERENCE_ONLY_SENTINEL",
            "domain": "test",
            "images": [{"bytes": payload, "path": None} for payload in images],
            "metadata": {
                "image_filenames": [f"panel_{n}.png" for n in range(len(images))],
                "answer_lower_bound": 11.0,
                "answer_upper_bound": 13.0,
                "secondary_tags": ["tag"],
            },
        }
        for task_id, images in [("single", [b"first"]), ("multi", [b"second", b"third"])]
    ]


def grader():
    spec = importlib.util.spec_from_file_location("vials_harbor_grader", TEMPLATE_DIR / "tests/grade.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {}), patch("sys.path", [str(TEMPLATE_DIR / "tests"), *sys.path]):
        spec.loader.exec_module(module)
    return module


def export(tmp_path: Path, rows: list[dict] | None = None, **kwargs: object) -> Path:
    dataset = Dataset.from_list(source_rows() if rows is None else rows)
    with patch("datasets.load_dataset", return_value=dataset) as load:
        index = prepare(output_dir=tmp_path / "harbor", docker_image="example/vials:tested", **kwargs)
    load.assert_called_once_with(HF_REPO, HF_CONFIG, split=HF_SPLIT, revision=HF_REVISION)
    return index


def test_direct_hf_export_keeps_panels_bounds_and_reference_private(tmp_path: Path) -> None:
    index = export(tmp_path)
    entries = [{"task_name": path.name} for path in index.glob("vials-*")]
    assert [r["task_name"] for r in entries] == ["vials-single", "vials-multi"]
    task = index / "vials-multi"
    payload = json.loads((task / "tests/verifier_input.json").read_text())
    assert payload["question"] == "How many cells in A & B? ${literal}"
    assert payload["expected_answer"] == "REFERENCE_ONLY_SENTINEL"
    assert payload["answer_lower_bound"] == 11 and payload["answer_upper_bound"] == 13
    assert payload["images"] == ["images/image_1.png", "images/image_2.png"]
    assert "responses_create_params" not in payload
    for name, expected in [("image_1.png", b"second"), ("image_2.png", b"third")]:
        assert (task / "environment/input" / name).read_bytes() == expected
        assert (task / "tests/images" / name).read_bytes() == expected
    # Agent edits cannot mutate the verifier's image copies.
    (task / "environment/input/image_1.png").write_bytes(b"edited")
    assert (task / "tests/images/image_1.png").read_bytes() == b"second"
    instruction = (task / "instruction.md").read_text()
    assert instruction.index("/task/image_1.png") < instruction.index("/task/image_2.png")
    assert "${literal}" in instruction and "/task/answer.txt" in instruction
    assert "REFERENCE_ONLY_SENTINEL" not in instruction
    for path in (task / "environment").rglob("*"):
        if path.is_file():
            assert b"REFERENCE_ONLY_SENTINEL" not in path.read_bytes()
    config = tomllib.loads((task / "task.toml").read_text())
    assert config["verifier"]["environment_mode"] == "separate"
    assert config["environment"]["network_mode"] == "public"
    assert config["verifier"]["env"]["VIALS_JUDGE_API_KEY"] == "${VIALS_JUDGE_API_KEY}"
    assert config["artifacts"] == ["/task/answer.txt"]
    assert not list(task.rglob("*.pyc"))
    assert (task / "tests/test.sh").stat().st_mode & 0o111
    assert sorted(p.name for p in tmp_path.iterdir()) == ["harbor"]
    assert (index / "dataset.toml").is_file()
    assert "REFERENCE_ONLY_SENTINEL" in (task / "solution/solve.sh").read_text()
    assert not (index / "gym.jsonl").exists()


def test_native_gym_loads_export_and_materializes_task_routes(tmp_path: Path) -> None:
    from nemo_gym.data.tasks.harbor import context_digest, load_task
    from nemo_gym.data.tasks.harbor.materialize import materialize_task, run_config
    from nemo_gym.data.tasks.harbor.records import write_json_record

    folder = export(tmp_path)
    builds = {}
    for task in folder.glob("vials-*"):
        for directory, service in [("environment", "main"), ("tests", "verifier")]:
            builds[context_digest(task / directory)] = {"services": {service: {"image": "example/unit-test:local"}}}
    write_json_record(folder / "built-images.json", builds)
    tasks = [load_task(path) for path in folder.glob("vials-*")]
    assert len(tasks) == 2
    for task in tasks:
        row = materialize_task(task, "vials")
        assert row.task_id.taskset == "vials"
        assert task.config.verifier.environment_mode == "separate"
        assert (task.path / "solution/solve.sh").is_file()
    config = run_config(
        folder=folder,
        tasks=tasks,
        rows_path=tmp_path / "rows.jsonl",
        taskset="vials",
        agent_instance_name="direct_vision_agent",
        agent_impl="direct_vision_agent",
    )
    assert config["environment_routing_mode"] == "taskset"
    assert (
        config["harbor_vials_environment"]["environment_servers"]["harbor"]["agent_server"]["name"]
        == "harbor_vials_agent"
    )


def test_overwrite_and_subset_are_explicit(tmp_path: Path) -> None:
    index = export(tmp_path)
    marker = index / "vials-single/marker"
    marker.touch()
    with pytest.raises(FileExistsError):
        export(tmp_path)
    with pytest.raises(ValueError, match="outside this selection"):
        export(tmp_path, limit=1, overwrite=True)
    assert marker.exists()
    export(tmp_path, overwrite=True)
    assert not marker.exists()


@pytest.mark.parametrize(
    "change", ["id", "duplicate", "question", "answer", "images", "image_bytes", "count", "bounds", "reversed_bounds"]
)
def test_invalid_rows_fail_before_replacing_existing_tasks(tmp_path: Path, change: str) -> None:
    index = export(tmp_path)
    marker = index / "vials-single/marker"
    marker.touch()
    rows = source_rows()
    if change == "id":
        rows[1]["id"] = "../escape"
    elif change == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif change in ("question", "answer"):
        rows[1][change] = ""
    elif change == "images":
        rows[1]["images"] = []
    elif change == "image_bytes":
        rows[1]["images"][0]["bytes"] = None
    elif change == "count":
        rows[1]["metadata"]["image_filenames"] = []
    elif change == "bounds":
        rows[1]["metadata"]["answer_upper_bound"] = float("inf")
    else:
        rows[1]["metadata"]["answer_lower_bound"] = 14
    with pytest.raises(ValueError):
        export(tmp_path, rows, overwrite=True)
    assert marker.exists()


def test_empty_dataset_fails(tmp_path: Path) -> None:
    with patch("datasets.load_dataset", return_value=Dataset.from_dict({"id": []})):
        with pytest.raises(ValueError, match="nonempty"):
            prepare(output_dir=tmp_path / "harbor", docker_image="vials:local")
    assert not (tmp_path / "harbor").exists()


def test_symlink_task_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "harbor"
    output.mkdir()
    (output / "vials-single").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        export(tmp_path, overwrite=True)


@pytest.mark.parametrize("options", [{"limit": 0}, {"docker_image": "bad\nRUN echo injected"}])
def test_invalid_options_do_not_download(options: dict) -> None:
    with patch("datasets.load_dataset") as load:
        with pytest.raises(ValueError):
            prepare(**({"docker_image": "vials:local"} | options))
    load.assert_not_called()


def test_cli_loads_hf_directly_with_limit(tmp_path: Path) -> None:
    with (
        patch("datasets.load_dataset", return_value=Dataset.from_list(source_rows())),
        patch(
            "sys.argv",
            ["adapter.py", "--output-dir", str(tmp_path / "harbor"), "--docker-image", "vials:local", "--limit", "1"],
        ),
    ):
        runpy.run_path(str(TEMPLATE_DIR.parent / "adapter.py"), run_name="__main__")
    assert len(list((tmp_path / "harbor").glob("vials-*"))) == 1
    assert (tmp_path / "harbor/vials-single/tests/images/image.png").read_bytes() == b"first"
