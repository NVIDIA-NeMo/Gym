# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import importlib.util
import io
import json
import os
import runpy
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.vials.harbor.adapter import TEMPLATE_DIR, prepare
from benchmarks.vials.prepare import GYM_ROOT
from nemo_gym.tasks.harbor.models import HarborTaskConfig
from resources_servers.vials.app import VialsVerifyRequest
from resources_servers.vials.verifier_fixture import create_fixture_verifier, invoke_fixture


def source_file(tmp_path: Path) -> Path:
    rows = []
    for task_id, images in (("single", [b"first"]), ("multi", [b"second", b"third"])):
        paths = []
        for index, image in enumerate(images):
            path = tmp_path / f"{task_id}-{index}.png"
            path.write_bytes(image)
            paths.append(str(path))
        rows.append(
            {
                "task_id": task_id,
                "question": "How many cells are visible? ${literal}",
                "expected_answer": "REFERENCE_ONLY_SENTINEL",
                "answer_lower_bound": 11.0,
                "answer_upper_bound": 13.0,
                "image_paths": paths,
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                *[
                                    {
                                        "type": "input_image",
                                        "detail": "auto",
                                        "image_url": "data:image/png;base64," + base64.b64encode(b).decode(),
                                    }
                                    for b in images
                                ],
                                {"type": "input_text", "text": "How many cells are visible?"},
                            ],
                        }
                    ]
                },
            }
        )
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def grader():
    spec = importlib.util.spec_from_file_location("vials_harbor_grader", TEMPLATE_DIR / "tests" / "grade.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export(tmp_path: Path, **kwargs: object) -> Path:
    return prepare(
        source=source_file(tmp_path),
        output_dir=tmp_path / "harbor",
        docker_image="example/vials:tested",
        verifier_url="http://gym-verifier:1234/verify",
        **kwargs,
    )


def test_export_preserves_panels_bounds_and_private_verifier_data(tmp_path: Path) -> None:
    index = export(tmp_path)
    assert [json.loads(line)["task_name"] for line in index.read_text().splitlines()] == [
        "vials-single",
        "vials-multi",
    ]
    task = index.parent / "vials-multi"
    assert (task / "environment/input/image_1.png").read_bytes() == b"second"
    assert (task / "environment/input/image_2.png").read_bytes() == b"third"
    instruction = (task / "instruction.md").read_text()
    assert instruction.index("/task/image_1.png") < instruction.index("/task/image_2.png")
    assert "${literal}" in instruction
    assert "/task/answer.txt" in instruction
    config = tomllib.loads((task / "task.toml").read_text())
    HarborTaskConfig.model_validate(config)
    assert config["environment"]["allowed_hosts"] == ["gym-verifier"]
    assert config["verifier"]["env"]["VIALS_VERIFIER_URL"] == "http://gym-verifier:1234/verify"
    payload = json.loads((task / "tests/verifier_input.json").read_text())
    assert payload["expected_answer"] == "REFERENCE_ONLY_SENTINEL"
    assert payload["answer_lower_bound"] == 11
    assert payload["answer_upper_bound"] == 13
    assert len(payload["responses_create_params"]["input"][0]["content"]) == 3
    dockerfile = (task / "environment/Dockerfile").read_text()
    assert "FROM example/vials:tested" in dockerfile
    assert "COPY --chown=agent:agent input/ /task/" in dockerfile
    assert "/tests" not in dockerfile
    for path in (task / "environment").rglob("*"):
        if path.is_file():
            assert b"REFERENCE_ONLY_SENTINEL" not in path.read_bytes()
    assert "REFERENCE_ONLY_SENTINEL" not in instruction
    assert (task / "tests/test.sh").stat().st_mode & 0o111
    assert not list(task.rglob("*.pyc"))


def test_export_loads_in_native_harbor(tmp_path: Path) -> None:
    """Exercise Harbor's own parser when its optional runtime is installed."""
    harbor_task = pytest.importorskip("harbor.models.task.task")
    index = export(tmp_path)
    for line in index.read_text().splitlines():
        row = json.loads(line)
        task = harbor_task.Task(index.parent / row["task_name"])
        assert task.name == f"vials/{row['task_name']}"
        assert task.config.agent.user == "agent"
        assert task.config.verifier.user == "root"
        assert task.config.environment.network_mode.value == "allowlist"
        assert task.config.environment.allowed_hosts == ["gym-verifier"]


def test_missing_source_reuses_direct_download(tmp_path: Path) -> None:
    source = tmp_path / "download.jsonl"

    def download(*, output: Path) -> Path:
        output.write_bytes(source_file(tmp_path).read_bytes())
        return output

    with patch("benchmarks.vials.harbor.adapter.prepare_direct", side_effect=download) as download_mock:
        path = prepare(
            source=source,
            output_dir=tmp_path / "harbor",
            docker_image="vials:local",
            verifier_url="http://gym/verify",
            limit=1,
        )
    download_mock.assert_called_once_with(output=source)
    assert len(path.read_text().splitlines()) == 1
    assert (path.parent / "vials-single/environment/input/image.png").read_bytes() == b"first"


def test_overwrite_is_explicit_and_does_not_delete_unselected_tasks(tmp_path: Path) -> None:
    index = export(tmp_path)
    marker = index.parent / "vials-single/marker"
    marker.touch()
    with pytest.raises(FileExistsError):
        export(tmp_path)
    assert marker.exists()
    with pytest.raises(ValueError, match="outside this selection"):
        export(tmp_path, overwrite=True, limit=1)
    assert marker.exists()
    export(tmp_path, overwrite=True)
    assert not marker.exists()


@pytest.mark.parametrize("task_id", ["../escape", "/absolute", "", "bad.name"])
def test_invalid_task_ids_cannot_escape_output(tmp_path: Path, task_id: str) -> None:
    source = source_file(tmp_path)
    row = json.loads(source.read_text().splitlines()[0])
    row["task_id"] = task_id
    source.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Invalid VIALS task ID"):
        prepare(
            source=source, output_dir=tmp_path / "harbor", docker_image="vials:local", verifier_url="http://gym/verify"
        )
    assert not (tmp_path / "harbor").exists()


@pytest.mark.parametrize(
    "url", ["https://key@host/verify", "ftp://host/verify", "http://host/not-verify", "http://host/verify?key=secret"]
)
def test_invalid_verifier_urls_are_rejected(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="verifier_url"):
        prepare(source=tmp_path / "unused", docker_image="vials:local", verifier_url=url)


@pytest.mark.parametrize("candidate,reward", [("twelve", 1.0), ("eleven", 0.0)])
def test_grader_uses_native_gym_contract_and_scoring(tmp_path: Path, candidate: str, reward: float) -> None:
    index = export(tmp_path)
    answer = tmp_path / "answer.txt"
    answer.write_text(f"<think>private reasoning</think>\nANSWER: {candidate}")
    module = grader()
    captured = []

    def verify(request, **kwargs):
        assert request.full_url == "http://gym-verifier:1234/verify"
        body = VialsVerifyRequest.model_validate(json.loads(request.data))
        captured.append(body)
        result = asyncio.run(invoke_fixture(create_fixture_verifier(), body))
        return io.BytesIO(result.model_dump_json().encode())

    with patch.object(module, "urlopen", side_effect=verify):
        result = module.grade(
            answer_path=answer,
            input_path=index.parent / "vials-single/tests/verifier_input.json",
            output_dir=tmp_path / "logs",
            verifier_url="http://gym-verifier:1234/verify",
        )
    assert result["reward"] == reward
    assert len(captured) == 1
    assert json.loads((tmp_path / "logs/reward.json").read_text()) == {"reward": reward}


def test_missing_images_fail_before_replacing_existing_export(tmp_path: Path) -> None:
    index = export(tmp_path)
    marker = index.parent / "vials-single/marker"
    marker.touch()
    (tmp_path / "multi-1.png").unlink()
    with pytest.raises(FileNotFoundError, match="image does not exist"):
        prepare(
            source=tmp_path / "source.jsonl",
            output_dir=index.parent,
            docker_image="vials:local",
            verifier_url="http://gym/verify",
            overwrite=True,
        )
    assert marker.exists()


@pytest.mark.parametrize("change", ["duplicate", "empty", "question", "bounds", "images"])
def test_invalid_sources_do_not_create_partial_task_trees(tmp_path: Path, change: str) -> None:
    source = source_file(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "empty":
        rows = []
    elif change == "question":
        rows[0]["question"] = ""
    elif change == "bounds":
        rows[0]["answer_lower_bound"] = "Infinity"
    else:
        rows[0]["image_paths"] = []
    source.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        prepare(
            source=source, output_dir=tmp_path / "harbor", docker_image="vials:local", verifier_url="http://gym/verify"
        )
    assert not (tmp_path / "harbor").exists()


def test_symlink_task_is_never_overwritten(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    output = tmp_path / "harbor"
    output.mkdir()
    (output / "vials-single").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare(
            source=source,
            output_dir=output,
            docker_image="vials:local",
            verifier_url="http://gym/verify",
            overwrite=True,
        )
    assert source.exists()


def test_cli_exports_relative_source_images(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for row in rows:
        row["image_paths"] = [os.path.relpath(path, GYM_ROOT) for path in row["image_paths"]]
    source.write_text("\n\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "harbor"
    with patch(
        "sys.argv",
        [
            "adapter.py",
            "--source",
            str(source),
            "--output-dir",
            str(output),
            "--docker-image",
            "vials:local",
            "--verifier-url",
            "http://gym/verify",
            "--limit",
            "1",
        ],
    ):
        runpy.run_path(str(TEMPLATE_DIR.parent / "adapter.py"), run_name="__main__")
    assert len((output / "gym.jsonl").read_text().splitlines()) == 1
    assert (output / "vials-single/environment/input/image.png").read_bytes() == b"first"


@pytest.mark.parametrize("settings", [{"limit": 0}, {"docker_image": "image\nRUN echo injected"}])
def test_invalid_export_settings_fail_before_download(settings: dict) -> None:
    options = {"docker_image": "vials:local", "verifier_url": "http://gym/verify"} | settings
    with patch("benchmarks.vials.harbor.adapter.prepare_direct") as download:
        with pytest.raises(ValueError):
            prepare(**options)
    download.assert_not_called()


def test_missing_answer_scores_zero_without_calling_judge(tmp_path: Path) -> None:
    module = grader()
    with patch.object(module, "urlopen") as call:
        result = module.grade(
            answer_path=tmp_path / "absent",
            input_path=tmp_path / "unused",
            output_dir=tmp_path / "logs",
            verifier_url="http://gym/verify",
        )
    call.assert_not_called()
    assert result["reward"] == 0


@pytest.mark.parametrize("result", [{"mask_sample": True, "failure_reason": "judge unavailable"}, {"reward": 0.5}])
def test_verifier_failures_are_not_silently_scored_zero(tmp_path: Path, result: dict) -> None:
    module = grader()
    answer = tmp_path / "answer.txt"
    answer.write_text("ANSWER: twelve")
    payload = tmp_path / "payload.json"
    payload.write_text("{}")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "reward.json").write_text('{"reward":1}')
    with patch.object(module, "urlopen", return_value=io.BytesIO(json.dumps(result).encode())):
        with pytest.raises((ValueError, RuntimeError)):
            module.grade(answer_path=answer, input_path=payload, output_dir=logs, verifier_url="http://gym/verify")
    assert not (logs / "reward.json").exists()
