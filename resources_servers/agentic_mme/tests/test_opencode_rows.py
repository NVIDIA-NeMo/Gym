# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenCode-track and server-tools-track rows that benchmarks/agentic_mme/prepare.py writes."""

import base64
import importlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from PIL import Image

from benchmarks.agentic_mme.prepare import (
    EXAMPLE_FPATH,
    EXAMPLE_OPENCODE_FPATH,
    EXAMPLE_OPENCODE_SERVER_TOOLS_FPATH,
    EXAMPLE_SERVER_TOOLS_FPATH,
    OPENCODE_SYSTEM_PROMPT,
    convert,
    opencode_row,
    opencode_server_tools_row,
    server_tools_row,
)
from nemo_gym.config import GlobalConfigDictParser
from resources_servers.agentic_mme.image_tools import IMAGE_TOOLS


ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = ROOT / "benchmarks/agentic_mme"
PNG = b"\x89PNG\r\n\x1a\nfirst"
JPEG = b"\xff\xd8\xffsecond"


def _row() -> dict:
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "Use the atomic image tools."},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Image 0."},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64," + base64.b64encode(PNG).decode(),
                        },
                        {"type": "input_text", "text": "Image 1."},
                        {
                            "type": "input_image",
                            "image_url": "data:image/jpeg;base64," + base64.b64encode(JPEG).decode(),
                        },
                        {"type": "input_text", "text": "Which image is brighter?"},
                    ],
                },
            ]
        },
        "verifier_metadata": {"task_id": "t1", "golden_answer": {"value": "0"}},
    }


def test_opencode_row_writes_images_and_names_them(tmp_path: Path) -> None:
    row = _row()
    before = json.dumps(row)

    converted = opencode_row(row, tmp_path, "t1")

    assert json.dumps(row) == before, "the atomic-tool row must not change"
    assert (tmp_path / "t1" / "0.png").read_bytes() == PNG
    assert (tmp_path / "t1" / "1.jpg").read_bytes() == JPEG
    assert converted["image_files"] == [
        {"source": "t1/0.png", "target": "images/0.png"},
        {"source": "t1/1.jpg", "target": "images/1.jpg"},
    ]
    system, user = converted["responses_create_params"]["input"]
    assert system["content"] == OPENCODE_SYSTEM_PROMPT
    # The images stay inline, and one added line names each image's file.
    assert [part["type"] for part in user["content"]].count("input_image") == 2
    assert user["content"][-1] == {
        "type": "input_text",
        "text": "Image files in the working directory: Image 0 is images/0.png, Image 1 is images/1.jpg.",
    }
    assert converted["verifier_metadata"] == row["verifier_metadata"]


@pytest.mark.parametrize(
    "image_url, task_key, message",
    [
        ("https://example.com/a.png", "t1", "data URLs"),
        ("data:image/png;base64,not base64!", "t1", "base64"),
        ("data:image/png;base64,AAAA", "../t1", "folder"),
    ],
)
def test_opencode_row_refuses_what_it_cannot_write(
    tmp_path: Path, image_url: str, task_key: str, message: str
) -> None:
    row = _row()
    row["responses_create_params"]["input"][1]["content"][1]["image_url"] = image_url
    with pytest.raises(ValueError, match=message):
        opencode_row(row, tmp_path, task_key)


def test_committed_opencode_examples_match_their_image_files() -> None:
    root = EXAMPLE_OPENCODE_FPATH.with_name("example_images")
    for line in EXAMPLE_OPENCODE_FPATH.read_text().splitlines():
        row = json.loads(line)
        inline = [
            part["image_url"]
            for message in row["responses_create_params"]["input"]
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part["type"] == "input_image"
        ]
        assert len(inline) == len(row["image_files"])
        for url, image in zip(inline, row["image_files"]):
            assert base64.b64decode(url.split(",", 1)[1]) == (root / image["source"]).read_bytes()


def test_server_tools_row_adds_the_image_tools_and_the_image_files() -> None:
    row = _row()
    before = json.dumps(row)
    image_files = [{"source": "t1/0.png", "target": "images/0.png"}]

    converted = server_tools_row(row, image_files)

    assert json.dumps(row) == before, "the atomic-tool row must not change"
    params = converted["responses_create_params"]
    # The same prompt as the Agentic-MME agent, and the same tools, one call per turn.
    assert params["input"] == row["responses_create_params"]["input"]
    assert [tool["name"] for tool in params["tools"]] == list(IMAGE_TOOLS)
    assert params["parallel_tool_calls"] is False
    assert converted["image_files"] == image_files
    assert converted["verifier_metadata"] == row["verifier_metadata"]


def test_committed_server_tools_examples_match_the_example_rows() -> None:
    examples = [json.loads(line) for line in EXAMPLE_FPATH.read_text().splitlines()]
    opencode = [json.loads(line) for line in EXAMPLE_OPENCODE_FPATH.read_text().splitlines()]
    server_tools = [json.loads(line) for line in EXAMPLE_SERVER_TOOLS_FPATH.read_text().splitlines()]
    assert server_tools == [
        server_tools_row(row, oc["image_files"]) for row, oc in zip(examples, opencode, strict=True)
    ]


def test_opencode_server_tools_row_keeps_the_prompt_and_adds_only_the_image_files() -> None:
    row = _row()
    before = json.dumps(row)
    image_files = [{"source": "t1/0.png", "target": "images/0.png"}]

    converted = opencode_server_tools_row(row, image_files)

    assert json.dumps(row) == before, "the atomic-tool row must not change"
    # OpenCode refuses request fields beyond the prompt; it gets the tools over MCP.
    assert converted["responses_create_params"] == row["responses_create_params"]
    assert converted["image_files"] == image_files
    assert converted["verifier_metadata"] == row["verifier_metadata"]


def test_committed_opencode_server_tools_examples_match_the_example_rows() -> None:
    examples = [json.loads(line) for line in EXAMPLE_FPATH.read_text().splitlines()]
    opencode = [json.loads(line) for line in EXAMPLE_OPENCODE_FPATH.read_text().splitlines()]
    rows = [json.loads(line) for line in EXAMPLE_OPENCODE_SERVER_TOOLS_FPATH.read_text().splitlines()]
    assert rows == [
        opencode_server_tools_row(row, oc["image_files"]) for row, oc in zip(examples, opencode, strict=True)
    ]


def test_a_converted_row_carries_its_task_id_at_the_top_level(tmp_path: Path) -> None:
    """`gym eval plan` reads the task id from the row's top level; the verifier keeps its copy in verifier_metadata."""
    Image.new("RGB", (2, 2), "red").save(tmp_path / "0.png")
    task = {
        "task_id": "task-7",
        "input": {"image": "0.png", "prompt": "What colour is it?"},
        "meta": {"level": 1, "domain": "art"},
        "golden_answer": {"value": "red"},
        "process_evaluation": {"checkpoints": []},
    }

    row = convert(task, tmp_path)

    assert row["task_id"] == row["verifier_metadata"]["task_id"] == "task-7"


@pytest.mark.parametrize(
    ("recipe", "agent", "dataset_name"),
    [
        ("config", "agentic_mme_agent", "agentic_mme"),
        ("server_tools", "agentic_mme_server_tools_agent", "agentic_mme_server_tools"),
        ("opencode", "agentic_mme_opencode_agent", "agentic_mme_opencode"),
        ("opencode_server_tools", "agentic_mme_opencode_agent", "agentic_mme_opencode_server_tools"),
    ],
)
def test_each_recipe_prepares_the_file_it_reads_in_the_locked_project(
    recipe: str, agent: str, dataset_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`gym eval prepare` requires prepare() to return the dataset's jsonl_fpath, and runs it in prepare_project.

    prepare.py imports Pillow and, for the server-tools rows, OpenCV through the image tools; the resources server's
    locked project has both, and a dev install of Gym has neither.
    """
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(BENCHMARKS / f"{recipe}.yaml")])
    config = OmegaConf.merge(*configs)
    (agent_config,) = config[agent].responses_api_agents.values()
    (dataset,) = [dataset for dataset in agent_config.datasets if dataset.name == dataset_name]
    assert dataset.prepare_project == "resources_servers/agentic_mme"
    assert not dataset.get("prepare_dependencies")

    preparer = importlib.import_module(".".join(Path(dataset.prepare_script).with_suffix("").parts))
    written = []
    # Every preparer converts the release once and writes all four tracks; the conversion itself is tested above.
    monkeypatch.setattr(preparer, "prepare_all_tracks", lambda: written.append(recipe), raising=False)
    output = preparer.OUTPUT_FPATH if recipe == "config" else preparer.prepare()

    assert output.resolve() == (ROOT / dataset.jsonl_fpath).resolve()
    assert written == ([] if recipe == "config" else [recipe])
