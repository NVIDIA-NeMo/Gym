# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import fitz
import openpyxl
import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.gdpval import comparison
from resources_servers.gdpval.app import GDPValResourcesServer, GDPValResourcesServerConfig, GDPValVerifyRequest


def _text(blocks: list[dict]) -> str:
    return "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")


def _attachments(blocks: list[dict]) -> list[bytes]:
    return [
        base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        for block in blocks
        if block.get("type") == "image_url"
    ]


def _section(path: Path, **kwargs) -> list[dict]:
    cleanup: list[Path] = []
    try:
        return comparison.build_file_section(str(path), cleanup, **kwargs)
    finally:
        comparison.clean_up_paths(cleanup)


def _input_tree(root: Path) -> bytes:
    nested = root / "folder"
    nested.mkdir(parents=True)
    (root / "brief.txt").write_text("TOP_LEVEL_INPUT")
    (nested / "notes.txt").write_text("NESTED_INPUT_TEXT")
    with ZipFile(nested / "archive.zip", "w") as archive:
        archive.writestr("inner/member.txt", "NESTED_ARCHIVE_TEXT")
    (nested / "Plan.docx").write_bytes(b"Office source with a preconverted PDF")
    with fitz.open() as document:
        document.new_page().insert_text((72, 72), "Reference document")
        pdf = document.tobytes()
    (nested / "Plan.docx.pdf").write_bytes(pdf)
    return pdf


def test_nested_inputs_include_text_zip_and_office_sidecar_once(tmp_path: Path) -> None:
    pdf = _input_tree(tmp_path)

    shallow = _section(tmp_path)
    recursive = _section(tmp_path, recursive=True)

    assert _text(shallow).count("TOP_LEVEL_INPUT") == 1
    assert "NESTED_INPUT_TEXT" not in _text(shallow)
    assert "NESTED_ARCHIVE_TEXT" not in _text(shallow)
    assert _attachments(shallow) == []
    text = _text(recursive)
    assert text.count("TOP_LEVEL_INPUT") == 1
    assert text.count("NESTED_INPUT_TEXT") == 1
    assert text.count("NESTED_ARCHIVE_TEXT") == 1
    assert "folder/notes.txt:" in text
    assert "folder/archive.zip!/inner/member.txt:" in text
    assert text.count("folder/Plan.docx:") == 1
    assert "Plan.docx.pdf:" not in text
    assert _attachments(recursive) == [pdf]


def test_flat_inputs_and_root_zip_payloads_are_unchanged(tmp_path: Path) -> None:
    (tmp_path / "input.txt").write_text("PLAIN_INPUT")
    # Preserve the existing root ZIP order, including when it is not alphabetical.
    for name in ("z", "a"):
        with ZipFile(tmp_path / f"{name}.zip", "w") as archive:
            archive.writestr(f"inner/{name}.txt", f"ARCHIVE_{name}")

    shallow = _section(tmp_path)

    assert "ARCHIVE_z" in _text(shallow)
    assert "ARCHIVE_a" in _text(shallow)
    assert _section(tmp_path, recursive=True) == shallow


@pytest.mark.parametrize("media_mode", ["native_pdf", "images_and_text"])
def test_xlsx_without_pdf_keeps_existing_structured_text(tmp_path: Path, media_mode: str) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Forecast"
    sheet["A1"] = "BASELINE_SPREADSHEET"
    sheet["A2"] = 12
    sheet["A3"] = "=A2*2"
    workbook.save(tmp_path / "Budget.xlsx")
    workbook.close()

    shallow = _section(tmp_path, media_mode=media_mode)
    recursive = _section(tmp_path, media_mode=media_mode, recursive=True)

    assert recursive == shallow
    assert _attachments(shallow) == []
    text = _text(shallow)
    assert "structured spreadsheet cells" in text
    assert "Sheet: Forecast" in text
    assert "BASELINE_SPREADSHEET" in text
    assert "A3: formula: =A2*2" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("recursive", [None, False, True], ids=["default", "disabled", "enabled"])
async def test_verify_only_recurses_into_reference_inputs(tmp_path: Path, recursive: bool | None) -> None:
    reference_root = tmp_path / "reference"
    reference = reference_root / "task_task-1" / "repeat_0"
    candidate = tmp_path / "candidate" / "task_task-1" / "repeat_0"
    for directory, label in ((reference, "REFERENCE"), (candidate, "CANDIDATE")):
        (directory / "nested").mkdir(parents=True)
        (directory / "finish_params.json").write_text("{}")
        (directory / "answer.txt").write_text(f"{label}_ANSWER")
        (directory / "nested" / "hidden.txt").write_text(f"{label}_NESTED_SUBMISSION")
    pdf = _input_tree(reference / "reference_files")
    options = {} if recursive is None else {"judge_reference_files_recursive": recursive}
    config = GDPValResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="",
        name="",
        reward_mode="comparison",
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        reference_deliverables_dir=str(reference_root),
        preconvert_office_to_pdf=False,
        num_comparison_trials=1,
        **options,
    )
    assert config.judge_reference_files_recursive is bool(recursive)
    server = GDPValResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    body = GDPValVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=NeMoGymResponse(
            id="resp-1",
            created_at=0.0,
            model="policy",
            object="response",
            output=[],
            status="completed",
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ),
        task_id="task-1",
        prompt="Compare the submissions against the provided inputs.",
        rubric_json=None,
        rubric_pretty="",
        deliverables_dir=str(candidate),
    )
    with (
        patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
        patch("openai.OpenAI", return_value=MagicMock()),
        patch(
            "resources_servers.gdpval.comparison.run_trials",
            return_value={"winner": "[[B]]", "win_count_a": 0, "win_count_b": 1, "tie_count": 0, "task_count": 1},
        ) as dispatch,
    ):
        result = await server.verify(body)

    assert result.total_wins == 1
    dispatch.assert_called_once()
    sent = dispatch.call_args.kwargs
    inputs = _text(sent["refs"])
    assert inputs.count("TOP_LEVEL_INPUT") == 1
    assert inputs.count("NESTED_INPUT_TEXT") == int(bool(recursive))
    assert inputs.count("NESTED_ARCHIVE_TEXT") == int(bool(recursive))
    assert _attachments(sent["refs"]) == ([pdf] if recursive else [])
    assert "REFERENCE_ANSWER" in _text(sent["submission_a"])
    assert "CANDIDATE_ANSWER" in _text(sent["submission_b"])
    for section in (sent["submission_a"], sent["submission_b"]):
        text = _text(section)
        assert "NESTED_SUBMISSION" not in text
        assert "TOP_LEVEL_INPUT" not in text
        assert "NESTED_INPUT_TEXT" not in text
    for sections in sent["sections_by_judge"].values():
        assert sections["refs"] == sent["refs"]
