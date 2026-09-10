# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference evidence and provider limits must survive the real comparison path."""

import base64
import shutil
import zipfile
from unittest.mock import MagicMock

import fitz
import pytest
from pydantic import ValidationError

from nemo_gym.server_utils import ServerClient
from resources_servers.gdpval import app, comparison


def _block(kind="image_url", mime="image/png", data=b"0123456789ab"):
    return {"type": kind, kind: {"url": f"data:{mime};base64," + base64.b64encode(data).decode()}}


def _preflight(blocks, **limits):
    judge = comparison.Judge(name="judge", client=None, model="test", **limits)
    return comparison.preflight_judge_transport(
        judge, "Compare the submissions", {"refs": blocks, "submission_a": [], "submission_b": []}
    )


def _section(root, **kwargs):
    cleanup = []
    try:
        return comparison.build_file_section(str(root), cleanup, **kwargs)
    finally:
        for path in cleanup:
            shutil.rmtree(path)


def test_recursive_inputs_keep_paths_and_pdf_provenance(tmp_path):
    for directory, text in (("a", "first input"), ("b", "second input")):
        folder = tmp_path / directory
        folder.mkdir()
        (folder / "notes.txt").write_text(text)
    (tmp_path / "a" / "report.docx").write_bytes(b"Office source with an existing render")
    with fitz.open() as document:
        document.new_page().insert_text((40, 40), "Rendered report")
        document.save(tmp_path / "a" / "report.docx.pdf")
    with zipfile.ZipFile(tmp_path / "b" / "inputs.zip", "w") as archive:
        archive.writestr("nested/notes.txt", "archived input")
    (tmp_path / "a" / "history.json").write_text("hidden run state")

    shallow = _section(tmp_path)
    assert shallow == [{"type": "text", "text": "None"}]
    blocks = _section(tmp_path, recursive=True)
    text = "".join(block.get("text", "") for block in blocks)
    assert all(label in text for label in ("a/notes.txt", "b/notes.txt", "b/inputs.zip!/nested/notes.txt"))
    assert all(value in text for value in ("first input", "second input", "archived input"))
    assert "hidden run state" not in text
    assert text.count("a/report.docx:\n") == 1
    assert "a/report.docx.pdf:\n" not in text
    assert sum(block.get("type") == "image_url" for block in blocks) == 1
    assert _preflight(blocks)["eligible"]


def test_missing_nested_office_render_is_ineligible(tmp_path):
    folder = tmp_path / "asset"
    folder.mkdir()
    (folder / "input.docx").write_bytes(b"unrendered input")
    receipt = _preflight(_section(tmp_path, recursive=True))
    assert receipt["reasons"] == ["lossy_attachment_omission"]
    assert "asset/input.docx" in receipt["loss_markers"][0]


@pytest.mark.parametrize("name", ["input.docx", "inputs.zip"])
def test_submission_with_unrepresented_file_is_ineligible(tmp_path, name):
    (tmp_path / "submission.txt").write_text("Visible evidence must not hide an omitted attachment")
    if name.endswith(".zip"):
        with zipfile.ZipFile(tmp_path / name, "w") as archive:
            archive.writestr("nested/input.docx", b"Office source without a PDF sidecar")
    else:
        (tmp_path / name).write_bytes(b"Source without a supported representation")
    blocks = _section(tmp_path)
    receipt = comparison.preflight_judge_transport(
        comparison.Judge(name="judge", client=None, model="test"),
        "Compare the submissions",
        {"refs": [], "submission_a": blocks, "submission_b": []},
    )
    assert receipt["reasons"] == ["lossy_attachment_omission"]
    assert name in receipt["loss_markers"][0]


@pytest.mark.parametrize("archived", [False, True])
def test_unsupported_files_are_logged_and_supported_evidence_is_judged(tmp_path, caplog, archived):
    files = {"design.step": b"CAD", "poster.psd": b"8BPS\x00", "notes.txt": b"Readable submission"}
    if archived:
        files["nested.zip"] = b"Nested archive is not expanded"
        with zipfile.ZipFile(tmp_path / "submission.zip", "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
    else:
        for name, content in files.items():
            (tmp_path / name).write_bytes(content)
    with caplog.at_level("INFO", logger=comparison.LOGGER.name):
        blocks = _section(tmp_path)
    text = "".join(block.get("text", "") for block in blocks)
    assert "Readable submission" in text
    assert "design.step" not in text and "poster.psd" not in text
    assert "design.step" in caplog.text and "poster.psd" in caplog.text
    if archived:
        assert "nested.zip" not in text and "nested.zip" in caplog.text
    assert _preflight(blocks)["eligible"]


def test_omission_survives_exhausted_text_budget(tmp_path, monkeypatch):
    (tmp_path / "bad.zip").write_bytes(b"invalid archive")
    (tmp_path / "notes.txt").write_text("valid evidence" * 10)
    monkeypatch.setattr(comparison, "MAX_SECTION_TEXT_CHARS_FOR_JUDGE", 1)
    receipt = _preflight(_section(tmp_path))
    assert receipt["reasons"] == ["lossy_attachment_omission"]
    assert receipt["loss_markers"] == ["[attachment omitted from bad.zip: unreadable archive]"]


@pytest.mark.parametrize("problem", ["unsafe", "duplicate", "corrupt", "crc", "member_limit", "size"])
def test_rejected_zip_evidence_cannot_pass_preflight(tmp_path, monkeypatch, problem):
    path = tmp_path / "inputs.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("first.txt", "original-evidence")
        if problem == "unsafe":
            archive.writestr("../escaped.txt", "missing evidence")
        elif problem == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("first.txt", "different evidence")
        elif problem == "member_limit":
            archive.writestr("second.txt", "missing evidence")
    if problem == "corrupt":
        path.write_bytes(b"invalid ZIP")
    elif problem == "crc":
        path.write_bytes(path.read_bytes().replace(b"original-evidence", b"modified-evidence"))
    elif problem == "member_limit":
        monkeypatch.setattr(comparison, "MAX_ZIP_MEMBERS_FOR_JUDGE", 1)
    elif problem == "size":
        monkeypatch.setattr(comparison, "MAX_ZIP_MEMBER_BYTES_FOR_JUDGE", 1)
    receipt = _preflight(_section(tmp_path, recursive=True))
    assert "lossy_attachment_omission" in receipt["reasons"]
    assert receipt["loss_markers"]
    assert not (tmp_path.parent / "escaped.txt").exists()


@pytest.mark.parametrize(
    ("blocks", "limits", "reason"),
    [
        ([_block()], {"max_image_base64_bytes": 15}, "provider_image_byte_cap"),
        ([_block(), _block()], {"max_total_image_base64_bytes": 32}, "provider_total_image_byte_cap"),
        ([_block("video_url", "video/mp4")] * 2, {"max_video_files": 1}, "provider_video_count_cap"),
    ],
)
def test_exact_provider_limits_reject_before_dispatch(blocks, limits, reason):
    receipt = _preflight(blocks, **limits)
    assert not receipt["eligible"]
    assert receipt["reasons"] == [reason]


def test_pdf_bytes_do_not_count_as_image_bytes_and_exact_image_limit_fits():
    receipt = _preflight(
        [_block(), _block(mime="application/pdf", data=b"large native PDF" * 10)],
        max_image_base64_bytes=16,
        max_total_image_base64_bytes=17,
        max_video_files=0,
    )
    assert receipt["eligible"]
    assert receipt["total_image_base64_bytes"] == receipt["largest_image_base64_bytes"] == 16
    assert receipt["video_file_count"] == 0


def test_provider_override_cannot_bypass_image_limit():
    blocks = [{"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}]
    receipt = _preflight(
        [],
        max_image_base64_bytes=16,
        create_overrides={"messages": [{"role": "user", "content": blocks}]},
    )
    assert receipt["reasons"] == ["provider_image_size_unknown"]


@pytest.mark.parametrize(
    "limits", [{"max_image_base64_bytes": 0}, {"max_total_image_base64_bytes": -1}, {"max_video_files": -1}]
)
def test_invalid_provider_limit_configuration_fails(limits):
    with pytest.raises(ValidationError):
        app.JudgePanelMember(**limits)


@pytest.mark.asyncio
async def test_actual_selected_judge_receives_caps_and_nested_inputs(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference" / "task_task" / "repeat_0"
    for path in (candidate, reference):
        path.mkdir(parents=True)
        (path / "finish_params.json").write_text("{}")
        (path / "submission.txt").write_text("A completed submission")
    inputs = reference / "reference_files" / "asset"
    inputs.mkdir(parents=True)
    (inputs / "picture.png").write_bytes(base64.b64decode("iVBORw0KGgo="))
    config = app.GDPValResourcesServerConfig(
        name="resources",
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        reward_mode="comparison",
        reference_models={"ref": {"deliverables_dir": str(tmp_path / "reference"), "elo": 1000}},
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        judge_panel=[{"name": "limited", "max_image_base64_bytes": 1}],
        judge_reference_files_recursive=True,
        preconvert_office_to_pdf=False,
    )
    server = app.GDPValResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    body = app.GDPValVerifyRequest(
        task_id="task",
        prompt="Use the supplied image",
        deliverables_dir=str(candidate),
        responses_create_params={"input": []},
        response={
            "id": "response",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        },
    )
    monkeypatch.setattr(app, "get_server_url", lambda _: "http://localhost:9999")
    client = MagicMock()
    monkeypatch.setattr("openai.OpenAI", lambda **_: client)
    result = await server._verify_comparison(body)
    assert result.model_dump()["_ng_failure_class"] == "transport_ineligible"
    client.chat.completions.create.assert_not_called()
