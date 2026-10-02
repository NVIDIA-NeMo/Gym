# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fitz
import pytest

from resources_servers.aa_briefcase_lite.app import (
    AABriefcaseLiteResourcesServer,
    AABriefcaseLiteResourcesServerConfig,
)
from resources_servers.gdpval.judge_panel import ResolvedJudge


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(AABriefcaseLiteResourcesServer, "model_post_init", lambda self, context: None)
    return AABriefcaseLiteResourcesServer.model_construct(
        config=AABriefcaseLiteResourcesServerConfig.model_construct(dataset_dir="unused")
    )


async def test_binary_inserts_artifacts_inside_released_template(server, monkeypatch):
    # Exact public template from ArtificialAnalysis/AA-Briefcase-Lite at
    # 4dec557b47d43867a1648c0974db1d8208c8b677, prompts/judge_user.txt.
    template = (Path(__file__).parent / "fixtures" / "judge_user.txt").read_text()
    server._aa_binary_user = template
    server._aa_binary_system = "Judge the artifact."
    check = {
        "check_id": "test-check",
        "check_description": "Check",
        "score_1_criteria": "Pass",
        "score_0_criteria": "Fail",
    }
    artifact = {"type": "image_url", "image_url": {"url": "data:application/pdf;base64,cGRm"}}
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"passed":true,"reasoning":"Observed"}'))]
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    monkeypatch.setattr("resources_servers.aa_briefcase_lite.app.AsyncOpenAI", lambda **kwargs: client)
    judge = ResolvedJudge(name="judge", model="model", base_url="http://judge.invalid/v1")

    await server._binary_call(judge, "Task", check, [artifact])

    blocks = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert len(blocks) == 3
    assert blocks[1] == artifact
    assert "<submitted_artifact>" in blocks[0]["text"]
    assert "</submitted_artifact>" not in blocks[0]["text"]
    assert blocks[2] == {"type": "text", "text": "\n</submitted_artifact>\n"}
    reconstructed = blocks[0]["text"] + "<<<SUBMISSION CONTENT MESSAGES>>>" + blocks[2]["text"]
    assert reconstructed == template.format(task_markdown="Task", **check)


@pytest.mark.parametrize("media_mode", ["native_pdf", "images_and_text"])
@pytest.mark.parametrize("sidecar", [False, True])
async def test_tex_source_and_rendered_pdf_both_reach_judge(server, tmp_path, media_mode, sidecar):
    source = r"\documentclass{article}\begin{document}Visible source\end{document}"
    (tmp_path / "report.tex").write_text(source)
    with fitz.open() as doc:
        doc.new_page().insert_text((72, 72), "Visible rendering")
        doc.save(tmp_path / "report.pdf")
    if sidecar:
        with fitz.open() as doc:
            doc.new_page().insert_text((72, 72), "Preferred sidecar rendering")
            doc.save(tmp_path / "report.tex.pdf")
    judge = ResolvedJudge(name="judge", model="model", base_url="http://judge.invalid/v1", media_mode=media_mode)

    blocks = await server._section(tmp_path, judge, [])

    assert any(source in block.get("text", "") for block in blocks)
    assert any(block["type"] != "text" for block in blocks)
    assert "report.tex (LaTeX source)" in str(blocks)
    pdf_name = "report.tex.pdf" if sidecar else "report.pdf"
    heading = {"type": "text", "text": f"\n{pdf_name} (submitted alongside report.tex):\n"}
    content_index = blocks.index(heading) + 1
    rendered = blocks[content_index + (media_mode == "images_and_text")]
    assert rendered["type"] == "image_url"
    if media_mode == "native_pdf":
        payload = base64.b64decode(rendered["image_url"]["url"].split(",", 1)[1])
        assert payload == (tmp_path / pdf_name).read_bytes()
    else:
        text = "Preferred sidecar rendering" if sidecar else "Visible rendering"
        assert text in blocks[content_index]["text"]


async def test_tex_without_render_does_not_claim_pdf_was_attached(server, tmp_path):
    (tmp_path / "report.tex").write_text("Visible source")
    judge = ResolvedJudge(name="judge", model="model", base_url="http://judge.invalid/v1")

    blocks = await server._section(tmp_path, judge, [])

    assert "report.pdf" not in str(blocks)
    assert "Visible source" in str(blocks)
    assert all(block["type"] == "text" for block in blocks)
