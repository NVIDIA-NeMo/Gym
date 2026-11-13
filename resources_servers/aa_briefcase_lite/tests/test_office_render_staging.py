# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An Office deliverable must reach the judge as a rendering, never as a filename."""

from contextlib import ExitStack

import pytest

from resources_servers.aa_briefcase_lite.app import (
    AABriefcaseLiteResourcesServer,
    AABriefcaseLiteResourcesServerConfig,
    _stage_submission,
)


@pytest.fixture(autouse=True)
def _no_dataset_on_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AABriefcaseLiteResourcesServer, "model_post_init", lambda self, context: None)


def _deliverable(tmp_path, render=None):
    (tmp_path / "budget.xlsx").write_bytes(b"xlsx")
    if render is not None:
        (tmp_path / render).write_bytes(b"%PDF-1.4")
    return str(tmp_path)


def _server(preconvert):
    return AABriefcaseLiteResourcesServer.model_construct(
        config=AABriefcaseLiteResourcesServerConfig.model_construct(
            dataset_dir="unused", preconvert_office_to_pdf=preconvert
        )
    )


@pytest.mark.parametrize("render", ["budget.pdf", "budget.xlsx.pdf"])
def test_an_existing_render_is_staged_when_we_do_not_convert(tmp_path, render):
    source = _deliverable(tmp_path, render)
    with ExitStack() as stack:
        stage, missing = _stage_submission(source, ["budget.xlsx"], stack, carry_renders=True)
        assert missing == []
        assert sorted(path.name for path in stage.iterdir()) == sorted(["budget.xlsx", render])


def test_a_missing_render_raises_instead_of_stubbing_the_judge(tmp_path):
    source = _deliverable(tmp_path)
    with ExitStack() as stack:
        with pytest.raises(RuntimeError, match="filename-only stub"):
            _stage_submission(source, ["budget.xlsx"], stack, carry_renders=True)


def test_a_render_shared_by_two_office_sources_raises(tmp_path):
    # ``plan.pdf`` cannot be attributed to either source, so there is no usable render.
    (tmp_path / "plan.pptx").write_bytes(b"pptx")
    (tmp_path / "plan.xlsx").write_bytes(b"xlsx")
    (tmp_path / "plan.pdf").write_bytes(b"%PDF-1.4")
    with ExitStack() as stack:
        with pytest.raises(RuntimeError, match="filename-only stub"):
            _stage_submission(str(tmp_path), ["plan.pptx"], stack, carry_renders=True)


def test_an_unrelated_pdf_is_never_staged(tmp_path):
    source = _deliverable(tmp_path, "budget.pdf")
    (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.4")
    with ExitStack() as stack:
        stage, _missing = _stage_submission(source, ["budget.xlsx"], stack, carry_renders=True)
        assert "notes.pdf" not in {path.name for path in stage.iterdir()}


def test_a_model_written_pdf_never_replaces_our_conversion(tmp_path):
    source = _deliverable(tmp_path, "budget.pdf")
    with ExitStack() as stack:
        stage, _missing = _stage_submission(source, ["budget.xlsx"], stack, carry_renders=False)
        assert [path.name for path in stage.iterdir()] == ["budget.xlsx"]


def test_a_requested_pdf_is_still_staged_when_we_convert(tmp_path):
    source = _deliverable(tmp_path, "budget.pdf")
    with ExitStack() as stack:
        stage, _missing = _stage_submission(source, ["budget.xlsx", "budget.pdf"], stack, carry_renders=False)
        assert sorted(path.name for path in stage.iterdir()) == ["budget.pdf", "budget.xlsx"]


def test_carry_renders_follows_the_conversion_setting():
    assert _server(preconvert=False)._carry_renders is True
    assert _server(preconvert=True)._carry_renders is False


def test_the_server_refuses_to_start_without_libreoffice(monkeypatch):
    import resources_servers.gdpval.setup_libreoffice as setup_libreoffice

    monkeypatch.setattr(setup_libreoffice, "ensure_libreoffice", lambda: False)
    with pytest.raises(RuntimeError, match="libreoffice"):
        _server(preconvert=True)._require_office_rendering()


def test_no_refusal_when_we_do_not_convert(monkeypatch):
    import resources_servers.gdpval.setup_libreoffice as setup_libreoffice

    monkeypatch.setattr(setup_libreoffice, "ensure_libreoffice", lambda: False)
    assert _server(preconvert=False)._require_office_rendering() is None
