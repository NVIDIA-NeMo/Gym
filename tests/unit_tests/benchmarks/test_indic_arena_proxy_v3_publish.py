# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publication must preserve source bytes and exclude unrelated local files."""

import json

import pytest

from benchmarks.indic.arena_proxy_v3 import publish_dataset


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(publish_dataset, "LANGUAGES", ["en"])
    monkeypatch.setattr(
        publish_dataset, "SOURCE_FILES", ["lmarena_v3_chat400_en.jsonl", "references/gpt-5.4-mini/en.jsonl"]
    )
    source = tmp_path / "source"
    (source / "references/gpt-5.4-mini").mkdir(parents=True)
    prompts = [{"question_id": str(i), "question": f"Question {i}", "metadata": {"chat400": {}}} for i in range(400)]
    refs = [
        {"question_id": str(i), "question": f"Question {i}", "language": "en", "generation": "Answer", "status": "ok"}
        for i in range(400)
    ]
    for name, rows in zip(publish_dataset.SOURCE_FILES, [prompts, refs], strict=True):
        (source / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (source / "private.key").write_text("not for upload")
    (source / "references/gpt-5.4-mini/en.jsonl.partial").write_text("not for upload")
    return source


def test_stage_preserves_bytes_and_allowlist(source, tmp_path):
    output = tmp_path / "staged"
    manifest = publish_dataset.stage_dataset(source, output)
    assert set(manifest["files"]) == set(publish_dataset.SOURCE_FILES)
    for name in publish_dataset.SOURCE_FILES:
        assert (output / name).read_bytes() == (source / name).read_bytes()
    assert {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()} == {
        *publish_dataset.SOURCE_FILES,
        "README.md",
        "checksums.json",
    }


def test_rejects_reference_from_different_conversation(source, tmp_path):
    ref = source / "references/gpt-5.4-mini/en.jsonl"
    ref.write_text(ref.read_text().replace('"question": "Question 1"', '"question": "Changed"'))
    with pytest.raises(ValueError, match="reference conversation mismatch"):
        publish_dataset.stage_dataset(source, tmp_path / "staged")


def test_rejects_duplicate_ids(source, tmp_path):
    prompt = source / "lmarena_v3_chat400_en.jsonl"
    prompt.write_text(prompt.read_text().replace('"question_id": "1"', '"question_id": "0"'))
    with pytest.raises(ValueError, match="400 unique"):
        publish_dataset.stage_dataset(source, tmp_path / "staged")


def test_does_not_print_credential_signature(source, tmp_path):
    ref = source / "references/gpt-5.4-mini/en.jsonl"
    fake_token = "hf_" + "x" * 30
    ref.write_text(ref.read_text().replace('"generation": "Answer"', f'"generation": "{fake_token}"'))
    with pytest.raises(ValueError, match="Potential credential signature") as error:
        publish_dataset.stage_dataset(source, tmp_path / "staged")
    assert fake_token not in str(error.value)


def test_upload_uses_allowlist_and_parent_revision(source, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub

    calls = {}

    class Api:
        def whoami(self):
            return {"name": "anushakamathofficial"}

        def create_repo(self, **kwargs):
            calls["create"] = kwargs

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="initial-revision")

        def upload_folder(self, **kwargs):
            calls["upload"] = kwargs
            return SimpleNamespace(oid="uploaded-revision")

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    output = tmp_path / "staged"
    publish_dataset.stage_dataset(source, output)
    (output / "private.key").write_text("must not be uploaded")
    assert publish_dataset.publish(output) == "uploaded-revision"
    assert calls["upload"]["allow_patterns"] == [*publish_dataset.SOURCE_FILES, "checksums.json", "README.md"]
    assert calls["upload"]["parent_commit"] == "initial-revision"
    assert calls["create"]["private"] is True
