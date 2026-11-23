# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock

import pytest

from benchmarks.genrm_mmrb2 import prepare as module


@pytest.mark.parametrize("mode", ["genrm", "llm_as_a_judge"])
def test_download_and_build_once(tmp_path, monkeypatch, mode):
    from benchmarks.genrm_mmrb2 import build_validation

    monkeypatch.setattr(module, "__file__", str(tmp_path / "prepare.py"))
    download = Mock(return_value=str(tmp_path / "snapshot"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)

    def build(source_dir, output, *, genrm):
        assert source_dir == tmp_path / "snapshot" / "reasoning"
        assert genrm == (mode == "genrm")
        output.parent.mkdir(parents=True)
        rows = [
            {
                "verifier_metadata": {
                    "pair_id": str(i),
                    "order": order,
                    "task": "reasoning",
                    "response_modality": "text_only",
                    "judge_mode": mode,
                    "pair_type": "chosen_text__rejected_text",
                }
            }
            for i in range(350)
            for order in ("forward", "reverse")
        ]
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))

    builder = Mock(side_effect=build)
    monkeypatch.setattr(build_validation, "prepare", builder)
    output = module.prepare(judge_mode=mode)
    assert len(output.read_text().splitlines()) == 700
    assert module.prepare(judge_mode=mode) == output
    download.assert_called_once_with(
        repo_id="rl-research/multimodal-rewardbench-2",
        repo_type="dataset",
        revision=module.DATASET_REVISION,
        allow_patterns="reasoning/test-*.parquet",
    )
    assert builder.call_count == 1


def test_missing_explicit_source_does_not_download(tmp_path, monkeypatch):
    download = Mock()
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    with pytest.raises(FileNotFoundError):
        module.prepare(tmp_path / "missing.jsonl")
    download.assert_not_called()


def test_prepare_rejects_wrong_scope(tmp_path):
    for kwargs in ({"task": "edit"}, {"response_modality": "all"}, {"judge_mode": "other"}):
        with pytest.raises(ValueError):
            module.prepare(**kwargs)


def test_prepared_input_requires_mode_modality_and_complete_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "__file__", str(tmp_path / "prepare.py"))
    source = tmp_path / "source.jsonl"
    rows = [
        {
            "id": f"{i}:{order}",
            "verifier_metadata": {
                "pair_id": str(i),
                "order": order,
                "task": "reasoning",
                "response_modality": "text_only",
                "judge_mode": "genrm",
                "pair_type": "chosen_text__rejected_text",
            },
        }
        for i in range(350)
        for order in ("forward", "reverse")
    ]
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    output = module.prepare(source)
    assert output.read_bytes() == source.read_bytes()
    original = output.read_bytes()
    with pytest.raises(ValueError, match="selected judge_mode"):
        module.prepare(source, judge_mode="llm_as_a_judge")
    rows.pop()
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="700 judgments"):
        module.prepare(source)
    assert output.read_bytes() == original
