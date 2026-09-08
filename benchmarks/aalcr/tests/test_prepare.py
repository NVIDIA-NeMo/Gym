# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from io import BytesIO
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pytest

from benchmarks.aalcr import prepare as v1_0
from benchmarks.aalcr import prepare_v1_1 as v1_1


ROW = {
    "document_category": "category",
    "document_set_id": "set",
    "question_id": 28,
    "question": "What percentage?",
    "answer": "65.8%",
    "data_source_filenames": "document.txt",
    "data_source_urls": "https://example.com/document",
    "input_tokens": 90000,
}


def _documents_zip() -> bytes:
    payload = BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("lcr/category/set/document.txt", "document contents")
    return payload.getvalue()


@pytest.mark.parametrize("module", [v1_0, v1_1], ids=["v1.0", "v1.1"])
def test_prepare_pins_dataset_and_records_provenance(module, tmp_path) -> None:
    output_fpath = tmp_path / module.OUTPUT_FPATH.name
    response = MagicMock(content=_documents_zip())

    with (
        patch.object(v1_0, "load_dataset", return_value=[ROW]) as load_dataset,
        patch.object(v1_0, "get_hf_token", return_value="token"),
        patch.object(v1_0.requests, "get", return_value=response) as get,
        patch.object(module, "OUTPUT_FPATH", output_fpath),
    ):
        result = module.prepare(dataset_revision=module.DATASET_REVISION)

    assert result == output_fpath
    load_dataset.assert_called_once_with(
        "ArtificialAnalysis/AA-LCR",
        revision=module.DATASET_REVISION,
        split="test",
        token="token",
    )
    requested_url = get.call_args.args[0]
    assert f"/resolve/{module.DATASET_REVISION}/" in requested_url
    response.raise_for_status.assert_called_once_with()

    prompt = """BEGIN INPUT DOCUMENTS

BEGIN DOCUMENT 1:
document contents
END DOCUMENT 1

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

What percentage?

END QUESTION
"""
    assert json.loads(output_fpath.read_text()) == {
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
        **ROW,
        "input_tokens_band": "80k-100k",
        "aa_lcr_version": module.BENCHMARK_VERSION,
        "aa_lcr_dataset_revision": module.DATASET_REVISION,
        "aa_lcr_judge_protocol": module.JUDGE_PROTOCOL,
    }


@pytest.mark.parametrize("module", [v1_0, v1_1], ids=["v1.0", "v1.1"])
def test_prepare_rejects_a_different_dataset_revision(module) -> None:
    with pytest.raises(ValueError, match="requires dataset revision"):
        module.prepare(dataset_revision="0" * 40)
