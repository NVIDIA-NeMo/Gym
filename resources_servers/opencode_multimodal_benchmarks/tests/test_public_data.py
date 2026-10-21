# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json

import pytest

from benchmarks.opencode_multimodal_benchmarks import canonical
from benchmarks.opencode_multimodal_benchmarks.agentx_prepare import prepare, validate_rows
from benchmarks.opencode_multimodal_benchmarks.agentx_source import reference_metadata
from resources_servers.opencode_multimodal_benchmarks.mcq import infer_option


@pytest.mark.parametrize(
    "prediction",
    [
        "The correct answer is **B**. Here is a long explanation of the image contents.",
        "The answer is c. A is incorrect, B is incorrect and so is D.",
    ],
)
def test_verbose_option_does_not_require_judge(prediction):
    assert infer_option(prediction, {"A": "cat", "B": "dog", "C": "bird", "D": "fish"}) == (
        "B" if "**B**" in prediction else "C"
    )


def test_verbose_option_must_be_available_and_api_failure_not_salvaged():
    assert infer_option("The answer is D. More explanation follows here.", {"A": "cat", "B": "dog"}) is None
    assert infer_option("Failed to obtain answer via API. The answer is B.", {"A": "cat", "B": "dog"}) is None


@pytest.mark.parametrize(
    "mutation", ["id", "label", "type", "question", "media", "missing_image", "assistant", "outside"]
)
def test_agentx_tampering_rejected_even_with_updated_checksum(tmp_path, monkeypatch, agentx_bundle, mutation):
    path, media, image, rows = agentx_bundle
    row = rows[-1]  # Even a sample outside the requested smoke limit must be checked.
    if mutation == "id":
        row["verifier_metadata"]["index"] = "1"  # Excluded generative ID.
    elif mutation == "label":
        row["verifier_metadata"]["GT_json"] = '{"value":"blue","justification":"changed"}'
    elif mutation == "type":
        row["verifier_metadata"]["query_type"] = "subjective"
    elif mutation == "question":
        row["responses_create_params"]["input"][0]["content"][0]["text"] = "Different question"
    elif mutation == "media":
        row["verifier_metadata"]["source_media_sha256"]["original.png"] = "0" * 64
    elif mutation == "missing_image":
        row["responses_create_params"]["input"][0]["content"].pop()
    elif mutation == "assistant":
        row["responses_create_params"]["input"][0]["role"] = "assistant"
    elif mutation == "outside":
        row["responses_create_params"]["input"][0]["content"][1]["image_url"] = str(image)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        prepare(str(path), hashlib.sha256(path.read_bytes()).hexdigest(), media_root=str(media), limit=1)


def test_arbitrary_605_rows_no_longer_accepted():
    rows = [{"verifier_metadata": {"index": str(i)}} for i in range(605)]
    with pytest.raises(ValueError, match="IDs differ"):
        validate_rows(rows)


def test_nested_public_reference_is_preserved():
    reference = {
        "query_type": "Objective",
        "tool_metadata": {
            "final_answer": {
                "value": "192.5 minutes; DINNER; 7",
                "justification": "timeline",
            }
        },
    }
    record = {"dialogs": [{"role": "user", "content": "Timeline question"}], "files": [{"type": "image"}]}
    result = reference_metadata("522", record, reference)
    assert result["answer"] == "192.5 minutes; DINNER; 7"
    assert json.loads(result["GT_json"])["justification"] == "timeline"
    reference["tool_metadata"] = {}
    with pytest.raises(ValueError, match="Missing Agent-X reference"):
        reference_metadata("522", record, reference)


@pytest.mark.parametrize("benchmark", ["VStarBench", "HRBench8K"])
def test_mcq_pins_ids_labels_cycles_and_media(monkeypatch, tmp_path, benchmark):
    image = tmp_path / (hashlib.sha256(b"original").hexdigest() + ".png")
    image.write_bytes(b"original")
    replacement = tmp_path / (hashlib.sha256(b"different").hexdigest() + ".png")
    replacement.write_bytes(b"different")
    task = {
        "index": "0",
        "question": "color?",
        "answer": "A",
        "options": {"A": "red", "B": "blue"},
        "category": "single",
        "l2_category": "",
        "cycle_category": "0",
        "split": "none",
        "images": [str(image)],
    }
    ref = {"tasks": {"0": canonical.fingerprint(canonical.mcq_identity(task))}}
    monkeypatch.setattr(canonical, "manifest", lambda _: ref)
    canonical.validate_mcq_tasks([task], benchmark)
    for field, value in [("answer", "B"), ("cycle_category", "1"), ("images", [str(replacement)]), ("index", "1")]:
        changed = copy.deepcopy(task)
        changed[field] = value
        with pytest.raises(ValueError):
            canonical.validate_mcq_tasks([changed], benchmark)
    with pytest.raises(ValueError):
        canonical.validate_mcq_tasks([task, task], benchmark)

    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="image checksum"):
        canonical.validate_mcq_tasks([task], benchmark)
