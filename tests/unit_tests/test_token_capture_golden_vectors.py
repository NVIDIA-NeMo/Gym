# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden vectors are the cross-repo wire contract for the lineage hashes.

An external resolver must reproduce these byte-for-byte or every continuation
silently resolves unresolved. If this test fails, either bump the fingerprint or
digest version machinery deliberately, or fix the regression — never regenerate
the vectors silently.
"""

import json
from pathlib import Path

from nemo_gym.token_id_capture.lineage import assistant_fingerprint, canonicalize_tool_arguments, conversation_digest
from nemo_gym.token_id_capture.records import compute_digest


VECTORS = json.loads((Path(__file__).parent / "token_capture_golden_vectors.json").read_text())


def _digest_input(spec):
    return list(range(1000)) if spec == "range(1000)" else spec


def test_fingerprint_vectors():
    for name, vector in VECTORS["fingerprint"].items():
        assert assistant_fingerprint(vector["input"]) == vector["expected"], name


def test_fingerprint_is_identical_across_dialects():
    values = {VECTORS["fingerprint"][k]["expected"] for k in ("chat_tool", "anthropic_tool", "responses_tool")}
    assert len(values) == 1


def test_conversation_digest_vectors():
    for name, vector in VECTORS["conversation_digest"].items():
        assert conversation_digest(vector["input"]) == vector["expected"], name


def test_tool_argument_canonicalization_vectors():
    for name, vector in VECTORS["canonicalize_tool_arguments"].items():
        assert canonicalize_tool_arguments(vector["input"]) == vector["expected"], name


def test_compute_digest_vectors():
    for name, vector in VECTORS["compute_digest"].items():
        assert compute_digest(_digest_input(vector["input"])) == vector["expected"], name
