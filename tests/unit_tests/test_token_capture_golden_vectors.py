# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the hash values shared by Gym and external lineage resolvers.

An external resolver must reproduce these values byte-for-byte.
Otherwise, unchanged continuations resolve as ``UNRESOLVED``.
When an intentional encoding change alters a value, update the corresponding fingerprint or digest version.
Do not regenerate the vectors to hide an unintended change.
"""

from nemo_gym.token_id_capture.fingerprint import (
    assistant_fingerprint,
    canonicalize_tool_arguments,
    conversation_digest,
)
from nemo_gym.token_id_capture.records import compute_digest


VECTORS = {
    "fingerprint": {
        "plain": {
            "input": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
            "expected": "5b20b01e7ca4ec80db6e221c277c02a796f6a2b90e81c7adaf7f1b8fc2c6846e",  # pragma: allowlist secret
        },
        "chat_tool": {
            "input": [
                {"role": "user", "content": "find x"},
                {
                    "role": "assistant",
                    "content": "calling",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "x", "k": 3}'},
                        }
                    ],
                },
            ],
            "expected": "82955bc805cccaf9d2faa840479ff6c30420d20e42bf2bc1ebef14d679ec3768",  # pragma: allowlist secret
        },
        "anthropic_tool": {
            "input": [
                {"role": "user", "content": "find x"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "calling"},
                        {"type": "tool_use", "id": "c1", "name": "search", "input": {"k": 3, "q": "x"}},
                    ],
                },
            ],
            "expected": "82955bc805cccaf9d2faa840479ff6c30420d20e42bf2bc1ebef14d679ec3768",  # pragma: allowlist secret
        },
        "responses_tool": {
            "input": [
                {"role": "user", "content": "find x"},
                {"role": "assistant", "content": "calling"},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "search",
                    "arguments": '{"k":3,"q":"x"}',
                },
            ],
            "expected": "82955bc805cccaf9d2faa840479ff6c30420d20e42bf2bc1ebef14d679ec3768",  # pragma: allowlist secret
        },
        "empty": {"input": [], "expected": ""},
    },
    "conversation_digest": {
        "plain": {
            "input": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
            "expected": "db10e497fe3d0ee04e81f35f2e3b7857e4c066f5ccaec3c5fcd279f2793ca81b",  # pragma: allowlist secret
        },
        "multimodal": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    ],
                },
                {"role": "assistant", "content": "seen"},
            ],
            "expected": "18d4045a0813dfbe915b43efde91849b483eaf207ba7d130f00473966dc28591",  # pragma: allowlist secret
        },
        "tool_result": {
            "input": [
                {"role": "tool", "tool_call_id": "c1", "content": "42"},
                {"role": "assistant", "content": "done"},
            ],
            "expected": "497f57bc226bd826c8d8cb340f7ab535585c0c86123085c958f880c39e83ec3e",  # pragma: allowlist secret
        },
        "empty": {
            "input": [],
            "expected": "edd3713968ca572ae44468ff61fabbec572e8de247023892e8bdf0f55aa8cb9f",  # pragma: allowlist secret
        },
    },
    "canonicalize_tool_arguments": {
        "reordered": {"input": '{"k": 3, "q": "x"}', "expected": '{"k":3,"q":"x"}'},
        "unicode": {
            "input": '{"s": "caf\u00e9 \\"quoted\\"", "n": 1.5}',
            "expected": '{"n":1.5,"s":"caf\u00e9 \\"quoted\\""}',
        },
        "none": {"input": None, "expected": ""},
    },
    "compute_digest": {
        "empty": {
            "input": [],
            "expected": "2e222d36d0c6ae1db1ec18b3f96d9a46df5db61b40137bbfbac4cdf0acde9763",  # pragma: allowlist secret
        },
        "zero": {
            "input": [0],
            "expected": "0596efeb21b51ec6eb122c9e470703df19a095a7c5ccb96bf87df0ba2447ba59",  # pragma: allowlist secret
        },
        "small": {
            "input": [1, 2, 3],
            "expected": "bf99633051449f0f3248b4995a7c8468b9561d7732447341916085f12ff9ff54",  # pragma: allowlist secret
        },
        "range1000": {
            "input": "range(1000)",
            "expected": "9b33490a72c5e86cc27fcc0916a08dbb8f1edeb5f108d010dad1b616bd7f0a2b",  # pragma: allowlist secret
        },
    },
}


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
