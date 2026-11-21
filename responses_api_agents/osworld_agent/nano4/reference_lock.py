# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Producer version and content digests; each allowlisted literal is a fixed checksum."""

from typing import Any


REFERENCE_LOCK: dict[str, Any] = {
    "repository": "externally supplied prompt-format producer",
    "commit": "35d35a7c38b356ff1549feda11572a4bceda4e49",  # pragma: allowlist secret -- producer/source checksum
    "agent_profile": "rfc0015_v1_feedback_v2",
    "prompt_profile_sha256": "6b96c5d118cf5fb9cbac4b179b50011e4160cf89f796a4229bff047f72adf056",  # pragma: allowlist secret -- producer/source checksum
    "runtime_files": {
        "convert_procua_sft.py": {
            "bytes": 128540,
            "sha256": "65df27dfca51a54525c0b55f60498b7b5cda4b9af5d9b9422b8ffbbbf87994f6",  # pragma: allowlist secret -- producer/source checksum
        },
        "sft_tool_prompt.py": {
            "bytes": 13488,
            "sha256": "3ff02d105794bb24d1ac1f6b1c09675d861f1eb9d25dfcde4a38f1e1d69f444d",  # pragma: allowlist secret -- producer/source checksum
        },
        "sft_coordinate_normalization.py": {
            "bytes": 25804,
            "sha256": "c3f3bdc96c9adc57ae6c8d939cf09c4eb679c9033f2ed280bd258758e895c4d9",  # pragma: allowlist secret -- producer/source checksum
        },
    },
    "default_actions_per_chunk": 10,
    "default_boundary_image_window": 1,
    "default_wait_seconds": 20,
    "wait_seconds_authority": "sft_tool_prompt.py:57-60; argument_free_v1 description",
    "authority_repository_commit": "6d9fc17479949f94b1fc85c428161a150b6ba838",  # pragma: allowlist secret -- producer/source checksum
    "authority_osworld_submodule_commit": "85382cce017040d3c3743772641a76fef41a965c",  # pragma: allowlist secret -- producer/source checksum
    "training_consumer_ready": False,
    "validation_files": {
        "README.md": {
            "bytes": 39089,
            "sha256": "2dfe84e8d6d2b3ccaac776172d29b49c7d18ef105e2fc419620e203a04db8f5d",  # pragma: allowlist secret -- producer/source checksum
        },
        "tests/test_convert_procua_sft.py": {
            "bytes": 88858,
            "sha256": "8b1c1467d8d4e0869e3e9644134bbb0f80fadf461b69789c5d10a54562c60414",  # pragma: allowlist secret -- producer/source checksum
        },
        "tests/test_sft_coordinate_normalization.py": {
            "bytes": 2701,
            "sha256": "c9d40a84d6b2d07f4e661253919ab214e5611e4e51948195246ca9a43d60adc9",  # pragma: allowlist secret -- producer/source checksum
        },
        "samples/converted/manifest.json": {
            "bytes": 5484,
            "sha256": "90f89ada4f26b6932361853a1e04ec80024661ff4d7a6dac4ec91b5175f4368b",  # pragma: allowlist secret -- producer/source checksum
        },
        "samples/converted/stats.json": {
            "bytes": 736,
            "sha256": "01a33ed415f6c318b2bcdfb8375a5bf4cfe0f82e409514ab6babfad966af032c",  # pragma: allowlist secret -- producer/source checksum
        },
        "samples/converted/examples/index.json": {
            "bytes": 2809,
            "sha256": "04bcf39edb69b5c8984917443b198826e6403cfcd4b0bdce901bf4bb6cc61780",  # pragma: allowlist secret -- producer/source checksum
        },
        "samples/converted/annotations.jsonl": {
            "bytes": 51527,
            "sha256": "af6aedab28682c33e744e1c0e721fd0774085202870d7c1174e56d35762b214a",  # pragma: allowlist secret -- producer/source checksum
        },
    },
}
