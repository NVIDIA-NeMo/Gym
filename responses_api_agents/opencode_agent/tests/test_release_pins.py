# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The release digest table and the releases the adapter's notes were checked on."""

import re

from nemo_gym.models import HARNESS_NOTES
from responses_api_agents.opencode_agent.runtime import OPENCODE_RELEASE_SHA256, OPENCODE_V2_VERSION, OPENCODE_VERSION


# The assets install_opencode_runtime.sh and setup_opencode._linux_release_arch pick: x86_64 -> x64-baseline,
# aarch64 -> arm64, each with a -musl variant. Stated here, not read from the table, so a missing entry fails.
LINUX_ASSETS = {"x64-baseline", "x64-baseline-musl", "arm64", "arm64-musl"}
# The releases the README and the seed warning call validated: the two OpenCode 1 releases request sampling accepts and
# the two OpenCode 2 releases with pinned digests.
VALIDATED_RELEASES = ("1.17.11", OPENCODE_VERSION, "2.2.7", OPENCODE_V2_VERSION)


def test_every_pinned_release_has_one_well_formed_digest_per_linux_asset():
    # A missing or malformed entry would only show when a sandbox on that platform refuses the download.
    for version, digests in OPENCODE_RELEASE_SHA256.items():
        assert re.fullmatch(r"\d+\.\d+\.\d+", version), version
        assert digests.keys() == LINUX_ASSETS, version
        for asset, digest in digests.items():
            assert re.fullmatch(r"[0-9a-f]{64}", digest), (version, asset)
    every = [digest for digests in OPENCODE_RELEASE_SHA256.values() for digest in digests.values()]
    # Distinct assets never share bytes: a repeated digest is a copy-paste error.
    assert len(set(every)) == len(every)


def test_every_supported_release_was_checked_for_the_harness_notes():
    # Token capture recognizes OpenCode's max-steps note by its exact text, which is checked per release. A version
    # added to the validated sets without re-checking that text fails here, not in a training run.
    notes = [note for note in HARNESS_NOTES if note.harness == "opencode"]
    assert notes
    for version in VALIDATED_RELEASES:
        assert all(version in note.checked_versions for note in notes), version
