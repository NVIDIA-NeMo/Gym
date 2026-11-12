# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Git object's abbreviation length is not a workspace difference."""

import pytest

from resources_servers.agentic_if.episode.opencode_replay import workspace_patches_equal


PATCH = "diff --git a/a.py b/a.py\nindex a123456..b234567 100644\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
FULL = PATCH.replace("a123456..b234567", "a123456" + "1" * 33 + "..b234567" + "2" * 33)


def test_abbreviated_and_full_object_ids_are_the_same_patch():
    assert workspace_patches_equal(PATCH, FULL) and workspace_patches_equal(FULL, PATCH)
    assert workspace_patches_equal(PATCH, PATCH)


@pytest.mark.parametrize(
    "changed",
    [
        FULL.replace("+new", "+wrong"),
        FULL.replace("100644", "100755"),
        FULL.replace("a/a.py", "a/b.py"),
        FULL.replace("b234567", "c234567"),
        FULL + "diff --git a/extra b/extra\nnew file mode 100644\n",
        FULL.replace("+new", "+index a123456..b234567 100644"),
    ],
)
def test_content_paths_modes_and_incompatible_hashes_still_diverge(changed):
    assert not workspace_patches_equal(PATCH, changed)


def test_binary_body_is_not_normalized():
    patch = "diff --git a/x b/x\nindex a123456..b234567 100644\nGIT binary patch\nliteral 2\nABCD\n"
    assert not workspace_patches_equal(patch, patch.replace("ABCD", "WXYZ"))
