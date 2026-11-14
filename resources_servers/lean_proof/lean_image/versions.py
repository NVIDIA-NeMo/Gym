# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinned Lean/Mathlib versions, one image per version.

Compiled oleans do not carry across Mathlib versions, so a benchmark must run against exactly
the version its statements were written for. The wrong one does not error at startup: it fails
individual tasks with ordinary-looking compile errors, indistinguishable from a model that
could not do the problem.

``lean_sha256`` is the sha256 of the release tarball named by ``lean_version``;
``mathlib_commit`` is the commit the Mathlib tag of the same name resolves to.

To add a version, resolve the tag and hash the tarball:

    git ls-remote https://github.com/leanprover-community/mathlib4 refs/tags/<tag>
    curl -fsSL https://github.com/leanprover/lean4/releases/download/<tag>/lean-<v>-linux.tar.zst | sha256sum

Take the Lean version from the Mathlib tag rather than choosing it: each Mathlib release names
the toolchain it requires, and the build asserts the two agree rather than trusting the pair.

A Python module rather than JSON so the pins can carry `pragma: allowlist secret`: a pinned
commit or checksum is indistinguishable from a high-entropy secret to a scanner, and JSON has
no comment syntax to say otherwise.
"""

VERSIONS = {
    "v4.19.0": {
        "lean_version": "v4.19.0",
        "mathlib_commit": "c44e0c8ee63ca166450922a373c7409c5d26b00b",  # pragma: allowlist secret
        "lean_sha256": "6fe3ce97a58f44e2b3567d455b994eacec5bfe9ae7774f2a573444480ba813fe",  # pragma: allowlist secret
    },
}


def pins(version: str) -> dict:
    """Return the pins for ``version``, or exit listing the known ones."""
    if version not in VERSIONS:
        raise SystemExit(f"unknown version {version!r}; known: {' '.join(VERSIONS)}")
    return VERSIONS[version]
