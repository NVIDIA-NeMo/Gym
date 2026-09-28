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

"""Pins for each Mathlib version a Gym benchmark needs.

One image per version: compiled oleans do not carry across versions, so a benchmark run against
the wrong one fails in ways that look like model errors rather than environment errors.

``lean_sha256`` is the sha256 of the release tarball named by ``lean_version``;
``mathlib_commit`` is the commit the Mathlib tag of the same name resolves to. Both were
computed from the published artifacts, and the v4.34.0 pair reproduces the one in
``responses_api_agents/opencode_sandboxed_agent/offline_science_image``.

To add a version, resolve the tag and hash the tarball:

    git ls-remote https://github.com/leanprover-community/mathlib4 refs/tags/<tag>
    curl -fsSL https://github.com/leanprover/lean4/releases/download/<tag>/lean-<v>-linux.tar.zst | sha256sum

The build asserts the Mathlib commit's own ``lean-toolchain`` matches ``lean_version``, so a
mismatched pair fails the build rather than producing a subtly wrong image.

A Python module rather than JSON so the pins can carry `pragma: allowlist secret` markers: a
pinned commit or checksum is indistinguishable from a high-entropy secret to a scanner, and
JSON has no comment syntax to say otherwise.
"""

VERSIONS = {
    "v4.12.0": {
        "lean_version": "v4.12.0",
        "mathlib_commit": "809c3fb3b5c8f5d7dace56e200b426187516535a",  # pragma: allowlist secret
        "lean_sha256": "2659fc3bfb3955c118e727346c4d894799c69cfbbc2238f6d625fd4f5ad2d929",  # pragma: allowlist secret
        "benchmarks": ["minif2f", "proofnet", "putnam_bench", "mobench"],
    },
    "v4.19.0": {
        "lean_version": "v4.19.0",
        "mathlib_commit": "c44e0c8ee63ca166450922a373c7409c5d26b00b",  # pragma: allowlist secret
        "lean_sha256": "6fe3ce97a58f44e2b3567d455b994eacec5bfe9ae7774f2a573444480ba813fe",  # pragma: allowlist secret
        "benchmarks": ["leancat"],
    },
    "v4.33.0": {
        "lean_version": "v4.33.0",
        "mathlib_commit": "db584cd6d46c92f209a44c0f1c829460d327499d",  # pragma: allowlist secret
        "lean_sha256": "4b3fb03c29a1e0a253fb1d11f9bae3725f19a0dc6fc09b3ea16d2c9df3349e2c",  # pragma: allowlist secret
        "benchmarks": [],
    },
    "v4.34.0": {
        "lean_version": "v4.34.0",
        "mathlib_commit": "5ed2965256430c3649e86755f9576b54eca72435",  # pragma: allowlist secret
        "lean_sha256": "caaa98356098c85dc0fcbbd28e1ec66f39eb6551829972b752ff20e1286b646b",  # pragma: allowlist secret
        "benchmarks": [],
    },
}


def pins(version: str) -> dict:
    """Return the pins for ``version``, or exit listing the known ones."""
    if version not in VERSIONS:
        raise SystemExit(f"unknown version {version!r}; known: {' '.join(VERSIONS)}")
    return VERSIONS[version]
