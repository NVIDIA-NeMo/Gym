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

"""Pinned versions for the Kimina Lean Server image, one image per Lean version.

Same rule as ``lean_proof/lean_image/versions.py``, and for the same reason: compiled
oleans do not carry across Mathlib versions, and the wrong one does not error at
startup -- it fails tasks with ordinary-looking compile errors that are
indistinguishable from a model that could not do the problem. CombiBench's statements
are written against ``v4.24.0``.

Four things are pinned rather than three, because this image also carries the Lean REPL
and the server that pools it:

* ``lean_version`` / ``lean_sha256`` -- the release tarball, verified by checksum.
* ``mathlib_commit`` -- what the Mathlib tag of the same name resolves to.
* ``repl_commit`` -- ``leanprover-community/repl`` at the tag of the same name. Kimina's
  own default REPL is ``FrederickPu/repl@lean415compat``, which is for older Leans.
* ``kimina_commit`` -- the server. The project publishes no tags at all, so a commit is
  the only way to pin it; this is the head of ``main``.

To add a version, resolve the tags and hash the tarball:

    git ls-remote https://github.com/leanprover-community/mathlib4 refs/tags/<tag>
    git ls-remote https://github.com/leanprover-community/repl refs/tags/<tag>
    curl -fsSL https://github.com/leanprover/lean4/releases/download/<tag>/lean-<v>-linux.tar.zst | sha256sum

Take the Lean version from the Mathlib tag rather than choosing it: each Mathlib release
names the toolchain it requires, and the build asserts the two agree rather than trusting
the pair. The REPL's own ``lean-toolchain`` is asserted the same way.

A Python module rather than JSON so the pins can carry ``pragma: allowlist secret``: a
pinned commit or checksum is indistinguishable from a high-entropy secret to a scanner,
and JSON has no comment syntax to say otherwise.
"""

VERSIONS = {
    "v4.24.0": {
        "lean_version": "v4.24.0",
        "mathlib_commit": "f897ebcf72cd16f89ab4577d0c826cd14afaafc7",  # pragma: allowlist secret
        "lean_sha256": "b14f5e5159219dd1a1956c3b806813319f5e94ccd5bdfd56f54520609a5bb5ec",  # pragma: allowlist secret
        "repl_commit": "8fff8552292860d349b459d6a811e6915671dc0d",  # pragma: allowlist secret
        "kimina_commit": "fb2393de3461db35eda4c714e3fd21187e92ec90",  # pragma: allowlist secret
    },
}


def pins(version: str) -> dict:
    """Return the pins for ``version``, or exit listing the known ones."""
    if version not in VERSIONS:
        raise SystemExit(f"unknown version {version!r}; known: {' '.join(VERSIONS)}")
    return VERSIONS[version]


def main() -> None:
    """Print every pin as the ``--build-arg`` flags the documented build takes.

    The README tells the reader to run this module to see the pins, so it has to
    print them; printing them in the form ``docker build`` wants means the
    command below it can be pasted rather than retyped from the dict above.
    """
    for version, values in VERSIONS.items():
        flags = " \\\n    ".join(f"--build-arg {key.upper()}={value}" for key, value in values.items())
        print(f"# {version}\n    {flags}")


if __name__ == "__main__":
    main()
