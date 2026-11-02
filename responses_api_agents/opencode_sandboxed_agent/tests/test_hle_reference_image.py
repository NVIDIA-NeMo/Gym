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
"""hle_reference_image: the HLE-Diamond reference sandbox recipe stays upstream's and the science image's."""

import hashlib
import re
from pathlib import Path


AGENT_DIR = Path(__file__).parents[1]
REFERENCE_IMAGE = AGENT_DIR / "hle_reference_image"
SCIENCE_IMAGE = AGENT_DIR / "offline_science_image"

# docs/requirements-tools.txt of centerforaisafety/hle at 22ed3074b1e7b134bcbc09028d0ba320839b0655,
# the protocol revision named in the image README: the file in the image directory is that text verbatim.
UPSTREAM_REQUIREMENTS_SHA256 = (
    "c1c626beb8f2fab11609f6ed41a2721d9acd139554d6ce3c8c83f52a49797068"  # pragma: allowlist secret
)
UPSTREAM_PYTHON = "3.11.16"
# docs/evaluation-with-tools.md, section 2: "Do not add specialist solvers, chess engines, OCR tools, or
# other task libraries". The science image's extras that would violate it, as Dockerfile and lock tokens.
FORBIDDEN_TOKENS = ("sage", "lean", "mathlib", "z3", "rdkit", "tesseract", "stockfish", "ocr", "pysat", "cvxpy")

PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\]+)", re.MULTILINE)


def _pins(text: str) -> dict[str, str]:
    return {name.lower().replace("_", "-"): version for name, version in PIN.findall(text)}


def test_requirements_file_is_the_pinned_upstream_text_verbatim():
    text = (REFERENCE_IMAGE / "requirements-tools.txt").read_bytes()

    assert hashlib.sha256(text).hexdigest() == UPSTREAM_REQUIREMENTS_SHA256
    assert text.decode().splitlines()[0] == f"# Python {UPSTREAM_PYTHON}"
    assert len(_pins(text.decode())) == 12


def test_lock_and_dockerfile_install_exactly_the_upstream_pins_on_upstream_python():
    pins = _pins((REFERENCE_IMAGE / "requirements-tools.txt").read_text())
    locked = _pins((REFERENCE_IMAGE / "requirements.lock").read_text())
    dockerfile = (REFERENCE_IMAGE / "Dockerfile").read_text()

    # Every upstream pin is in the hashed lock at upstream's version; the lock adds only their dependencies.
    assert {name: locked.get(name) for name in pins} == pins
    assert "--require-hashes -r /opt/image-provenance/requirements.lock" in dockerfile
    assert "COPY requirements-tools.txt requirements.lock /opt/image-provenance/" in dockerfile
    assert re.search(
        rf"^FROM python:{re.escape(UPSTREAM_PYTHON)}-slim-bookworm@sha256:[0-9a-f]{{64}} AS base$", dockerfile, re.M
    )
    assert f"test \"$(python --version)\" = 'Python {UPSTREAM_PYTHON}'" in dockerfile

    # Instructions only (the header comment names the forbidden tools); "lean" would also match "clean",
    # so tokens are matched as whole words.
    instructions = "\n".join(line for line in dockerfile.splitlines() if not line.lstrip().startswith("#"))
    lowered = (instructions + "\n" + "\n".join(locked)).lower()
    assert not [token for token in FORBIDDEN_TOKENS if re.search(rf"(?<![a-z0-9]){token}(?![a-z0-9])", lowered)]


def test_opencode_is_installed_exactly_as_the_science_image_installs_it():
    reference = (REFERENCE_IMAGE / "Dockerfile").read_text()
    science = (SCIENCE_IMAGE / "Dockerfile").read_text()

    def pinned_lines(text: str) -> set[str]:
        """The OpenCode release and Node download URLs and their checksum lines, whitespace-normalised."""
        return {
            " ".join(line.split())
            for line in text.splitlines()
            if any(
                marker in line
                for marker in (
                    "opencode-linux-x64.tar.gz",
                    "node-v22",
                    "opencode' | sha256sum",
                    "node.tar.xz' | sha256sum",
                )
            )
        }

    assert pinned_lines(reference) == pinned_lines(science) and len(pinned_lines(reference)) == 4
    for name in ("package.json", "package-lock.json"):
        assert (REFERENCE_IMAGE / "opencode" / name).read_bytes() == (SCIENCE_IMAGE / "opencode" / name).read_bytes()
    assert "npm ci --ignore-scripts --omit=dev --no-audit --no-fund" in reference
    assert "test \"$(opencode --version)\" = '1.17.11'" in reference
