# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Dockerfile must build the sandbox the vendored GDPval-AA v2 manifests describe.

These tests tie the container definition to the published Python pin set and
apt closure.
"""

import re
from pathlib import Path


_CONTAINERS = Path(__file__).resolve().parents[1] / "containers"
_PY_MANIFEST = _CONTAINERS / "gdpval_aa_v2_python_requirements.txt"
_APT_MANIFEST = _CONTAINERS / "gdpval_aa_v2_apt_closure.txt"
_DOCKERFILE = _CONTAINERS / "Dockerfile"
_ARM64_EXCLUSIONS = _CONTAINERS / "gdpval_aa_v2_arm64_exclusions.txt"


def _pins(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = "==" if "==" in line else "="
        name, _, ver = line.partition(sep)
        out[name.lower().replace("_", "-")] = ver
    return out


def test_python_manifest_is_the_published_419_pins():
    pins = _pins(_PY_MANIFEST)
    assert len(pins) == 419, f"expected the published 419 pins, found {len(pins)}"
    # Spot-check anchors from the published snapshot, including the ones that
    # force an x86_64 build.
    assert pins["numpy"] == "2.4.4"
    assert pins["pymupdf"] == "1.27.2.2"
    assert pins["nvidia-nccl-cu12"] == "2.29.7"
    assert all(v and v[0].isdigit() for v in pins.values()), "a pin has no concrete version"


def test_apt_manifest_is_the_published_762_pins_on_trixie():
    pins = _pins(_APT_MANIFEST)
    assert len(pins) == 762, f"expected the published 762 pins, found {len(pins)}"
    # The closure being trixie-pinned is why the Dockerfile uses a trixie base.
    assert any("deb13" in v for v in pins.values()), "closure is not Debian 13 pinned"
    assert pins["python3.13"].startswith("3.13"), "reference sandbox is not CPython 3.13"


def test_dockerfile_installs_the_pinned_manifest_rather_than_loose_names():
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert "FROM debian:trixie" in text, "base image must match the trixie-pinned closure"
    # Debian's interpreter, not the docker python image's later patch release.
    assert "python3.13 \\" in text, "the Debian interpreter must be installed explicitly"
    assert '-r "$EFFECTIVE"' in text, "pins must be installed from a requirements file"
    assert _PY_MANIFEST.name in text and _APT_MANIFEST.name in text, "manifests are not staged into the image"
    # A loose `pip install pkg1 pkg2 ...` block would let versions drift away
    # from the published sandbox, which is the whole point of pinning.
    assert not re.search(r"pip install[^\n]*\\\n\s+[a-z0-9-]+ [a-z0-9-]+", text), "loose pip block reintroduced"


def test_dockerfile_pins_the_interpreter_to_the_published_micro_version():
    text = _DOCKERFILE.read_text(encoding="utf-8")
    want = _pins(_APT_MANIFEST)["python3.13"].split("-", 1)[0]
    assert want == "3.13.5", f"closure pins python3.13={want}; update this test deliberately"
    major, minor, micro = want.split(".")
    assert f"({major},{minor},{micro})" in text.replace(" ", ""), (
        "the build must assert the interpreter micro version, not just 3.13"
    )


def test_arm64_exclusions_are_a_closed_documented_subset():
    excluded = _pins(_ARM64_EXCLUSIONS)
    published = _pins(_PY_MANIFEST)
    assert excluded, "the exclusion list must not be empty while arm64 builds are supported"
    for name, ver in excluded.items():
        assert name in published, f"{name} is excluded but is not in the published manifest"
        assert published[name] == ver, f"{name} exclusion pins {ver}, manifest pins {published[name]}"
    # Keep it small and deliberate: these are packages with no aarch64
    # distribution at all, not a dumping ground for build failures.
    assert len(excluded) <= 8, f"exclusion list has grown to {len(excluded)}; justify each addition"


def test_dockerfile_does_not_reintroduce_deep_learning_frameworks():
    installed = _pins(_PY_MANIFEST)
    for pkg in ("torch", "torchvision", "torchaudio", "keras", "jax", "tensorflow"):
        assert pkg not in installed, f"{pkg} is not part of GDPval-AA v2; it would add GB for nothing"


def test_verifier_script_is_staged_into_the_image():
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY verify_gdpval_sandbox.py /opt/gdpval/verify_gdpval_sandbox.py" in text
    assert (_CONTAINERS / "verify_gdpval_sandbox.py").exists()
