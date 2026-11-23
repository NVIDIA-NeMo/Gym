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


def _dockerfile() -> str:
    """The Dockerfile without comment lines, so only what the build executes is matched."""
    lines = _DOCKERFILE.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def _published_count(path: Path) -> int:
    """The package count the manifest's header states it was published with."""
    header = path.read_text(encoding="utf-8").split("\n\n", 1)[0]
    match = re.search(r"(\d+) (?:Python |system )?packages", header)
    assert match, f"{path.name} header states no package count"
    return int(match.group(1))


def test_python_manifest_is_the_published_set():
    pins = _pins(_PY_MANIFEST)
    assert len(pins) == _published_count(_PY_MANIFEST)
    # Spot-check anchors from the published snapshot, including the ones that
    # force an x86_64 build.
    assert pins["numpy"] == "2.4.4"
    assert pins["pymupdf"] == "1.27.2.2"
    assert pins["nvidia-nccl-cu12"] == "2.29.7"
    assert all(v and v[0].isdigit() for v in pins.values()), "a pin has no concrete version"


def test_apt_manifest_is_the_published_set_on_trixie():
    pins = _pins(_APT_MANIFEST)
    assert len(pins) == _published_count(_APT_MANIFEST)
    # The closure being trixie-pinned is why the Dockerfile uses a trixie base.
    assert any("deb13" in v for v in pins.values()), "closure is not Debian 13 pinned"
    assert pins["python3.13"].startswith("3.13"), "reference sandbox is not CPython 3.13"


def test_dockerfile_installs_the_pinned_manifest_rather_than_loose_names():
    text = _dockerfile()
    assert re.search(r"^FROM debian:trixie\b", text, re.MULTILINE), "base image must match the trixie-pinned closure"
    # Debian's interpreter, not the docker python image's later patch release.
    assert re.search(r"^\s+python3\.13 \\$", text, re.MULTILINE), "the Debian interpreter must be installed explicitly"
    assert re.search(r'^\$PIP -r "\$EFFECTIVE"$', text, re.MULTILINE), (
        "pins must be installed from a requirements file"
    )
    for manifest in (_PY_MANIFEST, _APT_MANIFEST):
        assert re.search(rf"^COPY {re.escape(manifest.name)} ", text, re.MULTILINE), f"{manifest.name} is not staged"
    # A loose `pip install pkg1 pkg2 ...` block would let versions drift away
    # from the published sandbox, which is the whole point of pinning.
    assert not re.search(r"pip install[^\n]*\\\n\s+[a-z0-9-]+ [a-z0-9-]+", text), "loose pip block reintroduced"


def test_dockerfile_pins_the_interpreter_to_the_published_micro_version():
    want = tuple(_pins(_APT_MANIFEST)["python3.13"].split("-", 1)[0].split("."))
    asserted = re.search(r"\(v\.major,\s*v\.minor,\s*v\.micro\)\s*==\s*\((\d+),\s*(\d+),\s*(\d+)\)", _dockerfile())
    assert asserted, "the build must assert the interpreter micro version, not just 3.13"
    assert asserted.groups() == want


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
    assert re.search(r"^COPY verify_gdpval_sandbox.py /opt/gdpval/verify_gdpval_sandbox.py$", _dockerfile(), re.M)
    assert (_CONTAINERS / "verify_gdpval_sandbox.py").exists()
