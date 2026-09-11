# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GDPval prompt must describe the sandbox gdpval.def actually builds.

The expensive failure is a prompt that advertises a package the sif does not
carry: the model plans around it and only finds out mid-trajectory. These tests
tie the prompt, the container definition, and the vendored GDPval-AA v2
manifests to each other.
"""

import re
from pathlib import Path

import pytest


_CONTAINERS = Path(__file__).resolve().parents[1] / "containers"
_PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
_PY_MANIFEST = _CONTAINERS / "gdpval_aa_v2_python_requirements.txt"
_APT_MANIFEST = _CONTAINERS / "gdpval_aa_v2_apt_closure.txt"
_DEF = _CONTAINERS / "gdpval.def"
_ARM64_EXCLUSIONS = _CONTAINERS / "gdpval_aa_v2_arm64_exclusions.txt"


def _prompt() -> str:
    """The GDPval user prompt as this tree defines it.

    Read from disk rather than via ``_build_gdpval_user_prompt`` so the test
    checks *this* checkout: an editable install can resolve the package to a
    different worktree and silently validate the wrong file.
    """
    template = (_PROMPTS / "gdpval_user_prompt.txt").read_text(encoding="utf-8")
    return template.format(task="a task", reference_files="- ref.docx")


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
    # The closure being trixie-pinned is why gdpval.def uses a trixie base.
    assert any("deb13" in v for v in pins.values()), "closure is not Debian 13 pinned"
    assert pins["python3.13"].startswith("3.13"), "reference sandbox is not CPython 3.13"


def test_def_installs_the_pinned_manifest_rather_than_loose_names():
    text = _DEF.read_text(encoding="utf-8")
    assert "From: debian:trixie" in text, "base image must match the trixie-pinned closure"
    # Debian's interpreter, not the docker python image's later patch release.
    assert "python3.13 \\" in text, "the Debian interpreter must be installed explicitly"
    assert '-r "$EFFECTIVE"' in text, "pins must be installed from a requirements file"
    assert _PY_MANIFEST.name in text and _APT_MANIFEST.name in text, "manifests are not staged into the image"
    # A loose `pip install pkg1 pkg2 ...` block would let versions drift away
    # from the published sandbox, which is the whole point of pinning.
    assert not re.search(r"pip install[^\n]*\\\n\s+[a-z0-9-]+ [a-z0-9-]+", text), "loose pip block reintroduced"


def test_def_pins_the_interpreter_to_the_published_micro_version():
    text = _DEF.read_text(encoding="utf-8")
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


def test_def_does_not_reintroduce_deep_learning_frameworks():
    installed = _pins(_PY_MANIFEST)
    for pkg in ("torch", "torchvision", "torchaudio", "keras", "jax", "tensorflow"):
        assert pkg not in installed, f"{pkg} is not part of GDPval-AA v2; it would add GB for nothing"


def test_prompt_does_not_advertise_packages_the_sandbox_lacks():
    prompt = _prompt()
    installed = _pins(_PY_MANIFEST)
    # The stack summary is the part that tells the model what it *has*; the
    # trailing disclaimer deliberately names the frameworks that are absent, so
    # only the summary bullets are searched for availability claims.
    summary = "\n".join(ln for ln in prompt.split("## Reference Files")[0].splitlines() if ln.startswith("- "))
    assert summary, "the prompt lost its stack summary bullets"
    # Each of these was advertised by the GDPval-AA v1 era prompt and is absent
    # from the v2 manifest, so the sif does not have it.
    for pkg in ("torch", "keras", "jax", "dlib", "mtcnn", "pygraphviz", "pdfkit", "imgkit", "fuzzywuzzy"):
        assert pkg not in installed, f"{pkg} unexpectedly present; revisit the prompt claim"
        assert not re.search(rf"\b{re.escape(pkg)}\b", summary, re.IGNORECASE), (
            f"prompt advertises {pkg} as available, but the sandbox does not install it"
        )


def test_prompt_tells_the_model_there_is_no_deep_learning_framework():
    # Without this the model plans a torch solution, burns turns discovering the
    # gap, and submits nothing.
    prompt = _prompt()
    disclaimer = next(ln for ln in prompt.splitlines() if ln.startswith("There is no "))
    for framework in ("PyTorch", "TensorFlow", "JAX", "Keras"):
        assert framework in disclaimer, f"{framework} is not covered by the absence disclaimer"


def test_prompt_only_advertises_python_packages_that_are_pinned():
    prompt = _prompt()
    installed = _pins(_PY_MANIFEST)
    advertised = [
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "matplotlib",
        "plotly",
        "seaborn",
        "bokeh",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "statsmodels",
        "python-docx",
        "python-pptx",
        "openpyxl",
        "PyMuPDF",
        "pdfplumber",
        "reportlab",
        "weasyprint",
        "fpdf2",
        "Pillow",
        "playwright",
        "nltk",
        "spacy",
        "gensim",
        "librosa",
        "soundfile",
        "pydub",
        "moviepy",
        "av",
        "shapely",
        "geopandas",
        "fiona",
        "rasterio",
        "folium",
        "sympy",
        "pymc",
        "h5py",
        "tables",
        "rdkit",
        "biopython",
        "graphviz",
        "networkx",
        "cairosvg",
        "trimesh",
        "wordcloud",
    ]
    for name in advertised:
        key = name.lower().replace("_", "-")
        assert key in installed, f"{name} is advertised in the prompt but is not a pinned package"
        assert re.search(rf"\b{re.escape(name)}\b", prompt, re.IGNORECASE), (
            f"{name} is pinned and expected in the prompt's stack summary but is absent"
        )
    # opencv is named in prose; the distribution is opencv-python.
    assert "opencv" in prompt.lower() and "opencv-python" in installed


def test_prompt_states_the_real_command_timeout():
    prompt = _prompt()
    base = pytest.importorskip("stirrup.tools.code_backends.base")
    SHELL_TIMEOUT = base.SHELL_TIMEOUT

    minutes = SHELL_TIMEOUT // 60
    assert f"{minutes} minutes" in prompt, (
        f"prompt must state the real per-command limit ({minutes} min), not the upstream AA value"
    )


def test_prompt_describes_the_persistent_shell_not_e2b_semantics():
    prompt = _prompt()
    # Our Apptainer backend feeds commands to one long-lived bash, so state does
    # carry over. The published AA prompt says the opposite because E2B runs
    # each command independently; copying that text would misinform the model.
    assert "persistent" in prompt.lower()
    assert "carry over" in prompt.lower()


def test_verifier_script_is_staged_into_the_image():
    text = _DEF.read_text(encoding="utf-8")
    assert "verify_gdpval_sandbox.py /opt/gdpval/verify_gdpval_sandbox.py" in text
    assert (_CONTAINERS / "verify_gdpval_sandbox.py").exists()
