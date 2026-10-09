# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GDPval prompt must describe the sandbox gdpval.def actually builds.

The expensive failure is a prompt that advertises a package the sif does not
carry: the model plans around it and only finds out mid-trajectory. These tests
tie the prompt templates to the vendored GDPval-AA v2 Python manifest, the
per-command timeout, and the provider's working directory.
"""

import re
from pathlib import Path

import pytest


_CONTAINERS = Path(__file__).resolve().parents[1] / "containers"
_PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
_PY_MANIFEST = _CONTAINERS / "gdpval_aa_v2_python_requirements.txt"


@pytest.fixture(params=["gdpval_user_prompt.txt", "user_prompt.j2"])
def prompt(request) -> str:
    """Both GDPval user prompt templates as this tree defines them.

    Read from disk rather than via ``_build_gdpval_user_prompt`` so the test
    checks *this* checkout: an editable install can resolve the package to a
    different worktree and silently validate the wrong file.
    """
    return (_PROMPTS / request.param).read_text(encoding="utf-8")


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


def test_prompt_does_not_advertise_packages_the_sandbox_lacks(prompt):
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


def test_prompt_tells_the_model_there_is_no_deep_learning_framework(prompt):
    # Without this the model plans a torch solution, burns turns discovering the
    # gap, and submits nothing.
    disclaimer = next(ln for ln in prompt.splitlines() if ln.startswith("There is no "))
    for framework in ("PyTorch", "TensorFlow", "JAX", "Keras"):
        assert framework in disclaimer, f"{framework} is not covered by the absence disclaimer"


def test_prompt_only_advertises_python_packages_that_are_pinned(prompt):
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


def test_prompt_states_the_real_command_timeout(prompt):
    base = pytest.importorskip("stirrup.tools.code_backends.base")
    SHELL_TIMEOUT = base.SHELL_TIMEOUT

    minutes = SHELL_TIMEOUT // 60
    assert f"{minutes} minutes" in prompt, (
        f"prompt must state the real per-command limit ({minutes} min), not the upstream AA value"
    )


def test_prompt_tells_the_model_state_does_not_carry_over(prompt):
    """Wording check, paired with the behavioural test
    ``test_exec_backend_really_discards_shell_state_between_calls`` in
    ``test_gdpval_sandbox_alignment.py``.

    On its own this proves nothing -- it is the behavioural test that anchors
    it. Together they fail in opposite directions if prompt and backend drift.
    """
    runtime = " ".join(prompt.split("## Reference Files")[0].lower().split())
    assert (
        "every command runs independently: no working directory, environment variable, "
        "or other shell state carries over from one call to the next." in runtime
    )
    assert "files you write do persist." in runtime
    # The claim that was wrong. Guard the exact phrasing so it cannot return.
    assert "shell is persistent" not in runtime


def test_prompt_advertises_the_working_dir_the_provider_actually_uses(prompt):
    """The prompt's example path must be this sandbox's, not the reference one.

    The published Artificial Analysis prompt says `/home/user`, because their
    sandbox runs as a non-root user. GDPValTask constructs the provider with
    `working_dir="/root"`, and every command is prefixed with a `cd` to it, so
    an example rooted at /home/user sends the model to a directory that does
    not exist here.
    """
    task_src = (Path(__file__).resolve().parents[1] / "tasks" / "gdpval.py").read_text(encoding="utf-8")
    m = re.search(r'^\s*working_dir\s*=\s*"([^"]+)"', task_src, re.MULTILINE)
    assert m, "could not find the working_dir the GDPval provider is constructed with"
    working_dir = m.group(1)

    assert working_dir in prompt, f"prompt never names the real working dir {working_dir}"
    assert "/home/user" not in prompt, (
        f"prompt carries the reference sandbox's /home/user path; this sandbox uses {working_dir}"
    )
