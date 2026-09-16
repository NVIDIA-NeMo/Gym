# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GDPval prompt must describe the sandbox gdpval.def actually builds.

The expensive failure is a prompt that advertises a package the sif does not
carry: the model plans around it and only finds out mid-trajectory. These tests
tie the prompt, the container definition, and the vendored GDPval-AA v2
manifests to each other.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


_CONTAINERS = Path(__file__).resolve().parents[1] / "containers"
_PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
_PY_MANIFEST = _CONTAINERS / "gdpval_aa_v2_python_requirements.txt"
_APT_MANIFEST = _CONTAINERS / "gdpval_aa_v2_apt_closure.txt"
_DEF = _CONTAINERS / "gdpval.def"
_ARM64_EXCLUSIONS = _CONTAINERS / "gdpval_aa_v2_arm64_exclusions.txt"


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


@pytest.mark.parametrize("has_timeout", [False, True], ids=["without-timeout", "with-timeout"])
def test_exec_backend_really_discards_shell_state_between_calls(tmp_path, monkeypatch, has_timeout):
    """Execute two provider-built commands in one parent bash; only files persist."""
    pytest.importorskip("stirrup", reason="apptainer_provider imports stirrup; runs in the per-server venv")
    from responses_api_agents.stirrup_agent import apptainer_provider

    assert Path(apptainer_provider.__file__).resolve() == _CONTAINERS.parent / "apptainer_provider.py"
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")
    if has_timeout:
        timeout = shutil.which("timeout")
        if (
            timeout is None
            or "GNU coreutils"
            not in subprocess.run([timeout, "--version"], capture_output=True, text=True, timeout=5).stdout
        ):
            pytest.skip("GNU timeout is unavailable")

    monkeypatch.setattr(apptainer_provider, "IO_MOUNT_DEST", str(tmp_path))
    monkeypatch.delenv("GDPVAL_SHELL_TEST", raising=False)
    (tmp_path / "subdir").mkdir()
    provider = apptainer_provider.ApptainerCodeExecToolProvider("/nonexistent.sif", working_dir=str(tmp_path))
    provider._has_timeout_cmd = has_timeout
    commands = [
        "cd subdir && export GDPVAL_SHELL_TEST=changed && printf 'saved\\n' > saved.txt",
        'pwd -P; printf "%s\\n" "${GDPVAL_SHELL_TEST-unset}"; cat subdir/saved.txt',
    ]
    scripts = [provider._build_command_script(cmd, 5, f".stderr_{i}", f"DONE{i}")[0] for i, cmd in enumerate(commands)]
    result = subprocess.run([bash], input="".join(scripts), capture_output=True, text=True, check=True, timeout=10)
    assert result.stdout.splitlines() == ["", "DONE0:0", str(tmp_path.resolve()), "unset", "saved", "", "DONE1:0"]


def test_prompt_tells_the_model_state_does_not_carry_over(prompt):
    """Wording check, paired with the behavioural test above.

    On its own this proves nothing -- it is the behavioural test that anchors
    it. Together they fail in opposite directions if prompt and backend drift.
    """
    runtime = prompt.split("## Reference Files")[0].lower()
    assert "every command runs independently" in runtime
    assert "does not" in runtime or "no working directory" in runtime
    # The claim that was wrong. Guard the exact phrasing so it cannot return.
    assert "shell is persistent" not in runtime
    assert "carry over from one call to the next" not in runtime or "no working directory" in runtime


@pytest.mark.integration
async def test_apptainer_shell_state_and_timeout_recovery():
    """Exercise the real provider when an existing sandbox image is supplied."""
    image = os.environ.get("GDPVAL_CONTAINER_PATH")
    if not image:
        pytest.skip("set GDPVAL_CONTAINER_PATH to run the real Apptainer check")
    assert Path(image).is_file(), f"sandbox image does not exist: {image}"
    assert shutil.which("apptainer"), "apptainer is required for the configured sandbox check"
    from responses_api_agents.stirrup_agent import apptainer_provider

    assert Path(apptainer_provider.__file__).resolve() == _CONTAINERS.parent / "apptainer_provider.py"
    provider = apptainer_provider.ApptainerCodeExecToolProvider(image, working_dir="/root", capture_git_diff=False)
    async with provider:
        assert provider._has_timeout_cmd, "the sandbox must provide GNU timeout"
        first = await provider.run_command(
            "mkdir -p /root/shell_test && cd /root/shell_test && "
            "export GDPVAL_SHELL_TEST=changed && printf 'saved\\n' > saved.txt"
        )
        assert first.exit_code == 0, first.stderr
        second = await provider.run_command(
            'pwd; printf "%s\\n" "${GDPVAL_SHELL_TEST-unset}"; cat /root/shell_test/saved.txt'
        )
        assert second.exit_code == 0, second.stderr
        assert second.stdout.strip().splitlines() == ["/root", "unset", "saved"]

        timed_out = await provider.run_command("sleep 10; echo unexpected", timeout=1)
        assert timed_out.exit_code != 0, timed_out
        assert "Command timed out after 1 seconds" in timed_out.stderr
        assert "unexpected" not in timed_out.stdout
        recovered = await provider.run_command("cat /root/shell_test/saved.txt", timeout=5)
        assert recovered.exit_code == 0, recovered.stderr
        assert recovered.stdout.strip() == "saved"

    assert provider._process is None
    assert provider._temp_dir is None


def test_verifier_script_is_staged_into_the_image():
    text = _DEF.read_text(encoding="utf-8")
    assert "verify_gdpval_sandbox.py /opt/gdpval/verify_gdpval_sandbox.py" in text
    assert (_CONTAINERS / "verify_gdpval_sandbox.py").exists()


def test_prompt_advertises_the_working_dir_the_provider_actually_uses(prompt):
    """The prompt's example path must be this sandbox's, not the reference one.

    The published Artificial Analysis prompt says `/home/user`, because their
    sandbox runs as a non-root user. GDPValTask constructs the provider with
    `working_dir="/root"`, and every command is prefixed with a `cd` to it, so
    an example rooted at /home/user sends the model to a directory that does
    not exist here.
    """
    import re

    task_src = (Path(__file__).resolve().parents[1] / "tasks" / "gdpval.py").read_text(encoding="utf-8")
    m = re.search(r'^\s*working_dir\s*=\s*"([^"]+)"', task_src, re.MULTILINE)
    assert m, "could not find the working_dir the GDPval provider is constructed with"
    working_dir = m.group(1)

    assert working_dir in prompt, f"prompt never names the real working dir {working_dir}"
    assert "/home/user" not in prompt, (
        f"prompt carries the reference sandbox's /home/user path; this sandbox uses {working_dir}"
    )
