# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GDPval prompt must describe the sandbox the Dockerfile actually builds.

The expensive failure is a prompt that advertises a package the sif does not
carry: the model plans around it and only finds out mid-trajectory. These tests
tie the prompt templates to the vendored GDPval-AA v2 Python manifest, the
per-command timeout, and the provider's working directory.
"""

import re
from pathlib import Path

import pytest
import yaml

from resources_servers.gdpval.task_sandbox import WORKDIR


_REPO = Path(__file__).resolve().parents[3]
_CONTAINERS = Path(__file__).resolve().parents[1] / "containers"
_PROMPTS = _REPO / "benchmarks" / "gdpval" / "prompts"
_PY_MANIFEST = _CONTAINERS / "gdpval_aa_v2_python_requirements.txt"


def _user_prompt(name: str) -> str:
    return yaml.safe_load((_PROMPTS / name).read_text(encoding="utf-8"))["user"]


@pytest.fixture(params=["default-stirrup.yaml", "harness-neutral.yaml"])
def prompt(request) -> str:
    """Both GDPval user prompts as this tree defines them."""
    return _user_prompt(request.param)


@pytest.fixture
def stirrup_prompt() -> str:
    """The prompt that states Stirrup's exec contract."""
    return _user_prompt("default-stirrup.yaml")


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


# Names the prompt uses for distributions published under another name.
_DISTRIBUTION = {"opencv": "opencv-python"}


def _advertised_python_packages(prompt: str) -> list[str]:
    """The package names in the prompt's Python bullets: the parenthesised lists and the trailing 'plus X'."""
    names: list[str] = []
    for line in prompt.split("## Reference Files")[0].splitlines():
        if not line.startswith("- ") or "Python" not in line:
            continue
        for group in re.findall(r"\(([^)]*)\)", line):
            names += [name.strip() for name in group.split(",")]
        names += re.findall(r"\bplus (\w+)\b", line)
    return names


def test_prompt_only_advertises_python_packages_that_are_pinned(prompt):
    installed = _pins(_PY_MANIFEST)
    advertised = _advertised_python_packages(prompt)

    assert len(advertised) > 40, f"the prompt's package lists were not parsed: {advertised}"
    for name in advertised:
        key = _DISTRIBUTION.get(name.lower(), name.lower()).replace("_", "-")
        assert key in installed, f"{name} is advertised in the prompt but is not a pinned package"


def test_prompt_states_the_real_command_timeout(stirrup_prompt):
    try:
        from stirrup.tools.code_backends.base import SHELL_TIMEOUT
    except ImportError:
        pytest.skip("stirrup is only installed with responses_api_agents/stirrup_agent; run there to check the limit")

    assert f"{SHELL_TIMEOUT // 60} minutes" in stirrup_prompt


def test_prompt_states_a_command_limit_in_minutes(stirrup_prompt):
    assert re.search(r"[Cc]ommands are terminated after \d+ minutes", stirrup_prompt)


def test_prompt_tells_the_model_state_does_not_carry_over(stirrup_prompt):
    """Wording check for the exec contract Stirrup's sandbox provides."""
    runtime = " ".join(stirrup_prompt.split("## Reference Files")[0].lower().split())
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
    sandbox runs as a non-root user. The task sandbox's working directory is
    `WORKDIR`, and every command is prefixed with a `cd` to it, so an example
    rooted at /home/user sends the model to a directory that does not exist here.
    """
    working_dir = WORKDIR

    assert working_dir in prompt, f"prompt never names the real working dir {working_dir}"
    assert "/home/user" not in prompt, (
        f"prompt carries the reference sandbox's /home/user path; this sandbox uses {working_dir}"
    )
