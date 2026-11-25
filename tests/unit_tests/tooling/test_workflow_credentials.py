# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the credentials referenced by GitHub workflows and local actions."""

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
SECRET_REFERENCE = re.compile(r"\bsecrets\s*(?:\.\s*(\w+)|\[\s*([^\]]*)\])", re.IGNORECASE)
EXTERNAL_CREDENTIALS = {
    ".github/workflows/cicd-main.yml": (
        "pre-flight",
        "NVIDIA-NeMo/FW-CI-templates/.github/workflows/_cicd_preflight.yml",
        "NVIDIA_MANAGEMENT_ORG_PAT",
        "NVIDIA_MANAGEMENT_ORG_PAT",
    ),
    ".github/workflows/validate-codeowners-membership.yml": (
        "validate-codeowners-membership",
        "NVIDIA-NeMo/FW-CI-templates/.github/workflows/_validate_codeowners_membership.yml",
        "ROSTER_TOKEN",
        "NVIDIA_MANAGEMENT_ORG_PAT",
    ),
    ".github/workflows/request-nvskills-ci.yml": (
        "request",
        "NVIDIA/skills/.github/workflows/team-request.yml",
        "NVSKILLS_CI_DISPATCH_TOKEN",
        "NVSKILLS_CI_DISPATCH_TOKEN",
    ),
}


def _secret_names(dot_name: str, index: str, matrix: dict | str) -> list[str]:
    """Resolve direct names and selectors over a literal include-only job matrix."""
    if dot_name:
        return [dot_name]
    if literal := re.fullmatch(r"""(['"])(\w+)\1""", index.strip()):
        return [literal[2]]
    field = re.fullmatch(r"matrix\.(\w+)", index.strip(), re.IGNORECASE)
    if field and isinstance(matrix, dict) and matrix.keys() == {"include"} and isinstance(matrix["include"], list):
        return [row.get(field[1], "") if isinstance(row, dict) else "" for row in matrix["include"]]
    return []


def _credential_violations(node: dict | list | str, matrix: dict | str = "") -> Iterator[str]:
    """Yield inherited secrets and references to personal or retired GitHub credentials."""
    if isinstance(node, dict):
        if node.get("secrets") == "inherit":
            yield "secrets: inherit"
        if isinstance(node.get("strategy"), dict):
            matrix = node["strategy"].get("matrix", "")
        for value in node.values():
            yield from _credential_violations(value, matrix)
    elif isinstance(node, list):
        for item in node:
            yield from _credential_violations(item, matrix)
    elif isinstance(node, str):
        for expression in re.findall(r"\$\{\{(.*?)\}\}", node, flags=re.DOTALL):
            if re.search(r"\btoJSON\s*\(\s*secrets\s*\)", expression, re.IGNORECASE):
                yield "toJSON(secrets)"
            for dot_name, index in SECRET_REFERENCE.findall(expression):
                names = _secret_names(dot_name, index, matrix)
                if not names or not all(isinstance(name, str) and re.fullmatch(r"\w+", name) for name in names):
                    yield "dynamic secret reference"
                else:
                    for name in map(str.upper, names):
                        if (
                            "PAT" in name.split("_")
                            or "PERSONAL_ACCESS_TOKEN" in name
                            or name == "NVSKILLS_CI_DISPATCH_TOKEN"
                        ):
                            yield name


def _workflow_violations(path: str, definition: dict) -> Iterator[str]:
    """Exclude an explicitly approved binding only at its designated external call site."""
    if path in EXTERNAL_CREDENTIALS:
        job_id, callee, parameter, secret = EXTERNAL_CREDENTIALS[path]
        jobs = definition.get("jobs", {})
        job = jobs.get(job_id, {})
        bindings = job.get("secrets", {})
        if (
            job.get("uses", "").partition("@")[0] == callee
            and isinstance(bindings, dict)
            and bindings.get(parameter) == "${{ secrets." + secret + " }}"
        ):
            remaining = {key: value for key, value in bindings.items() if key != parameter}
            definition = {**definition, "jobs": {**jobs, job_id: {**job, "secrets": remaining}}}
    yield from _credential_violations(definition)


def test_workflow_credentials() -> None:
    """Check referenced secrets in every workflow and action manifest in the checkout."""
    paths = [*(ROOT / ".github/workflows").glob("*.y*ml"), *(ROOT / ".github/actions").rglob("action.y*ml")]
    violations = []
    for path in sorted(paths):
        relative_path = path.relative_to(ROOT).as_posix()
        definition = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        violations.extend(
            f"{relative_path}: {violation}" for violation in _workflow_violations(relative_path, definition)
        )
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize(
    ("expression", "violations"),
    [
        ("toJSON(secrets)", ["toJSON(secrets)"]),
        ("secrets[matrix.name]", ["dynamic secret reference"]),
        ("secrets['GITHUB_PAT']", ["GITHUB_PAT"]),
        ("secrets.NVSKILLS_CI_DISPATCH_TOKEN", ["NVSKILLS_CI_DISPATCH_TOKEN"]),
        ("secrets.DOCS_FERN_TOKEN || secrets['PROVIDER_PATH']", []),
    ],
)
def test_secret_expressions(expression: str, violations: list[str]) -> None:
    """Check literal, dynamic and whole-context secret references."""
    assert list(_credential_violations("${{ " + expression + " }}")) == violations


def test_matrix_secret_candidates() -> None:
    """Inspect each provider's selected secret, including added providers."""
    job = {
        "strategy": {"matrix": {"include": [{"key": "FIREWORKS"}, {"key": "GITHUB_PAT"}]}},
        "env": {"MODEL_API_KEY": "${{ secrets[matrix.key] }}"},
    }
    assert list(_credential_violations(job)) == ["GITHUB_PAT"]


@pytest.mark.parametrize("path", EXTERNAL_CREDENTIALS)
def test_external_credential_is_limited_to_its_call_site(path: str) -> None:
    """Other jobs, aliases, files and callees must not inherit an integration exception."""
    job_id, callee, parameter, secret = EXTERNAL_CREDENTIALS[path]
    reference = "${{ secrets." + secret + " }}"
    other_parameter = secret if secret != parameter else "OTHER"
    job = {"uses": callee + "@main", "secrets": {parameter: reference, other_parameter: reference}}
    workflow = {"jobs": {job_id: job, "extra": {"steps": [{"env": {"TOKEN": reference}}]}}}
    assert list(_workflow_violations(path, workflow)) == [secret] * 2
    assert list(_workflow_violations(".github/workflows/other.yml", workflow)) == [secret] * 3
    job["uses"] = "./.github/workflows/other.yml"
    assert list(_workflow_violations(path, workflow)) == [secret] * 3
