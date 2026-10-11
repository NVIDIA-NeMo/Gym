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

"""Per-repo/version evaluation specs for the SWE-Gym repos, plus the pytest log parser.

Vendored from the SWE-bench harness fork the ``swe_agents`` evaluator already uses
(https://github.com/HeyyyyyyG/SWE-bench, MIT; ``swebench/harness/constants/python.py``,
``test_spec/python.py`` and ``log_parsers/python.py``), restricted to the 200 (repo, version) pairs
that occur in SWE-Gym/SWE-Gym. Vendoring rather than importing keeps the grader pinned: an upstream
spec change cannot silently move scores between runs, and the server needs no clone at start-up.

Each entry carries the fields the eval script consumes: ``test_cmd`` (always pytest here), the
``install`` step the official harness re-runs before testing, and optional ``eval_commands``. Most
repos use one entry for every version, so entries are stored once per repo and versions index them.
"""

from __future__ import annotations

import re
from typing import Any


# Statuses pytest prints with ``-rA``; a test counts as passing when PASSED or XFAIL (SWE-bench rule).
TEST_STATUSES = ("FAILED", "PASSED", "SKIPPED", "ERROR", "XFAIL")
PASSING_STATUSES = frozenset({"PASSED", "XFAIL"})

# Files a test patch may touch that are data rather than tests, so they are never passed to pytest.
NON_TEST_EXTS = (".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg", ".pkl", ".yml", ".yaml", ".toml")

_ENTRIES: dict[str, list[dict[str, Any]]] = {
    "Project-MONAI/MONAI": [
        {
            "install": "sed -i '/^git+https:\\/\\/github.com\\/Project-MONAI\\//d' "
            "requirements-dev.txt; python -m pip install types-pkg-resources==0.1.3 "
            "pytest; pip install -r requirements-dev.txt;python setup.py develop;",
            "test_cmd": "pytest -rA ",
        }
    ],
    "bokeh/bokeh": [
        {
            "install": "python -m pip install -e .; python -m pip install bokeh_sampledata;",
            "test_cmd": "pytest -rA -n0",
        }
    ],
    "conan-io/conan": [
        {
            "eval_commands": ["export PYTHONPATH=${PYTHONPATH:-}:$(pwd)"],
            "install": "echo 'cython<3' > /tmp/constraint.txt; export "
            "PIP_CONSTRAINT=/tmp/constraint.txt; python -m pip install -r "
            "conans/requirements.txt; python -m pip install -r "
            "conans/requirements_server.txt; python -m pip install -r "
            "conans/requirements_dev.txt ",
            "test_cmd": "pytest -n0 -rA",
        },
        {
            "eval_commands": ["export PYTHONPATH=${PYTHONPATH:-}:$(pwd)"],
            "install": "python -m pip install -r conans/requirements.txt; python -m pip install -r "
            "conans/requirements_server.txt; python -m pip install -r "
            "conans/requirements_dev.txt ",
            "test_cmd": "pytest -n0 -rA",
        },
    ],
    "dask/dask": [{"install": "python -m pip install --no-deps -e .", "test_cmd": "pytest -n0 -rA  --color=no"}],
    "facebookresearch/hydra": [
        {
            "install": "sed -i "
            "'s|isort@git+git://github.com/timothycrosley/isort|isort@git+https://github.com/timothycrosley/isort|g' "
            "requirements/dev.txt; { tail -n1 requirements/requirements.txt | "
            'grep -q "." && echo ""; } >> requirements/requirements.txt; echo '
            '"pip==24.0" >> requirements/requirements.txt;pip install '
            '"pip==24.0"; pip install -r requirements/dev.txt; pip install -e .;',
            "test_cmd": "pytest -rA --tb=long",
        },
        {"install": "pip install -r requirements/dev.txt; pip install -e .;", "test_cmd": "pytest -rA --tb=long"},
    ],
    "getmoto/moto": [{"install": "make init", "test_cmd": "pytest -n0 -rA"}],
    "iterative/dvc": [
        {
            "install": "python -m pip install --upgrade pip wheel GitPython; python -m pip install "
            '"cython<3.0.0" && python -m pip install --no-build-isolation pyyaml==5.4.1; '
            "python -m pip install git+https://github.com/iterative/mock-ssh-server.git "
            "|| true; python -m pip install -r tests/requirements.txt || true; python -m "
            "pip install -r test-requirements.txt || true; python -m pip install -e "
            '".[tests,dev,all_remotes,all,testing]"; python -m pip install "numpy<=1.20"; '
            'python -m pip install "pytest<8";',
            "test_cmd": "pytest -rA",
        },
        {
            "install": "python -m pip install --upgrade pip wheel GitPython; python -m pip install "
            '"cython<3.0.0" && python -m pip install --no-build-isolation pyyaml==5.4.1; '
            "python -m pip install git+https://github.com/iterative/mock-ssh-server.git "
            "|| true; python -m pip install -r tests/requirements.txt || true; python -m "
            "pip install -r test-requirements.txt || true; python -m pip install -e "
            '".[tests,dev,all_remotes,all,testing]";',
            "test_cmd": "pytest -rA",
        },
    ],
    "modin-project/modin": [{"install": "python -m pip install -e .;", "test_cmd": "pytest -n0 -rA"}],
    "pandas-dev/pandas": [
        {
            "install": "unset CFLAGS; unset LDFLAGS; unset CPPFLAGS; python -m pip install "
            "'numpy<2'; python -m pip install -ve . --no-build-isolation "
            "-Ceditable-verbose=true; pip uninstall pytest-qt -y;",
            "test_cmd": "pytest -rA --tb=long",
        },
        {
            "install": "unset CFLAGS; unset LDFLAGS; unset CPPFLAGS; python -m pip install -ve . "
            "--no-build-isolation -Ceditable-verbose=true; pip uninstall pytest-qt "
            "-y;",
            "test_cmd": "pytest -rA --tb=long",
        },
    ],
    "pydantic/pydantic": [
        {
            "install": 'export PATH="$HOME/.local/bin:$PATH"; pdm add pre-commit; make install;',
            "test_cmd": "pytest -rA --tb=short -vv -o console_output_style=classic --no-header",
        }
    ],
    "python/mypy": [
        {
            "install": "python -m pip install -r test-requirements.txt; python -m pip install -e .; "
            "pip install pytest pytest-xdist; hash -r;",
            "test_cmd": "pytest -n0 -rA -k",
        },
        {
            "install": "python -m pip install -r test-requirements.txt; python -m pip install -e .; "
            "pip install pytest pytest-xdist; hash -r",
            "test_cmd": "pytest -n0 -rA -k",
        },
        {
            "install": "python -m pip install -r test-requirements.txt; python -m pip install -e .; hash -r",
            "test_cmd": "pytest -n0 -rA -k",
        },
        {
            "install": "python -m pip install -r test-requirements.txt; python -m pip install -e .; hash -r",
            "test_cmd": "pytest -rA -k",
        },
    ],
}

_VERSION_TO_ENTRY: dict[str, dict[str, int]] = {
    "Project-MONAI/MONAI": {
        "0.1": 0,
        "0.2": 0,
        "0.3": 0,
        "0.4": 0,
        "0.5": 0,
        "0.6": 0,
        "0.7": 0,
        "0.8": 0,
        "0.9": 0,
        "1.0": 0,
        "1.1": 0,
        "1.2": 0,
        "1.3": 0,
    },
    "bokeh/bokeh": {"3.0": 0, "3.3": 0, "3.4": 0, "3.5": 0},
    "conan-io/conan": {
        "1.33": 0,
        "1.38": 0,
        "1.40": 0,
        "1.44": 0,
        "1.45": 0,
        "1.46": 0,
        "1.47": 0,
        "1.48": 0,
        "1.49": 0,
        "1.50": 0,
        "1.51": 0,
        "1.52": 0,
        "1.53": 0,
        "1.54": 0,
        "1.55": 0,
        "1.57": 0,
        "1.60": 1,
        "1.61": 1,
        "2.0": 0,
        "2.1": 1,
        "2.2": 1,
        "2.3": 1,
        "2.4": 1,
    },
    "dask/dask": {
        "2.25": 0,
        "2.27": 0,
        "2.28": 0,
        "2.30": 0,
        "2020.12": 0,
        "2021.01": 0,
        "2021.02": 0,
        "2021.03": 0,
        "2021.04": 0,
        "2021.05": 0,
        "2021.07": 0,
        "2021.08": 0,
        "2021.09": 0,
        "2021.10": 0,
        "2021.11": 0,
        "2021.12": 0,
        "2022.01": 0,
        "2022.02": 0,
        "2022.03": 0,
        "2022.04": 0,
        "2022.05": 0,
        "2022.12": 0,
        "2022.6": 0,
        "2022.7": 0,
        "2022.8": 0,
        "2022.9": 0,
        "2023.1": 0,
        "2023.10": 0,
        "2023.11": 0,
        "2023.12": 0,
        "2023.2": 0,
        "2023.3": 0,
        "2023.4": 0,
        "2023.5": 0,
        "2023.6": 0,
        "2023.7": 0,
        "2023.8": 0,
        "2023.9": 0,
        "2024.1": 0,
        "2024.2": 0,
        "2024.3": 0,
        "2024.4": 0,
        "2024.5": 0,
    },
    "facebookresearch/hydra": {"1.0": 0, "1.1": 0, "1.2": 0, "1.3": 1, "1.4": 1},
    "getmoto/moto": {"3.0": 0, "3.1": 0, "4.0": 0, "4.1": 0, "4.2": 0, "5.0": 0},
    "iterative/dvc": {
        "0.27": 0,
        "0.28": 0,
        "0.29": 0,
        "0.30": 0,
        "0.31": 0,
        "0.32": 0,
        "0.33": 0,
        "0.34": 0,
        "0.35": 0,
        "0.40": 0,
        "0.41": 0,
        "0.50": 0,
        "0.51": 0,
        "0.52": 0,
        "0.89": 0,
        "0.90": 0,
        "0.91": 0,
        "0.92": 0,
        "0.93": 0,
        "1.0": 0,
        "1.1": 0,
        "1.10": 0,
        "1.11": 0,
        "1.3": 0,
        "1.4": 0,
        "1.6": 0,
        "1.7": 0,
        "1.8": 0,
        "1.9": 0,
        "2.0": 0,
        "2.1": 0,
        "2.19": 0,
        "2.20": 0,
        "2.21": 0,
        "2.24": 0,
        "2.27": 0,
        "2.28": 0,
        "2.45": 0,
        "2.5": 0,
        "2.50": 0,
        "2.51": 0,
        "2.52": 0,
        "2.55": 0,
        "2.56": 0,
        "2.58": 0,
        "2.6": 0,
        "2.7": 0,
        "2.8": 0,
        "3.0": 0,
        "3.1": 0,
        "3.10": 1,
        "3.12": 1,
        "3.13": 1,
        "3.15": 1,
        "3.17": 1,
        "3.37": 1,
        "3.4": 1,
        "3.43": 1,
        "3.48": 1,
        "3.49": 1,
        "3.6": 1,
    },
    "modin-project/modin": {
        "0.20": 0,
        "0.22": 0,
        "0.23": 0,
        "0.24": 0,
        "0.25": 0,
        "0.26": 0,
        "0.27": 0,
        "0.28": 0,
        "0.29": 0,
    },
    "pandas-dev/pandas": {"1.5": 0, "2.0": 0, "2.1": 0, "2.2": 1, "3.0": 1},
    "pydantic/pydantic": {
        "2.0": 0,
        "2.01": 0,
        "2.02": 0,
        "2.03": 0,
        "2.04": 0,
        "2.4": 0,
        "2.5": 0,
        "2.6": 0,
        "2.7": 0,
    },
    "python/mypy": {
        "0.800": 0,
        "0.810": 0,
        "0.820": 0,
        "0.910": 0,
        "0.920": 0,
        "0.940": 1,
        "0.950": 1,
        "0.960": 1,
        "0.970": 1,
        "0.980": 1,
        "0.990": 1,
        "1.0": 2,
        "1.10": 3,
        "1.11": 3,
        "1.2": 2,
        "1.3": 2,
        "1.4": 2,
        "1.5": 2,
        "1.6": 2,
        "1.7": 3,
        "1.8": 3,
        "1.9": 3,
    },
}

SPECS: dict[str, dict[str, dict[str, Any]]] = {
    repo: {version: _ENTRIES[repo][i] for version, i in versions.items()}
    for repo, versions in _VERSION_TO_ENTRY.items()
}


def spec_for(repo: str, version: str) -> dict[str, Any]:
    """The eval spec for one row, or a KeyError naming the gap (a row we cannot grade)."""
    try:
        return SPECS[repo][str(version)]
    except KeyError as exc:
        raise KeyError(f"no SWE-bench spec for {repo} version {version!r}") from exc


def touched_test_files(repo: str, test_patch: str) -> list[str]:
    """Test files the held-out patch touches; what the harness hands to pytest."""
    directives = re.findall(r"diff --git a/.* b/(.*)", test_patch)
    directives = [d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]
    if repo == "django/django":  # not in SWE-Gym; kept so the helper matches the harness exactly
        directives = [d[: -len(".py")] if d.endswith(".py") else d for d in directives]
        directives = [d[len("tests/") :] if d.startswith("tests/") else d for d in directives]
        directives = [d.replace("/", ".") for d in directives]
    return directives


def pytest_command(repo: str, version: str, test_patch: str) -> str:
    """``test_cmd`` plus its targets.

    mypy's data-driven suites are selected by ``[case name]`` keys from the test patch rather than by
    file (its ``test_cmd`` ends in ``-k`` for exactly that reason); every other repo takes the touched
    test files.
    """
    cmd = spec_for(repo, version)["test_cmd"]
    if repo == "python/mypy":
        cases = re.findall(r"\[case ([^\]]+)\]", test_patch)
        if cases:
            return f'{cmd} "{" or ".join(cases)}"'
        # A mypy test patch that adds plain pytest files has no [case] keys. The harness recipe would then
        # emit `-k <path>`, which pytest rejects ("unexpected character /"), so run the touched files instead.
        cmd = cmd.removesuffix(" -k").rstrip()
    return " ".join([cmd, *touched_test_files(repo, test_patch)])


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ESCAPE = re.compile(r"\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|[nrt])")


def normalize_test_id(test_id: str) -> str:
    """One spelling for a test id on both sides of the comparison.

    pytest escapes non-ASCII and control characters in parameter ids (``test_x[un\\xeecode]``), while
    SWE-Gym's FAIL_TO_PASS / PASS_TO_PASS lists carry the decoded characters (``test_x[unîcode]``). Decoding
    the escapes here makes the two agree; an id that already holds the characters is unchanged.
    """

    def decode(match: re.Match) -> str:
        body = match.group(0)[1:]
        if body[0] in "xuU":
            return chr(int(body[1:], 16))
        return {"n": "\n", "r": "\r", "t": "\t"}[body]

    return _ESCAPE.sub(decode, test_id)


def parse_log_pytest(log: str) -> dict[str, str]:
    """Test id -> status from pytest output.

    Reads both line shapes pytest produces: ``STATUS name [- reason]`` from the ``-rA`` summary (what
    SWE-bench's ``parse_log_pytest`` reads) and ``name STATUS`` from ``-v`` progress lines (SWE-bench's
    ``parse_log_pytest_v2`` fallback). The second matters for pydantic, whose pytest-pretty plugin
    replaces the summary block, so the verbose lines are the only per-test record in the log.
    """
    statuses: dict[str, str] = {}
    for raw in log.split("\n"):
        line = _ANSI.sub("", raw).strip()
        if any(line.startswith(status) for status in TEST_STATUSES):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) <= 1:
                continue
            statuses[parts[1]] = parts[0]
            continue
        for status in TEST_STATUSES:
            if line.endswith(" " + status):
                # Whitespace-split like the harness: SWE-Gym's FAIL_TO_PASS / PASS_TO_PASS ids were produced that
                # way, so an id with spaces inside its parameters is recorded truncated at the first space.
                parts = line[: -len(status)].split()
                if parts and "::" in parts[0]:
                    statuses[parts[0]] = status
                break
    return statuses
