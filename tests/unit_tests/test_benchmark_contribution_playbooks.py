# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise documented Git journeys against local remotes, never production services."""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.cli.main import build_parser


ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "fern/versions/latest/pages/contribute/environments"


def _block(page: str, name: str) -> str:
    path = PAGES / f"{page}.mdx"
    assert path.is_file(), f"Missing contribution playbook: {path.name}"
    matches = re.findall(r'```bash title="' + re.escape(name) + r'"\n(.*?)\n```', path.read_text(), re.S)
    assert len(matches) == 1, f"Expected one executable block: {name}"
    return matches[0]


def _run(argv: list[str], cwd: Path, env: dict[str, str]) -> str:
    result = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def _step(page: str, name: str, cwd: Path, env: dict[str, str]) -> str:
    return _run(["bash", "-euo", "pipefail", "-c", _block(page, name)], cwd, env)


@pytest.mark.parametrize(
    "route,base,remote", [("public", "main", "origin"), ("private", "nv-internal-main", "internal")]
)
def test_contribution_git_journey_preserves_base_and_excludes_unlisted_files(tmp_path, route, base, remote):
    """Run the actual clone/scaffold/stage/sign-off/push instructions with synthetic data only."""
    env = os.environ.copy()
    # Ignore user credentials, signing configuration, hooks, and remote rewrites.
    for key in list(env):
        if key.startswith("GIT_"):
            del env[key]
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="Offline Contributor",
        GIT_AUTHOR_EMAIL="contributor@example.invalid",
        GIT_COMMITTER_NAME="Offline Contributor",
        GIT_COMMITTER_EMAIL="contributor@example.invalid",
        GIT_TERMINAL_PROMPT="0",
        GIT_ALLOW_PROTOCOL="file",
        PATH=f"{Path(sys.executable).parent}:{env['PATH']}",
        PYTHONPATH=str(ROOT),
    )
    seed = tmp_path / "seed"
    seed.mkdir()
    _run(["git", "init", "--initial-branch", base], seed, env)
    (seed / "README.md").write_text("Synthetic repository for offline documentation tests.\n")
    _run(["git", "add", "README.md"], seed, env)
    _run(["git", "commit", "-m", "Initialize offline fixture"], seed, env)
    base_sha = _run(["git", "rev-parse", "HEAD"], seed, env)
    upstream = tmp_path / "upstream.git"
    fork = tmp_path / "fork.git"
    _run(["git", "clone", "--bare", str(seed), str(upstream)], tmp_path, env)
    _run(["git", "clone", "--bare", str(seed), str(fork)], tmp_path, env)
    env.update(
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0=f"url.{upstream}.insteadOf",
        GIT_CONFIG_VALUE_0="https://github.com/NVIDIA-NeMo/Gym.git",
        GYM_PUBLIC_FORK_URL=str(fork),
        GYM_INTERNAL_REPO_URL=str(upstream),
    )
    page = f"benchmark-{route}-contribution"
    _step(page, f"{route}-branch", tmp_path, env)
    checkout = tmp_path / f"Gym-{route}"
    env["NEMO_GYM_EXTRA_ROOTS"] = str(checkout)
    assert _run(["git", "branch", "--show-current"], checkout, env) == "benchmark/safety_example"
    assert _run(["git", "rev-parse", "HEAD"], checkout, env) == base_sha
    _step("benchmark-contribution", "contribution-scaffold", checkout, env)

    # Preparation must replace the already-present arithmetic control, not reuse a stale cache.
    data = checkout / "benchmarks/safety_example/data"
    row = {"question": "What is 5 + 3?", "expected_answer": "8"}
    (data / "source.jsonl").write_text(json.dumps(row) + "\n")
    _step("benchmark-contribution", "contribution-prepare", checkout, env)
    assert [json.loads(line) for line in (data / "example.jsonl").read_text().splitlines()] == [row]

    # A private source dump beside the fixture and unrelated edits must not be staged.
    (checkout / "benchmarks/safety_example/private-dump.jsonl").write_text('"PRIVATE_CANARY"\n')
    (checkout / "README.md").write_text("Unrelated user edit.\n")
    _step("benchmark-contribution", "contribution-stage", checkout, env)
    staged = _run(["git", "diff", "--cached", "--name-only"], checkout, env).splitlines()
    assert len(staged) == 17
    assert "benchmarks/safety_example/manifest.yaml" in staged
    assert "resources_servers/safety_example/tests/verifier_cases.jsonl" in staged
    assert "benchmarks/safety_example/private-dump.jsonl" not in staged
    assert "README.md" not in staged
    _step("benchmark-contribution", "contribution-commit", checkout, env)
    assert "Signed-off-by: Offline Contributor <contributor@example.invalid>" in _run(
        ["git", "log", "-1", "--format=%B"], checkout, env
    )
    _step(page, f"{route}-push", checkout, env)
    destination = fork if route == "public" else upstream
    head = _run(["git", "rev-parse", "HEAD"], checkout, env)
    assert _run(["git", "rev-parse", "refs/heads/benchmark/safety_example"], destination, env) == head
    assert _run(["git", "rev-parse", f"refs/heads/{base}"], destination, env) == base_sha
    assert "PRIVATE_CANARY" not in _run(["git", "show", "HEAD"], checkout, env)
    assert _run(["git", "config", "branch.benchmark/safety_example.remote"], checkout, env) == remote

    # A net diff hides content added and then deleted; the documented review must reveal it.
    history_file = checkout / "history-control.txt"
    history_file.write_text("SYNTHETIC_HISTORY_CANARY\n")
    _run(["git", "add", "history-control.txt"], checkout, env)
    _run(["git", "commit", "-m", "Add synthetic history control"], checkout, env)
    history_file.unlink()
    _run(["git", "add", "history-control.txt"], checkout, env)
    _run(["git", "commit", "-m", "Remove synthetic history control"], checkout, env)
    comparison = "upstream/main" if route == "public" else "internal/nv-internal-main"
    assert not _run(["git", "diff", f"{comparison}...HEAD", "--", "history-control.txt"], checkout, env)
    assert "SYNTHETIC_HISTORY_CANARY" in _step(page, f"{route}-review", checkout, env)


def test_private_registry_download_uses_gitlab_and_loads_external_config_without_dispatch():
    command = _block("benchmark-private-contribution", "private-registry-download")
    # Substitute only a non-secret filename; never execute the download.
    command = command.replace(
        "${GYM_DATASET_CONFIG:?Set the private registry config path first}", "/private/config with spaces.yaml"
    )
    tokens = shlex.split(command.replace("\\\n", " "))
    assert tokens[:3] == ["gym", "dataset", "download"]
    args, overrides = build_parser().parse_known_args(tokens[1:])
    assert args.storage == "gitlab"
    assert args.name == "safety_example"
    assert args.revision == "0.0.1"
    assert args.artifact == "example.jsonl"
    assert len(overrides) == 1 and overrides[0].startswith("+config_paths=")
    assert OmegaConf.from_dotlist([overrides[0].removeprefix("+")]).config_paths == [
        "/private/config with spaces.yaml"
    ]


def test_recipe_index_routes_contributors_before_optional_adapters():
    text = (ROOT / "fern/versions/latest/pages/get-started/benchmark-onboarding.mdx").read_text()
    for route in ("public", "private"):
        link = f"/main/contribute/environments/benchmark-{route}-contribution"
        assert link in text, f"Missing first-class {route} contribution route"
        assert text.index(link) < text.index("## Validation scope")
    assert "optional" in text.lower()
