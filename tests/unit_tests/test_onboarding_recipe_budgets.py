# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard documented sample budgets; never execute the model-facing commands."""

import re
import shlex
from importlib.metadata import version
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from nemo_gym.cli.main import build_parser


ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "fern/versions/latest/pages/get-started"
HARBOR_CONFIG = ROOT / "responses_api_agents/harbor_agent/example/onboarding/harbor_onboarding.yaml"
TOKEN_CAPS = {"native": 128, "verifiers": 256, "harbor": 512}


def _documented_runs(markdown: str) -> list[list[str]]:
    commands = []
    for block in re.findall(r"^```(?:bash|sh|shell|zsh)[^\n]*\n(.*?)^```", markdown, re.M | re.S):
        # Join shell continuations before parsing; do not expand variables or execute anything.
        for line in re.sub(r"\\\r?\n", " ", block).splitlines():
            if not re.search(r"\bgym\s+eval\s+run\b", line):
                continue
            tokens = shlex.split(line, comments=True)
            if not tokens:
                continue
            assert tokens[:3] == ["gym", "eval", "run"], f"Review wrapped run before skipping it: {line}"
            commands.append(tokens)
    return commands


def _assert_budget(command: list[str], token_cap: int) -> None:
    assert command[:3] == ["gym", "eval", "run"]
    args, overrides = build_parser().parse_known_args(command[1:])
    for name, expected in (
        ("limit", 1),
        ("num_repeats", 1),
        ("concurrency", 1),
        ("max_output_tokens", token_cap),
    ):
        assert getattr(args, name) == str(expected), f"{name} must be explicitly {expected}: {command}"

    # Raw Hydra overrides take precedence over flags. Review any new one instead of
    # allowing a later +limit / +responses_create_params override to defeat this guard.
    allowed_overrides = {"uv_venv_dir", "skip_venv_if_present"}
    for override in overrides:
        key, separator, _ = override.lstrip("+").partition("=")
        assert override.startswith("+") and separator and key in allowed_overrides, (
            f"Unreviewed override could bypass the documented budget: {override}"
        )


@pytest.mark.parametrize(("route", "token_cap"), TOKEN_CAPS.items())
def test_every_documented_eval_run_has_explicit_smoke_limits(route: str, token_cap: int) -> None:
    commands = _documented_runs((PAGES / f"{route}-onboarding.mdx").read_text())
    assert commands, f"No executable gym eval run command found for {route}; update the budget guard."
    for command in commands:
        _assert_budget(command, token_cap)


@pytest.mark.parametrize("flag", ["--limit", "--num-repeats", "--concurrency", "--max-output-tokens"])
def test_budget_guard_rejects_missing_limits(flag: str) -> None:
    command = shlex.split("gym eval run --limit 1 --num-repeats 1 --concurrency 1 --max-output-tokens 128")
    index = command.index(flag)
    del command[index : index + 2]
    with pytest.raises(AssertionError, match="must be explicitly"):
        _assert_budget(command, 128)


@pytest.mark.parametrize(
    "change",
    ["--limit 2", "--max-output-tokens 512", "+limit=2", "+responses_create_params={max_output_tokens:4096}"],
)
def test_budget_guard_rejects_raised_or_bypassed_limits(change: str) -> None:
    command = shlex.split("gym eval run --limit 1 --num-repeats 1 --concurrency 1 --max-output-tokens 128 " + change)
    with pytest.raises(AssertionError):
        _assert_budget(command, 128)


def test_budget_guard_extracts_every_continued_command() -> None:
    markdown = '```bash title="run"\nexport UNUSED=1\ngym eval run \\\n  --limit 1\ngym eval run --limit 2\n```'
    assert _documented_runs(markdown) == [
        ["gym", "eval", "run", "--limit", "1"],
        ["gym", "eval", "run", "--limit", "2"],
    ]


def test_budget_guard_does_not_silently_skip_wrapped_commands() -> None:
    with pytest.raises(AssertionError, match="Review wrapped run"):
        _documented_runs("```bash\nuv run gym eval run --limit 100\n```")


def test_tracked_adapter_configs_keep_their_turn_and_token_caps() -> None:
    harbor = yaml.safe_load(HARBOR_CONFIG.read_text())["harbor_onboarding"]["responses_api_agents"]["harbor_agent"]
    assert harbor["concurrency"] == 1
    assert harbor["harbor_agent_kwargs"]["max_turns"] == 3
    assert harbor["harbor_agent_kwargs"]["model_info"]["max_output_tokens"] == TOKEN_CAPS["harbor"]
    verifiers_path = ROOT / "responses_api_agents/verifiers_agent/examples/onboarding/config.yaml"
    verifiers = yaml.safe_load(verifiers_path.read_text())["verifiers_onboarding"]["responses_api_agents"]
    assert verifiers["verifiers_agent"]["max_tokens"] == TOKEN_CAPS["verifiers"]


def test_harbor_job_budget_has_one_attempt_and_zero_job_retries(monkeypatch, tmp_path: Path) -> None:
    """This is a job-config check, not proof that LLM or transport retries are disabled."""
    pytest.importorskip("harbor")
    assert version("harbor") == "0.1.42", "Use the pinned Harbor adapter dependencies for this check."
    from responses_api_agents.harbor_agent.app import HarborAgent, HarborAgentConfig

    monkeypatch.setenv("GYM_REPO_ROOT", str(ROOT))
    monkeypatch.setenv("GYM_HARBOR_RUN_DIR", str(tmp_path))
    config = OmegaConf.to_container(OmegaConf.load(HARBOR_CONFIG), resolve=True)
    values = config["harbor_onboarding"]["responses_api_agents"]["harbor_agent"]
    agent = HarborAgent.model_construct(
        config=HarborAgentConfig(name="harbor_onboarding", host="127.0.0.1", port=8080, **values)
    )
    job = agent._build_job_config(
        dataset_alias="onboarding",
        task_name="write-answer",
        model_name="offline-budget-check",
        api_base="http://127.0.0.1:9000/v1",
        job_name="offline-budget-check",
        jobs_dir=tmp_path,
        responses_create_params={"max_output_tokens": TOKEN_CAPS["harbor"]},
    )
    assert job["agents"][0]["kwargs"]["max_turns"] == 3
    assert job["orchestrator"]["n_concurrent_trials"] == 1
    assert job["orchestrator"]["retry"]["max_retries"] == 0
    assert job["n_attempts"] == 1
