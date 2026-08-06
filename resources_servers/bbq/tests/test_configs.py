from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_main_config_is_prototype_only() -> None:
    config = yaml.safe_load((ROOT / "configs/bbq_two_judge.yaml").read_text(encoding="utf-8"))
    server = config["bbq_two_judge"]["resources_servers"]["bbq_two_judge"]
    assert server["verified"] is False
    assert server["judge_model_server"]["name"] == "bbq_judge_model"
    assert "grpo" not in str(config).lower()
    assert "training" not in config


def test_judge_config_is_environment_backed_and_contains_no_secret() -> None:
    text = (ROOT / "configs/bbq_judge_model.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    judge = config["bbq_judge_model"]["responses_api_models"]["vllm_model"]
    assert "BBQ_JUDGE_BASE_URL" in judge["base_url"]
    assert "BBQ_JUDGE_API_KEY" in judge["api_key"]
    assert "BBQ_JUDGE_MODEL" in judge["model"]
    assert judge["base_url"] == "${oc.env:BBQ_JUDGE_BASE_URL}"
    assert judge["api_key"] == "${oc.env:BBQ_JUDGE_API_KEY}"
    # Model is env-overridable but defaults to the calibrated judge.
    assert judge["model"] == "${oc.env:BBQ_JUDGE_MODEL,nvidia/google/gemma-4-31b-it}"
    assert "<api_key>" not in text
    assert "http://<host_url>" not in text


def test_prompt_contract_uses_existing_plural_metadata() -> None:
    text = (ROOT / "configs/verifier_prompt_templates.yaml").read_text(encoding="utf-8")
    prompts = yaml.safe_load(text)
    assert set(prompts) == {"answer_check", "explanation_quality_check"}
    assert "{{label}}" in text
    assert "{{actor_aliases_json}}" in text
    assert "{{protected_axis}}" in text
    assert "expected_answer" not in text
    assert "stereotyped_actor_id" not in text
    assert "actor_registry" not in text


def test_no_git_dependency_or_hash_pin() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "git+" not in requirements
    assert "917c3d" not in requirements


def test_requirements_do_not_pin_nemo_gym() -> None:
    # nemo-gym is not on PyPI; it must come from an editable source-checkout
    # reference (-e nemo-gym @ ../../, the upstream convention, resolved at
    # runtime placement), never from a PyPI-style version pin.
    # requirements-dev.txt is optional (dev tooling comes from the
    # nemo-gym[dev] extra); check it only where present.
    for path in ("requirements.txt", "requirements-dev.txt"):
        if not (ROOT / path).exists():
            continue
        lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
        for line in lines:
            spec = line.split("#", 1)[0].strip()
            assert not spec.lower().startswith("nemo-gym"), f"{path} must not pin nemo-gym"
