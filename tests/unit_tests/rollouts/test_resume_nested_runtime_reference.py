# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A runtime reference nested in another interpolation keeps the resume identity resolvable.

Judge-model configs fall the judge endpoint back to the policy endpoint with
``${oc.select:judge_base_url,${policy_base_url}}``. Normalizing the inner ``${policy_base_url}``
to a bare ``<runtime:...>`` marker left the surrounding ``oc.select`` unparseable
(``GrammarParseError: token recognition error at: '<'``) when ``_configuration_identity``
resolved it. The marker is now quoted only when nested, so the config still resolves, while a
top-level reference is replaced byte-for-byte as before and already-persisted digests are unchanged.
"""

import json
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.rollouts.rollout_recovery import (
    RunManifest,
    _configuration_identity,
    _credential_safe,
    _normalize_runtime_references,
)
from tests.unit_tests.rollouts.test_rollout_collection import failing_row


REPO_ROOT = Path(__file__).parents[3]
POLICY = {
    "policy_base_url": "http://policy:8000/v1",
    "policy_api_key": "sk-policy-canary",  # pragma: allowlist secret
    "policy_model_name": "policy-model",
}


def _judge_fallback_config() -> dict:
    """The shipped terminus_judge judge-model server plus the policy values it falls back to."""
    servers = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "resources_servers/terminus_judge/configs/judge_model.yaml"),
        resolve=False,
    )
    servers.update(POLICY)
    return servers


def test_nested_runtime_reference_is_quoted_and_resolves() -> None:
    """The inner reference becomes a quoted marker that OmegaConf can still parse and resolve."""
    normalized = _normalize_runtime_references("${oc.select:judge_base_url,${policy_base_url}}")
    assert normalized == "${oc.select:judge_base_url,'<runtime:policy_base_url>'}"
    # The quoted marker is valid grammar: it resolves to the plain marker string.
    resolved = OmegaConf.to_container(OmegaConf.create({"v": normalized}), resolve=True)
    assert resolved["v"] == "<runtime:policy_base_url>"


def test_top_level_runtime_reference_stays_an_unquoted_marker() -> None:
    """A reference that is the whole value keeps the byte-identical marker (resume compatibility)."""
    assert _normalize_runtime_references("${policy_base_url}") == "<runtime:policy_base_url>"


def test_judge_fallback_config_resolves_the_run_identity() -> None:
    """The judge-falls-back-to-policy config builds an identity instead of raising GrammarParseError."""
    servers = _judge_fallback_config()
    identity = _configuration_identity({}, servers, [])
    judge = identity["servers"]["judge_model"]["responses_api_models"]["openai_model"]
    # The model name is semantic (not a runtime option), so it survives and resolves to the fallback.
    assert judge["openai_model"] == "policy-model"
    # The policy credential never reaches the persisted identity.
    assert "sk-policy-canary" not in json.dumps(identity)


def test_judge_fallback_identity_ignores_the_runtime_address_but_tracks_the_model(tmp_path: Path) -> None:
    """A changed judge endpoint keeps the digest; a changed judge model changes it."""
    servers = _judge_fallback_config()
    source = tmp_path / "tasks.jsonl"
    source.write_text("{}\n")
    base = RunManifest.create(source, [], {}, servers).config_digest
    moved_endpoint = RunManifest.create(source, [], {}, servers | {"judge_base_url": "http://judge:9000/v1"})
    changed_model = RunManifest.create(source, [], {}, servers | {"judge_model_name": "other-model"})
    assert moved_endpoint.config_digest == base
    assert changed_model.config_digest != base


def test_nested_credential_reference_under_a_plain_key_redacts_without_breaking_grammar() -> None:
    """A credential reference nested in an oc.select default, below a non-credential key, is redacted safely."""
    servers = _judge_fallback_config()
    # A credential-named top-level value whose reference is not a server runtime option, so
    # `_normalize_runtime_references` leaves it and the credential substitution handles the nested ${...}.
    servers["my_service_pw"] = "sk-tracking-canary"  # pragma: allowlist secret
    model = servers["judge_model"]["responses_api_models"]["openai_model"]
    # A non-credential-named field (not wholesale-redacted) whose fallback default is that credential reference.
    model["usage_report_url"] = "${oc.select:usage_report_override,${my_service_pw}}"

    safe = _credential_safe(_normalize_runtime_references(model["usage_report_url"]))
    assert safe == "${oc.select:usage_report_override,'<redacted>'}"
    assert OmegaConf.to_container(OmegaConf.create({"v": safe}), resolve=True)["v"] == "<redacted>"

    identity = _configuration_identity({}, servers, [])
    judge = identity["servers"]["judge_model"]["responses_api_models"]["openai_model"]
    assert judge["usage_report_url"] == "<redacted>"
    assert "sk-tracking-canary" not in json.dumps(identity)


def test_agent_routed_nested_fallback_config_resolves(tmp_path: Path) -> None:
    """The same fallback shape resolves when the identity is reached through agent-routed rows."""
    servers = _judge_fallback_config()
    rows = [failing_row() | {"agent_ref": {"name": "judge_model"}}]
    source = tmp_path / "tasks.jsonl"
    source.write_text(json.dumps(rows[0]) + "\n")
    # Must not raise GrammarParseError while building the manifest digest.
    assert RunManifest.create(source, rows, {}, servers).config_digest
