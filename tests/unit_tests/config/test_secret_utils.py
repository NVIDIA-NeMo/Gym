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
import pytest
from omegaconf import OmegaConf

from nemo_gym.config.secret_utils import (
    MASKED_VALUE,
    hide_secrets_in_container,
    hide_secrets_in_overrides,
    is_credential_name,
    mask_credentials,
    mask_url_passwords,
    recursively_hide_secrets,
    url_without_credentials,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "https://user:s3cr3t-token@api.example.com/v1?key=abc123",  # pragma: allowlist secret
            "https://user:****@api.example.com/v1?key=abc123",  # pragma: allowlist secret
        ),
        ("postgres://admin:pw@db:5432/x", "postgres://admin:****@db:5432/x"),  # pragma: allowlist secret
        ("redis://:pw@cache:6379/0", "redis://:****@cache:6379/0"),
        (
            "mongodb://u:pw@h1:27017,h2:27017/db?replicaSet=r",  # pragma: allowlist secret
            "mongodb://u:****@h1:27017,h2:27017/db?replicaSet=r",
        ),
        ("postgres://u:p@ss@host/db", "postgres://u:****@host/db"),  # pragma: allowlist secret
        ("https://u:a:b@h/x", "https://u:****@h/x"),  # pragma: allowlist secret
        (
            "curl https://a:b@h1/x and redis://:pw@r:6379/0",  # pragma: allowlist secret
            "curl https://a:****@h1/x and redis://:****@r:6379/0",
        ),
    ],
)
def test_the_password_in_a_url_is_masked_and_the_rest_of_the_url_stays(text: str, expected: str) -> None:
    assert mask_url_passwords(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "ssh://git@github.com/org/repo.git",
        "https://token-as-user@host/x",
        "https://api.example.com/v1?page=2&model=m",
        "https://u:@h",
        "someone@example.com",
        "key:value@x",
        "C:\\\\path\\\\file",
        "https://u:${oc.env:DB_PASSWORD}@h",
        "https://${oc.env:DB_USER}:${oc.env:DB_PASSWORD}@h",
        "https://u:****@h",  # pragma: allowlist secret
    ],
)
def test_text_without_a_url_password_is_left_alone(text: str) -> None:
    assert mask_url_passwords(text) == text


def test_an_interpolated_user_name_is_not_cut_in_two() -> None:
    assert mask_url_passwords("https://${oc.env:DB_USER}:literal@h") == "https://${oc.env:DB_USER}:****@h"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "https://user:secret@host.example/v1?token=secret#frag",  # pragma: allowlist secret
            "https://host.example/v1",
        ),
        ("//user:secret@host.example/v1?token=secret", "//host.example/v1"),  # pragma: allowlist secret
        ("user:p@ss@host/model", "host/model"),  # pragma: allowlist secret
        ("user:secret@host/model?token=secret", "host/model"),  # pragma: allowlist secret
        ("ghp_TOKEN@github.com/org/repo", "github.com/org/repo"),
        ("ghp_TOKEN@localhost:8080/model", "localhost:8080/model"),
        ("nvidia-eval-factory-benchmarking/configs/benchmarks/gpqa@v3", None),
        ("model@v3", None),
        ("meta/llama-3.1-70b-instruct", None),
        ("/checkpoints/super-3/step-900", None),
    ],
)
def test_a_recorded_name_keeps_no_url_credentials(value: str, expected: str | None) -> None:
    assert url_without_credentials(value) == (value if expected is None else expected)


@pytest.mark.parametrize("name", ["api_key", "X-Api-Key", "auth", "Authorization", "secrets", "HF_TOKEN", "cookie"])
def test_a_credential_is_named_as_one(name: str) -> None:
    assert is_credential_name(name)


@pytest.mark.parametrize("name", ["max_tokens", "author", "authors", "reference_key", "vllm_serve_env_vars", "env"])
def test_a_setting_is_not_named_as_a_credential(name: str) -> None:
    assert not is_credential_name(name)


def test_masking_twice_changes_nothing() -> None:
    once = mask_url_passwords("postgres://admin:pw@db/x")  # pragma: allowlist secret

    assert mask_url_passwords(once) == once


def test_the_saved_config_masks_a_url_password_under_an_ordinary_key_and_in_lists() -> None:
    config = {
        "base_url": "https://user:s3cr3t-token@api.example.com/v1?key=abc123",  # pragma: allowlist secret
        "servers": ["http://a:pw@one", "http://two"],  # pragma: allowlist secret
        "nested": {"dsn": "postgres://u:pw@db/x", "name": "m"},  # pragma: allowlist secret
        "policy_api_key": "sk-1",
    }

    assert hide_secrets_in_container(config) == {
        "base_url": "https://user:****@api.example.com/v1?key=abc123",  # pragma: allowlist secret
        "servers": ["http://a:****@one", "http://two"],  # pragma: allowlist secret
        "nested": {"dsn": "postgres://u:****@db/x", "name": "m"},  # pragma: allowlist secret
        "policy_api_key": MASKED_VALUE,
    }
    assert config["base_url"].endswith("?key=abc123") and "s3cr3t-token" in config["base_url"]


def test_the_saved_config_keeps_an_interpolated_password_as_a_reference() -> None:
    config = {"base_url": "https://user:${oc.env:API_PASSWORD}@api.example.com"}

    assert hide_secrets_in_container(config) == config


def test_the_exporters_config_masks_a_url_password_too() -> None:
    config = OmegaConf.create(
        {
            "mlflow_tracking_uri": "https://u:pw@mlflow.example",  # pragma: allowlist secret
            "tags": ["http://a:pw@h"],  # pragma: allowlist secret
            "model": "m",
        }
    )

    recursively_hide_secrets(config)

    assert OmegaConf.to_container(config) == {
        "mlflow_tracking_uri": "https://u:****@mlflow.example",  # pragma: allowlist secret
        "tags": ["http://a:****@h"],  # pragma: allowlist secret
        "model": "m",
    }


@pytest.mark.parametrize(
    "value", [False, True, 17, None, "diagnostic-canary", {"arbitrary": "diagnostic-canary"}, ["diagnostic-canary"]]
)
def test_config_projection_masks_arbitrary_secret_values(value):
    source = {"secrets": {"JUDGE_AUTH": value}, "headers": {"arbitrary": value}}
    masked = hide_secrets_in_container(source)
    assert "diagnostic-canary" not in str(masked)
    assert masked["headers"]["arbitrary"] == MASKED_VALUE
    expected = (
        {"arbitrary": MASKED_VALUE}
        if isinstance(value, dict)
        else [MASKED_VALUE]
        if isinstance(value, list)
        else MASKED_VALUE
    )
    assert masked["secrets"]["JUDGE_AUTH"] == expected
    assert source["secrets"]["JUDGE_AUTH"] == value


def test_capture_namespace_cannot_escape_a_secret_ancestor():
    source = {
        "token_id_capture": {"enabled": False, "sink": "package:Sink", "headers": {"X-Auth": "diagnostic-canary"}},
        "secrets": {"token_id_capture": {"enabled": True, "sink": "diagnostic-canary"}},
        "headers": {"token_id_capture": {"enabled": True}},
        "my_token_id_capture": {"enabled": True},
        "nested": {"token_id_capture": {"enabled": True}},
    }
    masked = hide_secrets_in_container(source)
    assert masked["token_id_capture"] == {
        "enabled": False,
        "sink": "package:Sink",
        "headers": {"X-Auth": MASKED_VALUE},
    }
    assert masked["secrets"]["token_id_capture"] == {"enabled": MASKED_VALUE, "sink": MASKED_VALUE}
    assert masked["headers"]["token_id_capture"] == MASKED_VALUE
    assert masked["my_token_id_capture"]["enabled"] == MASKED_VALUE
    assert masked["nested"]["token_id_capture"]["enabled"] == MASKED_VALUE


def test_public_capture_projection_fields_match_the_supported_schema():
    from nemo_gym.config.secret_utils import _CAPTURE_PUBLIC_FIELDS
    from nemo_gym.models.token_capture.config import TokenCaptureSettings as TokenIdCaptureSettings

    assert _CAPTURE_PUBLIC_FIELDS | {"sink_kwargs", "lineage_store_kwargs", "control_auth_token_env"} == set(
        TokenIdCaptureSettings.model_fields
    )
    source = {
        "token_id_capture": {
            "enabled": True,
            "allow_system_suffix": True,
            "auth": "diagnostic-canary",
            "access": {"opaque": "diagnostic-canary"},
            "arbitrary": ["diagnostic-canary"],
        }
    }
    assert hide_secrets_in_container(source) == {
        "token_id_capture": {
            "enabled": True,
            "allow_system_suffix": True,
            "auth": MASKED_VALUE,
            "access": {"opaque": MASKED_VALUE},
            "arbitrary": [MASKED_VALUE],
        }
    }


_TOKEN_SETTINGS = {
    "policy": {
        "max_tokens": None,
        "max_output_tokens": 4096,
        "num_speculative_tokens": 3,
        "max_num_batched_tokens": 8192,
        "token_budget_fraction": 0.5,
        "return_token_ids": True,
    },
    "hf_token": "hf-FAKE-CANARY",  # pragma: allowlist secret
    "api_token": "${oc.env:API_TOKEN}",
    "limits": {"cap": 7},
    "capped_token": "${limits.cap}",
    "string_max_tokens": "4096",
    "token_key": 3,
    "secrets": {"max_tokens": 4096, "enabled": True},
    "auth_tokens": {"max_tokens": 1},
    "stop_tokens": [1, 2],
}

_TOKEN_SETTINGS_MASKED = {
    "policy": {
        "max_tokens": None,
        "max_output_tokens": 4096,
        "num_speculative_tokens": 3,
        "max_num_batched_tokens": 8192,
        "token_budget_fraction": 0.5,
        "return_token_ids": True,
    },
    "hf_token": MASKED_VALUE,
    "api_token": MASKED_VALUE,
    "limits": {"cap": 7},
    "capped_token": MASKED_VALUE,
    "string_max_tokens": MASKED_VALUE,
    "token_key": MASKED_VALUE,
    "secrets": {"max_tokens": MASKED_VALUE, "enabled": MASKED_VALUE},
    "auth_tokens": {"max_tokens": MASKED_VALUE},
    "stop_tokens": [MASKED_VALUE, MASKED_VALUE],
}


def test_the_saved_config_shows_token_counts_and_flags_and_masks_token_credentials() -> None:
    assert hide_secrets_in_container(_TOKEN_SETTINGS) == _TOKEN_SETTINGS_MASKED


def test_the_exporters_config_shows_token_counts_and_flags_and_masks_token_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_TOKEN", "tok-FAKE-CANARY")  # pragma: allowlist secret
    config = OmegaConf.create(_TOKEN_SETTINGS)

    recursively_hide_secrets(config)

    # `capped_token` resolves to an int, but it is an interpolation in the source config, so it stays masked.
    assert OmegaConf.to_container(config) == _TOKEN_SETTINGS_MASKED


def test_overrides_show_token_counts_and_flags_and_mask_token_credentials() -> None:
    overrides = [
        "++policy.max_tokens=4096",
        "++policy.max_output_tokens=null",
        "+num_speculative_tokens=-1",
        "++policy.token_budget_fraction=0.5",
        "++policy.return_token_ids=True",
        "++hf_token=hf-FAKE-CANARY",  # pragma: allowlist secret
        "++api_token=${oc.env:API_TOKEN}",
        "++hf_token='4096'",
        "++token_key=3",
        "++harbor.secrets.max_tokens=4096",
        "++auth_tokens.max_tokens=1",
        "++stop_tokens=[1,2]",
    ]

    assert hide_secrets_in_overrides(overrides) == [
        "++policy.max_tokens=4096",
        "++policy.max_output_tokens=null",
        "+num_speculative_tokens=-1",
        "++policy.token_budget_fraction=0.5",
        "++policy.return_token_ids=True",
        "++hf_token=****",
        "++api_token=****",
        "++hf_token=****",
        "++token_key=****",
        "++harbor.secrets.max_tokens=****",
        "++auth_tokens.max_tokens=****",
        "++stop_tokens=****",
    ]


def test_mask_credentials_masks_only_credential_values() -> None:
    config = {
        "policy_api_key": "sk-live",
        "hf_token": "hf-live",
        "token": "bare-token",
        "policy_model": {
            "responses_api_models": {
                "vllm_model": {
                    "api_key": "${policy_api_key}",
                    "extra_headers": {"Authorization": "Bearer live", "X-Trace": "abc"},
                    "max_tokens": 131072,
                    "assistant_header": "<assistant>",
                    "return_token_id_information": True,
                }
            }
        },
        "env_key": "${oc.env:EXAMPLE_MODEL_API_KEY}",
        "secret_api_key": "${oc.env:EXAMPLE_MODEL_API_KEY}",
        "credentials": {"user": "me", "password": "pw", "nested": ["a", {"x": "b"}]},
        "unset_api_key": "",
        "missing_api_key": "???",
        "cache_key": "k1",
        "token_ids": [1, 2],
        "max_tokens": 4096,
        "client_secrets": ["s1", "s2"],
        "openai_default_headers": {"x-litellm-key": "sk-litellm", "x-trace-id": "t1"},
        "mlflow_tracking_uri": "https://user:pw@mlflow.example/api",  # pragma: allowlist secret
        "docs_url": "https://example.com/path",
    }

    assert mask_credentials(config) == {
        "policy_api_key": "****",
        "hf_token": "****",
        "token": "****",
        "policy_model": {
            "responses_api_models": {
                "vllm_model": {
                    "api_key": "****",
                    "extra_headers": {"Authorization": "****", "X-Trace": "abc"},
                    "max_tokens": 131072,
                    "assistant_header": "<assistant>",
                    "return_token_id_information": True,
                }
            }
        },
        "env_key": "${oc.env:EXAMPLE_MODEL_API_KEY}",
        "secret_api_key": "****",
        "credentials": {"user": "****", "password": "****", "nested": ["****", {"x": "****"}]},
        "unset_api_key": "",
        "missing_api_key": "???",
        "cache_key": "k1",
        "token_ids": [1, 2],
        "max_tokens": 4096,
        "client_secrets": ["****", "****"],
        # A `*-key` header is a credential (LiteLLM's own key header); other headers stay.
        "openai_default_headers": {"x-litellm-key": "****", "x-trace-id": "t1"},
        # A password inside a URL is masked wherever the URL sits; the rest of the URL stays.
        "mlflow_tracking_uri": "https://user:****@mlflow.example/api",
        "docs_url": "https://example.com/path",
    }
    # A copy: the input is untouched.
    assert config["policy_api_key"] == "sk-live"


@pytest.mark.parametrize("header_first", [False, True])
def test_mask_credentials_follows_shared_header_and_scalar_sources(header_first: bool) -> None:
    config = {
        "shared": {"x-litellm-key": "sentinel-header", "x-trace-id": "trace-kept"},
        "alias": "${shared}",
        "second_alias": "${alias}",
        "openai_default_headers": "${second_alias}",
        "values": ["not-secret", "sentinel-scalar"],
        "policy_api_key": "${values[1]}",
        "api_key_via_container": "${alias[x-litellm-key]}",
        "cache_key": "cache-kept",
        "max_tokens": 4096,
        "unrelated": "${oc.env:MASK_TEST_UNSET}",
    }
    if header_first:
        config = dict(reversed(list(config.items())))
    masked = mask_credentials(config)
    assert "sentinel" not in repr(masked)
    assert masked["shared"] == {"x-litellm-key": "****", "x-trace-id": "trace-kept"}
    assert masked["values"] == ["not-secret", "****"]
    assert masked["openai_default_headers"] == "${second_alias}"
    assert OmegaConf.create(masked).openai_default_headers["x-trace-id"] == "trace-kept"
    assert OmegaConf.create(masked).openai_default_headers["x-litellm-key"] == "****"
    assert masked["cache_key"] == "cache-kept" and masked["max_tokens"] == 4096
    assert masked["unrelated"] == "${oc.env:MASK_TEST_UNSET}"
    assert config["shared"]["x-litellm-key"] == "sentinel-header"
    assert config["values"][1] == "sentinel-scalar"


def test_mask_credentials_follows_relative_and_embedded_references() -> None:
    config = {
        "sources": ["sentinel-root"],
        "nested": {
            "values": [{"value": "sentinel-nested", "ordinary": "kept"}],
            "api_key": "${.values[0].value}",
            "headers": {"Authorization": "Bearer ${...sources[0]}"},
        },
    }
    masked = mask_credentials(config)
    assert "sentinel" not in repr(masked)
    assert masked["sources"] == ["****"]
    assert masked["nested"]["values"] == [{"value": "****", "ordinary": "kept"}]
    assert masked["nested"]["api_key"] == "****"
    assert masked["nested"]["headers"]["Authorization"] == "****"
    assert config["sources"] == ["sentinel-root"]


def test_mask_credentials_masks_url_password_and_its_source() -> None:
    config = {
        "shared": "sentinel-password",
        "mlflow_tracking_uri": "https://user:prefix${shared}suffix@host/path",
        "unrelated": "${oc.env:MASK_TEST_UNSET}",
    }
    masked = mask_credentials(config)
    assert masked["mlflow_tracking_uri"] == "https://user:****@host/path"
    assert masked["shared"] == "****"
    assert "sentinel" not in repr(masked)
    assert config["shared"] == "sentinel-password"
    # Other uses of the existing URL helper keep interpolations unchanged.
    assert mask_url_passwords(config["mlflow_tracking_uri"]) == config["mlflow_tracking_uri"]


def test_mask_credentials_never_evaluates_environment_or_custom_resolvers(monkeypatch) -> None:
    monkeypatch.delenv("MASK_TEST_UNSET", raising=False)
    called = []
    OmegaConf.register_new_resolver("mask_test_resolver", lambda: called.append(True), replace=True)
    try:
        config = {
            "env_name": "MASK_TEST_UNSET",
            "default_value": "sentinel-default",
            "api_key": "${oc.env:${env_name},${default_value}}",
            "unrelated": "${mask_test_resolver:}",
            "unrelated_env": "${oc.env:MASK_TEST_UNSET}",
        }
        masked = mask_credentials(config)
        assert masked["api_key"] == masked["default_value"] == "****"
        assert masked["env_name"] == "MASK_TEST_UNSET"
        assert masked["unrelated"] == "${mask_test_resolver:}"
        assert masked["unrelated_env"] == "${oc.env:MASK_TEST_UNSET}"
        with pytest.raises(ValueError, match="Cannot safely mask"):
            mask_credentials({"api_key": "${mask_test_resolver:}"})
        assert not called
    finally:
        OmegaConf.clear_resolver("mask_test_resolver")


@pytest.mark.parametrize(
    "config",
    [
        {"which": "value", "values": {"value": "sentinel"}, "api_key": "${values.${which}}"},
        {"api_key": "${custom:sentinel}"},
        {"extra_headers": "${oc.env:MASK_TEST_UNSET}"},
        {"first": "${second}", "second": "${first}", "api_key": "${first.value}"},
    ],
)
def test_mask_credentials_refuses_unanalyzable_sensitive_references(config) -> None:
    with pytest.raises(ValueError, match="Cannot safely mask") as error:
        mask_credentials(config)
    assert "sentinel" not in str(error.value)


def test_mask_credentials_handles_cycles_and_escaped_interpolation_literals() -> None:
    config = {
        "first": "${second}",
        "second": "${first}",
        "api_key": "${first}",
        "ordinary": r"\${not_a_reference}",
        "extra_headers": {"x-trace-id": r"\${not_a_reference}"},
    }
    masked = mask_credentials(config)
    assert masked["first"] == masked["second"] == masked["api_key"] == "****"
    assert masked["ordinary"] == config["ordinary"]
    assert masked["extra_headers"] == config["extra_headers"]


@pytest.mark.parametrize("present", [False, True])
def test_mask_credentials_supports_static_select_and_unresolved_fallbacks(present: bool, monkeypatch) -> None:
    monkeypatch.delenv("MASK_TEST_UNSET", raising=False)
    config = {
        "default_value": "sentinel-default",
        "nested": {
            "value": "sentinel-relative",
            "api_key": "${oc.select:.value,unused}",
            "openai_api_key": "${oc.select:shared,${oc.env:MASK_TEST_UNSET,${default_value}}}",
            "password": "${oc.select:missing.path,null}",
        },
        "unrelated": "${oc.env:MASK_TEST_UNSET}",
    }
    if present:
        config["shared"] = "sentinel-selected"
    masked = mask_credentials(config)
    assert "sentinel" not in repr(masked)
    assert masked["default_value"] == "****"
    assert masked["nested"] == {"value": "****", "api_key": "****", "openai_api_key": "****", "password": "****"}
    assert masked["unrelated"] == config["unrelated"]
    assert config["default_value"] == "sentinel-default"


@pytest.mark.parametrize("present", [False, True])
def test_mask_credentials_supports_repository_decode_select_wrapper(present: bool) -> None:
    # The critpt recipe decodes the selected credential or null. Its dependency can be found without decoding it.
    config = {"api_key": "${oc.decode:${oc.select:shared,null}}"}
    if present:
        config["shared"] = "sentinel-selected"
    masked = mask_credentials(config)
    assert masked["api_key"] == "****"
    assert "sentinel" not in repr(masked)
    if present:
        assert masked["shared"] == "****"
        assert config["shared"] == "sentinel-selected"


@pytest.mark.parametrize(
    "config",
    [
        {"which": "value", "api_key": "${oc.select:${which},unused}"},
        {"extra_headers": "${oc.select:shared,{x-litellm-key:sentinel}}"},
        {"api_key": "${oc.decode:${oc.env:MASK_TEST_UNSET}}"},
        {"shared": r"\${unclassified}", "unclassified": "sentinel", "api_key": "${oc.decode:${shared}}"},
        {"shared": "{unclassified}", "unclassified": "sentinel", "api_key": "${oc.decode:$${shared}}"},
        {"unclassified": "sentinel", "api_key": r"${oc.decode:${oc.select:missing,'\${unclassified}'}}"},
        {
            "unclassified": "sentinel",
            "api_key": r"${oc.decode:${oc.select:missing,${oc.select:also_missing,'\${unclassified}'}}}",
        },
        {
            "piece": "{unclassified}",
            "unclassified": "sentinel",
            "api_key": "${oc.decode:${oc.select:missing,$${piece}}}",
        },
    ],
)
def test_mask_credentials_refuses_dynamic_select_or_decode(config) -> None:
    with pytest.raises(ValueError, match="Cannot safely mask") as error:
        mask_credentials(config)
    assert "sentinel" not in str(error.value)
