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
    mask_url_passwords,
    recursively_hide_secrets,
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
