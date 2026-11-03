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
import re
from typing import Any, List

from omegaconf import DictConfig, ListConfig, open_dict


MASKED_VALUE = "****"

# Matched case-insensitively anywhere in a key name, so `DB_PASSWORD`, `client_secret`, `apiKey` and an
# `Authorization` header entry all count. `authorization`, not `auth`, so keys like `author` stay visible.
_SECRET_KEY_SUBSTRINGS = (
    "token",
    "key",
    "header",
    "password",
    "passwd",
    "secret",
    "credential",
    "bearer",
    "cookie",
    "authorization",
)


# The password in a URL's user info (`user:PASSWORD@host` after the scheme). `${...}` stays whole inside the user name,
# so an interpolation is never cut in two.
# The password runs to the last `@` of the authority, because passwords are often pasted without percent-encoding.
_URL_PASSWORD = re.compile(
    r"(?P<head>[A-Za-z][A-Za-z0-9+.\-]*://(?:\$\{[^}]*\}|[^\s/?#:@$])*):(?P<password>[^\s/?#]*)@(?P<host>[^\s/?#@]*)"
)


def mask_url_passwords(text: str) -> str:
    """Mask the password in the user info of every URL inside `text`, keeping the user name and the rest of the URL.

    Only a precaution for a credential pasted into a URL: it is the user's job to keep secrets out of the config
    in the first place (behind `${oc.env:NAME}` or `env.yaml`). A password that is an interpolation is a reference
    rather than a secret, so it stays as it is.
    """

    def mask(match: re.Match) -> str:
        if not match["password"] or "${" in match["password"] or match["password"] == MASKED_VALUE:
            return match.group(0)
        return f"{match['head']}:{MASKED_VALUE}@{match['host']}"

    return _URL_PASSWORD.sub(mask, text)


def looks_like_secret_key(key: object) -> bool:
    # OmegaConf allows non-string keys (ints, bools, enums), so match on their string form.
    lowered_key = str(key).lower()
    return any(substring in lowered_key for substring in _SECRET_KEY_SUBSTRINGS)


def hide_secrets_in_overrides(tokens: List[str]) -> List[str]:
    """Mask `+key=value` / `++key=value` tokens whose key looks secret-shaped."""
    override_re = re.compile(r"^(\+{1,2})([^=]+)=(.*)$")
    redacted = []
    for token in tokens:
        match = override_re.match(token)
        if match:
            prefix, key, _value = match.groups()
            # Any secret-shaped segment, not only the leaf: `...harbor.secrets.JUDGE_AUTH` is a secret too.
            if any(looks_like_secret_key(segment) for segment in key.split(".")):
                token = f"{prefix}{key}={MASKED_VALUE}"
        redacted.append(token)
    return redacted


def recursively_hide_secrets(dict_config: DictConfig) -> None:
    """Mask every secret-shaped leaf in place with '****' so a config can be printed or exported.

    A leaf is secret-shaped by its own name, or by sitting anywhere below a secret-shaped key: every
    value of a `secrets:` mapping is a secret whatever its environment variable is called.

    Used by the config parser and the exporters.
    """

    def mask_all(node) -> None:
        for k, v in list(node.items()):
            if isinstance(v, (DictConfig, dict)):
                mask_all(v)
            elif isinstance(v, (ListConfig, list)):
                node[k] = [MASKED_VALUE] * len(v)
            else:
                node[k] = MASKED_VALUE

    def hide(node) -> None:
        for k, v in list(node.items()):
            if isinstance(v, (DictConfig, dict)):
                if looks_like_secret_key(k):
                    mask_all(v)
                else:
                    hide(v)
            elif isinstance(v, (ListConfig, list)):
                if looks_like_secret_key(k):
                    node[k] = [MASKED_VALUE] * len(v)
                else:
                    for index, inner_v in enumerate(v):
                        if isinstance(inner_v, (DictConfig, dict)):
                            hide(inner_v)
                        elif isinstance(inner_v, str) and mask_url_passwords(inner_v) != inner_v:
                            v[index] = mask_url_passwords(inner_v)
            else:
                if looks_like_secret_key(k):
                    node[k] = MASKED_VALUE
                elif isinstance(v, str) and mask_url_passwords(v) != v:
                    node[k] = mask_url_passwords(v)

    with open_dict(dict_config):
        hide(dict_config)


def hide_secrets_in_container(data: Any) -> Any:
    """Copy of a plain dict/list tree with every secret-shaped leaf masked.

    Unlike `recursively_hide_secrets` this never touches OmegaConf, so an unresolved `${...}`
    interpolation (e.g. `${oc.env:API_KEY}`) is neither evaluated nor required to be resolvable.
    """
    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # A header map's names are arbitrary (`X-Auth`, `Api-Gateway-Token`), so its values are masked
                # whatever the name looks like; the names stay for inspection.
                if "header" in str(key).lower():
                    masked[key] = {name: MASKED_VALUE for name in value}
                else:
                    masked[key] = hide_secrets_in_container(value)
            elif isinstance(value, list):
                masked[key] = (
                    [MASKED_VALUE] * len(value) if looks_like_secret_key(key) else hide_secrets_in_container(value)
                )
            else:
                masked[key] = MASKED_VALUE if looks_like_secret_key(key) else hide_secrets_in_container(value)
        return masked
    if isinstance(data, list):
        return [hide_secrets_in_container(item) for item in data]
    if isinstance(data, str):
        return mask_url_passwords(data)
    return data
