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
from typing import Any, List, NoReturn

from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from omegaconf._utils import split_key
from omegaconf.errors import GrammarParseError
from omegaconf.grammar.gen.OmegaConfGrammarParser import OmegaConfGrammarParser
from omegaconf.grammar_parser import parse
from omegaconf.grammar_visitor import GrammarVisitor


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


# Noncredential controls from TokenIdCaptureSettings. Unknown/custom children stay private; adding a
# capture setting must not accidentally make arbitrary backend credentials visible in run artifacts.
_CAPTURE_PUBLIC_FIELDS = frozenset(
    {
        "enabled",
        "all_agents",
        "dir",
        "sink",
        "lineage_store",
        "rebuild_response",
        "parent_resolution",
        "allow_system_suffix",
        "allow_unresolved_continuations",
        "delta_records",
        "mask_incomplete_when_attributed",
        "max_mask_fraction",
        "mask_fraction_min_samples",
        "external_staging",
        "external_staging_backend",
    }
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


def _secret_shaped_only_by_token(key: object) -> bool:
    lowered_key = str(key).lower()
    return {substring for substring in _SECRET_KEY_SUBSTRINGS if substring in lowered_key} == {"token"}


def _is_token_count_or_flag(key: object, value: Any) -> bool:
    """Whether a leaf is secret-shaped only through "token" and holds a number, bool or null.

    `max_tokens`, `num_speculative_tokens` and the like are counts or flags, not credentials, and the
    saved config must show them. A credential is a string, so any string (an interpolation included)
    stays masked, as does a key that also matches another secret substring (`api_token_key`).
    """
    return _secret_shaped_only_by_token(key) and (value is None or isinstance(value, (bool, int, float)))


# An override value that Hydra parses as an int, float, bool or null.
_OVERRIDE_NON_STRING_SCALAR = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|true|false|null", re.IGNORECASE
)


# Narrower than _SECRET_KEY_SUBSTRINGS: key names whose values are credentials, for a config that clients still use
# (the head server's copy feeds routing, assistant headers and token limits, so `header`, `key` and `tokens` stay
# visible there). A bare `token` or a `*_token` name counts; `max_tokens` and `token_ids` do not.
_CREDENTIAL_KEY_SUBSTRINGS = (
    "api_key",
    "apikey",
    "api-key",
    "access_key",
    "private_key",
    "secret",
    "password",
    "passwd",
    "credential",
    "bearer",
    "authorization",
    "cookie",
)


def looks_like_credential_key(key: object) -> bool:
    lowered_key = str(key).lower()
    return (
        any(substring in lowered_key for substring in _CREDENTIAL_KEY_SUBSTRINGS)
        or lowered_key == "token"
        or lowered_key.endswith(("_token", "-token"))
    )


def _looks_like_header_credential(key: object) -> bool:
    """Inside a headers mapping, a `*-key` / `*_key` header (e.g. LiteLLM's `x-litellm-key`) carries a credential."""
    lowered_key = str(key).lower()
    return lowered_key == "key" or lowered_key.endswith(("-key", "_key"))


def mask_credentials(value: Any, under_credential_key: bool = False, in_headers: bool = False) -> Any:
    """Copy a config without credential values, including their referenced source nodes.

    Static OmegaConf references carry credential/header context to their targets without resolving any values.
    Unrelated interpolations stay intact. Dynamic references and custom resolvers in credential-bearing paths
    fail closed. Static ``oc.select`` and ``oc.env`` credential leaves are analyzed without evaluation;
    ``oc.decode`` supports static inputs, refusing expressions whose decoded references depend on runtime values.
    URL passwords are masked here even when interpolated; ``mask_url_passwords`` retains its existing behavior
    for other callers.
    Empty strings, MISSING, non-string leaves, ordinary headers and routing settings remain usable.
    """
    path_type = tuple[object, ...]
    pending: list[tuple[path_type, bool, bool, bool]] = [((), under_credential_key, in_headers, False)]
    visited: set[tuple[path_type, bool, bool, bool]] = set()
    masked: set[path_type] = set()
    urls: dict[path_type, str] = {}

    def unsafe_reference() -> NoReturn:
        # Never include an interpolation's text: resolver arguments can themselves contain credentials.
        raise ValueError(
            "Cannot safely mask a credential interpolation. Use static config references, oc.select or oc.env for "
            "credential values, and static config references for header mappings."
        )

    def references(text: str, *, allow_env: bool, decoded: bool = False) -> tuple[list[tuple[str, bool, bool]], bool]:
        if "${" not in text:
            return [], False
        try:
            tree = parse(text)
        except GrammarParseError:
            unsafe_reference()
        result: list[tuple[str, bool, bool]] = []
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if decoded and hasattr(node, "symbol") and node.symbol.type == OmegaConfGrammarParser.ESC_INTER:
                unsafe_reference()
            if isinstance(node, OmegaConfGrammarParser.InterpolationNodeContext):
                # A config key containing an interpolation is dynamic, e.g. ${keys.${which}}.
                if any(
                    isinstance(child, OmegaConfGrammarParser.ConfigKeyContext)
                    and isinstance(child.getChild(0), OmegaConfGrammarParser.InterpolationContext)
                    for child in node.getChildren()
                ):
                    unsafe_reference()
                result.append((node.getText()[2:-1], False, decoded))
                continue
            if isinstance(node, OmegaConfGrammarParser.InterpolationResolverContext):
                name = node.getChild(1).getText()
                if not allow_env or name not in ("oc.env", "oc.select", "oc.decode"):
                    unsafe_reference()
                sequence = node.getChild(3)
                if not isinstance(sequence, OmegaConfGrammarParser.SequenceContext):
                    unsafe_reference()
                arguments = list(sequence.getChildren())
                if not arguments or not isinstance(arguments[0], OmegaConfGrammarParser.ElementContext):
                    unsafe_reference()
                if name == "oc.select":
                    # Parse the static selector with OmegaConf's grammar, refusing every interpolation callback.
                    # This matches the builtin's root-relative lookup (unless the selector begins with a dot).
                    selector = GrammarVisitor(
                        node_interpolation_callback=lambda *_: unsafe_reference(),
                        resolver_interpolation_callback=lambda **_: unsafe_reference(),
                        memo=None,
                    ).visit(arguments[0])
                    if not isinstance(selector, str):
                        unsafe_reference()
                    result.append((selector, True, decoded))
                elif name == "oc.decode":
                    # decode reparses its result. Only static inputs can be inspected without running a resolver;
                    # propagate this requirement through references/selectors, and refuse a second decode layer.
                    if decoded or len(arguments) != 1:
                        unsafe_reference()
                    refs, _ = references(arguments[0].getText(), allow_env=True, decoded=True)
                    result.extend(refs)
                    continue
                elif decoded:
                    # An environment value could decode into an arbitrary config reference.
                    unsafe_reference()
                # An env variable's name is not a credential; an optional env/select default is. Inspect defaults
                # even when a selector exists, since their literal sources must not survive elsewhere in the copy.
                if decoded:
                    for argument in arguments[2::2]:
                        refs, _ = references(argument.getText(), allow_env=True, decoded=True)
                        result.extend(refs)
                else:
                    nodes.extend(arguments[2:])
                continue
            if hasattr(node, "getChildren"):
                nodes.extend(node.getChildren())
        text_node = tree.getChild(0)
        sole_reference = (
            text_node.getChildCount() == 1
            and isinstance(text_node.getChild(0), OmegaConfGrammarParser.InterpolationContext)
            and isinstance(text_node.getChild(0).getChild(0), OmegaConfGrammarParser.InterpolationNodeContext)
        )
        if decoded:
            sole_expression = text_node.getChildCount() == 1 and isinstance(
                text_node.getChild(0), OmegaConfGrammarParser.InterpolationContext
            )
            if result and not sole_expression:
                unsafe_reference()
            if not result:
                literal = GrammarVisitor(
                    node_interpolation_callback=lambda *_: unsafe_reference(),
                    resolver_interpolation_callback=lambda **_: unsafe_reference(),
                    memo=None,
                ).visit(tree)
                if isinstance(literal, str) and "${" in literal:
                    unsafe_reference()
        return result, sole_reference

    def get(path: path_type) -> Any:
        node = value
        for key in path:
            node = node[key]
        return node

    def target(
        reference: str,
        origin: path_type,
        active: frozenset[path_type] = frozenset(),
        *,
        optional: bool = False,
    ) -> path_type | None:
        parts = split_key(reference)
        relative = 0
        while parts and parts[0] == "":
            relative += 1
            parts.pop(0)
        path = origin[:-1] if relative else ()
        if relative > len(origin):
            unsafe_reference()
        if relative > 1:
            path = path[: -(relative - 1)]
        for part in parts:
            # A path may traverse a container alias before reaching its leaf: ${alias.key}, alias=${shared}.
            seen = set(active)
            node = get(path)
            while isinstance(node, str):
                if path in seen:
                    unsafe_reference()
                seen.add(path)
                refs, sole_reference = references(node, allow_env=False)
                if not sole_reference or len(refs) != 1:
                    if optional:
                        return None
                    unsafe_reference()
                destination = target(refs[0][0], path, frozenset(seen))
                assert destination is not None
                path = destination
                node = get(path)
            if isinstance(node, dict) and part in node:
                path += (part,)
            elif isinstance(node, list) and part.isdecimal() and int(part) < len(node):
                path += (int(part),)
            else:
                if optional:
                    return None
                unsafe_reference()
        return path

    def follow(refs: list[tuple[str, bool, bool]], origin: path_type, credential: bool, headers: bool) -> None:
        for reference, optional, decoded in refs:
            destination = target(reference, origin, optional=optional)
            if destination is not None:
                pending.append((destination, credential, headers, decoded))

    while pending:
        path, credential, headers, decoded = pending.pop()
        context = path, credential, headers, decoded
        if context in visited:
            continue
        visited.add(context)
        node = get(path)
        if decoded and isinstance(node, (dict, list)):
            unsafe_reference()
        if isinstance(node, dict):
            pending.extend(
                (
                    path + (key,),
                    credential or looks_like_credential_key(key) or (headers and _looks_like_header_credential(key)),
                    str(key).lower().endswith("headers"),
                    False,
                )
                for key in node
            )
        elif isinstance(node, list):
            pending.extend((path + (index,), credential, headers, False) for index in range(len(node)))
        elif isinstance(node, str):
            if credential and node not in ("", "???"):
                masked.add(path)
            if credential or headers:
                refs, _ = references(node, allow_env=credential, decoded=decoded)
                follow(refs, path, credential, headers)
            if not credential:

                def mask_password(match: re.Match) -> str:
                    password = match["password"]
                    if not password or password == MASKED_VALUE:
                        return match.group(0)
                    refs, _ = references(password, allow_env=True)
                    follow(refs, path, True, False)
                    return f"{match['head']}:{MASKED_VALUE}@{match['host']}"

                urls[path] = _URL_PASSWORD.sub(mask_password, node)

    def copy(node: Any, path: path_type = ()) -> Any:
        if path in masked:
            return MASKED_VALUE
        if isinstance(node, dict):
            return {key: copy(child, path + (key,)) for key, child in node.items()}
        if isinstance(node, list):
            return [copy(child, path + (index,)) for index, child in enumerate(node)]
        return urls.get(path, node)

    return copy(value)


# A setting holding a credential, matched on its name lower-cased with `-`, `.`, spaces and case changes as `_`:
# `exa_api_key`, `X-Api-Key`, `service_key`, `Authorization`, `auth`, `X-Session`, `X_TOKEN`, `my_service_pw`, but not
# `max_tokens`, `author`, `reference_key` or `record_terminal_session`. A `..._key` counts only after a word that names
# an account or a service, since other `..._key` settings (`metric_key`, `reference_key`) select what is measured.
_CREDENTIAL_NAME = re.compile(
    r"api_?key|secret|passw(or)?d|credential|private_?key|access_?key|cookie"
    r"|(^|_)(token|pw|pwd|pass|bearer|o?auth)$|(^|_)o?auth(orization|entication)?(_|$)"
    r"|(^|_)(service|client|signing|license|master|account|subscription|encryption|session|app)_key$"
    r"|^(x_)?session(_id)?$"
)


def is_credential_name(name: object) -> bool:
    """Whether a setting's name says it holds a credential; narrower than `looks_like_secret_key`, so it can decide
    what a persisted identity redacts without hiding settings such as `max_tokens`."""
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    return bool(_CREDENTIAL_NAME.search(re.sub(r"[\s.\-]", "_", normalized).lower()))


_URL_HEAD = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.\-]*:)?//")
_URL_PARTS = re.compile(r"^(?P<authority>[^/?#]*)(?P<path>[^?#]*)")


def url_without_credentials(value: str) -> str:
    """`value` without the user info, query and fragment of the URL it is, so it can be recorded as a name.

    A URL is one with a scheme, a protocol-relative `//host/...`, or a schemeless `user:password@host/...` (also
    `TOKEN@host.example/...`). The user info runs to the authority's last `@`, since passwords are often pasted without
    percent-encoding. Anything else, such as `org/benchmark@v3` or a model name, is returned as is.
    """
    head = _URL_HEAD.match(value)
    rest = value[head.end() :] if head else value
    parts = _URL_PARTS.match(rest)
    userinfo, separator, host = parts["authority"].rpartition("@")
    if not head:
        if not separator:
            return value
        hostname, _, port = host.partition(":")
        if not ("." in hostname or hostname == "localhost" or port or ":" in userinfo):
            return value
    return (head[0] if head else "") + host + parts["path"]


def hide_secrets_in_overrides(tokens: List[str]) -> List[str]:
    """Mask `+key=value` / `++key=value` tokens whose key looks secret-shaped."""
    override_re = re.compile(r"^(\+{1,2})([^=]+)=(.*)$")
    redacted = []
    for token in tokens:
        match = override_re.match(token)
        if match:
            prefix, key, value = match.groups()
            *parents, leaf = key.split(".")
            # Any secret-shaped segment, not only the leaf: `...harbor.secrets.JUDGE_AUTH` is a secret too.
            secret_parent = any(looks_like_secret_key(segment) for segment in parents)
            # Same rule as for a config leaf: a token count or flag stays visible.
            visible_leaf = _secret_shaped_only_by_token(leaf) and _OVERRIDE_NON_STRING_SCALAR.fullmatch(value)
            if secret_parent or (looks_like_secret_key(leaf) and not visible_leaf):
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
                # An interpolation is a string in the source config, whatever it resolves to.
                interpolation = isinstance(node, DictConfig) and OmegaConf.is_interpolation(node, k)
                if looks_like_secret_key(k) and (interpolation or not _is_token_count_or_flag(k, v)):
                    node[k] = MASKED_VALUE
                elif isinstance(v, str) and mask_url_passwords(v) != v:
                    node[k] = mask_url_passwords(v)

    with open_dict(dict_config):
        hide(dict_config)


def hide_secrets_in_container(data: Any) -> Any:
    """Return the credential-safe config projection shared by saved artifacts and exporters.

    Preserve unresolved interpolations without evaluating them. Secret ancestry masks every value,
    including arbitrary names in a ``secrets`` mapping. The root ``token_id_capture`` mapping is a
    settings namespace, not a credential; its own secret-shaped children still undergo masking.
    Custom sink/resolver kwargs can contain credentials under arbitrary names, so their values stay masked.
    """

    def hide(node: Any, secret: bool = False, *, root: bool = False, capture: bool = False) -> Any:
        if isinstance(node, dict):
            masked = {}
            for key, value in node.items():
                if not secret and "header" in str(key).lower() and isinstance(value, dict):
                    # Header names are arbitrary. Never retain any header value, even a container.
                    masked[key] = {name: MASKED_VALUE for name in value}
                    continue
                capture_settings = root and key == "token_id_capture" and isinstance(value, dict)
                credential_options = capture and key not in _CAPTURE_PUBLIC_FIELDS
                child_secret = (
                    secret
                    or credential_options
                    or (
                        looks_like_secret_key(key) and not capture_settings and not _is_token_count_or_flag(key, value)
                    )
                )
                masked[key] = hide(value, child_secret, capture=capture_settings)
            return masked
        if isinstance(node, list):
            return [MASKED_VALUE] * len(node) if secret else [hide(item) for item in node]
        if secret:
            return MASKED_VALUE
        return mask_url_passwords(node) if isinstance(node, str) else node

    return hide(data, root=True)
