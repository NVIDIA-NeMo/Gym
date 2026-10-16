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
"""Derive the HLE-Diamond search exclusion policy from the upstream HLE blocklist.

The upstream harness (https://github.com/centerforaisafety/hle, ``docs/blocklist.json``, MIT licence) ships
seven rule families for its ``web_search``/``web_fetch`` wrapper. NeMo Gym's Tavily resource
(``resources_servers/tavily_search/app.py``, ``URLExclusionPolicy``) can express exactly two property types:

* ``domain``: an exact hostname or any of its subdomains;
* ``url_substring``: a case-insensitive substring of the canonical URL (also checked after percent-decoding).

This script maps upstream ``domains`` to ``domain`` rules (IDNA-normalised, lower-cased, validated as a
hostname, deduplicated) and upstream ``url_prefixes`` to ``url_substring`` rules (lower-cased, scheme and
leading ``www.`` removed as upstream's matching does, deduplicated), then unions the result with
``benchmarks/hle/search_exclusions.json`` so the Diamond policy is at least as strict as Gym's HLE policy.
Prefix rules become substring rules, so the Gym policy is a superset of upstream's for these two families.

The regex-based upstream families (``serp_url_patterns``, ``url_patterns``, ``patterns_case_sensitive``,
``answer_title_patterns``, ``answer_title_allow_hosts``) have no equivalent in Gym's format. They are listed by
name and count under the top-level ``unrepresented_upstream_rules`` key, which the Tavily parser ignores
because it reads only ``notices``.

Usage::

    python benchmarks/hle_diamond/build_search_exclusions.py --blocklist /path/to/blocklist.json
    python benchmarks/hle_diamond/build_search_exclusions.py --download [--blocklist /path/to/save/copy.json]

Nothing is downloaded unless ``--download`` is passed. The blocklist's SHA-256 must match the pinned upstream
snapshot; to move to a newer upstream revision update ``UPSTREAM_COMMIT`` and ``UPSTREAM_BLOCKLIST_SHA256``
together, regenerate, and review the diff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import urlopen


UPSTREAM_REPOSITORY = "https://github.com/centerforaisafety/hle"
UPSTREAM_COMMIT = "22ed3074b1e7134bcbc09028d0ba320839b0655"  # pragma: allowlist secret
UPSTREAM_BLOCKLIST_PATH = "docs/blocklist.json"
UPSTREAM_BLOCKLIST_SHA256 = (
    "3abaf35e03d6d5d361d2611746683a32315fe73ef851ec5f33c54b3c6a526575"  # pragma: allowlist secret
)
UPSTREAM_BLOCKLIST_URL = (
    f"https://raw.githubusercontent.com/centerforaisafety/hle/{UPSTREAM_COMMIT}/{UPSTREAM_BLOCKLIST_PATH}"
)
UPSTREAM_LICENSE = "MIT (Copyright (c) 2025 centerforaisafety)"

REPO_ROOT = Path(__file__).resolve().parents[2]
HLE_EXCLUSIONS_PATH = REPO_ROOT / "benchmarks" / "hle" / "search_exclusions.json"
OUTPUT_PATH = Path(__file__).resolve().with_name("search_exclusions.json")

# Upstream rule families that map onto a Gym property type.
REPRESENTED_UPSTREAM_KEYS = {"domains": "domain", "url_prefixes": "url_substring"}
# Upstream regex families known at the pinned revision; any other non-represented key is reported as well.
UNREPRESENTED_RULE_FAMILIES = (
    "serp_url_patterns",
    "url_patterns",
    "patterns_case_sensitive",
    "answer_title_patterns",
    "answer_title_allow_hosts",
)
GYM_PROPERTY_TYPES = frozenset(REPRESENTED_UPSTREAM_KEYS.values())

_HOSTNAME_LABEL = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")
_URL_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://")
_MAX_HOSTNAME_LENGTH = 253


def normalize_domain(value: Any) -> tuple[str | None, str | None]:
    """Return ``(hostname, None)`` for a usable ``domain`` rule or ``(None, reason)`` when it must be dropped.

    Mirrors ``URLExclusionPolicy`` (IDNA encode, lower-case, strip the trailing dot) and then rejects strings
    the parser would accept but that could never equal a URL hostname (paths, schemes, spaces, single labels).
    """
    if not isinstance(value, str):
        return None, "not a string"
    try:
        hostname = value.strip().encode("idna").decode().lower().rstrip(".")
    except UnicodeError as exc:
        return None, f"rejected by the IDNA codec used by the Gym parser ({exc})"
    if not hostname:
        return None, "empty"
    if len(hostname) > _MAX_HOSTNAME_LENGTH:
        return None, f"longer than {_MAX_HOSTNAME_LENGTH} characters"
    labels = hostname.split(".")
    if len(labels) < 2 or not all(_HOSTNAME_LABEL.match(label) for label in labels):
        return None, "not a hostname"
    return hostname, None


def normalize_url_substring(value: Any) -> tuple[str | None, str | None]:
    """Return ``(substring, None)`` for a usable ``url_substring`` rule or ``(None, reason)`` when dropped.

    Upstream matches ``url_prefixes`` case-insensitively after removing the scheme and a leading ``www.``;
    the same normalisation is applied here. An empty substring would block every URL, so it is dropped.
    """
    if not isinstance(value, str):
        return None, "not a string"
    substring = _URL_SCHEME.sub("", value.strip().lower(), count=1)
    if substring.startswith("www."):
        substring = substring[len("www.") :]
    if not substring:
        return None, "empty"
    return substring, None


_NORMALIZERS = {"domain": normalize_domain, "url_substring": normalize_url_substring}


@dataclass
class ExclusionPolicy:
    """Gym-format rules plus the audit trail of what could not be carried over."""

    domains: list[str] = field(default_factory=list)
    url_substrings: list[str] = field(default_factory=list)
    dropped: list[tuple[str, Any, str]] = field(default_factory=list)  # (source, original value, reason)
    unrepresented: dict[str, int] = field(default_factory=dict)

    def add(self, property_type: str, value: Any, source: str) -> None:
        normalizer = _NORMALIZERS.get(property_type)
        if normalizer is None:
            self.dropped.append((source, value, f"property type {property_type!r} is not carried into this policy"))
            return
        normalized, reason = normalizer(value)
        if normalized is None:
            self.dropped.append((source, value, reason))
            return
        target = self.domains if property_type == "domain" else self.url_substrings
        if normalized not in target:
            target.append(normalized)


def build_policy(upstream: dict[str, Any], hle_exclusions: dict[str, Any] | None = None) -> ExclusionPolicy:
    """Map the upstream blocklist onto Gym rules and union them with Gym's HLE exclusion file."""
    if not isinstance(upstream, dict):
        raise ValueError("upstream blocklist must be a JSON object")
    policy = ExclusionPolicy()
    for key, property_type in REPRESENTED_UPSTREAM_KEYS.items():
        entries = upstream.get(key, [])
        if not isinstance(entries, list):
            raise ValueError(f"upstream {key!r} must be a list")
        for value in entries:
            policy.add(property_type, value, f"upstream {key}")
    for key, entries in upstream.items():
        if key not in REPRESENTED_UPSTREAM_KEYS:
            policy.unrepresented[key] = len(entries) if isinstance(entries, (list, dict)) else 1
    if hle_exclusions is not None:
        for notice in hle_exclusions["notices"]:
            for prop in notice["properties"]:
                policy.add(prop["type"], prop["value"], "benchmarks/hle/search_exclusions.json")
    return policy


def render(policy: ExclusionPolicy) -> dict[str, Any]:
    """Serialise the policy in the Tavily resource's ``exclude_domains_file_path`` format.

    ``URLExclusionPolicy`` reads only ``notices[*].properties``; ``source`` and ``unrepresented_upstream_rules``
    are attribution and audit metadata it ignores (verified by tests/unit_tests/test_hle_diamond_search_exclusions.py).
    """
    properties = [{"type": "domain", "value": domain} for domain in policy.domains]
    properties += [{"type": "url_substring", "value": substring} for substring in policy.url_substrings]
    return {
        "source": {
            "description": (
                "Search exclusions for the HLE-Diamond with-tools presets: the upstream HLE blocklist's 'domains' "
                "and 'url_prefixes' (the two rule families NeMo Gym's Tavily resource can express) unioned with "
                "benchmarks/hle/search_exclusions.json. Generated by benchmarks/hle_diamond/build_search_exclusions.py; "
                "do not edit by hand."
            ),
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_path": UPSTREAM_BLOCKLIST_PATH,
            "upstream_commit": (
                f"{UPSTREAM_COMMIT[:8]} (full revision pinned as UPSTREAM_COMMIT in build_search_exclusions.py)"
            ),
            "upstream_license": UPSTREAM_LICENSE,
        },
        "unrepresented_upstream_rules": dict(policy.unrepresented),
        "notices": [{"properties": properties}],
    }


def download_pinned_blocklist() -> bytes:
    with urlopen(UPSTREAM_BLOCKLIST_URL, timeout=60) as response:  # noqa: S310 - fixed https URL
        return response.read()


def report(policy: ExclusionPolicy, output: Path) -> None:
    print(
        f"wrote {output}: {len(policy.domains)} domain rules, {len(policy.url_substrings)} url_substring rules",
        file=sys.stderr,
    )
    families = ", ".join(f"{name}={count}" for name, count in policy.unrepresented.items()) or "none"
    print(f"unrepresented upstream rule families (not expressible in Gym's format): {families}", file=sys.stderr)
    for name in policy.unrepresented:
        if name not in UNREPRESENTED_RULE_FAMILIES:
            print(f"warning: upstream rule family {name!r} is new to this generator; review it", file=sys.stderr)
    print(f"dropped {len(policy.dropped)} entries", file=sys.stderr)
    for source, value, reason in policy.dropped:
        print(f"  {source}: {value!r}: {reason}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--blocklist",
        type=Path,
        help="Local copy of the upstream docs/blocklist.json (required unless --download is given; with "
        "--download the fetched snapshot is also saved here).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=f"Fetch the blocklist at the pinned upstream revision from {UPSTREAM_BLOCKLIST_URL}",
    )
    parser.add_argument("--hle-exclusions", type=Path, default=HLE_EXCLUSIONS_PATH, help="Gym HLE policy to union.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Where to write the Gym-format policy.")
    args = parser.parse_args(argv)

    if args.download:
        raw = download_pinned_blocklist()
        if args.blocklist is not None:
            args.blocklist.write_bytes(raw)
    elif args.blocklist is not None:
        raw = args.blocklist.read_bytes()
    else:
        parser.error("pass --blocklist PATH (a local copy of the upstream blocklist) or --download")

    digest = hashlib.sha256(raw).hexdigest()
    if digest != UPSTREAM_BLOCKLIST_SHA256:
        parser.error(
            f"blocklist SHA-256 {digest} does not match the pinned upstream snapshot at commit {UPSTREAM_COMMIT}; "
            "update UPSTREAM_COMMIT and UPSTREAM_BLOCKLIST_SHA256 together before regenerating"
        )

    policy = build_policy(json.loads(raw), json.loads(args.hle_exclusions.read_text()))
    args.output.write_text(json.dumps(render(policy), indent=2, ensure_ascii=False) + "\n")
    report(policy, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
