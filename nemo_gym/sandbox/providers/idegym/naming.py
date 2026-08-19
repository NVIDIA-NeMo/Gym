# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Naming rules for IdeGYM clients and servers.

Names carry more weight in IdeGYM than in most backends, which is why they get their
own module. A *client* name is what resource-quota rules match by regex and what groups
a job in the dashboard. A *server* name becomes the Kubernetes resource name, is what
IdeGYM matches when looking for a server to reuse, and must satisfy RFC-1035 syntax
with room left for the ``-<server_id>`` suffix the orchestrator appends.
"""

import re
import uuid

from nemo_gym.sandbox.providers.idegym.config import (
    MAX_SERVER_NAME_LENGTH,
    SERVER_ID_SUFFIX_RESERVE,
    SERVER_NAME_UNIQUE_SUFFIX_LENGTH,
)


# Names end up in Kubernetes objects, log lines, and operator-written regexes, so
# they are reduced to a conservative DNS-ish alphabet: an attribution value or task
# id picked up from the environment must not be able to produce something invalid.
_UNSAFE_NAME_CHARS = re.compile(r"[^a-z0-9-]+")

MAX_CLIENT_NAME_LENGTH = 63
# RFC-1035 label names must start with a letter, so a stem that survives
# sanitization but starts with a digit needs a fallback.
FALLBACK_SERVER_NAME_STEM = "idegym"


def sanitize_name(value: str) -> str:
    """Lowercase ``value`` and collapse everything outside ``[a-z0-9-]``."""
    return _UNSAFE_NAME_CHARS.sub("-", value.strip().lower()).strip("-")


def clamp_client_name(value: str) -> str:
    """Trim a client name to what IdeGYM accepts."""
    return value[:MAX_CLIENT_NAME_LENGTH].rstrip("-")


def refresh_unique_suffix(name: str) -> str:
    """Return ``name`` with a fresh uniqueness suffix.

    Used when a start-server call is retried: the previous attempt may have landed
    even though its response was lost, and reusing the name would make IdeGYM treat
    that half-created server as a reuse candidate.
    """
    stem, _, _ = name.rpartition("-")
    return f"{stem or name}-{uuid.uuid4().hex[:SERVER_NAME_UNIQUE_SUFFIX_LENGTH]}"


def generate_server_name(prefix: str, hints: list[str], *, unique_suffix: str | None = None) -> str:
    """Build a unique RFC-1035 server name from ``prefix`` and ``hints``.

    ``hints`` are best-effort context (a benchmark instance id, say) and are
    truncated away first when the budget runs out; the uniqueness suffix is never
    dropped, because two concurrent sandboxes sharing a name would make IdeGYM
    treat one as a candidate for reusing the other.
    """
    suffix = unique_suffix or uuid.uuid4().hex[:SERVER_NAME_UNIQUE_SUFFIX_LENGTH]
    budget = MAX_SERVER_NAME_LENGTH - SERVER_ID_SUFFIX_RESERVE - len(suffix) - 1
    if budget < 1:
        raise ValueError(f"Server name suffix {suffix!r} leaves no room for a name stem")
    stem = sanitize_name("-".join([prefix, *hints]))[:budget].rstrip("-")
    if not stem or not stem[0].isalpha():
        stem = sanitize_name(prefix)[:budget].rstrip("-")
    if not stem or not stem[0].isalpha():
        stem = FALLBACK_SERVER_NAME_STEM[:budget]
    return f"{stem}-{suffix}"
