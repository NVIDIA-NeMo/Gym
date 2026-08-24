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
a job in the dashboard. A *server* name is one of the filters IdeGYM matches on when a
``RESTART``/``RESET`` reuse strategy sends it looking for a server to reuse, and it
prefixes the Kubernetes resource name, so it must satisfy RFC-1035 syntax with room
left for the ``-<server_id>`` suffix the orchestrator appends. It need not be unique:
that suffix is what makes the Kubernetes name unique.
"""

import re

from nemo_gym.sandbox.providers.idegym.config import MAX_SERVER_NAME_LENGTH, SERVER_ID_SUFFIX_RESERVE


# Names end up in Kubernetes objects, log lines, and operator-written regexes, so an
# attribution value or task id picked up from the environment is reduced to a
# conservative DNS-ish alphabet rather than trusted to be valid.
_UNSAFE_NAME_CHARS = re.compile(r"[^a-z0-9-]+")

MAX_CLIENT_NAME_LENGTH = 63
# RFC-1035 labels must start with a letter, so a stem that survives sanitization but
# starts with a digit needs a fallback.
FALLBACK_SERVER_NAME_STEM = "idegym"


def sanitize_name(value: str) -> str:
    """Lowercase ``value`` and collapse everything outside ``[a-z0-9-]``."""
    return _UNSAFE_NAME_CHARS.sub("-", value.strip().lower()).strip("-")


def clamp_client_name(value: str) -> str:
    """Trim a client name to what IdeGYM accepts."""
    return value[:MAX_CLIENT_NAME_LENGTH].rstrip("-")


def generate_server_name(prefix: str, hints: list[str]) -> str:
    """Build an RFC-1035 server name from ``prefix`` and ``hints``.

    ``hints`` are best-effort context (a benchmark instance id) and are truncated away
    first when the budget runs out. The name deliberately carries no uniqueness suffix:
    IdeGYM appends its own autoincrement id to derive ``<server_name>-<server_id>``,
    and that is the name Kubernetes sees and the column the database keeps unique.
    """
    budget = MAX_SERVER_NAME_LENGTH - SERVER_ID_SUFFIX_RESERVE
    if budget < 1:
        raise ValueError(
            f"MAX_SERVER_NAME_LENGTH={MAX_SERVER_NAME_LENGTH} leaves no room for a name after "
            f"reserving {SERVER_ID_SUFFIX_RESERVE} for the server id IdeGYM appends"
        )
    name = sanitize_name("-".join([prefix, *hints]))[:budget].rstrip("-")
    if not name or not name[0].isalpha():
        name = sanitize_name(prefix)[:budget].rstrip("-")
    if not name or not name[0].isalpha():
        name = FALLBACK_SERVER_NAME_STEM[:budget]
    return name
