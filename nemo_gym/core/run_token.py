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
"""The token that a run's own servers and clients send with their calls to each other.

Code in a sandbox can often reach a run's servers over the network, and a caller's address does not tell it apart from
the run's servers: a provider's port forwarder makes sandbox traffic arrive from loopback. So the launcher
(``ServerSupervisor``) makes one token per run, and every server it starts inherits it through the environment
(``RUN_TOKEN_ENV``). ``ServerClient`` sends it in ``RUN_TOKEN_HEADER`` with every call, and a server that requires it
answers its orchestration routes only for a caller that sends it (``SimpleServer.run_token_routes``).

The token never enters the config, so the head server never serves it. A sandbox provider that passes its own
environment on to the code it runs drops the variable (``nemo_gym.core.run_session_secrets``). Clients that run as other
processes of the same user on the same host, such as rollout collection started in another terminal, read it from a
file only that user can read (``run_token_file``).
"""

import hmac
import os
import re
import secrets
import tempfile
from pathlib import Path
from typing import Optional


# The variable that carries the run's token to every process the launcher starts.
RUN_TOKEN_ENV = "NEMO_GYM_RUN_TOKEN"
# The request header that carries it. Not Authorization, which already carries MCP bearer tokens, checkpoint control
# tokens and model API keys.
RUN_TOKEN_HEADER = "X-NeMo-Gym-Run-Token"
# The directory for the files that hold run tokens, when not the user's cache directory.
RUN_TOKEN_DIR_ENV = "NEMO_GYM_RUN_TOKEN_DIR"


def run_token() -> Optional[str]:
    """This process's run token, or ``None`` outside a run that has one."""
    return os.environ.get(RUN_TOKEN_ENV) or None


def mint_run_token() -> bool:
    """Give this process a new random run token unless it has one, and return whether it made one.

    The launcher calls it before it starts any server, so every server inherits the token. A token set beforehand is
    kept: setting ``RUN_TOKEN_ENV`` shares one token among processes that no one launcher starts, such as a run that
    spans hosts.
    """
    if run_token() is not None:
        return False
    os.environ[RUN_TOKEN_ENV] = secrets.token_urlsafe(32)
    return True


def is_run_token(value: Optional[str]) -> bool:
    """Whether ``value`` is this process's run token, compared in constant time."""
    token = run_token()
    # A header value can hold characters outside ASCII, which a constant-time comparison of strings refuses.
    return token is not None and value is not None and value.isascii() and hmac.compare_digest(value, token)


def run_token_file(head_server_host: str, head_server_port: int) -> Path:
    """The file that holds the token of the run whose head server a client of this host reaches at
    ``head_server_host``:``head_server_port``.

    It is in the directory ``RUN_TOKEN_DIR_ENV`` names, or else in the user's cache directory, and only the user can
    read it.
    """
    directory = os.environ.get(RUN_TOKEN_DIR_ENV)
    if not directory:
        directory = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "nemo-gym" / "run-tokens"
    host = re.sub(r"[^A-Za-z0-9.-]", "_", head_server_host)
    return Path(directory) / f"head-{host}-{head_server_port}"


def write_run_token_file(head_server_host: str, head_server_port: int, token: str) -> Path:
    """Write ``token`` to its ``run_token_file``, readable by this user only, and return the file."""
    path = run_token_file(head_server_host, head_server_port)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # mkstemp creates the file readable by this user only; the rename then replaces a file a previous run left.
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(token)
    os.replace(temporary, path)
    return path


def read_run_token_file(head_server_host: str, head_server_port: int) -> Optional[str]:
    """The token in the ``run_token_file`` of that head server, or ``None`` when there is none."""
    try:
        token = run_token_file(head_server_host, head_server_port).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return token or None


def remove_run_token_file(head_server_host: str, head_server_port: int, token: str) -> None:
    """Remove the ``run_token_file`` of that head server if it still holds ``token``.

    A run that started at the same address since has replaced the file, and keeps it.
    """
    if read_run_token_file(head_server_host, head_server_port) == token:
        run_token_file(head_server_host, head_server_port).unlink(missing_ok=True)
