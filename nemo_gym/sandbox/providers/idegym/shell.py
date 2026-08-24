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

"""Turning a sandbox exec request into one script for IdeGYM's bash tool.

``/api/tools/bash`` takes a command and nothing else — each call is a fresh
``bash -c`` in the server's project directory with a cleaned environment — so
everything ``exec(command, cwd=, env=, user=)`` promises is expressed inside the
script. Two sandbox-side details shape it: the executor runs
``source <bash-integration> && <script>`` where ``&&`` binds only to the first
statement, so the script is emitted as one ``{ ... }`` group; and it travels as a single
``execve()`` argument, which Linux caps at 128 KiB, so an oversized script is rejected
here against ``MAX_COMMAND_BYTES`` rather than failing with a confusing ``E2BIG``.
"""

import logging
import re
import shlex

from nemo_gym.sandbox.providers.idegym.config import MAX_COMMAND_BYTES, IdeGymExecConfig, UserMode
from nemo_gym.sandbox.providers.idegym.errors import IdeGymCommandTooLongError


LOGGER = logging.getLogger(__name__)

# POSIX portable environment variable names. `export` cannot carry anything else, so an
# invalid name has to be a hard error rather than a silently dropped value. Matched with
# `fullmatch`, since `$` would also accept a trailing newline.
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def quote(value: str) -> str:
    """Shell-quote one value for inclusion in a generated script."""
    return shlex.quote(value)


def directory_exists_script(path: str) -> str:
    """A script that succeeds only if ``path`` is a directory in the sandbox."""
    return f"[ -d {quote(path)} ]"


class BashScriptBuilder:
    """Builds the bash script for one sandbox command.

    Stateless apart from its config, so one builder is shared by every sandbox of
    a provider.
    """

    def __init__(self, config: IdeGymExecConfig) -> None:
        self._config = config
        self._warned_about_user = False

    def build(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
    ) -> str:
        """Return the script that runs ``command`` under the requested context.

        Raises:
            ValueError: If ``env`` holds a name ``export`` cannot carry, or a user
                switch is configured but the requested user cannot be expressed.
            IdeGymCommandTooLongError: If the finished script exceeds what the
                sandbox's shell can accept as a single argument.
        """
        inner = self._inner_script(command, cwd=cwd, env=env)
        script = self._apply_user(inner, user)
        # A group keeps every statement conditional on the executor's `source` prefix
        # and makes the group's last statement supply the exit code. The leading `:`
        # keeps the group non-empty: a blank or comment-only command would otherwise
        # make bash fail on `{ }` with a syntax error attributed to the caller.
        script = f"{{ :\n{script}\n}}"
        self._check_size(script, command)
        return script

    def _inner_script(self, command: str, *, cwd: str | None, env: dict[str, str] | None) -> str:
        lines: list[str] = []
        if cwd:
            # Fail loudly: a missing working directory is a configuration problem,
            # and silently running in the image's project directory instead would
            # make the command look like it merely failed.
            lines.append(f"cd -- {quote(cwd)} || exit 1")
        for key, value in (env or {}).items():
            name = str(key)
            if not _ENV_NAME.fullmatch(name):
                raise ValueError(
                    f"Environment variable name {name!r} is not a valid shell identifier, so the idegym "
                    f"provider cannot export it into the sandbox"
                )
            lines.append(f"export {name}={quote(str(value))}")
        lines.append(command)
        return "\n".join(lines)

    def _apply_user(self, script: str, user: str | int | None) -> str:
        """Wrap ``script`` so it runs as ``user``, per the configured user mode."""
        if user is None:
            return script
        mode = UserMode(self._config.user_mode)
        if mode is UserMode.IGNORE:
            if not self._warned_about_user:
                self._warned_about_user = True
                LOGGER.warning(
                    f"The idegym provider cannot run commands as user={user!r}: the IdeGYM bash tool has no "
                    f"user field. Commands run as the server container's user, which "
                    f"provider_options.run_as_root controls. Set exec.user_mode to 'runuser' or 'su' if the "
                    f"image ships one of those tools."
                )
            return script
        # Both tools resolve the user through getpwnam, so a bare uid is not something
        # they can be handed.
        if isinstance(user, int) or str(user).isdigit():
            raise ValueError(
                f"exec.user_mode={mode.value!r} needs a user name, but got the numeric id {user!r}. Pass a "
                f"name, or use exec.user_mode='ignore' to run as the container's own user."
            )
        name = quote(str(user))
        # Both pin the shell to bash rather than inherit the target user's login shell,
        # which is `/sbin/nologin` on a service account and would refuse the command.
        if mode is UserMode.RUNUSER:
            return f"exec runuser -u {name} -- bash -c {quote(script)}"
        return f"exec su -s /bin/bash -c {quote(script)} {name}"

    def _check_size(self, script: str, command: str) -> None:
        size = len(script.encode())
        if size <= MAX_COMMAND_BYTES:
            return
        raise IdeGymCommandTooLongError(
            f"The generated sandbox script is {size} bytes, over the {MAX_COMMAND_BYTES} byte limit the "
            f"IdeGYM bash tool can pass to the shell as one argument. Write the payload to a file and run "
            f"that instead (command starts with {command[:80]!r})."
        )
