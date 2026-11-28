# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor's system-dependency step for Claude Code, as one POSIX sh script.

Before Harbor 0.24 installs Claude Code in a task container, its claude-code agent calls
``BaseInstalledAgent.ensure_system_dependencies(("curl", "bash", "nodejs", "npm", "procps"))``:
each dependency whose commands are all found with ``command -v`` is skipped, and the rest are
installed as root with the first package manager found (apt-get, dnf, yum, apk), using Harbor's
package names and command lines. Agents running under Harbor therefore have ``node``, ``npm``,
``ps`` and ``pkill`` even where the task image lacks them. The script below makes the same
decisions in one exec, logs each one on a ``harbor_deps:`` line, and exits with the package
manager's status (0 when nothing is missing or no package manager exists, where Harbor only warns).
It uses shell builtins until it runs the package manager, so it works without coreutils.
"""

import shlex
from dataclasses import dataclass, field


PACKAGE_MANAGERS = ("apt-get", "dnf", "yum", "apk")
_INSTALL_COMMANDS = {
    "apt-get": "apt-get update && apt-get install -y",
    "dnf": "dnf install -y",
    "yum": "yum install -y",
    "apk": "apk add --no-cache",
}


@dataclass(frozen=True)
class PackageSpec:
    """One of Harbor's ``SYSTEM_PACKAGES`` entries: present when every command resolves."""

    name: str
    commands: tuple[str, ...]
    packages: dict[str, tuple[str, ...]] = field(default_factory=dict)
    always_install: bool = False

    @classmethod
    def standard(cls, name: str) -> "PackageSpec":
        return cls(name=name, commands=(name,), packages={manager: (name,) for manager in PACKAGE_MANAGERS})


# The entries of Harbor 0.24's SYSTEM_PACKAGES that its claude-code install() ensures, in its order.
CLAUDE_CODE_DEPENDENCIES: tuple[PackageSpec, ...] = (
    PackageSpec.standard("curl"),
    PackageSpec.standard("bash"),
    PackageSpec(name="nodejs", commands=("node",), packages={manager: ("nodejs",) for manager in PACKAGE_MANAGERS}),
    PackageSpec.standard("npm"),
    PackageSpec(
        name="procps",
        commands=("pgrep",),
        packages={"apt-get": ("procps",), "dnf": ("procps-ng",), "yum": ("procps-ng",), "apk": ("procps",)},
    ),
)


def _log(message: str) -> str:
    return f"harbor_deps_log {shlex.quote(message)}"


def system_dependencies_command(specs: tuple[PackageSpec, ...] = CLAUDE_CODE_DEPENDENCIES) -> str:
    """Return the sh script that runs Harbor's ``ensure_system_dependencies`` for ``specs`` as root."""
    names = ", ".join(spec.name for spec in specs)
    lines = [
        "harbor_deps_log() { printf 'harbor_deps: %s\\n' \"$*\"; }",
        _log(f"ensure_system_dependencies({names}) as in Harbor 0.24 claude-code install()"),
        "harbor_deps_needed=''",
    ]
    for spec in specs:
        need = f'harbor_deps_needed="$harbor_deps_needed {spec.name}"'
        if spec.always_install:
            lines += [_log(f"{spec.name}: always installed"), need]
        elif not spec.commands:
            # Harbor's check is then an empty command, which succeeds.
            lines.append(_log(f"{spec.name}: present (no commands to check)"))
        else:
            check = " && ".join(f"command -v {shlex.quote(command)} >/dev/null 2>&1" for command in spec.commands)
            lines += [
                f"if {check}; then",
                "  " + _log(f"{spec.name}: present ({', '.join(spec.commands)})"),
                "else",
                "  " + _log(f"{spec.name}: missing"),
                f"  {need}",
                "fi",
            ]
    lines += [
        "if [ -z \"$harbor_deps_needed\" ]; then harbor_deps_log 'nothing to install'; exit 0; fi",
        "harbor_deps_manager=''",
        f"for harbor_deps_candidate in {' '.join(PACKAGE_MANAGERS)}; do",
        '  if command -v "$harbor_deps_candidate" >/dev/null 2>&1; then '
        'harbor_deps_manager="$harbor_deps_candidate"; break; fi',
        "done",
        'if [ -z "$harbor_deps_manager" ]; then',
        "  " + _log(f"warning: No supported package manager found; cannot ensure system dependencies: {names}"),
        "  exit 0",
        "fi",
        "harbor_deps_packages=''",
        "for harbor_deps_dep in $harbor_deps_needed; do",
        '  case "$harbor_deps_manager:$harbor_deps_dep" in',
    ]
    for spec in specs:
        for manager in PACKAGE_MANAGERS:
            packages = shlex.quote(" ".join(spec.packages.get(manager, ())))
            lines.append(f"    {shlex.quote(f'{manager}:{spec.name}')}) harbor_deps_add={packages} ;;")
    lines += [
        "    *) harbor_deps_add='' ;;",
        "  esac",
        # Harbor de-duplicates package names in first-seen order (dict.fromkeys).
        "  for harbor_deps_package in $harbor_deps_add; do",
        '    case " $harbor_deps_packages " in',
        '      *" $harbor_deps_package "*) ;;',
        '      *) harbor_deps_packages="${harbor_deps_packages:+$harbor_deps_packages }$harbor_deps_package" ;;',
        "    esac",
        "  done",
        "done",
        'case "$harbor_deps_manager" in',
        *(
            f'  {manager}) harbor_deps_install="{command} $harbor_deps_packages" ;;'
            for manager, command in _INSTALL_COMMANDS.items()
        ),
        "esac",
        'if [ "$harbor_deps_manager" = apt-get ]; then',
        '  harbor_deps_log "install with $harbor_deps_manager (DEBIAN_FRONTEND=noninteractive): $harbor_deps_install"',
        "  DEBIAN_FRONTEND=noninteractive",
        "  export DEBIAN_FRONTEND",
        "else",
        '  harbor_deps_log "install with $harbor_deps_manager: $harbor_deps_install"',
        "fi",
        'eval "$harbor_deps_install"',
        "harbor_deps_status=$?",
        'harbor_deps_log "install exit status $harbor_deps_status"',
    ]
    for spec in specs:
        for command in spec.commands:
            quoted = shlex.quote(command)
            lines.append(
                f"if command -v {quoted} >/dev/null 2>&1; then "
                f'{_log(f"after install: {command} at")} "$(command -v {quoted})"; '
                f"else {_log(f'after install: {command} missing')}; fi"
            )
    lines += [
        'if command -v node >/dev/null 2>&1; then harbor_deps_log "node $(node --version 2>&1)"; fi',
        'if command -v npm >/dev/null 2>&1; then harbor_deps_log "npm $(npm --version 2>&1)"; fi',
        'exit "$harbor_deps_status"',
    ]
    return "\n".join(lines)
