# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
New CLI implementation for NeMo Gym.

This module provides a redesigned CLI with a single entry point and organized subcommands.
"""
import json
import os
import platform
import sys

import click
import psutil
from importlib.metadata import version as md_version

from nemo_gym import PARENT_DIR, __version__
from nemo_gym.cli_new.config import config
from nemo_gym.cli_new.data import data
from nemo_gym.cli_new.dataset import dataset
from nemo_gym.cli_new.server import server
from nemo_gym.cli_new.test import test


@click.group()
@click.version_option(version=__version__, package_name="nemo_gym")
def cli():
    """
    NeMo Gym - Build and train LLMs with Reinforcement Learning.

    Use 'ng <command> --help' for more information on a specific command.
    """
    pass


# Register command groups
cli.add_command(server)
cli.add_command(data)
cli.add_command(dataset)
cli.add_command(config)
cli.add_command(test)


def _extract_cli_structure(group: click.Group, parent_names: list[str] = None) -> dict:
    """Extract the CLI structure for static completion generation."""
    if parent_names is None:
        parent_names = []

    structure = {"commands": {}, "options": []}

    # Extract options for this group/command
    for param in group.params:
        if isinstance(param, click.Option):
            opts = list(param.opts) + list(param.secondary_opts)
            structure["options"].extend(opts)

    # Extract subcommands if this is a group
    if isinstance(group, click.Group):
        for name, cmd in group.commands.items():
            cmd_info = {"name": name, "options": [], "subcommands": {}}

            # Extract options for this command
            for param in cmd.params:
                if isinstance(param, click.Option):
                    opts = list(param.opts) + list(param.secondary_opts)
                    cmd_info["options"].extend(opts)
                elif isinstance(param, click.Argument):
                    # Store argument info (e.g., for shell choices)
                    if isinstance(param.type, click.Choice):
                        cmd_info["argument_choices"] = param.type.choices

            # Recursively extract subcommands
            if isinstance(cmd, click.Group):
                cmd_info["subcommands"] = _extract_cli_structure(cmd, parent_names + [name])["commands"]

            structure["commands"][name] = cmd_info

    return structure


def _generate_static_bash_completion(cli_group: click.Group) -> str:
    """Generate a truly static bash completion script by introspecting the CLI."""
    structure = _extract_cli_structure(cli_group)

    # Build command cases
    cases = []
    for cmd_name, cmd_info in structure["commands"].items():
        opts = " ".join(cmd_info["options"] + ["--help"])

        if cmd_info["subcommands"]:
            # Has subcommands
            subcmd_names = " ".join(cmd_info["subcommands"].keys())
            subcases = []
            for subcmd_name, subcmd_info in cmd_info["subcommands"].items():
                subopts = " ".join(subcmd_info["options"] + ["--help"])
                subcases.append(f'            elif [ "${{commands[1]}}" = "{subcmd_name}" ]; then\n                COMPREPLY=($(compgen -W "{subopts}" -- "$cur"))')

            subcommand_handling = "\n".join(subcases)
            cases.append(f'''        {cmd_name})
            if [ ${{#commands[@]}} -eq 1 ]; then
                COMPREPLY=($(compgen -W "{subcmd_names} --help" -- "$cur"))
{subcommand_handling}
            fi
            ;;''')
        else:
            # No subcommands, just handle arguments/options
            if "argument_choices" in cmd_info:
                # Has argument choices (like shell types)
                choices = " ".join(cmd_info["argument_choices"])
                cases.append(f'''        {cmd_name})
            if [ ${{#commands[@]}} -eq 1 ]; then
                COMPREPLY=($(compgen -W "{choices}" -- "$cur"))
            else
                COMPREPLY=($(compgen -W "{opts}" -- "$cur"))
            fi
            ;;''')
            else:
                cases.append(f'''        {cmd_name})
            COMPREPLY=($(compgen -W "{opts}" -- "$cur"))
            ;;''')

    cases_str = "\n".join(cases)

    # Top-level commands
    top_commands = " ".join(structure["commands"].keys()) + " " + " ".join(structure["options"])

    return f'''# NeMo Gym bash completion (static, auto-generated)

_ng_completion() {{
    local cur prev words cword

    # Use _init_completion if available (from bash-completion package)
    # Otherwise, use basic fallback
    if declare -F _init_completion >/dev/null 2>&1; then
        _init_completion || return
    else
        # Simple fallback
        cur="${{COMP_WORDS[COMP_CWORD]}}"
        prev="${{COMP_WORDS[COMP_CWORD-1]}}"
        words=("${{COMP_WORDS[@]}}")
        cword=$COMP_CWORD
    fi

    # Get the command chain
    local commands=()
    local i=1
    while [ $i -lt $cword ]; do
        commands+=("${{words[$i]}}")
        ((i++))
    done

    # Top-level commands
    local top_commands="{top_commands}"

    # If we're at the top level
    if [ ${{#commands[@]}} -eq 0 ]; then
        COMPREPLY=($(compgen -W "$top_commands" -- "$cur"))
        return 0
    fi

    # Handle subcommands
    case "${{commands[0]}}" in
{cases_str}
    esac
}}

complete -F _ng_completion ng
'''


def _generate_static_zsh_completion(cli_group: click.Group) -> str:
    """Generate a truly static zsh completion script by introspecting the CLI."""
    structure = _extract_cli_structure(cli_group)

    # Build subcommand definitions
    subcommand_defs = []

    for cmd_name, cmd_info in structure["commands"].items():
        if cmd_info["subcommands"]:
            # Command has subcommands
            subcmd_list = []
            for subcmd_name, subcmd_info in cmd_info["subcommands"].items():
                # Get options for this subcommand
                opts_lines = []
                for opt in subcmd_info["options"]:
                    if opt.startswith("--"):
                        opt_name = opt[2:]
                        opts_lines.append(f"            '{opt}[{opt_name}]'")

                if opts_lines:
                    opts_str = " \\\n".join(opts_lines)
                    subcmd_list.append(f'''        {subcmd_name})
            _arguments \\
{opts_str} \\
                '--help[Show help]'
            ;;''')
                else:
                    subcmd_list.append(f'''        {subcmd_name})
            _arguments '--help[Show help]'
            ;;''')

            subcmd_cases = "\n".join(subcmd_list)
            subcmd_names = " ".join(cmd_info["subcommands"].keys())

            subcommand_defs.append(f'''    {cmd_name})
        local subcmds="{subcmd_names}"
        _describe 'subcommand' subcmds

        case $line[1] in
{subcmd_cases}
        esac
        ;;''')
        else:
            # No subcommands, just options
            opts_lines = []
            for opt in cmd_info["options"]:
                if opt.startswith("--"):
                    opt_name = opt[2:]
                    opts_lines.append(f"        '{opt}[{opt_name}]'")

            if "argument_choices" in cmd_info:
                choices = " ".join(cmd_info["argument_choices"])
                opts_lines.insert(0, f"        '1:shell:({choices})'")

            if opts_lines:
                opts_str = " \\\n".join(opts_lines)
                subcommand_defs.append(f'''    {cmd_name})
        _arguments \\
{opts_str} \\
            '--help[Show help]'
        ;;''')
            else:
                subcommand_defs.append(f'''    {cmd_name})
        _arguments '--help[Show help]'
        ;;''')

    subcommand_cases = "\n".join(subcommand_defs)

    # Top-level commands
    cmd_list = " ".join(f"'{name}:{cmd['options']}'" for name, cmd in structure["commands"].items())

    return f'''#compdef ng
# NeMo Gym zsh completion (static, auto-generated)

_ng() {{
    local curcontext="$curcontext" state line
    typeset -A opt_args

    local -a commands
    commands=(
        'server:Server management'
        'data:Data collection and preparation'
        'dataset:Dataset registry operations'
        'config:Configuration utilities'
        'test:Test commands'
        'version:Show version information'
        'generate-shell-completion:Generate shell completion'
    )

    _arguments -C \\
        '1: :->command' \\
        '*::arg:->args' \\
        '--version[Show version]' \\
        '--help[Show help]'

    case $state in
        command)
            _describe 'command' commands
            ;;
        args)
            case $line[1] in
{subcommand_cases}
            esac
            ;;
    esac
}}

_ng
'''


def _generate_static_fish_completion(cli_group: click.Group) -> str:
    """Generate a truly static fish completion script by introspecting the CLI."""
    structure = _extract_cli_structure(cli_group)

    lines = ["# NeMo Gym fish completion (static, auto-generated)\n"]

    # Top-level commands
    for cmd_name, cmd_info in structure["commands"].items():
        desc = cmd_info.get("description", f"{cmd_name} command")
        lines.append(f'complete -c ng -n "__fish_use_subcommand" -a "{cmd_name}" -d "{desc}"')

    lines.append("")

    # Subcommands and options
    for cmd_name, cmd_info in structure["commands"].items():
        if cmd_info["subcommands"]:
            # This command has subcommands
            subcmd_names = " ".join(cmd_info["subcommands"].keys())
            for subcmd_name, subcmd_info in cmd_info["subcommands"].items():
                # Define the subcommand
                lines.append(f'complete -c ng -n "__fish_seen_subcommand_from {cmd_name}; and not __fish_seen_subcommand_from {subcmd_names}" -a "{subcmd_name}" -d "{subcmd_name} command"')

                # Add options for this subcommand
                for opt in subcmd_info["options"]:
                    if opt.startswith("--"):
                        opt_name = opt[2:]
                        short_opt = ""
                        if f"-{opt_name[0]}" in subcmd_info["options"]:
                            short_opt = f" -s {opt_name[0]}"
                        lines.append(f'complete -c ng -n "__fish_seen_subcommand_from {cmd_name}; and __fish_seen_subcommand_from {subcmd_name}" -l {opt_name}{short_opt} -d "{opt_name}"')
        else:
            # No subcommands, just add options
            for opt in cmd_info["options"]:
                if opt.startswith("--"):
                    opt_name = opt[2:]
                    short_opt = ""
                    # Check if there's a short option
                    for o in cmd_info["options"]:
                        if o.startswith("-") and not o.startswith("--") and len(o) == 2:
                            short_opt = f" -s {o[1]}"
                            break
                    lines.append(f'complete -c ng -n "__fish_seen_subcommand_from {cmd_name}" -l {opt_name}{short_opt} -d "{opt_name}"')

            # Handle argument choices (like shell types)
            if "argument_choices" in cmd_info:
                for choice in cmd_info["argument_choices"]:
                    lines.append(f'complete -c ng -n "__fish_seen_subcommand_from {cmd_name}" -a "{choice}" -d "{choice}"')

    return "\n".join(lines) + "\n"


@cli.command(name="generate-shell-completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]), required=True)
@click.option("--install", is_flag=True, help="Install completion script to the appropriate location")
def generate_shell_completion(shell: str, install: bool):
    """
    Generate shell completion script for the specified shell.

    This generates a static completion script that can be sourced in your shell
    configuration file.

    Usage:

        # Bash
        eval "$(ng generate-shell-completion bash)"

        # Zsh
        eval "$(ng generate-shell-completion zsh)"

        # Fish
        ng generate-shell-completion fish > ~/.config/fish/completions/ng.fish

    The --install flag automates the setup:

        ng generate-shell-completion bash --install

    This writes the completion script to a file and adds the source command
    to your shell RC file.
    """
    from pathlib import Path

    # Generate the static completion script by introspecting the CLI
    if shell == "bash":
        completion_script = _generate_static_bash_completion(cli)
    elif shell == "zsh":
        completion_script = _generate_static_zsh_completion(cli)
    elif shell == "fish":
        completion_script = _generate_static_fish_completion(cli)
    else:
        click.echo(f"Unsupported shell: {shell}", err=True)
        sys.exit(1)

    if install:
        # Install the completion script to appropriate location
        home = Path.home()

        if shell == "bash":
            cache_file = home / ".ng-complete.bash"
            rc_file = home / ".bashrc"
            source_line = f"\n# NeMo Gym CLI completion\n[ -f {cache_file} ] && . {cache_file}\n"
        elif shell == "zsh":
            cache_file = home / ".ng-complete.zsh"
            rc_file = home / ".zshrc"
            source_line = f"\n# NeMo Gym CLI completion\n[ -f {cache_file} ] && . {cache_file}\n"
        elif shell == "fish":
            cache_file = home / ".config/fish/completions/ng.fish"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            rc_file = None  # Fish doesn't need RC modification
            source_line = None
        else:
            click.echo(f"Unsupported shell: {shell}", err=True)
            sys.exit(1)

        # Write the completion script to cache file
        cache_file.write_text(completion_script)
        click.echo(f"✓ Completion script written to: {cache_file}")

        # Add source line to RC file (except for fish)
        if rc_file and source_line:
            rc_content = rc_file.read_text() if rc_file.exists() else ""

            if str(cache_file) not in rc_content and "ng-complete" not in rc_content:
                with rc_file.open("a") as f:
                    f.write(source_line)
                click.echo(f"✓ Added completion hook to: {rc_file}")
                click.echo(f"\nRestart your shell or run: source {rc_file}")
            else:
                click.echo(f"✓ Completion hook already exists in: {rc_file}")
        elif shell == "fish":
            click.echo("\nFish completion installed. Restart your shell to enable it.")
    else:
        # Just print the completion script
        click.echo(completion_script)


@cli.command()
@click.option("--json", "json_format", is_flag=True, help="Output in JSON format for programmatic use.")
def version(json_format: bool):
    """
    Display gym version and system information.

    Example:

        ng version

        ng version --json
    """
    version_info = {
        "nemo_gym": __version__,
        "python": platform.python_version(),
        "python_path": sys.executable,
        "installation_path": str(PARENT_DIR),
    }

    key_deps = [
        "openai",
        "ray",
    ]

    dependencies = {dep: md_version(dep) for dep in key_deps}
    version_info["dependencies"] = dependencies

    # System info
    version_info["system"] = {
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpus": os.cpu_count(),
    }

    # Memory info
    mem = psutil.virtual_memory()
    version_info["system"]["memory_gb"] = round(mem.total / (1024**3), 2)

    # Output
    if json_format:
        print(json.dumps(version_info))
    else:
        click.echo(f"NeMo Gym v{version_info['nemo_gym']}")
        click.echo(f"Python {version_info['python']} ({version_info['python_path']})")
        click.echo(f"Installation: {version_info['installation_path']}")
        click.echo()
        click.echo("Key Dependencies:")
        for dep, ver in version_info["dependencies"].items():
            click.echo(f"  {dep}: {ver}")
        click.echo()
        click.echo("System:")
        for key, value in version_info["system"].items():
            click.echo(f"  {key.replace('_', ' ').title()}: {value}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
