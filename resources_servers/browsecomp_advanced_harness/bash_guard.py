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
"""Command guards and output formatting for the read-only bash_command tool.

Stdlib only: the sandboxed agent runs this module inside a stock python image
(pod_bash.py), and the resources server imports it for the host path, so both
paths block the same commands and render the same output.
"""

import re
from pathlib import Path


_BASH_HEAD_BYTES = 8192
_BASH_TAIL_BYTES = 4096


def _bash_truncate(s: str) -> str:
    if len(s) <= _BASH_HEAD_BYTES + _BASH_TAIL_BYTES + 64:
        return s
    omitted = len(s) - _BASH_HEAD_BYTES - _BASH_TAIL_BYTES
    return (
        f"{s[:_BASH_HEAD_BYTES]}\n"
        f"[...truncated {omitted} bytes; head shown above, tail follows...]\n"
        f"{s[-_BASH_TAIL_BYTES:]}"
    )


def format_bash_output(stdout: str, stderr: str, exit_code: int) -> str:
    """Render a command result as stdout / --- stderr --- / [exit_code=N]."""
    truncated = (len(stdout) + len(stderr)) > (_BASH_HEAD_BYTES + _BASH_TAIL_BYTES) * 2
    stdout = _bash_truncate(stdout)
    stderr = _bash_truncate(stderr)
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append("--- stderr ---")
        parts.append(stderr)
    parts.append(f"[exit_code={exit_code}{' truncated' if truncated else ''}]")
    return "\n".join(parts)


# --- bash_command guards (ported byte-for-byte from an internal reference
# harness: _bash_denylisted + _bash_allowlisted). Layered deny-first,
# then default-deny allow-list. NOT a security boundary (the ulimit -f 0 in
# _run_bash_readonly is the kernel backstop); these block destructive/network/
# interpreter commands + command/process substitution + find -exec / sed -i. ---
_BASH_DENY_COMMANDS = {
    # destructive file ops / writes
    "rm",
    "rmdir",
    "unlink",
    "shred",
    "srm",
    "wipe",
    "mv",
    "cp",
    "dd",
    "truncate",
    "install",
    "ln",
    "tee",
    "mkfs",
    "mke2fs",
    "mkswap",
    "fsck",
    # permissions / metadata
    "chmod",
    "chown",
    "chgrp",
    "chattr",
    "touch",
    # privilege / escape / process / system control
    "sudo",
    "su",
    "doas",
    "pkexec",
    "chroot",
    "unshare",
    "nsenter",
    "mount",
    "umount",
    "kill",
    "pkill",
    "killall",
    "reboot",
    "shutdown",
    "halt",
    "poweroff",
    "init",
    "systemctl",
    "service",
    "crontab",
    "at",
    "batch",
    # network / exfiltration
    "curl",
    "wget",
    "nc",
    "ncat",
    "netcat",
    "telnet",
    "ssh",
    "scp",
    "sftp",
    "rsync",
    "ftp",
    "tftp",
    "socat",
    # interpreters / shells / arbitrary-code engines & wrappers
    "bash",
    "sh",
    "zsh",
    "ksh",
    "dash",
    "csh",
    "tcsh",
    "fish",
    "python",
    "python2",
    "python3",
    "perl",
    "ruby",
    "node",
    "nodejs",
    "php",
    "lua",
    "rscript",
    "awk",
    "gawk",
    "mawk",
    "eval",
    "exec",
    "command",
    "builtin",
    "source",
    "xargs",
    "nohup",
    "setsid",
    "timeout",
    "nice",
    "watch",
    "stdbuf",
    "ionice",
    # environment / secret disclosure
    "env",
    "printenv",
    "export",
    "set",
    "declare",
    "typeset",
}

# Patterns blocked anywhere (evaluated on the quote-stripped command).
_BASH_DENY_GLOBAL_PATTERNS = [
    (r"`", "backtick command substitution"),
    (r"\$\((?!\()", "$(...) command substitution"),
    (r"[<>]\(", "process substitution"),
    (r":\s*\(\s*\)\s*\{", "fork bomb"),
    (r"(?:^|[;|&])\s*\.\s+\S", "sourcing a script with '.'"),
]

_FIND_WRITE_RE = re.compile(r"-(delete|exec|execdir|fprintf|fprint|fls|ok|okdir)\b")
_SED_INPLACE_RE = re.compile(r"(?:^|\s)-i\b")


def _bash_denylisted(keystrokes):
    """Return a reason string if the command is denied, else None.

    Pragmatic speed-bump only — NOT a security boundary. See _BASH_DENY_*.
    """
    s = keystrokes or ""
    # Strip quoted strings so search terms (grep ">" / grep "rm") don't trip rules.
    unq = re.sub(r"'[^']*'", " ", s)
    unq = re.sub(r'"[^"]*"', " ", unq)

    for pat, why in _BASH_DENY_GLOBAL_PATTERNS:
        if re.search(pat, unq):
            return why

    # Output redirection to a real file (allow /dev/null and fd dups like 2>&1).
    redir = re.sub(r"(?:\d*|&)>{1,2}\s*(?:/dev/null|&\s*\d+|&\s*-)", " ", unq)
    if re.search(r">{1,2}", redir):
        return "output redirection to a file (writes blocked)"

    # Command-position check on each pipeline / list segment.
    for seg in re.split(r"[;\n|&]+", unq):
        words = seg.split()
        idx = 0
        while idx < len(words) and re.match(r"^[A-Za-z_]\w*=", words[idx]):
            idx += 1  # skip leading VAR=val assignments
        if idx >= len(words):
            continue
        prog = Path(words[idx].lstrip("\\")).name
        if prog in _BASH_DENY_COMMANDS:
            return f"command '{prog}' is blocked"
        if prog == "find" and _FIND_WRITE_RE.search(seg):
            return "find write/exec action (-delete/-exec/...) is blocked"
        if prog == "sed" and _SED_INPLACE_RE.search(seg):
            return "sed -i (in-place file edit) is blocked"
    return None


# Default-deny companion: a command passes only if EVERY command-position program
# is in this read-only set. Closes the denylist's gap for unenumerated binaries.
_BASH_ALLOW_COMMANDS = {
    "grep",
    "cat",
    "head",
    "tail",
    "sed",
    "ls",
    "wc",
    "sort",
    "uniq",
    "cut",
    "tr",
    "nl",
    "strings",
    "file",
    "find",
    "diff",
    "echo",
    "printf",
    "cd",
}

# Shell keywords permitted as structural glue (pipelines, for-loops, conditionals).
_BASH_KEYWORD_SKIP = {"if", "elif", "then", "else", "while", "until", "do", "!", "time", "{", "(", "[["}
_BASH_KEYWORD_STANDALONE = {"done", "fi", "esac", "}", ")"}
_BASH_KEYWORD_HEADER = {"for", "case", "select", "function"}

# fd-dups / &-redirections contain '&' which would otherwise phantom-split a segment.
_BASH_FD_DUP_RE = re.compile(r"\d*[<>]&\s*[-\d]+")
_BASH_AMP_REDIR_RE = re.compile(r"&>>?\s*\S+")


def _bash_allowlisted(keystrokes):
    """Return a reason string if any command-position program is NOT read-only,
    else None. Default-deny companion to _bash_denylisted."""
    s = keystrokes or ""
    # Strip quoted strings so search terms (grep ">" / grep "rm") don't trip rules.
    unq = re.sub(r"'[^']*'", " ", s)
    unq = re.sub(r'"[^"]*"', " ", unq)
    # Neutralize &-bearing redirections so they don't split segments.
    unq = _BASH_FD_DUP_RE.sub(" ", unq)
    unq = _BASH_AMP_REDIR_RE.sub(" ", unq)

    for seg in re.split(r"[;\n|&]+", unq):
        words = seg.split()
        idx = 0
        while idx < len(words):
            w = words[idx]
            if re.match(r"^[A-Za-z_]\w*=", w):
                idx += 1  # leading VAR=val assignment
            elif w in _BASH_KEYWORD_HEADER:
                idx = len(words)  # loop/case header: no command to check
            elif w in _BASH_KEYWORD_SKIP or w in _BASH_KEYWORD_STANDALONE:
                idx += 1  # structural keyword; real command (if any) follows
            else:
                break
        if idx >= len(words):
            continue
        prog = Path(words[idx].lstrip("\\")).name
        if prog not in _BASH_ALLOW_COMMANDS:
            return f"command '{prog}' not in read-only allow-list"
    return None
