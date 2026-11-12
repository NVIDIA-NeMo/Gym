# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the opt-in tmux bootstrap that hides unrelated service volumes."""

import shlex
from pathlib import PurePosixPath


_BOOTSTRAP = r"""
set -euo pipefail
test "$(id -u)" = 0
for tool in tmux unshare mount mountpoint; do
    command -v "$tool" >/dev/null || { echo "Private terminal requires $tool" >&2; exit 1; }
done
if tmux list-sessions >/dev/null 2>&1; then
    echo 'Private terminal requires a fresh sandbox with no existing tmux server' >&2
    exit 1
fi
probe_dir=$(mktemp -d /tmp/gym-private-terminal.XXXXXX)
chmod 777 "$probe_dir"
probe_pid=
cleanup() {
    if [[ -n "$probe_pid" ]]; then kill "$probe_pid" 2>/dev/null || true; fi
    rm -rf "$probe_dir"
}
trap cleanup EXIT
unshare --user --mount --propagation private bash -c '
    set -eu
    probe_dir=$1
    shift
    echo $$ > "$probe_dir/ready"
    while [[ ! -f "$probe_dir/mapped" ]]; do sleep 0.05; done
    for hidden_mount in "$@"; do
        mountpoint -q -- "$hidden_mount"
    done
    for hidden_mount in "$@"; do
        mount -t tmpfs -o size=1m tmpfs "$hidden_mount"
    done
    TERM=xterm-256color tmux new-session -d -s gym-internal-mount-bootstrap "sleep 120"
' bash "$probe_dir" "$@" &
probe_pid=$!
for attempt in {1..100}; do
    if [[ -f "$probe_dir/ready" ]]; then break; fi
    kill -0 "$probe_pid"
    sleep 0.05
done
test -f "$probe_dir/ready"
printf '0 0 4294967295\n' > "/proc/$probe_pid/uid_map"
printf '0 0 4294967295\n' > "/proc/$probe_pid/gid_map"
touch "$probe_dir/mapped"
wait "$probe_pid"
probe_pid=
"""


def private_terminal_bootstrap(hidden_mounts: list[str]) -> str:
    """Hide mount points only in tmux's namespace, preserving task files and UID/GID identities.

    Requires Linux, root, an installed tmux, and permission to create user/mount
    namespaces with identity mappings. This changes the terminal's capability
    context as well as mount visibility; it is not a security boundary. The
    caller must remove the bootstrap session once Harbor creates its own session.
    """
    for path in hidden_mounts:
        parsed = PurePosixPath(path)
        if (
            not parsed.is_absolute()
            or parsed == PurePosixPath("/")
            or path.startswith("//")
            or ".." in parsed.parts
            or "\0" in path
            or str(parsed) != path
        ):
            raise ValueError(f"terminal_hidden_mounts requires normalized absolute mount paths other than /: {path!r}")
    return shlex.join(["timeout", "--kill-after=2", "20", "bash", "-c", _BOOTSTRAP, "bash", *hidden_mounts])
