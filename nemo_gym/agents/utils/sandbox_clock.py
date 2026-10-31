# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep a sandbox's wall clock continuous across a restore that keeps its processes.

A provider whose snapshot keeps a sandbox's memory and processes (AgentENV) resumes the guest with uptime running on
from the snapshot, but sets the guest's wall clock to the host's: after a restore on a later day, ``date``, the
modification times of new files and a harness's own idea of today jump by however long the sandbox was stopped.
Nothing inside the sandbox can stop the resume from setting it, so a restore sets it back instead:
``rewind_clock`` moves the wall clock to where it would be had the sandbox never stopped, from the wall clock and
uptime a pause recorded before the snapshot. ``disable_clock_sync`` stops the time daemons some images run, which
would otherwise pull the clock forward again.

The wall clock then lags real time by the time the sandbox spent stopped. Requests the sandbox signs with the time
(cloud APIs, short-lived tokens) can notice that; the sandbox's own processes cannot.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from shlex import quote

from nemo_gym.sandbox.api import AsyncSandbox


# Services that keep a guest's clock in step with the network or the host; stopped and masked where they exist.
CLOCK_SYNC_SERVICES = ("systemd-timesyncd", "chronyd", "chrony", "ntpd", "ntp", "openntpd")

# Prints one line per service it stopped. Masks systemd units so that systemd does not start them again; without
# systemd, ends the daemons themselves (/proc comm names are cut at 15 characters).
_DISABLE_CLOCK_SYNC = r"""
stopped=()
if command -v systemctl >/dev/null 2>&1; then
    for unit in "$@"; do
        systemctl cat "$unit.service" >/dev/null 2>&1 || continue
        systemctl stop "$unit.service" >/dev/null 2>&1
        systemctl mask "$unit.service" >/dev/null 2>&1 && stopped+=("$unit")
    done
fi
for dir in /proc/[0-9]*; do
    read -r comm <"$dir/comm" 2>/dev/null || continue
    for name in "$@"; do
        if [[ $comm == "${name:0:15}" ]]; then
            kill "${dir#/proc/}" 2>/dev/null && stopped+=("$comm")
        fi
    done
done
printf '%s\n' "${stopped[@]}"
"""

# Arguments: pause wall clock and uptime in milliseconds, the shift (ms) below which nothing changes, the command
# that sets the clock, whether to read the clock back, then the directories whose newer files to reset. Prints one
# JSON object. Integer arithmetic only, so that images without awk or Python can run it: /proc/uptime always has
# two decimals, and EPOCHREALTIME (bash 5) or printf's %(%s)T (bash 4.2) gives the wall clock.
_REWIND_CLOCK = r"""
pause_wall_ms=$1 pause_uptime_ms=$2 min_shift_ms=$3 set_clock=$4 verify=$5
shift 5
wall_ms() {
    if [[ -n ${EPOCHREALTIME-} ]]; then
        local now=${EPOCHREALTIME/[.,]/}
        echo $((10#$now / 1000))
    else
        local now
        printf -v now '%(%s)T' -1
        echo $((now * 1000))
    fi
}
read -r uptime _ </proc/uptime 2>/dev/null || { echo '{"applied": false, "reason": "no_uptime"}'; exit 0; }
uptime_ms=$((10#${uptime/./} * 10))
target_ms=$((pause_wall_ms + uptime_ms - pause_uptime_ms))
shift_ms=$(($(wall_ms) - target_ms))
if ((shift_ms <= min_shift_ms)); then
    echo "{\"applied\": false, \"shift_ms\": $shift_ms, \"reason\": \"continuous\"}"
    exit 0
fi
target=$((target_ms / 1000)).$(printf '%03d' $((target_ms % 1000)))
if ! $set_clock "@$target" >/dev/null 2>&1; then
    echo "{\"applied\": false, \"shift_ms\": $shift_ms, \"reason\": \"set_failed\"}"
    exit 0
fi
if [[ $verify == 1 ]]; then
    off_ms=$(($(wall_ms) - target_ms))
    if ((off_ms > 2000 || off_ms < -2000)); then
        echo "{\"applied\": false, \"shift_ms\": $shift_ms, \"reason\": \"readback_mismatch\"}"
        exit 0
    fi
fi
touched=0
for dir in "$@"; do
    [[ -d $dir ]] || continue
    count=$(find "$dir" -xdev -newermt "@$target" -print -exec touch -h -d "@$target" {} + 2>/dev/null | wc -l)
    touched=$((touched + count))
done
echo "{\"applied\": true, \"shift_ms\": $shift_ms, \"touched\": $touched, \"target\": \"$target\"}"
"""


@dataclass(frozen=True)
class ClockRewind:
    """What ``rewind_clock`` did.

    ``applied``: the wall clock was set back by ``shift_s`` seconds and ``touched`` files newer than the new time had
    their modification times reset to it. Otherwise ``reason`` says why not: ``continuous`` (the clock had moved on
    less than the threshold, as without a restore), ``set_failed`` (not root, or no ``date -s``),
    ``readback_mismatch``, ``no_uptime`` or ``check_failed``.
    """

    applied: bool
    shift_s: float | None = None
    touched: int = 0
    reason: str | None = None


async def disable_clock_sync(sandbox: AsyncSandbox, *, timeout: float) -> tuple[str, ...]:
    """Stop and mask the time daemons running in ``sandbox``, so that they do not move its clock again.

    Best effort, as root: services that are absent are skipped, and a failure to stop one is not an error. Returns the
    services it stopped.
    """
    result = await sandbox.exec(disable_clock_sync_command(), timeout_s=timeout, user="root")
    return tuple(line for line in (result.stdout or "").splitlines() if line.strip())


def disable_clock_sync_command(services: Sequence[str] = CLOCK_SYNC_SERVICES) -> str:
    """The one command ``disable_clock_sync`` runs, for ``services``."""
    return f"bash -c {quote(_DISABLE_CLOCK_SYNC)} clock {' '.join(quote(name) for name in services)}"


def rewind_clock_command(
    *,
    pause_wall: float,
    pause_uptime: float,
    touch_dirs: Sequence[str],
    min_shift_s: float = 5.0,
    set_clock: str = "date -s",
    verify: bool = True,
) -> str:
    """The one command ``rewind_clock`` runs; ``set_clock`` and ``verify`` exist so tests can run it unprivileged."""
    arguments = [
        str(round(pause_wall * 1000)),
        str(round(pause_uptime * 1000)),
        str(round(min_shift_s * 1000)),
        set_clock,
        "1" if verify else "0",
        *touch_dirs,
    ]
    return f"bash -c {quote(_REWIND_CLOCK)} rewind {' '.join(quote(argument) for argument in arguments)}"


async def rewind_clock(
    sandbox: AsyncSandbox,
    *,
    pause_wall: float,
    pause_uptime: float,
    touch_dirs: Sequence[str],
    timeout: float,
    min_shift_s: float = 5.0,
) -> ClockRewind:
    """Set a restored sandbox's wall clock back to where it would be had the sandbox never stopped.

    The new time is ``pause_wall + (uptime now - pause_uptime)``, from a pause recorded before the snapshot: uptime
    does not count the time the sandbox spent stopped. A clock that moved on less than ``min_shift_s`` past it is
    left alone. Files under ``touch_dirs`` written after the restore and before the rewind, which would otherwise
    date from the future, get the new time as their modification time. One command, as root; never raises for a
    clock it cannot set.
    """
    command = rewind_clock_command(
        pause_wall=pause_wall, pause_uptime=pause_uptime, touch_dirs=touch_dirs, min_shift_s=min_shift_s
    )
    result = await sandbox.exec(command, timeout_s=timeout, user="root")
    return parse_clock_rewind(result.stdout or "", error=result.error_type or None)


def parse_clock_rewind(output: str, *, error: str | None = None) -> ClockRewind:
    """Read the JSON line the rewind command prints."""
    if error is not None:
        return ClockRewind(applied=False, reason="check_failed")
    try:
        report = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return ClockRewind(applied=False, reason="check_failed")
    shift_ms = report.get("shift_ms")
    return ClockRewind(
        applied=bool(report.get("applied")),
        shift_s=None if shift_ms is None else shift_ms / 1000,
        touched=int(report.get("touched", 0)),
        reason=report.get("reason"),
    )
