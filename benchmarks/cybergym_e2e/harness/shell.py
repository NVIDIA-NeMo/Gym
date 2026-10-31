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
"""The agent's persistent, hardened sandbox shell: one long-lived bwrapped bash driven over the E2B PTY.

This is the boundary the untrusted agent runs behind, and the mechanism that lets it keep background
jobs (fuzzers) alive across `bash` calls. See PersistentShell for the details.
"""

import asyncio
import os
import re
import secrets
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
from e2b.sandbox.commands.command_handle import PtySize  # noqa: E402
from isolation import REMOTE_SECCOMP, SANDBOX_RUNTIME_DIR  # noqa: E402
from template import AGENT_UID  # noqa: E402


WORKDIR = "/work"            # the agent's scratch cwd (uid 1000); repo lives at /src/<repo>
# setpriv cap-drop flags. probe_agent_setpriv() clears this to "" on old-libcap-ng images where setpriv
# can't parse "-all" (uid 1000 + --no-new-privs + stripped setuid + seccomp + bwrap still hold).
AGENT_CAPS_ARGS = "--inh-caps=-all --ambient-caps=-all --bounding-set=-all "
OUT_CAP = 16000              # max chars of one command's output returned to the model
BUF_CAP = 4_000_000          # hard cap on the live PTY buffer (head+tail kept, middle elided)
BUF_ELIDE = "\n...[PTY buffer elided to bound memory]...\n"


class PersistentShell:
    """One long-lived bwrapped bash (uid 1000, no network, seccomp) driven over the E2B PTY, so cwd,
    exported vars AND background jobs persist across commands (a per-command `bash -c` would kill any
    backgrounded fuzzer when the call returned). A per-command timeout sends Ctrl-C, which the PTY
    delivers to the foreground group only; `&` jobs are in their own groups and keep running.

    start() launches /bin/bash as root over the PTY, then `exec`s it into the hardened bwrap shell.
    --new-session is not used (it would detach the shell from the PTY and break SIGINT); the VM +
    --unshare-net + seccomp + uid 1000 remain the boundary."""

    def __init__(self, sb):
        self.sb = sb
        self.pid = None
        self.buf = ""

    def _on_data(self, data):
        if isinstance(data, bytes):
            data = data.decode("utf-8", "replace")
        self.buf += data.replace("\r", "")
        # Bound one noisy command: keep the head and (larger) tail; exec() resets the buffer and reads
        # from offset 0 each command, so collapsing the middle here never breaks marker/output slicing.
        if len(self.buf) > BUF_CAP:
            self.buf = self.buf[:BUF_CAP // 3] + BUF_ELIDE + self.buf[-(BUF_CAP * 2 // 3):]

    async def start(self, pty_timeout):
        # Capture the VM's ORIGINAL (root) net namespace so we can later prove the agent shell is in a
        # different one. sb.commands.run executes as root in the VM's main namespace.
        vm_netns = (await self.sb.commands.run(
            "readlink /proc/self/ns/net", user="root", timeout=30)).stdout.strip()
        h = await self.sb.pty.create(size=PtySize(rows=50, cols=220), on_data=self._on_data,
                                     user="root", timeout=pty_timeout)
        self.pid = h.pid
        # TEST HOOK: CYBERGYM_TEST_BREAK_ISOLATION=1 points the seccomp fd at a nonexistent path so the
        # `exec` fails, leaving the root bash alive (the exact fail-open the identity check must catch).
        # Only for deliberately exercising the reject path; never set in a real run.
        seccomp_src = REMOTE_SECCOMP
        if os.environ.get("CYBERGYM_TEST_BREAK_ISOLATION") == "1":
            seccomp_src = "/nonexistent/__cybergym_break__"
            print("[run] TEST: CYBERGYM_TEST_BREAK_ISOLATION set -- forcing bwrap exec to fail")
        launch = (
            f"exec 10<{seccomp_src} "
            "bwrap --unshare-net --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try "
            "--die-with-parent --seccomp 10 "
            "--setenv PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
            f"--setenv HOME {WORKDIR} --setenv LANG C.UTF-8 "
            f"--bind / / --dev /dev --proc /proc --tmpfs {SANDBOX_RUNTIME_DIR} -- "
            f"setpriv --reuid={AGENT_UID} --regid={AGENT_UID} --clear-groups "
            f"{AGENT_CAPS_ARGS}--no-new-privs -- bash -i\n")
        await self.sb.pty.send_stdin(self.pid, launch.encode())
        # Quiet the PTY (no input echo, empty prompt) so output capture is just program output, then
        # source the build env and start in /work -- ONCE, since state now persists.
        await self.sb.pty.send_stdin(
            self.pid, (f"stty -echo 2>/dev/null; PS1=''; PROMPT_COMMAND=''; "
                       f". /etc/cybergym.env 2>/dev/null; cd {WORKDIR}\n").encode())
        out, rc = await self.exec("echo __shell_ready__", 60)
        if "__shell_ready__" not in out:
            raise RuntimeError(f"persistent shell did not come up: rc={rc} out={out[:300]!r}")
        # FAIL CLOSED: prove this shell is the hardened identity, not a root shell left behind by a
        # failed `exec ... bwrap ...` (which would still answer __shell_ready__). A failed network
        # request is NOT sufficient -- the VM firewall drops egress for root too -- so check privilege
        # and namespace directly. Check CapEff AND CapPrm AND NoNewPrivs, not just CapEff: on the
        # cap-drop-fallback image (AGENT_CAPS_ARGS=""), the bounding set is NOT dropped; what makes that
        # safe is no_new_privs (execve then ignores setuid bits + file caps). CapPrm must be empty too
        # -- a nonzero permitted set could be raised back into effective. netns must differ from the
        # VM's (proves --unshare-net took). Any mismatch means the wrapper did not take; caller kills VM.
        probe, _ = await self.exec(
            "printf 'uid=%s eff=%s prm=%s nnp=%s ns=%s\\n' \"$(id -u)\" "
            "\"$(awk '/^CapEff:/{print $2}' /proc/self/status)\" "
            "\"$(awk '/^CapPrm:/{print $2}' /proc/self/status)\" "
            "\"$(awk '/^NoNewPrivs:/{print $2}' /proc/self/status)\" "
            "\"$(readlink /proc/self/ns/net)\"", 60)
        m = re.search(r"uid=(\d+) eff=(\S+) prm=(\S+) nnp=(\S+) ns=(\S+)", probe)
        if not m:
            raise RuntimeError(f"could not verify agent shell identity: {probe[:300]!r}")
        uid, eff, prm, nnp, netns = m.groups()
        if (uid != str(AGENT_UID) or eff.lstrip("0") != "" or prm.lstrip("0") != ""
                or nnp != "1" or netns == vm_netns):
            raise RuntimeError(
                f"agent shell NOT isolated (uid={uid} capEff={eff} capPrm={prm} noNewPrivs={nnp} "
                f"netns={netns} vm_netns={vm_netns}); bwrap/setpriv wrapper failed to exec -- killing VM")

    async def exec(self, command, timeout):
        """Send one command; return (output, exit_code). On timeout, Ctrl-C the foreground job."""
        nonce = secrets.token_hex(4)
        end = f"__E{nonce}__"
        pat = re.compile(re.escape(end) + r"(-?\d+)__")
        # Reset the buffer so it never accumulates across commands (prior output is already returned and
        # never read again). No await between here and send_stdin, so _on_data can't interleave.
        self.buf = ""
        start = 0
        marker = f"printf '\\n{end}%s__\\n' \"$?\"\n"
        await self.sb.pty.send_stdin(self.pid, (command.rstrip("\n") + "\n" + marker).encode())
        t0, timed_out = time.time(), False
        while True:
            m = pat.search(self.buf, start)
            if m:
                return self.buf[start:m.start()].strip("\n"), int(m.group(1))
            if not timed_out and time.time() - t0 > timeout:
                timed_out = True
                # SIGINT the foreground group, then ask for the marker again WITHOUT re-running the
                # command (so side effects don't repeat); $? is now the interrupted command's status.
                await self.sb.pty.send_stdin(self.pid, b"\x03")
                await asyncio.sleep(0.3)
                await self.sb.pty.send_stdin(self.pid, marker.encode())
                t0 = time.time()
            if timed_out and time.time() - t0 > 20:
                body = self.buf[start:].strip("\n")
                return (body + f"\n[timed out after {timeout}s; sent Ctrl-C. Background jobs (started "
                               "with &) keep running; foreground command was interrupted.]"), None
            await asyncio.sleep(0.1)


def truncate(text, cap=OUT_CAP):
    if len(text) <= cap:
        return text
    head, tail = cap * 2 // 3, cap // 3
    return f"{text[:head]}\n\n...[{len(text) - cap} chars elided]...\n\n{text[-tail:]}"


async def run_agent_command(shell, command, timeout):
    """Run one agent command in the persistent shell; return a text block (exit code + output)."""
    try:
        out, rc = await shell.exec(command, timeout)
    except Exception as e:
        out, rc = f"[shell error: {type(e).__name__}: {e}]", None
    body = truncate(out)
    return f"exit code: {rc}\n--- output ---\n{body}" if body.strip() else f"exit code: {rc}\n(no output)"


async def probe_agent_setpriv(sb):
    """Detect whether setpriv can drop the '-all' cap sets on this image; fall back if not.

    Old libcap-ng exits 127 with "setpriv: libcap-ng is too old for \"all\" caps" before running the
    command, which would make EVERY agent command fail silently. On failure we drop only the cap flags
    (reuid/regid/clear-groups/no-new-privs still apply); setuid bits are already stripped and seccomp +
    bwrap remain, so there is no path for the uid-1000 agent to regain capabilities."""
    global AGENT_CAPS_ARGS
    probe = await sb.commands.run(
        f"setpriv --reuid={AGENT_UID} --regid={AGENT_UID} --clear-groups "
        "--inh-caps=-all --ambient-caps=-all --bounding-set=-all --no-new-privs -- true "
        "2>&1 || echo __SETPRIV_CAPS_FAIL__", user="root", timeout=30)
    if "__SETPRIV_CAPS_FAIL__" in probe.stdout or "libcap-ng" in probe.stdout:
        AGENT_CAPS_ARGS = ""
        print(f"[run] WARN setpriv cap-drop unsupported ({probe.stdout.strip().splitlines()[0][:80]}); "
              "dropping to uid/no-new-privs only (setuid bits stripped, seccomp+bwrap active)")
    else:
        AGENT_CAPS_ARGS = "--inh-caps=-all --ambient-caps=-all --bounding-set=-all "
        print("[run] setpriv cap-drop supported")
