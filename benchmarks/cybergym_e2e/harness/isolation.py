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
"""Layered in-sandbox hardening primitives for CyberGym, applied inside an E2B microVM.

CyberGym runs untrusted agent code and untrusted patches that get compiled and executed, so a
microVM-class backend (real kernel boundary) is required and the following layers are applied as
root BEFORE the agent starts, then verified fail-closed:

  firewall   iptables OUTPUT default DROP (loopback, envd replies, established, --allow IPs:443),
             ip6tables DROP, blackhole route to the cloud-metadata address.
  lockdown   guest-kernel attack surface cut: no user namespaces, no unprivileged BPF, perf locked,
             kptr/dmesg restricted, kernel modules disabled (one-way), setuid bits stripped.
  seccomp    classic-BPF filter (for `bwrap --seccomp`): kill non-x86_64, EPERM the x32 ABI and a
             deny list of dangerous syscalls.
  (bwrap + uid-1000 drop live in run.py / validate.py, which build their own launch commands.)

E2B's in-VM agent (envd) listens on tcp/49983; its replies must pass the firewall or exec/file
transfer hangs. The agent's own commands additionally run in `bwrap --unshare-net`, so these
root-level rules only govern setup/validation traffic.

No endpoint, key, or deployment-specific address is hardcoded here. SANDBOX_RUNTIME_DIR below is the
E2B runtime directory to hide behind an empty tmpfs; adjust it for your E2B deployment if it differs.
"""

import struct


# Egress deny set for E2B's `network=` rules (deny_out). Standard non-routable ranges + default.
DENY = ["0.0.0.0/0", "169.254.0.0/16", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10"]

# The sandbox runtime directory, hidden behind an empty tmpfs inside bwrap so the agent can't read
# the sandbox's own runtime files. Depends on the E2B build; override for your deployment.
SANDBOX_RUNTIME_DIR = "/agentenv"

REMOTE_SECCOMP = "/etc/agent-seccomp.bpf"


def firewall_script(allow_ips):
    """Root script: iptables OUTPUT default-DROP, allowing loopback, envd, established, and allow_ips."""
    allow = "\n".join(f"iptables -A OUTPUT -d {ip} -p tcp --dport 443 -j ACCEPT" for ip in allow_ips)
    # envd (E2B's in-VM agent) listens on :49983; its replies must pass or exec/file calls hang.
    return f"""set -e
iptables --version
iptables -F OUTPUT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -p tcp --sport 49983 -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT 2>/dev/null \\
  || echo "WARN: conntrack match unavailable"
{allow}
iptables -P OUTPUT DROP
ip6tables -P OUTPUT DROP 2>/dev/null || true
ip6tables -A OUTPUT -o lo -j ACCEPT 2>/dev/null || true
ip route add blackhole 169.254.169.254/32 2>/dev/null || true
iptables -S OUTPUT
"""


# Applied AFTER the firewall (modules_disabled=1 would stop nftables from loading its modules).
LOCKDOWN_SCRIPT = """
sysctl -w user.max_user_namespaces=0
sysctl -w kernel.unprivileged_bpf_disabled=1
sysctl -w kernel.perf_event_paranoid=3
sysctl -w kernel.kptr_restrict=2
sysctl -w kernel.dmesg_restrict=1
sysctl -w vm.mmap_rnd_bits=28 || echo "WARN: could not set vm.mmap_rnd_bits=28 (ASan may misbehave)"
find / -xdev -perm -4000 -type f -exec chmod u-s {} + 2>/dev/null || true
sysctl -w kernel.modules_disabled=1
echo lockdown-applied
"""

# Syscalls the agent gets EPERM for (x86_64 numbers). The VM is the real boundary; this trims the
# guest-kernel attack surface. ptrace is blocked here (run_poc.sh sets detect_leaks=0, so no LSan);
# note that also disables gdb/strace for the agent -- validate.py allows ptrace for the test suite.
SECCOMP_DENY = {
    "ptrace": 101, "process_vm_readv": 310, "process_vm_writev": 311, "kcmp": 312,
    "mount": 165, "umount2": 166, "pivot_root": 155, "unshare": 272, "setns": 308,
    "open_tree": 428, "move_mount": 429, "fsopen": 430, "fsconfig": 431, "fsmount": 432,
    "fspick": 433, "mount_setattr": 442, "open_by_handle_at": 304,
    "bpf": 321, "perf_event_open": 298, "userfaultfd": 323,
    "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
    "add_key": 248, "request_key": 249, "keyctl": 250,
    "init_module": 175, "finit_module": 313, "delete_module": 176,
    "kexec_load": 246, "kexec_file_load": 320, "reboot": 169,
    "swapon": 167, "swapoff": 168, "acct": 163, "quotactl": 179,
}


def seccomp_bpf(deny=SECCOMP_DENY):
    """Classic-BPF seccomp filter for bwrap --seccomp: kill non-x86_64, EPERM x32 and the deny set."""
    ld, jeq, jge, ret = 0x20, 0x15, 0x35, 0x06
    allow, eperm, kill = 0x7FFF0000, 0x00050000 | 1, 0x80000000
    nrs = sorted(deny.values())
    errno_at = 5 + len(nrs) + 1  # index of the final "ret EPERM"
    prog = [(ld, 0, 0, 4),                    # 0: A = seccomp_data.arch
            (jeq, 1, 0, 0xC000003E),          # 1: AUDIT_ARCH_X86_64 -> skip the kill
            (ret, 0, 0, kill),                # 2
            (ld, 0, 0, 0),                    # 3: A = seccomp_data.nr
            (jge, errno_at - 5, 0, 0x40000000)]  # 4: x32 ABI -> EPERM
    for i, nr in enumerate(nrs):
        prog.append((jeq, errno_at - (5 + i) - 1, 0, nr))
    prog += [(ret, 0, 0, allow), (ret, 0, 0, eperm)]
    return b"".join(struct.pack("<HBBI", *ins) for ins in prog)


async def sh(sb, name, script, timeout=60):
    """Run a root script in the sandbox and print its combined output under a [name] tag."""
    r = await sb.commands.run(script, user="root", timeout=timeout)
    print(f"[{name}] {(r.stdout + r.stderr).strip()}")
    return r


# Required sysctls checked explicitly after lockdown. mmap_rnd_bits is ASan hygiene, not isolation,
# so it only warns (some kernels cap it); the rest are security-critical and fail closed.
REQUIRED_SYSCTLS = {
    "user/max_user_namespaces": "0", "kernel/unprivileged_bpf_disabled": "1",
    "kernel/kptr_restrict": "2", "kernel/dmesg_restrict": "1",
}


async def verify_isolation(sb, name):
    """Fail closed: confirm IPv4 default-DROP egress, no off-link IPv6 egress (DROP policy or no usable
    IPv6 route), and the required sysctls are actually in force -- firewall_script/LOCKDOWN_SCRIPT
    tolerate individual failures and print success regardless. Raises RuntimeError if anything
    security-critical is missing; prints the real state (does not claim DROP when the policy is ACCEPT)."""
    v4 = (await sb.commands.run("iptables -S OUTPUT", user="root", timeout=30)).stdout
    if "-P OUTPUT DROP" not in v4:
        raise RuntimeError(f"IPv4 egress not default-DROP:\n{v4}")
    # IPv6: the real property is "no off-link IPv6 egress". ip6tables -P OUTPUT DROP is the direct way,
    # but it can silently fail to apply on these VMs. When it hasn't, require that no USABLE IPv6 route
    # exists: link-local (fe80) and multicast (ff00) routes are confined to the local link, but a
    # default/global route can carry traffic off-host, so that must be DROP-policed. ip6tables itself
    # can be unusable on some base-image kernels (no IPv6 netfilter), so read it tolerantly.
    v6 = (await sb.commands.run("ip6tables -S OUTPUT 2>/dev/null || echo __IP6_UNAVAIL__",
                                user="root", timeout=30)).stdout
    v6_avail = "__IP6_UNAVAIL__" not in v6
    v6_drop = v6_avail and "-P OUTPUT DROP" in v6
    if v6_drop:
        v6_state = "OUTPUT DROP"
    else:
        routes = (await sb.commands.run("ip -6 route show 2>/dev/null", user="root", timeout=30)).stdout
        usable = [r for r in routes.splitlines()
                  if r.strip() and not r.startswith(("fe80", "ff00", "unreachable", "::1"))]
        if usable:
            why = "ip6tables unavailable" if not v6_avail else "IPv6 egress not default-DROP"
            raise RuntimeError(f"{why} and a usable IPv6 route exists:\n{routes}")
        v6_state = (("ip6tables unavailable" if not v6_avail else "ACCEPT policy")
                    + "; no usable (off-link) IPv6 route")
    bad = []
    for key, want in REQUIRED_SYSCTLS.items():
        cur = (await sb.commands.run(f"cat /proc/sys/{key} 2>/dev/null || echo MISSING",
                                     user="root", timeout=30)).stdout.strip()
        if cur != want:
            bad.append(f"{key}={cur} (want {want})")
    if bad:
        raise RuntimeError(f"lockdown sysctls not in force: {', '.join(bad)}")
    rnd = (await sb.commands.run("cat /proc/sys/vm/mmap_rnd_bits 2>/dev/null || echo ?",
                                 user="root", timeout=30)).stdout.strip()
    print(f"[{name}] isolation verified: v4 OUTPUT DROP, v6 {v6_state}, sysctls OK"
          f"{'' if rnd == '28' else f' (WARN mmap_rnd_bits={rnd}, want 28; ASan may misbehave)'}")
