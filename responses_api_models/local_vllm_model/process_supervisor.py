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
"""POSIX process-group owner, intentionally independent of Gym/vLLM imports.

stdin is a lifetime pipe from Gym, never inherited by the managed command.
Even if Gym is killed before lifespan cleanup, EOF initiates group cleanup.
Do not kill this supervisor before it has had time to clean up its group.
"""

import argparse
import os
import selectors
import signal
import subprocess
import sys
import time


def signal_group(pgid: int, sig: int) -> bool:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    return True


def supervise(command: list[str], shutdown_timeout: float) -> int:
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # No shell, preexec_fn, Python runtime imports, or inherited lifetime pipe.
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(sys.stdin.buffer, selectors.EVENT_READ)
            while not stopping and child.poll() is None:
                if selector.select(timeout=0.05):
                    # EOF means owner death; any explicit message also requests stop.
                    os.read(sys.stdin.fileno(), 1)
                    stopping = True
        return_code = child.poll()
    finally:
        # The group can still have children after its original leader has exited.
        signal_group(child.pid, signal.SIGTERM)
        deadline = time.monotonic() + shutdown_timeout
        while time.monotonic() < deadline:
            child.poll()  # Reap the leader before checking group existence.
            if not signal_group(child.pid, 0):
                break
            time.sleep(0.05)
        signal_group(child.pid, signal.SIGKILL)
        child.wait()
    # A caller-requested shutdown is successful. Preserve early command failures.
    return return_code if return_code is not None and return_code >= 0 else (128 - return_code if return_code else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shutdown-timeout", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.shutdown_timeout <= 0:
        parser.error("a command and positive shutdown timeout are required")
    try:
        return supervise(command, args.shutdown_timeout)
    except OSError as exc:
        print(f"Cannot launch managed executable: {exc}", file=sys.stderr, flush=True)
        return 127


if __name__ == "__main__":
    sys.exit(main())
