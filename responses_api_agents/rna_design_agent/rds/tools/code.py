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
"""Let a model write and run its own design code. It returns sequences only, so scoring stays ours.

Scripts run in the `$RDS_SANDBOX_IMAGE` image with no network and only their own directory mounted, under docker
or bubblewrap. Each result records which sandbox ran it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..registry import tool
from ..spec import Cost, EnvironmentFault, field


PREAMBLE = '''"""Written by the agent. Prints {"sequences": [...]} and nothing else."""
'''


def _sandbox_command(script: Path, argument: str, scratch: Path) -> tuple[list[str], str]:
    """How to run the script, and which isolation mode that gives.

    Only a container keeps a script from the rest of the filesystem, so plain subprocess is opt-in.
    """
    import shutil

    mode = os.environ.get("RDS_SANDBOX", "auto")
    if mode == "auto":
        mode = "docker" if shutil.which("docker") else "bwrap"

    if mode in ("bwrap", "enroot") and shutil.which("bwrap"):
        # For hosts without docker: the same image, unpacked, as a read-only root.
        # `--unshare-all` also cuts the network, which enroot alone does not.
        rootfs = os.environ.get("RDS_SANDBOX_ROOTFS", "")
        if not rootfs or not Path(rootfs).is_dir():
            raise EnvironmentFault(
                "RDS_SANDBOX=bwrap needs RDS_SANDBOX_ROOTFS pointing at an "
                f"unpacked image directory; {rootfs or '<unset>'} is not one."
            )
        command = [
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--ro-bind",
            rootfs,
            "/",  # the image, and nothing of this host
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",  # scratch that dies with the call
            "--ro-bind",
            str(script.parent),
            "/work",
            "--chdir",
            "/work",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "ARNIEFILE",
            "/opt/arniefile",
            "--setenv",
            "PATH",
            "/usr/local/bin:/usr/bin:/bin",
            "python3",
            "-I",
            f"/work/{script.name}",
        ]
        if argument:
            command.append(argument)
        return command, "bwrap"

    if mode == "docker" and shutil.which("docker"):
        image = os.environ.get("RDS_SANDBOX_IMAGE", "rds-sandbox:1")
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",  # no gateway, no package index
            "--read-only",  # the image is not writable
            "--tmpfs",
            "/tmp:exec",  # scratch that dies with the run
            "--memory",
            os.environ.get("RDS_SANDBOX_MEMORY", "4g"),
            "--pids-limit",
            "256",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{script.parent}:/work:ro",  # this script, nothing else
            "-w",
            "/work",
            "--entrypoint",
            "python3",
            image,
            "-I",
            f"/work/{script.name}",
        ]
        if argument:
            command.append(argument)
        return command, "docker"
    if mode == "subprocess":
        # Unisolated, so only ever chosen by name, never as a fallback.
        command = [sys.executable, "-I", str(script)]
        if argument:
            command.append(argument)
        return command, "subprocess"
    raise EnvironmentFault(
        f"no sandbox for RDS_SANDBOX={mode}: docker and bwrap were not found, and agent-written code does not "
        "run unisolated by default. Install one, or set RDS_SANDBOX=subprocess to run it as this user with "
        "network access."
    )


def _designs_in(text: str) -> list[str]:
    """The sequences in the last complete JSON line, or none."""
    for line in reversed((text or "").strip().splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        found = [str(s).strip().upper().replace("T", "U") for s in (payload.get("sequences") or []) if str(s).strip()]
        if found and not ({b for s in found for b in s} - set("ACGU")):
            return found
    return []


def _result(tool: str, purpose: str, sequences: list[str], sandbox: str, note: str = "") -> dict:
    return {
        "tool": tool,
        "purpose": purpose,
        "n": len(sequences),
        "sandbox": sandbox,
        "note": note,
        "results": [{"sequence": s} for s in sequences],
    }


def _execute(script: Path, timeout_s: int, argument: str = "", tool: str = "code.run", purpose: str = "") -> dict:
    """Run one script under the strongest isolation available."""
    # Pass no variables that point at the campaign, the API key or the weights.
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "ARNIEFILE", "PYTHONHASHSEED")}
    env["PYTHONPATH"] = ""
    with tempfile.TemporaryDirectory(prefix="agent_tmp_") as scratch:
        command, sandbox = _sandbox_command(script, argument, Path(scratch))
        try:
            result = subprocess.run(
                command, cwd=str(script.parent), env=env, capture_output=True, text=True, timeout=timeout_s
            )
        except subprocess.TimeoutExpired as expired:
            # Keep whatever designs the script printed before it was killed.
            printed = expired.stdout or b""
            text = printed.decode() if isinstance(printed, bytes) else printed
            salvaged = _designs_in(text)
            if salvaged:
                return _result(
                    tool, purpose, salvaged, sandbox, f"killed after {timeout_s}s; read what it had printed by then"
                )
            raise RuntimeError(
                f"the script was killed after {timeout_s}s and printed nothing "
                f"usable. Raise timeout_s, cut the search, or print "
                f'{{"sequences": [...]}} as you go — the last complete line is '
                f"read even if the script is killed"
            ) from None
    if result.returncode != 0:
        raise RuntimeError(f"the script exited {result.returncode}\n--- stderr ---\n{result.stderr[-1200:]}")
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise ValueError(
            "the script must print one JSON object as its last line, like "
            '{"sequences": ["GGGCAAAGCC"]}; it printed:\n'
            f"{result.stdout[-400:]}"
        ) from None
    sequences = [str(s).strip().upper().replace("T", "U") for s in (payload.get("sequences") or []) if str(s).strip()]
    bad = {b for s in sequences for b in s} - set("ACGU")
    if bad:
        raise ValueError(f"designs contain non-RNA letters: {sorted(bad)}")
    if not sequences:
        raise ValueError(
            'the script printed {"sequences": []}. A filter that rejects '
            "everything spends the round and returns nothing: return the best "
            "few you have and let them be scored here, or loosen the filter."
        )
    return {
        "tool": tool,
        "purpose": purpose,
        "n": len(sequences),
        "sandbox": sandbox,
        "results": [{"sequence": s} for s in sequences],
        "stderr_tail": result.stderr[-200:] if result.stderr else "",
        "_streams": result,
    }


def _write_target(work: Path) -> None:
    """Write target.json beside the script, so a long structure is read rather than retyped."""
    from ..targets import TARGETS

    target = TARGETS.get(os.environ.get("RDS_TARGET", ""))
    if target is None:
        return
    payload = {
        "name": target.name,
        "length": target.length,
        "structure": getattr(target, "target_structure", "") or "",
        "constraints": {
            "fixed_prefix": getattr(target, "fixed_prefix", "") or "",
            "conserved": getattr(target, "conserved", None) or {},
        },
    }
    (work / "target.json").write_text(json.dumps(payload, indent=1))


def _toolbox() -> Path:
    """Where a saved tool lives: a directory of Python files that spans the targets of one run."""
    import os

    box = Path(os.environ.get("RDS_TOOLBOX") or Path(tempfile.gettempdir()) / "rds_toolbox")
    box.mkdir(parents=True, exist_ok=True)
    return box


@dataclass
class RunCodeIn:
    code: str = field(
        "Python that prints one JSON object to stdout: "
        '{"sequences": ["GGGC...", ...]}. The standard library is available, '
        "and so are numpy and the folding packages already installed. There is "
        "no network and no access to the campaign state, so anything the script "
        "needs — the target structure, sequences to start from — must be "
        "written into the code itself."
    )
    purpose: str = field(
        "What this script does, in a sentence. Recorded with the designs it "
        "produces, so a method can be read back later.",
        default="",
    )
    save_as: str = field(
        "Save this script under a name so you can call it again with "
        "code.call, on this target or the next one. Leave empty to run it "
        "once and discard it.",
        default="",
    )
    timeout_s: int = field(
        "Seconds before the script is killed. Raise it for "
        "a search; three scripts in a row died at the old "
        "default of 120 with nothing to show for the round",
        default=300,
        operator=True,
    )


@tool(
    name="code.run",
    summary=(
        "Run Python you write, and take its designs. Use it for a method the "
        "tools do not offer — a local search around a candidate, a filter of "
        "your own, a combination of two scores. The script prints "
        '{"sequences": [...]} and that is all this tool reads: it cannot score '
        "anything, reach the network, or see the campaign state. Designs it "
        "returns enter the population like any others and are evaluated the "
        "same way.\n"
        'Print {"sequences": [...]} as you go, not only at the end: a '
        "script killed at `timeout_s` keeps its last complete line, and three "
        "searches in a row have otherwise spent a round and returned nothing.\n"
        "What is in there: python3 with numpy, scipy and arnie, and EternaFold "
        "as the folding engine (`from arnie.mfe import mfe; mfe(seq, "
        "package='eternafold')`). ViennaRNA is not installed and `import RNA` "
        "fails; there is no network, so nothing can be installed either.\n"
        "The script takes no arguments from you: write what it needs into the "
        "code itself (`code.call` is the one that passes an `argument`).\n"
        "A file `target.json` sits beside the script — read it rather than "
        'retyping a structure into your code: {"name", "length", '
        '"structure", "constraints"}. Anything else the script needs '
        "must be written into it."
    ),
    cost=Cost.CHEAP,
    model=RunCodeIn,
)
def code_run(inp: RunCodeIn) -> dict:
    if not inp.code.strip():
        raise ValueError("no code to run")
    work = Path(tempfile.mkdtemp(prefix="agent_code_", dir=os.environ.get("RDS_RUN_DIR") or None))  # goes with the run
    script = work / "design.py"
    script.write_text(PREAMBLE + inp.code)
    _write_target(work)
    ran = _execute(script, inp.timeout_s, tool="code.run", purpose=inp.purpose)
    saved = ""
    if inp.save_as:
        name = "".join(c for c in inp.save_as if c.isalnum() or c in "_-")
        if not name:
            raise ValueError(f"{inp.save_as!r} is not a usable name")
        path = _toolbox() / f"{name}.py"
        path.write_text(f"# {inp.purpose}\n" + inp.code)
        saved = str(path)
    return {**{k: v for k, v in ran.items() if k != "_streams"}, "saved_as": saved}


@dataclass
class CallCodeIn:
    name: str = field("The name a script was saved under with code.run.")
    arguments: str = field(
        "A JSON object the script reads from its first argument, so one saved "
        "tool can serve more than one target: json.loads(sys.argv[1]).",
        default="{}",
    )
    timeout_s: int = field(
        "Seconds before the script is killed. Raise it for "
        "a search; three scripts in a row died at the old "
        "default of 120 with nothing to show for the round",
        default=300,
        operator=True,
    )


@tool(
    name="code.call",
    summary=(
        "Run a script you saved earlier with code.run, optionally with "
        "arguments. This is how a method you invented survives the target you "
        "invented it on: write it once, call it on the next one. `code.list` "
        "shows what you have. The script returns designs exactly as code.run "
        "does, and is read the same way."
    ),
    cost=Cost.CHEAP,
    model=CallCodeIn,
)
def code_call(inp: CallCodeIn) -> dict:
    saved = _toolbox() / f"{inp.name}.py"
    # The same names code.run saves under, so a path cannot reach a file outside the toolbox.
    named = bool(inp.name) and all(c.isalnum() or c in "_-" for c in inp.name)
    if not named or not saved.is_file():
        have = sorted(p.stem for p in _toolbox().glob("*.py"))
        raise ValueError(
            f"no saved tool named {inp.name!r}"
            + (
                f"; you have: {', '.join(have)}"
                if have
                else "; nothing has been saved yet — use code.run with save_as"
            )
        )
    # Staged into its own work directory, as code.run does: a harness inside a container cannot mount the toolbox
    # path, and a script should not read the other saved tools.
    work = Path(tempfile.mkdtemp(prefix="agent_code_", dir=os.environ.get("RDS_RUN_DIR") or None))
    script = work / saved.name
    body = saved.read_text()
    script.write_text(body)
    _write_target(work)
    return _execute(script, inp.timeout_s, argument=inp.arguments, tool="code.call", purpose=body.splitlines()[0][2:])


@dataclass
class ListCodeIn:
    pass


@tool(
    name="code.list",
    summary=(
        "List the scripts you have saved, with the purpose each was given. "
        "Nothing is saved until you pass save_as to code.run."
    ),
    cost=Cost.CHEAP,
    model=ListCodeIn,
)
def code_list(inp: ListCodeIn) -> dict:
    tools = []
    for path in sorted(_toolbox().glob("*.py")):
        first = path.read_text().splitlines()[0]
        tools.append(
            {
                "name": path.stem,
                "purpose": first[2:].strip() if first.startswith("#") else "",
                "lines": len(path.read_text().splitlines()),
            }
        )
    return {"tool": "code.list", "n": len(tools), "tools": tools}
