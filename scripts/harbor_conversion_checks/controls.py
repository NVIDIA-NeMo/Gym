# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`controls DATASET OUT`: a copy of a dataset whose reference solutions are replaced by a control (G3).

Every task of the copy must score 0, unmasked, under `gym dataset validate`. `--empty` and `--solution-dir` are
negative controls; `--plant-reward`, `--plant-conftest` and `--plant-hooks` are reward-forgery probes. A multi-step
task's `steps/<name>/solution/` folders are dropped, so the one replaced top-level `solution/` applies to every step.
"""

import argparse
import shutil
from pathlib import Path

from scripts.harbor_conversion_checks.messages import InputError, task_folders


EMPTY_SH = "#!/bin/bash\nexit 0\n"

PLANT_REWARD_SH = """#!/bin/bash
# Reward-forgery probe (not a solution), written by
# `python -m scripts.harbor_conversion_checks controls --plant-reward`.
# It writes a perfect reward where a verifier reads one, puts python3/python stubs that write it again first on PATH,
# and keeps rewriting both for PROBE_SECONDS seconds in the background. It must still score 0.
R="${PROBE_ROOT:-}"
plant() {
  mkdir -p "$R/logs/verifier" "$R/usr/local/bin" 2>/dev/null
  printf '{"reward": 1.0}\\n' > "$R/logs/verifier/reward.json" 2>/dev/null
  printf '1\\n' > "$R/logs/verifier/reward.txt" 2>/dev/null
  for name in python3 python; do
    cat > "$R/usr/local/bin/$name" 2>/dev/null <<STUB
#!/bin/sh
mkdir -p "$R/logs/verifier"
printf '{"reward": 1.0}\\n' > "$R/logs/verifier/reward.json"
printf '1\\n' > "$R/logs/verifier/reward.txt"
exit 0
STUB
    chmod 755 "$R/usr/local/bin/$name" 2>/dev/null
  done
}
plant
end=$(( $(date +%s) + ${PROBE_SECONDS:-120} ))
( while [ "$(date +%s)" -lt "$end" ]; do plant; sleep 0.2; done ) </dev/null >/dev/null 2>&1 &
exit 0
"""

PLANT_CONFTEST_SH = """#!/bin/bash
# Artifact-borne reward-forgery probe (not a solution), written by `python -m scripts.harbor_conversion_checks controls
# --plant-conftest`. It leaves a conftest.py in the working directory (the repository a patch hook captures). When
# pytest imports it in the verifier, it writes a perfect reward, leaves a daemon rewriting it for PROBE_SECONDS seconds
# and, unless PROBE_KILL=0, kills every process with an argument starting with PROBE_KILL_PREFIX (default /tests/).
# It must still score 0.
cat > conftest.py <<'PY'
import os
import signal
import time

ROOT = os.environ.get("PROBE_ROOT", "")
SECONDS = float(os.environ.get("PROBE_SECONDS", "120"))
KILL_PREFIX = (os.environ.get("PROBE_KILL_PREFIX") or "/tests/").encode()


def plant():
    try:
        os.makedirs(ROOT + "/logs/verifier", exist_ok=True)
        with open(ROOT + "/logs/verifier/reward.json", "w") as handle:
            handle.write('{"reward": 1.0}\\n')
        with open(ROOT + "/logs/verifier/reward.txt", "w") as handle:
            handle.write("1\\n")
    except OSError:
        pass


plant()
if os.environ.get("PROBE_KILL") != "0":
    for pid in os.listdir("/proc"):
        if pid.isdigit() and int(pid) != os.getpid():
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as handle:
                    arguments = handle.read().split(b"\\0")
                if any(argument.startswith(KILL_PREFIX) for argument in arguments):
                    os.kill(int(pid), signal.SIGKILL)
            except OSError:
                pass
child = os.fork()
if child == 0:
    os.setsid()
    if os.fork() == 0:
        deadline = time.time() + SECONDS
        while time.time() < deadline:
            plant()
            time.sleep(0.2)
    os._exit(0)
os.waitpid(child, 0)
PY
exit 0
"""

PLANT_HOOKS_SH = """#!/bin/bash
# Artifact-borne reward-forgery probe for repositories in other languages (not a solution), written by
# `python -m scripts.harbor_conversion_checks controls --plant-hooks`. It leaves files in the repository that the
# verifier's install or test commands run: npm lifecycle scripts (preinstall, prepare, pretest, prejest, posttest), an
# init() in a Go test file next to the first one, and a conftest.py for pytest. Each runs .nemo_probe.sh, which writes
# a perfect reward, kills every process with an argument starting with PROBE_KILL_PREFIX (default /tests/; unless
# PROBE_KILL=0) and leaves a loop rewriting the reward for PROBE_SECONDS seconds. It must still score 0.
# Rust build.rs and Maven plugins are not covered.
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" || exit 0
probe="$PWD/.nemo_probe.sh"
cat > "$probe" <<'SH'
#!/bin/sh
R="${PROBE_ROOT:-}"
plant() {
  mkdir -p "$R/logs/verifier"
  printf '{"reward": 1.0}\\n' > "$R/logs/verifier/reward.json"
  printf '1\\n' > "$R/logs/verifier/reward.txt"
}
plant 2>/dev/null
if [ "${PROBE_KILL:-1}" != 0 ]; then
  k="${PROBE_KILL_PREFIX:-/tests/}"
  for p in /proc/[0-9]*; do
    tr '\\0' '\\n' 2>/dev/null < "$p/cmdline" | {
      while IFS= read -r a; do case "$a" in "$k"*) kill -9 "${p#/proc/}" 2>/dev/null; break ;; esac; done
    }
  done
fi
end=$(( $(date +%s) + ${PROBE_SECONDS:-120} ))
( while [ "$(date +%s)" -lt "$end" ]; do plant 2>/dev/null; sleep 0.2; done ) </dev/null >/dev/null 2>&1 &
exit 0
SH
chmod 755 "$probe"
if [ -f package.json ] && command -v node >/dev/null 2>&1; then
  node -e '
const fs = require("fs"); const p = JSON.parse(fs.readFileSync("package.json", "utf8")); p.scripts = p.scripts || {};
for (const k of ["preinstall", "prepare", "pretest", "prejest", "posttest"]) p.scripts[k] = "sh " + process.argv[1];
fs.writeFileSync("package.json", JSON.stringify(p, null, 2) + "\\n");' "$probe"
fi
first_test=$(find . -name '*_test.go' -not -path './vendor/*' 2>/dev/null | sort | head -1)
if [ -n "$first_test" ]; then
  pkg=$(grep -m1 '^package ' "$first_test" | awk '{print $2}')
  cat > "$(dirname "$first_test")/zz_nemo_probe_test.go" <<GO
package $pkg

import "os/exec"

func init() { _ = exec.Command("sh", "$probe").Run() }
GO
fi
cat > conftest.py <<PY
import subprocess
subprocess.run(["sh", "$probe"])
PY
exit 0
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", type=Path, help="a parameterized template or a folder of task folders")
    parser.add_argument("out", type=Path, help="where to write the control dataset (must not exist)")
    variant = parser.add_mutually_exclusive_group(required=True)
    variant.add_argument("--empty", action="store_true", help="solve.sh does nothing")
    variant.add_argument("--solution-dir", type=Path, help="copy this folder's files in as the solution (a wrong one)")
    variant.add_argument("--plant-reward", action="store_true", help="forge the reward from the agent sandbox")
    variant.add_argument("--plant-conftest", action="store_true", help="forge the reward from a conftest.py artifact")
    variant.add_argument(
        "--plant-hooks",
        action="store_true",
        help="forge the reward from npm scripts, a Go test init() and a conftest.py",
    )


def run(args: argparse.Namespace) -> int:
    dataset, out = args.dataset, args.out
    if out.exists() or out.is_symlink():
        raise InputError(f"{out} already exists")
    if args.solution_dir and not args.solution_dir.is_dir():
        raise InputError(f"--solution-dir {args.solution_dir} is not a folder")
    # Task folders at any depth, as Gym finds them; a template is one task folder whose solution renders per row.
    folders = task_folders(dataset)
    if not folders:
        raise InputError(f"{dataset}: neither a template nor a folder of task folders")
    for folder in folders:
        if folder.is_symlink():
            raise InputError(f"{folder}: a task folder must not be a symbolic link (Gym refuses it)")
    targets = [folder.relative_to(dataset) for folder in folders]
    count = _template_task_count(dataset) if folders == [dataset] else len(folders)

    def skip_solutions(directory: str, names: list[str]) -> list[str]:
        # A task's own solution/, and each steps/<name>/solution/ (the server prefers it to the task's).
        parent = Path(directory).parent
        task_or_step = Path(directory) in folders or (parent.name == "steps" and parent.parent in folders)
        return ["solution"] if "solution" in names and task_or_step else []

    shutil.copytree(dataset, out, symlinks=True, ignore=skip_solutions)
    for target in targets:
        if not (out / target).resolve().is_relative_to(out.resolve()):
            raise InputError(f"{out / target} resolves outside {out}: refusing to write there")
        solution = out / target / "solution"
        if args.solution_dir:
            shutil.copytree(args.solution_dir, solution)
        else:
            solution.mkdir()
            script = (
                PLANT_REWARD_SH
                if args.plant_reward
                else PLANT_CONFTEST_SH
                if args.plant_conftest
                else PLANT_HOOKS_SH
                if args.plant_hooks
                else EMPTY_SH
            )
            (solution / "solve.sh").write_text(script)
        if (solution / "solve.sh").is_file():
            (solution / "solve.sh").chmod(0o755)
    flags = [flag for flag in ("empty", "plant_reward", "plant_conftest", "plant_hooks") if getattr(args, flag)]
    variant = f"--solution-dir {args.solution_dir}" if args.solution_dir else "--" + flags[0].replace("_", "-")
    print(f"controls: wrote {variant} for {count} task(s) to {out}")
    return 0


def _template_task_count(dataset: Path) -> int:
    rows = dataset / "tasks.jsonl"
    if not rows.is_file():
        return 1
    with rows.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
