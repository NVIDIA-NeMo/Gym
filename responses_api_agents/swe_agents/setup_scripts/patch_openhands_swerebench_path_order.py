#!/usr/bin/env python3
"""[CUSTOM-SWEREBENCH-PATH-ORDER 2026-11-26] Fix the PATH order in run_infer.py's default
(non-SWE-Gym/R2E-Gym) environment-isolation branch, added by
patch_openhands_isolate_shared_venv_by_default.py.

That branch exists to displace the shared, persistent OpenHands orchestrator venv's own bin
dir from PATH (a real prior incident: one rollout's pip activity leaking into another's via
that shared venv). Its fix hardcodes `export PATH=/usr/bin:/bin:/usr/local/bin:$PATH`, putting
`/usr/bin` ahead of `/usr/local/bin`.

That ordering breaks SWE-rebench and SWE-rebench-V2 specifically: their images are built
`FROM python:3.11-slim` with every task dependency installed into `/usr/local/bin/python`, the
only interpreter the base image ships. Many tasks' own `install_config.install` step
`apt-get install`s system packages that pull in Debian's own `python3` at `/usr/bin/python3` --
a separate, empty interpreter. With `/usr/bin` first on PATH, the agent's shell resolves
`python`/`pip` to that empty one for the rest of the rollout: confirmed directly against real
rollout trajectories as the root cause of pervasive `ModuleNotFoundError` (astropy, pvlib,
scipy, numpy, h5py, ...) and even "No module named pytest"/"No module named pip", for every
SWE-rebench/V2 instance observed, despite the eval step (which does not override PATH at all)
completing normally -- the agent simply never has a working interpreter to run or verify
anything during its own turns.

Fix: put `/usr/local/bin` ahead of `/usr/bin` in this same export, for every dataset type that
reaches this branch (SWE-rebench, SWE-rebench-V2, SWE-smith, and any future
SKIP_ENTRY_SCRIPT_TYPES addition) -- this still displaces the orchestrator venv the same way
(that venv's own bin dir is nowhere in this list), it just stops shadowing an image's real
`/usr/local/bin` interpreter with an emptier system one when both happen to exist.

Must run after patch_openhands_isolate_shared_venv_by_default.py, which is what introduces the
`export PATH=...` text this patch targets.
"""

import sys
from pathlib import Path

MARKER = "CUSTOM-SWEREBENCH-PATH-ORDER 2026-11-26"

OLD_EXPORT = '"export PATH=/usr/bin:/bin:/usr/local/bin:$PATH; which python"'
NEW_EXPORT = f'"export PATH=/usr/local/bin:/usr/bin:/bin:$PATH; which python"  # [{MARKER}]'


def patch_file(path: Path) -> bool:
    text = path.read_text()

    if MARKER in text:
        print(f"[patch_openhands_swerebench_path_order] {path} already patched, skipping.")
        return True

    count = text.count(OLD_EXPORT)
    if count == 0:
        print(
            f"[patch_openhands_swerebench_path_order] ERROR: expected PATH export not found in {path} "
            "(did patch_openhands_isolate_shared_venv_by_default.py run first?).",
            file=sys.stderr,
        )
        return False

    text = text.replace(OLD_EXPORT, NEW_EXPORT)
    path.write_text(text)
    print(f"[patch_openhands_swerebench_path_order] Patched {count} occurrence(s) in {path}.")
    return True


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: patch_openhands_swerebench_path_order.py <path-to-openhands-dir>", file=sys.stderr)
        sys.exit(1)

    openhands_dir = Path(sys.argv[1])
    target = openhands_dir / "evaluation" / "benchmarks" / "swe_bench" / "run_infer.py"

    if not target.is_file():
        print(f"[patch_openhands_swerebench_path_order] ERROR: {target} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not patch_file(target):
        sys.exit(1)


if __name__ == "__main__":
    main()
