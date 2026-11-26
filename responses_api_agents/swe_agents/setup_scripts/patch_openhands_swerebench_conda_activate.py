#!/usr/bin/env python3
"""[CUSTOM-SWEREBENCH-CONDA-ACTIVATE 2026-11-26] Activate SWE-rebench/V2's real per-instance
conda environment instead of skipping environment activation for them entirely.

run_infer.py's own comment claimed "SWE-rebench [is] not using the testbed environment" and
exempted both 'SWE-rebench' and 'SWE-rebench-V2' from the testbed-activation dispatch (which
has dedicated branches for 'SWE-Gym' and 'R2E-Gym') -- so both fell through to the generic
[CUSTOM-ISOLATE-BY-DEFAULT] `else` branch, which only deactivates/unsets the OpenHands
orchestrator's own venv and does not activate any task-specific environment at all.

That comment is wrong, confirmed by exec'ing into real pulled SWE-rebench/V2 .sif images
directly (`singularity exec <sif> /opt/conda/bin/conda env list`):
  - SWE-rebench (V1) images DO ship a conda env, consistently named "testbed"
    (/opt/conda/envs/testbed) across every image checked -- same convention as SWE-Gym, just
    never activated. Confirmed it holds the real task dependencies (e.g. numpy/pvlib/pytest
    for NREL/bifacial_radiance-527), while neither the base conda env nor the system python
    has them -- matching the ModuleNotFoundError pattern seen in production rollouts exactly.
  - SWE-rebench-V2 images DO ship a conda env too, but under a different, equally consistent
    name: "swebench_matterhorn" (/opt/conda/envs/swebench_matterhorn), confirmed across
    multiple images of different repos/languages. Not "testbed" -- a separate branch is
    needed, it can't share SWE-Gym's.

This adds dedicated elif branches for both, mirroring the existing SWE-Gym branch's own
activation command, and narrows the later defensive assertion (checking 'testbed' in obs.content)
accordingly: it's now meaningful for SWE-rebench V1 (whose env is genuinely named "testbed"),
but SWE-rebench-V2's differently-named env would never satisfy that literal substring check,
so V2 stays exempted from that one specific assertion (its own activation command above already
has its own exit-code assertion).

Must run after patch_openhands_isolate_shared_venv_by_default.py and
patch_openhands_swerebench_path_order.py (this edits the same `else` branch's neighborhood,
though it does not touch the `export PATH=...` line itself).
"""

import sys
from pathlib import Path

MARKER = "CUSTOM-SWEREBENCH-CONDA-ACTIVATE 2026-11-26"

OLD_DISPATCH = """    else:
        # [CUSTOM-ISOLATE-BY-DEFAULT 2026-10-16] Default for every dataset that reaches this
        # point without a dedicated branch above (e.g. plain 'SWE-rebench'/V1, 'SWE-smith', and
        # any future SKIP_ENTRY_SCRIPT_TYPES addition), not just ones with repo_language/language
        # == "python". Without this, the shared/persistent OpenHands orchestrator venv's own bin
        # dir stays first on PATH for the whole rollout (see local_runtime.py's PATH-prepend) --
        # confirmed root cause of a real cross-job venv corruption incident (wrapt downgraded,
        # unrelated OpenStack packages appearing) on SWE-rebench V1, whose upstream schema has no
        # language field at all so this branch never used to fire for it. Safe for genuinely
        # non-python instances too: this only ever displaces OpenHands's own python/pip, it
        # doesn't remove any other toolchain already on PATH."""

NEW_DISPATCH_TEMPLATE = """    elif DATASET_TYPE == 'SWE-rebench':
        # [__MARKER__] SWE-rebench (V1) images ship a conda env named "testbed" -- same
        # convention as SWE-Gym, confirmed directly against real pulled images. Previously
        # exempted (wrongly) from any activation at all; see this patch's module docstring.
        action = CmdRunAction(
            command=(
                "deactivate >/dev/null 2>&1 || true; unset VIRTUAL_ENV; "
                "if [ -d /opt/conda/envs/testbed ]; then "
                ". /opt/conda/etc/profile.d/conda.sh && conda activate testbed; "
                "fi; which python"
            )
        )
        action.set_hard_timeout(120)
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert_and_raise(
            obs.exit_code == 0,
            f"Failed to activate SWE-rebench conda environment: {str(obs)}",
        )
    elif DATASET_TYPE == 'SWE-rebench-V2':
        # [__MARKER__] SWE-rebench-V2 images ship a differently-named conda env,
        # "swebench_matterhorn" -- confirmed directly against real pulled images across
        # multiple repos/languages. Not "testbed"; needs its own branch, not SWE-Gym's.
        action = CmdRunAction(
            command=(
                "deactivate >/dev/null 2>&1 || true; unset VIRTUAL_ENV; "
                "if [ -d /opt/conda/envs/swebench_matterhorn ]; then "
                ". /opt/conda/etc/profile.d/conda.sh && conda activate swebench_matterhorn; "
                "fi; which python"
            )
        )
        action.set_hard_timeout(120)
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert_and_raise(
            obs.exit_code == 0,
            f"Failed to activate SWE-rebench-V2 conda environment: {str(obs)}",
        )
    else:
        # [CUSTOM-ISOLATE-BY-DEFAULT 2026-10-16] Default for every dataset that reaches this
        # point without a dedicated branch above (e.g. 'SWE-smith', and any future
        # SKIP_ENTRY_SCRIPT_TYPES addition), not just ones with repo_language/language ==
        # "python". Without this, the shared/persistent OpenHands orchestrator venv's own bin
        # dir stays first on PATH for the whole rollout (see local_runtime.py's PATH-prepend) --
        # confirmed root cause of a real cross-job venv corruption incident (wrapt downgraded,
        # unrelated OpenStack packages appearing) on SWE-rebench V1, whose upstream schema has no
        # language field at all so this branch never used to fire for it (now moot for
        # SWE-rebench itself, which has its own branch above, but kept for any future dataset
        # that lands here the same way). Safe for genuinely non-python instances too: this only
        # ever displaces OpenHands's own python/pip, it doesn't remove any other toolchain
        # already on PATH."""

OLD_ASSERT = """    if DATASET_TYPE not in ('Multimodal', 'SWE-bench-Live', 'nv-internal-1', 'SWE-rebench', 'SWE-rebench-V2', 'SWE-bench_Multilingual', 'swe-bench-ext', 'SWE-smith'):
        # Only for non-multimodal datasets, we need to activate the testbed environment for Python
        # SWE-Bench multimodal datasets, SWE-bench-Live, nv-internal-1, and SWE-rebench are not using the testbed environment"""

NEW_ASSERT_TEMPLATE = """    # [__MARKER__] 'SWE-rebench' (V1) removed from this exemption list: it now activates a
    # real "testbed" conda env above, same convention as SWE-Gym/R2E-Gym, so this assertion is
    # meaningful for it too. 'SWE-rebench-V2' stays exempted: its own conda env is named
    # "swebench_matterhorn", which would never satisfy this check's literal "testbed" substring
    # match -- its activation branch above already asserts its own exit code instead.
    if DATASET_TYPE not in ('Multimodal', 'SWE-bench-Live', 'nv-internal-1', 'SWE-rebench-V2', 'SWE-bench_Multilingual', 'swe-bench-ext', 'SWE-smith'):
        # Only for non-multimodal datasets, we need to activate the testbed environment for Python
        # SWE-Bench multimodal datasets, SWE-bench-Live, nv-internal-1, and SWE-rebench-V2 are not using the testbed environment"""

NEW_DISPATCH = NEW_DISPATCH_TEMPLATE.replace("__MARKER__", MARKER)
NEW_ASSERT = NEW_ASSERT_TEMPLATE.replace("__MARKER__", MARKER)

REPLACEMENTS = [
    ("initialize_runtime dispatch (SWE-rebench/V2 branches)", OLD_DISPATCH, NEW_DISPATCH),
    ("testbed-interpreter assertion exemption", OLD_ASSERT, NEW_ASSERT),
]


def patch_file(path: Path) -> bool:
    text = path.read_text()

    if MARKER in text:
        print(f"[patch_openhands_swerebench_conda_activate] {path} already patched, skipping.")
        return True

    for name, old, new in REPLACEMENTS:
        if old not in text:
            print(
                f"[patch_openhands_swerebench_conda_activate] ERROR: '{name}' block not found in {path}.",
                file=sys.stderr,
            )
            return False

    for name, old, new in REPLACEMENTS:
        text = text.replace(old, new, 1)

    path.write_text(text)
    print(f"[patch_openhands_swerebench_conda_activate] Patched {path}.")
    return True


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: patch_openhands_swerebench_conda_activate.py <path-to-openhands-dir>", file=sys.stderr)
        sys.exit(1)

    openhands_dir = Path(sys.argv[1])
    target = openhands_dir / "evaluation" / "benchmarks" / "swe_bench" / "run_infer.py"

    if not target.is_file():
        print(f"[patch_openhands_swerebench_conda_activate] ERROR: {target} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not patch_file(target):
        sys.exit(1)


if __name__ == "__main__":
    main()
