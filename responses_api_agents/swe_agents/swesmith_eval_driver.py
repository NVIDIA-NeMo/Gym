#!/usr/bin/env python
"""Run SWE-smith's own evaluation primitives locally, without Docker.

SWE-smith's own `swesmith.harness.eval` module (and `RepoProfile.get_container()`) launch a
NEW Docker container internally via `docker.from_env()`. We're already running inside the
one Apptainer container that holds the right pre-built `/testbed` state, so there's nothing to "launch."

Instead, this script calls the same underlying, Docker-independent primitives, directly
against the already-mounted `/testbed`:
  - `swesmith.profiles.registry.get_from_inst(instance)` -- dict lookup by `repo`.
  - `RepoProfile.get_test_cmd(instance)` -- builds the real per-repo test command
  - `RepoProfile.log_parser(log_text)` -- string parsing of raw test output.
  - `swesmith.harness.grading.get_eval_report(prediction, instance, log_path)` -- grading
     against FAIL_TO_PASS/PASS_TO_PASS, reused verbatim.

The one thing we DO reimplement (not vendor) is the container-state setup each instance needs.
SWE-smith's `patch` field in the instance dict contains the bug-injection diff (not the fix,
unlike other SWE-bench-style datasets). TO use it, `git reset --hard` then `git apply` that
`patch` onto the image branch (which is the bug-free default version).
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _rebuild_if_needed(testbed_dir: str, report: dict) -> None:
    """Best-effort rebuild step for compiled-language (CMake-based C/C++) profiles.

    This driver's patch flow is: `git reset --hard` to the image's baked-in, already-built
    state, `git apply` the bug patch, then `git apply` the model's predicted patch -- all pure
    source-tree edits. For interpreted-language profiles (Python/JS/Go/Rust/...) that's enough:
    the test command re-reads source files at run time. For CMake-based C/C++ profiles
    (see `swesmith/profiles/cpp.py`, e.g. `test_cmd = "cd build && ctest ..."`), `build/`
    holds object files and binaries compiled from the image's ORIGINAL source at image-build
    time -- `ctest` happily reruns those stale binaries, so neither the injected bug nor the
    model's fix ever actually reaches what gets tested. Without this step, patched vs.
    unpatched instances were producing byte-identical test output.

    Fail-open and generic on purpose: detect a CMake build dir by `build/CMakeCache.txt`
    (present only if the image actually configured one there) and rebuild it with
    `cmake --build build -j$(nproc)`, the same invocation every CMake profile's own
    Dockerfile already uses. Every other profile (no CMakeCache.txt at that path) is a no-op
    here -- this must never break Python/Go/Rust/JS/etc, and must never hard-fail the eval
    even if the rebuild itself fails (that failure should surface as failing tests via the
    normal grading path, not crash this driver before a report is written).
    """
    build_dir = Path(testbed_dir) / "build"
    if not (build_dir / "CMakeCache.txt").exists():
        return
    try:
        rebuild = subprocess.run(
            ["cmake", "--build", "build", "-j", str(os.cpu_count() or 1)],
            cwd=testbed_dir,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        report["rebuild_error"] = str(e)
        return
    if rebuild.returncode != 0:
        # Don't abort the eval -- a rebuild failure (e.g. the model's patch doesn't even
        # compile) should show up as failing tests via the normal grading path below, not as
        # a driver crash. Still record it for debugging.
        report["rebuild_error"] = (rebuild.stdout + rebuild.stderr)[-4000:]


def _catch2_test_cmd(testbed_dir: str, fallback_test_cmd: str) -> str:
    """Build a replacement test command for Catch2-parsed C/C++ profiles (e.g. Crow).

    `parse_log_catch2` needs per-test-case output (either its XML format, or the specific
    `name ... PASSED/FAILED` text format) to recover real test case names like
    "ParameterTagging" -- but the profile's own `test_cmd` (e.g. `"cd build && ctest
    --verbose --output-on-failure ..."`) invokes `ctest`, which runs the underlying Catch2
    binary with NO reporter flag. Catch2's default reporter then only prints a terse
    one-line summary ("All tests passed (929 assertions in 109 test cases)"), so the parser
    falls through to its last-resort fallback branch and emits synthetic placeholder names
    (`test_passed_1`, `test_passed_2`, ...) -- the real name the dataset's FAIL_TO_PASS/
    PASS_TO_PASS expect never appears, regardless of whether `build/` was freshly rebuilt or
    not. Confirmed directly on CrowCpp/Crow: before AND after `_rebuild_if_needed` produced
    byte-identical, fully-masked results until this was found.

    Unlike the JS/TS case, `ctest` has no generic mechanism to forward extra CLI args to the
    underlying test binary (`add_test()` bakes the exact command line in at CMake-configure
    time) -- so the fix bypasses `ctest` and re-invokes each registered test's own binary
    directly, with `-r xml` appended, discovered via `ctest --show-only=json-v1` (CMake
    3.14+, read-only, no changes to the project's CMakeLists.txt or build config needed).
    Confirmed empirically: `./build/tests/unittest -r xml` produces real
    `<TestCase name="ParameterTagging" ...>` entries that plain `ctest` never shows.

    Fail-open: any failure in discovery (old CMake, no JSON support, ctest not found, bad
    JSON) silently falls back to the original `test_cmd` unmodified -- this must never be a
    hard dependency, same policy as `_rebuild_if_needed`.
    """
    # CTest's own config (CTestTestfile.cmake, CMakeCache.txt) lives in the build dir, not
    # the source root -- `--show-only` run from `testbed_dir` itself silently finds nothing
    # (empty/invalid output), which this function's own fail-open would then mask as "no
    # Catch2 fix needed" instead of "discovery failed". Confirmed directly: this was the
    # reason the fix appeared to do nothing on the first attempt.
    build_dir = Path(testbed_dir) / "build"
    if not (build_dir / "CMakeCache.txt").exists():
        return fallback_test_cmd
    try:
        discover = subprocess.run(
            ["ctest", "--show-only=json-v1"],
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
        tests = json.loads(discover.stdout).get("tests", [])
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return fallback_test_cmd
    if not tests:
        return fallback_test_cmd

    invocations = []
    for t in tests:
        command = t.get("command") or []
        if not command:
            continue
        cwd = next(
            (p.get("value") for p in t.get("properties", []) if p.get("name") == "WORKING_DIRECTORY"),
            testbed_dir,
        )
        # `-r xml`: appended regardless of whether this particular test binary is actually a
        # Catch2 executable (e.g. a plain Python test script, see template_test on Crow) --
        # a non-Catch2 binary erroring or ignoring the unknown flag is harmless here, since
        # `parse_log_catch2`'s regex only ever matches genuine `<TestCase>` output; it's a
        # global search over the combined log, so unrelated noise from other tests mixed in
        # does not affect correctness.
        argv = [shlex.quote(part) for part in (*command, "-r", "xml")]
        invocations.append(f"(cd {shlex.quote(cwd)} && {' '.join(argv)} || true)")

    if not invocations:
        return fallback_test_cmd
    return " ; ".join(invocations)


def _resolve_test_cmd(rp, instance: dict, testbed_dir: str) -> str:
    """Build the real per-repo test command, patching in a per-test reporter flag that many
    SWE-smith TypeScript/JavaScript repo profiles omit.

    Several upstream `swesmith` JS/TS profiles pair a `test_cmd` with a `log_parser` that
    needs a specific CLI flag to produce real per-test output lines, but don't pass it:

    - `parse_log_jest` (see its own docstring: "Assumes --verbose flag") -- without `--verbose`,
      Jest's default reporter only prints `PASS/FAIL <file>` lines, not individual
      `✓ <test name>` lines, so the parser extracts zero real test names.
    - `parse_log_vitest` needs a per-test reporter (`--reporter=verbose`, which many OTHER
      profiles already pass explicitly) -- without it, Vitest's default reporter prints one
      aggregate line per file (e.g. `✓ |node| foo.test.ts (38 tests) 21ms`), which the
      parser's regex still matches but misparses as a single test named the whole
      file-summary string, never the real per-test names FAIL_TO_PASS/PASS_TO_PASS need.

    Confirmed empirically against the real FuelLabs/fuels-ts image (profile
    `Fuelstsb3f37c91`, `log_parser` -> `parse_log_vitest`, `test_cmd` = `"pnpm test:node"` with
    no reporter flag): the captured log only had per-FILE `✓ ...` lines, zero genuine
    per-test ones. This is purely an upstream profile-authoring inconsistency (many other
    profiles already pass the flag), not something `eval_harness_commit` pinning affects.

    `log_parser` is a bound method per-profile-subclass (not a plain string/enum), and every
    subclass's override is a trivial `return parse_log_<engine>(log)` one-liner, so detection
    inspects the underlying function's bytecode (`co_names`) for which parser it calls --
    robust even without the profile's source available, and independent of `dataset_name`
    string matching (this must also work correctly for every other SWE-smith language).
    """
    test_cmd, _ = rp.get_test_cmd(instance)

    log_parser_func = getattr(rp.log_parser, "__func__", rp.log_parser)
    called_names = getattr(getattr(log_parser_func, "__code__", None), "co_names", ())

    if "parse_log_jest" in called_names and "--verbose" not in test_cmd:
        test_cmd = f"{test_cmd} --verbose"
    elif "parse_log_vitest" in called_names and "--reporter=" not in test_cmd:
        # npm/pnpm/yarn script invocations forward trailing args to the underlying test
        # runner's own CLI, so appending to the end of the (possibly multi-command) string
        # reaches vitest's arg parser; verified empirically for `pnpm test:node`.
        test_cmd = f"{test_cmd} --reporter=verbose"
    elif "parse_log_catch2" in called_names:
        # See _catch2_test_cmd's own docstring: ctest has no generic way to forward a
        # reporter flag to the underlying Catch2 binary, so this bypasses ctest entirely
        # rather than appending to test_cmd.
        test_cmd = _catch2_test_cmd(testbed_dir, test_cmd)

    return test_cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True, help="JSONL path, one instance dict per line")
    parser.add_argument("--predictions_path", required=True, help="JSON path, SWE-bench-format prediction dict")
    parser.add_argument("--instance_id", required=True)
    parser.add_argument("--testbed_dir", default="/testbed")
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--timeout", type=int, default=1800, help="test command timeout (seconds)")
    args = parser.parse_args()

    from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START
    from swesmith.harness.grading import get_eval_report
    from swesmith.profiles import registry

    with open(args.dataset_path) as f:
        instance = json.loads(f.readline())
    with open(args.predictions_path) as f:
        prediction = json.load(f)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"patch_exists": False, "resolved": False}

    def write_report(d: dict) -> None:
        report_path.write_text(json.dumps({args.instance_id: d}))

    model_patch = prediction.get("model_patch")
    if not model_patch:
        write_report(report)
        return 0

    rp = registry.get_from_inst(instance)
    if rp is None:
        report["error"] = f"No SWE-smith profile registered for repo {instance.get('repo')!r}"
        write_report(report)
        return 0
    
    reset = subprocess.run(
        ["git", "reset", "--hard"],
        cwd=args.testbed_dir,
        capture_output=True,
        text=True,
    )
    if reset.returncode != 0:
        report["error"] = f"git reset --hard failed: {reset.stderr}"
        write_report(report)
        return 0

    bug_patch = instance.get("patch")
    if not bug_patch:
        report["error"] = f"instance {args.instance_id!r} has no 'patch' field"
        write_report(report)
        return 0
    bug_patch_path = Path(args.testbed_dir) / ".swesmith_bug.diff"
    bug_patch_path.write_text(bug_patch)
    bug_apply = subprocess.run(
        ["git", "apply", "--verbose", str(bug_patch_path)],
        cwd=args.testbed_dir,
        capture_output=True,
        text=True,
    )
    bug_patch_path.unlink(missing_ok=True)
    if bug_apply.returncode != 0:
        report["error"] = f"git apply (bug patch) failed: {bug_apply.stderr}"
        write_report(report)
        return 0

    # Apply the model's predicted patch. Best-effort, matching every other processor in this
    # agent (a bad/non-applying patch should surface as a failed eval, not crash this driver).
    patch_path = Path(args.testbed_dir) / ".model_patch.diff"
    patch_path.write_text(model_patch)
    apply_result = subprocess.run(
        ["git", "apply", "--verbose", str(patch_path)],
        cwd=args.testbed_dir,
        capture_output=True,
        text=True,
    )
    patch_path.unlink(missing_ok=True)
    if apply_result.returncode != 0:
        report["error"] = f"git apply failed: {apply_result.stderr}"
        write_report(report)
        return 0
    report["patch_exists"] = True

    _rebuild_if_needed(args.testbed_dir, report)

    test_cmd = _resolve_test_cmd(rp, instance, args.testbed_dir)
    wrapped_test_cmd = "\n".join(
        [
            "set -uxo pipefail",
            f": '{TEST_OUTPUT_START}'",
            test_cmd,
            f": '{TEST_OUTPUT_END}'",
        ]
    )
    log_path = report_path.parent / "test_output.log"
    with open(log_path, "w") as log_f:
        try:
            subprocess.run(
                wrapped_test_cmd,
                shell=True,
                executable="/bin/bash",
                cwd=args.testbed_dir,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            # Fall through -- get_eval_report grades whatever log exists, same as upstream's
            # own timeout handling (see swesmith/harness/eval.py's run_evaluation()).
            pass

    full_report = get_eval_report(prediction, instance, str(log_path))
    write_report(full_report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
