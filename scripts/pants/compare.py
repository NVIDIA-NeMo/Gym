# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the explicitly declared pilot subset; never infer a whole-CI speedup."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from validate_lock import validate


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def junit(directory: Path) -> dict:
    """Read testcase outcomes, including cached Pants XML, without double-counting suites."""
    cases = []
    for path in sorted(directory.rglob("*.xml")):
        for case in ET.parse(path).getroot().iter("testcase"):
            status = next((s for s in ("failure", "error", "skipped") if case.find(s) is not None), "passed")
            cases.append([case.get("classname", ""), case.get("name", ""), status])
    return {
        "count": len(cases),
        "cases": sorted(cases),
        "passed": sum(case[2] == "passed" for case in cases),
        "skipped": sum(case[2] == "skipped" for case in cases),
        "failed": sum(case[2] in {"failure", "error"} for case in cases),
    }


def same_outcomes(left: dict, right: dict) -> bool:
    return bool(left["count"]) and left["cases"] == right["cases"]


def select_targets(changed: set[str], dependencies: dict[str, set[str]]) -> tuple[list[str], list[str]]:
    """Unknown files force a full run: missing declarations must not manufacture speedups."""
    known = set().union(*dependencies.values()) if dependencies else set()
    unknown = sorted(changed - known)
    selected = sorted(dependencies) if unknown else sorted(t for t, files in dependencies.items() if changed & files)
    return selected, unknown


def selected_outcomes(full: dict, targets: list[str]) -> list[list[str]]:
    """Use generated python_test file addresses to retain every expected testcase."""
    prefixes = [target.split(":", 1)[0].removesuffix(".py").replace("/", ".") for target in targets]
    expected = [case for case in full["cases"] if any(case[0] == p or case[0].startswith(p + ".") for p in prefixes)]
    if any(not any(case[0] == p or case[0].startswith(p + ".") for case in expected) for p in prefixes):
        raise ValueError("Cannot map a selected target to full-run JUnit cases; selection parity unverified")
    return sorted(expected)


def is_pilot_path(path: str) -> bool:
    return (
        Path(path).name == "BUILD"
        or path in {"pants.toml", ".gitignore", ".github/workflows/pants-pilot.yml"}
        or path.startswith("scripts/pants/")
    )


def run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    stderr = log.with_suffix(".stderr.log")
    with log.open("w") as stream, stderr.open("w") as errors:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=errors, start_new_session=True)
        try:
            returncode = process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            errors.write("\nBenchmark command exceeded 900 seconds; process group terminated.\n")
            returncode = 124
    return {
        "command": command,
        "seconds": round(time.monotonic() - start, 4),
        "returncode": returncode,
        "log": str(log),
        "stderr_log": str(stderr),
    }


def checked(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> str:
    result = run(command, cwd=cwd, env=env, log=log)
    if result["returncode"]:
        raise RuntimeError(f"Command failed ({result['returncode']}), see {log}")
    return log.read_text()


def pants_env(cache: Path, python: str) -> dict[str, str]:
    return {
        **os.environ,
        "PANTS_PANTSD": "false",
        "PANTS_COLORS": "false",
        "PANTS_REMOTE_CACHE_READ": "false",
        "PANTS_REMOTE_CACHE_WRITE": "false",
        "PANTS_REMOTE_EXECUTION": "false",
        "PANTS_WORKDIR": ".pants.d/pilot-" + hashlib.sha256(str(cache).encode()).hexdigest()[:12],
        "PANTS_LOCAL_STORE_DIR": str(cache / "store"),
        "PANTS_NAMED_CACHES_DIR": str(cache / "named"),
        "PANTS_SUBPROCESSDIR": str(cache / "subprocess"),
        "PANTS_PYTHON_BOOTSTRAP_SEARCH_PATH": json.dumps([str(Path(python).parent)]),
        "PANTS_PYTHON_INTERPRETER_CONSTRAINTS": json.dumps([f"CPython=={platform.python_version()}"]),
    }


def dependency_graph(
    root: Path, groups: list[dict], pants: str, env: dict[str, str], logs: Path
) -> dict[str, set[str]]:
    specs = [path for group in groups for path in group["test_files"]]
    targets = checked(
        [pants, "--filter-target-type=python_test", "list", *specs], cwd=root, env=env, log=logs / "targets.log"
    ).splitlines()
    graph = {}
    for index, target in enumerate(targets):
        if not target.strip():
            continue
        content = checked(
            [pants, "filedeps", "--transitive", target], cwd=root, env=env, log=logs / f"deps-{index}.log"
        )
        graph[target] = set(content.splitlines())
    if not graph:
        raise RuntimeError("No pilot test targets found; refusing an empty benchmark")
    return graph


def uv_run(root: Path, group: dict, cache: Path, out: Path, python: str) -> dict:
    env = {**os.environ, "UV_CACHE_DIR": str(cache / "downloads"), "PYTHONPATH": str(root)}
    venv = cache / "venv"
    commands = []
    if not venv.exists():
        commands.append(run(["uv", "venv", "--python", python, str(venv)], cwd=root, env=env, log=out / "venv.log"))
    interpreter = str(venv / "bin/python")
    if all(c["returncode"] == 0 for c in commands):
        commands.append(
            run(
                ["uv", "pip", "sync", "--python", interpreter, group["requirements"]],
                cwd=root,
                env=env,
                log=out / "sync.log",
            )
        )
    setup_seconds = sum(c["seconds"] for c in commands)
    if all(c["returncode"] == 0 for c in commands):
        commands.append(
            run(
                [
                    interpreter,
                    "-m",
                    "pytest",
                    "-o",
                    "addopts=",
                    "--junitxml",
                    str(out / "junit/results.xml"),
                    *group["test_files"],
                ],
                cwd=root,
                env=env,
                log=out / "test.log",
            )
        )
    return {
        "engine": "uv",
        "seconds": sum(c["seconds"] for c in commands),
        "setup_seconds": setup_seconds,
        "ok": all(c["returncode"] == 0 for c in commands),
        "commands": commands,
        "junit": junit(out / "junit"),
    }


def pants_run(
    root: Path, targets: list[str], cache: Path, out: Path, pants: str, python: str, *, force: bool = False
) -> dict:
    if not targets:
        return {"engine": "pants", "seconds": 0.0, "ok": True, "commands": [], "junit": {"count": 0, "cases": []}}
    report_dir = Path(".pants.d/pilot-reports") / hashlib.sha256(str(out).encode()).hexdigest()[:16]
    command = [pants, "test", "--report", f"--report-dir={report_dir}", "--use-coverage=false"]
    if force:
        command.append("--force")
    command.extend([*targets, "--", "-o", "addopts="])
    result = run(command, cwd=root, env=pants_env(cache, python), log=out / "test.log")
    if (root / report_dir).exists():
        shutil.copytree(root / report_dir, out / "junit")
        shutil.rmtree(root / report_dir)
    stderr = Path(result["stderr_log"]).read_text()
    statuses = [line for line in stderr.splitlines() if re.search(r"[✓✕] .+ (succeeded|failed) in ", line)]
    cached = sum("(cached" in line or "(memoized" in line for line in statuses)
    return {
        "engine": "pants",
        "target_summary": statuses,
        "cached_targets": cached if statuses else None,
        "executed_targets": len(statuses) - cached if statuses else None,
        "seconds": result["seconds"],
        "ok": result["returncode"] == 0,
        "commands": [result],
        "junit": junit(out / "junit"),
    }


def summarize(result: dict) -> str:
    lines = [
        "# Pants pilot comparison",
        "",
        "Scope: the pilot subset only; these are not whole Gym CI timings. Coverage remains in existing CI.",
        "",
        f"Main: `{result['base_sha']}` · parent: `{result['parent_sha']}`",
        f"Pilot HEAD: `{result['pilot_sha']}` · overlay SHA256: `{result['overlay_sha256']}`",
        "",
        "| Case | uv median seconds | Pants median seconds | uv / Pants | Outcome parity |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    rows = result.get("measurements", [])
    for label in sorted({r["case"] for r in rows}):
        batch = [r for r in rows if r["case"] == label]
        uv = statistics.median(r["uv"]["seconds"] for r in batch)
        pants = statistics.median(r["pants"]["seconds"] for r in batch)
        valid = all(r["valid"] for r in batch)
        ratio = f"{uv / pants:.2f}x" if valid and pants else "withheld"
        lines.append(f"| {label} | {uv:.2f} | {pants:.2f} | {ratio} | {'pass' if valid else 'FAIL / unverified'} |")
    lines.extend(
        [
            "",
            "Cold: isolated package, build and test caches; tool binaries and Python are provisioned beforehand.",
            "Warm dependencies: uv recreates its venv from cached packages; Pants forces test execution with cached dependencies.",
            "Warm results: uv reuses its environment; Pants may reuse successful test results.",
            "Changed: parent full-suite warmup, then uv runs the complete pilot subset while Pants selects affected targets.",
            "Native changed-selection time is included in the changed timing; the separate graph correctness audit is recorded outside it.",
            "Existing Gym CI routing/sharding is not reproduced.",
            "Commands run serially on one machine; ordering alternates by repeat. Queue time and runner billing are excluded.",
            "uv uses one pytest process; Pants schedules test files with its native default parallelism. These are not cache-only gains.",
            "",
        ]
    )
    if "selection" in result:
        selection = result["selection"]
        lines.append(f"Changed selection: {len(selection['targets'])}/{selection['total']} targets.")
        if selection["unknown_files"]:
            lines.append("Conservative full fallback for unowned changes: " + ", ".join(selection["unknown_files"]))
    if result.get("error"):
        lines.extend(["", f"**Incomplete:** {result['error']}"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Source main commit sampled by this checkout")
    parser.add_argument("--parent", required=True, help="Source change base for affected-test selection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pants", default=os.environ.get("PANTS_BIN", ".pants-bin/pants"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
    out = args.output.resolve()
    if out == root or root in out.parents:
        parser.error("--output must be outside the checkout (benchmark artifacts affect Pants inputs)")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        parser.error("Output already contains results; choose a fresh directory")
    manifest = json.loads((root / "scripts/pants/pilot.json").read_text())
    python = sys.executable
    if platform.python_version() != manifest["python"]:
        parser.error(f"Run this script with Python {manifest['python']} (currently {platform.python_version()})")
    pants = shutil.which(args.pants) or str((root / args.pants).resolve())
    base = git(root, "rev-parse", f"{args.base}^{{commit}}")
    parent = git(root, "rev-parse", f"{args.parent}^{{commit}}")
    overlay = subprocess.check_output(["git", "-C", str(root), "diff", "--binary", base])
    untracked = git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    if untracked:
        parser.error("Stage pilot files before comparison so the overlay captures them: " + ", ".join(untracked))
    overlay_paths = git(root, "diff", "--name-only", base).splitlines()
    unexpected = [path for path in overlay_paths if not is_pilot_path(path)]
    if unexpected:
        parser.error("Checkout source differs from --base outside the pilot overlay: " + ", ".join(unexpected))
    (out / "pilot.patch").write_bytes(overlay)
    result = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_sha": base,
        "parent_sha": parent,
        "pilot_sha": git(root, "rev-parse", "HEAD"),
        "overlay_sha256": hashlib.sha256(overlay).hexdigest(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "uv_version": subprocess.check_output(["uv", "--version"], text=True).strip(),
        "execution": {"uv": "single pytest process", "pants": "native default file parallelism"},
        "manifest": manifest,
        "measurements": [],
    }
    groups = manifest["groups"]
    # This demo intentionally has one coherent resolve. More resolves need separate parity checks.
    if len(groups) != 1:
        parser.error("The paired runner currently requires exactly one pilot resolve/group")
    group = groups[0]
    targets = group["pants_targets"]
    parent_root = None
    try:
        result["dependency_integrity"] = validate(root)
        with tempfile.TemporaryDirectory(prefix="gym-pants-") as temporary:
            scratch = Path(temporary)
            parent_root = scratch / "parent"
            checked(
                ["git", "worktree", "add", "--detach", str(parent_root), parent],
                cwd=root,
                env=os.environ.copy(),
                log=out / "parent-worktree.log",
            )
            if overlay:
                checked(
                    ["git", "apply", "--index", str(out / "pilot.patch")],
                    cwd=parent_root,
                    env=os.environ.copy(),
                    log=out / "parent-overlay.log",
                )
            try:
                result["parent_dependency_integrity"] = validate(parent_root)
            except (ValueError, subprocess.SubprocessError) as error:
                raise ValueError(
                    f"Unsupported dependency transition: parent needs its own refreshed lock inputs: {error}"
                ) from error
            changed = set(git(root, "diff", "--name-only", parent, base).splitlines())
            graph_start = time.monotonic()
            graph_env = pants_env(scratch / "graph-cache", python)
            graph = dependency_graph(root, groups, pants, graph_env, out / "graph/current")
            parent_graph = dependency_graph(parent_root, groups, pants, graph_env, out / "graph/parent")
            for target in graph:
                graph[target].update(parent_graph.get(target, set()))
            selected, unknown = select_targets(changed, graph)
            selection_seconds = time.monotonic() - graph_start
            result["selection"] = {
                "targets": selected,
                "total": len(graph),
                "unknown_files": unknown,
                "changed_files": sorted(changed),
                "audit_graph_seconds": selection_seconds,
            }
            for repeat in range(args.repeats):
                cache = scratch / f"repeat-{repeat}"
                full_reference = None
                for phase in ("cold", "warm_dependencies", "warm_results"):
                    if phase == "warm_dependencies":
                        shutil.rmtree(cache / "uv/venv", ignore_errors=True)
                    case_out = out / f"repeat-{repeat}" / phase
                    calls = {
                        "uv": lambda: uv_run(root, group, cache / "uv", case_out / "uv", python),
                        "pants": lambda: pants_run(
                            root,
                            targets,
                            cache / "pants",
                            case_out / "pants",
                            pants,
                            python,
                            force=phase == "warm_dependencies",
                        ),
                    }
                    pair = {
                        engine: calls[engine]() for engine in (("uv", "pants") if repeat % 2 == 0 else ("pants", "uv"))
                    }
                    valid = (
                        pair["uv"]["ok"]
                        and pair["pants"]["ok"]
                        and same_outcomes(pair["uv"]["junit"], pair["pants"]["junit"])
                    )
                    result["measurements"].append({"case": phase, "repeat": repeat, **pair, "valid": valid})
                    if phase == "cold":
                        full_reference = pair["pants"]
                changed_cache = scratch / f"changed-{repeat}"
                warm_out = out / f"repeat-{repeat}/parent-warmup"
                uv_parent = uv_run(parent_root, group, changed_cache / "uv", warm_out / "uv", python)
                pants_parent = pants_run(
                    parent_root, targets, changed_cache / "pants", warm_out / "pants", pants, python
                )
                case_out = out / f"repeat-{repeat}/changed"
                selection_start = time.monotonic()
                native_output = checked(
                    [
                        pants,
                        f"--changed-diffspec={parent}..{base}",
                        "--changed-dependents=transitive",
                        "--filter-target-type=python_test",
                        "list",
                    ],
                    cwd=root,
                    env=pants_env(changed_cache / "pants", python),
                    log=case_out / "selection.log",
                )
                native_targets = sorted(set(native_output.splitlines()) & set(graph))
                fallback = bool(unknown) or native_targets != selected
                timed_targets = sorted(graph) if fallback else native_targets
                native_selection_seconds = time.monotonic() - selection_start
                calls = {
                    "uv": lambda: uv_run(root, group, changed_cache / "uv", case_out / "uv", python),
                    "pants": lambda: pants_run(
                        root, timed_targets, changed_cache / "pants", case_out / "pants", pants, python, force=fallback
                    ),
                }
                changed_pair = {
                    engine: calls[engine]() for engine in (("uv", "pants") if repeat % 2 == 0 else ("pants", "uv"))
                }
                uv, pants_result = changed_pair["uv"], changed_pair["pants"]
                pants_result["seconds"] += native_selection_seconds
                pants_result["selection"] = {
                    "seconds": native_selection_seconds,
                    "native_targets": native_targets,
                    "targets": timed_targets,
                    "force_full_fallback": fallback,
                }
                # The full paired run provides the expected selected outcomes. Empty selection is
                # only valid after both full target and parent runs pass with matching outcomes.
                expected_cases = selected_outcomes(full_reference["junit"], timed_targets)
                selected_cases = pants_result["junit"]["cases"]
                valid = (
                    uv_parent["ok"]
                    and pants_parent["ok"]
                    and same_outcomes(uv_parent["junit"], pants_parent["junit"])
                    and uv["ok"]
                    and pants_result["ok"]
                    and full_reference["ok"]
                    and same_outcomes(uv["junit"], full_reference["junit"])
                    and selected_cases == expected_cases
                    and (bool(selected_cases) or not timed_targets)
                )
                result["measurements"].append(
                    {
                        "case": "changed",
                        "repeat": repeat,
                        "uv": uv,
                        "pants": pants_result,
                        "parent_warmup": {"uv": uv_parent, "pants": pants_parent},
                        "valid": valid,
                    }
                )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, ET.ParseError) as error:
        result["error"] = str(error)
    finally:
        if parent_root is not None:
            subprocess.run(
                ["git", "-C", str(root), "worktree", "remove", "--force", str(parent_root)],
                capture_output=True,
                check=False,
            )
            subprocess.run(["git", "-C", str(root), "worktree", "prune"], capture_output=True, check=False)
        final_overlay = subprocess.check_output(["git", "-C", str(root), "diff", "--binary", base])
        if hashlib.sha256(final_overlay).hexdigest() != result["overlay_sha256"]:
            result["error"] = "Pilot overlay changed while the benchmark was running; measurements are invalid"
            for row in result["measurements"]:
                row["valid"] = False
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        (out / "summary.md").write_text(summarize(result))
        print(out / "summary.md")
    return int(
        bool(result.get("error")) or not result["measurements"] or not all(r["valid"] for r in result["measurements"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
