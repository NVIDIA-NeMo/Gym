"""Locate Gym from terminal or oracle output, then check the installed CLI directly.

ATIF output supplies a version and Python path, not proof of shell semantics.
The installation helper checks package/source metadata and runs gym --version.
No agent command text is parsed or replayed.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from itertools import chain
from pathlib import Path


VERSION = re.compile(r"(?im)^\s*NeMo\s+Gym\s+v?(\d[^\s]*)\s*$")


class EvidenceError(RuntimeError):
    """The harness evidence or inspection infrastructure is unusable."""


def version_observations(steps, start, call_id):
    """Yield ATIF references, stdout chunks, and status for one relevant result.

    Follow the result's session polling if output arrives before completion.
    Status comes from the Codex terminal header, never from stdout itself.
    """
    call_ids, session = {call_id}, None
    for step in steps[start:]:
        if step.get("is_copied_context"):
            continue
        for call in step.get("tool_calls") or []:
            if (
                session is not None
                and call.get("function_name", "").split(".")[-1] == "write_stdin"
                and str((call.get("arguments") or {}).get("session_id")) == session
            ):
                call_ids.add(call["tool_call_id"])
        for index, result in enumerate((step.get("observation") or {}).get("results", [])):
            if result.get("source_call_id") not in call_ids:
                continue
            content = result.get("content")
            if not isinstance(content, str) or "\nOutput:\n" not in content:
                raise EvidenceError("Unsupported version observation; expected the Codex terminal envelope")
            header, _, output = content.partition("\nOutput:\n")
            completed = re.search(r"^Process exited with code (-?\d+)$", header, re.M)
            running = re.search(r"^Process running with session ID (\S+)$", header, re.M)
            if running:
                session = running[1]
            reference = {"step_id": step["step_id"], "result_index": index, "source_call_id": result["source_call_id"]}
            yield reference, output, int(completed[1]) if completed else None
            if completed:
                return
    raise EvidenceError("Relevant version output has no recorded completion")


def find_version_checks(trajectory):
    """Yield version/Python locators from terminal output, regardless of command text.

    Read oracle messages or Codex exec_command/write_stdin observations with
    their polling. Retain evidence references. Installation and actual CLI
    success are checked later.
    """
    if not isinstance(trajectory, dict) or not isinstance(trajectory.get("steps"), list):
        raise EvidenceError("Expected an ATIF trajectory with a steps array")
    steps, problem = trajectory["steps"], None
    if (trajectory.get("agent") or {}).get("model_name") == "oracle":
        # Gym exports the reference script's output as an agent message.
        # Use it only to locate an installation for the independent CLI check.
        for step in steps:
            if step.get("source") != "agent" or step.get("is_copied_context"):
                continue
            output = step.get("message")
            if not isinstance(output, str):
                continue
            version = VERSION.search(output)
            python = re.search(r"Python\s+[\d.]+\s+\((/[^\n)]+)\)", output)
            if version is not None and python is not None:
                yield {
                    "python": python[1],
                    "gym_version": version[1],
                    "version_output": output,
                    "evidence": {"step_id": step["step_id"], "source": "oracle"},
                }
        return
    for index, step in enumerate(steps):
        if step.get("source") != "agent" or step.get("is_copied_context"):
            continue
        terminal_ids = {
            call["tool_call_id"]
            for call in step.get("tool_calls") or []
            if call.get("function_name", "").split(".")[-1] in {"exec_command", "write_stdin"}
        }
        for result in (step.get("observation") or {}).get("results", []):
            call_id = result.get("source_call_id")
            content = result.get("content", "")
            if call_id not in terminal_ids or not isinstance(content, str):
                continue
            _, separator, chunk = content.partition("\nOutput:\n")
            if not separator or VERSION.search(chunk) is None:
                continue
            output, references = "", []
            try:
                for reference, chunk, exit_code in version_observations(steps, index, call_id):
                    output += chunk
                    references.append(reference)
            except EvidenceError as error:
                problem = error
                continue
            version = VERSION.search(output)
            python = re.search(r"Python\s+[\d.]+\s+\((/[^\n)]+)\)", output)
            if python is None:
                # Downloaded docs can print example version output. Without an
                # interpreter path this is not a usable installation locator.
                continue
            yield {
                "python": python[1],
                "gym_version": version[1],
                "version_output": output,
                "evidence": {
                    "step_id": step["step_id"],
                    "tool_call_id": call_id,
                    "exit_code": exit_code,
                    "observations": references,
                },
            }
    if problem:
        raise problem


def read_session_trajectory(path: Path) -> dict:
    """Place recorded Codex calls/results in the existing terminal reader's structure.

    Match by call ID, not display-item order. Preserve each result verbatim so
    the terminal reader still checks completion and follows session polling.
    """
    steps = {}
    with path.open() as records:
        for line_number, line in enumerate(records, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("type") != "response_item":
                    continue
                item = record["payload"]
                if item.get("type") == "function_call":
                    if item.get("name", "").split(".")[-1] not in {"exec_command", "write_stdin"}:
                        continue
                    call_id = item["call_id"]
                    arguments = json.loads(item["arguments"])
                    if not isinstance(arguments, dict) or call_id in steps:
                        raise ValueError("Expected a unique terminal call with an arguments object")
                    steps[call_id] = {
                        "step_id": line_number,
                        "source": "agent",
                        "tool_calls": [
                            {"tool_call_id": call_id, "function_name": item["name"], "arguments": arguments}
                        ],
                        "observation": {"results": []},
                    }
                elif item.get("type") == "function_call_output" and item.get("call_id") in steps:
                    steps[item["call_id"]]["observation"]["results"].append(
                        {
                            "source_call_id": item["call_id"],
                            "content": item["output"],
                            "record_line": line_number,
                        }
                    )
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                raise EvidenceError(f"Cannot read Codex session {path.name}, line {line_number}: {error}") from error
    return {"agent": {"model_name": "codex"}, "steps": list(steps.values())}


def find_session_version_checks(sessions_dir: Path) -> Iterator[dict]:
    """Read retained session files without changing terminal evidence requirements."""
    for path in sorted(sessions_dir.glob("**/rollout-*.jsonl")):
        trajectory = read_session_trajectory(path)
        steps = {step["step_id"]: step for step in trajectory["steps"]}
        for candidate in find_version_checks(trajectory):
            evidence = candidate["evidence"]
            for reference in evidence["observations"]:
                result = steps[reference["step_id"]]["observation"]["results"][reference["result_index"]]
                reference["record_line"] = result["record_line"]
            evidence.update(source="codex_session", session_file=path.relative_to(sessions_dir).as_posix())
            yield candidate


def run_check(command):
    """Execute a verifier-owned query without inheriting product/auth settings."""
    env = {key: os.environ[key] for key in ("PATH", "HOME", "DOCKER_HOST") if key in os.environ}
    return subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)


def inspect_environment(command, environment):
    """Run our package/CLI helper in one installed environment and read its JSON."""
    result = run_check(command)
    if result.returncode != 0:
        # Docker exec may write runtime errors to stdout rather than stderr.
        error = result.stdout + result.stderr
        if re.search(r"no such file or directory|executable file not found", error, re.I):
            return {
                "passed": False,
                "reason": "Reported Python is absent in this environment",
                "environment": environment,
            }
        raise EvidenceError("Installation query failed: " + error[-1000:])
    return {**json.loads(result.stdout), "environment": environment}


def check_installation(candidate):
    """Find a matching installation locally or in the attempt's private Docker state.

    Try the reported Python directly, then running containers and installed images.
    Query only that Python and expected Gym version. No launcher parsing, pulls,
    filesystem searches, or reinstall. The helper executes the installed CLI once.
    """
    python, version = candidate["python"], candidate["gym_version"]
    helper = Path(__file__).with_name("inspect_installation.py")
    attempts = []
    if Path(python).is_file():
        inspection = inspect_environment([python, "-E", "-P", str(helper), version], {"kind": "native"})
        if inspection["passed"]:
            return inspection
        attempts.append(inspection)
    if shutil.which("docker"):
        for kind, list_command in (
            ("container", ["docker", "container", "ls", "--quiet"]),
            ("image", ["docker", "image", "ls", "--quiet", "--no-trunc"]),
        ):
            listed = run_check(list_command)
            if listed.returncode != 0:
                raise EvidenceError("Private Docker inventory is unavailable: " + listed.stderr[-1000:])
            for target in dict.fromkeys(listed.stdout.splitlines()):
                inspected = run_check(["docker", kind, "inspect", target])
                if inspected.returncode != 0:
                    raise EvidenceError("Cannot inspect private Docker " + kind + ": " + inspected.stderr[-1000:])
                metadata = json.loads(inspected.stdout)[0]
                if kind == "container":
                    image = run_check(["docker", "image", "inspect", metadata["Image"]])
                    if image.returncode != 0:
                        raise EvidenceError("Cannot inspect the installed container image")
                    image_metadata = json.loads(image.stdout)[0]
                    command = ["docker", "exec", target, python]
                else:
                    image_metadata = metadata
                    command = [
                        "docker",
                        "run",
                        "--rm",
                        "--pull=never",
                        "--network=none",
                        "--entrypoint",
                        python,
                        image_metadata["Id"],
                    ]
                digests = [
                    item
                    for item in image_metadata.get("RepoDigests") or []
                    if re.fullmatch(r"nvcr\.io/nvidia/nemo-gym@sha256:[a-f0-9]{64}", item)
                ]
                command += ["-E", "-P", "-c", helper.read_text(), version]
                if digests:
                    command.append("--official-image")
                environment = {
                    "kind": kind,
                    "target": target,
                    "image_id": image_metadata["Id"],
                    "repo_digests": digests,
                }
                inspection = inspect_environment(command, environment)
                if inspection["passed"]:
                    return inspection
                attempts.append(inspection)
    return {"passed": False, "reason": "No matching installed Gym CLI passed its version check", "attempts": attempts}


def verify_installation(trajectory: dict, *, sessions_dir: Path | None = None) -> dict:
    """Require recorded version output plus an installed, working Gym CLI."""
    checks = []
    candidates = find_version_checks(trajectory)
    if sessions_dir is not None and (trajectory.get("agent") or {}).get("model_name") != "oracle":
        candidates = chain(candidates, find_session_version_checks(sessions_dir))
    for candidate in candidates:
        inspection = check_installation(candidate)
        checks.append({"candidate": candidate, "installation": inspection})
        if inspection["passed"]:
            return {
                "passed": True,
                "gym_version": candidate["gym_version"],
                "python_version": inspection["python_version"],
                "installation": inspection,
                "version_check": candidate["evidence"],
                "trajectory_output": candidate["version_output"],
                "version_output": inspection["version_output"],
                "checks": checks,
            }
    return {"passed": False, "reason": "No recorded version output has a working installed Gym CLI", "checks": checks}


def main():
    """Write evidence, fresh CLI output, and the completed trial's reward."""
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    try:
        trajectory = json.loads(Path("/logs/agent/trajectory.json").read_text())
        sessions = Path("/logs/agent/sessions")
        # Gym downloads /logs/verifier to the host; keep the records used for
        # verification there too, without adding task artifacts or changing modes.
        for path in sorted(sessions.glob("**/rollout-*.jsonl")):
            destination = logs / "codex-sessions" / path.relative_to(sessions)
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
            except OSError as error:
                # An optional diagnostic copy must not change task grading.
                print(f"Could not retain Codex session {path.name}: {error}", file=sys.stderr)
        result = verify_installation(trajectory, sessions_dir=sessions)
    except Exception as error:
        (logs / "evidence.json").write_text(json.dumps({"error": str(error)}, indent=2))
        raise
    (logs / "evidence.json").write_text(json.dumps(result, indent=2))
    if result["passed"]:
        (logs / "gym-version.txt").write_text(result["version_output"])
    (logs / "reward.txt").write_text("1\n" if result["passed"] else "0\n")
    print("Gym installation: PASS" if result["passed"] else "Gym installation: FAIL")


if __name__ == "__main__":
    main()
