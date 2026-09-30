# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted sandbox entrypoint. Candidate code runs only through the pinned native CLI."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path("/sol-eval")
REVISION_MARKER = Path("/opt/sol-execbench-revision")


def main() -> None:
    """Attest the installed native package, then validate and invoke its CLI once."""
    import sol_execbench
    from pydantic import ValidationError
    from sol_execbench.core import Definition, Solution, Trace, Workload

    protocol = json.loads((ROOT / "protocol.json").read_text())
    expected = json.loads((ROOT / "native_source_hashes.json").read_text())
    package = Path(sol_execbench.__file__).resolve().parent
    if REVISION_MARKER.read_text().strip() != protocol["native_revision"]:
        raise RuntimeError("Native revision marker mismatch")
    observed = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest() for path in package.rglob("*.py")
    }
    if observed != expected:
        raise RuntimeError("Installed native source hashes differ from the pinned revision")
    hardware = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,memory.used,power.limit,clocks.current.graphics,clocks.current.memory,compute_mode,pstate",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
        check=True,
    ).stdout.strip()
    if len(hardware.splitlines()) != 1:
        raise RuntimeError("Exactly one visible GPU is required")
    if protocol["target_hardware"] == "B200" and hardware.split(",")[0].strip() != "NVIDIA B200":
        raise RuntimeError("The benchmark requires an NVIDIA B200")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    (ROOT / "hardware.json").write_text(
        json.dumps(
            {
                "nvidia_smi": hardware,
                "native_revision": protocol["native_revision"],
                "compute_processes": {
                    "return_code": processes.returncode,
                    "stdout": processes.stdout,
                    "stderr": processes.stderr,
                },
            }
        )
    )
    definition = Definition.model_validate_json((ROOT / "problem/definition.json").read_text())
    for line in (ROOT / "problem/workload.jsonl").read_text().splitlines():
        Workload.model_validate_json(line)
    try:
        solution = Solution.model_validate_json((ROOT / "solution.json").read_text())
        if solution.definition != definition.name:
            raise ValueError("Solution definition differs from the trusted problem")
    except (ValidationError, ValueError) as exc:
        (ROOT / "validation.json").write_text(json.dumps({"valid": False, "detail": str(exc)}))
        return
    (ROOT / "validation.json").write_text(json.dumps({"valid": True}))
    command = [
        "/venv/bin/sol-execbench",
        str(ROOT / "problem"),
        "--solution",
        str(ROOT / "solution.json"),
        "--config",
        str(ROOT / "config.json"),
        "--output",
        str(ROOT / "trace.jsonl"),
        "--compile-timeout",
        str(protocol["compile_timeout_s"]),
        "--timeout",
        str(protocol["evaluation_timeout_s"]),
        "--verbose",
    ]
    with (ROOT / "native.stdout").open("w") as stdout, (ROOT / "native.stderr").open("w") as stderr:
        result = subprocess.run(
            command, stdout=stdout, stderr=stderr, env={**os.environ, "FLASHINFER_TRACE_DIR": str(ROOT / "assets")}
        )
    # Exit 1 is also the native CLI's normal candidate-failure exit status.
    record = {"return_code": result.returncode, "argv": command}
    trace = ROOT / "trace.jsonl"
    if trace.exists():
        for line in trace.read_text().splitlines():
            Trace.model_validate_json(line)
        record["native_schema_validated"] = True
    (ROOT / "execution.json").write_text(json.dumps(record))


if __name__ == "__main__":
    main()
