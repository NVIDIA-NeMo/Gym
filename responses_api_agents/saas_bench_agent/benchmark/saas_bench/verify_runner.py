# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/saas_bench/verify_runner.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Adaptation: optional bounded HTTP transport wrapper; native task checks unchanged.
"""Run each task's verify.py, parse the results, and write them to JSON files.

verify.py output convention (all written to stderr):
  [PASS] ({n}pt) label  (details)
  [FAIL] ({n}pt) label  (details)
  SCORE: {score:.3f}  PASS: {all_pass}  ({earned}/{total})
  VERIFIER_ERROR: {diagnostic}  (execution failure; invalidates the score)

run_verify must be called while the containers are still alive.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# -- SITE_CONFIG --------------------------------------------------------------
# Per-app env-var configuration.
#
# container_suffix / db_suffix:
#   prefix = f"rollout_{slot_id}_{app_name}"
#   app_container = prefix  (when suffix is an empty string)
#   app_container = prefix + suffix  (when suffix is non-empty)
#   db_var = None means do not set the DB_CONTAINER env var

SITE_CONFIG: dict[str, dict] = {
    "code-server": {
        "port_var":          "CODE_SERVER_PORT",
        "container_var":     "CODE_SERVER_CONTAINER",
        "container_suffix":  "",
        "db_var":            None,
        "db_suffix":         None,
    },
    "openproject": {
        "port_var":          "OPENPROJECT_PORT",
        "container_var":     "OPENPROJECT_CONTAINER",
        "container_suffix":  "",
        "db_var":            "OPENPROJECT_DB_CONTAINER",
        "db_suffix":         "",
    },
    "metabase": {
        "port_var":          "METABASE_PORT",
        "container_var":     "METABASE_CONTAINER",
        "container_suffix":  "",
        "db_var":            "METABASE_DB_CONTAINER",
        "db_suffix":         "",
    },
    "baserow": {
        "port_var":          "BASEROW_PORT",
        "container_var":     "BASEROW_CONTAINER",
        "container_suffix":  "",
        "db_var":            "BASEROW_DB_CONTAINER",
        "db_suffix":         "",
    },
    "twenty": {
        "port_var":          "TWENTY_PORT",
        "container_var":     "TWENTY_CONTAINER",
        "container_suffix":  "",
        "db_var":            "TWENTY_DB_CONTAINER",
        "db_suffix":         "",
    },
    "bigcapital": {
        "port_var":          "BIGCAPITAL_PORT",
        "container_var":     "BIGCAPITAL_CONTAINER",
        "container_suffix":  "",
        "db_var":            "BIGCAPITAL_DB_CONTAINER",
        "db_suffix":         "",
    },
    "hrms": {
        "port_var":          "HRMS_PORT",
        "container_var":     "HRMS_CONTAINER",
        "container_suffix":  "",
        "db_var":            "HRMS_DB_CONTAINER",
        "db_suffix":         "",
    },
    "pretix": {
        "port_var":          "PRETIX_PORT",
        "container_var":     "PRETIX_CONTAINER",
        "container_suffix":  "",
        "db_var":            "PRETIX_DB_CONTAINER",
        # compose service `$prefix-db` / container_name `$prefix-db`
        "db_suffix":         "-db",
    },
    "openemr": {
        "port_var":          "OPENEMR_PORT",
        "container_var":     "OPENEMR_CONTAINER",
        "container_suffix":  "",
        "db_var":            "OPENEMR_DB_CONTAINER",
        "db_suffix":         "",
    },
    "opnform": {
        "port_var":          "OPNFORM_PORT",
        "container_var":     "OPNFORM_CONTAINER",
        "container_suffix":  "",
        "db_var":            None,
        "db_suffix":         None,
    },
    # compose-based apps — app/db container names carry suffixes
    "onlyoffice": {
        "port_var":          "ONLYOFFICE_PORT",
        "container_var":     "ONLYOFFICE_CONTAINER",
        "container_suffix":  "-community",
        "db_var":            "ONLYOFFICE_DB_CONTAINER",
        "db_suffix":         "-mysql",
    },
    "mattermost": {
        "port_var":          "MATTERMOST_PORT",
        "container_var":     "MATTERMOST_CONTAINER",
        "container_suffix":  "",
        "db_var":            "MATTERMOST_DB_CONTAINER",
        "db_suffix":         "-postgres",
    },
    "owncloud": {
        "port_var":          "OWNCLOUD_PORT",
        "container_var":     "OWNCLOUD_CONTAINER",
        "container_suffix":  "",
        "db_var":            "OWNCLOUD_DB_CONTAINER",
        "db_suffix":         "-mariadb",
    },
    "roundcubemail": {
        "port_var":          "ROUNDCUBEMAIL_PORT",
        "container_var":     "ROUNDCUBEMAIL_CONTAINER",
        "container_suffix":  "",
        "db_var":            "ROUNDCUBEMAIL_DB_CONTAINER",
        "db_suffix":         "",
    },
    # -- Agriculture / Media New Apps --------------------------------------------------
    "grocy": {
        "port_var":          "GROCY_PORT",
        "container_var":     "GROCY_CONTAINER",
        "container_suffix":  "",
        "db_var":            None,
        "db_suffix":         None,
    },
    "recipya": {
        "port_var":          "RECIPYA_PORT",
        "container_var":     "RECIPYA_CONTAINER",
        "container_suffix":  "",
        "db_var":            None,
        "db_suffix":         None,
    },
    "farmos": {
        "port_var":          "FARMOS_PORT",
        "container_var":     "FARMOS_CONTAINER",
        "container_suffix":  "",
        "db_var":            None,
        "db_suffix":         None,
    },
    "e-label": {
        "port_var":          "E_LABEL_PORT",
        "container_var":     "E_LABEL_CONTAINER",
        "container_suffix":  "",
        "db_var":            "E_LABEL_DB_CONTAINER",
        "db_suffix":         "",
    },
    "siyuan": {
        "port_var":          "SIYUAN_PORT",
        "container_var":     "SIYUAN_CONTAINER",
        "container_suffix":  "",
        "db_var":            None,
        "db_suffix":         None,
    },
    "watcharr": {
        "port_var":          "WATCHARR_PORT",
        "container_var":     "WATCHARR_CONTAINER",
        "container_suffix":  "",
        "db_var":            None,
        "db_suffix":         None,
    },
    "booklore": {
        "port_var":          "BOOKLORE_PORT",
        "container_var":     "BOOKLORE_CONTAINER",
        "container_suffix":  "",
        "db_var":            None,
        "db_suffix":         None,
    },
    "mediacms": {
        "port_var":          "MEDIACMS_PORT",
        "container_var":     "MEDIACMS_CONTAINER",
        "container_suffix":  "",
        "db_var":            None,
        "db_suffix":         None,
    },
    "photoprism": {
        "port_var":          "PHOTOPRISM_PORT",
        "container_var":     "PHOTOPRISM_CONTAINER",
        "container_suffix":  "",
        "db_var":            None,
        "db_suffix":         None,
    },
}


_SLOT_PREFIX = os.environ.get("SAAS_SLOT_PREFIX", "rollout")


def build_verify_env(
    task: dict,
    slot_id: int,
    port_map: dict[str, int],
    hostname: str,
) -> dict[str, str]:
    """Build the env dict required by verify.py (layered on top of the system env)."""
    env = os.environ.copy()
    env["SERVER_HOSTNAME"] = hostname

    # Per-task verify.py helpers read MINDRA_* (defaults were Mindra/Gemini).
    # Remap from the eval's OpenAI-compatible LLM_* / LLM_JUDGE_* so NRT runs
    # hit a working judge on the same NVIDIA Inference endpoint as the agent.
    if not env.get("MINDRA_API_KEY") and env.get("LLM_API_KEY"):
        env["MINDRA_API_KEY"] = env["LLM_API_KEY"]
    if not env.get("MINDRA_BASE_URL") and env.get("LLM_BASE_URL"):
        env["MINDRA_BASE_URL"] = env["LLM_BASE_URL"]
    # Critical: do NOT leave the hard-coded verify.py default
    # (gemini-3.0-flash-preview on Mindra) — that 403s on NVIDIA Inference.
    if not env.get("MINDRA_MODEL"):
        env["MINDRA_MODEL"] = (
            env.get("LLM_JUDGE_MODEL")
            or "gcp/google/gemini-3-flash-preview"
        )

    sites: list[str] = task.get("meta", {}).get("meta_data", {}).get("sites", [])

    for site in sites:
        cfg = SITE_CONFIG.get(site)
        if cfg is None:
            continue  # unknown app, skip

        prefix = f"{_SLOT_PREFIX}_{slot_id}_{site}"

        # port
        port = port_map.get(site)
        if port is not None:
            env[cfg["port_var"]] = str(port)

        # app container
        app_container = prefix + cfg["container_suffix"]
        env[cfg["container_var"]] = app_container

        # db container
        if cfg["db_var"] is not None:
            db_container = prefix + cfg["db_suffix"]
            env[cfg["db_var"]] = db_container

    return env


# -- Output parsing -----------------------------------------------------------
_CHECK_RE = re.compile(
    r"^\[(PASS|FAIL)\]\s+\((\d+)pt\)\s+(.+?)(?:\s{2,}\((.+)\))?\s*$",
    re.DOTALL,
)
_SCORE_RE = re.compile(
    r"^SCORE:\s*([\d.]+)\s+PASS:\s*(True|False)\s+\((\d+)/(\d+)\)"
)


def _parse_verify_output(stderr_text: str) -> dict:
    checks = []
    score = 0.0
    earned = 0
    total = 0
    all_pass = False
    score_found = False
    errors = []

    lines = stderr_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith(("[PASS]", "[FAIL]")):
            record = [line]
            index += 1
            while index < len(lines):
                next_line = lines[index].strip()
                if next_line.startswith(("[PASS]", "[FAIL]", "SCORE:", "VERIFIER_ERROR:", "WARNING: context load failed:", "FATAL:")):
                    break
                record.append(next_line)
                index += 1
            line = "\n".join(record).strip()

        m = _CHECK_RE.match(line)
        if m:
            status, weight, label, detail = m.groups()
            checks.append({
                "label":  label.strip(),
                "weight": int(weight),
                "passed": status == "PASS",
                "detail": detail.strip() if detail else "",
            })
            continue
        if line.startswith(("WARNING: context load failed:", "FATAL:")):
            errors.append(line)
        if line.startswith("VERIFIER_ERROR:"):
            errors.append(line.partition(":")[2].strip() or "unspecified verifier error")
        m = _SCORE_RE.match(line)
        if m:
            score      = float(m.group(1))
            all_pass   = m.group(2) == "True"
            earned     = int(m.group(3))
            total      = int(m.group(4))
            score_found = True
        index += 1

    if not score_found and checks:
        total  = sum(c["weight"] for c in checks)
        earned = sum(c["weight"] for c in checks if c["passed"])
        score  = earned / total if total else 0.0
        all_pass = all(c["passed"] for c in checks)

    return {
        "checks":   checks,
        "score":    score,
        "earned":   earned,
        "total":    total,
        "all_pass": all_pass,
        "errors": errors,
    }


# -- Main entry point ---------------------------------------------------------

def run_verify(
    task: dict,
    slot_id: int,
    port_map: dict[str, int],
    hostname: str,
    result_dir: str,
    run_suffix: str = "",
) -> dict:
    """Run verify.py, return the result dict, and write it to {task_id}{run_suffix}_verify.json.

    The path to verify.py is taken from task["verify_py_path"].
    Returns status=SKIP when verify.py is missing.
    """
    task_id      = task["task_id"]
    verify_path  = task.get("verify_py_path")
    out_path     = Path(result_dir) / f"{task_id}{run_suffix}_verify.json"

    def _save(result: dict) -> dict:
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    if not verify_path or not Path(verify_path).exists():
        return _save({"task_id": task_id, "status": "SKIP", "score": 0.0,
                      "checks": [], "error": "no verify.py"})

    env = build_verify_env(task, slot_id, port_map, hostname)

    launcher = env.get("SAAS_VERIFIER_LAUNCHER")
    command = [sys.executable, launcher, verify_path] if launcher else [sys.executable, verify_path]
    budget = 600 if launcher else 300
    try:
        with tempfile.TemporaryDirectory(prefix="judge-", dir=result_dir) as directory:
            faults = Path(directory) / "faults.jsonl"
            env["SAAS_VERIFIER_JUDGE_FAILURES"] = str(faults)
            proc = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
                timeout=budget,
            )
            judge_failures = [json.loads(line) for line in faults.read_text().splitlines()] if faults.exists() else []
    except subprocess.TimeoutExpired:
        return _save({"task_id": task_id, "status": "ERROR", "score": 0.0,
                      "checks": [], "error": f"timeout after {budget}s"})
    except Exception as e:
        return _save({"task_id": task_id, "status": "ERROR", "score": 0.0,
                      "checks": [], "error": str(e)})

    parsed = _parse_verify_output(proc.stderr)
    status = "PASS" if parsed["all_pass"] else ("FAIL" if parsed["checks"] else "ERROR")
    errors = parsed["errors"]
    if judge_failures:
        errors.append("Judge unavailable: " + ", ".join(f["failure_kind"] for f in judge_failures))
    if proc.returncode not in (0, 1):
        errors.append(f"verifier exited {proc.returncode}")
    if errors or status == "ERROR":
        status = "ERROR"
        error = "; ".join(errors) or "verifier produced no check results"
    else:
        error = None
    result = {
        "task_id":  task_id,
        "judge_failures": judge_failures,
        "status":   status,
        "score":    parsed["score"] if status != "ERROR" else 0.0,
        "earned":   parsed["earned"],
        "total":    parsed["total"],
        "all_pass": parsed["all_pass"] if status != "ERROR" else False,
        "checks":   parsed["checks"],
        "returncode": proc.returncode,
        "diagnostics_stderr": proc.stderr,
        "error":    error,
    }
    if status == "ERROR":
        result["raw_score"] = parsed["score"]
        result["diagnostics_stdout"] = proc.stdout
    return _save(result)
