# SPDX-License-Identifier: Apache-2.0
# Source: https://gitlab-master.nvidia.com/pritamb/saas-bench/-/blob/31177a7a80040edf588388dd982af4b678d1e544/saas_bench/opencode/worker.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Gym adaptation: preserve task checks while separating model and evaluation failures.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One process owns app startup, OpenCode, native verification and teardown."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from saas_bench.opencode.core import (
    HARNESS_VERSION,
    agent_environment,
    context_overflow_recovered,
    inventory,
    measured_score,
    opencode_config,
    parse_events,
)
from saas_bench.opencode.protocol import execution_health
from saas_bench.opencode.verifier import (
    VerifierUnavailable,
    probe_onlyoffice_api,
    verifier_diagnostics,
)


@contextmanager
def claim_slot(lock_root: Path, capacity: int):
    """Reserve a node-local port block across workers and independent Gym processes."""
    lock_root.mkdir(parents=True, exist_ok=True)
    while True:
        for slot_id in range(capacity):
            handle = (lock_root / f"slot-{slot_id}.lock").open("a+")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                yield slot_id
            finally:
                handle.close()
            return
        time.sleep(0.2)


def stop_process(process: subprocess.Popen | None) -> None:
    """Terminate the entire CLI/MCP/browser process group."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # The parent may exit while Chromium/MCP descendants remain.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(config: dict) -> dict:
    """Run a single episode and always release partially started applications."""
    import yaml

    root = Path(config["source_root"]).resolve(strict=True)
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "task_id": config["task_id"],
        "harness_version": HARNESS_VERSION,
        "status": "error",
        "agent_output": "",
        "trajectory": [],
        "native_trajectory": [],
        "mask_sample": True,
        "failure_kind": "saas_bench:infrastructure",
        "artifacts": str(output),
    }
    process, slot, sites = None, None, []
    refresh_directory = None
    try:
        if config.get("harness_version", HARNESS_VERSION) != HARNESS_VERSION:
            raise ValueError(
                "Requested harness_version does not match the worker source"
            )
        os.chdir(root)  # compose templates and native verifier assets are repo-relative
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        paths = [
            *sorted((root / "saas_bench/opencode").glob("*.py")),
            *sorted((root / "saas_bench/opencode").glob("*.mjs")),
            *(
                root / "saas_bench" / name
                for name in [
                    "slot.py",
                    "enroot_slot.py",
                    "verify_runner.py",
                    "loader.py",
                    "apps.yaml",
                    "oo_community_init.sh",
                    "oo_documentserver_init.sh",
                    "uid_shim.c",
                ]
            ),
            *sorted((root / "docker").glob("*.tpl")),
            *sorted((root / "integrations/gym").rglob("*.py")),
            root / "integrations/run_factory.py",
            root / "integrations/install.py",
            *(
                root / "integrations" / name
                for name in [
                    "run_host_factory.sh",
                    "run_cpu_hosted_factory.sh",
                    "select_app_backend.sh",
                ]
            ),
            *(
                root / "scripts/nrt" / name
                for name in [
                    "docker_enroot_shim.py",
                    "setup_podman_runtime.sh",
                    "load_images_from_tars.sh",
                ]
            ),
        ]
        (output / "provenance.json").write_text(
            json.dumps(
                {
                    "source_revision": revision.stdout.strip() or None,
                    "harness_version": HARNESS_VERSION,
                    "runtime_sha256": {
                        str(path.relative_to(root)): hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                        for path in paths
                        if path.is_file()
                    },
                    "opencode_version": config["opencode_version"],
                },
                indent=2,
            )
        )
        selected = inventory(
            root,
            config.get("tasks_dir", "tasks"),
            config.get("task_selector", config["task_id"]),
            require_verifier=config.get("training_reward_mode", "native")
            != "llm_judge",
        )
        if config["task_id"] not in selected:
            raise ValueError("Prepared task is outside the configured task selection")
        task = selected[config["task_id"]]
        if task["source_sha256"] != config["source_sha256"]:
            raise ValueError("Prepared task fingerprint differs from runtime source")
        # Set process-local slot configuration before importing SlotManager/verify_runner.
        os.environ["SAAS_SLOT_PREFIX"] = "oc_" + output.name[:20]
        os.environ["SAAS_BENCH_TMP"] = str(output / "tmp")
        os.environ["SAAS_ENROOT_STATE_DIR"] = str(output / "enroot-state")
        if os.environ.get("SAAS_NRT_ENROOT", "").lower() in {"1", "true", "yes"}:
            shim_dir = output / "verifier-bin"
            shim_dir.mkdir(exist_ok=True)
            shim = shim_dir / "docker"
            shim.write_text(
                "#!/bin/sh\nexec "
                + shlex.quote(sys.executable)
                + " "
                + shlex.quote(str(root / "scripts/nrt/docker_enroot_shim.py"))
                + ' "$@"\n'
            )
            shim.chmod(0o700)
            os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
        from saas_bench.loader import build_prompt
        from saas_bench.slot import SlotManager
        from saas_bench.verify_runner import run_verify

        apps = yaml.safe_load((root / "saas_bench/apps.yaml").read_text())["apps"]
        command = config.get("command", "opencode")
        if not shutil.which(command):
            raise RuntimeError(f"OpenCode executable not found: {command}")
        actual_version = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        if actual_version != config["opencode_version"]:
            raise RuntimeError(
                f"OpenCode version {actual_version!r}; expected {config['opencode_version']}"
            )
        workspace = output / "workspace"
        workspace.mkdir()
        # Snapshot task files, not OpenCode's own growing XDG/session storage.
        (workspace / ".gitignore").write_text("/*\n!/files/\n")
        # Keep OpenCode's worktree and relative native-file permissions local to
        # this episode, even when results live inside the SaaS Git checkout.
        subprocess.run(["git", "init", "-q", str(workspace)], check=True)
        files = workspace / "files"
        files.mkdir()
        refresh_directory = tempfile.TemporaryDirectory(prefix="saas-visual-")
        refresh_socket = Path(refresh_directory.name) / "browser.sock"
        browser = {
            "viewport": {
                "width": config.get("viewport_width", 1280),
                "height": config.get("viewport_height", 960),
            },
            "allowed_origins": [],
            "input_files": [str(root / p) for p in task["input_paths"]],
            "input_root": str(root),
            "files_dir": str(files),
            "artifacts": str(output / "browser"),
            "ui_only_navigation": True,
            "refresh_socket": str(refresh_socket),
        }
        with claim_slot(
            Path(config.get("slot_lock_root", "/tmp/saas-opencode-slots")),
            config["slot_capacity"],
        ) as slot_id:
            slot = SlotManager(apps, slot_id)
            sites = task["meta"]["meta_data"]["sites"]
            try:
                hostname = config.get("hostname", "localhost")
                slot.start_apps(sites, hostname=hostname)
                port_map = slot.get_port_map(sites)
                if (
                    "onlyoffice" in port_map
                    and config.get("training_reward_mode", "native") == "native"
                ):
                    probe = probe_onlyoffice_api(
                        f"http://{hostname}:{port_map['onlyoffice']}"
                    )
                    (output / "verifier_preflight.json").write_text(
                        json.dumps(probe, indent=2)
                    )
                    if not probe["available"]:
                        result["verifier_diagnostics"] = {
                            "unavailable": True,
                            "preflight": probe,
                        }
                        raise VerifierUnavailable(
                            "OnlyOffice grading API did not become ready; inspect verifier_preflight.json"
                        )
                prompt, todo, _ = build_prompt(task, port_map, hostname)
                browser["allowed_origins"] = [
                    f"http://{hostname}:{port}" for port in port_map.values()
                ]
                browser_path = output / "browser.json"
                browser_path.write_text(json.dumps(browser))
                oc_config = opencode_config(
                    endpoint=config["endpoint"],
                    model=config["model"],
                    python=sys.executable,
                    browser_config=browser_path,
                    context_tokens=config["context_tokens"],
                    output_tokens=config["output_tokens"],
                    max_steps=config["max_steps"],
                    files_dir=files,
                    input_files=browser["input_files"],
                    mcp_timeout_ms=config.get("mcp_timeout_ms", 180000),
                    refresh_socket=refresh_socket,
                )
                (workspace / "opencode.json").write_text(
                    json.dumps(oc_config, indent=2)
                )
                instruction = (
                    "Complete app interactions using the saas_browser tools. "
                    "Use observe for screenshots and accessibility state. "
                    "Inspect declared input images/PDFs with inspect_input. Never use native read for PDFs; "
                    "it cannot send PDF content to this model. inspect_input returns page count and page images. "
                    "For interact use role/name or CSS/Playwright selectors, e.g. role=button name=Save "
                    'or selector=\'text="Save"\'. Accessibility descriptions such as button "Save" are not selectors. '
                    "With multiple apps, page tools require expected_url for the intended app, selecting its tab atomically. "
                    "Open only assigned root URLs with navigate, and follow internal links through UI clicks. "
                    "Use observation_id and snapshot_next_offset to read long snapshots in chunks. "
                    "If a selector fails, inspect the fresh error observation and find_elements before retrying. "
                    "Do not assume an accessible role/name for unlabeled icons; inspect their attributes. "
                    "Tool descriptions remain authoritative after compaction; do not invent tool restrictions. "
                    "After compaction, get a fresh observe before acting on visual state. "
                    "Use frame_path selectors for iframe content; observe lists frame paths. "
                    "Use coordinate actions only with a current observation_id and visible target bounds. "
                    "A successful action with observation_errors must not be repeated merely to recover its screenshot. "
                    "Input tools accept basenames or declared paths. "
                    "Use native read/write/edit tools for task working files in "
                    + str(files)
                    + "; todo.md is prefilled there. Native todo tools are also available. "
                    "Report completion in your final response.\n\n" + prompt
                )
                if browser["input_files"]:
                    instruction += "\nDeclared input files:\n" + "\n".join(
                        "- " + path for path in browser["input_files"]
                    )
                (output / "prompt.txt").write_text(instruction)
                (output / "todo.md").write_text(todo)
                (files / "todo.md").write_text(todo)
                with (
                    (output / "events.jsonl").open("w") as stdout,
                    (output / "opencode.stderr").open("w") as stderr,
                ):
                    process = subprocess.Popen(
                        [
                            command,
                            "run",
                            "--format",
                            "json",
                            "--model",
                            f"nemo/{config['model']}",
                            instruction,
                        ],
                        cwd=workspace,
                        env=agent_environment(
                            workspace,
                            root,
                            config["output_tokens"],
                            refresh_socket=refresh_socket,
                        ),
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=True,
                    )
                    try:
                        process.wait(timeout=config["agent_timeout"])
                        timed_out = False
                    except subprocess.TimeoutExpired:
                        timed_out = True
                    finally:
                        stop_process(process)
                events, text = parse_events(
                    (output / "events.jsonl").read_text(errors="replace")
                )
                recovered = context_overflow_recovered(events)
                health = execution_health(events, timed_out=timed_out)
                (output / "execution_health.json").write_text(
                    json.dumps(health, indent=2)
                )
                result.update(
                    execution_health=health,
                    agent_output=text,
                    trajectory=events,
                    native_trajectory=events,
                    recovered_context_overflow=recovered,
                    timed_out=timed_out,
                    returncode=process.returncode,
                    slot_id=slot_id,
                    port_map=port_map,
                    source_sha256=task["source_sha256"],
                    opencode_version=actual_version,
                )
                # A permission abort after real activity still has a verifiable app
                # state. Keep it explicitly incomplete, rather than losing grading.
                permission_abort = any(
                    e.get("type") == "error" and "doom_loop" in json.dumps(e)
                    for e in events
                ) and any(e.get("type") in {"tool_use", "step_finish"} for e in events)
                if permission_abort:
                    health["termination"] = "permission_abort"
                    result["execution_error"] = (
                        "OpenCode aborted on doom_loop permission"
                    )
                model_endings = {
                    "timeout", "context_limit", "model_protocol_error",
                    "malformed_tool_call", "output_token_limit", "permission_abort",
                }
                if (not events and not timed_out) or (
                    any(e.get("type") == "error" for e in events)
                    and not recovered
                    and not permission_abort
                    and health["termination"] not in model_endings
                ):
                    raise RuntimeError(
                        "OpenCode returned no trajectory or an error event; inspect native artifacts"
                    )
                if (
                    process.returncode != 0
                    and not timed_out
                    and not (process.returncode == 1 and recovered)
                    and not permission_abort
                    and health["termination"] not in model_endings
                ):
                    raise RuntimeError(f"OpenCode exited {process.returncode}")
                if config.get("training_reward_mode", "native") == "llm_judge":
                    # Training-only opt-in: the Gym agent scores retained evidence.
                    # No native verifier is executed, and this placeholder is masked.
                    result.update(
                        status="completed",
                        verify={"status": "UNJUDGED"},
                        reward=0.0,
                        mask_sample=True,
                        training_judge_pending=True,
                    )
                    return result
                if config.get("training_reward_mode", "native") != "native":
                    raise ValueError("Unknown training reward mode")
                verify = run_verify(
                    task, slot_id, port_map, hostname, str(output), run_suffix="_r0"
                )
                diagnostics = verifier_diagnostics(verify)
                result["verifier_diagnostics"] = diagnostics
                health["task_success"] = (
                    None
                    if diagnostics["unavailable"]
                    else verify.get("status") == "PASS"
                )
                if (
                    not diagnostics["unavailable"]
                    and verify.get("status") == "FAIL"
                    and health["termination"] == "final_response"
                ):
                    health["termination"] = "task_incomplete"
                    health["execution_healthy"] = False
                (output / "execution_health.json").write_text(
                    json.dumps(health, indent=2)
                )
                result.update(
                    verify=verify,
                    reward=0.0
                    if diagnostics["unavailable"]
                    else 0.0 if health["termination"] in model_endings else measured_score(verify),
                    status="completed",
                    mask_sample=diagnostics["unavailable"],
                    failure_kind="saas_bench:verifier"
                    if diagnostics["unavailable"]
                    else "saas_bench:model_ending"
                    if health["termination"] in model_endings
                    else None,
                )
            finally:
                stop_process(process)
                slot.stop_apps(sites)
    except (Exception, SystemExit, KeyboardInterrupt) as exc:  # noqa: BLE001 -- episode boundary persists failure artifacts
        result.update(
            status="error",
            mask_sample=True,
            failure_kind="saas_bench:verifier"
            if isinstance(exc, VerifierUnavailable)
            else "saas_bench:infrastructure",
            error=f"{type(exc).__name__}: {exc}",
        )
        (output / "error.log").write_text(traceback.format_exc())
    finally:
        if refresh_directory is not None:
            refresh_directory.cleanup()
        (output / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False)
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    def terminate(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit("Episode cancelled")

    signal.signal(signal.SIGTERM, terminate)
    result = execute(json.loads(args.config.read_text()))
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
