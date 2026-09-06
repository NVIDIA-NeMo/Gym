# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


PACKAGE = Path(__file__).parents[1] / "hsg" / "checkpoint_e2e"
EXISTING_LAUNCHER = PACKAGE / "run_existing_judge.sh"
EXISTING_CONTROLLER = PACKAGE / "existing_judge_controller.sbatch"
MARS_HELPER = PACKAGE / "mars_node_local.sh"
JUDGE_PROGRESS_HELPER = PACKAGE / "judge_progress.sh"
JUDGE_PORTS_HELPER = PACKAGE / "judge_ports.sh"


def _write(path: Path, text: str = "fixture\n", *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)
    return path


def _bootstrap_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, dict[str, str]]:
    package = tmp_path / "package"
    fake_bin = tmp_path / "fake-bin"
    args_dir = tmp_path / "sbatch-args"
    package.mkdir()
    fake_bin.mkdir()
    args_dir.mkdir()

    safe_path = f"{fake_bin}:{os.environ['PATH']}"
    launcher = EXISTING_LAUNCHER.read_text(encoding="utf-8")
    launcher = re.sub(r"^SAFE_PATH=.*$", f"SAFE_PATH={shlex.quote(safe_path)}", launcher, count=1, flags=re.MULTILINE)
    _write(package / "run_existing_judge.sh", launcher, executable=True)
    _write(package / "slurm_receipts.sh", (PACKAGE / "slurm_receipts.sh").read_text(encoding="utf-8"))
    _write(package / "VERSION", "1.4.13\n")
    _write(package / "run_checkpoint_e2e.sh", "#!/usr/bin/env bash\nexit 0\n", executable=True)
    _write(package / "existing_judge_bootstrap.sbatch", "#!/usr/bin/env bash\nexit 0\n")
    _write(
        package / "prepare_existing_campaign.py",
        """from __future__ import annotations
import json
import sys

action = sys.argv[1]
if action == "identify":
    print(json.dumps({"import_id": "import-" + "a" * 24, "source": "fixture"}, sort_keys=True))
elif action == "identify-package":
    print(json.dumps({"inventory_sha256": "b" * 64}, sort_keys=True))
elif action == "verify":
    pass
else:
    raise SystemExit(f"unexpected action: {action}")
""",
    )
    _write(
        package / "campaign.py",
        """from __future__ import annotations
import sys
from pathlib import Path

assert sys.argv[1] == "locate"
campaign_root = Path(sys.argv[sys.argv.index("--campaign-root") + 1])
run_id = "fixture-" + "c" * 16
print(f"RUN_ID={run_id}")
print(f"RUN_DIR={campaign_root / run_id}")
""",
    )

    counter = _write(tmp_path / "job-counter", "900\n")
    _write(
        fake_bin / "sbatch",
        """#!/usr/bin/env bash
set -euo pipefail
next=$(( $(<"$FAKE_JOB_COUNTER") + 1 ))
printf '%s\n' "$next" > "$FAKE_JOB_COUNTER"
printf '%s\n' "$@" > "$FAKE_SBATCH_ARGS/$next.args"
printf '%s\n' "$next"
""",
        executable=True,
    )
    _write(
        fake_bin / "squeue",
        """#!/usr/bin/env bash
if [[ -n ${FAKE_ADOPT_JOB:-} && " $* " == *" -n "* ]]; then
    printf '%s\n' "$FAKE_ADOPT_JOB"
fi
""",
        executable=True,
    )
    _write(
        fake_bin / "scontrol",
        """#!/usr/bin/env bash
printf 'JobId=%s JobName=fixture Comment=%s State=RUNNING\n' "$FAKE_ADOPT_JOB" "$FAKE_ADOPT_COMMENT"
""",
        executable=True,
    )
    _write(fake_bin / "sacct", "#!/usr/bin/env bash\nexit 0\n", executable=True)

    checkpoint = tmp_path / "checkpoint"
    source = tmp_path / "deliverables"
    checkpoint.mkdir()
    source.mkdir()
    owner = tmp_path / "owner"
    aav2 = owner / "gdpval_colo" / "aav2"
    dataset = _write(aav2 / "dataset.jsonl", "{}\n")
    reference_overlay = _write(aav2 / "reference.yaml", "reference: one\n")
    env_file = _write(aav2 / "aav2.env", "JUDGE_API_KEY=fixture\n")
    existing_root = aav2 / "existing"
    environment = {
        **os.environ,
        "CHECKPOINT_E2E_PYTHON": sys.executable,
        "CHECKPOINT_E2E_OWNER_ROOT": str(owner),
        "CHECKPOINT_E2E_AAV2_ROOT": str(aav2),
        "CHECKPOINT_E2E_DATASET": str(dataset),
        "CHECKPOINT_E2E_REFERENCE_OVERLAY": str(reference_overlay),
        "CHECKPOINT_E2E_ENV_FILE": str(env_file),
        "CHECKPOINT_E2E_EXISTING_ROOT": str(existing_root),
        "CHECKPOINT_E2E_MODEL_NAME": "fixture-model",
        "CHECKPOINT_E2E_SLURM_ADOPTION_GRACE_SECONDS": "1",
        "CHECKPOINT_E2E_SLURM_ADOPTION_POLL_SECONDS": "1",
        "FAKE_JOB_COUNTER": str(counter),
        "FAKE_SBATCH_ARGS": str(args_dir),
    }
    return package, checkpoint, source, reference_overlay, env_file, environment


def _run_bootstrap(
    package: Path, checkpoint: Path, source: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(package / "run_existing_judge.sh"), "bootstrap", str(checkpoint), str(source)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _state_dir(result: subprocess.CompletedProcess[str]) -> Path:
    return Path(
        next(line.split("=", 1)[1] for line in result.stdout.splitlines() if line.startswith("BOOTSTRAP_STATE_DIR="))
    )


def test_existing_judge_controller_receives_execution_package() -> None:
    launcher = EXISTING_LAUNCHER.read_text(encoding="utf-8")
    controller = EXISTING_CONTROLLER.read_text(encoding="utf-8")
    lines = launcher.splitlines()
    controller_index = next(
        index for index, line in enumerate(lines) if '"$ACTIVE_PACKAGE/existing_judge_controller.sbatch"' in line
    )

    assert ': "${CHECKPOINT_E2E_EXECUTION_PACKAGE:?set CHECKPOINT_E2E_EXECUTION_PACKAGE}"' in controller
    assert 'CHECKPOINT_E2E_EXECUTION_PACKAGE="$ACTIVE_PACKAGE"' in lines[controller_index - 1]
    assert 'CHECKPOINT_E2E_JUDGE_NO_PROGRESS_SECONDS="$judge_no_progress_seconds"' in launcher

    full_controller = (PACKAGE / "controller.sbatch").read_text(encoding="utf-8")
    assert 'source "$E2E_DIR/judge_progress.sh"' in full_controller
    assert 'gdpval_judge_progress_signature "$output" "$journal" "$failures" "$JUDGE_CACHE_ROOT"' in full_controller
    assert 'find "$JUDGE_CACHE_ROOT"' not in full_controller

    finish = (PACKAGE / "finish_existing_prepare.sbatch").read_text(encoding="utf-8")
    assert 'CHECKPOINT_E2E_JUDGE_NO_PROGRESS_SECONDS="$JUDGE_NO_PROGRESS_SECONDS"' in finish


def test_existing_judge_progress_tracks_exact_assignment_caches(tmp_path: Path) -> None:
    output = tmp_path / "judge output.jsonl"
    journal = tmp_path / "judge journal.jsonl"
    failures = tmp_path / "judge output_failures.jsonl"
    candidate = tmp_path / "candidate view"

    def signature() -> str:
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; gdpval_judge_progress_signature "$2" "$3" "$4" "$5"',
                "judge-progress-test",
                str(JUDGE_PROGRESS_HELPER),
                str(output),
                str(journal),
                str(failures),
                str(candidate),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return result.stdout.strip()

    assert signature() == "0:0:0:0"
    output.write_text("abc", encoding="utf-8")
    journal.write_text("12345", encoding="utf-8")
    task_one = candidate / "task_one"
    task_one.mkdir(parents=True)
    first_cache = task_one / "repeat_0_verify_response_0123456789ab.json"
    first_cache.write_text("{}\n", encoding="utf-8")
    assert signature() == "3:5:0:1"

    failures.write_text("{}\n", encoding="utf-8")
    assert signature() == "3:5:3:1"

    # Rewriting one slot is not another completed assignment and therefore
    # cannot reset the watchdog indefinitely.
    first_cache.write_text('{"replacement": true}\n', encoding="utf-8")
    assert signature() == "3:5:3:1"

    # Only direct, regular, exact-assignment cache names are progress. Legacy,
    # malformed, nested, and symlinked artifacts must not keep a stuck job alive.
    (task_one / "repeat_0_verify_response.json").write_text("{}\n", encoding="utf-8")
    (task_one / "repeat_x_verify_response_0123456789ab.json").write_text("{}\n", encoding="utf-8")
    nested = task_one / "repeat_0"
    nested.mkdir()
    (nested / "repeat_1_verify_response_0123456789ab.json").write_text("{}\n", encoding="utf-8")
    (task_one / "repeat_2_verify_response_0123456789ab.json").symlink_to(first_cache)
    assert signature() == "3:5:3:1"

    task_two = candidate / "task_two"
    task_two.mkdir()
    (task_two / "repeat_3_verify_response_0123456789abcdef.json").write_text("{}\n", encoding="utf-8")
    assert signature() == "3:5:3:2"


def test_bootstrap_scopes_and_forwards_judge_watchdog_override(tmp_path: Path) -> None:
    package, checkpoint, source, _reference_overlay, _env_file, environment = _bootstrap_fixture(tmp_path)

    default = _run_bootstrap(package, checkpoint, source, environment)
    assert default.returncode == 0, (default.stdout, default.stderr)
    overridden = _run_bootstrap(
        package,
        checkpoint,
        source,
        {**environment, "CHECKPOINT_E2E_JUDGE_NO_PROGRESS_SECONDS": "7200"},
    )
    assert overridden.returncode == 0, (overridden.stdout, overridden.stderr)
    assert _state_dir(overridden) != _state_dir(default)
    export_argument = next(
        line
        for line in (Path(environment["FAKE_SBATCH_ARGS"]) / "902.args").read_text(encoding="utf-8").splitlines()
        if line.startswith("--export=")
    )
    assert "CHECKPOINT_E2E_JUDGE_NO_PROGRESS_SECONDS=7200" in export_argument

    bootstrap = (PACKAGE / "existing_judge_bootstrap.sbatch").read_text(encoding="utf-8")
    assert 'CHECKPOINT_E2E_JUDGE_NO_PROGRESS_SECONDS="$CHECKPOINT_E2E_JUDGE_NO_PROGRESS_SECONDS"' in bootstrap


def test_bootstrap_preserves_reference_inputs_and_scopes_adoption_to_their_identity(tmp_path: Path) -> None:
    package, checkpoint, source, reference_overlay, env_file, environment = _bootstrap_fixture(tmp_path)

    first = _run_bootstrap(package, checkpoint, source, environment)
    assert first.returncode == 0, (first.stdout, first.stderr)
    first_state = _state_dir(first)
    assert (Path(environment["FAKE_JOB_COUNTER"])).read_text(encoding="utf-8") == "901\n"
    export_argument = next(
        line
        for line in (Path(environment["FAKE_SBATCH_ARGS"]) / "901.args").read_text(encoding="utf-8").splitlines()
        if line.startswith("--export=")
    )
    overlay_sha = hashlib.sha256(reference_overlay.read_bytes()).hexdigest()
    assert f"CHECKPOINT_E2E_REFERENCE_OVERLAY={reference_overlay}" in export_argument
    assert f"CHECKPOINT_E2E_ENV_FILE={env_file}" in export_argument
    assert f"EXPECTED_REFERENCE_OVERLAY_SHA256={overlay_sha}" in export_argument

    # Simulate a disconnect after Slurm accepted the job but before its receipt
    # was durably published. The identical invocation must adopt that exact job.
    receipt = first_state / "submissions" / "attempt_1.jobid"
    intent = first_state / "submissions" / ".slurm_submit_intents" / "attempt_1.jobid.intent"
    receipt.unlink()
    (first_state / "BOOTSTRAP.jobid").unlink()
    comment = next(
        line.split("=", 1)[1]
        for line in intent.read_text(encoding="utf-8").splitlines()
        if line.startswith("comment=")
    )
    adopted = _run_bootstrap(
        package,
        checkpoint,
        source,
        {**environment, "FAKE_ADOPT_JOB": "901", "FAKE_ADOPT_COMMENT": comment},
    )
    assert adopted.returncode == 0, (adopted.stdout, adopted.stderr)
    assert _state_dir(adopted) == first_state
    assert "ADOPTED_SLURM_JOB role=existing-bootstrap-a1 job=901" in adopted.stderr
    assert Path(environment["FAKE_JOB_COUNTER"]).read_text(encoding="utf-8") == "901\n"

    # Scientific overlay content, overlay location, and environment-file
    # location each select a fresh bootstrap identity instead of adopting work
    # launched with different inputs.
    reference_overlay.write_text("reference: changed\n", encoding="utf-8")
    changed_content = _run_bootstrap(package, checkpoint, source, environment)
    assert changed_content.returncode == 0, (changed_content.stdout, changed_content.stderr)
    assert _state_dir(changed_content) != first_state

    overlay_copy = _write(reference_overlay.parent / "reference-copy.yaml", reference_overlay.read_text())
    changed_overlay_path = _run_bootstrap(
        package,
        checkpoint,
        source,
        {**environment, "CHECKPOINT_E2E_REFERENCE_OVERLAY": str(overlay_copy)},
    )
    assert changed_overlay_path.returncode == 0, (changed_overlay_path.stdout, changed_overlay_path.stderr)
    assert _state_dir(changed_overlay_path) not in {first_state, _state_dir(changed_content)}

    env_copy = _write(env_file.parent / "alternate.env", env_file.read_text())
    changed_env_path = _run_bootstrap(
        package,
        checkpoint,
        source,
        {
            **environment,
            "CHECKPOINT_E2E_REFERENCE_OVERLAY": str(overlay_copy),
            "CHECKPOINT_E2E_ENV_FILE": str(env_copy),
        },
    )
    assert changed_env_path.returncode == 0, (changed_env_path.stdout, changed_env_path.stderr)
    assert _state_dir(changed_env_path) != _state_dir(changed_overlay_path)
    assert Path(environment["FAKE_JOB_COUNTER"]).read_text(encoding="utf-8") == "904\n"


def test_bootstrap_rejects_noncanonical_reference_input_paths(tmp_path: Path) -> None:
    package, checkpoint, source, reference_overlay, env_file, environment = _bootstrap_fixture(tmp_path)
    overlay_link = reference_overlay.parent / "reference-link.yaml"
    overlay_link.symlink_to(reference_overlay)

    linked = _run_bootstrap(
        package,
        checkpoint,
        source,
        {**environment, "CHECKPOINT_E2E_REFERENCE_OVERLAY": str(overlay_link)},
    )
    assert linked.returncode == 64
    assert "path must be an absolute resolved real file" in linked.stderr

    relative = _run_bootstrap(
        package,
        checkpoint,
        source,
        {**environment, "CHECKPOINT_E2E_ENV_FILE": env_file.name},
    )
    assert relative.returncode == 64
    assert "path must be an absolute resolved real file" in relative.stderr
    assert Path(environment["FAKE_JOB_COUNTER"]).read_text(encoding="utf-8") == "900\n"


def test_node_local_cache_marker_read_is_idempotent(tmp_path: Path) -> None:
    package_id = (PACKAGE / "MARS_PACKAGE_ID").read_text(encoding="utf-8").strip()
    helper = MARS_HELPER.read_text(encoding="utf-8")
    assert f"MARS_PACKAGE_ID_EXPECTED={package_id}" in helper

    marker = tmp_path / ".mars-ready"
    script = r"""
set -euo pipefail
source "$1"
marker=$2
printf '%s\n' expected > "$marker"
chmod 0400 "$marker"
mars_marker_matches "$marker" expected
! mars_marker_matches "$marker" stale
ln -s "$marker" "$marker.link"
! mars_marker_matches "$marker.link" expected
"""

    subprocess.run(
        ["bash", "-c", script, "marker-test", str(MARS_HELPER), str(marker)],
        check=True,
    )
    assert '$(<"$marker" 2>/dev/null)' not in helper


def test_affected_shell_entrypoints_parse() -> None:
    subprocess.run(
        [
            "bash",
            "-n",
            str(EXISTING_LAUNCHER),
            str(EXISTING_CONTROLLER),
            str(MARS_HELPER),
            str(JUDGE_PROGRESS_HELPER),
            str(JUDGE_PORTS_HELPER),
            str(PACKAGE / "controller.sbatch"),
            str(PACKAGE / "finish_existing_prepare.sbatch"),
        ],
        check=True,
    )
