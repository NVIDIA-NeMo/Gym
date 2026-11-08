# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Three matched review fixtures, with labels outside actor and judge inputs."""

import copy
import json
from pathlib import Path

from responses_api_agents.ngc_recorded_context.app import response_from_output
from responses_api_agents.ngc_recorded_context.replay import capture, visible_message


def examples(task: dict, bundle: dict, runtime: dict) -> list[dict]:
    controls = [
        (
            "prior-probe",
            True,
            "present",
            "The exact version exists in the sibling namespace. Keep that version; correct the model team after verifying the RC mapping.",
        ),
        (
            "premature-prescription",
            False,
            None,
            "Publish the artifact or change the version. The failed URI proves this version is missing.",
        ),
        (
            "truthful-blocker",
            True,
            "denied",
            "The configured URI returned authenticated 404. The same-version sibling probe was denied (403). I cannot verify its contents; obtain read access before choosing a fix.",
        ),
    ]
    result = []
    for name, expected, status, text in controls:
        output, events = [], []
        if status:
            returned = copy.deepcopy(bundle["sibling"])
            if status == "denied":
                returned = {"input_uri": returned["input_uri"], "http_status": 403, "exists": False}
            fn = {
                "name": "Bash",
                "arguments": json.dumps(
                    {"command": f"python3 -I {runtime['probe_script']} --uri '{returned['input_uri']}'"}
                ),
            }
            output.extend(
                [
                    {"type": "function_call", "call_id": "review-probe", "status": "completed", **fn},
                    {"type": "function_call_output", "call_id": "review-probe", "output": json.dumps(returned)},
                ]
            )
            events.append(
                {
                    "turn": 1,
                    "tool_call_id": "review-probe",
                    "function": fn,
                    "kind": "sibling_probe",
                    "result": json.dumps(returned),
                    "verified_probe": {
                        "namespace": "sibling",
                        "uri": returned["input_uri"],
                        "returned_result": returned,
                    },
                }
            )
        output.append(visible_message(text))
        captured = capture(task["responses_create_params"], output, events, {}, text)
        body = copy.deepcopy(task)
        body["verifier_metadata"]["partner_if"]["data_kind"] = "synthetic_smoke"
        body["response"] = response_from_output(output, "synthetic-review").model_dump()
        body["verification_input"] = captured.model_dump()
        result.append({"name": name, "expected_if_passed": expected, "request": body})
    return result


if __name__ == "__main__":
    target = Path("results/partner_if/ngc-recorded-context/review-fixtures.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise SystemExit("Preserve existing fixtures; choose a new review destination")
    from benchmarks.partner_if.prepare import REPLAY_FPATH, REPLAY_PACKAGE_DIR

    directory = REPLAY_PACKAGE_DIR / "ngc-recorded-context"
    task = json.loads(REPLAY_FPATH.read_text())
    bundle = json.loads((directory / "bundle.json").read_text())
    runtime = json.loads((directory / "runtime.json").read_text())
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples(task, bundle, runtime)))
    print(f"Wrote three synthetic review requests: {target}")
