# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Uploads one gym run directory to EvalRDL. Runs in the EvalRDL exporter's own environment.

The payload comes from `EvalRDLExporter.build_payload`. Only the standard library and the `evalrdl` SDK are
imported here, so this file must not import `nemo_gym`.

The SDK is called with the arguments of `Client().uploads.upload` and reports one of four states per upload;
the exit codes below are the ones the EvalRDL command line reports for them.
"""

import sys


EXIT_ACCEPTED = 3
EXIT_REFUSED = 4


def _failed(error: str, exit_code: int, **detail) -> dict:
    return {"state": "failed", "error": error, "exit_code": exit_code, "detail": detail}


def _hint(exc: Exception) -> str:
    """What to do about a failure the user can fix. Matched by class name, so no extra import can break the worker."""
    kind = type(exc).__name__
    if kind == "ClientTooOld":
        return (
            f"The SDK in this exporter environment ({sys.prefix}) is not upgraded on its own, and the upgrade "
            "command in the message does not target it: delete that directory to rebuild it, or pin a newer version with "
            "+requirement=evalrdl==VERSION."
        )
    if kind == "AuthError" and getattr(exc, "reason", None) == "anonymous":
        return (
            "Set EVALRDL_TOKEN (or GITLAB_TOKEN or GITLAB_PRIVATE_TOKEN); the SDK falls back to `glab`, which "
            "cluster nodes usually lack."
        )
    return ""


def run(payload: dict) -> dict:
    from evalrdl.sdk.api import Client, IntakeRefused, RdlError

    client = Client(url=payload["url"]) if payload.get("url") else Client()
    try:
        results = client.uploads.upload(
            payload["run_dir"],
            label=payload.get("label"),
            tags=list(payload.get("tags") or ()),
            meta=payload.get("meta") or None,
            parallel=payload["parallel"],
            wait=payload["wait"],
            timeout=payload["timeout"],
        )
    except IntakeRefused as exc:
        return _failed(
            f"EvalRDL refused the upload at stage {exc.stage!r}: {exc.reason}",
            EXIT_REFUSED,
            evalrdl_state="refused",
            stage=exc.stage,
            reason=exc.reason,
        )
    except RdlError as exc:
        message = f"{type(exc).__name__}: {exc}"
        hint = _hint(exc)
        return _failed(f"{message.rstrip('.')}. {hint}" if hint else message, 1)

    if len(results) != 1:
        return _failed(f"Expected one upload for the run directory, EvalRDL reported {len(results)}.", 1)
    (result,) = results
    detail = {
        "evalrdl_state": result.state,
        "upload_token": result.upload_token,
        "run_id": result.run_id,
        "benchmark": result.benchmark,
        "n_rollouts": result.n_rollouts,
        "needs_binding": result.needs_binding,
    }
    if result.state == "committed":
        if result.needs_binding:
            print(
                f"EvalRDL stored the run but reports needs_binding=true (upload token {result.upload_token}).",
                file=sys.stderr,
            )
        return {"state": "succeeded", "detail": detail}
    if result.state == "deduped":
        return {"state": "skipped", "detail": detail}
    if result.state == "accepted":
        if not payload["wait"]:
            # Without waiting there is no verdict to report; the upload token in `detail` is how to look it up.
            return {"state": "succeeded", "detail": detail}
        return _failed(
            "EvalRDL stored the upload but gave no verdict before the timeout; check it with the upload token.",
            EXIT_ACCEPTED,
            **detail,
        )
    return _failed(f"Unexpected EvalRDL upload state {result.state!r}.", 1, **detail)
