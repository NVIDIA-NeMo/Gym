# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit a saved API attempt offline; never samples an actor or assigns a score."""

import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path

from resources_servers.partner_if.native_evidence import visible_events
from responses_api_agents.ngc_recorded_context.replay import ReplayConfig, actor_identity, execute, write_json


async def audit_attempt(source: Path, output: Path, package_dir: Path) -> dict:
    params = json.loads((package_dir / "ngc-recorded-context/actor-params.json").read_text())
    paths = sorted(source.glob("call-*.request.json"))
    if (source / "api-order.jsonl").exists():
        paths = [
            source / json.loads(line)["request"] for line in (source / "api-order.jsonl").read_text().splitlines()
        ]
        if any(not path.resolve().is_relative_to(source.resolve()) for path in paths):
            raise ValueError("Saved API call order escapes the capture")
    if not paths:
        raise ValueError("No saved calls")
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    consumed = 0

    async def request_api(payload, timeout_s):
        nonlocal consumed
        if consumed >= len(paths):
            raise AssertionError("Adapter attempted an additional model call")
        path = paths[consumed]
        if json.loads(path.read_text()) != payload:
            raise AssertionError("Reconstructed actor request differs from saved capture")
        prefix = path.name.removesuffix("request.json")
        consumed += 1
        receipt = json.loads((path.parent / (prefix + "receipt.json")).read_text())
        return receipt["http_status"], json.loads((path.parent / (prefix + "response.json")).read_text())

    # Gym may omit completed message-status metadata. Preserve the exact input
    # captured by the original execution, after checking semantic identity.
    params = copy.deepcopy(params)
    capture_path = source / "verification-input.json"
    if capture_path.exists():
        recorded_input = json.loads(capture_path.read_text())["input"]
        if actor_identity({"input": recorded_input}) != actor_identity({"input": params["input"]}):
            raise ValueError("Saved actor input differs from the canonical history")
        params["input"] = recorded_input
    model = json.loads(paths[0].read_text())["model"]
    identity_path = source / "identity.json"
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    settings = identity.get("settings", {})
    session_id = (identity.get("session_metadata") or {}).get("session_id")

    async def review_prefix(output, captured, turn):
        original_input = json.loads((source / f"prefix-{turn:02d}.input.json").read_text())
        if original_input["capture"] != captured.model_dump():
            raise ValueError("Saved prefix capture differs")
        if visible_events(original_input["output"]) != visible_events(output):
            raise ValueError("Saved prefix visible actions differ")
        result_path = source / f"prefix-{turn:02d}.grading-result.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        raise ValueError("Saved prefix judge failed; no new judge call is permitted during audit")

    root, _, captured, diagnostics = await execute(
        ReplayConfig(**settings, package_dir=package_dir, artifacts_dir=output),
        params,
        model=model,
        secret="",
        request_api=request_api,
        replay_id=session_id,
        review_prefix=review_prefix if settings.get("evaluation_boundary") == "ordering_prefix_v1" else None,
    )
    if consumed != len(paths) or diagnostics["termination"] in {"provider_error", "capture_error"}:
        raise ValueError("Saved actor could not be reconstructed exactly; see adapter capture")
    original_events = json.loads((source / "events.json").read_text())
    actual_events = json.loads((root / "events.json").read_text())
    stripped = [{k: v for k, v in event.items() if k != "verified_probe"} for event in actual_events]
    comparable = actual_events if any("verified_probe" in event for event in original_events) else stripped
    if original_events != comparable:
        raise ValueError("Replayed tools differ from the saved tool receipts")
    if json.loads((source / "drafts.json").read_text()) != json.loads((root / "drafts.json").read_text()):
        raise ValueError("Reconstructed new drafts differ")
    if session_id is not None:
        if (source / "session-transcript.jsonl").read_bytes() != (root / "session-transcript.jsonl").read_bytes():
            raise ValueError("Reconstructed session transcript differs")
    for path, expected in before.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Source evidence changed")
    result = {
        "source": str(source),
        "model": model,
        "calls_reconstructed": consumed,
        "source_checksums": before,
        "source_unchanged": True,
        "new_actor_requests": 0,
        "adapter_artifacts": str(root),
        **diagnostics,
        "judge_called": False,
        "semantic_score": None,
    }
    write_json(root / "saved-audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=Path("benchmarks/partner_if/data/ngc_replay/registry"))
    args = parser.parse_args()
    result = asyncio.run(audit_attempt(args.source, args.output, args.package))
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("calls_reconstructed", "termination", "new_sibling_probe_count", "source_unchanged")
            }
        )
    )


if __name__ == "__main__":
    main()
