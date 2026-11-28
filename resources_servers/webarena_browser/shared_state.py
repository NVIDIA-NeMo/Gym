# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initial-state evidence for VisualWebArena tasks that reuse mutable account data."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from nemo_gym.web import EvaluatorInfrastructureError, WebTask


# Runtime row indices and hashes of the maintained public population's intents.
STATE_EPOCHS = (("346", "678"), ("345", "751"), ("336",), ("335",), ("340",), ("92",))
STATE_INTENTS = {
    "92": "16cad01b727f77383732e1468c7c15c189dd4b74656eb9ef2f8fc6ff0c9424fd",  # pragma: allowlist secret
    "335": "a574c7bcbb949e48bf91009f8043ab376c66c6bb7d78a5a9bd5d8bab2778dc1d",  # pragma: allowlist secret
    "336": "5f57300e331ade7d69b63dfc681d5ed896593af64efe1e218d19080dea0ca0b7",  # pragma: allowlist secret
    "340": "f10f69e2b03f5d1efbf2e6e35ae1ff6dad55b0e5fe025596c222df1890c8527e",  # pragma: allowlist secret
    "345": "99043d0eb46501a5524e5ebdbd25e06cc33038a1fc951fe6691e93941ae24b1f",  # pragma: allowlist secret
    "346": "0b584aad170d479939d5a914cabe6909cf624ca1f71178c49e9f54b6c8ba25e8",  # pragma: allowlist secret
    "678": "ab69a3d5ba396d09917dc423adeb7dd97507ad258dc4415e0c0b5652197295e3",  # pragma: allowlist secret
    "751": "c01f00c13340d4a80daab72ebc04eac4ec22e0c906469a1881c426898caa677c",  # pragma: allowlist secret
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def validate_state_identity(task: WebTask) -> None:
    """Reject row-ID drift before applying the maintained clean-state policy."""
    expected = STATE_INTENTS.get(task.task_id)
    if task.benchmark != "visualwebarena" or expected is None:
        raise ValueError("clean-state policy requires an audited VisualWebArena task")
    if hashlib.sha256(task.intent.encode()).hexdigest() != expected:
        raise ValueError(f"clean-state task identity mismatch: {task.task_id}")


def _store_once(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Publish complete JSON atomically, preserving any existing observation."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as temporary:
        temporary_path = Path(temporary.name)
        json.dump(value, temporary, sort_keys=True, ensure_ascii=True)
    try:
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            pass
        return json.loads(path.read_text())
    finally:
        temporary_path.unlink(missing_ok=True)


def _active_epoch(root: Path, run_id: str, epoch: int) -> dict[str, Any]:
    with (root / "site.lease").open("r+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(handle, fcntl.LOCK_UN)
            raise EvaluatorInfrastructureError("clean-state coordinator no longer holds an exclusive site lease")
        record = json.load(handle)
    if record != {"run_id": run_id, "epoch": epoch, "phase": "running", "reset_verified": True}:
        raise EvaluatorInfrastructureError("clean-state epoch does not match the active site lease")
    return record


def _observe_account_state(page: Any, task: WebTask, urls: dict[str, str]) -> tuple[str, dict[str, Any]]:
    site = "shopping" if "shopping" in task.sites else "classifieds"
    if site not in task.sites or not urls.get(site):
        raise EvaluatorInfrastructureError("clean-state task has no configured audited site")
    probe = page.context.new_page()

    def visit(path: str) -> None:
        target = urls[site].rstrip("/") + path
        response = probe.goto(target, wait_until="load")
        expected, actual = urlsplit(target), urlsplit(probe.url)
        if (
            response is None
            or response.status != 200
            or (actual.scheme, actual.netloc, actual.path) != (expected.scheme, expected.netloc, expected.path)
            or parse_qs(actual.query, keep_blank_values=True) != parse_qs(expected.query, keep_blank_values=True)
        ):
            raise EvaluatorInfrastructureError("clean-state observation reached an unavailable or redirected page")

    try:
        if site == "shopping":
            visit("/customer/address/")
            state = {
                "billing": probe.locator(".box-address-billing").inner_text(),
                "shipping": probe.locator(".box-address-shipping").inner_text(),
                "addresses": probe.locator("address").all_inner_texts(),
            }
            visit("/checkout/cart/")
            state.update(
                cart_empty=probe.locator(".cart-empty").count() > 0,
                cart_items=[
                    text.strip()
                    for text in probe.locator("#shopping-cart-table tbody").all_inner_texts()
                    if text.strip()
                ],
                cart_quantities=probe.locator("#shopping-cart-table input.qty").evaluate_all(
                    "inputs => inputs.map(input => input.value)"
                ),
            )
            if state["cart_empty"] == bool(state["cart_items"]) or len(state["cart_items"]) != len(
                state["cart_quantities"]
            ):
                raise EvaluatorInfrastructureError("clean-state shopping cart observation is incomplete")
        else:
            state = {}
            for item in (84144, 84145, 84146):
                visit(f"/index.php?page=item&id={item}")
                state[str(item)] = {
                    "price": probe.locator(".price").first.inner_text(),
                    "description": probe.locator(".desc").first.inner_text(),
                }
        return site, state
    finally:
        probe.close()


def verify_initial_state(
    task: WebTask,
    page: Any,
    urls: dict[str, str],
    *,
    state_root: str | None,
    required: bool,
) -> dict[str, Any] | None:
    """Verify a protected task's initial state while its coordinator owns the site."""
    if task.benchmark != "visualwebarena":
        return None
    context = task.task_kwargs.get("shared_state")
    protected = task.benchmark == "visualwebarena" and task.task_id in STATE_INTENTS
    if context is None and not (required and protected):
        return None
    try:
        validate_state_identity(task)
        if not isinstance(context, dict) or state_root is None:
            raise ValueError("protected task requires a clean-state coordinator and configured state root")
        run_id, epoch = context.get("run_id"), context.get("epoch")
        if not isinstance(run_id, str) or not re.fullmatch("[0-9a-f]{32}", run_id):
            raise ValueError("invalid clean-state run identity")
        if (
            type(epoch) is not int
            or not 1 <= epoch <= len(STATE_EPOCHS)
            or task.task_id not in STATE_EPOCHS[epoch - 1]
        ):
            raise ValueError("task does not belong to the requested clean-state epoch")
        root = Path(state_root).expanduser().resolve()
        run_dir = root / run_id
        if run_dir.resolve().parent != root or not run_dir.is_dir():
            raise ValueError("clean-state run directory is missing or outside the configured root")
        _active_epoch(root, run_id, epoch)
        site, state = _observe_account_state(page, task, urls)
        _active_epoch(root, run_id, epoch)
        baseline = _store_once(run_dir / f"baseline-{site}.json", state)
        if baseline != state:
            raise EvaluatorInfrastructureError(f"reset did not restore the initial {site} state")
        receipt = {
            "run_id": run_id,
            "epoch": epoch,
            "task_id": task.task_id,
            "site": site,
            "state_sha256": _digest(state),
        }
        observed = _store_once(run_dir / f"epoch-{epoch}-task-{task.task_id}.json", receipt)
        if observed != receipt:
            raise EvaluatorInfrastructureError("clean-state task observation changed within its epoch")
        return receipt
    except EvaluatorInfrastructureError:
        raise
    except Exception as exc:
        raise EvaluatorInfrastructureError(f"clean-state verification failed: {exc}") from exc
