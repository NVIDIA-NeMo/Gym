# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``snapshots.py --retain-from``: snapshots named by retained partial-rollout checkpoints are kept."""

import asyncio
import shutil
from pathlib import Path

import pytest

from nemo_gym._checkpoint.store import write_participant_state
from nemo_gym.sandbox.providers.opensandbox import snapshots
from tests.unit_tests.test_opensandbox_snapshots import (
    BASE,
    TEST_ACCESS_KEY,
    Response,
    Session,
    install_session,
    page,
    snapshot,
)


def sandbox_state(snapshot_id: str | None) -> dict:
    return {
        "provider_name": "opensandbox",
        "descriptor": {"sandbox_id": "sb-1"},
        "snapshot_id": snapshot_id,
        "spec": {"image": "img"},
        "paused_at": 1.0,
    }


def write_checkpoints(root: Path) -> None:
    """Two checkpoints, with sandbox states nested the way each participant kind nests them."""
    write_participant_state(
        root / "c1",
        kind="resources",
        instance="notes",
        checkpoint_id="c1",
        records=[
            # A resources server whose session state is the sandbox state itself.
            {"episode_id": {"rollout_id": "r1"}, "session_id": "s1", "state": sandbox_state("snap-a")},
            # One that composes it with its own state, as the litmus agent does.
            {
                "episode_id": {"rollout_id": "r2"},
                "session_id": "s2",
                "state": {"cells": [], "sandbox": sandbox_state("snap-b")},
            },
            # A freeze-only backend keeps no snapshot.
            {"episode_id": {"rollout_id": "r3"}, "session_id": "s3", "state": sandbox_state(None)},
        ],
    )
    write_participant_state(
        root / "c2",
        kind="agent",
        instance="agent",
        checkpoint_id="c2",
        records=[
            {
                "episode_id": {"rollout_id": "r4"},
                "session_key": "k",
                "session": {"request": {}, "sandbox": sandbox_state("snap-c")},
            },
        ],
    )
    # Something else under the root that is not a participant directory.
    (root / "controller").mkdir()
    (root / "controller" / "manifest.json").write_text('{"not": "a participant manifest"}')


def run_cleanup(retain_from: Path, **overrides: object) -> int:
    options: dict = dict(
        domain="https://sandbox.example",
        protocol="http",
        access_key=TEST_ACCESS_KEY,
        sandbox_id=None,
        states=None,
        snapshot_ids=None,
        kill_paused=False,
        reap=True,
        retain_from=retain_from,
    )
    options.update(overrides)
    return asyncio.run(snapshots.cleanup_snapshots(**options))


def test_retained_snapshot_ids_reads_every_participant_record(tmp_path: Path) -> None:
    write_checkpoints(tmp_path)
    assert snapshots.retained_snapshot_ids(tmp_path) == {"snap-a", "snap-b", "snap-c"}


def test_reap_keeps_what_retained_checkpoints_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    write_checkpoints(tmp_path)
    session = Session(
        page(
            [snapshot("snap-a"), snapshot("snap-old"), snapshot("snap-b"), snapshot("snap-c"), snapshot("snap-gone")]
        ),
        page([]),
        delete_responses={
            f"{BASE}/snapshots/snap-old": Response(status=204),
            f"{BASE}/snapshots/snap-gone": Response(status=204),
        },
    )
    install_session(monkeypatch, session)

    assert run_cleanup(tmp_path) == 0

    assert session.urls("DELETE") == [f"{BASE}/snapshots/snap-old", f"{BASE}/snapshots/snap-gone"]
    out = capsys.readouterr().out
    assert "Keeping 3 snapshot(s)" in out and "Deleting snapshot snap-old" in out


def test_pruning_a_checkpoint_releases_its_snapshots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_checkpoints(tmp_path)
    shutil.rmtree(tmp_path / "c1")
    session = Session(
        page([snapshot("snap-a"), snapshot("snap-b"), snapshot("snap-c")]),
        page([]),
        delete_responses={
            f"{BASE}/snapshots/snap-a": Response(status=204),
            f"{BASE}/snapshots/snap-b": Response(status=204),
        },
    )
    install_session(monkeypatch, session)

    assert run_cleanup(tmp_path) == 0
    assert sorted(session.urls("DELETE")) == [f"{BASE}/snapshots/snap-a", f"{BASE}/snapshots/snap-b"]


def test_retain_from_fails_closed_before_any_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_checkpoints(tmp_path)
    records = next((tmp_path / "c1" / "gym" / "resources" / "notes").glob("records-*.jsonl"))
    records.write_text("not json\n")
    session = Session(page([snapshot("snap-x")]), delete_responses={f"{BASE}/snapshots/snap-x": Response(status=204)})
    install_session(monkeypatch, session)

    with pytest.raises(ValueError, match="cannot read retained checkpoint state"):
        run_cleanup(tmp_path)
    assert session.requests == []

    with pytest.raises(ValueError, match="is not a directory"):
        run_cleanup(tmp_path / "missing")
    assert session.requests == []


def test_retain_from_refuses_explicit_ids_and_kill_paused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_checkpoints(tmp_path)
    session = Session()
    install_session(monkeypatch, session)
    with pytest.raises(ValueError, match="snapshot_ids"):
        run_cleanup(tmp_path, snapshot_ids=["snap-a"])
    with pytest.raises(ValueError, match="kill_paused"):
        run_cleanup(tmp_path, kill_paused=True)
    assert session.requests == []


def test_cli_forwards_retain_from(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_checkpoints(tmp_path)
    session = Session(
        page([snapshot("snap-a"), snapshot("snap-old")]),
        page([]),
        delete_responses={f"{BASE}/snapshots/snap-old": Response(status=204)},
    )
    install_session(monkeypatch, session)

    code = snapshots.main(
        ["--domain", "https://sandbox.example", "--api-key", TEST_ACCESS_KEY, "--retain-from", str(tmp_path), "--reap"]
    )

    assert code == 0 and session.urls("DELETE") == [f"{BASE}/snapshots/snap-old"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--domain", "d", "--api-key", "k", "--retain-from", "x", "--snapshot-id", "s"],
        ["--domain", "d", "--api-key", "k", "--retain-from", "x", "--kill-paused"],
    ],
)
def test_cli_rejects_retain_from_with_explicit_ids_or_kill_paused(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        snapshots.main(argv)
