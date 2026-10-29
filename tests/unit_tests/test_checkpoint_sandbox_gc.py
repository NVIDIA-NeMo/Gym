# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandbox snapshot gc CLI releases a dead job's snapshots from its checkpoint state directory."""

from pathlib import Path

import pytest

from nemo_gym._checkpoint import sandbox_gc
from nemo_gym._checkpoint.sandbox import SandboxSessionCheckpointer
from tests.unit_tests.test_checkpoint_sandbox import SPEC, FakeSnapshotProvider


pytestmark = pytest.mark.sandbox


async def dead_job(provider: FakeSnapshotProvider, state_dir: Path) -> dict[str, SandboxSessionCheckpointer]:
    """Two servers that took snapshots for c1 and c2, then died without a final release."""
    servers = {}
    for server in ("swebench", "harbor"):
        checkpointer = SandboxSessionCheckpointer(provider, state_dir=state_dir, server_name=server)
        await checkpointer.create("s1", SPEC)
        for checkpoint_id in ("c1", "c2"):
            await checkpointer.export(["s1"], checkpoint_id=checkpoint_id)
        servers[server] = checkpointer
    return servers


async def test_collect_garbage_releases_every_server_outside_the_kept_checkpoints(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    servers = await dead_job(provider, tmp_path)
    kept = {snapshot_id for checkpointer in servers.values() for snapshot_id in checkpointer.indexed_snapshots["c2"]}

    assert sandbox_gc.server_names(tmp_path) == ["harbor", "swebench"]
    remaining = await sandbox_gc.collect_garbage(
        provider, state_dir=tmp_path, servers=["harbor", "swebench"], kept=frozenset({"c2"})
    )

    assert remaining == {"harbor": {}, "swebench": {}}
    assert set(provider.snapshots) == kept


async def test_a_dry_run_reports_and_deletes_nothing(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    await dead_job(provider, tmp_path)
    before = dict(provider.snapshots)

    remaining = await sandbox_gc.collect_garbage(
        provider, state_dir=tmp_path, servers=["swebench"], kept=frozenset(), dry_run=True
    )

    assert set(remaining["swebench"]) == {"c1", "c2"} and provider.snapshots == before


async def test_the_cli_releases_the_named_servers_and_reports_leftovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = FakeSnapshotProvider()
    servers = await dead_job(provider, tmp_path)
    loaded: list[tuple[str, str]] = []

    def load(config_path: str, provider_ref: str) -> FakeSnapshotProvider:
        loaded.append((config_path, provider_ref))
        return provider

    monkeypatch.setattr(sandbox_gc, "load_provider", load)
    # One of harbor's c1 snapshots cannot be deleted now: the CLI says so and fails.
    [stuck] = servers["harbor"].indexed_snapshots["c1"]
    provider.fail[("delete_snapshot", stuck)] = RuntimeError("HTTP 503")

    code = await sandbox_gc._main(
        sandbox_gc.argparse.Namespace(
            state_dir=str(tmp_path),
            provider_config="env.yaml",
            provider="sandbox",
            servers=None,
            keep=["c2"],
            dry_run=False,
        )
    )

    assert code == 1 and loaded == [("env.yaml", "sandbox")]
    captured = capsys.readouterr()
    assert "swebench: released" in captured.out
    assert "harbor: 1 snapshot(s) of dropped checkpoints were not deleted" in captured.err
    assert stuck in provider.snapshots


def test_the_cli_parses_its_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    async def fake_main(args: sandbox_gc.argparse.Namespace) -> int:
        seen.append(args)
        return 0

    monkeypatch.setattr(sandbox_gc, "_main", fake_main)
    assert (
        sandbox_gc.main(
            ["--state-dir", "/s", "--provider-config", "p.yaml", "--server", "a", "--server", "b", "--keep", "c9"]
        )
        == 0
    )
    [args] = seen
    assert (args.state_dir, args.provider_config, args.provider) == ("/s", "p.yaml", "sandbox")
    assert (args.servers, args.keep, args.dry_run) == (["a", "b"], ["c9"], False)
    with pytest.raises(SystemExit):
        sandbox_gc.main(["--provider-config", "p.yaml"])


def test_load_provider_resolves_the_named_block_of_a_gym_config(tmp_path: Path) -> None:
    pytest.importorskip("tenacity", reason="tenacity optional sandbox dependency is not installed")
    from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider

    config = tmp_path / "env.yaml"
    config.write_text("my_sandbox:\n  opensandbox:\n    connection:\n      domain: cell.example\n      api_key: k\n")

    provider = sandbox_gc.load_provider(str(config), "my_sandbox")

    assert isinstance(provider, OpenSandboxProvider)
