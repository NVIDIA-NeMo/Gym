# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""scripts/agentenv_cleanup.py against a fake E2B SDK: dry run by default, deletes only with --apply."""

import importlib.util
import types
from pathlib import Path

import pytest


pytestmark = pytest.mark.sandbox

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "agentenv_cleanup.py"


def _load():
    spec = importlib.util.spec_from_file_location("agentenv_cleanup", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Paginator:
    def __init__(self, items: list) -> None:
        self._pages = [items]

    @property
    def has_next(self) -> bool:
        return bool(self._pages)

    async def next_items(self) -> list:
        return self._pages.pop(0)


@pytest.fixture
def cleanup(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    module = _load()
    state = types.SimpleNamespace(module=module, deleted=[], killed=[], queries=[], provider_config=None)

    class FakeProvider:
        def __init__(self, **config: object) -> None:
            state.provider_config = config

        async def list_snapshots(self, prefix: str) -> list[types.SimpleNamespace]:
            return [types.SimpleNamespace(snapshot_id=f"{prefix}snap:default", name=f"{prefix}snap")]

        async def delete_snapshot(self, snapshot_id: str) -> None:
            state.deleted.append(snapshot_id)

    class FakeAsyncSandbox:
        @staticmethod
        def list(query):
            state.queries.append(query.metadata)
            # The last one is what a service that ignored the metadata filter would also hand over.
            return _Paginator(
                [
                    types.SimpleNamespace(sandbox_id="sbx-2", metadata={"purpose": "r1"}),
                    types.SimpleNamespace(sandbox_id="sbx-1", metadata={"purpose": "r1", "user": "me"}),
                    types.SimpleNamespace(sandbox_id="sbx-other", metadata={}),
                ]
            )

        @staticmethod
        async def kill(sandbox_id: str) -> bool:
            state.killed.append(sandbox_id)
            return True

    sdk = types.SimpleNamespace(
        AsyncSandbox=FakeAsyncSandbox, SandboxQuery=lambda metadata: types.SimpleNamespace(metadata=metadata)
    )
    monkeypatch.setattr(module, "E2BProvider", FakeProvider)
    monkeypatch.setattr(module, "_e2b_sdk", lambda: sdk)
    return state


async def test_a_dry_run_lists_and_deletes_nothing(
    cleanup: types.SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("tmpl-a:default\n\ntmpl-b:default\n")

    code = await cleanup.module.run(
        ["--snapshot-prefix", "ng-run-", "--snapshot-ids", str(ids), "--metadata", "purpose=r1"]
    )

    assert code == 0 and cleanup.deleted == [] and cleanup.killed == []
    out = capsys.readouterr().out
    assert "3 snapshot(s), 2 sandbox(es)" in out and "dry run" in out
    assert cleanup.queries == [{"purpose": "r1"}]


async def test_apply_deletes_every_selected_snapshot_and_sandbox(cleanup: types.SimpleNamespace) -> None:
    code = await cleanup.module.run(["--snapshot-prefix", "ng-run-", "--metadata", "purpose=r1", "--apply"])

    assert code == 0
    assert cleanup.deleted == ["ng-run-snap:default"]
    # Only the sandboxes that carry the label, whatever the service listed.
    assert sorted(cleanup.killed) == ["sbx-1", "sbx-2"]
    # Names on, or the provider would find no snapshot under a prefix.
    assert cleanup.provider_config == {"snapshots": {"send_names": True}}


def test_nothing_selected_is_an_error(cleanup: types.SimpleNamespace) -> None:
    with pytest.raises(SystemExit):
        cleanup.module.parse_args([])
    with pytest.raises(SystemExit):
        cleanup.module.parse_args(["--metadata", "no-equals"])


@pytest.mark.parametrize("arguments", [["--snapshot-prefix", ""], ["--metadata", "=x"], ["--metadata", "k="]])
def test_a_selection_that_would_match_everything_is_an_error(
    cleanup: types.SimpleNamespace, arguments: list[str]
) -> None:
    with pytest.raises(SystemExit):
        cleanup.module.parse_args(arguments)
