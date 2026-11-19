# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, call

from benchmarks.osworld.tools import cleanup_opensandbox_run


class _Filter:
    def __init__(self, **kwargs: object) -> None:
        self.metadata = kwargs["metadata"]
        self.page = kwargs["page"]
        self.page_size = kwargs["page_size"]


class _Manager:
    def __init__(self, pages: list[object]) -> None:
        self.pages = iter(pages)
        self.filters: list[_Filter] = []

    def list_sandbox_infos(self, sandbox_filter: _Filter) -> object:
        self.filters.append(sandbox_filter)
        return next(self.pages)


def _info(sandbox_id: str, metadata: dict[str, str]) -> object:
    return SimpleNamespace(id=sandbox_id, metadata=metadata)


def _page(infos: list[object], *, has_next_page: bool = False) -> object:
    return SimpleNamespace(
        sandbox_infos=infos,
        pagination=SimpleNamespace(has_next_page=has_next_page),
    )


def test_list_exact_ids_paginates_deduplicates_and_rechecks_metadata() -> None:
    manager = _Manager(
        [
            _page(
                [
                    _info("sandbox-b", {"run-id": "run-7"}),
                    _info("wrong-run", {"run-id": "run-70"}),
                ],
                has_next_page=True,
            ),
            _page([_info("sandbox-a", {"run-id": "run-7"})]),
            _page(
                [
                    _info("sandbox-a", {"nemo-gym.nvidia.com/run": "run-7"}),
                    _info("sandbox-c", {"nemo-gym.nvidia.com/run": "run-7"}),
                ]
            ),
        ]
    )

    assert cleanup_opensandbox_run._list_exact_ids(manager, _Filter, "run-7") == [
        "sandbox-a",
        "sandbox-b",
        "sandbox-c",
    ]
    assert [(item.metadata, item.page, item.page_size) for item in manager.filters] == [
        ({"run-id": "run-7"}, 1, 200),
        ({"run-id": "run-7"}, 2, 200),
        ({"nemo-gym.nvidia.com/run": "run-7"}, 1, 200),
    ]


def test_reap_kills_only_initial_exact_ids_and_waits_until_none(monkeypatch) -> None:
    manager = Mock()
    matches = iter(
        [
            ["sandbox-a", "sandbox-b"],
            ["sandbox-a"],
            [],
        ]
    )
    monkeypatch.setattr(
        cleanup_opensandbox_run,
        "_list_exact_ids",
        lambda *_args: next(matches),
    )
    monkeypatch.setattr(cleanup_opensandbox_run.time, "sleep", lambda _seconds: None)

    report = cleanup_opensandbox_run._reap_exact_ids(
        manager,
        _Filter,
        "run-7",
        timeout_s=10,
        poll_s=0.01,
    )

    assert manager.kill_sandbox.call_args_list == [call("sandbox-a"), call("sandbox-b")]
    assert report == {
        "run_id": "run-7",
        "matched_ids": ["sandbox-a", "sandbox-b"],
        "kill_errors": {},
        "remaining_ids": [],
        "all_gone": True,
    }
