# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import io

import pytest
from tqdm import tqdm as upstream_tqdm

from nemo_gym.progress_utils import tqdm


class _CollectingStream(io.StringIO):
    collect_on_write = False

    def write(self, value: str) -> int:
        if self.collect_on_write:
            self.collect_on_write = False
            gc.collect()
        return super().write(value)


@pytest.mark.parametrize("nolock", [False, True])
@pytest.mark.parametrize("bar_type", [tqdm, upstream_tqdm])
def test_write_survives_progress_bar_collection(nolock: bool, bar_type: type[upstream_tqdm]) -> None:
    """A stream write may collect abandoned bars while tqdm clears live ones."""
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    stream = _CollectingStream()
    anchor = tqdm(total=1, file=stream, disable=False, mininterval=0)
    try:
        bars = [bar_type(total=1, file=stream, disable=False, mininterval=0) for _ in range(8)]
        for bar in bars:
            bar.diagnostic_cycle = [bar]
        del bar, bars
        stream.collect_on_write = True
        tqdm.write("progress", file=stream, nolock=nolock)
        assert "progress\n" in stream.getvalue()
        anchor.update()
        assert anchor.n == 1
    finally:
        anchor.close()
        gc.collect()
        if was_enabled:
            gc.enable()


@pytest.mark.asyncio
async def test_async_completion_and_gather_preserve_results() -> None:
    async def value(index: int) -> int:
        await asyncio.sleep(0)
        return index

    completed = [await task for task in tqdm.as_completed([value(1), value(2)], disable=True)]
    assert sorted(completed) == [1, 2]
    assert await tqdm.gather(value(3), value(4), disable=True) == [3, 4]


def test_external_write_mode_propagates_errors_and_allows_later_writes() -> None:
    stream = io.StringIO()
    with tqdm(total=1, file=stream, disable=False) as progress:
        with pytest.raises(ValueError, match="failed operation"):
            with tqdm.external_write_mode(file=stream):
                raise ValueError("failed operation")
        tqdm.write("recovered", file=stream)
        progress.update()
        assert progress.n == 1
    assert "recovered\n" in stream.getvalue()


def test_rollout_helpers_share_safe_progress_class() -> None:
    from nemo_gym import rollout_collection, rollout_reverification

    assert rollout_collection.tqdm is tqdm
    assert rollout_reverification.tqdm is tqdm


@pytest.mark.parametrize("nolock", [False, True])
@pytest.mark.parametrize("bar_type", [tqdm, upstream_tqdm])
def test_write_survives_explicit_close_during_clear(monkeypatch, nolock, bar_type) -> None:
    """Keeping strong references does not prevent a callback from closing a bar."""
    stream = io.StringIO()
    bars = [bar_type(total=1, file=stream, disable=False, mininterval=0) for _ in range(4)]
    original_clear = upstream_tqdm.clear
    closed = False

    def clear_and_close(self, *args, **kwargs):
        nonlocal closed
        if not closed:
            closed = True
            bars[-1].close()
        return original_clear(self, *args, **kwargs)

    monkeypatch.setattr(upstream_tqdm, "clear", clear_and_close)
    try:
        tqdm.write("progress", file=stream, nolock=nolock)
        assert closed
        assert bars[-1].disable
        assert "progress\n" in stream.getvalue()
        tqdm.write("again", file=stream, nolock=nolock)
        assert "again\n" in stream.getvalue()
    finally:
        for bar in bars:
            bar.close()
