# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import TextIO

from tqdm.asyncio import tqdm as _tqdm


class tqdm(_tqdm):
    """Async progress bars whose console writes tolerate reentrant collection."""

    @classmethod
    @contextmanager
    def external_write_mode(cls, file: TextIO | None = None, nolock: bool = False) -> Iterator[None]:
        # Clear a strong snapshot: stream callbacks and cyclic finalizers may
        # explicitly close bars and mutate tqdm's weak registry, even under its
        # reentrant write lock. Do not replace the shared registry itself.
        fp = file if file is not None else sys.stdout
        with nullcontext() if nolock else cls.get_lock():
            cleared = []
            for bar in list(cls._instances):
                if hasattr(bar, "start_t") and (
                    bar.fp == fp or all(stream in (sys.stdout, sys.stderr) for stream in (fp, bar.fp))
                ):
                    bar.clear(nolock=True)
                    cleared.append(bar)
            try:
                yield
            finally:
                for bar in cleared:
                    bar.refresh(nolock=True)
