# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission of independent repeats after the collector settles the preceding attempt."""

from collections import defaultdict, deque
from collections.abc import Callable, Hashable, Iterable
from operator import itemgetter

from nemo_gym.config.global_config import ROLLOUT_INDEX_KEY_NAME


class RepeatAdmission:
    """Keep at most one repeat per task and agent eligible for dispatch.

    Each group initially offers its lowest pending rollout index. The collector settles
    each admitted row after recording its result, releasing its next repeat or retiring
    the rest. Independent groups stay eligible throughout.
    """

    def __init__(self, rows: Iterable[dict], group_key: Callable[[dict], Hashable]):
        self._group_key = group_key
        groups = defaultdict(list)
        for row in rows:
            groups[group_key(row)].append(row)
        self._remaining = {
            key: deque(sorted(siblings, key=itemgetter(ROLLOUT_INDEX_KEY_NAME))) for key, siblings in groups.items()
        }
        self._ready = deque(self._remaining)
        self.skipped = 0

    def take(self) -> dict | None:
        """Return the next eligible row, or None while all unfinished groups await a result."""
        if self._ready:
            return self._remaining[self._ready.popleft()].popleft()
        return None

    def settle(self, row: dict, *, solved: bool) -> list[dict]:
        """Release one more repeat or return the unstarted siblings for durable omission."""
        key = self._group_key(row)
        remaining = self._remaining[key]
        if remaining and not solved:
            self._ready.append(key)
            return []
        del self._remaining[key]
        self.skipped += len(remaining)
        return list(remaining)
