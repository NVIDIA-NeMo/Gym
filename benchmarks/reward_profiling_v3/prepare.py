"""No-op prepare() for reward-profiling datasets that already exist as jsonl files.

`gym eval prepare` imports every dataset's prepare module; with
+use_cached_prepared_benchmarks=true and the jsonl present it never calls prepare().
If it is called anyway, return the configured jsonl path untouched.
"""

from pathlib import Path


def prepare(jsonl_fpath: str | None = None, **_: object) -> Path:
    if jsonl_fpath is None:
        raise RuntimeError("reward_profiling_v3.prepare: the dataset jsonl must already exist (jsonl_fpath)")
    return Path(jsonl_fpath)
