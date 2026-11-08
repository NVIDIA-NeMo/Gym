# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select the replay row from the pinned private partner_if registry release."""

from pathlib import Path

from benchmarks.partner_if import prepare as registry


OUTPUT_FPATH = Path(__file__).parent / "data/ngc_replay/partner_if_ngc_replay.jsonl"


def is_prepared_data_current(fpath: Path) -> bool:
    """A matching row cannot hide an old or incomplete replay asset package."""
    try:
        return (
            registry.is_prepared_data_current(registry.OUTPUT_FPATH)
            and Path(fpath).read_bytes() == registry.REPLAY_FPATH.read_bytes()
        )
    except OSError:
        return False


def prepare(task_ids: list[str] | None = None) -> Path:
    if task_ids is not None and task_ids != ["pif_029"]:
        raise ValueError("The replay profile selects only pif_029")
    registry.prepare()
    if not OUTPUT_FPATH.is_file():
        raise ValueError("Registry release is missing its declared replay row")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
