# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The run summariser: upstream's per-sweep numbers and Welch against a published summary."""

import json
import sys
from pathlib import Path

import pytest
from scipy import stats


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_runs as sr  # noqa: E402


def _row(top1, top10, tani, answered=1.0, status="scored", **extra):
    return dict(top1=top1, top10=top10, tanimoto_top1=tani, answered=answered, status=status, **extra)


def test_per_run_metrics_match_upstream_definitions() -> None:
    rows = [_row(1.0, 1.0, 1.0), _row(0.0, 1.0, 0.5), _row(0.0, 0.0, None, answered=0.0, status="format_fail")]
    m = sr.per_run_metrics(rows)
    assert m["top1"] == pytest.approx(100 / 3) and m["top10"] == pytest.approx(200 / 3)
    # Tanimoto is conditional: the unanswered row is not in the denominator.
    assert m["tanimoto"] == pytest.approx(0.75) and m["tanimoto_n"] == 2 and m["answered"] == 2
    assert m["statuses"] == {"format_fail": 1, "scored": 2}


def test_welch_matches_scipy_from_summaries() -> None:
    a = [27.6, 25.7, 29.5]
    b = [22.0, 24.0, 20.5]
    ours = sr.welch(sum(a) / 3, stats.tstd(a), 3, sum(b) / 3, stats.tstd(b), 3)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert ours["t"] == pytest.approx(ref.statistic) and ours["p_two_sided"] == pytest.approx(ref.pvalue)
    assert ours["df"] == pytest.approx(ref.df)


def test_welch_is_none_when_both_spreads_are_zero() -> None:
    assert sr.welch(1.0, 0.0, 3, 2.0, 0.0, 3) is None


def test_main_summarises_files_and_attaches_published(tmp_path) -> None:
    files = []
    for i, top1 in enumerate((1.0, 0.0, 1.0)):
        p = tmp_path / f"run{i}.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in (_row(top1, 1.0, 0.8), _row(0.0, 0.0, None, 0.0))) + "\n")
        files.append(str(p))
    report = sr.main(
        [
            "--run",
            files[0],
            "--run",
            files[1],
            "--run",
            files[2],
            "--published",
            "top1=40,5",
            "--output",
            str(tmp_path / "s.json"),
        ]
    )
    s = report["summary"]
    assert s["top1"]["runs"] == [50.0, 0.0, 50.0] and s["top1"]["published"] == {"mean": 40.0, "sd": 5.0, "n_runs": 3}
    assert s["top1"]["welch"]["p_two_sided"] < 1 and "welch" not in s["top10"]
    assert (tmp_path / "s.json").exists()
