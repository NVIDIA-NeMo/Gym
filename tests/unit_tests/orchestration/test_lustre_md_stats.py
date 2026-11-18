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

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.orchestration.executors.slurm.lustre_md_stats import analyze, print_lustre_md_stats, stats_files


def _write(path: Path, rows: list[tuple]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("\t".join(str(field) for field in row) + "\n" for row in rows))
    return path


def _interval(report: dict, name: str) -> dict:
    [interval] = [item for item in report["intervals"] if item["interval"] == name]
    return interval


# One node, four phases: 100 s to services, 1000 s of driver, 10 s to the end.
_ONE_NODE = [
    ("start", 1000, "n1", "MDT0000", "close", 50),
    ("start", 1000, "n1", "MDT000b", "intent_lock", 1000),
    ("services_ready", 1100, "n1", "MDT0000", "close", 150),
    ("services_ready", 1100, "n1", "MDT000b", "intent_lock", 3000),
    ("driver_end", 2100, "n1", "MDT0000", "close", 1150),
    ("driver_end", 2100, "n1", "MDT000b", "intent_lock", 503000),
    ("driver_end", 2100, "n1", "MDT000b", "getattr", 7000),
    ("end", 2110, "n1", "MDT0000", "close", 1160),
    ("end", 2110, "n1", "MDT000b", "intent_lock", 503010),
    ("end", 2110, "n1", "MDT000b", "getattr", 7000),
]


def test_each_interval_is_the_difference_of_its_two_snapshots(tmp_path):
    report = analyze(_write(tmp_path / "s.tsv", _ONE_NODE))

    assert report["phases"] == ["start", "services_ready", "driver_end", "end"]
    assert [item["interval"] for item in report["intervals"]] == [
        "start->services_ready",
        "services_ready->driver_end",
        "driver_end->end",
        "total",
    ]
    ready = _interval(report, "start->services_ready")
    assert (ready["seconds"], ready["requests"], ready["requests_per_second"]) == (100, 2100, 21.0)
    assert ready["by_mdt"] == {"MDT000b": 2000, "MDT0000": 100}
    driver = _interval(report, "services_ready->driver_end")
    assert driver["requests"] == 1000 + 500000 + 7000
    assert driver["by_op"] == {"intent_lock": 500000, "getattr": 7000, "close": 1000}
    total = _interval(report, "total")
    assert (total["from"], total["to"], total["seconds"]) == ("start", "end", 1110)
    assert total["requests"] == 1110 + 502010 + 7000


def test_an_op_lctl_omitted_counts_zero(tmp_path):
    # getattr had a zero count at services_ready, so lctl printed no line for it.
    report = analyze(_write(tmp_path / "s.tsv", _ONE_NODE))

    assert _interval(report, "services_ready->driver_end")["by_op"]["getattr"] == 7000
    # An op with no change between two snapshots is left out of the breakdown.
    assert "getattr" not in _interval(report, "driver_end->end")["by_op"]


def test_two_nodes_are_summed_and_the_busier_one_is_the_peak(tmp_path):
    rows = [
        ("start", 1000, "n1", "MDT000b", "intent_lock", 0),
        ("start", 1000, "n2", "MDT000b", "intent_lock", 10),
        ("start", 1001, "n2", "MDT0003", "getattr", 5),
        ("services_ready", 1100, "n1", "MDT000b", "intent_lock", 1000),
        ("services_ready", 1101, "n2", "MDT000b", "intent_lock", 3010),
        ("services_ready", 1101, "n2", "MDT0003", "getattr", 5),
    ]
    interval = _interval(analyze(_write(tmp_path / "s.tsv", rows)), "start->services_ready")

    assert interval["nodes"] == 2
    assert interval["requests"] == 4000
    assert interval["seconds"] == 101
    assert interval["peak_node"] == "n2"
    assert interval["peak_node_requests_per_second"] == pytest.approx(3000 / 101)
    assert interval["requests_per_second"] == pytest.approx(4000 / 101)


def test_a_node_missing_from_one_phase_is_left_out_of_that_interval(tmp_path):
    rows = [
        ("start", 1000, "n1", "MDT000b", "close", 10),
        ("start", 1000, "n2", "MDT000b", "close", 10),
        ("services_ready", 1010, "n1", "MDT000b", "close", 20),
    ]
    interval = _interval(analyze(_write(tmp_path / "s.tsv", rows)), "start->services_ready")

    assert (interval["nodes"], interval["requests"]) == (1, 10)


def test_an_interval_within_one_second_has_no_rate(tmp_path, capsys):
    rows = [("driver_end", 50, "n1", "MDT000b", "close", 1), ("end", 50, "n1", "MDT000b", "close", 3)]
    path = _write(tmp_path / "s.tsv", rows)
    interval = _interval(analyze(path), "driver_end->end")

    assert interval["requests"] == 2
    assert interval["requests_per_second"] is None
    assert interval["peak_node"] is None
    # The table shows counts, without a rate, for the top MDTs and ops.
    print_lustre_md_stats(path)
    row = next(line for line in capsys.readouterr().out.splitlines() if "driver_end->end" in line)
    assert re.split(r"\s{2,}", row.strip()) == ["driver_end->end", "0", "2", "-", "-", "MDT000b 2", "close 2"]


def test_a_job_killed_before_its_end_reports_up_to_its_last_phase(tmp_path, capsys):
    path = _write(tmp_path / "s.tsv", _ONE_NODE[:4])
    report = analyze(path)

    assert [item["interval"] for item in report["intervals"]] == ["start->services_ready", "total"]
    assert _interval(report, "total")["to"] == "services_ready"
    print_lustre_md_stats(path)
    assert "no `end` snapshot" in capsys.readouterr().out


def test_a_requeued_job_counts_only_its_last_run(tmp_path):
    rows = [
        ("start", 1000, "n1", "MDT000b", "close", 10),
        ("services_ready", 1010, "n1", "MDT000b", "close", 99999),
        ("start", 2000, "n7", "MDT000b", "close", 100),
        ("services_ready", 2010, "n7", "MDT000b", "close", 130),
    ]
    report = analyze(_write(tmp_path / "s.tsv", rows))

    assert report["nodes"] == ["n7"]
    assert _interval(report, "start->services_ready")["requests"] == 30


def test_a_cut_off_row_is_skipped(tmp_path):
    path = _write(tmp_path / "s.tsv", _ONE_NODE[:2])
    path.write_text(path.read_text() + "services_ready\t1100\tn1\tMDT000b\tintent_lo\nend\tx\tn1\tMDT0000\tclose\t3\n")

    assert analyze(path)["phases"] == ["start"]


def test_one_phase_has_nothing_to_compare(tmp_path, capsys):
    print_lustre_md_stats(_write(tmp_path / "s.tsv", _ONE_NODE[:2]))

    assert "fewer than two phases recorded" in capsys.readouterr().out


def test_the_json_maps_each_file_to_its_report(tmp_path, capsys):
    path = _write(tmp_path / "logs" / "lustre-md-stats-12.tsv", _ONE_NODE)
    print_lustre_md_stats(tmp_path, json_output=True)

    captured = capsys.readouterr()
    assert "is experimental" in captured.err
    output = json.loads(captured.out)
    assert list(output) == [str(path)]
    assert output[str(path)]["nodes"] == ["n1"]
    assert set(output[str(path)]["intervals"][0]) == {
        "interval",
        "from",
        "to",
        "seconds",
        "nodes",
        "requests",
        "requests_per_second",
        "peak_node",
        "peak_node_requests_per_second",
        "by_mdt",
        "by_op",
    }


def test_the_text_table_has_one_row_per_interval_with_the_top_mdts_and_ops(tmp_path, capsys):
    print_lustre_md_stats(_write(tmp_path / "s.tsv", _ONE_NODE))

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].endswith("1 node(s), phases start, services_ready, driver_end, end")
    header = ["interval", "seconds", "requests", "req/s", "peak node req/s", "top MDTs", "top ops"]
    assert re.split(r"\s{2,}", lines[1].strip()) == header
    driver = next(line for line in lines if line.lstrip().startswith("services_ready->driver_end"))
    assert "508000" in driver
    assert "508.0 (n1)" in driver
    assert "MDT000b 507.0/s, MDT0000 1.0/s" in driver
    assert "intent_lock 500.0/s, getattr 7.0/s, close 1.0/s" in driver


def test_a_benchmark_dir_yields_every_job_in_job_id_order(tmp_path):
    logs = tmp_path / "logs"
    for job_id in ("999", "1000", "998"):
        _write(logs / f"lustre-md-stats-{job_id}.tsv", _ONE_NODE)
    (logs / "driver-999.log").write_text("")

    assert [path.name for path in stats_files(tmp_path)] == [
        "lustre-md-stats-998.tsv",
        "lustre-md-stats-999.tsv",
        "lustre-md-stats-1000.tsv",
    ]
    assert stats_files(logs) == stats_files(tmp_path)
    assert stats_files(logs / "lustre-md-stats-999.tsv") == [logs / "lustre-md-stats-999.tsv"]


def test_a_dir_without_stats_names_the_setting(tmp_path):
    with pytest.raises(FileNotFoundError, match="Slurm compute sets lustre_md_stats: true"):
        stats_files(tmp_path)


def test_the_reader_imports_without_the_executor():
    # In a fresh interpreter: this test module's own imports would hide what the reader pulls in.
    probe = (
        "import sys;"
        "import nemo_gym.orchestration.executors.slurm.lustre_md_stats;"
        "print(','.join(sorted(m for m in sys.modules if m.startswith('nemo_gym.orchestration.executors.'))))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip().split(",") == [
        "nemo_gym.orchestration.executors.slurm",
        "nemo_gym.orchestration.executors.slurm.lustre_md_stats",
    ]
