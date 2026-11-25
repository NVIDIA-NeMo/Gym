# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline execution of the exact installer download function with shell mocks.

No network, containers, HOME changes, installs, models or native admission.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path


INSTALLER = Path(__file__).resolve().parents[1] / "install_codex_runtime.sh"


def _download_function() -> str:
    source = INSTALLER.read_text()
    start = source.index("download_artifact() {\n")
    end = source.index("\n}\n", start) + 3
    return source[start:end]


class DownloadRetryTest(unittest.TestCase):
    def _run_downloads(self, statuses, costs, *, second=False, previous=False, deadline=180):
        with tempfile.TemporaryDirectory(prefix="codex-download-unit-") as directory:
            work = Path(directory)
            if previous:
                (work / "archive").write_text("previous-complete")
            # Unsetting Bash's special timer first makes this an actual mock
            # clock; filesystem scheduling must not consume simulated seconds.
            script = """set -Eeuo pipefail
unset SECONDS
SECONDS=0
mock_calls=0
mock_statuses=(%s)
mock_costs=(%s)
download_deadline=%d
curl() {
  local retry='' max_time='' connect_time='' output='' base_flags=0
  while (( $# )); do
    case "$1" in
      -fsSL) base_flags=1; shift ;;
      --retry) retry=$2; shift 2 ;;
      --connect-timeout) connect_time=$2; shift 2 ;;
      --max-time) max_time=$2; shift 2 ;;
      -o) output=$2; shift 2 ;;
      https://unit.invalid/*) shift ;;
      *) return 2 ;;
    esac
  done
  [[ "$retry" = 0 && "$base_flags" = 1 && -n "$output" ]] || return 2
  mock_calls=$((mock_calls + 1))
  local index=$((mock_calls - 1)) preexisting=0
  local status=${mock_statuses[$index]:-56} cost=${mock_costs[$index]:-1}
  if [[ -e "$output" ]]; then preexisting=1; fi
  (( cost <= max_time )) || return 2
  printf '%%s|%%s|%%s|%%s|%%s|%%s|%%s\n' "$mock_calls" "$retry" "$max_time" "$connect_time" "$preexisting" "$status" "$cost" >> calls
  if (( status == 0 )); then
    printf 'complete%%s' "$mock_calls" > "$output"
  else
    printf 'partial%%s' "$mock_calls" > "$output"
  fi
  SECONDS=$((SECONDS + cost))
  return "$status"
}
sleep() {
  printf '%%s\n' "$1" >> sleeps
  SECONDS=$((SECONDS + $1))
}
""" % (" ".join(map(str, statuses)), " ".join(map(str, costs)), deadline)
            script += _download_function() + "\n"
            script += "if download_artifact https://unit.invalid/archive archive; then first=0; else first=$?; fi\n"
            if second:
                script += "if (( first == 0 )); then if download_artifact https://unit.invalid/sums sums; then second=0; else second=$?; fi; else second=-1; fi\n"
                script += "printf '%s\\n' \"$second\" > second_status\n"
            script += "printf '%s\\n' \"$first\" > first_status\nprintf '%s\\n' \"$SECONDS\" > elapsed\n"
            result = subprocess.run(["bash", "-c", script], cwd=work, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = (
                [list(map(int, line.split("|"))) for line in (work / "calls").read_text().splitlines()]
                if (work / "calls").exists()
                else []
            )
            return {
                "first": int((work / "first_status").read_text()),
                "second": int((work / "second_status").read_text()) if second else None,
                "elapsed": int((work / "elapsed").read_text()),
                "calls": rows,
                "sleeps": [int(x) for x in (work / "sleeps").read_text().splitlines()]
                if (work / "sleeps").exists()
                else [],
                "archive": (work / "archive").read_text() if (work / "archive").exists() else None,
                "sums": (work / "sums").read_text() if (work / "sums").exists() else None,
                "partial_files": [p.name for p in work.glob("*.partial.*")],
            }

    def test_normal_first_success_keeps_tls_flags_and_disables_nested_retry(self) -> None:
        out = self._run_downloads([0], [1])
        self.assertEqual(out["first"], 0)
        self.assertEqual(out["archive"], "complete1")
        self.assertEqual(out["calls"], [[1, 0, 180, 10, 0, 0, 1]])
        self.assertEqual(out["sleeps"], [])
        self.assertEqual(out["partial_files"], [])

    def test_curl56_then_success_retries_and_discards_partial_output(self) -> None:
        out = self._run_downloads([56, 0], [1, 1])
        self.assertEqual(out["first"], 0)
        self.assertEqual(out["archive"], "complete2")
        self.assertEqual(len(out["calls"]), 2)
        self.assertTrue(all(row[4] == 0 for row in out["calls"]))
        self.assertEqual(out["sleeps"], [2])
        self.assertEqual(out["partial_files"], [])

    def test_four_failures_preserve_nonzero_code_and_publish_nothing(self) -> None:
        out = self._run_downloads([56] * 4, [1] * 4)
        self.assertEqual(out["first"], 56)
        self.assertEqual(len(out["calls"]), 4)
        self.assertEqual(out["sleeps"], [2, 2, 2])
        self.assertIsNone(out["archive"])
        self.assertEqual(out["partial_files"], [])
        self.assertLessEqual(out["elapsed"], 180)

    def test_partial_failures_do_not_overwrite_previous_complete_output(self) -> None:
        out = self._run_downloads([56] * 4, [1] * 4, previous=True)
        self.assertEqual(out["first"], 56)
        self.assertEqual(out["archive"], "previous-complete")
        self.assertEqual(out["partial_files"], [])

    def test_both_downloads_share180_second_deadline_without_reset(self) -> None:
        out = self._run_downloads([56, 56, 56, 0, 28], [45, 45, 45, 38, 1], second=True)
        self.assertEqual(out["first"], 0)
        self.assertEqual(out["archive"], "complete4")
        self.assertEqual(out["second"], 124)
        self.assertIsNone(out["sums"])
        self.assertEqual([row[2] for row in out["calls"]], [180, 133, 86, 39, 1])
        self.assertEqual(out["calls"][-1][3], 1)
        self.assertEqual(out["elapsed"], 180)
        self.assertEqual(out["partial_files"], [])

    def test_slow_healthy_transfer_can_use_more_than45_seconds(self) -> None:
        out = self._run_downloads([0, 0], [120, 2], second=True)
        self.assertEqual(out["first"], 0)
        self.assertEqual(out["second"], 0)
        self.assertEqual(out["archive"], "complete1")
        self.assertEqual(out["sums"], "complete2")
        self.assertEqual([row[2] for row in out["calls"]], [180, 60])
        self.assertEqual(out["elapsed"], 122)

    def test_expired_deadline_starts_no_transfer(self) -> None:
        out = self._run_downloads([0], [1], deadline=0)
        self.assertEqual(out["first"], 124)
        self.assertEqual(out["calls"], [])
        self.assertIsNone(out["archive"])


if __name__ == "__main__":
    unittest.main()
