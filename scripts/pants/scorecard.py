# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a daily scorecard from retained comparison artifacts."""

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def write_scorecards(output: Path) -> list[Path]:
    days = defaultdict(list)
    missing = defaultdict(list)
    for path in sorted((output / "batches").glob("*/*/comparison/results.json")):
        result = json.loads(path.read_text())
        days[result["started_at"][:10]].append((path, result))
    for path in sorted((output / "batches").glob("*/*/sample.json")):
        if (path.parent / "comparison/results.json").exists():
            continue
        stamp = path.parent.parent.name[:8]
        day = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
        missing[day].append((path, json.loads(path.read_text())))
        days.setdefault(day, [])
    written = []
    for day, reports in days.items():
        destination = output / "daily" / f"{day}.md"
        destination.parent.mkdir(exist_ok=True)
        lines = [
            f"# Pants pilot — {day} (UTC)",
            "",
            "Pilot subset only. Ratios compare serial command time on the same host, not production CI latency.",
            "Cold cases exclude provisioning the Python and Pants executables. Existing coverage gates remain unchanged.",
            "",
            f"Comparisons: {len(reports) + len(missing[day])}. "
            f"Incomplete: {sum(bool(r.get('error')) for _, r in reports) + len(missing[day])}.",
            "",
        ]
        grouped = defaultdict(list)
        for _, result in reports:
            scope = hashlib.sha256(
                json.dumps(
                    [result.get("manifest"), result.get("dependency_integrity", {}).get("requirements_sha256")],
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:12]
            for row in result.get("measurements", []):
                key = (result["platform"], result["python"], row["case"], scope)
                grouped[key].append(row)
        lines += [
            "| Platform / Python | Case | Pairs | uv median (s) | Pants median (s) | Ratio | Scope / dependencies |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
        for (platform, python, case, scope), rows in sorted(grouped.items()):
            uv = statistics.median(row["uv"]["seconds"] for row in rows)
            pants = statistics.median(row["pants"]["seconds"] for row in rows)
            valid = all(row["valid"] for row in rows)
            ratio = f"{uv / pants:.2f}x" if valid and pants else "withheld: outcome mismatch"
            lines.append(
                f"| {platform} / {python} | {case} | {len(rows)} | {uv:.2f} | {pants:.2f} | {ratio} | {scope} |"
            )
        lines += ["", "Individual comparisons (including failures):", ""]
        for path, result in reports:
            relative = Path("..") / path.parent.relative_to(output) / "summary.md"
            lines.append(
                f"- [`{result['base_sha'][:12]}`]({relative.as_posix()}) — overlay `{result['overlay_sha256'][:12]}`."
            )
        for path, result in missing[day]:
            relative = Path("..") / path.relative_to(output)
            logs = [
                f"[{name}]({(relative.parent / name).as_posix()})"
                for name in ("prepare.log", "compare.log")
                if (path.parent / name).exists()
            ]
            lines.append(
                f"- [`{result['sha'][:12]}`]({relative.as_posix()}) — "
                "incomplete before comparison results were written. " + ", ".join(logs)
            )
        lines += [
            "",
            "Different platforms, test scopes, and dependency locks are kept separate. "
            "Repeated pairs and sampled commits are exploratory measurements, "
            "not independent statistical evidence. Inspect each report for selection fallbacks and test counts.",
        ]
        destination.write_text("\n".join(lines) + "\n")
        written.append(destination)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    for written in write_scorecards(parser.parse_args().output.resolve()):
        print(written)
