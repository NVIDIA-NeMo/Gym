#!/usr/bin/env python3
"""
Convert a NeMo-Gym pass@k rollout JSONL into the full CVDP report layout:

    output/
      sample_1/   (report.json, report.txt, raw_result.json, prompt_response.jsonl, run.log, per-task dirs)
      sample_2/
      ...
      composite_report.json
      composite_report.txt

Splits the rollout JSONL by _ng_rollout_index, runs build_collateral on each split,
then calls combine_reports() to produce the composite.

Usage:
    python scripts/cvdp_pass_at_k_report.py \\
        --rollouts  results/rollouts.jsonl \\
        --output    results/report/ \\
        [--model    my-model-name] \\
        [--dataset  /path/to/cvdp_dataset.jsonl] \\
        [--k        1]
"""

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def split_by_rollout(rollouts_path: str) -> dict[int, list[dict]]:
    """Split rollout rows by _ng_rollout_index."""
    by_index: dict[int, list[dict]] = defaultdict(list)
    with open(rollouts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx = r.get("_ng_rollout_index", 0)
            by_index[idx].append(r)
    return dict(by_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CVDP pass@k report from NeMo-Gym rollouts")
    parser.add_argument("--rollouts", required=True, help="pass@k rollout JSONL from ng_collect_rollouts")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="nemo-gym", help="Model name for report metadata")
    parser.add_argument("--dataset", help="Path to original CVDP dataset JSONL (for report metadata)")
    parser.add_argument("--k", type=int, default=1, help="Pass@k threshold for composite report (default: 1)")
    args = parser.parse_args()

    # Import from the self-contained cvdp report module (no CVDP repo needed)
    cvdp_root = Path(__file__).parent.parent
    sys.path.insert(0, str(cvdp_root))
    sys.path.insert(0, str(cvdp_root / "scripts"))
    from cvdp_report import Report, build_collateral, combine_reports

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Split rollout JSONL by _ng_rollout_index
    # -----------------------------------------------------------------------
    by_index = split_by_rollout(args.rollouts)
    n_samples = len(by_index)
    print(f"Found {n_samples} rollout indices across {sum(len(v) for v in by_index.values())} rows")

    # -----------------------------------------------------------------------
    # For each rollout index, write a temp JSONL and run build_collateral
    # -----------------------------------------------------------------------
    sample_prefixes = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx in sorted(by_index.keys()):
            sample_name = f"sample_{idx + 1}"
            sample_dir = output_dir / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_prefixes.append(str(sample_dir))

            # Write temp JSONL for this rollout index
            tmp_jsonl = Path(tmpdir) / f"rollout_{idx}.jsonl"
            with open(tmp_jsonl, "w") as f:
                for row in by_index[idx]:
                    f.write(json.dumps(row) + "\n")

            print(f"\n=== Building {sample_name} ({len(by_index[idx])} tasks) ===")

            # Build per-task collateral
            raw_result = build_collateral(str(tmp_jsonl), sample_dir)
            print(f"  Built raw_result for {len(raw_result)} tasks")

            # Write raw_result.json
            raw_result_path = sample_dir / "raw_result.json"
            with open(raw_result_path, "w") as f:
                json.dump(raw_result, f, indent=2)

            # Generate report.json + report.txt via CVDP's Report class
            rpt = Report(
                raw_result,
                prefix=str(sample_dir),
                dataset_path=args.dataset,
                model_agent=args.model,
            )
            rpt.report_header()
            rpt.report_categories()
            rpt.report_timers()
            print(f"  Wrote {sample_dir / 'report.json'}")
            print(f"  Wrote {sample_dir / 'report.txt'}")

    # -----------------------------------------------------------------------
    # Combine all sample reports into composite
    # -----------------------------------------------------------------------
    print(f"\n=== Combining {n_samples} sample reports ===")
    combine_reports(
        sample_prefixes=sample_prefixes,
        output_prefix=str(output_dir),
        n_samples=n_samples,
        k_threshold=args.k,
    )

    print(f"\nDone. Output in: {output_dir}")
    print("  composite_report.json")
    print("  composite_report.txt")
    for sp in sample_prefixes:
        print(f"  {Path(sp).name}/report.txt")


if __name__ == "__main__":
    main()
