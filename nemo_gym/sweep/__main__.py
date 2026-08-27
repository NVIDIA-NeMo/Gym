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
"""CLI for sweep manifests: ``python -m nemo_gym.sweep {validate,build}``."""

from __future__ import annotations

import argparse
import sys

import yaml

from nemo_gym.sweep.build import build_sweep, container_config, run_command
from nemo_gym.sweep.manifest import DEFAULT_SAMPLE_ROWS, SweepValidationError, load_manifest, validate_manifest
from nemo_gym.sweep.materialize import materialize
from nemo_gym.sweep.split import SweepSplitError, split_sweep


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest", help="Path to the sweep manifest YAML.")
    parser.add_argument("--repo-root", default=".", help="Root that relative config paths resolve against.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nemo_gym.sweep", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Check the manifest against its configs and data.")
    _add_shared(validate)
    validate.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
        help="Rows to sample per data file when checking agent_ref. 0 scans the whole file.",
    )
    validate.add_argument("--skip-data", action="store_true", help="Check configs only; do not read data files.")

    build = sub.add_parser("build", help="Concatenate inputs and emit the run config.")
    _add_shared(build)
    build.add_argument("--out-dir", required=True, help="Directory for input.jsonl, sweep_config.yaml, report.")
    build.add_argument("--limit-per-entry", type=int, default=None, help="Take at most N rows per entry (smoke runs).")
    build.add_argument("--overwrite", action="store_true", help="Replace an existing input.jsonl.")
    build.add_argument("--output-jsonl", default="<rollouts-output>.jsonl", help="Shown in the rendered command.")
    build.add_argument("--policy-base-url", default="<router-ip>:8000/v1")
    build.add_argument("--policy-model-name", default="<checkpoint-path>")
    build.add_argument("--num-samples-in-parallel", type=int, default=64)
    build.add_argument("--skip-validate", action="store_true", help="Build without validating first.")

    mat = sub.add_parser(
        "materialize",
        help="Expand repeats in parallel and write Gym's materialized-inputs file directly.",
    )
    _add_shared(mat)
    mat.add_argument("--out-dir", required=True, help="Artifacts land under <out-dir>/<nickname>/.")
    mat.add_argument(
        "--jobs", type=int, default=None, help="Worker processes (default: one per CPU, capped at entries)."
    )
    mat.add_argument("--limit-per-entry", type=int, default=None, help="Take at most N source rows per entry.")
    mat.add_argument("--overwrite", action="store_true", help="Replace an existing materialized file.")
    mat.add_argument(
        "--shuffle",
        type=int,
        default=0,
        metavar="SEED",
        help="Seed for the dispatch-order shuffle; 0 (the default) keeps manifest order. "
        "Manifest order groups each entry contiguously, so the in-flight window shares system "
        "prompts and tool definitions and vLLM prefix caching hits. Shuffling spreads that window "
        "across every environment and thrashes the cache. Enable it when a partial run needs to be "
        "representative of the whole blend. Task identity is assigned before shuffling either way, "
        "so this never changes resume keys.",
    )
    mat.add_argument("--skip-validate", action="store_true", help="Materialize without validating first.")

    sp = sub.add_parser(
        "split",
        help="Split a sweep's inputs and rollouts into one directory per manifest entry.",
    )
    sp.add_argument("sweep_dir", help="The <out-dir>/<nickname> directory materialize wrote.")
    sp.add_argument("--out-dir", default=None, help="Where per-label directories go. Default <sweep_dir>/by_label.")

    cc = sub.add_parser(
        "container-config",
        help="Union several manifests' config_paths into one config for building a container.",
    )
    cc.add_argument("manifests", nargs="+", help="Manifest YAMLs to union.")
    cc.add_argument("-o", "--output", required=True, help="Where to write the container config.")

    args = parser.parse_args(argv)

    try:
        if args.command == "split":
            result = split_sweep(args.sweep_dir, args.out_dir)
            for label, counts in sorted(result.counts.items()):
                print(f"  {label:<32} {counts.inputs:>7,} inputs  {counts.rollouts:>7,} rollouts")
            print(f"wrote {len(result.counts)} label directories under {result.out_dir}")
            if result.labels_without_rollouts:
                missing = ", ".join(result.labels_without_rollouts)
                print(f"no rollouts for {len(result.labels_without_rollouts)}: {missing}")
            if result.unmapped_inputs or result.unmapped_rollouts:
                print(
                    f"warn: {result.unmapped_inputs:,} inputs and {result.unmapped_rollouts:,} rollouts "
                    "had no task index in any entry range and were dropped."
                )
            return 0

        if args.command == "container-config":
            manifests = [load_manifest(m) for m in args.manifests]
            doc = container_config(manifests)
            with open(args.output, "w") as handle:
                yaml.safe_dump(doc, handle, default_flow_style=False, sort_keys=False)
            print(f"wrote {args.output}: {len(doc['config_paths'])} config paths "
                  f"from {len(manifests)} manifest(s)")
            return 0

        manifest = load_manifest(args.manifest)
        if args.command == "validate":
            warnings = validate_manifest(
                manifest,
                repo_root=args.repo_root,
                sample_rows=args.sample_rows,
                check_data=not args.skip_data,
            )
            print(f"OK: {len(manifest.entries)} entries, {len(manifest.config_paths())} distinct configs")
            print(f"num_repeats: {manifest.num_repeats()}")
            for warning in warnings:
                print(f"warn: {warning}")
            return 0

        if args.command == "materialize":
            if not args.skip_validate:
                for warning in validate_manifest(manifest, repo_root=args.repo_root):
                    print(f"warn: {warning}")
            mreport = materialize(
                manifest,
                args.out_dir,
                jobs=args.jobs,
                limit_per_entry=args.limit_per_entry,
                overwrite=args.overwrite,
                shuffle_seed=args.shuffle,
            )
            print(f"\nwrote {mreport.materialized_fpath}")
            print(f"  {mreport.total_source_rows:,} source rows -> {mreport.total_materialized_rows:,} materialized")
            print(f"touched {mreport.output_fpath} (empty; completes the --resume gate)")
            print(f"wrote {mreport.config_fpath} ({len(manifest.config_paths())} config paths)")
            print(f"wrote {mreport.report_fpath} (observed per-entry row counts)")
            print(
                "\nRun with --resume so Gym loads these directly and skips preprocessing:\n"
                f"  gym eval run --no-serve --resume \\\n"
                f"      --input {mreport.materialized_fpath} \\\n"
                f"      --output {mreport.output_fpath} \\\n"
                f"      ++num_repeats=1"
            )
            return 0

        if not args.skip_validate:
            for warning in validate_manifest(manifest, repo_root=args.repo_root):
                print(f"warn: {warning}")
        report = build_sweep(
            manifest,
            args.out_dir,
            limit_per_entry=args.limit_per_entry,
            overwrite=args.overwrite,
        )
        print(f"wrote {report.input_jsonl} ({report.total_rows:,} rows)")
        print(f"wrote {report.config_yaml} ({len(report.config_paths)} config paths)")
        print(f"wrote {report.report_json}")
        if report.overrides_applied:
            print(f"agent_ref overrides applied: {report.overrides_applied}")
        print()
        print(
            run_command(
                report,
                output_jsonl=args.output_jsonl,
                policy_base_url=args.policy_base_url,
                policy_model_name=args.policy_model_name,
                num_samples_in_parallel=args.num_samples_in_parallel,
            )
        )
        return 0
    except (SweepValidationError, SweepSplitError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
