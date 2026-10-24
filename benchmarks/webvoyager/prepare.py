# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the maintained 552-task WebVoyager population for visual browsers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import tempfile
import urllib.request
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from nemo_gym.web.datasets import adapt_webvoyager_record, load_json_records, write_jsonl


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "webvoyager.jsonl"
DEFAULT_ENV_FPATH = BENCHMARK_DIR / "env.yaml"
DEFAULT_ROLLOUT_FPATH = REPO_ROOT / "results" / "webvoyager" / "rollouts.jsonl"
PROVENANCE_FPATH = BENCHMARK_DIR / "provenance.yaml"
PROVENANCE = yaml.safe_load(PROVENANCE_FPATH.read_text(encoding="utf-8"))
DATASET_PROVENANCE = PROVENANCE["dataset"]
SOURCE_COMMIT = DATASET_PROVENANCE["commit"]
SOURCE_URL = DATASET_PROVENANCE["raw_url"]
SOURCE_SHA256 = DATASET_PROVENANCE["sha256"]
EXPECTED_TASKS = int(DATASET_PROVENANCE["task_count"])
SOURCE_FPATH = BENCHMARK_DIR / "data" / "webvoyager_source.jsonl"
DATE_TEMPLATE_PROVENANCE = DATASET_PROVENANCE["date_template"]
DATE_TEMPLATE_URL = DATE_TEMPLATE_PROVENANCE["raw_url"]
DATE_TEMPLATE_SHA256 = DATE_TEMPLATE_PROVENANCE["sha256"]
DATE_TEMPLATE_FPATH = BENCHMARK_DIR / "data" / "webvoyager_date_template.jsonl"
SOURCE_REFERENCE_DATE = date.fromisoformat(DATE_TEMPLATE_PROVENANCE["source_reference_date"])
REFERENCE_DATE = date.fromisoformat(DATASET_PROVENANCE["reference_date"])
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
ABBREVIATED_MONTHS = ("Jan.", "Feb.", "Mar.", "Apr.", "May", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec.")

PROFILE_CONFIGS = {
    "nano_omni": (BENCHMARK_DIR / "configs" / "nano_omni.yaml",),
    "qwen35_122b_a10b": (BENCHMARK_DIR / "configs" / "qwen35_122b_a10b.yaml",),
}
PROFILE_AGENTS = {
    "nano_omni": "nano_omni_webvoyager_agent",
    "qwen35_122b_a10b": "qwen35_webvoyager_agent",
}
PROFILE_SAMPLING = {profile: dict(config["sampling"]) for profile, config in PROVENANCE["policy_profiles"].items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(label: str, url: str, sha256: str, destination: Path) -> Path:
    """Materialize an immutable maintained file in Gym's ignored cache."""

    if destination.is_file() and _sha256(destination) == sha256:
        print(f"Using cached {label}: {destination}", flush=True)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {label} from {url}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != sha256:
        raise ValueError(f"{label} hash mismatch: expected {sha256}, got {digest}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            handle.write(payload)
            temporary_path = Path(handle.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _pinned_file(configured: str | Path | None, label: str, url: str, sha256: str, cache: Path) -> Path:
    path = Path(configured).expanduser() if configured else _download(label, url, sha256, cache)
    digest = _sha256(path)
    if digest != sha256:
        raise ValueError(f"{label} hash mismatch: expected {sha256}, got {digest}")
    return path


def _ordinal(day: int) -> str:
    suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def _month_day(value: date) -> str:
    return f"{MONTHS[value.month - 1]} {value.day}"


def _month_day_ordinal(value: date) -> str:
    return f"{MONTHS[value.month - 1]} {_ordinal(value.day)}"


def _month_day_year(value: date) -> str:
    return f"{_month_day(value)}, {value.year}"


def _day_month_year(value: date) -> str:
    return f"{value.day} {MONTHS[value.month - 1]} {value.year}"


SINGLE_DATE_STYLES = {
    "month_day": _month_day,
    "month_day_ordinal": _month_day_ordinal,
    "month_day_year": _month_day_year,
    "month_day_ordinal_year": lambda value: f"{_month_day_ordinal(value)}, {value.year}",
    "abbr_month_day": lambda value: f"{ABBREVIATED_MONTHS[value.month - 1]} {value.day}",
    "day_month_year": _day_month_year,
}


def _render_date_value(style: str, start: date, end: date | None) -> str:
    """Render one placeholder style of the source's ``template/render_webvoyager.py``."""

    if style.startswith("end_"):
        single_style = style.removeprefix("end_")
        if end is None or single_style not in SINGLE_DATE_STYLES:
            raise ValueError(f"date style {style!r} needs an end date and a single-date style")
        return SINGLE_DATE_STYLES[single_style](end)
    if style in SINGLE_DATE_STYLES:
        return SINGLE_DATE_STYLES[style](start)
    if end is None:
        raise ValueError(f"date style {style!r} needs an end date")
    same_month = (start.year, start.month) == (end.year, end.month)
    match style:
        case "month_day_range":
            return f"{_month_day(start)}-{end.day}" if same_month else f"{_month_day(start)}-{_month_day(end)}"
        case "month_day_range_year":
            if same_month:
                return f"{_month_day(start)}-{end.day}, {start.year}"
            return f"{_month_day_year(start)} - {_month_day_year(end)}"
        case "ordinal_month_day_range":
            return f"{_month_day_ordinal(start)} - {_month_day_ordinal(end)}"
        case "month_day_to_day":
            return f"{_month_day(start)} to {end.day}" if same_month else f"{_month_day(start)} to {_month_day(end)}"
        case "month_day_to_day_year":
            if same_month:
                return f"{_month_day(start)} to {end.day}, {start.year}"
            return f"{_month_day_year(start)} to {_month_day_year(end)}"
        case "day_month_range_year":
            if same_month:
                return f"{start.day} {MONTHS[start.month - 1]} to {end.day} {MONTHS[end.month - 1]} {start.year}"
            return f"{_day_month_year(start)} to {_day_month_year(end)}"
        case "numeric_dmy_range":
            return f"{start:%d/%m/%Y} - {end:%d/%m/%Y}"
        case "sentence_range_year":
            return f"{_month_day_year(start)}, to {_month_day_year(end)}"
        case "between_ordinal_range_year":
            return f"{_month_day_ordinal(start)}, {start.year}, and {_month_day_ordinal(end)}, {end.year}"
    raise ValueError(f"unknown date style {style!r}")


def _render_question(template: Mapping[str, Any], index: int, reference_date: date) -> tuple[str, date]:
    """Render one template entry; return the question and the earliest date it names."""

    # The source renderer places entry N at min_days + (3N mod spread_days)
    # days after the reference date, counting entries in template order.
    policy = template["date_policy"]
    offset = int(policy.get("min_days_from_today", 30)) + (index * 3) % max(int(policy.get("spread_days", 90)), 1)
    start = reference_date + timedelta(days=offset)
    duration = policy.get("duration_days")
    end = None if duration is None else start + timedelta(days=int(duration))
    question = template["ques_template"]
    named = []
    for placeholder, style in template["render"].items():
        question = question.replace("{" + placeholder + "}", _render_date_value(style, start, end))
        named.append(end if style.startswith("end_") else start)
    if "{" in question or "}" in question:
        raise ValueError(f"unrendered date placeholder in {template['id']}: {question}")
    return question, min(named)


def _render_dated_questions(
    records: list[dict[str, Any]], templates: list[dict[str, Any]], reference_date: date
) -> dict[str, tuple[str, date]]:
    """Render the templated Booking and Google Flights questions for ``reference_date``.

    The maintained source renders these questions for ``SOURCE_REFERENCE_DATE``.
    Every template entry must first reproduce its source question for that date,
    so a new reference date changes nothing but the dates. Returns each task's
    question and the earliest date it names.
    """

    if (reference_date - SOURCE_REFERENCE_DATE).days % 7:
        raise ValueError(
            f"reference date {reference_date} must be a whole number of weeks after {SOURCE_REFERENCE_DATE} "
            "so that each question keeps its weekdays"
        )
    by_id = {record["id"]: record for record in records}
    rendered = {}
    for index, template in enumerate(templates):
        record = by_id.get(template["id"])
        if record is None or (record["web_name"], record["web"]) != (template["web_name"], template["web"]):
            raise ValueError(f"date template entry {template['id']} has no matching source task")
        if _render_question(template, index, SOURCE_REFERENCE_DATE)[0] != record["ques"]:
            raise ValueError(f"date template does not reproduce {template['id']} for {SOURCE_REFERENCE_DATE}")
        rendered[template["id"]] = _render_question(template, index, reference_date)
    return rendered


def _today() -> date:
    return date.today()


def prepare(
    source: str | Path | None = None,
    output: str | Path = OUTPUT_FPATH,
    *,
    date_template: str | Path | None = None,
    reference_date: date | str | None = None,
) -> Path:
    """Prepare one model-independent, hash-pinned 552-task dataset.

    ``gym eval prepare --benchmark webvoyager`` calls this function without
    arguments. Nano Omni and Qwen select different policy adapters and serving
    profiles at runtime but consume these exact same task rows.

    Booking and Google Flights questions name absolute dates, rendered for
    ``reference_date`` (default: ``reference_date`` in provenance.yaml). The live
    sites cannot search a past date, so preparation fails when any rendered date
    is not after today. ``original_metadata`` keeps the source record unchanged.
    """

    source_path = _pinned_file(
        source or os.environ.get("WEBVOYAGER_SOURCE_JSONL"),
        "WebVoyager source",
        SOURCE_URL,
        SOURCE_SHA256,
        SOURCE_FPATH,
    )
    template_path = _pinned_file(
        date_template or os.environ.get("WEBVOYAGER_DATE_TEMPLATE_JSONL"),
        "WebVoyager date template",
        DATE_TEMPLATE_URL,
        DATE_TEMPLATE_SHA256,
        DATE_TEMPLATE_FPATH,
    )
    if reference_date is None:
        reference_date = REFERENCE_DATE
    elif not isinstance(reference_date, date):
        reference_date = date.fromisoformat(reference_date)
    records = load_json_records(source_path)
    if len(records) != EXPECTED_TASKS:
        raise ValueError(f"maintained WebVoyager requires exactly {EXPECTED_TASKS} tasks, got {len(records)}")
    rendered = _render_dated_questions(records, load_json_records(template_path), reference_date)
    today = _today()
    expired = sorted(task_id for task_id, (_, earliest) in rendered.items() if earliest <= today)
    if expired:
        raise ValueError(
            f"{len(expired)} WebVoyager questions rendered for reference date {reference_date} name a date on or "
            f"before today ({today}), first {expired[0]} ({rendered[expired[0]][1]}); pass a later reference date"
        )
    rows = []
    for record in records:
        row = adapt_webvoyager_record(record)
        if record["id"] in rendered:
            row["web_task"]["intent"] = rendered[record["id"]][0]
        rows.append(row)
    count = write_jsonl(rows, output)
    if rendered:
        closest = min(earliest for _, earliest in rendered.values())
        print(f"Rendered {len(rendered)} dated questions for {reference_date}; the closest date is {closest}")
    print(f"Wrote {count} WebVoyager tasks to {output}", flush=True)
    return Path(output)


def _yaml_string(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def write_env(
    env_path: str | Path,
    *,
    profile: str,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    concurrency: int = 1,
    force: bool = False,
) -> bool:
    """Write a private, gitignored Gym composition for one policy profile."""

    if profile not in PROFILE_CONFIGS:
        raise ValueError(f"unsupported WebVoyager profile: {profile!r}")
    if concurrency != 1:
        raise ValueError(
            "one headed visual-browser process owns one DISPLAY; shard across isolated Gym processes instead"
        )
    config_paths = tuple(path.resolve() for path in PROFILE_CONFIGS[profile])
    missing = [path for path in config_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"WebVoyager Gym config does not exist: {missing[0]}")

    env_path = Path(env_path).expanduser().resolve()
    if env_path.exists() and not force:
        print(f"Keeping existing configuration: {env_path}")
        return False
    env_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampling = PROFILE_SAMPLING[profile]
    lines = [
        "# Generated by benchmarks/webvoyager/prepare.py. This file is gitignored.",
        "config_paths:",
        *(f"  - {_yaml_string(path)}" for path in config_paths),
        f"agent_name: {PROFILE_AGENTS[profile]}",
        f"input_jsonl_fpath: {_yaml_string(Path(input_jsonl).expanduser().resolve())}",
        f"output_jsonl_fpath: {_yaml_string(output_path)}",
        "num_repeats: 1",
        "num_samples_in_parallel: 1",
        "upload_rollouts: false",
        "responses_create_params:",
        f"  max_output_tokens: {sampling['max_output_tokens']}",
        f"  temperature: {sampling['temperature']}",
        f"  top_p: {sampling['top_p']}",
        "policy_base_url: ${oc.env:POLICY_BASE_URL,http://127.0.0.1:8000/v1}",
        "policy_api_key: ${oc.env:POLICY_API_KEY,local-vllm}",
        "policy_model_name: ${oc.env:POLICY_MODEL_NAME,webvoyager-policy}",
        "webvoyager_judge_base_url: ${oc.env:WEBARENA_JUDGE_BASE_URL,https://inference-api.nvidia.com/v1}",
        "webvoyager_judge_api_key: ${oc.env:WEBARENA_JUDGE_API_KEY,unset}",
        "webvoyager_judge_model_name: ${oc.env:WEBARENA_JUDGE_MODEL,gcp/google/gemini-3-flash-preview}",
        "",
    ]
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if force else os.O_EXCL)
    descriptor = os.open(env_path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    os.chmod(env_path, 0o600)
    print(f"Wrote private configuration: {env_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_CONFIGS), default="nano_omni")
    parser.add_argument("--source", type=Path, default=None, help="Source WebVoyager JSONL")
    parser.add_argument("--date-template", type=Path, default=None, help="Source date template JSONL")
    parser.add_argument(
        "--reference-date",
        type=date.fromisoformat,
        default=REFERENCE_DATE,
        help=f"Render dated questions for this YYYY-MM-DD, whole weeks after {SOURCE_REFERENCE_DATE}",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_FPATH, help="Prepared Gym JSONL")
    parser.add_argument("--rollout-output", type=Path, default=DEFAULT_ROLLOUT_FPATH)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FPATH)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-env", action="store_true", help="Prepare data without writing env.yaml")
    parser.add_argument("--force-env", action="store_true", help="Replace an existing generated env.yaml")
    args = parser.parse_args()

    prepared = prepare(args.source, args.output, date_template=args.date_template, reference_date=args.reference_date)
    if not args.no_env:
        write_env(
            args.env_file,
            profile=args.profile,
            input_jsonl=prepared,
            output_jsonl=args.rollout_output,
            concurrency=args.concurrency,
            force=args.force_env,
        )

    print("\nNext steps:")
    print(f"  cd {_yaml_string(args.env_file.expanduser().resolve().parent)}")
    gym_cli = shlex.quote(str(REPO_ROOT / ".venv" / "bin" / "gym"))
    print(f"  {gym_cli} env prefetch")
    print(f"  {gym_cli} env start")
    print(f"  {gym_cli} eval run --no-serve")


if __name__ == "__main__":
    main()
