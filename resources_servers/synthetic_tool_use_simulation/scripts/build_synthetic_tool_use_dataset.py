# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build raw-seed datasets for synthetic tool-use Gym rollouts."""

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.synthetic_tool_use.common.quality import (
    ArtifactValidationError,
    detect_leak,
    validate_tool_schema,
)
from resources_servers.synthetic_tool_use_simulation.app import (
    CustomerScenario,
    SyntheticToolUseSeedSessionRequest,
    ToolSignature,
)
from responses_api_agents.synthetic_tool_use_agent.app import SyntheticToolUseAgent


DEFAULT_SOURCE_ENV_VARS = (
    "SYNTHETIC_TOOL_USE_SIMPLE_SOURCE_DIR",
    "SYNTHETIC_TOOL_USE_PROACTIVE_SOURCE_DIR",
)
DEFAULT_SOURCE_NAMES = [
    "260625_nemotron_synthetic_tool_use_conversational_simple",
    "260625_nemotron_synthetic_tool_use_conversational_proactive",
]
DEFAULT_DATASET_NAME = "synthetic_tool_use_sample"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / f"{DEFAULT_DATASET_NAME}.jsonl"
DEFAULT_AGENT_NAME = "synthetic_tool_use_simulation_agent"
DEFAULT_DOMAIN_GENERATOR_MODEL = "Qwen3-235B-A22B-Thinking-2507"
DEFAULT_POLICY_TOOLS_MODEL = "DeepSeek-R1-0528"
DEFAULT_SCENARIO_GENERATOR_MODEL = "Qwen3-235B-A22B-Thinking-2507"
DEFAULT_POLICY_TEMPERATURE = 1.0

TOOL_REQUIRED_FIELDS = ("name", "doc", "params", "returns")
SCENARIO_REQUIRED_FIELDS = (
    "customer_persona",
    "reason_for_contact",
    "customer_details",
    "unknown_info",
    "task_instructions",
    "representative_domain",
    "outside_policy_scope",
)


@dataclass
class SourceReport:
    source_name: str
    source_index: int
    domains_seen: int = 0
    domains_accepted: int = 0
    rows_written: int = 0
    scenarios_seen: int = 0
    scenarios_accepted: int = 0
    skipped_domains: Counter[str] = field(default_factory=Counter)
    skipped_scenarios: Counter[str] = field(default_factory=Counter)
    skip_examples: list[dict[str, Any]] = field(default_factory=list)

    def record_domain_skip(self, domain_dir: Path, reason: str, detail: str, limit: int) -> None:
        self.skipped_domains[reason] += 1
        self._record_example("domain", domain_dir, reason, detail, limit)

    def record_scenario_skip(self, scenario_path: Path, reason: str, detail: str, limit: int) -> None:
        self.skipped_scenarios[reason] += 1
        self._record_example("scenario", scenario_path, reason, detail, limit)

    def _record_example(self, artifact_type: str, path: Path, reason: str, detail: str, limit: int) -> None:
        if len(self.skip_examples) >= limit:
            return
        example = {
            "type": artifact_type,
            "source_name": self.source_name,
            "reason": reason,
            "detail": detail,
        }
        if artifact_type == "domain":
            example["domain_dir_name"] = path.name
        else:
            domain_dir, relative_path = scenario_artifact_reference(path)
            example["domain_dir_name"] = domain_dir
            example["scenario_file"] = relative_path
        self.skip_examples.append(example)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_index": self.source_index,
            "domains_seen": self.domains_seen,
            "domains_accepted": self.domains_accepted,
            "rows_written": self.rows_written,
            "scenarios_seen": self.scenarios_seen,
            "scenarios_accepted": self.scenarios_accepted,
            "skipped_domains": dict(sorted(self.skipped_domains.items())),
            "skipped_scenarios": dict(sorted(self.skipped_scenarios.items())),
            "skip_examples": self.skip_examples,
        }


@dataclass(frozen=True)
class DatasetMetadataConfig:
    dataset_name: str
    domain_generator_model: str = DEFAULT_DOMAIN_GENERATOR_MODEL
    policy_tools_model: str = DEFAULT_POLICY_TOOLS_MODEL
    scenario_generator_model: str = DEFAULT_SCENARIO_GENERATOR_MODEL

    def to_dict(self) -> dict[str, str]:
        return {
            "dataset_name": self.dataset_name,
            "domain_generator_model": self.domain_generator_model,
            "policy_tools_model": self.policy_tools_model,
            "scenario_generator_model": self.scenario_generator_model,
        }


def numeric_domain_dirs(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        return []
    return sorted(
        [path for path in source_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            rows.append((line_number, json.loads(line)))
    return rows


def load_policy(domain_dir: Path) -> str:
    return (domain_dir / "policy.md").read_text(encoding="utf-8")


def load_tools(domain_dir: Path) -> list[dict[str, Any]]:
    raw_tools = read_jsonl(domain_dir / "tools.jsonl")
    converted_tools = []
    seen_names = set()
    for line_number, raw_tool in raw_tools:
        missing_fields = [field_name for field_name in TOOL_REQUIRED_FIELDS if field_name not in raw_tool]
        if missing_fields:
            reason = f"missing_{missing_fields[0]}" if len(missing_fields) == 1 else "missing_tool_fields"
            raise ArtifactValidationError(
                reason, f"tool line {line_number} missing fields: {', '.join(missing_fields)}"
            )
        tool_name = raw_tool["name"]
        if tool_name in seen_names:
            raise ArtifactValidationError("duplicate_tool_name", f"duplicate tool name: {tool_name}")
        seen_names.add(tool_name)
        validate_tool_schema(raw_tool["params"], tool_name=tool_name, field_name="params")
        validate_tool_schema(raw_tool["returns"], tool_name=tool_name, field_name="returns")
        converted_tool = {
            "name": tool_name,
            "doc": raw_tool["doc"],
            "params": raw_tool["params"],
            "description": raw_tool["doc"],
            "parameters": raw_tool["params"],
            "returns": raw_tool["returns"],
            "strict": True,
        }
        ToolSignature.model_validate(converted_tool)
        converted_tools.append(converted_tool)
    if not converted_tools:
        raise ArtifactValidationError("empty_tools", "tools.jsonl contains no tools")
    return converted_tools


def agent_system_message(policy: str, parallel_tool_calls: bool = False) -> str:
    template = (
        SyntheticToolUseAgent.AGENT_PARALLEL_SYSTEM_MESSAGE_TEMPLATE
        if parallel_tool_calls
        else SyntheticToolUseAgent.AGENT_SYSTEM_MESSAGE_TEMPLATE
    )
    return template.format(domain_policy=policy)


def responses_api_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["doc"],
            "parameters": tool["params"],
            "strict": True,
        }
        for tool in tools
    ]


def scenario_files(domain_dir: Path) -> list[Path]:
    scenarios_dir = domain_dir / "scenarios"
    if not scenarios_dir.is_dir():
        return []
    return sorted(scenarios_dir.glob("**/*.jsonl"))


def scenario_artifact_reference(scenario_path: Path) -> tuple[str | None, str]:
    parts = scenario_path.parts
    if "scenarios" not in parts:
        return None, scenario_path.name
    scenarios_index = parts.index("scenarios")
    domain_dir_name = parts[scenarios_index - 1] if scenarios_index > 0 else None
    return domain_dir_name, "/".join(parts[scenarios_index:])


def validate_scenario(raw_scenario: dict[str, Any]) -> dict[str, Any]:
    missing_fields = [field_name for field_name in SCENARIO_REQUIRED_FIELDS if field_name not in raw_scenario]
    if missing_fields:
        reason = f"missing_{missing_fields[0]}" if len(missing_fields) == 1 else "missing_scenario_fields"
        raise ArtifactValidationError(reason, f"scenario missing fields: {', '.join(missing_fields)}")
    scenario = CustomerScenario.model_validate(raw_scenario)
    return scenario.model_dump()


def domain_name_for_row(domain_dir: Path, scenario: dict[str, Any]) -> str:
    representative_domain = scenario.get("representative_domain")
    if representative_domain:
        return str(representative_domain)
    return domain_dir.name


def maybe_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def scenario_file_index(scenario_path: Path) -> int | None:
    suffix = scenario_path.stem.rsplit("_", maxsplit=1)[-1]
    return maybe_int(suffix)


def scenario_generator_run(domain_dir: Path, scenario_path: Path) -> str | None:
    scenarios_dir = domain_dir / "scenarios"
    try:
        relative_parts = scenario_path.relative_to(scenarios_dir).parts
    except ValueError:
        return None
    if len(relative_parts) <= 1:
        return None
    return relative_parts[0]


def source_manifest_metadata(source_dir: Path) -> dict[str, Any]:
    candidate_paths = (source_dir / "run_manifest.json", source_dir.parent / "run_manifest.json")
    manifest_path = next((path for path in candidate_paths if path.is_file()), None)
    if manifest_path is None:
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    config = manifest.get("config", {})
    metadata: dict[str, Any] = {
        "seed_protocol_version": manifest.get("protocol_version"),
        "seed_run_id": manifest.get("run_id"),
        "seed_run_name": manifest.get("run_name"),
        "generation_profile": manifest.get("generation_profile"),
        "random_seed": manifest.get("random_seed"),
        "code_revision": manifest.get("code_revision"),
    }
    asset_hashes = manifest.get("asset_hashes")
    if isinstance(asset_hashes, dict) and asset_hashes:
        canonical_hashes = json.dumps(asset_hashes, sort_keys=True, separators=(",", ":"))
        metadata["seed_asset_hashes_sha256"] = hashlib.sha256(canonical_hashes.encode("utf-8")).hexdigest()
    role_mapping = {
        "domain_model": "domain_generator_model",
        "policy_tools_model": "policy_tools_model",
        "scenario_model": "scenario_generator_model",
    }
    for role_name, metadata_name in role_mapping.items():
        role = config.get(role_name)
        if isinstance(role, dict) and role.get("model"):
            metadata[metadata_name] = role["model"]
    return {key: value for key, value in metadata.items() if value is not None}


def domain_artifact_metadata(domain_dir: Path) -> dict[str, Any]:
    path = domain_dir / "domain.json"
    if not path.is_file():
        return {}
    try:
        domain = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        key: domain[key]
        for key in ("domain_id", "normalized_name", "generation_profile", "request_index", "candidate_index")
        if key in domain
    }


def build_row(
    metadata_config: DatasetMetadataConfig,
    source_name: str,
    source_index: int,
    domain_dir: Path,
    scenario_path: Path,
    scenario_line_number: int,
    scenario_index: int,
    policy: str,
    tools: list[dict[str, Any]],
    scenario: dict[str, Any],
    agent_name: str,
    parallel_tool_calls: bool = False,
    policy_temperature: float = DEFAULT_POLICY_TEMPERATURE,
    seed_metadata: dict[str, Any] | None = None,
    domain_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    domain_name = domain_name_for_row(domain_dir, scenario)
    source_domain_index = maybe_int(domain_dir.name)
    tool_names = [tool["name"] for tool in tools]
    relative_scenario_file = str(scenario_path.relative_to(domain_dir))
    row_metadata = {
        **metadata_config.to_dict(),
        **(seed_metadata or {}),
        **(domain_metadata or {}),
        "source_name": source_name,
        "source_index": source_index,
        "domain_name": domain_name,
        "representative_domain": scenario.get("representative_domain"),
        "source_domain_index": source_domain_index,
        "domain_dir_name": domain_dir.name,
        "num_tools": len(tools),
        "tool_names": tool_names,
        "scenario_index": scenario_index,
        "scenario_file": relative_scenario_file,
        "scenario_file_index": scenario_file_index(scenario_path),
        "scenario_line": scenario_line_number,
        "scenario_generator_run": scenario_generator_run(domain_dir, scenario_path),
        "outside_policy_scope": scenario.get("outside_policy_scope"),
        "policy_num_chars": len(policy),
        "parallel_tool_calls": parallel_tool_calls,
    }
    row = {
        "id": f"{metadata_config.dataset_name}_{source_index}_{domain_dir.name}_{scenario_path.stem}_{scenario_line_number:06d}",
        "domain_name": domain_name,
        "policy": policy,
        "tools": tools,
        "customer_scenario": scenario,
        "metadata": row_metadata,
        "responses_create_params": {
            "input": [
                {
                    "role": "system",
                    "content": agent_system_message(policy, parallel_tool_calls=parallel_tool_calls),
                }
            ],
            "parallel_tool_calls": parallel_tool_calls,
            "temperature": policy_temperature,
            "tools": responses_api_tools(tools),
        },
        "agent_ref": {
            "type": "responses_api_agents",
            "name": agent_name,
        },
        "source_artifacts": {
            "source_name": source_name,
            "source_index": source_index,
            "domain_dir_name": domain_dir.name,
            "scenario_file": relative_scenario_file,
            "scenario_index": scenario_index,
            "scenario_line": scenario_line_number,
            "source_domain_index": source_domain_index,
            "scenario_generator_run": row_metadata["scenario_generator_run"],
            **(domain_metadata or {}),
            **{
                key: value
                for key, value in (seed_metadata or {}).items()
                if key
                in {
                    "seed_protocol_version",
                    "seed_run_id",
                    "seed_run_name",
                    "seed_asset_hashes_sha256",
                    "generation_profile",
                }
            },
        },
    }
    SyntheticToolUseSeedSessionRequest.model_validate(row)
    NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    return row


def validate_domain_static_artifacts(domain_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    policy_path = domain_dir / "policy.md"
    tools_path = domain_dir / "tools.jsonl"
    if not policy_path.is_file():
        raise ArtifactValidationError("missing_policy", "missing policy.md")
    if not tools_path.is_file():
        raise ArtifactValidationError("missing_tools", "missing tools.jsonl")
    scenario_paths = scenario_files(domain_dir)
    if not scenario_paths:
        raise ArtifactValidationError("missing_scenarios", "missing scenario jsonl files")

    policy = load_policy(domain_dir)
    leak_reason = detect_leak(policy)
    if leak_reason:
        raise ArtifactValidationError(leak_reason, f"{leak_reason} in policy.md")

    tools_text = tools_path.read_text(encoding="utf-8")
    leak_reason = detect_leak(tools_text)
    if leak_reason:
        raise ArtifactValidationError(leak_reason, f"{leak_reason} in tools.jsonl")
    tools = load_tools(domain_dir)
    return policy, tools


def accepted_rows_for_source(
    source_dir: Path,
    source_name: str,
    source_index: int,
    output_file,
    metadata_config: DatasetMetadataConfig,
    max_rows: int | None,
    max_rows_per_domain: int | None,
    scan_domains: int | None,
    agent_name: str,
    skip_examples_limit: int,
    parallel_tool_calls: bool,
    policy_temperature: float,
) -> tuple[int, SourceReport]:
    rows_written = 0
    report = SourceReport(source_name=source_name, source_index=source_index)
    seed_metadata = source_manifest_metadata(source_dir)
    for domain_dir in numeric_domain_dirs(source_dir):
        if scan_domains is not None and report.domains_seen >= scan_domains:
            break
        report.domains_seen += 1
        try:
            policy, tools = validate_domain_static_artifacts(domain_dir)
        except ArtifactValidationError as exc:
            report.record_domain_skip(domain_dir, exc.reason, exc.detail, skip_examples_limit)
            continue
        except json.JSONDecodeError as exc:
            reason = "malformed_json"
            report.record_domain_skip(domain_dir, reason, str(exc), skip_examples_limit)
            continue

        if max_rows is not None and rows_written >= max_rows:
            continue

        domain_had_accepted_scenario = False
        domain_row_count = 0
        domain_scenario_index = 0
        for scenario_path in scenario_files(domain_dir):
            if max_rows is not None and rows_written >= max_rows:
                break
            if max_rows_per_domain is not None and domain_row_count >= max_rows_per_domain:
                break
            try:
                for line_number, raw_scenario in read_jsonl(scenario_path):
                    if max_rows is not None and rows_written >= max_rows:
                        break
                    if max_rows_per_domain is not None and domain_row_count >= max_rows_per_domain:
                        break
                    report.scenarios_seen += 1
                    current_scenario_index = domain_scenario_index
                    domain_scenario_index += 1
                    serialized_scenario = json.dumps(raw_scenario, ensure_ascii=False)
                    leak_reason = detect_leak(serialized_scenario)
                    if leak_reason:
                        report.record_scenario_skip(scenario_path, leak_reason, leak_reason, skip_examples_limit)
                        continue
                    try:
                        scenario = validate_scenario(raw_scenario)
                        row = build_row(
                            metadata_config=metadata_config,
                            source_name=source_name,
                            source_index=source_index,
                            domain_dir=domain_dir,
                            scenario_path=scenario_path,
                            scenario_line_number=line_number,
                            scenario_index=current_scenario_index,
                            policy=policy,
                            tools=tools,
                            scenario=scenario,
                            agent_name=agent_name,
                            parallel_tool_calls=parallel_tool_calls,
                            policy_temperature=policy_temperature,
                            seed_metadata=seed_metadata,
                            domain_metadata=domain_artifact_metadata(domain_dir),
                        )
                    except ArtifactValidationError as exc:
                        report.record_scenario_skip(scenario_path, exc.reason, exc.detail, skip_examples_limit)
                        continue
                    except (ValueError, TypeError) as exc:
                        report.record_scenario_skip(scenario_path, "invalid_scenario", str(exc), skip_examples_limit)
                        continue
                    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_written += 1
                    domain_row_count += 1
                    report.rows_written += 1
                    report.scenarios_accepted += 1
                    domain_had_accepted_scenario = True
            except json.JSONDecodeError as exc:
                report.record_scenario_skip(scenario_path, "malformed_json", str(exc), skip_examples_limit)
        if domain_had_accepted_scenario:
            report.domains_accepted += 1
    return rows_written, report


def build_sample_dataset(
    source_dirs: list[Path],
    output_path: Path,
    report_path: Path,
    max_rows: int | None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    source_names: list[str] | None = None,
    agent_name: str = DEFAULT_AGENT_NAME,
    max_rows_per_domain: int | None = 1,
    scan_domains_per_source: int | None = 100,
    domain_generator_model: str = DEFAULT_DOMAIN_GENERATOR_MODEL,
    policy_tools_model: str = DEFAULT_POLICY_TOOLS_MODEL,
    scenario_generator_model: str = DEFAULT_SCENARIO_GENERATOR_MODEL,
    skip_examples_limit: int = 50,
    parallel_tool_calls: bool = False,
    policy_temperature: float = DEFAULT_POLICY_TEMPERATURE,
) -> dict[str, Any]:
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when set")
    if max_rows_per_domain is not None and max_rows_per_domain <= 0:
        raise ValueError("max_rows_per_domain must be positive when set")
    if source_names is None:
        source_names = default_source_names_for(source_dirs)
    if len(source_names) != len(source_dirs):
        raise ValueError("source_names must have the same length as source_dirs")
    metadata_config = DatasetMetadataConfig(
        dataset_name=dataset_name,
        domain_generator_model=domain_generator_model,
        policy_tools_model=policy_tools_model,
        scenario_generator_model=scenario_generator_model,
    )
    source_count = max(1, len(source_dirs))
    rows_per_source = None if max_rows is None else max(1, (max_rows + source_count - 1) // source_count)
    rows_written = 0
    reports = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for source_index, (source_dir, source_name) in enumerate(zip(source_dirs, source_names)):
            source_max_rows = None
            if rows_per_source is not None:
                source_max_rows = min(rows_per_source, max_rows - rows_written)
                if source_max_rows <= 0:
                    source_max_rows = 0
            written_for_source, report = accepted_rows_for_source(
                source_dir=source_dir,
                source_name=source_name,
                source_index=source_index,
                output_file=f,
                metadata_config=metadata_config,
                max_rows=source_max_rows,
                max_rows_per_domain=max_rows_per_domain,
                scan_domains=scan_domains_per_source,
                agent_name=agent_name,
                skip_examples_limit=skip_examples_limit,
                parallel_tool_calls=parallel_tool_calls,
                policy_temperature=policy_temperature,
            )
            rows_written += written_for_source
            reports.append(report)

    report_payload = {
        "output_path": str(output_path),
        "rows_written": rows_written,
        "metadata": metadata_config.to_dict(),
        "source_names": source_names,
        "max_rows": max_rows,
        "max_rows_per_domain": max_rows_per_domain,
        "scan_domains_per_source": scan_domains_per_source,
        "agent_name": agent_name,
        "parallel_tool_calls": parallel_tool_calls,
        "policy_temperature": policy_temperature,
        "sources": [report.to_dict() for report in reports],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_payload


def default_source_names_for(source_dirs: list[Path]) -> list[str]:
    names = []
    try:
        default_source_dirs = default_source_dirs_from_env()
    except ValueError:
        default_source_dirs = []
    for source_dir in source_dirs:
        if source_dir in default_source_dirs:
            names.append(DEFAULT_SOURCE_NAMES[default_source_dirs.index(source_dir)])
        else:
            names.append(f"source_{len(names)}")
    return names


def default_source_dirs_from_env() -> list[Path]:
    source_dirs = []
    missing_vars = []
    for env_var in DEFAULT_SOURCE_ENV_VARS:
        value = os.environ.get(env_var)
        if not value:
            missing_vars.append(env_var)
        else:
            source_dirs.append(Path(value))
    if missing_vars:
        raise ValueError(
            "Default source dirs require these environment variables: "
            + ", ".join(missing_vars)
            + ". Alternatively pass --source-dir once per source."
        )
    return source_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        action="append",
        type=Path,
        default=None,
        help="Raw seed policy directory. May be passed more than once.",
    )
    parser.add_argument(
        "--source-name",
        action="append",
        default=None,
        help="Neutral source name to store in row metadata. May be passed once per source-dir.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--max-rows-per-domain", type=int, default=1)
    parser.add_argument(
        "--scan-domains-per-source",
        type=int,
        default=100,
        help="Number of numeric domains to validate per source. Use 0 to scan all numeric domains.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--domain-generator-model", default=DEFAULT_DOMAIN_GENERATOR_MODEL)
    parser.add_argument("--policy-tools-model", default=DEFAULT_POLICY_TOOLS_MODEL)
    parser.add_argument("--scenario-generator-model", default=DEFAULT_SCENARIO_GENERATOR_MODEL)
    parser.add_argument("--policy-temperature", type=float, default=DEFAULT_POLICY_TEMPERATURE)
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--skip-examples-limit", type=int, default=50)
    parser.add_argument(
        "--parallel-tool-calls",
        action="store_true",
        help="Materialize responses_create_params with parallel_tool_calls=true and a parallel-capable agent prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_path
    report_path = args.report_path or output_path.with_suffix(".report.json")
    source_dirs = args.source_dir or default_source_dirs_from_env()
    source_names = args.source_name or default_source_names_for(source_dirs)
    max_rows = args.max_rows if args.max_rows > 0 else None
    max_rows_per_domain = args.max_rows_per_domain if args.max_rows_per_domain > 0 else None
    scan_domains_per_source = args.scan_domains_per_source if args.scan_domains_per_source > 0 else None
    report = build_sample_dataset(
        source_dirs=source_dirs,
        output_path=output_path,
        report_path=report_path,
        max_rows=max_rows,
        dataset_name=args.dataset_name,
        source_names=source_names,
        agent_name=args.agent_name,
        max_rows_per_domain=max_rows_per_domain,
        scan_domains_per_source=scan_domains_per_source,
        domain_generator_model=args.domain_generator_model,
        policy_tools_model=args.policy_tools_model,
        scenario_generator_model=args.scenario_generator_model,
        skip_examples_limit=args.skip_examples_limit,
        parallel_tool_calls=args.parallel_tool_calls,
        policy_temperature=args.policy_temperature,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
