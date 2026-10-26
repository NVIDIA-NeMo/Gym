# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run metadata kept with a result: allowlisted identifiers, fingerprints of prompts and inputs, and the
resolved config with secrets masked."""

import logging
import urllib.request
from collections.abc import Callable, Mapping
from hashlib import sha256
from itertools import islice
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

import orjson
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, Field, JsonValue

from nemo_gym.orchestration.jobs import installed_gym_commit
from nemo_gym.package_info import __version__
from nemo_gym.path_utils import provenance_path_for
from nemo_gym.prompt import load_prompt_config


logger = logging.getLogger(__name__)
INPUT_SAMPLING_ROW_LIMIT = 100
# Names settings found at the top level of the config, such as `policy_model_name` in a `--no-serve` run.
TOP_LEVEL_PATH = "<top-level>"


_SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_tokens",
        "max_output_tokens",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "seed",
        "n",
    }
)
_MODEL_KEYS = frozenset(
    {
        "model",
        "openai_model",
        "model_name",
        "served_model_name",
        "checkpoint",
        "checkpoint_path",
        "policy_model_name",
        "judge_model_name",
    }
)


#: Seconds to wait for one model backend's `/models` answer; a slow or absent backend never blocks the run.
SERVED_MODEL_TIMEOUT_S = 5.0
#: Backends asked per model server; a long `base_url` list is a replica set that normally agrees.
SERVED_MODEL_BACKEND_LIMIT = 8


class ServedBackend(BaseModel):
    """What one model backend reports serving at `/models`, or why that is unknown."""

    id: str | None = None
    root: str | None = None
    unknown_reason: str | None = None


class RunProvenance(BaseModel):
    """Reproducibility metadata; unknown values stay null and credentials are never selected."""

    schema_version: int = 1
    status: Literal["available", "unavailable"] = "available"
    error_type: str | None = None
    input_sampling_rows: int = 0
    input_sampling_rows_skipped: int = 0
    input_sampling_row_limit: int = INPUT_SAMPLING_ROW_LIMIT
    gym_version: str = __version__
    gym_commit: str | None = None
    models: dict[str, dict[str, str]] = Field(default_factory=dict)
    datasets: list[str] = Field(default_factory=list)
    materialized_inputs_sha256: str | None = None
    prompt_config: dict[str, str] | None = None
    prompt_config_paths: dict[str, str] = Field(default_factory=dict)
    template_sha256: dict[str, str] = Field(default_factory=dict)
    judge_prompt_versions: dict[str, str] = Field(default_factory=dict)
    sampling_parameters: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    input_sampling_parameters: list[dict[str, JsonValue]] = Field(default_factory=list)
    seed: int | None = None
    # What each model server's backends report serving, keyed by model server name, beside the configured `models`.
    served_models: dict[str, list[ServedBackend]] = Field(default_factory=dict)
    # Why `served_models` is empty: no model server could be found or asked. Absent when it was filled.
    served_models_unknown_reason: str | None = None
    # A versioned Eval Factory config the caller ran this under, recorded as given.
    efb_config_ref: str | None = None


def _identifier(value: str) -> str:
    if "://" not in value:
        userinfo, separator, remainder = value.partition("@")
        if not (separator and "/" not in userinfo):
            return value
        parsed = urlsplit("//" + remainder)
        host = parsed.hostname or ""
        if not ("." in host or ":" in host or host == "localhost" or parsed.port is not None or ":" in userinfo):
            return value
        return urlunsplit(("", parsed.netloc, parsed.path, "", "")).removeprefix("//")
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))


def _sampling(config: Mapping) -> dict[str, JsonValue]:
    return {
        key: value
        for key, value in config.items()
        if key in _SAMPLING_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def build_provenance(
    config: Mapping[str, object], *, materialized_fpath: Path | None = None, prompt_config: str | None = None
) -> RunProvenance:
    """Snapshot model, judge, template and sampling settings. The masked config itself is saved separately."""
    record = RunProvenance(gym_commit=installed_gym_commit())
    efb_config_ref = config.get("efb_config_ref")
    if isinstance(efb_config_ref, str):
        record.efb_config_ref = _identifier(efb_config_ref)
    seed = config.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        record.seed = seed

    def visit(node: Mapping, path: str) -> None:
        model = {
            key: _identifier(value) for key, value in node.items() if key in _MODEL_KEYS and isinstance(value, str)
        }
        if model:
            record.models[path or TOP_LEVEL_PATH] = model
        sampling = _sampling(node)
        if sampling:
            record.sampling_parameters[path or TOP_LEVEL_PATH] = sampling
        for key, value in node.items():
            location = f"{path}.{key}" if path else str(key)
            if key in {"input_jsonl_fpath", "dataset_name", "dataset_id"} and isinstance(value, str):
                record.datasets.append(_identifier(value))
            if key == "prompt_config" and isinstance(value, str):
                record.prompt_config_paths[location] = _identifier(value)
            if (
                key in {"chat_template", "chat_template_kwargs", "judge_prompt", "judge_prompt_template"}
                and value is not None
            ):
                template = (
                    OmegaConf.to_container(value, resolve=True)
                    if isinstance(value, (DictConfig, ListConfig))
                    else value
                )
                record.template_sha256[location] = sha256(
                    orjson.dumps(template, option=orjson.OPT_SORT_KEYS)
                ).hexdigest()
            elif key == "judge_prompt_version" and isinstance(value, (str, int)):
                record.judge_prompt_versions[location] = str(value)
            elif isinstance(value, Mapping) and key not in {
                "env",
                "headers",
                "default_headers",
                "openai_default_headers",
                "metadata",
                "extra_body",
            }:
                visit(value, location)
            if key == "extra_body" and isinstance(value, Mapping):
                selected = _sampling(value)
                if selected:
                    record.sampling_parameters[location] = selected

    visit(config, "")
    record.datasets = sorted(set(record.datasets))
    if prompt_config:
        prompt = load_prompt_config(prompt_config)
        record.prompt_config = {
            "path": _identifier(prompt_config),
            "sha256": sha256(orjson.dumps(prompt.model_dump(), option=orjson.OPT_SORT_KEYS)).hexdigest(),
        }
    if materialized_fpath is not None:
        digest = sha256()
        datasets: set[str] = set(record.datasets)
        sampling_profiles: dict[bytes, dict[str, JsonValue]] = {}
        with materialized_fpath.open("rb") as inputs:
            for chunk in iter(lambda: inputs.read(1024 * 1024), b""):
                digest.update(chunk)
            inputs.seek(0)
            # Bound JSON decoding: later rows still contribute to the exact byte hash.
            for line in islice(inputs, INPUT_SAMPLING_ROW_LIMIT):
                if not line.strip():
                    continue
                row = orjson.loads(line)
                record.input_sampling_rows += 1
                params = row.get("responses_create_params") or {}
                if not isinstance(params, Mapping):
                    record.input_sampling_rows_skipped += 1
                    continue
                metadata = params.get("metadata")
                if metadata is None:
                    metadata = {}
                if not isinstance(metadata, Mapping):
                    record.input_sampling_rows_skipped += 1
                    continue
                source = row.get("task_source")
                if isinstance(source, str):
                    datasets.add(_identifier(source))
                selected = _sampling(params)
                extra = metadata.get("extra_body") or {}
                if isinstance(extra, str):
                    try:
                        extra = orjson.loads(extra)
                    except orjson.JSONDecodeError:
                        extra = {}
                if isinstance(extra, Mapping):
                    selected.update(_sampling(extra))
                if selected:
                    sampling_profiles[orjson.dumps(selected, option=orjson.OPT_SORT_KEYS)] = selected
        record.materialized_inputs_sha256 = digest.hexdigest()
        record.datasets = sorted(datasets)
        record.input_sampling_parameters = [sampling_profiles[key] for key in sorted(sampling_profiles)]
    return record


def _backend_urls(server: Mapping) -> list[str]:
    endpoint_file = server.get("endpoint_file")
    if isinstance(endpoint_file, str) and endpoint_file:
        return [Path(endpoint_file).read_text(encoding="utf-8").strip()]
    # `base_url` (vLLM), `openai_base_url` (OpenAI-compatible), `switchyard_base_url`, ...
    base_url = next((value for key, value in server.items() if str(key).endswith("base_url") and value), None)
    urls = [base_url] if isinstance(base_url, str) else list(base_url or [])
    return [url for url in urls if isinstance(url, str)][:SERVED_MODEL_BACKEND_LIMIT]


def _ask_backend(base_url: str, api_key: object, configured: set[str]) -> list[ServedBackend]:
    request = urllib.request.Request(base_url.rstrip("/") + "/models")
    if isinstance(api_key, str) and api_key and not api_key.startswith("${"):
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=SERVED_MODEL_TIMEOUT_S) as response:
            models = orjson.loads(response.read()).get("data") or []
    except Exception as error:
        # The message may carry the URL and its credentials; keep only the type.
        return [ServedBackend(unknown_reason=type(error).__name__)]
    served = [
        ServedBackend(
            id=_identifier(model["id"]) if isinstance(model.get("id"), str) else None,
            root=_identifier(model["root"]) if isinstance(model.get("root"), str) else None,
        )
        for model in models
        if isinstance(model, Mapping)
    ]
    if len(served) > 1:
        # A hosted gateway lists its whole catalog; keep the configured model's entry. A single listed model is kept
        # as is, so a name or checkpoint that differs from the claim stays visible.
        matched = [backend for backend in served if backend.id in configured]
        return matched or [ServedBackend(unknown_reason=f"configured model not among {len(served)} listed")]
    return served or [ServedBackend(unknown_reason="no model listed")]


def _is_model_server(node: object) -> bool:
    return isinstance(node, Mapping) and isinstance(node.get("responses_api_models"), Mapping)


def served_models(config: Mapping[str, object]) -> dict[str, list[ServedBackend]]:
    """Ask each model server's backends what they serve (`GET <base_url>/models`), once per run.

    `root` is the checkpoint a vLLM backend loaded; hosted APIs report no `root`, which stays unknown. Nothing is
    taken from the configured names, so a claimed model can be checked against the served one."""
    result: dict[str, list[ServedBackend]] = {}
    for name, node in config.items():
        if not _is_model_server(node):
            continue
        backends: list[ServedBackend] = []
        for server in node["responses_api_models"].values():
            if not isinstance(server, Mapping):
                continue
            try:
                urls = _backend_urls(server)
                api_key = next((value for key, value in server.items() if str(key).endswith("api_key")), None)
                configured = {
                    _identifier(value)
                    for key, value in server.items()
                    if key in ("model", "openai_model", "served_model_name") and isinstance(value, str)
                }
            except Exception as error:
                backends.append(ServedBackend(unknown_reason=type(error).__name__))
                continue
            if not urls:
                backends.append(ServedBackend(unknown_reason="no base_url"))
            for url in urls:
                backends.extend(_ask_backend(url, api_key, configured))
        result[str(name)] = backends
    return result


def write_run_provenance(
    output_fpath: Path,
    config: Mapping[str, object],
    *,
    prompt_config: str | None = None,
    input_jsonl_fpath: str | None = None,
    model_server_config: Callable[[], Mapping[str, object]] | None = None,
) -> Path:
    """Write one provenance artifact next to a run's materialized inputs before dispatch.

    ``model_server_config`` returns the config that names the run's model servers when ``config`` does not: a
    ``--no-serve`` collection sees only its own config, and the model servers' entries come from the head server."""
    from nemo_gym.path_utils import materialized_path_for

    try:
        record = build_provenance(
            config, materialized_fpath=materialized_path_for(output_fpath), prompt_config=prompt_config
        )
        if input_jsonl_fpath is not None:
            record.datasets = sorted({*record.datasets, _identifier(input_jsonl_fpath)})
        try:
            servers = config
            if not any(_is_model_server(node) for node in config.values()) and model_server_config is not None:
                servers = model_server_config()
            record.served_models = served_models(servers)
            if not record.served_models:
                record.served_models_unknown_reason = "no model server in the run's config"
        except (Exception, SystemExit) as error:
            # Reporting never stops the run, not even when finding the head server exits on its arguments.
            # The message may carry a URL or a head-server address; keep only the type.
            record.served_models_unknown_reason = type(error).__name__
    except Exception as error:
        # Provenance is reporting, not a prerequisite for executing valid input rows.
        # Exception messages may contain prompts or credentials; retain only the type.
        logger.warning("Run provenance unavailable (%s).", type(error).__name__)
        record = RunProvenance(status="unavailable", error_type=type(error).__name__)
    path = provenance_path_for(output_fpath)
    try:
        path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        logger.warning("Could not write run provenance (%s).", type(error).__name__)
    return path
