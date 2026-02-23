# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from abc import abstractmethod
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
from shutil import copyfileobj
from typing import Any, Dict, List, Literal, Optional, Self, Tuple, Union

from devtools import pprint
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tdigest import TDigest
from tqdm.auto import tqdm

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.config_types import (
    AGENT_REF_KEY,
    AgentServerRef,
    BaseNeMoGymCLIConfig,
    DatasetConfig,
    DatasetType,
    DownloadJsonlDatasetGitlabConfig,
    ServerInstanceConfig,
)
from nemo_gym.gitlab_utils import download_jsonl_dataset
from nemo_gym.global_config import (
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_global_config_dict,
)


class TrainDataProcessorConfig(BaseNeMoGymCLIConfig):
    """
    Prepare and validate training data, generating metrics and statistics for datasets.

    Examples:

    ```bash
    config_paths="resources_servers/example_multi_step/configs/example_multi_step.yaml,\\
    responses_api_models/openai_model/configs/openai_model.yaml"
    ng_prepare_data "+config_paths=[${config_paths}]" \
        +output_dirpath=data/example_multi_step \
        +mode=example_validation
    ```
    """

    output_dirpath: str = Field(description="Directory path where processed datasets and metrics will be saved.")
    mode: Union[Literal["train_preparation"], Literal["example_validation"]] = Field(
        description="Processing mode: 'train_preparation' prepares train/validation datasets for training, 'example_validation' validates example data for PR submission."
    )
    should_download: bool = Field(
        default=False,
        description="Whether to automatically download missing datasets from remote registries (default: False).",
    )

    @property
    def in_scope_dataset_types(self) -> List[DatasetType]:
        if self.mode == "train_preparation":
            return ["train", "validation"]
        elif self.mode == "example_validation":
            return ["example"]
        else:
            raise NotImplementedError


class Accumulator(BaseModel):
    is_aggregated: bool = Field(default=False, exclude=True)

    def add(self: Self, other: Self) -> None:
        assert not self.is_aggregated
        assert not other.is_aggregated
        self._add(other)

    @abstractmethod
    def _add(self: Self, other: Self) -> None:
        pass

    def aggregate(self: Self) -> Self:
        res = self._aggregate()
        res.is_aggregated = True
        return res

    @abstractmethod
    def _aggregate(self: Self) -> Self:
        pass


class AvgMinMax(Accumulator):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    total: int = Field(serialization_alias="Total # non-null values", default=0)
    average: float = Field(serialization_alias="Average", default=0)
    min: float = Field(serialization_alias="Min", default=float("inf"))
    max: float = Field(serialization_alias="Max", default=float("-inf"))
    median: float = Field(serialization_alias="Median", default=0)
    stddev: float = Field(serialization_alias="Standard deviation", default=0)
    # Internal state
    mean: float = Field(default=0, exclude=True)  # running value (before final average)
    M2: float = Field(default=0, exclude=True)  # sum of squared differences (for variance)
    tdigest: TDigest = Field(default_factory=TDigest, exclude=True)
    """
    T-Digest is used to estimate the Median without storing and sorting all values. The Median is essentially an approximation using the 50th percentile, which is very close to the true Median.
    """

    def observe(self, x: float) -> None:
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x

        # Update running mean and variance
        self.total += 1
        delta = x - self.mean
        self.mean += delta / self.total
        self.M2 += delta * (x - self.mean)

        # Update quantile estimator (for median)
        self.tdigest.update(x)

    def _add(self: Self, other: Self) -> None:
        # Merge accumulators
        if other.total == 0:
            return
        if self.total == 0:
            self.total = other.total
            self.mean = other.mean
            self.M2 = other.M2
            self.min = other.min
            self.max = other.max
            self.tdigest = TDigest()
            self.tdigest = self.tdigest + other.tdigest
            return

        # Merge mean and variance
        n1, n2 = self.total, other.total
        delta = other.mean - self.mean
        n = n1 + n2
        self.mean = self.mean + delta * (n2 / n)
        self.M2 = self.M2 + other.M2 + (delta * delta) * (n1 * n2 / n)
        self.total = n

        if other.min < self.min:
            self.min = other.min
        if other.max > self.max:
            self.max = other.max

        # Merge t-digests for quantiles/median
        self.tdigest = self.tdigest + other.tdigest

    def _aggregate(self: Self) -> Self:
        def round_metric(x: float) -> float:
            if x >= 1 or x <= -1:
                return round(x, 2)
            return round(x, 3)

        n = self.total
        mean = self.mean if n > 0 else 0.0
        stddev = sqrt(self.M2 / (n - 1)) if n > 1 else 0.0
        med = float(self.tdigest.percentile(50)) if n > 0 and self.tdigest.n > 0 else 0.0

        params = {
            "total": self.total,
            "average": mean,
            "min": self.min if n > 0 else 0.0,
            "max": self.max if n > 0 else 0.0,
            "median": med,
            "stddev": stddev,
        }

        final_params = {k: round_metric(v) if isinstance(v, float) else v for k, v in params.items()}

        return AvgMinMax(**final_params)


class StringMetrics(BaseModel):
    unique_count: int
    total_count: int


class DatasetMetrics(Accumulator):
    model_config = ConfigDict(extra="allow")  # Allow any arbitrary fields

    number_of_examples: int = Field(serialization_alias="Number of examples", default=0)
    number_of_tools: AvgMinMax = Field(serialization_alias="Number of tools", default_factory=AvgMinMax)
    json_dumped_number_of_words: AvgMinMax = Field(
        serialization_alias="Json-dumped number of words (proxy for token count)",
        default_factory=AvgMinMax,
    )
    number_of_turns: AvgMinMax = Field(serialization_alias="Number of turns", default_factory=AvgMinMax)
    temperature: AvgMinMax = Field(serialization_alias="Temperature", default_factory=AvgMinMax)

    # TODO: Number of unique create params, Number of unique user messages, other sampling params, etc

    def _add(self: Self, other: Self) -> None:
        self.number_of_examples += other.number_of_examples
        self.number_of_tools.add(other.number_of_tools)
        self.json_dumped_number_of_words.add(other.json_dumped_number_of_words)
        self.number_of_turns.add(other.number_of_turns)
        self.temperature.add(other.temperature)

        # Merge extra fields safely
        if other.model_extra:
            for k, v in other.model_extra.items():
                if k in DatasetMetrics.model_fields.keys():
                    continue
                setattr(self, k, v)

    def _aggregate(self: Self) -> Self:
        extras = {}
        if self.model_extra:
            for k, v in self.model_extra.items():
                if k in DatasetMetrics.model_fields.keys():
                    continue
                extras[k] = v
        return DatasetMetrics(
            number_of_examples=self.number_of_examples,
            number_of_tools=self.number_of_tools.aggregate(),
            json_dumped_number_of_words=self.json_dumped_number_of_words.aggregate(),
            number_of_turns=self.number_of_turns.aggregate(),
            temperature=self.temperature.aggregate(),
            **extras,
        )


def aggregate_other_metrics(metrics: Dict[str, Any], sample: Dict[str, Any]) -> None:
    """Combines misc items (those other than response/response create params) into current metrics"""
    for k, v in sample.items():
        if k in ("responses_create_params", "response"):
            continue

        values = v if isinstance(v, list) else [v]

        for item in values:
            if isinstance(item, bool):
                item = int(item)
            if isinstance(item, (int, float)):
                if k not in metrics:
                    metrics[k] = AvgMinMax()
                metrics[k].observe(item)
            elif isinstance(item, str):
                if k not in metrics:
                    metrics[k] = Counter()
                metrics[k][item] += 1


def postprocess_other_metrics(metrics: DatasetMetrics, other_metrics: Dict[str, Any]) -> None:
    """Aggregates metrics and merges current metrics (containing only AvgMinMax) with StringMetrics"""
    for k, v in other_metrics.items():
        if isinstance(v, AvgMinMax):
            setattr(metrics, k, v.aggregate())
        elif isinstance(v, Counter):
            setattr(metrics, k, StringMetrics(unique_count=len(v), total_count=sum(v.values())))


def compute_sample_metrics(sample_dict_str: str) -> Tuple[DatasetMetrics, bool]:
    try:
        sample_dict = json.loads(sample_dict_str)
    except json.JSONDecodeError:
        return DatasetMetrics(), True

    try:
        sample = BaseRunRequest.model_validate(sample_dict)
    except ValidationError:
        return DatasetMetrics(), True

    responses_create_params = sample.responses_create_params
    responses_create_params = responses_create_params.model_dump(exclude_unset=True)
    inputs = responses_create_params.get("input")

    number_of_tools_metrics = AvgMinMax()
    if responses_create_params.get("tools") is not None:
        number_of_tools = len(responses_create_params["tools"])
        number_of_tools_metrics.observe(number_of_tools)

    if isinstance(inputs, str):
        inputs = [{"role": "user", "content": inputs}]
    user_inputs = [i for i in inputs if i.get("role") == "user"] if inputs else []
    number_of_turns_metrics = AvgMinMax()
    if user_inputs:
        number_of_turns = len(user_inputs)
        number_of_turns_metrics.observe(number_of_turns)

    temperature_metrics = AvgMinMax()
    if responses_create_params.get("temperature") is not None:
        temperature = responses_create_params["temperature"]
        temperature_metrics.observe(temperature)

    json_dumped_number_of_words_metrics = AvgMinMax()
    json_dumped_number_of_words = len(json.dumps(responses_create_params).split())
    json_dumped_number_of_words_metrics.observe(json_dumped_number_of_words)

    metrics = DatasetMetrics(
        number_of_examples=1,
        number_of_tools=number_of_tools_metrics,
        json_dumped_number_of_words=json_dumped_number_of_words_metrics,
        number_of_turns=number_of_turns_metrics,
        temperature=temperature_metrics,
    )
    return metrics, False


class DatasetValidatorState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: DatasetMetrics = Field(default_factory=DatasetMetrics)
    key_counts: Counter = Field(default_factory=Counter)
    offending_example_idxs: List[int] = Field(default_factory=list)
    other_metrics: Dict[str, Any] = Field(default_factory=dict)


class TrainDataProcessor(BaseModel):
    def run(self, global_config_dict: DictConfig):  # pragma: no cover
        """
        See the README section "How To: Prepare and validate data for PR submission or RL training"
        """
        config = TrainDataProcessorConfig.model_validate(global_config_dict)

        self._print_title("Load and validate server instance configs")
        server_instance_configs = self.load_and_validate_server_instance_configs(config, global_config_dict)

        self._print_title(
            f"Load datasets. Missing datasets {'**WILL**' if config.should_download else 'will **NOT**'} be downloaded."
        )
        self.load_datasets(config, server_instance_configs)

        self._print_title("Validate samples and aggregate metrics")
        dataset_type_to_aggregate_metrics = self.validate_samples_and_aggregate_metrics(server_instance_configs)

        self._print_title("Collate samples and aggregate metrics")
        self.collate_samples(config, server_instance_configs, dataset_type_to_aggregate_metrics)

        self._print_title("Finished!")

    def _print_title(self, title: str) -> None:  # pragma: no cover
        print(f"""

{"#" * 100}
#
# {title}
#
{"#" * 100}
""")

    def load_and_validate_server_instance_configs(
        self, config: TrainDataProcessorConfig, global_config_dict: DictConfig
    ) -> List[ServerInstanceConfig]:
        parser = GlobalConfigDictParser()
        server_instance_configs = parser.filter_for_server_instance_configs(global_config_dict)

        agent_configs: List[ServerInstanceConfig] = [
            c for c in server_instance_configs if c.SERVER_TYPE == "responses_api_agents"
        ]

        server_names_list_str = "\n- ".join([""] + [f"{c.name} ({c.SERVER_TYPE})" for c in server_instance_configs])
        print(
            f"Found {len(server_instance_configs)} server instance configs ({len(agent_configs)} agent configs):{server_names_list_str}\n\n"
        )

        agent_configs_with_data: List[ServerInstanceConfig] = []
        agent_configs_without_data: List[ServerInstanceConfig] = []
        for agent_config in agent_configs:
            if agent_config.datasets:
                agent_configs_with_data.append(agent_config)
            else:
                agent_configs_without_data.append(agent_config)

        server_names_list_str = "\n- ".join([""] + [f"{c.name} ({c.SERVER_TYPE})" for c in agent_configs_without_data])
        print(
            f"Found {len(agent_configs_without_data)} agent server instance configs WITHOUT datasets:{server_names_list_str}\n\n"
        )

        server_names_list_str = ""
        for c in agent_configs_with_data:
            server_str = f"\n- {c.name}"
            datasets_str = "\n  - ".join([""] + [f"{d.name} ({d.type})" for d in c.datasets])
            server_names_list_str += f"{server_str}{datasets_str}"
        print(
            f"Found {len(agent_configs_with_data)} agent server instance configs WITH datasets:{server_names_list_str}\n\n"
        )

        # Filter for in scope depending on the mode.
        in_scope_dataset_types = config.in_scope_dataset_types
        agent_configs_with_in_scope_datasets: List[ServerInstanceConfig] = []
        for agent_config in agent_configs_with_data:
            in_scope_datasets = [d for d in agent_config.datasets if d.type in in_scope_dataset_types]
            if not in_scope_datasets:
                continue

            inner_config = agent_config.get_inner_run_server_config()
            inner_config.datasets = in_scope_datasets
            agent_configs_with_in_scope_datasets.append(agent_config)

        server_names_list_str = ""
        for c in agent_configs_with_in_scope_datasets:
            server_str = f"\n- {c.name}"
            datasets_str = "\n  - ".join([""] + [f"{d.name} ({d.type})" for d in c.datasets])
            server_names_list_str += f"{server_str}{datasets_str}"
        print(f"In scope dataset types for `{config.mode}` mode: {in_scope_dataset_types}")
        print(
            f"Found {len(agent_configs_with_data)} agent server instance configs with in-scope datasets:{server_names_list_str}"
        )

        return agent_configs_with_data

    def load_datasets(
        self,
        config: TrainDataProcessorConfig,
        server_instance_configs: List[ServerInstanceConfig],
    ) -> None:
        # Check if all the dataset paths exist. Mapping of server name to list of dataset config
        local_datasets_found: Dict[str, List[DatasetConfig]] = defaultdict(list)
        local_datasets_not_found: Dict[str, List[DatasetConfig]] = defaultdict(list)
        for c in server_instance_configs:
            for d in c.datasets:
                jsonl_fpath = Path(d.jsonl_fpath)
                if jsonl_fpath.exists():
                    local_datasets_found[c.name].append(d)
                else:
                    local_datasets_not_found[c.name].append(d)

        server_names_list_str = ""
        for server_name, datasets in local_datasets_found.items():
            datasets_str = "\n  - ".join([""] + [f"{d.name} ({d.type})" for d in datasets])
            server_names_list_str += f"\n- {server_name}{datasets_str}"
        print(f"FOUND the following datasets at their local paths:{server_names_list_str}\n\n")

        server_names_list_str = ""
        for server_name, datasets in local_datasets_not_found.items():
            datasets_str = "\n  - ".join([""] + [f"{d.name} ({d.type})" for d in datasets])
            server_names_list_str += f"\n- {server_name}{datasets_str}"
        print(f"MISSING the following datasets:{server_names_list_str}\n\n")

        if config.mode == "example_validation":
            assert not local_datasets_not_found, "You must provide the above missing example jsonl files!"
        if not config.should_download:
            assert not local_datasets_not_found, (
                "Missing local datasets. You must provide local datasets since download is disabled. Run with `+should_download=true` to enable downloading."
            )

        for (
            server_name,
            datasets,
        ) in local_datasets_not_found.items():  # pragma: no cover
            for d in datasets:
                download_config = DownloadJsonlDatasetGitlabConfig.model_validate(
                    d.gitlab_identifier.model_dump() | {"output_fpath": d.jsonl_fpath}
                )
                print(f"Downloading dataset `{d.name}` from `{server_name}` using {download_config}")
                download_jsonl_dataset(download_config)

    ########################################
    # Validate samples and aggregate metrics
    ########################################

    def _validate_samples_and_aggregate_metrics_single_sample(
        self, state: DatasetValidatorState, sample_idx: int, sample_dict_str: str
    ) -> None:
        metrics, is_offending = compute_sample_metrics(sample_dict_str)
        if is_offending:
            state.offending_example_idxs.append(sample_idx)
            return

        sample_dict = json.loads(sample_dict_str)
        state.key_counts.update(sample_dict.keys())
        state.metrics.add(metrics)

        aggregate_other_metrics(state.other_metrics, sample_dict)

    def _iter_dataset_lines(self, dataset_config: DatasetConfig):
        repeats = dataset_config.num_repeats

        # Print dataset repetition info
        if repeats > 1:
            print(
                f"Dataset {dataset_config.name} repeating {repeats}x: each line repeated {repeats} times (e.g. pattern: abc -> aaaabbbbcccc)"
            )

        # Don't load everything into memory at once. Throw things away immediately.
        with open(dataset_config.jsonl_fpath) as f:
            for line in tqdm(f, desc=f"{dataset_config.jsonl_fpath}"):
                for _ in range(repeats):
                    yield line

    def _validate_samples_and_aggregate_metrics_single_dataset(
        self, dataset_config: DatasetConfig
    ) -> DatasetValidatorState:
        state = DatasetValidatorState()

        map_fn = self._validate_samples_and_aggregate_metrics_single_sample
        for sample_idx, sample_dict_str in enumerate(self._iter_dataset_lines(dataset_config)):
            map_fn(state, sample_idx, sample_dict_str)

        postprocess_other_metrics(state.metrics, state.other_metrics)

        return state

    def _validate_aggregate_metrics(self, aggregate_metrics_dict: Dict, metrics_fpath: Path) -> Optional[Path]:
        """
        Returns the conflicting metrics fpath if invalid. Else returns None
        """
        if metrics_fpath.exists():
            with open(metrics_fpath) as f:
                previous_aggregate_metrics_dict = json.load(f)

            def numeric_close(a: float, b: float) -> bool:
                """Helper to compare numbers with a tolerance"""
                if a == b:
                    return True
                try:
                    a_f = float(a)
                    b_f = float(b)
                except Exception:
                    return False
                scale = max(abs(a_f), abs(b_f))  # Adjuster for tolerance

                # may need to adjust this threshold:
                tol = 5e-3 if scale >= 1 else 5e-4  # Higher threshold for larger numbers
                return abs(a_f - b_f) <= max(tol, 1e-9)  # Allow small differences

            def diff_values(prev_v, new_v, path: str, diffs: List[str]) -> None:
                """
                Recursively compare values at the given path.
                Keys from previous dict must be present in new dict.
                Additional fields in new dict are allowed.
                """
                if isinstance(prev_v, dict) and isinstance(new_v, dict):
                    for k in prev_v.keys():
                        sub_path = f"{path}.{k}" if path else k
                        if k not in new_v:
                            diffs.append(f"Missing key in new metrics: {sub_path}")
                            continue
                        diff_values(prev_v[k], new_v[k], sub_path, diffs)
                    return

                # Lists: Check for equality regardless of order
                if isinstance(prev_v, list) and isinstance(new_v, list):
                    if len(prev_v) != len(new_v):
                        diffs.append(f"List length differs at {path}: {len(prev_v)} != {len(new_v)}")
                        return
                    try:
                        prev_counter = Counter(prev_v)
                        new_counter = Counter(new_v)
                        if prev_counter != new_counter:
                            diffs.append(f"Multiset mismatch at {path}: {prev_counter} != {new_counter}")
                        return
                    except TypeError:
                        # Manual fallback for unhashable elements
                        used = set()
                        for i, pv in enumerate(prev_v):
                            found = False
                            for j, nv in enumerate(new_v):
                                if j in used:
                                    continue
                                sub_diffs = []
                                diff_values(pv, nv, f"{path}[{i}]", sub_diffs)
                                if not sub_diffs:
                                    used.add(j)
                                    found = True
                                    break
                            if not found:
                                diffs.append(f"No matching element for {path}[{i}] in new metrics (unordered)")
                        return

                if isinstance(prev_v, float) and isinstance(new_v, float):
                    if not numeric_close(prev_v, new_v):
                        diffs.append(f"Numeric mismatch at {path}: {prev_v} != {new_v}")
                    return

                if prev_v != new_v:
                    diffs.append(f"Value differs at {path}: {prev_v} != {new_v}")

            diffs: List[str] = []
            diff_values(previous_aggregate_metrics_dict, aggregate_metrics_dict, path="", diffs=diffs)

            if diffs:
                print("Differences found in aggregate metrics:")
                pprint(diffs)

                conflicting_metrics_fpath = metrics_fpath.with_name(f"{metrics_fpath.stem}_conflict.json")
                with open(conflicting_metrics_fpath, "w") as f:
                    json.dump(aggregate_metrics_dict, f, indent=4)

                return conflicting_metrics_fpath

    def validate_samples_and_aggregate_metrics(
        self, server_instance_configs: List[ServerInstanceConfig]
    ) -> Dict[str, DatasetMetrics]:
        conflicting_fpaths: List[str] = []
        dataset_type_to_aggregate_metrics: Dict[str, DatasetMetrics] = defaultdict(DatasetMetrics)
        for c in server_instance_configs:
            for d in c.datasets:
                state = self._validate_samples_and_aggregate_metrics_single_dataset(d)

                dataset_type_to_aggregate_metrics[d.type].add(state.metrics)

                aggregate_metrics = state.metrics.aggregate()

                aggregate_metrics_dict = aggregate_metrics.model_dump(mode="json", by_alias=True)
                aggregate_metrics_dict = d.model_dump() | aggregate_metrics_dict

                data_fpath = Path(d.jsonl_fpath)
                metrics_fpath = data_fpath.with_name(f"{data_fpath.stem}_metrics.json")

                maybe_conflicting_metrics_fpath = self._validate_aggregate_metrics(
                    aggregate_metrics_dict, metrics_fpath
                )
                if maybe_conflicting_metrics_fpath is not None:
                    conflicting_fpaths.append(str(maybe_conflicting_metrics_fpath))
                    continue

                with open(metrics_fpath, "w") as f:
                    json.dump(aggregate_metrics_dict, f, indent=4)

                print(f"Aggregate metrics for {metrics_fpath}")
                pprint(aggregate_metrics_dict)

        if conflicting_fpaths:
            conflicting_fpaths_str = "\n- ".join([""] + conflicting_fpaths)
            target_fpaths_str = "\n- ".join(
                [""] + [fp.replace("_conflict.json", ".json") for fp in conflicting_fpaths]
            )
            raise ValueError(f"""
Found conflicting aggregate metrics that need to be corrected:{conflicting_fpaths_str}

This could be due to a change in how metrics are calculated, leading to outdated metrics. Try deleting the below file(s) and rerunning data preparation:{target_fpaths_str}
""")

        return dict(dataset_type_to_aggregate_metrics)

    ########################################
    # Collate samples
    ########################################

    def _collate_samples_single_type(
        self,
        type: DatasetType,
        server_instance_configs: List[ServerInstanceConfig],
    ) -> List[Path]:
        paths_to_collate = []
        for c in server_instance_configs:
            for d in c.datasets:
                if d.type != type:
                    continue

                data_path = Path(d.jsonl_fpath)
                prepare_path = data_path.with_name(f"{data_path.stem}_prepare.jsonl")
                with open(prepare_path, "w") as target:
                    for line in self._iter_dataset_lines(d):
                        d = json.loads(line)
                        d[AGENT_REF_KEY] = AgentServerRef(type="responses_api_agents", name=c.name).model_dump()
                        target.write(f"{json.dumps(d)}\n")

                paths_to_collate.append(prepare_path)

        return paths_to_collate

    def collate_samples(
        self,
        config: TrainDataProcessorConfig,
        server_instance_configs: List[ServerInstanceConfig],
        dataset_type_to_aggregate_metrics: Dict[str, DatasetMetrics],
    ) -> None:
        final_fpaths: Dict[DatasetType, Path] = dict()
        conflicting_fpaths: List[str] = []
        for type in config.in_scope_dataset_types:
            if type not in dataset_type_to_aggregate_metrics:
                continue

            aggregate_metrics = dataset_type_to_aggregate_metrics[type]
            aggregate_metrics = aggregate_metrics.aggregate()

            aggregate_metrics_dict = aggregate_metrics.model_dump(mode="json", by_alias=True)

            # Add metadata for collated metrics (similar to validate step)
            parent = Path(config.output_dirpath)
            parent.mkdir(exist_ok=True)
            collated_fpath = parent / f"{type}.jsonl"

            # Get metadata from first dataset of this type
            dataset_metadata = {}
            for c in server_instance_configs:
                for d in c.datasets:
                    if d.type == type:
                        dataset_metadata = {
                            "name": type,
                            "type": type,
                            "jsonl_fpath": str(collated_fpath),
                            "num_repeats": 1,
                            "gitlab_identifier": d.gitlab_identifier.model_dump() if d.gitlab_identifier else None,
                            "license": d.license,
                        }
                        break
                if dataset_metadata:
                    break

            # Merge metadata with aggregate metrics
            aggregate_metrics_dict = dataset_metadata | aggregate_metrics_dict

            metrics_fpath = parent / f"{type}_metrics.json"
            maybe_conflicting_metrics_fpath = self._validate_aggregate_metrics(
                aggregate_metrics_dict=aggregate_metrics_dict,
                metrics_fpath=metrics_fpath,
            )
            if maybe_conflicting_metrics_fpath is not None:
                conflicting_fpaths.append(str(maybe_conflicting_metrics_fpath))
                continue

            with open(metrics_fpath, "w") as f:
                json.dump(aggregate_metrics_dict, f, indent=4)

            paths_to_collate = self._collate_samples_single_type(
                type=type,
                server_instance_configs=server_instance_configs,
            )
            collated_fpath = parent / f"{type}.jsonl"
            with open(collated_fpath, "wb") as outfile:
                for path in tqdm(paths_to_collate, desc=f"Collating {type} datasets"):
                    with open(path, "rb") as infile:
                        copyfileobj(infile, outfile)

            print(f"Aggregate metrics for {metrics_fpath}")
            pprint(aggregate_metrics_dict)

            final_fpaths[type] = collated_fpath

        if conflicting_fpaths:
            conflicting_fpaths_str = "\n- ".join([""] + conflicting_fpaths)
            target_fpaths_str = "\n- ".join(
                [""] + [fp.replace("_conflict.json", ".json") for fp in conflicting_fpaths]
            )
            raise ValueError(f"""
Found conflicting aggregate metrics that need to be corrected:{conflicting_fpaths_str}

This could be due to a change in how metrics are calculated, leading to outdated metrics. Try deleting the below file(s) and rerunning data preparation:{target_fpaths_str}
""")

        final_fpaths_str = "\n- ".join([""] + [f"{type}: {fpath}" for type, fpath in final_fpaths.items()])
        print(f"View your final data!{final_fpaths_str}")


def prepare_data():  # pragma: no cover
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )

    data_processor = TrainDataProcessor()
    data_processor.run(global_config_dict)
