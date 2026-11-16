# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seeded source-first sampling of raw Gym metadata, without rendering assets."""

# Adapted from https://github.com/NVIDIA-NeMo/RL, Apache-2.0,
# revision f2c87657d570be0e5293b57bd9b5e45e60087848,
# nemo_rl/data/source_operation_sampler.py.

import hashlib
import json
from array import array
from collections.abc import Iterable, Iterator
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator


class SourceOperationSamplingConfig(BaseModel, extra="forbid"):
    source_field: str = "deepcad_lazy.record.dataset"
    operations_field: str = "deepcad_lazy.record.reference_operations"
    sources: list[str] = ["zero-to-cad", "cadevolve", "op-cad", "histcad"]
    uniform_sources: list[str] = ["histcad"]
    tau: float = Field(default=1000.0, gt=0, allow_inf_nan=False)
    operation_fraction: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0)
    samples_per_epoch: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if not self.sources or len(set(self.sources)) != len(self.sources):
            raise ValueError("sources must be nonempty and unique")
        if any(not s for s in self.sources) or not set(self.uniform_sources) <= set(self.sources):
            raise ValueError("uniform_sources must be a subset of nonempty sources")
        for path in (self.source_field, self.operations_field):
            if any(not part for part in path.split(".")):
                raise ValueError("metadata fields must be nonempty dotted paths")
        return self


def validate_source_operation_sampling(data_config: dict[str, Any]) -> SourceOperationSamplingConfig | None:
    """Reject conflicting static balancing before it expands the dataset."""
    hooks = data_config.get("gym_dataset_hooks")
    if hooks is None:
        return None
    value = hooks["config"].get("source_operation_sampling")
    if value is None:
        return None
    config = SourceOperationSamplingConfig.model_validate(value)
    train = data_config["train"]
    train = train if isinstance(train, list) else [train]
    if len(train) != 1 or train[0].get("dataset_name") != "NemoGymDataset":
        raise ValueError("source_operation_sampling requires one NemoGymDataset manifest")
    defaults = data_config.get("default") or {}
    if train[0].get("source_balance", defaults.get("source_balance")) is not None:
        raise ValueError("Remove source_balance when using source_operation_sampling")
    if data_config.get("shuffle") or data_config.get("use_multiple_dataloader"):
        raise ValueError("source_operation_sampling requires shuffle=false and one dataloader")
    return config


class SourceOperationSampler:
    """Equal source quotas per batch; on-demand draws with replacement.

    Batch-specific RNG seeds make resume independent of prefetching. State is
    just the epoch/cursor and a fingerprint of metadata and sampling settings.
    Indexing reads raw JSON metadata only, never processed dataset items.
    """

    def __init__(
        self,
        raw_rows: Iterable[str],
        config: SourceOperationSamplingConfig,
        batch_size: int,
    ) -> None:
        self.config = SourceOperationSamplingConfig.model_validate(config)
        self.batch_size = batch_size
        if batch_size <= 0 or batch_size % len(config.sources):
            raise ValueError("Prompt batch size must be divisible by the number of sources")
        members = {s: array("q") for s in config.sources}
        buckets = {s: {} for s in config.sources}
        digest = hashlib.sha256()
        digest.update(config.model_dump_json().encode())
        digest.update(str(batch_size).encode())
        count = 0
        for index, raw in enumerate(raw_rows):
            digest.update(raw.encode())
            digest.update(b"\0")
            row = json.loads(raw)
            try:
                source = row
                for key in config.source_field.split("."):
                    source = source[key]
                if not isinstance(source, str) or source not in members:
                    raise ValueError(f"Unknown source at row {index}: {source!r}")
                members[source].append(index)
                if source not in config.uniform_sources:
                    ops = row
                    for key in config.operations_field.split("."):
                        ops = ops[key]
                    if not isinstance(ops, list) or any(not isinstance(o, str) or not o for o in ops):
                        raise ValueError(f"Invalid operation list at row {index}")
                    for op in sorted(set(ops)):
                        buckets[source].setdefault(op, array("q")).append(index)
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Missing sampling metadata at row {index}") from exc
            count += 1
        if any(not indices for indices in members.values()):
            raise ValueError("Every configured source must have at least one row")
        self.members = [np.asarray(members[s], dtype=np.int64) for s in config.sources]
        self.buckets = []
        self.operation_probabilities = []
        for source in config.sources:
            groups = [np.asarray(buckets[source][op], dtype=np.int64) for op in sorted(buckets[source])]
            if not groups and source not in config.uniform_sources and config.operation_fraction > 0:
                raise ValueError(f"No operations in {source}; explicitly add it to uniform_sources")
            weights = np.log1p(np.array([len(g) for g in groups], dtype=float) / config.tau)
            self.buckets.append(groups)
            self.operation_probabilities.append(weights / weights.sum() if groups else weights)
        self.num_samples = config.samples_per_epoch or (count // batch_size * batch_size)
        if self.num_samples == 0 or self.num_samples % batch_size:
            raise ValueError("samples_per_epoch must be a positive multiple of prompt batch size")
        self.fingerprint = digest.hexdigest()
        self.epoch = 0
        self.position = 0

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        if self.position == self.num_samples:
            self.epoch += 1
            self.position = 0
        while self.position < self.num_samples:
            batch_index, offset = divmod(self.position, self.batch_size)
            rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, self.epoch, batch_index]))
            batch = []
            per_source = self.batch_size // len(self.members)
            for members, groups, probabilities in zip(self.members, self.buckets, self.operation_probabilities):
                for _ in range(per_source):
                    if groups and rng.random() < self.config.operation_fraction:
                        group = groups[int(rng.choice(len(groups), p=probabilities))]
                    else:
                        group = members
                    batch.append(int(group[rng.integers(len(group))]))
            rng.shuffle(batch)
            for index in batch[offset:]:
                self.position += 1
                yield index

    def state_dict(self) -> dict[str, int | str]:
        return {
            "version": 1,
            "fingerprint": self.fingerprint,
            "epoch": self.epoch,
            "position": self.position,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1 or state.get("fingerprint") != self.fingerprint:
            raise ValueError("Sampler manifest/config changed; use a fresh dataloader state")
        epoch, position = state["epoch"], state["position"]
        if (
            not isinstance(epoch, int)
            or epoch < 0
            or not isinstance(position, int)
            or not 0 <= position <= self.num_samples
        ):
            raise ValueError("Invalid sampler checkpoint cursor")
        self.epoch, self.position = epoch, position


def source_operation_dataloader_kwargs(dataset: Any, data_config: dict[str, Any], batch_size: int) -> dict[str, Any]:
    config = validate_source_operation_sampling(data_config)
    if config is None:
        return {"shuffle": data_config["shuffle"]}
    return {
        "shuffle": False,
        "sampler": SourceOperationSampler(dataset.dataset["extra_env_info"], config, batch_size),
    }
