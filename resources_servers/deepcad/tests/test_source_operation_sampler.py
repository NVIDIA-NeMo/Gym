# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/NVIDIA-NeMo/RL, Apache-2.0,
# revision f2c87657d570be0e5293b57bd9b5e45e60087848,
# tests/unit/data/test_source_operation_sampler.py.

import copy
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from resources_servers.deepcad.dataset.source_operation_sampler import (
    SourceOperationSampler,
    SourceOperationSamplingConfig,
    source_operation_dataloader_kwargs,
    validate_source_operation_sampling,
)


def rows():
    return [
        json.dumps({"deepcad_lazy": {"record": {"dataset": s, "reference_operations": ops}}})
        for s, ops in [
            ("a", ["common"]),
            ("a", ["common", "rare", "rare"]),
            ("a", []),
            ("b", ["common"]),
            ("b", ["other"]),
            ("histcad", []),
        ]
    ]


def cfg(**kwargs):
    return SourceOperationSamplingConfig(sources=["a", "b", "histcad"], samples_per_epoch=600, **kwargs)


def test_quota_determinism_epochs_and_resume():
    config = cfg()
    sampler = SourceOperationSampler(rows(), config, 12)
    expected = list(sampler)
    assert expected == list(SourceOperationSampler(rows(), config, 12))
    for start in range(0, len(expected), 12):
        assert Counter(["a" if i < 3 else "b" if i < 5 else "histcad" for i in expected[start : start + 12]]) == {
            "a": 4,
            "b": 4,
            "histcad": 4,
        }
    assert list(sampler) != expected
    sampler = SourceOperationSampler(rows(), config, 12)
    iterator = iter(sampler)
    for _ in range(17):
        next(iterator)
    restored = SourceOperationSampler(rows(), config, 12)
    restored.load_state_dict(sampler.state_dict())
    assert list(restored) == expected[17:]
    baseline = SourceOperationSampler(rows(), config, 12)
    list(baseline)
    assert list(restored) == list(baseline)


@pytest.mark.parametrize("workers", [0, 2])
def test_stateful_loader_resume_with_prefetch(workers):
    StatefulDataLoader = pytest.importorskip("torchdata.stateful_dataloader").StatefulDataLoader
    config = cfg()

    def loader():
        return StatefulDataLoader(
            list(range(6)),
            sampler=SourceOperationSampler(rows(), config, 12),
            batch_size=12,
            num_workers=workers,
        )

    original = loader()
    iterator = iter(original)
    for _ in range(7):
        next(iterator)
    state = copy.deepcopy(original.state_dict())
    expected = [batch.tolist() for batch in iterator]
    restored = loader()
    restored.load_state_dict(state)
    assert [batch.tolist() for batch in restored] == expected
    assert [batch.tolist() for batch in original] == [batch.tolist() for batch in restored]


@pytest.mark.parametrize("tau", [1, 10, 100, 1000, 10000])
def test_matches_exact_mixture_and_deduplicates_operations(tau):
    config = cfg(tau=tau).model_copy(update={"samples_per_epoch": 120000})
    sampler = SourceOperationSampler(rows(), config, 12)
    actual = np.bincount(list(sampler), minlength=6) / len(sampler)
    q = np.log1p(np.array([2, 1]) / tau)
    q /= q.sum()
    expected = np.array(
        [
            (0.5 / 3 + 0.5 * q[0] / 2) / 3,
            (0.5 / 3 + 0.5 * (q[0] / 2 + q[1])) / 3,
            0.5 / 9,
            1 / 6,
            1 / 6,
            1 / 3,
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=0.004)
    np.testing.assert_allclose(sampler.operation_probabilities[0], q)


def test_uniform_and_operation_only():
    for fraction in [0, 1]:
        sampler = SourceOperationSampler(rows(), cfg(operation_fraction=fraction), 12)
        selected = list(sampler)
        assert 5 in selected  # HistCAD fallback even when operation_fraction=1.
        assert (2 in selected) == (fraction == 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau": 0},
        {"tau": float("inf")},
        {"tau": float("nan")},
        {"operation_fraction": 1.1},
        {"sources": ["a", "a"]},
        {"source_field": ""},
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        SourceOperationSamplingConfig(**kwargs)


def test_bad_metadata_and_state():
    with pytest.raises(ValueError):
        SourceOperationSampler(rows(), cfg(), 4)
    with pytest.raises(ValueError):
        SourceOperationSampler(rows()[:-1], cfg(), 12)
    bad = rows()
    bad[0] = "{}"
    with pytest.raises(ValueError):
        SourceOperationSampler(bad, cfg(), 12)
    bad = rows()
    bad[0] = bad[0].replace('["common"]', '"common"')
    with pytest.raises(ValueError):
        SourceOperationSampler(bad, cfg(), 12)
    first = SourceOperationSampler(rows(), cfg(), 12)
    for changed_rows, config in [
        (rows()[::-1], cfg()),
        (rows(), cfg(tau=10)),
        (rows(), cfg(seed=43)),
    ]:
        other = SourceOperationSampler(changed_rows, config, 12)
        with pytest.raises(ValueError):
            other.load_state_dict(first.state_dict())


def test_factory_never_processes_or_renders_rows():
    class LazyDataset:
        dataset = {"extra_env_info": rows()}

        def __getitem__(self, _):
            raise AssertionError("Must not process prompts")

    data = {
        "train": [{"dataset_name": "NemoGymDataset"}],
        "shuffle": False,
        "gym_dataset_hooks": {"config": {"source_operation_sampling": cfg().model_dump()}},
    }
    kwargs = source_operation_dataloader_kwargs(LazyDataset(), data, 12)
    assert len(list(kwargs["sampler"])) == 600
    assert source_operation_dataloader_kwargs(None, {"shuffle": True}, 12) == {"shuffle": True}
    for key in ["shuffle", "use_multiple_dataloader"]:
        with pytest.raises(ValueError):
            validate_source_operation_sampling(dict(data, **{key: True}))
    data["train"][0]["source_balance"] = {"seed": 42}
    with pytest.raises(ValueError):
        validate_source_operation_sampling(data)


def test_real_arrow_dataset_and_loader():
    StatefulDataLoader = pytest.importorskip("torchdata.stateful_dataloader").StatefulDataLoader
    from datasets import Dataset

    dataset = SimpleNamespace(dataset=Dataset.from_dict({"extra_env_info": rows()}))
    data = {
        "train": {"dataset_name": "NemoGymDataset"},
        "shuffle": False,
        "gym_dataset_hooks": {"config": {"source_operation_sampling": cfg().model_dump()}},
    }
    loader = StatefulDataLoader(
        list(range(6)),
        batch_size=12,
        **source_operation_dataloader_kwargs(dataset, data, 12),
    )
    assert len(list(loader)) == 50
