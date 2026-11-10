# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from resources_servers.visgym.scripts import create_hf_rl_manifests as manifests
from resources_servers.visgym.task_data import VISGYM_BINARY_SUCCESS_ENV_IDS
from resources_servers.visgym.tests.conftest import requires_visgym


def test_resolve_hf_revision_returns_immutable_commit(monkeypatch) -> None:
    commit = "a" * 40
    seen = {}

    def fake_request_json(url: str, timeout: int) -> dict[str, str]:
        seen.update(url=url, timeout=timeout)
        return {"sha": commit}

    monkeypatch.setattr(manifests, "request_json", fake_request_json)

    assert manifests.resolve_hf_revision("VisGym/visgym_data", "main", 17) == commit
    assert seen == {
        "url": "https://huggingface.co/api/datasets/VisGym/visgym_data/revision/main",
        "timeout": 17,
    }


def test_resolve_hf_revision_rejects_non_commit_response(monkeypatch) -> None:
    monkeypatch.setattr(manifests, "request_json", lambda _url, _timeout: {"sha": "main"})

    try:
        manifests.resolve_hf_revision("VisGym/visgym_data", "main", 17)
    except ValueError as exc:
        assert "full commit SHA" in str(exc)
    else:  # pragma: no cover - assertion helper
        raise AssertionError("non-commit Hugging Face revision was accepted")


@requires_visgym
def test_every_hf_environment_mapping_is_registered_by_pinned_runtime() -> None:
    import gymnasium as gym

    missing = []
    for source_name, env_id in manifests.LOCAL_ENV_BY_HF_ENV.items():
        try:
            gym.spec(env_id)
        except Exception as exc:  # pragma: no cover - assertion reports all missing IDs
            missing.append(f"{source_name} -> {env_id}: {type(exc).__name__}: {exc}")

    assert not missing, "\n".join(missing)
    assert manifests.LOCAL_ENV_BY_HF_ENV["refdot"] == "referring_dot_pointing/easy"
    assert set(manifests.LOCAL_ENV_BY_HF_ENV.values()).issubset(VISGYM_BINARY_SUCCESS_ENV_IDS)
    assert len(VISGYM_BINARY_SUCCESS_ENV_IDS) == 34
