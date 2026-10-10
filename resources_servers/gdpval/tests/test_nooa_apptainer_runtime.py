# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest

from benchmarks.gdpval import prepare_nooa_apptainer_runtime as runtime


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    config = tmp_path / "system.conf"
    config.write_text(
        "# original\nallow user ns = yes\nsessiondir max size = 64 # existing comment\nmount hostfs = no\n"
    )
    root = tmp_path / "owned-run"
    root.mkdir()
    monkeypatch.delenv("APPTAINER_CONFIG_FILE", raising=False)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/usr/bin/apptainer")
    state = {"suid": "0"}
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"APPTAINER_CONF_FILE={config}\nAPPTAINER_SUID_INSTALL={state['suid']}\n"
        ),
    )
    return config, root, state


def test_copy_changes_only_size_and_reuses_identical_private_file(configuration):
    config, root, _ = configuration
    before = config.read_bytes()
    result = runtime.prepare_apptainer_config(run_root=root)
    private = root / "private/apptainer.conf"
    assert config.read_bytes() == before
    assert private.read_bytes() == before.replace(b"size = 64", b"size = 8192")
    assert private.stat().st_mode & 0o777 == 0o600
    assert result["override_required"] and result["created_private_copy"]
    assert result["environment_variable"] == "APPTAINER_CONFIG_FILE"
    assert runtime.prepare_apptainer_config(run_root=root)["created_private_copy"] is False


def test_existing_sufficient_config_is_not_modified(configuration):
    config, root, _ = configuration
    config.write_text("sessiondir max size = 30720\n")
    result = runtime.prepare_apptainer_config(run_root=root)
    assert result["config_path"] == str(config)
    assert result["sessiondir_max_mib"] == 30720 and not result["override_required"]
    assert not (root / "private").exists()


@pytest.mark.parametrize("suid", ["1", "unknown"])
def test_refuses_setuid_or_unknown_install_override_for_nonroot(configuration, suid):
    config, root, state = configuration
    state["suid"] = suid
    if os.geteuid() == 0:
        pytest.skip("This case requires an actual nonroot process")
    with pytest.raises(PermissionError, match="non-setuid"):
        runtime.prepare_apptainer_config(run_root=root)
    assert not (root / "private").exists()
    assert "size = 64" in config.read_text()


@pytest.mark.parametrize("problem", ["symlink", "conflict", "duplicate", "minimum"])
def test_rejects_unsafe_or_ambiguous_configuration(configuration, problem):
    config, root, _ = configuration
    if problem == "symlink":
        (root / "private").symlink_to(root.parent, target_is_directory=True)
    elif problem == "conflict":
        (root / "private").mkdir()
        (root / "private/apptainer.conf").write_text("human-edited config")
    elif problem == "duplicate":
        config.write_text(config.read_text() + "sessiondir max size = 64\n")
    with pytest.raises((ValueError, FileExistsError)):
        runtime.prepare_apptainer_config(run_root=root, minimum_mib=64 if problem == "minimum" else 8192)


def test_honors_existing_config_override(configuration, monkeypatch):
    _, root, _ = configuration
    active = root / "selected.conf"
    active.write_text("sessiondir max size = 12288\n")
    monkeypatch.setenv("APPTAINER_CONFIG_FILE", str(active))
    result = runtime.prepare_apptainer_config(run_root=root)
    assert result["config_path"] == str(active) and result["override_required"]
    assert result["sessiondir_max_mib"] == 12288


def test_refuses_preexisting_unsupported_override(configuration, monkeypatch):
    config, root, state = configuration
    if os.geteuid() == 0:
        pytest.skip("This case requires an actual nonroot process")
    state["suid"] = "1"
    config.write_text("sessiondir max size = 12288\n")
    monkeypatch.setenv("APPTAINER_CONFIG_FILE", str(config))
    with pytest.raises(PermissionError, match="non-setuid"):
        runtime.prepare_apptainer_config(run_root=root)
