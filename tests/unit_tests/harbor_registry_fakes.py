# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fake registry client the Harbor tests use in place of ``RegistryClient``.

``conftest.py`` installs it (``fake_registry``, and ``no_real_registry`` for every ``test_harbor*`` module), and
test_harbor_tasks.py and test_harbor_task_data.py read its ``resolved`` and ``missing`` state."""

from nemo_gym.tasks.harbor.dockerfile import RECORD_VERSION
from nemo_gym.tasks.harbor.registry import ImageConfigError


IMAGE_PATH = "/usr/local/bin:/usr/bin:/bin"


class FakeRegistryClient:
    """Stands in for ``RegistryClient`` at prepare: records every image, except the ones told to fail."""

    missing: set[str] = set()
    resolved: list[str] = []

    def __init__(self, *args, **kwargs):
        pass

    def image_config(self, reference: str) -> dict:
        FakeRegistryClient.resolved.append(reference)
        if reference in FakeRegistryClient.missing:
            raise ImageConfigError(
                f"library/{reference.partition(':')[0]}: registry returned HTTP 404 for manifests/x"
            )
        return {
            "architecture": "amd64",
            "config": {"Env": [f"PATH={IMAGE_PATH}"], "WorkingDir": "/"},
            "config_digest": "sha256:c",
            "image": reference,
            "os": "linux",
            "record_version": RECORD_VERSION,
        }
