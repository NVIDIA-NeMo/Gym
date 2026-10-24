# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the Compose field allow-list and normalisation.

The unsupported field messages, deploy resources and gpus, restart, pull_policy, platform,
extra_hosts, hooks, networks and service options, and the fixed point of check_compose_fields.
"""

from tests.unit_tests.compose_testing import IMAGE_CONFIGS, MAIN, SIDECAR  # noqa: F401


class TestUnsupportedComposeFields:
    """The sandbox Compose adapter's field allow-list as a pure function, so prepare can report a refused field
    per task before any sandbox exists and the adapter raises the same message at start."""

    def test_names_the_service_and_its_fields_in_document_order(self):
        from nemo_gym.sandbox.compose_fields import unsupported_compose_fields

        document = {
            "x-extra": {},
            "services": {
                "main": {
                    "image": "a",
                    "deploy": {"resources": {"reservations": {"devices": [{"driver": "nvidia", "count": "all"}]}}},
                    "extra_hosts": ["pypi.org:0.0.0.0"],
                },
                "task-server": {
                    "image": "b",
                    "platform": "linux/amd64",
                    "expose": ["9111"],
                    "deploy": {"resources": {"limits": {"cpus": "2", "memory": "4096M"}}},
                },
                "api": {"image": "c", "build": {"context": "."}, "pull_policy": "always"},
            },
        }
        assert unsupported_compose_fields(document) == [
            "Unsupported Compose fields: ['x-extra']",
            "Service 'main': unsupported Compose fields ['deploy', 'extra_hosts']",
            "Service 'task-server': unsupported Compose fields ['deploy', 'platform']",
            "Service 'api': unsupported Compose fields ['build', 'pull_policy']",
        ]

    def test_every_supported_key_passes_and_shapes_are_left_to_the_adapter(self):
        from nemo_gym.sandbox.compose_fields import SUPPORTED_SERVICE_FIELDS, unsupported_compose_fields

        service = dict.fromkeys(SUPPORTED_SERVICE_FIELDS) | {"image": "a"}
        document = {"name": "p", "services": {"app": service}, "networks": {}, "volumes": {}}
        assert unsupported_compose_fields(document) == []
        # Not a mapping, a null service, a scalar service: the adapter's shape checks name those, not this one.
        assert unsupported_compose_fields([]) == []
        assert unsupported_compose_fields({"services": {"main": None, "x": "text"}}) == []
        assert unsupported_compose_fields({"services": []}) == []
