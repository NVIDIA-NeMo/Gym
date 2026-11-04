# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the Compose field allow-list and normalisation.

The unsupported field messages, deploy resources and gpus, restart, pull_policy, platform,
extra_hosts, hooks, networks and service options, and the fixed point of check_compose_fields.
"""

import re
from copy import deepcopy

import pytest
import yaml

from nemo_gym.sandbox.compose_fields import (
    ComposeFieldError,
    check_compose_fields,
    normalize_platform,
    service_gpus,
    service_resources,
)
from tests.unit_tests.compose_testing import (
    IMAGE_CONFIGS,  # noqa: F401
    MAIN,
    SIDECAR,
    _fixture,
)


class TestUnsupportedComposeFields:
    """The sandbox Compose adapter's field allow-list as a pure function, the first step of ``check_compose_fields``,
    so prepare can report a refused key per task before any sandbox exists and the adapter raises the same message
    at start. The M28 fields (``deploy``, ``extra_hosts``, ``platform``, ``pull_policy``, ``post_start`` and the
    provider service options) are on the list; what they are given to run with is checked after this step."""

    def test_names_the_service_and_its_fields_in_document_order(self):
        from nemo_gym.sandbox.compose_fields import unsupported_compose_fields

        document = {
            "x-extra": {},
            "services": {
                "main": {
                    "image": "a",
                    "deploy": {"resources": {"reservations": {"devices": [{"driver": "nvidia", "count": "all"}]}}},
                    "logging": {"driver": "json-file"},
                },
                "task-server": {
                    "image": "b",
                    "platform": "linux/amd64",
                    "expose": ["9111"],
                    "deploy": {"resources": {"limits": {"cpus": "2", "memory": "4096M"}}},
                    "privileged": True,
                    "devices": ["/dev/fuse"],
                },
                "api": {"image": "c", "build": {"context": "."}, "pull_policy": "always"},
            },
        }
        assert unsupported_compose_fields(document) == [
            "Unsupported Compose fields: ['x-extra']",
            "Service 'main': unsupported Compose fields ['logging']",
            "Service 'task-server': unsupported Compose fields ['devices', 'privileged']",
            "Service 'api': unsupported Compose fields ['build']",
        ]

    def test_a_sidecar_build_and_a_key_no_provider_starts_are_refused_fields(self):
        from nemo_gym.sandbox.compose_fields import unsupported_compose_fields

        document = {
            "services": {
                "main": {"image": "a", "extra_hosts": ["pypi.org:0.0.0.0"], "cap_drop": ["ALL"]},
                "api": {"image": "b", "build": {"context": "."}, "pull_policy": "always", "cgroup_parent": "x"},
            }
        }
        assert unsupported_compose_fields(document) == [
            "Service 'main': unsupported Compose fields ['cap_drop']",
            "Service 'api': unsupported Compose fields ['build', 'cgroup_parent']",
        ]

    def test_check_compose_fields_refuses_the_same_keys_first_naming_the_field(self):
        from nemo_gym.sandbox.compose_fields import UnsupportedComposeFieldError, check_compose_fields

        document = {"services": {"api": {"image": "b", "build": {"context": "."}, "platform": "windows"}}}
        with pytest.raises(UnsupportedComposeFieldError) as info:
            check_compose_fields(document)
        assert str(info.value) == "Service 'api': unsupported Compose fields ['build']" and info.value.field == "build"
        with pytest.raises(UnsupportedComposeFieldError, match=r"Unsupported Compose fields: \[.x-extra.\]"):
            check_compose_fields({"x-extra": 1, "services": {"main": {"image": "a"}}})

    def test_every_supported_key_passes_and_shapes_are_left_to_the_adapter(self):
        from nemo_gym.sandbox.compose_fields import SUPPORTED_SERVICE_FIELDS, unsupported_compose_fields

        service = dict.fromkeys(SUPPORTED_SERVICE_FIELDS) | {"image": "a"}
        document = {"name": "p", "services": {"app": service}, "networks": {}, "volumes": {}}
        assert unsupported_compose_fields(document) == []
        # Not a mapping, a null service, a scalar service: the adapter's shape checks name those, not this one.
        assert unsupported_compose_fields([]) == []
        assert unsupported_compose_fields({"services": {"main": None, "x": "text"}}) == []
        assert unsupported_compose_fields({"services": []}) == []


def _gpu_document(devices, **model_fields):
    return {
        "services": {
            "main": {"image": MAIN, "deploy": {"resources": {"limits": {"cpus": "2", "memory": "1g"}}}},
            "model": {
                "image": SIDECAR,
                "deploy": {"resources": {"limits": {"memory": "2g"}, "reservations": {"devices": devices}}},
                **model_fields,
            },
        }
    }


def test_restart_always_and_a_custom_network_are_accepted_as_alibaba_ctf_writes_them():
    document = check_compose_fields(_fixture("ctf"))
    services = document["services"]
    # restart is validated and kept for the adapter to record; the list-form networks become mappings.
    assert services["main"]["restart"] == "always" and services["target"]["restart"] == "always"
    assert services["main"]["networks"] == {"agent_network": {}}
    assert services["target"]["networks"] == {"agent_network": {"aliases": ["challenge"]}}
    assert document["networks"] == {"agent_network": {"driver": "bridge"}}


def test_a_named_shared_network_is_accepted_as_o11y_bench_writes_it():
    document = check_compose_fields(_fixture("o11y"))
    assert document["networks"] == {"harbor-shared": {"name": "harbor-shared"}}
    assert all(service["networks"] == {"harbor-shared": {}} for service in document["services"].values())


def test_deploy_resources_map_to_sandbox_resources_with_limits_first():
    service = {
        "image": SIDECAR,
        "cpus": "0.5",
        "mem_limit": "128m",
        "deploy": {
            "resources": {
                "limits": {"cpus": "2", "memory": "1g"},
                "reservations": {
                    "cpus": "1",
                    "memory": "512M",
                    "devices": [{"driver": "nvidia", "capabilities": ["gpu"], "count": 2}],
                },
            },
            "replicas": 1,
            "restart_policy": {"condition": "on-failure"},
        },
    }
    document = check_compose_fields({"services": {"gpu": service}})
    deploy = document["services"]["gpu"]["deploy"]
    assert deploy["resources"] == {
        "limits": {"cpus": 2.0, "memory": 1024**3},
        "reservations": {"cpus": 1.0, "memory": 512 * 1024**2, "gpus": 2},
    }
    assert deploy["restart_policy"] == {"condition": "on-failure"}
    assert service_resources(document["services"]["gpu"]) == {"cpu": 2.0, "memory_mib": 1024, "gpu": 2}
    assert service_gpus(document["services"]["gpu"]) == 2
    # Reservations stand in when there is no limit; the legacy keys when there is no deploy block.
    assert service_resources({"deploy": {"resources": {"reservations": {"cpus": 1.5}}}}) == {"cpu": 1.5}
    assert service_resources({"cpus": "0.5", "mem_limit": 1048577}) == {"cpu": 0.5, "memory_mib": 2}
    assert service_gpus({"image": "x"}) == 0


def test_device_ids_count_gpus_and_count_all_is_refused():
    devices = [{"capabilities": ["gpu"], "device_ids": ["0", "1", "2"]}]
    document = check_compose_fields(
        {"services": {"s": {"deploy": {"resources": {"reservations": {"devices": devices}}}}}}
    )
    assert service_gpus(document["services"]["s"]) == 3
    with pytest.raises(ComposeFieldError, match="count: all is not supported") as info:
        check_compose_fields(
            {
                "services": {
                    "s": {
                        "deploy": {
                            "resources": {"reservations": {"devices": [{"capabilities": ["gpu"], "count": "all"}]}}
                        }
                    }
                }
            }
        )
    assert info.value.field == "deploy.resources.reservations.devices[0].count"
    with pytest.raises(ComposeFieldError, match="only the gpu capability"):
        check_compose_fields(
            {"services": {"s": {"deploy": {"resources": {"reservations": {"devices": [{"capabilities": ["tpu"]}]}}}}}}
        )


def test_a_decimal_count_string_from_interpolation_means_its_integer():
    """Review (M28): ``count: "${GPUS:-1}"`` reaches the check as ``"1"`` (interpolation hands back strings and the
    type restoration covers healthchecks alone) and was refused, so a valid GPU-sidecar task was skipped at prepare.
    A decimal string counts like its integer and normalizes to the same ``gpus``, a fixed point of the YAML round
    trip; anything else a string can say is still the field's error."""
    document = check_compose_fields(_gpu_document([{"capabilities": ["gpu"], "count": "2"}]))
    assert document["services"]["model"]["deploy"]["resources"]["reservations"] == {"gpus": 2}
    written = yaml.safe_load(yaml.safe_dump(document, sort_keys=False))
    assert check_compose_fields(deepcopy(written)) == written == document
    assert service_gpus(written["services"]["model"]) == 2
    for count in ("0", "-1", "1.5", "", "two", True):
        with pytest.raises(ComposeFieldError, match="count") as info:
            check_compose_fields(_gpu_document([{"capabilities": ["gpu"], "count": count}]))
        assert info.value.field == "deploy.resources.reservations.devices[0].count"


@pytest.mark.parametrize(
    "service, field, message",
    [
        ({"deploy": {"placement": {"constraints": []}}}, "deploy.placement", "deploy.placement is not supported"),
        ({"deploy": {"replicas": 3}}, "deploy.replicas", "replicas must be 1"),
        (
            {"deploy": {"resources": {"limits": {"pids": 100}}}},
            "deploy.resources.limits.pids",
            "pids is not supported",
        ),
        ({"deploy": {"resources": {"limits": {"memory": "lots"}}}}, "deploy.resources.limits.memory", "memory size"),
        # Review (M28): a float, a mapping, a list or a boolean escaped ``_bytes`` as an ``AttributeError``, past
        # prepare's per-task boundary, which aborted the dataset instead of skipping the task.
        ({"deploy": {"resources": {"limits": {"memory": 1.5}}}}, "deploy.resources.limits.memory", "got 1.5"),
        (
            {"deploy": {"resources": {"reservations": {"memory": {"size": "1g"}}}}},
            "deploy.resources.reservations.memory",
            "memory size such as 512M, got {'size': '1g'}",
        ),
        ({"deploy": {"resources": {"limits": {"memory": ["1g"]}}}}, "deploy.resources.limits.memory", r"got \['1g'\]"),
        ({"deploy": {"resources": {"limits": {"memory": True}}}}, "deploy.resources.limits.memory", "got True"),
        ({"restart": "sometimes"}, "restart", "restart must be one of"),
        ({"pull_policy": "build"}, "pull_policy", "build is not supported"),
        ({"pull_policy": "whenever"}, "pull_policy", "not a Compose value"),
        ({"platform": "linux/arm64"}, "platform", "does not match the deployment platform linux/amd64"),
        ({"platform": "windows"}, "platform", "platform must be os/arch"),
        ({"networks": ["backend"]}, "networks", "'backend' is not declared"),
        ({"networks": {"default": {"ipv4_address": "10.0.0.2"}}}, "networks.ipv4_address", "is not supported"),
        ({"extra_hosts": ["gateway:host-gateway"]}, "extra_hosts", "host-gateway"),
        ({"extra_hosts": ["db:not-an-ip"]}, "extra_hosts", "needs an IP address"),
        ({"extra_hosts": ["no-separator"]}, "extra_hosts", "is not host:address"),
        ({"post_start": [{"command": "x", "privileged": True}]}, "post_start.privileged", "is not supported"),
        ({"post_start": [{"user": "root"}]}, "post_start", "must be a mapping with a command"),
        ({"post_start": [{"command": "x", "user": "1000:1000"}]}, "post_start.user", "user:group overrides"),
        ({"post_start": [{"command": "x", "environment": "A=1"}]}, "post_start.environment", "list or a mapping"),
        ({"networks": {"default": 1}}, "networks", "networks.default must be a mapping, got 1"),
        ({"networks": {"default": "bridge"}}, "networks", "networks.default must be a mapping, got 'bridge'"),
        ({"pre_stop": [{"command": "x"}]}, "pre_stop", "pre_stop hooks are not supported"),
        ({"tmpfs": ["relative/path"]}, "tmpfs", "absolute container paths"),
        ({"ulimits": {"nofile": {"soft": 1}}}, "ulimits", "needs integer soft and hard"),
        ({"sysctls": ["net.core.somaxconn"]}, "sysctls", "key=value"),
        ({"security_opt": "seccomp=unconfined"}, "security_opt", "list of strings"),
    ],
)
def test_a_field_the_group_cannot_run_is_named(service, field, message):
    with pytest.raises(ComposeFieldError, match=message) as info:
        check_compose_fields({"services": {"svc": {"image": SIDECAR, **service}}})
    assert info.value.field == field


@pytest.mark.parametrize(
    "field, message",
    [
        ("internal", "needs network isolation"),
        ("ipam", "needs network isolation"),
        ("driver_opts", "needs network isolation"),
    ],
)
def test_a_network_option_the_pod_cannot_provide_is_named(field, message):
    value = {"config": [{"subnet": "10.0.0.0/24"}]} if field == "ipam" else True
    with pytest.raises(ComposeFieldError, match=message) as info:
        check_compose_fields({"services": {"a": {"image": SIDECAR}}, "networks": {"backend": {field: value}}})
    assert info.value.field == f"networks.backend.{field}"
    # An empty ipam block, as `docker compose config` emits it, means nothing and is dropped.
    document = check_compose_fields({"services": {"a": {}}, "networks": {"backend": {"ipam": {}}}})
    assert document["networks"] == {"backend": {}}


@pytest.mark.parametrize("options", ["bridge", 1, ["a"]], ids=["string", "int", "list"])
def test_a_network_declaration_that_is_not_a_mapping_is_the_fields_own_error(options):
    """``networks: {front: bridge}`` raised a bare ``ValueError`` from ``dict("bridge")`` and ``{front: 1}`` on a
    service a ``TypeError``, past the ``ComposeFieldError`` boundary prepare records a task by (review finding)."""
    with pytest.raises(
        ComposeFieldError, match=re.escape(f"network 'front' must be a mapping, got {options!r}")
    ) as info:
        check_compose_fields({"services": {"a": {"image": SIDECAR}}, "networks": {"front": options}})
    assert info.value.field == "networks.front"
    with pytest.raises(ComposeFieldError, match="service 'a' networks.front must be a mapping") as info:
        check_compose_fields(
            {"services": {"a": {"image": SIDECAR, "networks": {"front": options}}}, "networks": {"front": {}}}
        )
    assert info.value.field == "networks"
    # ``null`` is Compose's own spelling of a network with no options, on both levels.
    document = check_compose_fields({"services": {"a": {"networks": {"front": None}}}, "networks": {"front": None}})
    assert document["networks"] == {"front": {}} and document["services"]["a"]["networks"] == {"front": {}}


def test_pull_policy_platform_extra_hosts_and_hooks_take_their_defined_forms():
    document = check_compose_fields(
        {
            "services": {
                "svc": {
                    "image": SIDECAR,
                    "pull_policy": "if_not_present",
                    "platform": "linux/x86_64",
                    "extra_hosts": ["db:10.0.0.5", "api=[fd00::1]"],
                    "post_start": [
                        {"command": "echo ready", "user": "root"},
                        {"command": ["sh", "-c", "touch /tmp/x"], "working_dir": "/tmp", "environment": ["A=1"]},
                        {"command": "id", "user": 0},
                        {"command": "id", "user": "1000"},
                    ],
                },
                "other": {"image": SIDECAR, "pull_policy": "every_12h", "extra_hosts": {"db": "10.0.0.6"}},
                "third": {"image": SIDECAR, "pull_policy": "ALWAYS", "restart": "on-failure:3"},
            }
        }
    )
    svc, other, third = (document["services"][name] for name in ("svc", "other", "third"))
    assert svc["pull_policy"] == "missing" and other["pull_policy"] == "missing" and third["pull_policy"] == "always"
    assert svc["platform"] == "linux/amd64"
    assert svc["extra_hosts"] == {"db": "10.0.0.5", "api": "fd00::1"} and other["extra_hosts"] == {"db": "10.0.0.6"}
    # A decimal user is an integer id, whatever the file's spelling: the provider switches by uid, not by a name "0".
    assert svc["post_start"] == [
        {"command": "echo ready", "user": "root"},
        {"command": ["sh", "-c", "touch /tmp/x"], "working_dir": "/tmp", "environment": {"A": "1"}},
        {"command": "id", "user": 0},
        {"command": "id", "user": 1000},
    ]
    assert third["restart"] == "on-failure:3"


def test_service_options_are_normalized_and_checked_against_the_provider():
    service = {
        "image": SIDECAR,
        "tmpfs": "/run",
        "ulimits": {"nofile": {"soft": 1024, "hard": 2048}, "nproc": 64},
        "sysctls": ["net.core.somaxconn=1024"],
        "security_opt": ["seccomp=unconfined"],
    }
    document = check_compose_fields({"services": {"svc": dict(service)}})
    assert document["services"]["svc"]["tmpfs"] == ["/run"]
    assert document["services"]["svc"]["sysctls"] == {"net.core.somaxconn": "1024"}
    assert document["services"]["svc"]["ulimits"] == {"nofile": {"soft": 1024, "hard": 2048}, "nproc": 64}
    # Named provider without the capability: the error names the field and the provider.
    with pytest.raises(
        ComposeFieldError, match="service 'svc' security_opt is not supported by the opensandbox sandbox provider"
    ) as info:
        check_compose_fields(
            {"services": {"svc": dict(service)}},
            provider="opensandbox",
            service_option_fields=frozenset({"tmpfs", "ulimits", "sysctls"}),
        )
    assert info.value.field == "security_opt"
    full = frozenset({"tmpfs", "ulimits", "sysctls", "security_opt"})
    assert check_compose_fields({"services": {"svc": dict(service)}}, provider="docker", service_option_fields=full)


def test_normalize_platform_folds_architecture_aliases():
    assert normalize_platform("linux/x86_64") == "linux/amd64"
    assert normalize_platform("Linux/aarch64/v8") == "linux/arm64"
    with pytest.raises(ValueError):
        normalize_platform("amd64")


@pytest.mark.parametrize(
    "document, gpus",
    [
        (_fixture("ctf"), None),
        (_fixture("o11y"), None),
        (_gpu_document([{"driver": "nvidia", "capabilities": ["gpu"], "count": 2}]), 2),
        (_gpu_document([{"capabilities": ["gpu"], "device_ids": ["0", "1", "3"]}]), 3),
        (
            {
                "services": {
                    "model": {
                        "image": SIDECAR,
                        "post_start": [{"command": "id", "user": "0"}, {"command": "id", "user": 7}],
                    }
                }
            },
            None,
        ),
    ],
    ids=["ctf", "o11y", "gpu-count", "gpu-device-ids", "hook-users"],
)
def test_check_compose_fields_is_a_fixed_point_of_its_own_output(document, gpus):
    """The server normalizes once and writes the file; the adapter reloads it and normalizes again. The second
    pass must accept the first one's output unchanged, GPU reservations included (``devices`` became ``gpus``)."""
    once = check_compose_fields(document)
    written = yaml.safe_load(yaml.safe_dump(once, sort_keys=False))
    assert check_compose_fields(deepcopy(written)) == written == once
    if gpus is not None:
        assert once["services"]["model"]["deploy"]["resources"]["reservations"] == {"gpus": gpus}
        assert service_gpus(written["services"]["model"]) == gpus


@pytest.mark.parametrize(
    "reservations, message",
    [
        ({"gpus": 0}, "deploy.resources.reservations.gpus must be a positive integer, got 0"),
        ({"gpus": True}, "deploy.resources.reservations.gpus must be a positive integer, got True"),
        ({"gpus": "1"}, "deploy.resources.reservations.gpus must be a positive integer"),
        (
            {"gpus": 1, "devices": [{"capabilities": ["gpu"]}]},
            "deploy.resources.reservations: give devices or gpus, not both",
        ),
    ],
)
def test_the_normalized_gpus_form_is_checked_like_devices(reservations, message):
    document = {"services": {"model": {"image": SIDECAR, "deploy": {"resources": {"reservations": reservations}}}}}
    with pytest.raises(ComposeFieldError, match=f"service 'model' {message}") as info:
        check_compose_fields(document)
    assert info.value.field == "deploy.resources.reservations.gpus"


def test_a_gpu_sidecar_is_checked_against_the_gpu_provider():
    docker, opensandbox = frozenset({"tmpfs", "ulimits", "sysctls", "security_opt"}), frozenset()
    # Docker task block, OpenSandbox GPU block: the sidecar reserving GPUs runs on the GPU block, which lacks tmpfs.
    with pytest.raises(
        ComposeFieldError, match="service 'model' tmpfs is not supported by the opensandbox sandbox provider"
    ):
        check_compose_fields(
            _gpu_document([{"capabilities": ["gpu"], "count": 1}], tmpfs=["/run"]),
            provider="docker",
            service_option_fields=docker,
            gpu_provider="opensandbox",
            gpu_service_option_fields=opensandbox,
        )
    # The reverse pairing runs it; without a GPU block every service is checked against the task's provider.
    document = check_compose_fields(
        _gpu_document([{"capabilities": ["gpu"], "count": 1}], tmpfs=["/run"]),
        provider="opensandbox",
        service_option_fields=opensandbox,
        gpu_provider="docker",
        gpu_service_option_fields=docker,
    )
    assert document["services"]["model"]["tmpfs"] == ["/run"]
    with pytest.raises(ComposeFieldError, match="tmpfs is not supported by the opensandbox sandbox provider"):
        check_compose_fields(
            _gpu_document([{"capabilities": ["gpu"], "count": 1}], tmpfs=["/run"]),
            provider="opensandbox",
            service_option_fields=opensandbox,
        )
    # main follows the task, not its own deploy block: a GPU reservation on main stays with the task's provider.
    document = {
        "services": {
            "main": {
                "image": MAIN,
                "tmpfs": ["/run"],
                **_gpu_document([{"capabilities": ["gpu"]}])["services"]["model"],
            }
        }
    }
    with pytest.raises(
        ComposeFieldError, match="service 'main' tmpfs is not supported by the docker sandbox provider"
    ):
        check_compose_fields(
            document,
            provider="docker",
            service_option_fields=frozenset(),
            gpu_provider="opensandbox",
            gpu_service_option_fields=docker,
        )


def test_a_network_declared_internal_false_asks_for_nothing_and_is_kept():
    """Every kumo task declares ``networks: {internal: {internal: false}}`` (Compose's default, spelled out). A
    false value asks for no isolation, as an empty ``ipam`` block asks for no addressing, so the network is kept
    with the option dropped instead of failing prepare with ``needs network isolation`` (M35 D1, 10,350 tasks)."""
    document = check_compose_fields(
        {"services": {"a": {"image": SIDECAR}}, "networks": {"internal": {"internal": False, "enable_ipv6": False}}}
    )
    assert document["networks"] == {"internal": {}}
    for option in ("internal", "enable_ipv6"):
        with pytest.raises(ComposeFieldError, match="needs network isolation") as info:
            check_compose_fields({"services": {"a": {"image": SIDECAR}}, "networks": {"internal": {option: True}}})
        assert info.value.field == f"networks.internal.{option}"


class TestPorts:
    """M35 D4: ``ports: ["8080:8080"]`` passed prepare as a string and the adapter refused it at seed (``resolve ports
    to a list of mappings upstream``), so the task failed every episode. Prepare now gives ``ports`` the long syntax
    and applies the group's endpoint rules, the same for the short and the long form."""

    @pytest.mark.parametrize(
        "entry, expected",
        [
            ("8080", {"target": 8080}),
            (8080, {"target": 8080}),
            # The bounds the sandbox spec applies, inclusive.
            ("1", {"target": 1}),
            (65535, {"target": 65535}),
            ("0:8080", {"target": 8080, "published": "0"}),
            ("9000/tcp", {"target": 9000, "protocol": "tcp"}),
            # The protocol is canonical lowercase: the adapter compares it exactly, so an upper-case spelling
            # passed prepare and was refused at every seed (repair round 1).
            ("8080/TCP", {"target": 8080, "protocol": "tcp"}),
            ({"target": 80, "protocol": "Tcp"}, {"target": 80, "protocol": "tcp"}),
            # Control: a long entry the adapter always accepted is left as it is.
            ({"target": 8080}, {"target": 8080}),
            (
                {"target": "8080", "published": 0, "protocol": "tcp", "mode": "ingress"},
                {"target": 8080, "published": 0, "protocol": "tcp", "mode": "ingress"},
            ),
        ],
    )
    def test_short_and_long_entries_take_one_shape(self, entry, expected):
        document = check_compose_fields({"services": {"svc": {"image": SIDECAR, "ports": [entry]}}})
        assert document["services"]["svc"]["ports"] == [expected]
        # The adapter's own rule, applied to what prepare wrote.
        assert expected.get("protocol", "tcp") == "tcp" and expected.get("mode", "ingress") == "ingress"
        # The fixed point: the adapter re-reads the written file through the same check.
        assert check_compose_fields(deepcopy(document)) == document

    @pytest.mark.parametrize(
        "entry, message",
        [
            ("8080:8080", "fixed published ports cannot be honoured by sandbox endpoints (published '8080')"),
            ("127.0.0.1:8080:80", "fixed published ports cannot be honoured by sandbox endpoints (published '8080')"),
            (
                {"target": 80, "published": 8080},
                "fixed published ports cannot be honoured by sandbox endpoints (published 8080)",
            ),
            ({"target": 80, "published": 0, "host_ip": "127.0.0.1"}, "host_ip cannot be honoured"),
            ("53/udp", "only TCP ports"),
            ({"target": 80, "protocol": "udp"}, "only TCP ports"),
            ({"target": 80, "mode": "host"}, "only ingress mode"),
            ("8080-8090", "not a range"),
            # Review (M28): these passed prepare and failed every seed in ``SandboxSpec`` (ports 1 through 65535).
            ("0", "target must be between 1 and 65535, got 0"),
            ({"target": 65536}, "target must be between 1 and 65535, got 65536"),
            ("0:0", "target must be between 1 and 65535, got 0"),
            ({"target": 80, "name": "web"}, "unsupported option 'name'"),
            ("a:b:c:d", "cannot read"),
            ("http", "must be a port number"),
            ([80], "must be a string, a number or a mapping"),
        ],
    )
    def test_what_sandbox_endpoints_cannot_honour_fails_prepare_naming_the_service(self, entry, message):
        with pytest.raises(ComposeFieldError, match=re.escape(message)) as info:
            check_compose_fields({"services": {"next-waf": {"image": SIDECAR, "ports": [entry]}}})
        assert info.value.field == "ports"
        assert "service 'next-waf' ports" in str(info.value)

    def test_ports_must_be_a_list(self):
        with pytest.raises(ComposeFieldError, match="service 'svc' ports must be a list"):
            check_compose_fields({"services": {"svc": {"image": SIDECAR, "ports": "8080"}}})
        assert (
            check_compose_fields({"services": {"svc": {"image": SIDECAR, "ports": None}}})["services"]["svc"]["ports"]
            == []
        )
