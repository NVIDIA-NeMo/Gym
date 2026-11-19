# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from resources_servers.terminal_bench_4.compose_config import _bytes, prebuilt_compose, resolve_compose


def image_record(**config):
    return {"image": "repo@sha256:digest", "os": "linux", "architecture": "amd64", "config": config}


def test_harbor_main_command_image_startup_and_compose_literals():
    source = {
        "services": {
            "main": {"depends_on": ["peer"]},
            "peer": {
                "image": "peer",
                "environment": ["URL=http://main:8000", "VALUE=a=b"],
                "shm_size": "1gb",
                "mem_limit": 4096,
                "healthcheck": {"test": ["CMD-SHELL", "echo $$HOME; test $? -eq 0"]},
            },
        }
    }
    original = deepcopy(source)
    records = {
        "main": image_record(Entrypoint=["/entrypoint.sh"], Cmd=["python3"], WorkingDir="/app"),
        "peer": image_record(Cmd=["serve"], User="appuser", ExposedPorts={"8000/tcp": {}}),
    }
    result = resolve_compose(source, "main", records)
    main, peer = result["services"]["main"], result["services"]["peer"]
    assert main["entrypoint"] == ["/entrypoint.sh"]
    assert main["command"] == ["sh", "-c", "sleep infinity"]
    assert main["working_dir"] == "/app"
    assert main["depends_on"] == {"peer": {"condition": "service_started"}}
    assert peer["command"] == ["serve"] and peer["user"] == "appuser"
    assert peer["expose"] == ["8000/tcp"]
    assert peer["environment"] == {"URL": "http://main:8000", "VALUE": "a=b"}
    assert peer["healthcheck"]["test"][1] == "echo $HOME; test $? -eq 0"
    assert (peer["shm_size"], peer["mem_limit"]) == (1024**3, 4096)
    assert source == original


def test_explicit_entrypoint_does_not_inherit_image_cmd_and_healthcheck_defaults():
    records = {
        "main": image_record(),
        "peer": image_record(
            Cmd=["wrong"],
            Healthcheck={
                "Test": ["CMD", "true"],
                "Interval": 1_000_000_000,
                "Timeout": 2_000_000_000,
                "StartPeriod": 3_000_000_000,
                "Retries": 4,
            },
        ),
    }
    result = resolve_compose({"services": {"peer": {"image": "peer", "entrypoint": ["right"]}}}, "main", records)
    peer = result["services"]["peer"]
    assert peer["command"] == [] and peer["entrypoint"] == ["right"]
    assert peer["healthcheck"] == {
        "test": ["CMD", "true"],
        "interval": "1.0s",
        "timeout": "2.0s",
        "start_period": "3.0s",
        "retries": 4,
    }


@pytest.mark.parametrize(
    "source,records,match",
    [
        (None, {}, "services mapping"),
        ({"services": {}}, {}, "no recorded OCI"),
        ({"services": {"main": {"environment": ["MISSING"]}}}, {"main": image_record()}, "unresolved"),
        ({"services": {"main": {"command": "echo $UNRESOLVED"}}}, {"main": image_record()}, "substitutions"),
        ({"services": {}}, {"main": {**image_record(), "architecture": "arm64"}}, "Linux/amd64"),
    ],
)
def test_invalid_or_unresolved_compose_fails_before_provisioning(source, records, match):
    with pytest.raises(ValueError, match=match):
        resolve_compose(source, "main", records)


def test_memory_units():
    assert _bytes("1.5GiB") == 1610612736
    assert _bytes("512") == 512
    with pytest.raises(ValueError, match="memory size"):
        _bytes("unlimited")


def test_prebuilt_compose_replaces_builds_and_applies_harbor_variables():
    source = {
        "services": {
            "main": {"build": {"context": "."}, "networks": ["default", "task"], "command": ["sleep", "infinity"]},
            "evaluator": {
                "build": {"context": "${CONTEXT_DIR}/evaluator"},
                "pull_policy": "build",
                "cpus": "${CPUS:-4}",
                "mem_limit": "${MEMORY-4096M}",
                "read_only": True,
                "cap_drop": ["ALL"],
                "cap_add": ["KILL"],
                "networks": ["task"],
                "healthcheck": {"test": ["CMD-SHELL", "test $$HOME = /root"]},
            },
            "cache": {"image": "redis:7-alpine", "environment": {"LIMIT": "${UNSET:-64}", "EMPTY": "${BLANK-}"}},
        },
        "networks": {"default": {}, "task": {"internal": True}},
    }
    original = deepcopy(source)
    variables = {"CPUS": "2", "MEMORY": "8192M", "BLANK": ""}
    ignored = ["read_only", "cap_drop", "networks"]
    result = prebuilt_compose(source, {"evaluator": "registry/evaluator@sha256:1"}, variables, ignored)
    assert source == original
    assert "networks" not in result
    main, evaluator, cache = (result["services"][name] for name in ("main", "evaluator", "cache"))
    # main runs the task image, which resolve_compose supplies.
    assert main == {"command": ["sleep", "infinity"]}
    assert evaluator == {
        "image": "registry/evaluator@sha256:1",
        "cpus": "2",
        "mem_limit": "8192M",
        "cap_add": ["KILL"],
        "healthcheck": {"test": ["CMD-SHELL", "test $$HOME = /root"]},
    }
    assert cache["environment"] == {"LIMIT": "64", "EMPTY": ""}
    # Without the task's resources, Compose falls back to the inline defaults.
    defaults = prebuilt_compose(source, {"evaluator": "image"}, {})["services"]["evaluator"]
    assert (defaults["cpus"], defaults["mem_limit"], defaults["networks"]) == ("4", "4096M", ["task"])


def test_prebuilt_compose_requires_images_and_leaves_unknown_variables_unresolved():
    with pytest.raises(ValueError, match="'peer' builds from a Dockerfile"):
        prebuilt_compose({"services": {"peer": {"build": "."}}}, {}, {})
    document = prebuilt_compose({"services": {"main": {"command": "echo ${UNKNOWN}"}}}, {}, {"CPUS": "1"})
    with pytest.raises(ValueError, match="substitutions"):
        resolve_compose(document, "main", {"main": image_record()})


def test_records_of_dockerfile_builds_are_keyed_by_task_and_service():
    records = {
        "task/main": {"os": "linux", "architecture": "amd64", "config": {"Cmd": ["python3"], "WorkingDir": "/app"}},
        "task/peer": {"os": "linux", "architecture": "amd64", "config": {"Cmd": ["serve"], "User": "app"}},
        "pinned": image_record(Cmd=["published"]),
    }
    source = {"services": {"peer": {"image": "registry/peer@sha256:2"}, "other": {"image": "pinned"}}}
    result = resolve_compose(source, "registry/main@sha256:1", records, task="task")
    main, peer, other = (result["services"][name] for name in ("main", "peer", "other"))
    assert (main["image"], main["working_dir"]) == ("registry/main@sha256:1", "/app")
    assert (peer["image"], peer["command"], peer["user"]) == ("registry/peer@sha256:2", ["serve"], "app")
    # A record for the exact image keeps its digest-pinned reference.
    assert (other["image"], other["command"]) == ("repo@sha256:digest", ["published"])
    with pytest.raises(ValueError, match="no recorded OCI"):
        resolve_compose(source, "registry/main@sha256:1", records)
