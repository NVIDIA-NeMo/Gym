# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from responses_api_agents.harbor_agent_general.compose_config import _bytes, resolve_compose


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
