# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle


pytestmark = pytest.mark.sandbox


def make_compose(provider, document, *, image_configs=None, **kwargs):
    """Stub external normalization/inspection for focused lifecycle tests."""
    from unittest.mock import AsyncMock

    from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose

    group = AsyncSandboxCompose(provider, "compose.yaml", **kwargs)
    group.document = document
    group._normalize = AsyncMock(return_value=document)
    group._image_configs = dict(image_configs or {})
    return group


class Provider:
    name = "test"

    def __init__(self):
        self.created = []
        self.closed = []

    async def create(self, spec):
        self.created.append(spec)
        return SandboxHandle(str(len(self.created)), self.name, None)

    async def close(self, handle):
        self.closed.append(handle.sandbox_id)

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_rejects_unsupported_network_before_creating_anything():
    provider = Provider()
    group = make_compose(provider, {"services": {"db": {"image": "db"}, "app": {"image": "app"}}})
    with pytest.raises(NotImplementedError, match="network"):
        await group.start()
    assert provider.created == []


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("privileged", True), ("restart", "always"), ("devices", ["/dev/fuse"])])
async def test_rejects_unsupported_semantics_before_creating(field, value):
    provider = Provider()
    group = make_compose(provider, {"services": {"app": {"image": "app", field: value}}})
    with pytest.raises((ValueError, NotImplementedError), match=field):
        await group.start()
    assert provider.created == []


@pytest.mark.asyncio
async def test_detects_dependency_cycle_before_creating():
    provider = Provider()
    group = make_compose(
        provider,
        {
            "services": {
                "a": {"image": "a", "depends_on": {"b": {"condition": "service_started"}}},
                "b": {"image": "b", "depends_on": {"a": {"condition": "service_started"}}},
            }
        },
    )
    with pytest.raises(ValueError, match="cycle"):
        await group.start()
    assert not provider.created


@pytest.mark.asyncio
async def test_published_ports_require_endpoint_capability():
    provider = Provider()
    group = make_compose(provider, {"services": {"app": {"image": "app", "ports": [{"target": 8000}]}}})
    with pytest.raises(NotImplementedError, match="endpoint"):
        await group.start()
    assert not provider.created


@pytest.mark.asyncio
async def test_shared_volumes_require_provider_support():
    provider = Provider()
    group = make_compose(
        provider,
        {
            "services": {
                "app": {
                    "image": "app",
                    "volumes": [{"type": "volume", "source": "data", "target": "/data"}],
                }
            },
            "volumes": {"data": {}},
        },
    )
    with pytest.raises(NotImplementedError, match="volume|storage"):
        await group.start()
    assert not provider.created


class ShellProvider(Provider):
    """Execute actual subprocesses; only remote provisioning is replaced."""

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        import asyncio
        import os

        spec = self.created[int(handle.sandbox_id) - 1]
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            cwd=cwd,
            env={**os.environ, **spec.env, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(process.communicate(), timeout_s)
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        except TimeoutError:
            if process.returncode is None:
                process.kill()
            await process.wait()
            return SandboxExecResult("", "timeout", 124)
        return SandboxExecResult(out.decode(), err.decode(), process.returncode)

    async def upload_file(self, handle, source_path, target_path):
        import shutil

        shutil.copyfile(source_path, target_path)

    def validate_networking(self):
        pass

    async def network_address(self, handle):
        return "192.0.2." + handle.sandbox_id

    async def set_hosts(self, handle, hosts):
        pass


@pytest.mark.asyncio
async def test_health_and_completed_dependency_gate_real_processes(tmp_path):
    import shlex

    ready = shlex.quote(str(tmp_path / "ready"))
    seeded = shlex.quote(str(tmp_path / "seeded"))
    result = tmp_path / "result"
    document = {
        "services": {
            "app": {
                "image": "image",
                "command": ["sh", "-c", f'test -f {seeded} && printf "$VALUE" > {result}'],
                "environment": {"VALUE": "compose"},
                "depends_on": {"seed": {"condition": "service_completed_successfully"}},
            },
            "seed": {
                "image": "image",
                "command": ["sh", "-c", f"test -f {ready} && touch {seeded}"],
                "depends_on": {"db": {"condition": "service_healthy"}},
            },
            "db": {
                "image": "image",
                "command": ["sh", "-c", f"sleep .1; touch {ready}; sleep 1"],
                "healthcheck": {
                    "test": ["CMD", "test", "-f", str(tmp_path / "ready")],
                    "interval": "20ms",
                    "retries": 30,
                },
            },
        }
    }
    provider = ShellProvider()
    async with make_compose(
        provider, document, image_configs={"image": {"Cmd": ["false"]}}, poll_interval_s=0.01
    ) as group:
        await group._wait("app", "service_completed_successfully")
        assert result.read_text() == "compose"
    assert provider.closed == ["3", "2", "1"]


@pytest.mark.asyncio
async def test_failed_one_shot_prevents_dependent_start_and_cleans_up(tmp_path):
    result = tmp_path / "must-not-exist"
    provider = ShellProvider()
    group = make_compose(
        provider,
        {
            "services": {
                "seed": {"image": "image", "command": ["sh", "-c", "exit 7"]},
                "app": {
                    "image": "image",
                    "command": ["touch", str(result)],
                    "depends_on": {"seed": {"condition": "service_completed_successfully"}},
                },
            }
        },
        image_configs={"image": {}},
        poll_interval_s=0.01,
    )
    with pytest.raises(RuntimeError, match="exited.*7"):
        await group.start()
    assert not result.exists()
    assert provider.closed == ["2", "1"]


@pytest.mark.asyncio
async def test_explicit_entrypoint_clears_image_command_and_argv_preserves_empty(tmp_path):
    output = tmp_path / "output"
    provider = ShellProvider()
    group = make_compose(
        provider,
        {
            "services": {
                "app": {
                    "image": "image",
                    "entrypoint": ["sh", "-c", f'printf "%s|%s" "$1" "$2" > {output}', "sh", "", "two words"],
                }
            }
        },
        image_configs={"image": {"Cmd": ["incorrect"]}},
        poll_interval_s=0.01,
    )
    async with group:
        await group._wait("app", "service_completed_successfully")
    assert output.read_text() == "|two words"


@pytest.mark.asyncio
async def test_invalid_health_dependency_rejected_before_provisioning():
    provider = ShellProvider()
    group = make_compose(
        provider,
        {
            "services": {
                "db": {"image": "image"},
                "app": {"image": "image", "depends_on": {"db": {"condition": "service_healthy"}}},
            }
        },
        image_configs={"image": {"Cmd": ["sleep", "1"]}},
    )
    with pytest.raises(ValueError, match="no healthcheck"):
        await group.start()
    assert not provider.created


@pytest.mark.asyncio
async def test_collection_closes_provider_only_after_all_sandboxes(tmp_path):
    class SharedProvider(ShellProvider):
        shutdown = False

        async def close(self, handle):
            if self.shutdown:
                raise RuntimeError("provider transport already closed")
            await super().close(handle)

        async def aclose(self):
            self.shutdown = True

    provider = SharedProvider()
    async with make_compose(
        provider,
        {
            "services": {
                "one": {"image": "image"},
                "two": {"image": "image"},
            }
        },
        image_configs={"image": {"Cmd": ["sleep", ".1"]}},
    ):
        assert not provider.shutdown
    assert provider.closed == ["2", "1"]
    assert provider.shutdown


@pytest.mark.asyncio
async def test_cancelled_cleanup_finishes_every_member():
    import asyncio

    began = asyncio.Event()
    release = asyncio.Event()

    class SlowProvider(ShellProvider):
        async def close(self, handle):
            began.set()
            await release.wait()
            await super().close(handle)

    provider = SlowProvider()
    group = make_compose(
        provider,
        {
            "services": {
                "one": {"image": "image"},
                "two": {"image": "image"},
            }
        },
        image_configs={"image": {"Cmd": ["sleep", ".1"]}},
        poll_interval_s=0.01,
    )
    await group.start()
    cleanup = asyncio.create_task(group.stop())
    await began.wait()
    cleanup.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await cleanup
    await group.stop()
    assert provider.closed == ["2", "1"]


@pytest.mark.asyncio
async def test_failed_cleanup_can_be_retried():
    class RetryProvider(ShellProvider):
        fail = True

        async def close(self, handle):
            if self.fail:
                self.fail = False
                raise RuntimeError("temporary delete failure")
            await super().close(handle)

    provider = RetryProvider()
    group = make_compose(
        provider,
        {"services": {"one": {"image": "image"}}},
        image_configs={"image": {"Cmd": ["sleep", ".1"]}},
        poll_interval_s=0.01,
    )
    await group.start()
    with pytest.raises(ExceptionGroup):
        await group.stop()
    await group.stop()
    assert provider.closed == ["1"]


@pytest.mark.asyncio
async def test_single_service_receives_its_name_and_alias():
    class NetworkProvider(ShellProvider):
        async def set_hosts(self, handle, hosts):
            self.hosts = hosts

    provider = NetworkProvider()
    async with make_compose(
        provider,
        {
            "services": {
                "one": {
                    "image": "image",
                    "networks": {"default": {"aliases": ["self-name"]}},
                }
            }
        },
        image_configs={"image": {"Cmd": ["sleep", ".1"]}},
        poll_interval_s=0.01,
    ):
        assert provider.hosts == {"one": "192.0.2.1", "self-name": "192.0.2.1"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "volume",
    [
        {"type": "volume", "source": "../escape", "target": "/data"},
        {"type": "bind", "source": "/local/path", "target": "/data"},
    ],
)
async def test_volume_preflight_prevents_unsafe_or_unmapped_mounts(volume):
    class VolumeProvider(ShellProvider):
        def shared_volume_options(self, source, target, *, read_only=False):
            return {}

        def shared_volume_metadata(self):
            return {}

    provider = VolumeProvider()
    group = make_compose(provider, {"services": {"one": {"image": "image", "volumes": [volume]}}})
    with pytest.raises((ValueError, NotImplementedError)):
        await group.start()
    assert not provider.created


@pytest.mark.asyncio
async def test_volume_helper_create_failure_preserves_error_and_closes_provider():
    from unittest.mock import AsyncMock

    provider = ShellProvider()
    provider.create = AsyncMock(side_effect=RuntimeError("volume helper unavailable"))
    provider.aclose = AsyncMock()
    provider.shared_volume_metadata = lambda: {}
    provider.shared_volume_options = lambda *args, **kwargs: {"volumes": []}
    group = make_compose(
        provider,
        {
            "services": {
                "app": {"image": "app", "volumes": [{"type": "volume", "source": "data", "target": "/data"}]}
            },
            "volumes": {"data": {}},
        },
        image_configs={"app": {"Cmd": ["true"]}},
    )
    with pytest.raises(RuntimeError, match="volume helper unavailable"):
        await group.start()
    provider.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_yaml_input_normalizes_environment_and_runs_services(tmp_path):
    import shutil
    import sys

    import yaml

    from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose

    compose = shutil.which("docker-compose")
    if compose is None:
        pytest.skip("docker-compose executable is not installed")
    (tmp_path / ".env").write_text("COMPOSE_TEST_VALUE=from-dotenv\n")
    output = tmp_path / "result"
    path = tmp_path / "compose.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "app": {
                        "image": "test-image",
                        "environment": ["VALUE=${COMPOSE_TEST_VALUE}"],
                        "x-sandbox": {"hosts": []},
                        "command": ["sh", "-c", f'printf "%s" "$$VALUE" > {output}'],
                    }
                }
            }
        )
    )
    provider = ShellProvider()
    provider.set_hosts = AsyncMock(side_effect=AssertionError("explicit host opt-out was lost"))
    async with AsyncSandboxCompose(
        provider,
        path,
        compose_command=(compose,),
        image_config_command=(sys.executable, "-c", "print('{\"config\": {}}')"),
        poll_interval_s=0.01,
    ) as group:
        await group._wait("app", "service_completed_successfully")
        assert output.read_text() == "from-dotenv"
        assert provider.created[0].env["VALUE"] == "from-dotenv"
    assert provider.closed == ["1"]


@pytest.mark.asyncio
async def test_invalid_yaml_fails_before_provisioning(tmp_path):
    import shutil

    from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose

    compose = shutil.which("docker-compose")
    if compose is None:
        pytest.skip("docker-compose executable is not installed")
    path = tmp_path / "compose.yaml"
    path.write_text("services: [\n")
    provider = ShellProvider()
    with pytest.raises(RuntimeError, match="failed"):
        await AsyncSandboxCompose(provider, path, compose_command=(compose,)).start()
    assert provider.created == []


@pytest.mark.asyncio
async def test_healthcheck_override_inherits_image_test_and_defaults():
    compose = make_compose(
        SimpleNamespace(),
        {"services": {"app": {"image": "image", "healthcheck": {"interval": "1s"}}}},
        image_configs={
            "image": {
                "Cmd": ["sleep", "60"],
                "Healthcheck": {
                    "Test": ["CMD", "true"],
                    "Timeout": 2_000_000_000,
                    "Retries": 5,
                },
            }
        },
    )
    await compose._prepare()
    assert compose._plans["app"]["health"] == {
        "test": ["CMD", "true"],
        "interval": "1s",
        "timeout": 2.0,
        "retries": 5,
    }


@pytest.mark.asyncio
async def test_compose_cli_normalization_decodes_literal_dollars_and_empty_ipam(tmp_path, monkeypatch):
    path = tmp_path / "compose.yaml"
    normalized = {
        "networks": {"default": {"name": "project_default", "ipam": {}}},
        "services": {
            "app": {
                "image": "alpine",
                "command": ["sh", "-c", 'echo $$HOME "$${FOO}"'],
                "environment": {"FOO": "$$literal"},
                "healthcheck": {"test": ["CMD-SHELL", 'test "$$FOO" = "$$literal"']},
                "shm_size": "1073741824",
            }
        },
    }
    monkeypatch.setattr(
        "nemo_gym.sandbox.adapters.docker_compose._command", AsyncMock(return_value=json.dumps(normalized))
    )
    compose = AsyncSandboxCompose(
        SimpleNamespace(validate_networking=lambda: None, network_address=AsyncMock(), set_hosts=AsyncMock()),
        path,
    )
    compose.document = await compose._normalize()
    assert compose.document["services"]["app"].pop("shm_size") == 1073741824
    compose._image_configs = {"alpine": {}}
    await compose._prepare()
    assert shlex.split(compose._plans["app"]["command"])[2] == 'echo $HOME "${FOO}"'
    assert compose._plans["app"]["spec"].env["FOO"] == "$literal"
    assert compose._plans["app"]["health"]["test"][1] == 'test "$FOO" = "$literal"'
    assert compose._validate() == ["app"]


@pytest.mark.asyncio
async def test_cancel_during_initial_file_upload_deletes_created_service():
    import asyncio

    from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxSpec

    uploading = asyncio.Event()

    async def upload(*args):
        uploading.set()
        await asyncio.Future()

    provider = SimpleNamespace(
        create=AsyncMock(return_value=SandboxHandle("service", "test", None)),
        close=AsyncMock(),
        aclose=AsyncMock(),
        upload_file=upload,
        validate_networking=lambda: None,
        network_address=AsyncMock(return_value="192.0.2.1"),
        set_hosts=AsyncMock(),
    )
    group = make_compose(
        provider,
        {"services": {"app": {"image": "image"}}},
        image_configs={"image": {"Cmd": ["true"]}},
        service_specs={"app": SandboxSpec(files={"/input": "data"})},
    )
    starting = asyncio.create_task(group.start())
    await uploading.wait()
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    await group.stop()
    assert [call.args[0].sandbox_id for call in provider.close.await_args_list] == ["service"]


@pytest.mark.asyncio
async def test_failed_volume_seed_delete_is_retried_by_collection_cleanup():
    from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle

    live = set()
    failed_seed_delete = False

    async def create(spec):
        name = "helper" if not live else "seed"
        live.add(name)
        return SandboxHandle(name, "test", None)

    async def close(handle):
        nonlocal failed_seed_delete
        if handle.sandbox_id == "seed" and not failed_seed_delete:
            failed_seed_delete = True
            raise RuntimeError("temporary seed deletion failure")
        live.remove(handle.sandbox_id)

    provider = SimpleNamespace(
        create=create,
        close=close,
        aclose=AsyncMock(),
        exec=AsyncMock(return_value=SandboxExecResult("", "", 0)),
        validate_networking=lambda: None,
        network_address=AsyncMock(return_value="192.0.2.1"),
        set_hosts=AsyncMock(),
        shared_volume_metadata=lambda: {},
        shared_volume_options=lambda *args, **kwargs: {"volumes": []},
    )
    group = make_compose(
        provider,
        {
            "services": {
                "app": {
                    "image": "image",
                    "volumes": [{"type": "volume", "source": "data", "target": "/data"}],
                }
            },
            "volumes": {"data": {}},
        },
        image_configs={"image": {"Cmd": ["true"]}},
    )
    with pytest.raises(RuntimeError, match="temporary seed deletion failure"):
        await group.start()
    await group.stop()
    assert not live


@pytest.mark.asyncio
async def test_external_command_preserves_stdout_and_reports_stderr():
    import sys

    from nemo_gym.sandbox.adapters.docker_compose import _command

    assert await _command([sys.executable, "-c", "print('normalized configuration')"]) == "normalized configuration\n"
    with pytest.raises(RuntimeError, match=r"failed \(7\): invalid input"):
        await _command([sys.executable, "-c", "import sys; sys.stderr.write('invalid input'); sys.exit(7)"])


@pytest.mark.asyncio
async def test_external_command_cancellation_reaps_child(tmp_path):
    import asyncio
    import os
    import sys

    from nemo_gym.sandbox.adapters.docker_compose import _command

    marker = tmp_path / "child.pid"
    task = asyncio.create_task(
        _command(
            [
                sys.executable,
                "-c",
                "import os, pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)",
                str(marker),
            ]
        )
    )
    try:
        async with asyncio.timeout(5):
            while not marker.exists():
                await asyncio.sleep(0.01)
        pid = int(marker.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields, message",
    [
        ({"healthcheck": {"test": ["CMD", "true"], "interval": "one-second"}}, "duration"),
        ({"healthcheck": {"test": ["CMD", "true"], "timeout": "0s"}}, "timeout"),
        ({"healthcheck": {"test": ["CMD-SHELL"]}}, "requires a command"),
        ({"healthcheck": {"test": ["CMD", "true"], "retries": 0}}, "retries"),
        ({"healthcheck": {"test": ["BAD", "true"]}}, "invalid healthcheck"),
        ({"image": ""}, "requires a prebuilt image"),
        ({"entrypoint": [], "command": []}, "no command"),
        ({"depends_on": {"missing": {"condition": "service_started"}}}, "unknown dependency"),
        ({"depends_on": {"app": {"condition": "invalid"}}}, "invalid dependency"),
        ({"networks": {"default": {"aliases": ["invalid alias"]}}}, "alias"),
        ({"ports": [{"target": 80, "published": "8080"}]}, "published ports"),
        ({"ports": [{"target": 80, "protocol": "udp"}]}, "TCP"),
    ],
)
async def test_invalid_service_configuration_fails_before_provisioning(fields, message):
    provider = SimpleNamespace(
        create=AsyncMock(),
        aclose=AsyncMock(),
        validate_networking=lambda: None,
        network_address=AsyncMock(),
        set_hosts=AsyncMock(),
    )
    group = make_compose(
        provider,
        {"services": {"app": {"image": "image", **fields}}},
        image_configs={"image": {"Cmd": ["true"]}},
    )
    with pytest.raises((ValueError, NotImplementedError), match=message):
        await group.start()
    assert provider.create.await_count == 0


@pytest.mark.asyncio
async def test_health_probe_timeout_retries_then_starts_dependent(tmp_path):
    output = tmp_path / "dependent-started"

    class HealthProvider(ShellProvider):
        probes = 0

        async def exec(self, handle, command, **kwargs):
            if command == "test -d /":
                self.probes += 1
                if self.probes == 1:
                    raise TimeoutError("probe deadline")
            return await super().exec(handle, command, **kwargs)

    provider = HealthProvider()
    async with make_compose(
        provider,
        {
            "services": {
                "db": {
                    "image": "image",
                    "healthcheck": {"test": ["CMD", "test", "-d", "/"], "interval": "1ms", "retries": 2},
                },
                "app": {
                    "image": "image",
                    "command": ["touch", str(output)],
                    "depends_on": {"db": {"condition": "service_healthy"}},
                },
            }
        },
        image_configs={"image": {"Cmd": ["sleep", "1"]}},
        poll_interval_s=0.01,
    ) as group:
        await group._wait("app", "service_completed_successfully")
        assert output.exists()
        assert provider.probes >= 2
    assert provider.closed == ["2", "1"]


@pytest.mark.asyncio
async def test_volume_copy_after_nocopy_mount_and_external_source_cleanup(tmp_path):
    shared = tmp_path / "shared"
    external = shared / "operator-data"
    external.mkdir(parents=True)
    (external / "existing").write_text("external data")
    image_data = tmp_path / "image-data"
    image_data.mkdir()
    (image_data / "seed").write_text("image data")

    class VolumeProvider(ShellProvider):
        def shared_volume_metadata(self):
            return {"placement": "shared-disk"}

        def shared_volume_options(self, source, target, *, read_only=False):
            return {"volumes": [{"source": source, "target": target, "read_only": read_only}]}

        async def exec(self, handle, command, **kwargs):
            spec = self.created[int(handle.sandbox_id) - 1]
            mounts = spec.provider_options.get("volumes", [])
            replacements = {"/data": str(image_data)}
            for mount in mounts:
                backing = shared / mount["source"] if mount["source"] else shared
                replacements[mount["target"]] = str(backing)
            pattern = "|".join(re.escape(target) for target in sorted(replacements, key=len, reverse=True))
            command = re.sub(pattern, lambda match: replacements[match[0]], command)
            return await super().exec(handle, command, **kwargs)

    provider = VolumeProvider()
    group = make_compose(
        provider,
        {
            "services": {
                "first": {
                    "image": "image",
                    "volumes": [{"type": "volume", "source": "data", "target": "/data", "volume": {"nocopy": True}}],
                },
                "second": {
                    "image": "image",
                    "volumes": [
                        {"type": "volume", "source": "data", "target": "/data"},
                        {"type": "bind", "source": "/operator/source", "target": "/external", "read_only": True},
                    ],
                },
            },
            "volumes": {"data": {}},
        },
        image_configs={"image": {"Cmd": ["sleep", "1"]}},
        volume_sources={"/operator/source": "operator-data"},
        poll_interval_s=0.01,
    )
    async with group:
        for name in ("first", "second"):
            result = await group.services[name].exec("cat /data/seed")
            assert result.return_code == 0
            assert result.stdout == "image data"
        result = await group.services["second"].exec("cat /external/existing")
        assert result.stdout == "external data"
        second_spec = provider.created[-1]
        assert second_spec.metadata["placement"] == "shared-disk"
        assert second_spec.provider_options["volumes"][-1]["read_only"] is True
    assert not (shared / group.project).exists()
    assert (external / "existing").read_text() == "external data"
    assert len(provider.closed) == len(provider.created)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "document, message",
    [
        ({"services": {}}, "non-empty services"),
        ({"services": {"invalid name": {"image": "image"}}}, "service name"),
        (
            {
                "services": {
                    "app": {"image": "image", "networks": {"default": {"aliases": ["db"]}}},
                    "db": {"image": "image"},
                }
            },
            "Ambiguous",
        ),
        (
            {
                "services": {
                    "app": {
                        "image": "image",
                        "volumes": [
                            {"type": "volume", "source": "data", "target": "relative"},
                        ],
                    }
                },
                "volumes": {"data": {}},
            },
            "absolute target",
        ),
        (
            {
                "services": {
                    "app": {
                        "image": "image",
                        "volumes": [
                            {"type": "volume", "source": "missing", "target": "/data"},
                        ],
                    }
                }
            },
            "Undeclared",
        ),
    ],
)
async def test_invalid_project_and_volumes_never_provision(document, message):
    provider = SimpleNamespace(
        create=AsyncMock(),
        aclose=AsyncMock(),
        shared_volume_options=lambda *args, **kwargs: {"volumes": []},
        shared_volume_metadata=lambda: {},
    )
    with pytest.raises(ValueError, match=message):
        await make_compose(provider, document).start()
    assert not provider.create.await_count


@pytest.mark.asyncio
@pytest.mark.parametrize("unset, disabled", [(True, False), (False, True)])
async def test_image_environment_unset_and_health_disable(tmp_path, unset, disabled):
    provider = ShellProvider()
    output = tmp_path / "output"
    group = make_compose(
        provider,
        {
            "services": {
                "app": {
                    "image": "image",
                    "environment": {"IMAGE_VALUE": None} if unset else {"ABSENT": None},
                    "healthcheck": {"disable": disabled},
                }
            }
        },
        image_configs={
            "image": {
                "Cmd": ["sh", "-c", f'printf "$IMAGE_VALUE" > {output}'],
                "Env": ["IMAGE_VALUE=inherited"],
                "Healthcheck": {"Test": ["CMD", "false"]},
            }
        },
        poll_interval_s=0.01,
    )
    if unset:
        with pytest.raises(NotImplementedError, match="unsetting inherited"):
            await group.start()
        assert not provider.created
    else:
        async with group:
            await group._wait("app", "service_completed_successfully")
            assert output.read_text() == "inherited"
        assert provider.closed == ["1"]


@pytest.mark.asyncio
async def test_image_inspection_resources_and_environment(tmp_path):
    import sys

    from nemo_gym.sandbox.providers.base import SandboxSpec

    output = tmp_path / "output"
    config = {
        "User": "1000",
        "WorkingDir": str(tmp_path),
        "Env": ["IMAGE=from-image", "OVERRIDE=old"],
        "Shell": ["/bin/bash", "-c"],
    }
    inspector = tmp_path / "inspect.py"
    inspector.write_text(
        "import json, sys\nassert sys.argv[1] == 'original-image'\n" + f"print(json.dumps({{'config': {config!r}}}))\n"
    )

    class UserProvider(ShellProvider):
        users = None

        async def exec(self, handle, command, **kwargs):
            if self.users is None:
                self.users = []
            self.users.append(kwargs.get("user"))
            return await super().exec(handle, command, **kwargs)

    provider = UserProvider()
    group = make_compose(
        provider,
        {
            "services": {
                "app": {
                    "image": "original-image",
                    "user": "1000",
                    "entrypoint": "sh -c",
                    "command": shlex.quote(
                        f'printf "%s|%s|%s|%s" "$IMAGE" "$OVERRIDE" "$SPEC" "$PWD" > {output}; sleep 1'
                    ),
                    "environment": {"OVERRIDE": "new"},
                    "cpus": "1.5",
                    "mem_limit": 1048577,
                    "healthcheck": {"test": ["CMD-SHELL", f"[[ -f {output} ]]"], "interval": "1ms"},
                }
            }
        },
        image_config_command=[sys.executable, str(inspector), "{image}"],
        service_specs={"app": SandboxSpec(env={"SPEC": "from-spec"}, ttl_s=30)},
        poll_interval_s=0.01,
    )
    async with group:
        assert output.read_text() == f"from-image|new|from-spec|{tmp_path}"
        spec = provider.created[0]
        assert spec.image == "original-image"
        assert spec.ttl_s == 30
        assert spec.resources.cpu == 1.5
        assert spec.resources.memory_mib == 2
        assert 1000 in provider.users
        with pytest.raises(RuntimeError, match="already started"):
            await group.start()
    assert provider.closed == ["1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["service_completed_successfully", "service_healthy", "service_started"])
async def test_dependency_timeout_or_failure_cleans_every_service(tmp_path, condition):
    provider = ShellProvider()
    output = tmp_path / "dependent"
    service = {"image": "image"}
    if condition == "service_healthy":
        service["healthcheck"] = {"test": ["CMD", "false"], "retries": 1}
    group = make_compose(
        provider,
        {
            "services": {
                "source": service,
                "dependent": {
                    "image": "image",
                    "command": ["touch", str(output)],
                    "depends_on": {"source": {"condition": condition}},
                },
            }
        },
        image_configs={"image": {"Cmd": ["sleep", "1"]}},
        timeout_s=0.3 if condition == "service_completed_successfully" else 5,
        poll_interval_s=0.01,
    )
    if condition == "service_started":
        async with group:
            await group._wait("dependent", "service_completed_successfully")
            assert output.exists()
    else:
        with pytest.raises(TimeoutError if condition == "service_completed_successfully" else RuntimeError):
            await group.start()
        assert not output.exists()
    assert provider.closed == ["2", "1"]


@pytest.mark.asyncio
async def test_nonroot_image_default_exec_and_explicit_host_opt_out(tmp_path):
    class NonrootProvider(ShellProvider):
        async def exec(self, handle, command, **kwargs):
            assert kwargs.get("user") is None, "image default must not require switching user"
            return await super().exec(handle, command, **kwargs)

        async def set_hosts(self, handle, hosts):
            assert handle.sandbox_id == "2", "nonroot service explicitly disables host writes"
            self.hosts = hosts

    ready = tmp_path / "ready"
    provider = NonrootProvider()
    group = make_compose(
        provider,
        {
            "services": {
                "kafka": {
                    "image": "image",
                    "x-sandbox": {"hosts": []},
                    "command": ["sh", "-c", f"touch {ready}; while test ! -f {ready}.release; do sleep .01; done"],
                    "healthcheck": {"test": ["CMD", "test", "-f", str(ready)], "interval": "1ms"},
                },
                "client": {"image": "image", "command": ["true"]},
            }
        },
        image_configs={"image": {"User": "appuser"}},
        poll_interval_s=0.01,
    )
    async with group:
        ready.with_name(ready.name + ".release").touch()
        await group._wait("kafka", "service_completed_successfully")
        await group._wait("client", "service_completed_successfully")
        assert provider.hosts == {"kafka": "192.0.2.1", "client": "192.0.2.2"}
    assert provider.closed == ["2", "1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options", [{"cap_add": ["SYS_PTRACE"]}, {"shm_size": 67108864}, {"x-sandbox": {"hosts": ["peer"]}}]
)
async def test_runtime_and_host_options_fail_preflight_without_support(options):
    provider = ShellProvider()
    group = make_compose(provider, {"services": {"app": {"image": "image", **options}}})
    with pytest.raises((ValueError, NotImplementedError)):
        await group.start()
    assert not provider.created


@pytest.mark.asyncio
async def test_runtime_requirements_configured_before_workload(tmp_path):
    configured = tmp_path / "configured"
    output = tmp_path / "output"

    class RuntimeProvider(ShellProvider):
        async def create(self, spec):
            assert self.validated
            return await super().create(spec)

        def validate_runtime_requirements(self, *, cap_add, shm_size):
            self.validated = True
            assert cap_add == ("SYS_PTRACE",)
            assert shm_size == 67108864

        async def configure_runtime(self, handle, *, cap_add, shm_size):
            assert cap_add == ("SYS_PTRACE",)
            assert shm_size == 67108864
            configured.touch()

    provider = RuntimeProvider()
    async with make_compose(
        provider,
        {
            "services": {
                "app": {
                    "image": "image",
                    "cap_add": ["SYS_PTRACE"],
                    "shm_size": 67108864,
                    "command": ["sh", "-c", f"test -f {configured} && touch {output}"],
                }
            }
        },
        image_configs={"image": {}},
        poll_interval_s=0.01,
    ) as group:
        await group._wait("app", "service_completed_successfully")
        assert output.exists()


@pytest.mark.asyncio
async def test_network_mode_requires_forwarding_opt_in_before_create():
    provider = ShellProvider()
    group = make_compose(
        provider,
        {
            "services": {
                "app": {"image": "image", "network_mode": "service:db"},
                "db": {"image": "image", "expose": [5432]},
            }
        },
    )
    with pytest.raises(NotImplementedError, match="forward"):
        await group.start()
    assert not provider.created


@pytest.mark.asyncio
async def test_network_mode_orders_target_waits_for_relay_and_cancels_on_cleanup(tmp_path):
    import asyncio
    from pathlib import Path

    listening = tmp_path / "listening"
    output = tmp_path / "output"
    cancelled = asyncio.Event()

    class ForwardProvider(ShellProvider):
        async def create(self, spec):
            assert self.validated
            return await super().create(spec)

        def validate_port_forwarding(self):
            self.validated = True

        async def forward_ports(self, handle, target_address, ports, *, ready_file):
            assert handle.sandbox_id == "2"
            assert target_address == "192.0.2.1"
            assert ports == (5432,)
            listening.touch()
            Path(ready_file).touch()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def close(self, handle):
            assert cancelled.is_set(), "relay must stop before deleting its sandbox"
            await super().close(handle)

    provider = ForwardProvider()
    async with make_compose(
        provider,
        {
            "services": {
                "app": {
                    "image": "image",
                    "network_mode": "service:db",
                    "command": ["sh", "-c", f"test -f {listening} && touch {output}"],
                },
                "db": {"image": "image", "expose": [5432], "command": ["sleep", ".1"]},
            }
        },
        image_configs={"image": {}},
        poll_interval_s=0.01,
    ) as group:
        await group._wait("app", "service_completed_successfully")
        assert output.exists()
    assert cancelled.is_set()
    assert provider.closed == ["2", "1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("address,authority", [("192.0.2.10", "192.0.2.10"), ("2001:db8::10", "[2001:db8::10]")])
async def test_resolve_selected_environment_urls_for_nonroot_service(tmp_path, address, authority):
    output = tmp_path / "url"
    provider = ShellProvider()
    provider.network_address = AsyncMock(return_value=address)
    provider.set_hosts = AsyncMock(side_effect=AssertionError("host injection must be disabled"))
    original = "http://user:pass@workspace:18073/path?next=workspace#fragment"
    expected = f"http://user:pass@{authority}:18073/path?next=workspace#fragment"
    group = make_compose(
        provider,
        {
            "services": {
                "workspace": {
                    "image": "image",
                    "environment": {"URL": original, "UNCHANGED": original},
                    "x-sandbox": {"hosts": [], "resolve_environment": ["URL"]},
                    "command": ["sh", "-c", f'printf "%s" "$URL" > {output}; sleep 30'],
                    "healthcheck": {
                        "test": [
                            "CMD-SHELL",
                            f'test "$URL" = {shlex.quote(expected)} && test "$UNCHANGED" = {shlex.quote(original)}',
                        ]
                    },
                }
            }
        },
        image_configs={"image": {}},
        poll_interval_s=0.01,
    )
    async with group:
        assert output.read_text() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection,value",
    [
        ("URL", "http://workspace"),
        (["MISSING"], "http://workspace"),
        (["URL"], "http://unknown"),
        (["URL"], "workspace"),
        (["URL"], "http://workspace:bad"),
    ],
)
async def test_invalid_environment_resolution_fails_before_provisioning(selection, value):
    provider = ShellProvider()
    group = make_compose(
        provider,
        {
            "services": {
                "workspace": {
                    "image": "image",
                    "command": ["true"],
                    "environment": {"URL": value},
                    "x-sandbox": {"resolve_environment": selection},
                }
            }
        },
        image_configs={"image": {}},
    )
    with pytest.raises(ValueError, match="environment|[Pp]ort"):
        await group.start()
    assert provider.created == []


@pytest.mark.asyncio
async def test_process_finishing_during_marker_probe_is_not_startup_failure(monkeypatch):
    import asyncio

    provider = ShellProvider()
    group = make_compose(
        provider,
        {"services": {"app": {"image": "image", "command": ["true"]}}},
        image_configs={"image": {}},
        poll_interval_s=0.001,
    )
    execute = provider.exec
    launch = asyncio.Event()
    marker_reads = 0

    async def race(handle, command, **kwargs):
        nonlocal marker_reads
        if command.startswith("sh ") and "/run.sh >" in command:
            await launch.wait()
        if command.startswith("test -f ") and command.endswith("/started"):
            marker_reads += 1
            if marker_reads == 1:
                result = await execute(handle, command, **kwargs)
                assert result.return_code == 1
                launch.set()
                await group._processes["app"]
                return result
        return await execute(handle, command, **kwargs)

    monkeypatch.setattr(provider, "exec", race)
    async with group:
        await group._wait("app", "service_completed_successfully")
        assert marker_reads == 2
    assert provider.closed == ["1"]
