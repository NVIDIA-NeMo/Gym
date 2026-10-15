# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle


pytestmark = pytest.mark.sandbox


def make_compose(provider, document, *, default_command=None, **kwargs):
    """Supply normalized YAML and optional commands for focused lifecycle tests."""
    from unittest.mock import Mock

    from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose

    group = AsyncSandboxCompose(provider, "compose.yaml", **kwargs)
    if default_command is not None:
        for service in document["services"].values():
            if not service.get("entrypoint"):
                service.setdefault("command", default_command)
    group.document = document
    group._load = Mock(return_value=document)
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
@pytest.mark.parametrize(
    "field,value", [("privileged", True), ("pre_stop", [{"command": "x"}]), ("devices", ["/dev/fuse"])]
)
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
        import signal

        spec = self.created[int(handle.sandbox_id) - 1]
        # A real sandbox ends every process inside it, so stop the whole group and not only the shell.
        # Killing the shell alone would leave the service script and its children running after the test.
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            cwd=cwd,
            env={**os.environ, **spec.env, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        def kill_group():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        try:
            out, err = await asyncio.wait_for(process.communicate(), timeout_s)
        except asyncio.CancelledError:
            kill_group()
            await process.wait()
            raise
        except TimeoutError:
            kill_group()
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
    async with make_compose(provider, document, default_command=["false"], poll_interval_s=0.01) as group:
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
        poll_interval_s=0.01,
    )
    with pytest.raises(RuntimeError, match="exited.*7"):
        await group.start()
    assert not result.exists()
    assert provider.closed == ["2", "1"]


@pytest.mark.asyncio
async def test_explicit_entrypoint_preserves_empty_arguments(tmp_path):
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
        default_command=["sleep", "1"],
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
        default_command=["sleep", ".1"],
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
        default_command=["sleep", ".1"],
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
        default_command=["sleep", ".1"],
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
        default_command=["sleep", ".1"],
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
    class VolumeProvider(ConnectableShellProvider):
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
        default_command=["true"],
    )
    with pytest.raises(RuntimeError, match="volume helper unavailable"):
        await group.start()
    provider.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_yaml_input_runs_services_without_local_tools(tmp_path, monkeypatch):
    import asyncio

    import yaml

    from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose

    output = tmp_path / "result"
    path = tmp_path / "compose.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "app": {
                        "image": "test-image",
                        "environment": {"VALUE": "from-yaml"},
                        "x-sandbox": {"hosts": []},
                        "command": ["sh", "-c", f'printf "%s" "$VALUE" > {output}'],
                    }
                }
            }
        )
    )
    original_subprocess = asyncio.create_subprocess_exec

    async def sandbox_shell_only(*argv, **kwargs):
        assert argv[0] == "/bin/sh"
        return await original_subprocess(*argv, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", sandbox_shell_only)
    provider = ShellProvider()
    provider.set_hosts = AsyncMock(side_effect=AssertionError("explicit host opt-out was lost"))
    async with AsyncSandboxCompose(
        provider,
        path,
        poll_interval_s=0.01,
    ) as group:
        await group._wait("app", "service_completed_successfully")
        assert output.read_text() == "from-yaml"
        assert provider.created[0].env["VALUE"] == "from-yaml"
    assert provider.closed == ["1"]


@pytest.mark.asyncio
async def test_invalid_yaml_fails_before_provisioning(tmp_path):
    import yaml

    path = tmp_path / "compose.yaml"
    path.write_text("services: [\n")
    provider = ShellProvider()
    with pytest.raises(yaml.YAMLError):
        await AsyncSandboxCompose(provider, path).start()
    assert provider.created == []


def test_yaml_preserves_literal_dollars_and_does_not_load_dotenv(tmp_path, monkeypatch):
    import yaml

    monkeypatch.setenv("VALUE", "from-process")
    (tmp_path / ".env").write_text("VALUE=from-dotenv\n")
    document = {
        "services": {
            "app": {
                "image": "test-image",
                "command": ["sh", "-c", 'echo $$ "$VALUE"'],
                "environment": {"VALUE": "${VALUE}"},
            }
        }
    }
    path = tmp_path / "compose.yaml"
    path.write_text(yaml.safe_dump(document))
    assert AsyncSandboxCompose(Provider(), path)._load() == document


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
        default_command=["true"],
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
        default_command=["true"],
    )
    with pytest.raises(RuntimeError, match="temporary seed deletion failure"):
        await group.start()
    await group.stop()
    assert not live


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
        ({"entrypoint": [], "command": []}, "resolve entrypoint or command upstream"),
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
        default_command=["true"],
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
        default_command=["sleep", "1"],
        poll_interval_s=0.01,
    ) as group:
        await group._wait("app", "service_completed_successfully")
        assert output.exists()
        assert provider.probes >= 2
    assert provider.closed == ["2", "1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("declaration", [{}, None], ids=["mapping", "null"])
async def test_volume_copy_after_nocopy_mount_and_external_source_cleanup(tmp_path, declaration):
    # ``volumes:\n  data:`` (a null declaration, the canonical Compose spelling) provisions like ``data: {}``.
    shared = tmp_path / "shared"
    external = shared / "operator-data"
    external.mkdir(parents=True)
    (external / "existing").write_text("external data")
    image_data = tmp_path / "image-data"
    image_data.mkdir()
    (image_data / "seed").write_text("image data")

    class VolumeProvider(ConnectableShellProvider):
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
            "volumes": {"data": declaration},
        },
        default_command=["sleep", "1"],
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
        assert group._seeds and all(seed._stopped for seed in group._seeds)
        descriptor = json.loads(json.dumps(await group.serialize()))
        receiver = VolumeProvider()
        receiver.created = provider.created.copy()
        receiver.create = AsyncMock(side_effect=AssertionError("must not provision"))
        async with await AsyncSandboxCompose.connect(descriptor, provider=receiver) as connected:
            assert (await connected.services["second"].exec("cat /data/seed")).stdout == "image data"
        assert group._volume_helper._handle.sandbox_id not in receiver.closed
        assert (shared / group.project / "data" / "seed").read_text() == "image data"
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
@pytest.mark.parametrize("user", ["1000", "0", 0])
async def test_service_resources_environment_and_workdir(tmp_path, user):
    from nemo_gym.sandbox.providers.base import SandboxSpec

    output = tmp_path / "output"

    class UserProvider(ShellProvider):
        users = None

        async def exec(self, handle, command, **kwargs):
            if self.users is None:
                self.users = []
            self.users.append((command, kwargs.get("user")))
            return await super().exec(handle, command, **kwargs)

    provider = UserProvider()
    group = make_compose(
        provider,
        {
            "services": {
                "app": {
                    "image": "original-image",
                    "user": user,
                    "entrypoint": "sh -c",
                    "command": shlex.quote(f'printf "%s|%s|%s" "$OVERRIDE" "$SPEC" "$PWD" > {output}; sleep 1'),
                    "environment": {"OVERRIDE": "new", "UNSET": None},
                    "working_dir": str(tmp_path),
                    "cpus": "1.5",
                    "mem_limit": 1048577,
                    "healthcheck": {"test": ["CMD-SHELL", f"test -f {output}"], "interval": "1ms"},
                }
            }
        },
        service_specs={"app": SandboxSpec(env={"SPEC": "from-spec"}, ttl_s=30)},
        poll_interval_s=0.01,
    )
    async with group:
        assert output.read_text() == f"new|from-spec|{tmp_path}"
        spec = provider.created[0]
        assert spec.image == "original-image"
        assert spec.ttl_s == 30
        assert spec.resources.cpu == 1.5
        assert spec.resources.memory_mib == 2
        service_users = [
            uid for command, uid in provider.users if str(output) in command or command.startswith("sh /tmp/")
        ]
        assert len(service_users) >= 2  # Service launch and health check.
        assert all(uid == int(user) for uid in service_users)
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
        default_command=["sleep", "1"],
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
@pytest.mark.parametrize("with_metadata", [True, False])
async def test_runtime_requirements_configured_before_workload(tmp_path, with_metadata):
    configured = tmp_path / "configured"
    output = tmp_path / "output"

    class RuntimeProvider(ShellProvider):
        async def create(self, spec):
            assert self.validated
            assert spec.metadata["example.test/shm"] == ("67108864" if with_metadata else "true")
            assert spec.metadata["purpose"] == "kept"
            return await super().create(spec)

        def validate_runtime_requirements(self, *, cap_add, shm_size):
            self.validated = True
            assert cap_add == ("SYS_PTRACE",)
            assert shm_size == 67108864
            return {"example.test/shm": str(shm_size)} if with_metadata else None

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
                    "labels": {"example.test/shm": "true", "purpose": "kept"},
                    "command": ["sh", "-c", f"test -f {configured} && touch {output}"],
                }
            }
        },
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
    # Synthetic credentials exercise preservation of URL userinfo during hostname replacement.
    original = "http://user:pass@workspace:18073/path?next=workspace#fragment"  # pragma: allowlist secret
    expected = f"http://user:pass@{authority}:18073/path?next=workspace#fragment"  # pragma: allowlist secret
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


class ConnectableShellProvider(ShellProvider):
    async def endpoint(self, handle, port):
        from nemo_gym.sandbox.providers.base import SandboxEndpoint

        return SandboxEndpoint(endpoint=f"http://127.0.0.1:{port}")

    async def serialize_handle(self, handle, *, scope=None):
        return {"sandbox_id": handle.sandbox_id, "scope": scope}

    async def connect(self, descriptor):
        return SandboxHandle(descriptor["sandbox_id"], self.name, None)


@pytest.mark.asyncio
async def test_compose_connect_round_trip_without_yaml_or_provisioning(tmp_path):
    original = ConnectableShellProvider()
    owner = make_compose(
        original,
        {
            "services": {
                "main": {
                    "image": "image",
                    "command": ["sleep", "60"],
                    "working_dir": str(tmp_path),
                    "expose": ["8000"],
                },
                "db": {"image": "image", "command": ["sleep", "60"]},
            }
        },
        poll_interval_s=0.01,
    )
    async with owner:
        endpoint = await owner.services["main"].endpoint(8000)
        descriptor = json.loads(json.dumps(await owner.serialize(scope="operate")))
        assert set(descriptor["services"]) == {"main", "db"}
        assert descriptor["services"]["main"]["scope"] == "operate"
        receiver = ConnectableShellProvider()
        receiver.created = original.created.copy()  # backing store shared by independent clients
        receiver.create = AsyncMock(side_effect=AssertionError("must not provision"))
        receiver.aclose = AsyncMock()
        connected = await AsyncSandboxCompose.connect(descriptor, provider=receiver)
        async with connected:
            assert await connected.services["main"].endpoint(8000) == endpoint
            with pytest.raises(ValueError, match="not declared"):
                await connected.services["main"].endpoint(8001)
            assert (await connected.services["main"].exec("pwd")).stdout.strip() == str(tmp_path)
            assert await connected.serialize(scope="operate") == descriptor
        assert receiver.closed == ["2", "1"]
        receiver.aclose.assert_awaited_once()
        await connected.stop()
        receiver.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_compose_partial_connect_failure_does_not_destroy_services(cancel):
    import asyncio

    error = asyncio.CancelledError() if cancel else RuntimeError("attach failed")
    provider = ConnectableShellProvider()
    provider.connect = AsyncMock(side_effect=[SandboxHandle("one", provider.name, None), error])
    provider.aclose = AsyncMock()
    descriptor = {
        "services": {"main": {"sandbox_id": "one"}, "db": {"sandbox_id": "two"}},
    }
    with pytest.raises(type(error)):
        await AsyncSandboxCompose.connect(descriptor, provider=provider)
    assert provider.closed == []
    provider.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        {},
        {"project": "../../x", "services": {"main": {}}},
        {"services": {"main": "bad"}},
        {"project": "compose-" + "a" * 32, "services": {"main": {}}, "volume_helper": "bad"},
        {"project": "compose-" + "a" * 32, "services": {"main": {}}, "seeds": ["bad"]},
    ],
)
async def test_compose_connect_validates_descriptor_before_attaching(descriptor):
    provider = ConnectableShellProvider()
    provider.connect = AsyncMock()
    with pytest.raises(ValueError):
        await AsyncSandboxCompose.connect(descriptor, provider=provider)
    provider.connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_compose_serialize_requires_running_collection():
    group = AsyncSandboxCompose(ConnectableShellProvider(), "compose.yaml")
    with pytest.raises(RuntimeError, match="running"):
        await group.serialize()


@pytest.mark.asyncio
async def test_compose_connect_without_helpers_and_closed_serialize():
    provider = ConnectableShellProvider()
    descriptor = {"services": {"main": {"sandbox_id": "one"}}}
    connected = await AsyncSandboxCompose.connect(descriptor, provider=provider)
    assert set(await connected.serialize()) == {"services"}
    await connected.stop()
    with pytest.raises(RuntimeError, match="running"):
        await connected.serialize()
    with pytest.raises(RuntimeError, match="closed"):
        await connected.__aenter__()


@pytest.mark.asyncio
async def test_compose_connect_requires_provider_capability():
    descriptor = {"services": {"main": {"sandbox_id": "one"}}}
    with pytest.raises(RuntimeError, match="does not support"):
        await AsyncSandboxCompose.connect(descriptor, provider=Provider())


@pytest.mark.asyncio
async def test_compose_start_requires_yaml():
    with pytest.raises(ValueError, match="YAML"):
        await AsyncSandboxCompose(Provider(), None).start()


@pytest.mark.asyncio
async def test_explicit_entrypoint_runs_service(tmp_path):
    output = tmp_path / "output"
    group = make_compose(
        ShellProvider(),
        {
            "services": {
                "app": {
                    "image": "example/app:1",
                    "entrypoint": ["sh", "-c"],
                    "command": [f"printf started > {output}"],
                    "working_dir": str(tmp_path),
                }
            }
        },
    )
    async with group:
        await group._wait("app", "service_completed_successfully")
        assert output.read_text() == "started"


@pytest.mark.asyncio
@pytest.mark.parametrize("startup", [{}, {"entrypoint": [], "command": []}])
async def test_unresolved_startup_rejected_before_provisioning(startup):
    provider = ShellProvider()
    group = make_compose(provider, {"services": {"app": {"image": "example/app:1", **startup}}})
    with pytest.raises(ValueError, match="resolve entrypoint or command upstream"):
        await group.start()
    assert provider.created == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "document",
    [
        None,
        [],
        {"services": {"app": None}},
        {"services": {"app": {"image": "image", "environment": ["KEY=value"]}}},
        {"services": {"app": {"image": "image", "ports": ["8000:80"]}}},
        {"services": {"app": {"image": "image", "volumes": ["data:/data"]}}},
        {"services": {"app": {"image": "image", "depends_on": {"app": None}}}},
        {"services": {"app": {}}, "networks": ["default"]},
    ],
)
async def test_unresolved_yaml_shapes_fail_before_provisioning(tmp_path, document):
    import yaml

    path = tmp_path / "compose.yaml"
    path.write_text(yaml.safe_dump(document))
    provider = ShellProvider()
    with pytest.raises(ValueError, match="mapping|upstream"):
        await AsyncSandboxCompose(provider, path).start()
    assert provider.created == []


async def test_cleanup_failure_still_attempts_volume_helper_and_transport():
    provider = Provider()
    provider.aclose = AsyncMock()
    group = AsyncSandboxCompose(provider, None)
    broken = SimpleNamespace(stop=AsyncMock(side_effect=RuntimeError("service delete failed")))
    healthy = SimpleNamespace(stop=AsyncMock())
    helper = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=1, stderr="volume failed")), stop=AsyncMock()
    )
    group.services = {"broken": broken, "healthy": healthy}
    group._volume_helper = helper
    with pytest.raises(ExceptionGroup) as exc:
        await group.stop()
    assert len(exc.value.exceptions) == 2
    healthy.stop.assert_awaited_once()
    helper.stop.assert_awaited_once()
    provider.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_spec_env_wins_over_compose_environment_only_where_the_caller_gives_one(tmp_path):
    """A spec's env is the caller's host-resolved override, so it beats the service's Compose ``environment`` (a
    ``${HF_TOKEN:-}`` that interpolated to empty upstream must not hide the host value, as in Harbor, where the
    task's env override file is merged after the task's Compose file). A key only the Compose file sets is kept,
    and a sidecar given an empty spec env keeps its Compose ``environment``."""
    from nemo_gym.sandbox.providers.base import SandboxSpec

    main_out, side_out = tmp_path / "main", tmp_path / "side"
    provider = ShellProvider()
    group = make_compose(
        provider,
        {
            "services": {
                "main": {
                    "image": "image",
                    "command": [
                        "sh",
                        "-c",
                        f'printf "%s|%s|%s|%s" "$HF_TOKEN" "$ONLY_COMPOSE" "$TASK_ID" "$ONLY_SPEC" > {main_out}; sleep 1',
                    ],
                    "environment": {"HF_TOKEN": "", "ONLY_COMPOSE": "c", "TASK_ID": "t016"},
                    "healthcheck": {"test": ["CMD-SHELL", f"test -f {main_out}"], "interval": "1ms"},
                },
                "side": {
                    "image": "image",
                    "command": ["sh", "-c", f'printf "%s" "$HF_TOKEN" > {side_out}; sleep 1'],
                    "environment": {"HF_TOKEN": "sidecar-literal"},
                    "healthcheck": {"test": ["CMD-SHELL", f"test -f {side_out}"], "interval": "1ms"},
                },
            }
        },
        service_specs={
            "main": SandboxSpec(env={"HF_TOKEN": "hostval", "ONLY_SPEC": "s"}, ttl_s=30),
            "side": SandboxSpec(env={}, ttl_s=30),
        },
        poll_interval_s=0.01,
    )
    async with group:
        assert group._plans["main"]["spec"].env == {
            "HF_TOKEN": "hostval",
            "ONLY_COMPOSE": "c",
            "TASK_ID": "t016",
            "ONLY_SPEC": "s",
        }
        assert group._plans["side"]["spec"].env == {"HF_TOKEN": "sidecar-literal"}
        assert main_out.read_text() == "hostval|c|t016|s"
        assert side_out.read_text() == "sidecar-literal"


@pytest.mark.asyncio
async def test_unsupported_fields_are_refused_with_the_shared_message_before_creating():
    """The adapter raises the first message of the pure key allow-list prepare also runs (the first step of the
    one field check), before it asks the provider for anything."""
    from nemo_gym.sandbox.compose_config import unsupported_compose_fields

    provider = Provider()
    document = {
        "services": {
            "main": {"image": "a"},
            "task-server": {"image": "b", "platform": "linux/amd64", "logging": {}, "privileged": True},
        }
    }
    expected = "Service 'task-server': unsupported Compose fields ['logging', 'privileged']"
    assert unsupported_compose_fields(document) == [expected]
    group = make_compose(provider, document)
    with pytest.raises(NotImplementedError, match=re.escape(expected)):
        await group.start()
    assert provider.created == []
    top = make_compose(provider, {"x-extra": 1, "services": {"main": {"image": "a", "deploy": {}}}})
    with pytest.raises(NotImplementedError, match=re.escape("Unsupported Compose fields: ['x-extra']")):
        await top.start()
    assert provider.created == []


def test_a_bodiless_named_volume_declaration_counts_as_declared():
    """`volumes:` followed by a bare name is valid Compose and loads as None; it must not read as undeclared."""

    class VolumeProvider(ConnectableShellProvider):
        def shared_volume_metadata(self):
            return {"placement": "shared-disk"}

        def shared_volume_options(self, source, target, *, read_only=False):
            return {"volumes": [{"source": source, "target": target, "read_only": read_only}]}

    document = {
        "services": {"app": {"image": "app", "volumes": [{"type": "volume", "source": "data", "target": "/data"}]}},
        "volumes": {"data": None},
    }
    group = make_compose(VolumeProvider(), document)
    assert group._validate()
    missing = make_compose(VolumeProvider(), {**document, "volumes": {}})
    with pytest.raises(ValueError, match="Undeclared Compose volume: 'data'"):
        missing._validate()


# --------------------------------------------------------------------------------------------------------------
# Compose service fields (M28)
# --------------------------------------------------------------------------------------------------------------


class HostsProvider(ShellProvider):
    """Records the hosts each sandbox received."""

    def __init__(self):
        super().__init__()
        self.hosts = {}

    async def set_hosts(self, handle, hosts):
        self.hosts[handle.sandbox_id] = dict(hosts)


@pytest.mark.asyncio
async def test_custom_networks_scope_which_services_see_each_other_and_extra_hosts_are_added():
    """alibaba-ctf style: `restart: always` is recorded, a custom network scopes name resolution."""
    provider = HostsProvider()
    async with make_compose(
        provider,
        {
            "services": {
                "main": {"image": "img", "restart": "always", "networks": ["agent_network"]},
                "target": {
                    "image": "img",
                    "restart": "always",
                    "networks": {"agent_network": {"aliases": ["challenge"]}},
                    "extra_hosts": ["licence.example:198.51.100.7"],
                },
                "aside": {"image": "img"},
            },
            "networks": {"agent_network": {"driver": "bridge"}},
        },
        default_command=["sleep", ".1"],
        poll_interval_s=0.01,
    ) as group:
        assert group.restart_policies == {"main": "always", "target": "always", "aside": "no"}
        by_name = {
            name: provider.hosts[sandbox._require_handle().sandbox_id] for name, sandbox in group.services.items()
        }
        main_ip, target_ip, aside_ip = (
            "192.0.2." + group.services[n]._require_handle().sandbox_id for n in ("main", "target", "aside")
        )
        # main and target share agent_network: each sees the other by name and alias, neither sees aside.
        assert by_name["main"] == {"main": main_ip, "target": target_ip, "challenge": target_ip}
        assert by_name["target"] == {
            "target": target_ip,
            "challenge": target_ip,
            "main": main_ip,
            "licence.example": "198.51.100.7",
        }
        # aside is alone on the default network: only itself.
        assert by_name["aside"] == {"aside": aside_ip}


@pytest.mark.asyncio
async def test_a_shared_named_network_puts_every_service_in_reach():
    """o11y-bench style: one named network every service joins behaves like the default network."""
    provider = HostsProvider()
    document = {
        "services": {name: {"image": "img", "networks": ["harbor-shared"]} for name in ("main", "grafana", "loki")},
        "networks": {"harbor-shared": {"name": "harbor-shared"}},
    }
    async with make_compose(provider, document, default_command=["sleep", ".1"], poll_interval_s=0.01) as group:
        for name, sandbox in group.services.items():
            assert set(provider.hosts[sandbox._require_handle().sandbox_id]) == {"main", "grafana", "loki"}


@pytest.mark.asyncio
async def test_resolved_environment_must_name_a_peer_on_a_shared_network():
    provider = HostsProvider()
    group = make_compose(
        provider,
        {
            "services": {
                "app": {
                    "image": "img",
                    "networks": ["front"],
                    "environment": {"DB_URL": "http://db:5432"},
                    "x-sandbox": {"hosts": [], "resolve_environment": ["DB_URL"]},
                },
                "db": {"image": "img", "networks": ["back"]},
            },
            "networks": {"front": {}, "back": {}},
        },
        default_command=["sleep", ".1"],
        poll_interval_s=0.01,
    )
    with pytest.raises(ValueError, match="shares no network"):
        await group.start()
    assert provider.closed  # whatever was created was torn down


@pytest.mark.asyncio
async def test_extra_hosts_need_host_injection():
    provider = ShellProvider()
    group = make_compose(
        provider,
        {"services": {"app": {"image": "img", "extra_hosts": ["db:10.0.0.1"], "x-sandbox": {"hosts": []}}}},
    )
    with pytest.raises(ValueError, match="extra_hosts cannot be applied without host injection"):
        await group.start()
    assert not provider.created


@pytest.mark.asyncio
async def test_deploy_resources_size_the_service_sandbox_and_gpu_sidecars_run_on_their_own_provider():
    from nemo_gym.sandbox.providers.base import SandboxSpec

    cpu_provider, gpu_provider = HostsProvider(), HostsProvider()
    gpu_provider.name = "gpu"
    document = {
        "services": {
            "main": {"image": "img", "deploy": {"resources": {"limits": {"cpus": "2", "memory": "1g"}}}},
            "model": {
                "image": "img",
                "cpus": "0.5",
                "deploy": {
                    "resources": {
                        "reservations": {"devices": [{"driver": "nvidia", "capabilities": ["gpu"], "count": 1}]}
                    }
                },
            },
        }
    }
    async with make_compose(
        cpu_provider,
        document,
        default_command=["sleep", ".1"],
        poll_interval_s=0.01,
        service_providers={"model": gpu_provider},
        service_specs={"model": SandboxSpec(resources={"gpu_type": "H100"}, ttl_s=30)},
    ) as group:
        assert group.services["main"]._provider is cpu_provider and group.services["model"]._provider is gpu_provider
        (main_spec,) = cpu_provider.created
        (model_spec,) = gpu_provider.created
        assert (main_spec.resources.cpu, main_spec.resources.memory_mib, main_spec.resources.gpu) == (2.0, 1024, None)
        # The legacy cpus key still counts when deploy says nothing about CPUs; the spec's gpu_type is kept.
        assert (model_spec.resources.cpu, model_spec.resources.gpu, model_spec.resources.gpu_type) == (0.5, 1, "H100")
        # Both providers resolved addresses and the sidecar on the GPU provider still sees main by name.
        assert set(gpu_provider.hosts["1"]) == {"main", "model"}
    assert cpu_provider.closed == ["1"] and gpu_provider.closed == ["1"]


@pytest.mark.asyncio
async def test_service_providers_must_name_services_and_every_provider_must_network():
    provider = ShellProvider()
    group = make_compose(provider, {"services": {"app": {"image": "img"}}}, service_providers={"ghost": Provider()})
    with pytest.raises(ValueError, match="service_providers names unknown services"):
        await group.start()
    group = make_compose(provider, {"services": {"app": {"image": "img"}}}, service_providers={"app": Provider()})
    with pytest.raises(NotImplementedError, match="networking between sandboxes"):
        await group.start()
    assert not provider.created


@pytest.mark.asyncio
async def test_post_start_hooks_run_in_order_after_the_service_starts_and_a_failure_fails_the_seed(tmp_path):
    log = tmp_path / "hooks"
    provider = ShellProvider()
    async with make_compose(
        provider,
        {
            "services": {
                "app": {
                    "image": "img",
                    "command": ["sh", "-c", f"echo service >> {log}; sleep 5"],
                    "post_start": [
                        {"command": f"echo first >> {log}"},
                        {"command": ["sh", "-c", f'echo "$HOOK_VAR" >> {log}'], "environment": {"HOOK_VAR": "second"}},
                        {"command": f"pwd >> {log}", "working_dir": str(tmp_path)},
                    ],
                }
            }
        },
        poll_interval_s=0.01,
    ):
        # Hooks run once the service's command has started, in order; the service kept running.
        lines = log.read_text().splitlines()
        assert lines[0] == "service" and lines[1:] == ["first", "second", str(tmp_path)]
    failing = make_compose(
        provider,
        {
            "services": {
                "app": {
                    "image": "img",
                    "command": ["sleep", "5"],
                    "post_start": [{"command": "echo boom >&2; exit 3"}, {"command": f"touch {tmp_path}/never"}],
                }
            }
        },
        poll_interval_s=0.01,
    )
    with pytest.raises(RuntimeError, match=r"post_start hook 0 exited \(3\): boom"):
        await failing.start()
    assert not (tmp_path / "never").exists()
    assert provider.closed == ["1", "2"]


class RecordingShellProvider(ShellProvider):
    """A ShellProvider that keeps every exec command and upload it was asked for."""

    def __init__(self):
        super().__init__()
        self.commands = []
        self.uploads = []

    async def exec(self, handle, command, **kwargs):
        self.commands.append(command)
        return await super().exec(handle, command, **kwargs)

    async def upload_file(self, handle, source_path, target_path):
        self.uploads.append((str(source_path), str(target_path)))
        await super().upload_file(handle, source_path, target_path)


@pytest.mark.asyncio
async def test_uploads_land_in_the_service_before_its_command_runs(tmp_path):
    """M22: a task-folder bind mount, planned as ``x-sandbox.uploads``, is copied into the service after its
    sandbox starts and before the service command, files and folders alike, readable by any user. The chmod is
    never recursive and covers exactly the folders and files the upload touched, every folder ``mkdir -p``
    created included (intermediates and the parent of a single file, which a restrictive root umask would leave
    unreadable); a folder upload merges over what the target already holds (review findings)."""
    task = tmp_path / "task"
    (task / "data" / "nested").mkdir(parents=True)
    (task / "data" / "deep" / "er").mkdir(parents=True)
    (task / "seed.json").write_text('{"ok": true}')
    (task / "data" / "a.txt").write_text("a")
    (task / "data" / "nested" / "b.txt").write_text("b")
    (task / "data" / "nested" / "b.txt").chmod(0o600)
    (task / "data" / "deep" / "er" / "c.txt").write_text("c")
    sandbox_root = tmp_path / "sandbox"
    # What the image already ships under the target: kept (a merge), never chmod-ed.
    (sandbox_root / "data" / "image").mkdir(parents=True)
    (sandbox_root / "data" / "image" / "default.conf").write_text("image")
    (sandbox_root / "data" / "image" / "default.conf").chmod(0o600)
    result = tmp_path / "result"
    document = {
        "services": {
            "api": {
                "image": "image",
                "command": [
                    "sh",
                    "-c",
                    f"cat {sandbox_root}/app/seed.json {sandbox_root}/data/a.txt {sandbox_root}/data/nested/b.txt "
                    f"{sandbox_root}/data/deep/er/c.txt > {result}",
                ],
                "x-sandbox": {
                    "uploads": [
                        {"source": str(task / "seed.json"), "target": f"{sandbox_root}/app/seed.json"},
                        {"source": str(task / "data"), "target": f"{sandbox_root}/data"},
                    ]
                },
            }
        }
    }
    provider = RecordingShellProvider()
    # The sandbox's root shell creates the folders with its own umask; a restrictive one leaves them 0700.
    umask = os.umask(0o077)
    try:
        async with make_compose(provider, document, poll_interval_s=0.01, upload_root=task) as group:
            await group._wait("api", "service_completed_successfully")
    finally:
        os.umask(umask)
    assert result.read_text() == '{"ok": true}abc'
    assert (sandbox_root / "data" / "nested" / "b.txt").stat().st_mode & 0o044 == 0o044
    for created in ("app", "data/nested", "data/deep", "data/deep/er"):
        assert (sandbox_root / created).stat().st_mode & 0o055 == 0o055, created
    assert (sandbox_root / "data" / "image" / "default.conf").read_text() == "image"
    assert (sandbox_root / "data" / "image" / "default.conf").stat().st_mode & 0o044 == 0
    assert (sandbox_root / "data" / "image").stat().st_mode & 0o077 == 0o055  # pre-existing: untouched
    chmods = [command for command in provider.commands if command.startswith("chmod")]
    assert not any(command.startswith("chmod -R") for command in provider.commands)
    touched = [shlex.split(command.removesuffix(" 2>/dev/null || true"))[2:] for command in chmods]
    assert touched == [
        [f"{sandbox_root}/app", f"{sandbox_root}/app/seed.json"],
        [
            f"{sandbox_root}/data",
            f"{sandbox_root}/data/deep",
            f"{sandbox_root}/data/deep/er",
            f"{sandbox_root}/data/nested",
            f"{sandbox_root}/data/a.txt",
            f"{sandbox_root}/data/deep/er/c.txt",
            f"{sandbox_root}/data/nested/b.txt",
        ],
    ]
    # Nothing above the sandbox root (which existed) is ever chmod-ed, "/" least of all.
    assert all(path.startswith(f"{sandbox_root}/") for paths in touched for path in paths)


@pytest.mark.asyncio
async def test_a_link_inside_an_uploaded_folder_is_refused_before_it_is_read(tmp_path):
    # Review: ``is_file()`` follows links, so a link in a bound folder would upload a host file the planner never
    # checked. The adapter refuses the folder, naming the link, before any file of it is uploaded.
    task = tmp_path / "task"
    (task / "data").mkdir(parents=True)
    (task / "data" / "ok.txt").write_text("ok")
    secret = tmp_path / "secret"
    secret.write_text("host secret")
    (task / "data" / "leak").symlink_to(secret)
    sandbox_root = tmp_path / "sandbox"
    document = {
        "services": {
            "api": {
                "image": "image",
                "command": ["sleep", "1"],
                "x-sandbox": {"uploads": [{"source": str(task / "data"), "target": f"{sandbox_root}/data"}]},
            }
        }
    }
    provider = RecordingShellProvider()
    with pytest.raises(RuntimeError, match=r"contains a link: .*data/leak"):
        await make_compose(provider, document, poll_interval_s=0.01, upload_root=task).start()
    assert provider.uploads == []
    assert not (sandbox_root / "data" / "leak").exists()
    assert len(provider.closed) == len(provider.created) == 1


@pytest.mark.asyncio
async def test_upload_sources_must_lie_under_the_upload_root(tmp_path):
    # Review blocker: upload sources are read in this process. A document assembled without the planner (or a
    # root the planner never saw) cannot reach /etc/passwd or any file outside the root; no sandbox is created.
    task = tmp_path / "task"
    task.mkdir()
    (task / "seed.json").write_text("{}")
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (task / "link.json").symlink_to(outside)
    for source in ("/etc/passwd", str(outside), str(task / "link.json"), str(task / ".." / "outside.json")):
        provider = Provider()
        document = {
            "services": {"api": {"image": "image", "x-sandbox": {"uploads": [{"source": source, "target": "/x"}]}}}
        }
        with pytest.raises(ValueError, match="outside the upload root"):
            await make_compose(provider, document, upload_root=task).start()
        assert provider.created == []
    # Without a root, uploads are refused outright: no caller may forget to vouch for one.
    provider = Provider()
    document = {
        "services": {
            "api": {"image": "image", "x-sandbox": {"uploads": [{"source": str(task / "seed.json"), "target": "/x"}]}}
        }
    }
    with pytest.raises(ValueError, match="need an upload_root"):
        await make_compose(provider, document).start()
    assert provider.created == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field, value",
    [
        ("tmpfs", ["/run"]),
        ("ulimits", {"nofile": 1024}),
        ("sysctls", {"net.core.somaxconn": "1024"}),
        ("security_opt", ["seccomp=unconfined"]),
    ],
)
async def test_service_options_need_the_provider_capability_and_the_error_names_field_and_provider(field, value):
    provider = ShellProvider()
    group = make_compose(provider, {"services": {"app": {"image": "img", field: value}}})
    with pytest.raises(
        NotImplementedError, match=f"Service 'app': {field} is not supported by the test sandbox provider"
    ):
        await group.start()
    assert not provider.created

    class PartialProvider(ShellProvider):
        SERVICE_OPTION_FIELDS = frozenset({"tmpfs"})

        def service_options(self, **kwargs):
            return {"run_args": ["--tmpfs", *kwargs["tmpfs"]]}

    provider = PartialProvider()
    group = make_compose(
        provider,
        {"services": {"app": {"image": "img", field: value}}},
        default_command=["sleep", ".1"],
        poll_interval_s=0.01,
    )
    if field == "tmpfs":
        async with group:
            assert provider.created[0].provider_options == {"run_args": ["--tmpfs", "/run"]}
    else:
        with pytest.raises(NotImplementedError, match=f"{field} is not supported by the test sandbox provider"):
            await group.start()
        assert not provider.created


@pytest.mark.asyncio
async def test_service_options_and_pull_policy_merge_into_the_spec_provider_options():
    from nemo_gym.sandbox.providers.base import SandboxSpec

    class FullProvider(ShellProvider):
        SERVICE_OPTION_FIELDS = frozenset({"tmpfs", "ulimits", "sysctls", "security_opt"})

        def service_options(self, *, tmpfs=(), ulimits=None, sysctls=None, security_opt=()):
            self.options = {"tmpfs": tmpfs, "ulimits": ulimits, "sysctls": sysctls, "security_opt": security_opt}
            return {"run_args": ["--tmpfs", *tmpfs], "extensions": {"from": "options"}}

        def pull_policy_options(self, policy):
            return {"run_args": ["--pull", policy], "extensions": {"pull": policy}}

    provider = FullProvider()
    async with make_compose(
        provider,
        {
            "services": {
                "app": {
                    "image": "img",
                    "tmpfs": "/run",
                    "ulimits": {"nofile": {"soft": 1, "hard": 2}},
                    "sysctls": ["a=b"],
                    "security_opt": ["x"],
                    "pull_policy": "if_not_present",
                    "platform": "linux/x86_64",
                }
            }
        },
        default_command=["sleep", ".1"],
        poll_interval_s=0.01,
        service_specs={
            "app": SandboxSpec(ttl_s=30, provider_options={"run_args": ["--init"], "extensions": {"k": "v"}})
        },
    ) as group:
        assert provider.options == {
            "tmpfs": ("/run",),
            "ulimits": {"nofile": {"soft": 1, "hard": 2}},
            "sysctls": {"a": "b"},
            "security_opt": ("x",),
        }
        assert provider.created[0].provider_options == {
            "run_args": ["--init", "--tmpfs", "/run", "--pull", "missing"],
            "extensions": {"k": "v", "from": "options", "pull": "missing"},
        }
        assert group.pull_policies == {"app": "missing"}


@pytest.mark.asyncio
async def test_pull_policy_is_recorded_when_the_provider_cannot_apply_it_and_platform_must_match():
    provider = ShellProvider()
    async with make_compose(
        provider,
        {"services": {"app": {"image": "img", "pull_policy": "always", "platform": "linux/amd64"}}},
        default_command=["sleep", ".1"],
        poll_interval_s=0.01,
    ) as group:
        assert group.pull_policies == {"app": "always"}
        assert provider.created[0].provider_options == {}
    provider = ShellProvider()
    group = make_compose(
        provider, {"services": {"app": {"image": "img", "platform": "linux/amd64"}}}, platform="linux/arm64"
    )
    with pytest.raises(ValueError, match="platform 'linux/amd64' does not match the deployment platform linux/arm64"):
        await group.start()
    assert not provider.created


@pytest.mark.asyncio
async def test_network_mode_service_shares_its_targets_networks():
    class ForwardingProvider(HostsProvider):
        def validate_port_forwarding(self):
            pass

        async def forward_ports(self, handle, target_address, ports, *, ready_file):
            import asyncio
            from pathlib import Path

            Path(ready_file).touch()
            await asyncio.sleep(3600)

    provider = ForwardingProvider()
    async with make_compose(
        provider,
        {
            "services": {
                "api": {"image": "img", "networks": ["back"], "expose": ["80"]},
                "main": {"image": "img", "network_mode": "service:api"},
                "web": {"image": "img", "networks": ["front"]},
            },
            "networks": {"back": {}, "front": {}},
        },
        default_command=["sleep", ".1"],
        poll_interval_s=0.01,
    ) as group:
        main_hosts = provider.hosts[group.services["main"]._require_handle().sandbox_id]
        assert set(main_hosts) == {"main", "api"}


class RecordingProvider(Provider):
    """Sandboxes that exist only as records and answer every exec with success; records which provider each
    forwarding relay, host injection and shared-storage call went through."""

    def __init__(self, name, subnet):
        super().__init__()
        self.name = name
        self.subnet = subnet
        self.execs = []
        self.hosts = {}
        self.forwarded = []

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        import asyncio

        self.execs.append((handle.sandbox_id, command, user))
        await asyncio.sleep(0)  # a real exec suspends, which lets the forwarding relay task run
        return SandboxExecResult("", "", 0)

    async def upload_file(self, handle, source_path, target_path):
        pass

    def validate_networking(self):
        pass

    async def network_address(self, handle):
        return f"10.{self.subnet}.0.{handle.sandbox_id}"

    async def set_hosts(self, handle, hosts):
        self.hosts[handle.sandbox_id] = dict(hosts)

    def validate_port_forwarding(self):
        pass

    async def forward_ports(self, handle, target_address, ports, *, ready_file):
        import asyncio

        self.forwarded.append((handle.sandbox_id, target_address, tuple(ports)))
        await asyncio.Event().wait()

    def shared_volume_metadata(self):
        return {"storage": self.name}

    def shared_volume_options(self, source, target, *, read_only=False):
        return {"volumes": [{"source": source, "target": target, "read_only": read_only, "storage": self.name}]}


@pytest.mark.asyncio
async def test_forwarding_and_volumes_of_a_service_on_its_own_provider_go_through_that_provider():
    """A sidecar on its own provider (a GPU block): its relay runs where its sandbox is, and its named volume is
    initialized, seeded and mounted on that provider's shared storage, never the collection provider's."""
    cpu, gpu = RecordingProvider("cpu", 1), RecordingProvider("gpu", 2)
    document = {
        "services": {
            "db": {"image": "img", "expose": [5432]},
            "side": {
                "image": "img",
                "network_mode": "service:db",
                "volumes": [{"type": "volume", "source": "data", "target": "/data"}],
            },
        },
        "volumes": {"data": {}},
    }
    async with make_compose(
        cpu, document, default_command=["sleep", "1"], poll_interval_s=0.01, service_providers={"side": gpu}
    ) as group:
        side_id = group.services["side"]._require_handle().sandbox_id
        # The relay was started on the provider that owns the forwarder's sandbox, towards db's address on cpu.
        assert cpu.forwarded == [] and gpu.forwarded == [(side_id, "10.1.0.1", (5432,))]
        # Init helper, copy-up seed and the service itself were all created on gpu; cpu only ran db.
        assert [spec.image for spec in gpu.created] == ["alpine:3.22", "img", "img"]
        assert [spec.image for spec in cpu.created] == ["img"]
        helper, seed, side = gpu.created
        assert helper.metadata == {"storage": "gpu"} and seed.metadata == {"storage": "gpu"}
        assert side.metadata["storage"] == "gpu"
        assert seed.provider_options["volumes"] == [
            {"source": f"{group.project}/data", "target": "/tmp/compose-volumes", "read_only": False, "storage": "gpu"}
        ]
        assert side.provider_options["volumes"] == [
            {"source": f"{group.project}/data", "target": "/data", "read_only": False, "storage": "gpu"}
        ]
        # The directory was made through gpu's helper; cpu, which runs no volumes, got no helper at all.
        assert any(command.startswith("mkdir -p /tmp/compose-volumes/") for _, command, _ in gpu.execs)
        assert group._volume_helper is None and len(group._volume_helpers) == 1
    # Cleanup went through gpu's helper; both providers' sandboxes were closed.
    assert any("rm -rf /tmp/compose-volumes/" in command for _, command, _ in gpu.execs)
    assert not any("compose-volumes" in command for _, command, _ in cpu.execs)
    assert sorted(gpu.closed) == ["1", "2", "3"] and cpu.closed == ["1"]


@pytest.mark.asyncio
async def test_volumes_need_shared_storage_on_the_providers_that_run_them():
    cpu, gpu = RecordingProvider("cpu", 1), Provider()
    gpu.name = "plain"
    document = {
        "services": {"side": {"image": "img", "volumes": [{"type": "volume", "source": "data", "target": "/data"}]}},
        "volumes": {"data": {}},
    }
    group = make_compose(cpu, document, service_providers={"side": gpu})
    with pytest.raises(
        NotImplementedError, match="Service 'side': the plain sandbox provider does not support shared"
    ):
        await group.start()
    assert not cpu.created and not gpu.created


class ChainForwardingProvider(HostsProvider):
    def validate_port_forwarding(self):
        pass

    async def forward_ports(self, handle, target_address, ports, *, ready_file):
        import asyncio
        from pathlib import Path

        Path(ready_file).touch()
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_chained_network_mode_resolves_to_the_end_of_the_chain_in_declaration_order():
    """a -> service:b -> service:c, declared in that order: every link is on c's networks (M27 started this
    layout; the first M28 membership pass looked b up before it was filled)."""
    provider = ChainForwardingProvider()
    async with make_compose(
        provider,
        {
            "services": {
                "a": {"image": "img", "network_mode": "service:b"},
                "b": {"image": "img", "network_mode": "service:c", "expose": [81]},
                "c": {"image": "img", "networks": ["back"], "expose": [80]},
                "d": {"image": "img"},
            },
            "networks": {"back": {}},
        },
        default_command=["sleep", ".1"],
        poll_interval_s=0.01,
    ) as group:
        assert group._memberships() == {"a": {"back"}, "b": {"back"}, "c": {"back"}, "d": {"default"}}
        by_name = {
            name: set(provider.hosts[sandbox._require_handle().sandbox_id]) for name, sandbox in group.services.items()
        }
        assert by_name["a"] == {"a", "b", "c"} and by_name["c"] == {"a", "b", "c"} and by_name["d"] == {"d"}


@pytest.mark.asyncio
async def test_a_network_mode_cycle_is_named_before_provisioning():
    provider = ChainForwardingProvider()
    group = make_compose(
        provider,
        {
            "services": {
                "a": {"image": "img", "network_mode": "service:b", "expose": [1]},
                "b": {"image": "img", "network_mode": "service:c", "expose": [2]},
                "c": {"image": "img", "network_mode": "service:b", "expose": [3]},
            }
        },
    )
    with pytest.raises(ValueError, match="Service 'a': network_mode cycle: b -> c -> b"):
        await group.start()
    assert not provider.created


@pytest.mark.asyncio
async def test_the_servers_normalized_gpu_reservation_passes_the_adapters_own_pass(tmp_path):
    """What the environment server writes (deploy normalized once) is what the adapter reloads and normalizes again."""
    import yaml

    from nemo_gym.sandbox.compose_config import check_compose_fields

    cpu, gpu = HostsProvider(), HostsProvider()
    document = check_compose_fields(
        {
            "services": {
                "main": {"image": "img", "command": ["sleep", ".1"]},
                "model": {
                    "image": "img",
                    "command": ["sleep", ".1"],
                    "deploy": {"resources": {"reservations": {"devices": [{"capabilities": ["gpu"], "count": 1}]}}},
                },
            }
        }
    )
    assert document["services"]["model"]["deploy"] == {"resources": {"reservations": {"gpus": 1}}}
    path = tmp_path / "compose.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    async with AsyncSandboxCompose(cpu, path, service_providers={"model": gpu}, poll_interval_s=0.01) as group:
        assert group.services["model"]._provider is gpu
        (model_spec,) = gpu.created
        assert model_spec.resources.gpu == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uploads,message",
    [
        ("seed.json:/app/seed.json", "list of source/target mappings"),
        ([{"source": "x"}], "list of source/target mappings"),
        ([{"source": "/nonexistent/seed.json", "target": "/app/seed.json"}], "does not exist"),
        ([{"source": ".", "target": "app/seed.json"}], "absolute container path"),
        ([{"source": ".", "target": "/app/../seed.json"}], "absolute container path"),
        ([{"source": ".", "target": "/"}], "container root"),
        ([{"source": ".", "target": "//"}], "container root"),
    ],
)
async def test_malformed_uploads_fail_before_provisioning(uploads, message):
    provider = Provider()
    group = make_compose(
        provider, {"services": {"api": {"image": "image", "x-sandbox": {"uploads": uploads}}}}, upload_root="."
    )
    with pytest.raises(ValueError, match=message):
        await group.start()
    assert provider.created == []


@pytest.mark.asyncio
async def test_upload_folder_creation_falls_back_to_the_default_user_and_reports_failure(tmp_path):
    seed = tmp_path / "seed.json"
    seed.write_text("{}")

    class NoRootProvider(ShellProvider):
        def __init__(self):
            super().__init__()
            self.users = []

        async def exec(self, handle, command, *, user=None, **kwargs):
            if command.startswith("mkdir -p "):
                self.users.append(user)
                if user == "root":
                    raise RuntimeError("identity switch not permitted")
                if "/forbidden/" in command:
                    return SandboxExecResult("", "mkdir: permission denied", 1)
            return await super().exec(handle, command, user=user, **kwargs)

    provider = NoRootProvider()
    target = tmp_path / "sandbox" / "app" / "seed.json"
    document = {
        "services": {
            "api": {
                "image": "image",
                "command": ["sleep", "1"],
                "x-sandbox": {"uploads": [{"source": str(seed), "target": str(target)}]},
            }
        }
    }
    async with make_compose(provider, document, poll_interval_s=0.01, upload_root=tmp_path):
        assert target.read_text() == "{}"
    assert provider.users[:2] == ["root", None]

    provider = NoRootProvider()
    document["services"]["api"]["x-sandbox"]["uploads"][0]["target"] = "/forbidden/seed.json"
    with pytest.raises(RuntimeError, match="cannot create '/forbidden/seed.json' for upload"):
        await make_compose(provider, document, poll_interval_s=0.01, upload_root=tmp_path).start()
    assert len(provider.closed) == len(provider.created) == 1
